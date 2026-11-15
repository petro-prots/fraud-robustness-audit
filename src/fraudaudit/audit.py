"""Minimal episode audit: VER of normal fraud vs VER under attack on the target cards.

Steps of ``run_audit``:

1. fraud episodes of the target periods (``test`` by default) are split by card 50/50 into target
   and spare cards (``split_cards(episodes, seed)``); the attacker never sees a target card;
2. reconnaissance: ``attack`` searches a strategy on the spare cards through ``ReconOracle`` with the
   budget ``tm.recon_budget_ops``;
3. the best strategy is realised on every target card (its own ``rng``) and executed with the bank's
   reaction (``run_episode``); every executed operation is checked against the threat model
   (``feasible_share``, ruling R3);
4. baseline: the real operations of every target episode are replayed the same way (``clean_ver``),
   the VER of normal, non-adversarial fraud;
5. metrics: mean VER with a percentile bootstrap interval over episodes (``ver_ci``), ASR (share of
   successful target episodes) with its Clopper-Pearson upper bound, clean recall/FPR/precision.

Determinism (ruling R7): the random streams are derived from ``seed`` only: card split
``split_cards(…, seed)``, oracle ``default_rng([seed, 1])``, attack ``default_rng([seed, 2])``,
target ``i`` ``default_rng([seed, 3, i])``, bootstrap ``seed``. Target executions run in a spawn
process pool (``n_jobs > 1``, each worker loads its own model from ``model_dir``); results are
identical for any ``n_jobs``. Entry points that use ``n_jobs > 1`` need an
``if __name__ == "__main__":`` guard.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .attacks.random_search import Attack
from .data import fraud_episodes, period, split_cards
from .episode import run_episode
from .metrics import clopper_pearson_upper, uplift_ci, ver_ci
from .oracle import ReconOracle
from .serving.core import Serving
from .serving.state import Tx
from .threat import EpisodeContext, MerchantTable, Strategy, ThreatModel, contexts_for, feasible_ops, realize

CONFIDENCE = 0.95
N_BOOT = 2000
_STREAM_ORACLE, _STREAM_ATTACK, _STREAM_TARGET = 1, 2, 3


@dataclass(frozen=True)
class AuditResult:
    ver_adv: tuple[float, float, float]    # (mean, lower, upper), 95 % bootstrap over target episodes
    ver_clean: tuple[float, float, float]  # the same for the real operations of the target episodes
    uplift: tuple[float, float, float]     # paired bootstrap of ver_adv_i − ver_clean_i (gate: upper ≤ δ)
    asr: float                             # share of target episodes with success (passed >= rho·A)
    asr_upper: float                       # Clopper-Pearson upper bound of asr (95 %)
    n_targets: int
    recon_ops: int                         # operations spent on spare cards
    best: Strategy
    clean_quality: dict                    # {"val": quality(...), "test": quality(...)} at tau
    meta: dict
    feasible_share: float                  # executed adversarial ops on target cards passing feasible_ops
    per_target: list[dict]                 # card, A, ver_adv, ver_clean, success, ops, declines, blocked
    spare_cards: list[int]


def episode_txs(df: pd.DataFrame, episode) -> list[Tx]:
    """The real operations of ``episode`` (its ``row`` ids in ``df``) as ``Tx``, in (ts, row) order."""
    ops = df[df["row"].isin(episode.rows)].sort_values(["ts", "row"], kind="stable")
    if len(ops) != len(episode.rows):
        raise ValueError(f"episode of card {episode.card}: {len(ops)} of {len(episode.rows)} rows found")
    return [Tx(ts=pd.Timestamp(r.ts), amt=float(r.amt), merchant=str(r.merchant), category=str(r.category),
               merch_lat=float(r.merch_lat), merch_long=float(r.merch_long)) for r in ops.itertuples(index=False)]


def clean_ver(serving, targets: list[EpisodeContext], df: pd.DataFrame, max_declines: int | None,
              rho: float) -> np.ndarray:
    """VER of the real (non-adversarial) operations of every target episode, replayed through
    ``run_episode`` from the same pre-episode context with the same bank reaction.

    Replay assumption (``REPLAY``): the context is the card history strictly before the episode start
    (episode rows excluded); the card's legitimate operations inside the episode window are not
    replayed, in this clean run as in the adversarial one. So this is VER of the fraud operations
    alone, not the bank's historical outcome."""
    return np.array([_clean_outcome(serving, ctx, episode_txs(df, ctx.episode), max_declines, rho).ver
                     for ctx in targets], dtype=np.float64)


REPLAY = {"context": "history ts < episode.start (episode rows excluded)",
          "in_window_legit_ops": "excluded in both runs"}


def _clean_outcome(serving, ctx: EpisodeContext, real_txs: list[Tx], max_declines: int | None, rho: float):
    """The clean leg: the episode's real operations through ``run_episode`` (used by ``clean_ver`` and
    by ``run_audit``)."""
    return run_episode(serving, ctx, real_txs, max_declines, rho)


def _execute_target(serving, ctx: EpisodeContext, clean_txs: list[Tx], best: Strategy, tm: ThreatModel,
                    merchants: MerchantTable, seed: int, i: int) -> dict:
    """Adversarial and clean run of target ``i``; plain types only (crosses process boundaries)."""
    txs = realize(best, ctx, tm, merchants, np.random.default_rng([seed, _STREAM_TARGET, i]))
    adv = run_episode(serving, ctx, txs, tm.max_declines, tm.rho)
    executed = txs[:adv.n_ops]  # time-ordered prefix: the ops the bank actually saw
    n_ok = sum(feasible_ops(executed, ctx, tm, merchants))
    clean = _clean_outcome(serving, ctx, clean_txs, tm.max_declines, tm.rho)
    return {"card": int(ctx.episode.card), "start": pd.Timestamp(ctx.episode.start).isoformat(),
            "amount": float(ctx.episode.amount),
            "ver_adv": float(adv.ver), "success": bool(adv.success), "n_ops": int(adv.n_ops),
            "n_declines": int(adv.n_declines), "blocked": bool(adv.blocked),
            "n_feasible_ops": int(n_ok),
            "ver_clean": float(clean.ver), "clean_success": bool(clean.success),
            "clean_n_ops": int(clean.n_ops), "clean_n_real_ops": len(clean_txs),
            "clean_blocked": bool(clean.blocked)}


_WORKER: dict = {}  # per worker process: set once by _init_worker


def _load_serving(model_dir) -> Serving:
    from .model.artifacts import load_artifacts
    from .model.pipeline import FraudModel
    return Serving(FraudModel.from_artifacts(load_artifacts(Path(model_dir))))


def _init_worker(model_dir, targets, clean, best, tm, merchants, seed) -> None:
    _WORKER.update(serving=_load_serving(model_dir), targets=targets, clean=clean, best=best, tm=tm,
                   merchants=merchants, seed=seed)


def _target_job(i: int) -> dict:
    w = _WORKER
    return _execute_target(w["serving"], w["targets"][i], w["clean"][i], w["best"], w["tm"], w["merchants"],
                           w["seed"], i)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _versions() -> dict:
    import keras
    import scipy
    import sklearn
    import tensorflow as tf

    from . import __version__
    return {"python": platform.python_version(), "fraudaudit": __version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "tensorflow": tf.__version__, "keras": keras.__version__}


def clean_quality(df: pd.DataFrame, model) -> dict:
    """Recall/FPR/precision of ``model`` at its tau on clean ``val`` and ``test`` operations."""
    from .model.pipeline import features, quality
    per = period(df).to_numpy()
    F = features(df, model.encoder)
    y = df["is_fraud"].to_numpy(dtype=np.int64)
    out = {}
    for name in ("val", "test"):
        m = per == name
        p = np.asarray(model.predict_proba(F[m]), dtype=np.float64) if m.any() else np.empty(0)
        out[name] = quality(y[m], p, model.tau)
    return out


def run_audit(df: pd.DataFrame, model, tm: ThreatModel, attack: Attack, seed: int, *,
              n_jobs: int | None = None, model_dir: Path | str | None = None,
              target_periods: tuple[str, ...] = ("test",), threat_path: Path | str | None = None) -> AuditResult:
    """Episode audit of ``model`` (see the module docstring).

    ``model`` is used in-process (``n_jobs=1``) and for the clean quality; ``n_jobs > 1`` needs
    ``model_dir`` (the same model's artifacts), loaded by every worker. ``n_jobs=None`` means
    ``os.cpu_count()`` with ``model_dir`` and 1 without. ``target_periods`` other than ``("test",)``
    only for tests on small synthetic data. ``threat_path`` (the YAML ``tm`` came from) is hashed
    into ``meta``.
    """
    t0 = time.perf_counter()
    jobs = n_jobs if n_jobs is not None else ((os.cpu_count() or 1) if model_dir is not None else 1)
    jobs = max(1, int(jobs))
    if jobs > 1 and model_dir is None:
        raise ValueError("n_jobs > 1 needs model_dir: every worker loads its own model")

    episodes = fraud_episodes(df, set(target_periods))
    target_eps, spare_eps = split_cards(episodes, seed)
    if not target_eps:
        raise ValueError("no target episodes")
    targets = contexts_for(df, target_eps)
    spares = contexts_for(df, spare_eps)
    merchants = MerchantTable.from_df(df)
    serving = Serving(model)

    with ReconOracle(serving, spares, tm, merchants, tm.recon_budget_ops,
                     np.random.default_rng([seed, _STREAM_ORACLE]), n_jobs=jobs, model_dir=model_dir) as oracle:
        run = attack.run(oracle, tm, np.random.default_rng([seed, _STREAM_ATTACK]))
        recon = {"attack": type(attack).__name__, "evals": int(run.evals), "ops_used": int(run.ops_used),
                 "best_fitness": float(run.best_fitness), "cards_blocked": int(oracle.cards_blocked),
                 "n_spare_episodes": len(spares)}
    if run.best is None:
        raise ValueError(f"the recon budget {tm.recon_budget_ops} did not allow a single evaluation")
    best = run.best.clip(tm, len(merchants))

    clean = [episode_txs(df, ctx.episode) for ctx in targets]
    idx = list(range(len(targets)))
    if jobs > 1 and len(targets) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(targets)), mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(str(model_dir), targets, clean, best, tm, merchants, seed)) as ex:
            per_target = list(ex.map(_target_job, idx, chunksize=max(1, len(idx) // (4 * jobs))))
    else:
        per_target = [_execute_target(serving, targets[i], clean[i], best, tm, merchants, seed, i) for i in idx]

    v_adv = np.array([t["ver_adv"] for t in per_target])
    v_clean = np.array([t["ver_clean"] for t in per_target])
    n = len(per_target)
    k = sum(t["success"] for t in per_target)
    n_exec = sum(t["n_ops"] for t in per_target)
    feasible_share = sum(t["n_feasible_ops"] for t in per_target) / n_exec if n_exec else float("nan")
    quality = clean_quality(df, model)

    from .cli import git_info
    meta = {
        "seed": int(seed),
        **git_info(),
        "platform": {"system": platform.system(), "machine": platform.machine(), "platform": platform.platform(),
                     "cpu_count": os.cpu_count()},
        "versions": _versions(),
        "threat_model": {"file": None if threat_path is None else str(threat_path),
                         "sha256": None if threat_path is None else _sha256(Path(threat_path)),
                         "params": {f: getattr(tm, f) for f in tm.__dataclass_fields__}},
        "model": {"dir": None if model_dir is None else str(model_dir),
                  "model_json_sha256": None if model_dir is None else _sha256(Path(model_dir) / "model.json"),
                  "tau": float(model.tau)},
        "target_periods": list(target_periods),
        "n_target_cards": len({t["card"] for t in per_target}),
        "recon_budget_ops": int(tm.recon_budget_ops),
        "recon": recon,
        "n_jobs": jobs,
        "confidence": CONFIDENCE,
        "n_boot": N_BOOT,
        "n_success": int(k),
        "n_clean_success": int(sum(t["clean_success"] for t in per_target)),
        "n_executed_adv_ops": int(n_exec),
        "replay": dict(REPLAY),
        "n_ver_adv_gt_1": int((v_adv > 1).sum()), "max_ver_adv": float(v_adv.max()),
        "n_ver_clean_gt_1": int((v_clean > 1).sum()), "max_ver_clean": float(v_clean.max()),
        "median_amount": float(np.median([t["amount"] for t in per_target])),
    }
    result = AuditResult(
        ver_adv=ver_ci(v_adv, CONFIDENCE, N_BOOT, seed), ver_clean=ver_ci(v_clean, CONFIDENCE, N_BOOT, seed),
        uplift=uplift_ci(v_adv, v_clean, CONFIDENCE, N_BOOT, seed),
        asr=k / n, asr_upper=clopper_pearson_upper(int(k), n, CONFIDENCE), n_targets=n,
        recon_ops=int(run.ops_used), best=best, clean_quality=quality, meta=meta,
        feasible_share=float(feasible_share), per_target=per_target,
        spare_cards=sorted({c.episode.card for c in spares}))
    meta["wall_seconds"] = round(time.perf_counter() - t0, 1)
    return result


def strategy_dict(s: Strategy, merchants: MerchantTable | None = None) -> dict:
    """The strategy as JSON (full ``k_max`` arrays) plus its first ``k`` operations in plain terms."""
    k = s.k
    shares = np.asarray(s.shares[:k], dtype=np.float64)
    w = shares / shares.sum() if shares.sum() > 0 else np.full(k, 1.0 / k)
    ops = []
    for i in range(k):
        j = int(s.merchant[i])
        off = float(s.day[i]) * 24 + float(s.hour[i])  # hours from midnight of the start day
        op = {"share_of_A": float(s.u * w[i]), "offset_hours": off, "day_index": int(off // 24),
              "time_of_day": f"{int(off % 24):02d}:{int((off % 1) * 60):02d}", "merchant_index": j}
        if merchants is not None:
            op.update(merchant=merchants.names[j], category=merchants.category[j])
        ops.append(op)
    ops.sort(key=lambda o: o["offset_hours"])  # planned order (before wrapping into the window)
    return {"k": int(k), "u": float(s.u), "shares": [float(x) for x in s.shares],
            "day": [float(x) for x in s.day], "hour": [float(x) for x in s.hour],
            "merchant": [int(x) for x in s.merchant], "ops": ops}


def to_json(r: AuditResult, merchants: MerchantTable | None = None) -> dict:
    """Every field of ``r``; the best strategy via ``strategy_dict`` (categories need ``merchants``)."""
    def ci(t):
        return {"mean": t[0], "lower": t[1], "upper": t[2]}
    best = strategy_dict(r.best, merchants)
    return {"ver_adv": ci(r.ver_adv), "ver_clean": ci(r.ver_clean), "uplift": ci(r.uplift), "asr": r.asr, "asr_upper": r.asr_upper,
            "n_targets": r.n_targets, "recon_ops": r.recon_ops, "feasible_share": r.feasible_share,
            "best": best, "clean_quality": r.clean_quality, "meta": r.meta, "per_target": r.per_target,
            "spare_cards": r.spare_cards}


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %".replace(".", ",")


def _num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def summary_markdown(data: dict) -> str:
    """Ukrainian Markdown summary of ``to_json`` output (no verdict: the gate is a later step)."""
    m, b = data["meta"], data["best"]
    va, vc, up = data["ver_adv"], data["ver_clean"], data["uplift"]
    med = m["median_amount"]
    conf = int(round(100 * m["confidence"]))
    lines = [
        "# Аудит стійкості до епізодних атак (мінімальний)",
        "",
        f"Атака: `{m['recon']['attack']}`, модель загроз: `{m['threat_model']['file']}`. "
        "Вердикту немає: правила шлюзу ще не застосовуються.",
        "",
        "| показник | значення | 95 % інтервал |",
        "|---|---:|---|",
        f"| VER_clean – повтор реальних шахрайських операцій без легітимних операцій вікна | {_pct(vc['mean'])} | [{_pct(vc['lower'])}; {_pct(vc['upper'])}] |",
        f"| VER_adv (найкраща стратегія атаки) | {_pct(va['mean'])} | [{_pct(va['lower'])}; {_pct(va['upper'])}] |",
        f"| Приріст VER_adv − VER_clean (парний bootstrap) | {_pct(up['mean'])} "
        f"| [{_pct(up['lower'])}; {_pct(up['upper'])}] |",
        f"| ASR (частка успішних епізодів, ≥ ρ·A) | {_pct(data['asr'])} ({m['n_success']} з {data['n_targets']}) "
        f"| верхня межа {conf} %: {_pct(data['asr_upper'])} |",
        "",
        "Обидва прогони (VER_clean і VER_adv) стартують зі стану картки до початку епізоду, а легітимні "
        "операції картки всередині вікна епізоду не відтворюються, тож VER_clean – це повтор лише "
        "шахрайських операцій, а не історичний результат банку.",
        "",
        f"- Приріст у доларах при медіанній сумі епізоду A = {_num(med, 2)} $: "
        f"≈ {_num(up['mean'] * med, 0)} $ [{_num(up['lower'] * med, 0)}; {_num(up['upper'] * med, 0)}] "
        "на епізод (частка × медіана A).",
        f"- VER > 1 (через нижню межу суми операції): змагальних епізодів {m['n_ver_adv_gt_1']} "
        f"(макс. VER_adv {_num(m['max_ver_adv'])}), реальних {m['n_ver_clean_gt_1']} "
        f"(макс. VER_clean {_num(m['max_ver_clean'])}); значення не обрізаються.",
        f"- Цільових епізодів: {data['n_targets']} (карток: {m['n_target_cards']}); запасних карток для "
        f"розвідки: {len(data['spare_cards'])}.",
        f"- Інтервали VER: percentile bootstrap за епізодами, {m['n_boot']} повторів, seed {m['seed']}; "
        f"межа ASR: Clopper–Pearson, однобічна.",
        f"- Розвідка: використано {data['recon_ops']} з {m['recon_budget_ops']} операцій, "
        f"{m['recon']['evals']} оцінок стратегій, найкращий середній VER на запасних картках "
        f"{_pct(m['recon']['best_fitness'])}.",
        f"- Частка виконаних змагальних операцій, що відповідають моделі загроз: {_pct(data['feasible_share'])} "
        f"({m['n_executed_adv_ops']} операцій).",
        f"- Без атаки поріг ρ·A досягли {m['n_clean_success']} з {data['n_targets']} епізодів.",
        "",
        "## Найкраща стратегія",
        "",
        f"Операцій k = {b['k']}, сумарно ≈ {_pct(b['u'])} від A. Таблиця – план стратегії в порядку "
        "запланованого часу, до перенесення у вікно і розсування операцій (min_gap_secs), а не порядок "
        "фактично виконаних операцій:",
        "",
        "| № | частка A | день | година | категорія |",
        "|---:|---:|---:|---:|---|",
    ]
    for i, op in enumerate(b["ops"], 1):
        lines.append(f"| {i} | {_pct(op['share_of_A'])} | {op['day_index']} | {op['time_of_day']} "
                     f"| {op.get('category', '–')} |")
    lines += [
        "",
        "День рахується від опівночі дня початку епізоду (0 – день початку). Операцію, заплановану раніше "
        f"за початок епізоду, переносять на добу пізніше, а ту, що виходить за "
        f"{m['threat_model']['params']['window_days']}-денне вікно, – на добу "
        "раніше; година доби зберігається. Суми обрізаються до меж однієї операції й округлюються до "
        "центів, тож фактична сума може трохи відрізнятися від частки A.",
        "",
        "## Якість моделі на чистих даних (при τ)",
        "",
        "| період | recall | FPR | precision | операцій | шахрайських |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in ("val", "test"):
        q = data["clean_quality"][p]
        lines.append(f"| {p} | {_num(q['recall'], 4)} | {_num(q['fpr'], 4)} | {_num(q['precision'], 4)} "
                     f"| {q['n']} | {q['n_fraud']} |")
    v = m["versions"]
    lines += [
        "",
        "## Метадані",
        "",
        f"- seed {m['seed']}; git `{m['git_sha']}` (незакомічені зміни: {m['git_dirty']})",
        f"- платформа: {m['platform']['platform']}; процесів: {m['n_jobs']}; час аудиту {m['wall_seconds']} с"
        + (f" (з завантаженням даних і моделі {m['wall_seconds_with_loading']} с)"
           if "wall_seconds_with_loading" in m else ""),
        f"- модель: `{m['model']['dir']}`, τ = {m['model']['tau']}, SHA-256 `model.json`: "
        f"`{m['model']['model_json_sha256']}`",
        f"- модель загроз: SHA-256 `{m['threat_model']['sha256']}`",
        "- версії: " + ", ".join(f"{k} {val}" for k, val in v.items()),
        "",
    ]
    return "\n".join(lines)


def write_outputs(r: AuditResult, out_dir: Path | str, merchants: MerchantTable | None = None) -> dict:
    """``out_dir/audit.json`` (every field, the best strategy as a dict) and ``out_dir/summary.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = to_json(r, merchants)
    (out / "audit.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(data), encoding="utf-8")
    return data
