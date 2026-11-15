"""Round-based adversarial retraining (``fraudaudit harden``; design §11, rulings R7, R20).

Data (rulings R20, test isolation): the fit-period fraud episodes are split by card into H and S
(``fit_split(…, FIT_SPLIT_SEED)``); the gate reconnoitres only on S, hardening only ever touches H.
H is split by card again (``split_cards(H, seed)``) into ``recon`` cards (the attacker's reconnaissance
during generation) and ``host`` cards (where the found strategies are executed and become training rows).
Nothing of the ``val`` or ``test`` periods is generated, inserted or trained on; ``val`` only serves for τ
(as in ``train``) and for the mini-audit; ``test`` is not read at all by the hardening decisions.

Host slots (``host_slots``): every host card has one real fit episode. Adversarial episodes are realised on
the host card at start times ``real start + m·(window_days + 30) days`` (m = ±1, ±2, …) with the target sum
A of the real episode, keeping a slot only if no fraud operation of the card lies in
``[start − 30 d, start + window_days + 30 d)``, the full 30-day history lies in the data, and
``start + window_days + 30 d ≤ VAL_START``. Hence episodes of one card never share a 30-day aggregate window
(neither with each other nor with real fraud), so the context the attacker saw at generation time (the card's
history before the slot, as in the audit) is exactly what the batch features see after insertion, and no
``val`` row's aggregates change. Several attacks on the one real episode start would pile up in one window
and inflate the count aggregates of every inserted row.

One round r (1 … ``rounds``):

1. generation (process pool over the units, ruling R7; each worker loads the current model): for every
   unit (ES seed ``s`` ∈ ``GEN_SEEDS``, ``max_declines`` ∈ ``GEN_MAX_DECLINES``, 16 units) a (μ+λ) ES with
   the gate's ES parameters and the budget ``HARDEN_BUDGET_OPS`` reconnoitres on the recon contexts against
   the CURRENT model (the bank blocks after the unit's ``max_declines``); its best strategy is realised on the
   unit's slots (slot i belongs to unit ``(i + r − 1) mod 16``, so every slot gets a fresh episode each round
   and the units rotate) and executed with the bank's reaction; every executed operation up to the blocking
   decline becomes a candidate row (``is_fraud = 1``, owner fields from the card, real merchant/category and
   merchant coordinates of the realised operation, ``trans_num = adv-r{generation round}-…``);
2. selection: ``quota = round(share · real fraud rows of fit)``. Every slot holds one adversarial episode in
   the training stream: the episode of a uniformly drawn generation round 1 … r (a replay buffer, so a later
   model still sees the strategies that beat the earlier models); slots are taken in a random order, whole
   episodes, the last one cut to a time-ordered prefix, until ``quota`` rows; if fewer rows exist, all are
   taken and the shortfall is reported;
3. insertion and retraining: the rows are inserted into the ORIGINAL stream (``insert_rows``), features are
   recomputed with the batch path, and the whole pipeline is retrained with ``train_model`` (same procedure,
   same seed: encoder, scaler, IsolationForest, MLP; τ at FPR 1 % on clean val);
4. mini-audit: the gate's ES with ``attack_seeds`` restricted to ``MINI_AUDIT_SEEDS``, the gate's threat model
   and budget, targets = every ``val`` fraud episode, spares = every H episode (``audit_episodes``), one-sided
   uplift as in the gate; clean recall/FPR on ``val``. Stop when the worst upper bound ≤ δ or after
   ``rounds`` rounds.

Round 0 is the same mini-audit on the input model (the starting point of the curve). Every round's model is
saved to ``out/round-{r}/``; ``out/harden-report.json`` and ``out/harden-report.md`` hold the per-round
numbers. Only ES is used for hardening; the gate's random search never is, so it partially acts as a held-out
attack.

Determinism (R7): unit (s, md) of round r: oracle ``default_rng([seed, 5, r, s, md, 1])``, ES
``default_rng([seed, 5, r, s, md, 2])``; slot i realisation ``default_rng([seed, 6, r, i])``; selection
``default_rng([seed, 7, r])``; host split ``split_cards(H, seed)``. Results are identical for any ``n_jobs``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FIT_SPLIT_SEED, VAL_START, Episode, fit_split, fit_split_meta, fraud_episodes, period, split_cards
from .episode import run_episode
from .features.aggregates import WINDOW_DAYS
from .oracle import ReconOracle
from .serving.core import Serving
from .threat import EpisodeContext, MerchantTable, ThreatModel, contexts_for, feasible_ops, realize

HARDEN_BUDGET_OPS = 20_000
GEN_SEEDS = tuple(range(1, 9))
GEN_MAX_DECLINES = (1, 3)
SHARE = 0.5
ROUNDS = 3
MINI_AUDIT_SEEDS = (1, 2, 3)
AGG_MEMORY_DAYS = max(WINDOW_DAYS)  # 30: the longest aggregate window
_STREAM_GEN, _STREAM_SLOT, _STREAM_SELECT = 5, 6, 7
_DAY = pd.Timedelta(days=1)
OWNER_COLUMNS = ("first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job",
                 "dob")


# --- data: H split, host slots, rows ------------------------------------------------------------------------


def generation_split(h: list[Episode], seed: int) -> tuple[list[Episode], list[Episode]]:
    """(recon, hosts): the H episodes split by card 50/50 (``split_cards``; hosts get the larger half)."""
    hosts, recon = split_cards(h, seed)
    return recon, hosts


def host_slots(df: pd.DataFrame, hosts: list[Episode], tm: ThreatModel) -> list[Episode]:
    """Synthetic episodes (``rows=()``) on the host cards, see the module docstring; sorted by (card, start).
    A card with several host episodes (not in Sparkov: one fit episode per card) keeps its slots greedily in time
    order, at least ``window_days + 30`` days apart."""
    step = pd.Timedelta(days=tm.window_days + AGG_MEMORY_DAYS)
    mem = pd.Timedelta(days=AGG_MEMORY_DAYS)
    first = pd.Timestamp(df["ts"].min()).normalize() + mem
    last = VAL_START - step  # start + window + 30 d <= VAL_START
    fraud = df[df["is_fraud"] == 1]
    fraud_ts = {c: g["ts"].to_numpy(dtype="datetime64[ns]") for c, g in
                fraud[fraud["cc_num"].isin({e.card for e in hosts})].groupby("cc_num")}
    out = []
    for e in hosts:
        f = fraud_ts.get(e.card, np.array([], dtype="datetime64[ns]"))
        start0 = pd.Timestamp(e.start)
        lo = -int((start0 - first) // step) - 1
        hi = int((last - start0) // step) + 1
        for m in range(lo, hi + 1):
            if m == 0:
                continue
            s = start0 + m * step
            if not first <= s <= last:
                continue
            a, b = np.datetime64(s - mem), np.datetime64(s + step)
            if ((f >= a) & (f < b)).any():
                continue
            out.append(Episode(card=e.card, start=s, amount=e.amount, rows=()))
    out.sort(key=lambda x: (x.card, x.start, x.amount))
    kept: list[Episode] = []  # a card with several H episodes: keep slots in time order, >= step apart
    for x in out:
        if not kept or kept[-1].card != x.card or x.start - kept[-1].start >= step:
            kept.append(x)
    out = kept
    for x, y in zip(out, out[1:]):
        assert x.card != y.card or y.start - x.start >= step
    return out


def owner_rows(df: pd.DataFrame, cards) -> dict[int, dict]:
    """Owner fields of every card (its first row; ``profile_of`` checks they are constant)."""
    first = df[df["cc_num"].isin(set(cards))].drop_duplicates("cc_num", keep="first")
    return {int(r["cc_num"]): {c: r[c] for c in OWNER_COLUMNS} for _, r in first.iterrows()}


def adversarial_frame(df: pd.DataFrame, selected: list[dict], slots: list[Episode]) -> pd.DataFrame:
    """Rows of the selected adversarial operations with every column of ``df`` (``row`` and ``part`` are set by
    ``insert_rows``). ``selected`` items: ``{"slot", "round", "ops": [op dicts]}``."""
    owners = owner_rows(df, {slots[x["slot"]].card for x in selected})
    recs = []
    for x in selected:
        card = slots[x["slot"]].card
        for j, op in enumerate(x["ops"]):
            ts = pd.Timestamp(op["ts"])
            recs.append({"trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"), "cc_num": card,
                         "merchant": op["merchant"], "category": op["category"], "amt": float(op["amt"]),
                         **owners[card], "trans_num": f"adv-r{x['round']}-slot{x['slot']}-op{j}",
                         "unix_time": int(ts.value // 1_000_000_000), "merch_lat": float(op["merch_lat"]),
                         "merch_long": float(op["merch_long"]), "is_fraud": 1, "ts": ts})
    cols = [c for c in df.columns if c not in ("row", "part")]
    if not recs:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame.from_records(recs)
    return out[cols].astype({c: df[c].dtype for c in cols if c in df}, errors="ignore")


def insert_rows(df: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """``df`` plus ``adv`` (``part = "train"``, new ``row`` ids after the largest one), sorted stably by
    (ts, row) as ``data.finalize``; every original row keeps its values and ``row`` id."""
    if len(adv) and (pd.to_datetime(adv["ts"]) >= VAL_START).any():
        raise ValueError("adversarial rows must lie in the fit period (ts < VAL_START)")
    a = adv.copy()
    a["part"] = "train"
    a["row"] = np.arange(len(a), dtype="int64") + int(df["row"].max()) + 1
    out = pd.concat([df, a[df.columns]], ignore_index=True)
    return out.sort_values(["ts", "row"], kind="stable").reset_index(drop=True)


# --- generation ----------------------------------------------------------------------------------------------


def units(gen_seeds=GEN_SEEDS, gen_max_declines=GEN_MAX_DECLINES) -> list[tuple[int, int]]:
    """Generation units (ES seed, max_declines): max_declines outer, seeds inner."""
    return [(int(s), int(md)) for md in gen_max_declines for s in gen_seeds]


def _op_dict(tx, declined: bool) -> dict:
    return {"ts": pd.Timestamp(tx.ts).isoformat(), "amt": float(tx.amt), "merchant": tx.merchant,
            "category": tx.category, "merch_lat": float(tx.merch_lat), "merch_long": float(tx.merch_long),
            "declined": bool(declined)}


def _run_unit(serving, recon: list[EpisodeContext], slot_ctx: list[EpisodeContext], tm: ThreatModel,
              merchants: MerchantTable, seed: int, r: int, unit: tuple[int, int], slot_ids: list[int],
              budget_ops: int) -> dict:
    """ES of one unit against ``serving`` on the recon contexts, its best strategy executed on ``slot_ids``."""
    from .attacks.evolution import EvolutionStrategy
    s, md = unit
    tm_u = dataclasses.replace(tm, max_declines=md, recon_budget_ops=int(budget_ops))
    with ReconOracle(serving, recon, tm_u, merchants, tm_u.recon_budget_ops,
                     np.random.default_rng([seed, _STREAM_GEN, r, s, md, 1]), n_jobs=1) as oracle:
        run = EvolutionStrategy.from_params(tm.es).run(oracle, tm_u, np.random.default_rng([seed, _STREAM_GEN, r, s,
                                                                                           md, 2]))
    if run.best is None:
        raise ValueError(f"harden budget {budget_ops} did not allow a single evaluation")
    best = run.best.clip(tm_u, len(merchants))
    episodes, n_ok, n_ops = [], 0, 0
    for i in slot_ids:
        ctx = slot_ctx[i]
        txs = realize(best, ctx, tm_u, merchants, np.random.default_rng([seed, _STREAM_SLOT, r, i]))
        o = run_episode(serving, ctx, txs, md, tm_u.rho, tau_after_decline=tm_u.tau_after_decline)
        ex = txs[:o.n_ops]
        n_ok += sum(feasible_ops(ex, ctx, tm_u, merchants))
        n_ops += len(ex)
        episodes.append({"slot": int(i), "round": int(r), "ver": float(o.ver), "blocked": bool(o.blocked),
                         "ops": [_op_dict(t, d) for t, d in zip(ex, o.decisions)]})
    from .audit import strategy_dict
    return {"seed": s, "max_declines": md, "best_fitness": float(run.best_fitness), "evals": int(run.evals),
            "ops_used": int(run.ops_used), "n_slots": len(slot_ids), "n_ops": int(n_ops),
            "n_passed": int(sum(not op["declined"] for e in episodes for op in e["ops"])),
            "n_feasible_ops": int(n_ok), "mean_ver": float(np.mean([e["ver"] for e in episodes])) if episodes
            else float("nan"), "best": strategy_dict(best, merchants), "episodes": episodes}


_WORKER: dict = {}


def _init_worker(model_dir, recon, slot_ctx, tm, merchants, seed, r, assign, budget_ops) -> None:
    from .audit import _load_serving
    _WORKER.update(serving=_load_serving(model_dir), recon=recon, slot_ctx=slot_ctx, tm=tm, merchants=merchants,
                   seed=seed, r=r, assign=assign, budget_ops=budget_ops)


def _unit_job(job: tuple[int, tuple[int, int]]) -> dict:
    u, unit = job
    w = _WORKER
    return _run_unit(w["serving"], w["recon"], w["slot_ctx"], w["tm"], w["merchants"], w["seed"], w["r"], unit,
                     w["assign"][u], w["budget_ops"])


def slot_assignment(n_slots: int, n_units: int, r: int) -> list[list[int]]:
    """Slot ids of every unit in round ``r``: slot i belongs to unit ``(i + r − 1) mod n_units``."""
    out: list[list[int]] = [[] for _ in range(n_units)]
    for i in range(n_slots):
        out[(i + r - 1) % n_units].append(i)
    return out


def generate_round(model, recon: list[EpisodeContext], slot_ctx: list[EpisodeContext], tm: ThreatModel,
                   merchants: MerchantTable, seed: int, r: int, *, unit_list: list[tuple[int, int]],
                   budget_ops: int = HARDEN_BUDGET_OPS, n_jobs: int = 1,
                   model_dir: Path | str | None = None) -> list[dict]:
    """Every unit of round ``r`` (see the module docstring); a list of unit results in ``unit_list`` order.
    ``n_jobs > 1`` runs the units in a spawn process pool (each worker loads the model from ``model_dir``)."""
    assign = slot_assignment(len(slot_ctx), len(unit_list), r)
    jobs = list(enumerate(unit_list))
    if n_jobs > 1 and len(jobs) > 1:
        if model_dir is None:
            raise ValueError("n_jobs > 1 needs model_dir: every worker loads its own model")
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)), mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(str(model_dir), recon, slot_ctx, tm, merchants, seed, r, assign,
                                           budget_ops)) as ex:
            return list(ex.map(_unit_job, jobs))
    serving = Serving(model)
    return [_run_unit(serving, recon, slot_ctx, tm, merchants, seed, r, unit, assign[u], budget_ops)
            for u, unit in jobs]


def select_rows(pool: dict[int, dict[int, dict]], r: int, quota: int, seed: int) -> list[dict]:
    """Episodes for the training stream of round ``r`` (step 2 of the module docstring).

    ``pool[g][slot]`` is the episode generated in round g for that slot. One generation round per slot is drawn
    uniformly from those with a non-empty episode; slots in a random order; whole episodes until ``quota`` rows,
    the last one cut to a prefix. Returns ``[{"slot", "round", "ops"}]`` with at most ``quota`` ops in total."""
    rng = np.random.default_rng([seed, _STREAM_SELECT, r])
    slots = sorted({i for g in pool.values() for i in g})
    choice = {}
    for i in slots:
        rounds = [g for g in sorted(pool) if i in pool[g] and pool[g][i]["ops"]]
        k = int(rng.integers(0, len(rounds))) if rounds else -1
        if rounds:
            choice[i] = rounds[k]
    order = rng.permutation(len(slots))
    out, left = [], int(quota)
    for j in order:
        i = slots[int(j)]
        if left <= 0:
            break
        if i not in choice:
            continue
        ops = pool[choice[i]][i]["ops"][:left]
        out.append({"slot": i, "round": choice[i], "ops": ops})
        left -= len(ops)
    return out


# --- rounds ------------------------------------------------------------------------------------------------


@dataclass
class RoundReport:
    round: int
    model_dir: str | None
    tau: float
    val_quality: dict                      # recall/FPR/precision on clean val at tau
    val_uplift: dict                       # {attack_seed: {"mean", "lower", "upper"}}, one-sided upper
    val_uplift_worst_upper: float
    val_uplift_mean: float                 # mean over the mini-audit seeds of the uplift means
    val_ver_adv: dict                      # {attack_seed: mean VER_adv}
    val_ver_clean: float
    n_adv_rows: int = 0                    # adversarial rows in this round's training stream
    quota: int = 0
    shortfall: int = 0                     # quota − n_adv_rows when fewer rows were generated
    n_generated_rows: int = 0              # rows generated in this round (all slots, all units)
    n_generated_passed: int = 0            # of which passed at generation time
    n_rows_by_generation_round: dict = field(default_factory=dict)
    n_slots: int = 0
    units: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)
    train_report: dict = field(default_factory=dict)


def _mini_audit(df, model, tm, gate_cfg, seed, val_eps, h_eps, *, n_jobs, model_dir, seeds, threat_path) -> dict:
    from .attacks.evolution import EvolutionStrategy
    from .audit import audit_episodes
    res = {}
    for s in seeds:
        res[int(s)] = audit_episodes(df, model, tm, EvolutionStrategy.from_params(tm.es), seed, val_eps, h_eps,
                                     spares_period="harden-H", one_sided=gate_cfg.one_sided, n_jobs=n_jobs,
                                     model_dir=model_dir, target_periods=("val",), threat_path=threat_path,
                                     attack_seed=int(s), confidence=gate_cfg.confidence, quality_periods=("val",))
    return res


def _round_metrics(res: dict) -> dict:
    up = {s: {"mean": r.uplift[0], "lower": r.uplift[1], "upper": r.uplift[2]} for s, r in res.items()}
    first = next(iter(res.values()))
    return {"val_uplift": up, "val_uplift_worst_upper": max(v["upper"] for v in up.values()),
            "val_uplift_mean": float(np.mean([v["mean"] for v in up.values()])),
            "val_ver_adv": {s: r.ver_adv[0] for s, r in res.items()}, "val_ver_clean": first.ver_clean[0],
            "val_quality": first.clean_quality["val"]}


def harden(df: pd.DataFrame, model, tm: ThreatModel, gate_cfg, rounds: int = ROUNDS, share: float = SHARE,
           seed: int = 42, *, model_dir: Path | str, out_dir: Path | str, n_jobs: int | None = None,
           budget_ops: int = HARDEN_BUDGET_OPS, epochs: int | None = None, gen_seeds=GEN_SEEDS,
           gen_max_declines=GEN_MAX_DECLINES, mini_audit_seeds=MINI_AUDIT_SEEDS, baseline_audit: bool = True,
           threat_path: Path | str | None = None, log=None) -> tuple[object, list[RoundReport], dict]:
    """Round-based adversarial retraining (module docstring). ``model``/``model_dir`` is the input model.
    Returns (final model, round reports – round 0 first when ``baseline_audit`` –, info dict with the data
    split, the augmented stream of the last round (``info["df"]``) and the selected rows)."""
    from .model.pipeline import EPOCHS, FraudModel, train_model
    say = log or (lambda m: None)
    jobs = max(1, int(n_jobs if n_jobs is not None else (os.cpu_count() or 1)))
    epochs = EPOCHS if epochs is None else int(epochs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    unit_list = units(gen_seeds, gen_max_declines)

    h_eps, s_eps = fit_split(fraud_episodes(df, {"fit"}), FIT_SPLIT_SEED)
    recon_eps, host_eps = generation_split(h_eps, seed)
    slots = host_slots(df, host_eps, tm)
    if not slots:
        raise ValueError("no host slots: the fit period is too short for adversarial episodes")
    val_eps = fraud_episodes(df, {"val"})
    recon_ctx = contexts_for(df, recon_eps)
    slot_ctx = contexts_for(df, slots)
    merchants = MerchantTable.from_df(df)
    real_fraud_fit = int(((period(df) == "fit") & (df["is_fraud"] == 1)).sum())
    quota = int(round(share * real_fraud_fit))
    info = {"fit_split": fit_split_meta(h_eps, s_eps, FIT_SPLIT_SEED),
            "h_cards": sorted({e.card for e in h_eps}), "s_cards": sorted({e.card for e in s_eps}),
            "recon_cards": sorted({e.card for e in recon_eps}), "host_cards": sorted({e.card for e in host_eps}),
            "n_recon_episodes": len(recon_eps), "n_host_episodes": len(host_eps), "n_slots": len(slots),
            "n_val_targets": len(val_eps), "real_fraud_rows_fit": real_fraud_fit, "quota": quota,
            "units": [{"seed": s, "max_declines": md} for s, md in unit_list], "budget_ops": int(budget_ops),
            "epochs": epochs, "share": float(share), "seed": int(seed), "n_jobs": jobs,
            "mini_audit_seeds": [int(s) for s in mini_audit_seeds]}
    say(f"H {len(h_eps)} episodes ({len(recon_eps)} recon, {len(host_eps)} hosts), S {len(s_eps)}; "
        f"{len(slots)} host slots; quota {quota} rows (share {share} of {real_fraud_fit})")

    reports: list[RoundReport] = []
    if baseline_audit:
        t0 = time.perf_counter()
        res = _mini_audit(df, model, tm, gate_cfg, seed, val_eps, h_eps, n_jobs=jobs, model_dir=model_dir,
                          seeds=mini_audit_seeds, threat_path=threat_path)
        m = _round_metrics(res)
        reports.append(RoundReport(round=0, model_dir=str(model_dir), tau=float(model.tau),
                                   seconds={"mini_audit": round(time.perf_counter() - t0, 1)}, **m))
        say(f"round 0: val uplift worst upper {m['val_uplift_worst_upper']:.4f}, mean {m['val_uplift_mean']:.4f}")

    pool: dict[int, dict[int, dict]] = {}
    cur_model, cur_dir = model, Path(model_dir)
    selected: list[dict] = []
    df_r = df
    for r in range(1, int(rounds) + 1):
        t0 = time.perf_counter()
        gen = generate_round(cur_model, recon_ctx, slot_ctx, tm, merchants, seed, r, unit_list=unit_list,
                             budget_ops=budget_ops, n_jobs=jobs, model_dir=cur_dir)
        pool[r] = {e["slot"]: e for u in gen for e in u["episodes"]}
        n_gen = sum(len(e["ops"]) for e in pool[r].values())
        n_pass = sum(not op["declined"] for e in pool[r].values() for op in e["ops"])
        t_gen = time.perf_counter() - t0
        say(f"round {r}: generated {n_gen} rows ({n_pass} passed) on {len(pool[r])} slots in {t_gen:.0f} s")

        selected = select_rows(pool, r, quota, seed)
        adv = adversarial_frame(df, selected, slots)
        df_r = insert_rows(df, adv)
        t1 = time.perf_counter()
        new_model, rep = train_model(df_r, seed=seed, epochs=epochs)
        t_train = time.perf_counter() - t1
        rdir = out / f"round-{r}"
        by_round = {str(g): sum(len(x["ops"]) for x in selected if x["round"] == g) for g in range(1, r + 1)}
        # the test-period quality that train_model reports is dropped: hardening never looks at test
        meta = dict({k: v for k, v in rep.items() if k != "test"}, harden={"round": r, "input_model": str(model_dir), "n_adv_rows": len(adv),
                                 "quota": quota, "n_rows_by_generation_round": by_round,
                                 "fit_split": info["fit_split"], "seed": int(seed), "budget_ops": int(budget_ops)},
                    platform={"system": platform.system(), "machine": platform.machine(),
                              "platform": platform.platform()})
        from .cli import git_info
        meta.update(git_info())
        new_model.save(rdir, meta=meta)
        (rdir / "adversarial-rows.csv").write_text(adv.to_csv(index=False), encoding="utf-8")
        say(f"round {r}: trained on {len(adv)} adversarial rows in {t_train:.0f} s, tau {new_model.tau:.6g}, "
            f"val recall {rep['val']['recall']:.4f} FPR {rep['val']['fpr']:.4f}")
        from .model.artifacts import load_artifacts
        cur_model, cur_dir = FraudModel.from_artifacts(load_artifacts(rdir)), rdir
        t2 = time.perf_counter()
        res = _mini_audit(df, cur_model, tm, gate_cfg, seed, val_eps, h_eps, n_jobs=jobs, model_dir=cur_dir,
                          seeds=mini_audit_seeds, threat_path=threat_path)
        m = _round_metrics(res)
        t_aud = time.perf_counter() - t2
        unit_rows = [{k: v for k, v in u.items() if k != "episodes"} for u in gen]
        reports.append(RoundReport(
            round=r, model_dir=str(rdir), tau=float(cur_model.tau), n_adv_rows=len(adv), quota=quota,
            shortfall=max(0, quota - len(adv)), n_generated_rows=n_gen, n_generated_passed=n_pass,
            n_rows_by_generation_round=by_round, n_slots=len(slots), units=unit_rows,
            seconds={"generation": round(t_gen, 1), "training": round(t_train, 1), "mini_audit": round(t_aud, 1),
                     "total": round(time.perf_counter() - t0, 1)},
            train_report={k: rep[k] for k in ("rows", "fraud_rows", "train_balanced_rows", "val", "seconds")},
            **m))
        say(f"round {r}: val uplift worst upper {m['val_uplift_worst_upper']:.4f}, mean {m['val_uplift_mean']:.4f} "
            f"({t_aud:.0f} s)")
        if gate_cfg.delta is not None and m["val_uplift_worst_upper"] <= gate_cfg.delta:
            say(f"round {r}: val uplift upper bound <= delta {gate_cfg.delta}: stop")
            break
    info.update(df=df_r, selected=selected)
    return cur_model, reports, info


# --- report --------------------------------------------------------------------------------------------------


def _sha(path) -> str | None:
    return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_json(reports: list[RoundReport], info: dict, meta: dict) -> dict:
    return {"rounds": [dataclasses.asdict(r) for r in reports],
            "final_round": reports[-1].round, "final_model_dir": reports[-1].model_dir,
            "data": {k: v for k, v in info.items() if k not in ("df", "selected")}, "meta": meta}


def _pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.1f} %"


def report_markdown(data: dict) -> str:
    d, m = data["data"], data["meta"]
    lines = [
        "# Adversarial hardening (round-based retraining)",
        "",
        f"Input model `{m['input_model']}`, threat model `{m['threat_file']}`, gate `{m['gate_file']}` "
        f"(δ = {m['delta']}). Rounds run: {len(data['rounds']) - (1 if data['rounds'][0]['round'] == 0 else 0)} "
        f"of {m['rounds']}; final model: `{data['final_model_dir']}`.",
        "",
        "Validation curve: the mini-audit runs the gate's ES (seeds "
        f"{d['mini_audit_seeds']}, gate threat model and budget) on all {d['n_val_targets']} fraud episodes of the "
        "val period, with the H episodes as spares; uplift = VER_adv − VER_clean, one-sided upper bound as in the "
        "gate. The test period is not used by hardening (no rows, no targets, no decisions).",
        "",
        "| round | adversarial rows (quota) | generated rows (passed) | val uplift mean | val uplift worst upper "
        "| val recall | val FPR | τ | time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in data["rounds"]:
        q = r["val_quality"]
        t = r["seconds"]
        tt = t.get("total", t.get("mini_audit"))
        adv = "–" if r["round"] == 0 else f"{r['n_adv_rows']} ({r['quota']})"
        gen = "–" if r["round"] == 0 else f"{r['n_generated_rows']} ({r['n_generated_passed']})"
        lines.append(f"| {r['round']} | {adv} | {gen} | {_pct(r['val_uplift_mean'])} "
                     f"| {_pct(r['val_uplift_worst_upper'])} | {q['recall']:.4f} | {q['fpr']:.4f} "
                     f"| {r['tau']:.6g} | {tt:.0f} s |")
    lines += [
        "",
        "Round 0 is the input model. Per round: adversarial rows in the training stream (one episode per host "
        "slot, drawn from the episodes of all rounds so far), rows generated in the round, mini-audit uplift "
        "(mean over the seeds and worst one-sided upper bound), clean val recall/FPR at τ (τ at 1 % FPR on clean "
        "val), wall time (generation + training + mini-audit).",
        "",
        "## Data",
        "",
        f"- Fit-period episodes split by card (seed {d['fit_split']['seed']}, ruling R20): H "
        f"{d['fit_split']['n_h_cards']} cards (hardening), S {d['fit_split']['n_s_cards']} cards (gate spares, "
        "never used here).",
        f"- H split by card (seed {d['seed']}): {d['n_recon_episodes']} recon episodes (attacker reconnaissance "
        f"during generation), {d['n_host_episodes']} host cards with {d['n_slots']} host slots (adversarial "
        "episodes realised at the real episode start ± multiples of window + 30 days, away from real fraud and "
        "ending ≥ 30 days before the val period).",
        f"- Quota: round({d['share']} × {d['real_fraud_rows_fit']} real fraud rows of fit) = {d['quota']} rows.",
        f"- Generation: {len(d['units'])} ES units (seeds × max_declines "
        f"{sorted({u['max_declines'] for u in d['units']})}), budget {d['budget_ops']} operations each, "
        f"{d['n_jobs']} processes; training epochs {d['epochs']}.",
        "",
        "## Attacks",
        "",
        "Only the evolution strategy generates hardening data. The gate's random search was never used for "
        "hardening, so in the final gate it is a partial held-out check; the held-out stepwise attack of the plan is "
        "not implemented yet.",
        "",
        "## Metadata",
        "",
        f"- seed {m['seed']}; git `{m['git_sha']}` (uncommitted changes: {m['git_dirty']}); "
        f"platform {m['platform']}; wall time {m['wall_seconds']} s",
        f"- threat model SHA-256 `{m['threat_sha256']}`, gate SHA-256 `{m['gate_sha256']}`",
        "- versions: " + ", ".join(f"{k} {v}" for k, v in m["versions"].items()),
        "",
    ]
    return "\n".join(lines)


def write_report(out_dir, reports, info, meta) -> dict:
    data = report_json(reports, info, meta)
    out = Path(out_dir)
    (out / "harden-report.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
                                            encoding="utf-8")
    (out / "harden-report.md").write_text(report_markdown(data), encoding="utf-8")
    return data
