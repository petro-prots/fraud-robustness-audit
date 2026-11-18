"""``fraudaudit policy-sweep``: the bank-policy grid of Task 19 (rulings R18, R19, R30) – the same pre-registered
policies on several models, measured by the gate's statistic and by the friction they cause.

Grid per model (one cell = one combination; the gate's ensemble – attacks × attack seeds of ``gate.yaml`` – on all
test targets with the fit-period spares):

* τ from clean val at a target FPR (``threshold_at_fpr``; default 0.005, 0.01 = the model's own, 0.02); the τ at the
  model's own target FPR must reproduce ``model.json`` (checked for every model, whatever the grid);
* bank reaction ``md1`` (``max_declines: 1``), ``md3`` (``max_declines: 3``, no escalation), ``md3+esc``
  (``max_declines: 3``, ``tau_after_decline`` = τ at val FPR ``esc_fpr`` = 0.05 of the same model, which must be below
  every τ of the grid); ``md2`` and ``mdinf`` (never blocked) are the extra cells of the R19 experiment;
* corridor ``off`` / ``on`` (the corridor section of ``--corridor-threat``, the gate's current corridor);
* corridor strength (Task 19b, ruling R32): optional variants of the corridor-on cells, κ of the V1 daily split ×
  quantile q (``kappas`` / ``qs``, applied to the corridor of ``--corridor-threat`` by ``dataclasses.replace``; every
  other corridor parameter as in the file; the population floor is refitted per q). A variant equal to the file's
  (κ, q) is the file's corridor itself: its cell key and config hash are exactly those of a run without variants, so
  Task 19 records stay valid; another variant adds ``|kappa=…|q=…`` to the key and its parameters to the hash.

Per cell: U = worst (over the runs) one-sided upper bound of the paired uplift VER_adv − VER_clean (the gate's
statistic) with the mean uplift and the per-run table; U$ = the same statistic on dollars per target,
(VER_adv,i − VER_clean,i) · A_i, same paired bootstrap (not U × mean A); clean test recall / FPR at the cell's τ;
per run the per-target VERs (for R19 and the seed spread). Friction on the legitimate test stream (a separate stage,
from the p of every legitimate test operation cached once per model): false blocks (``metrics.false_block_rate``,
30-day card windows) for (τ, reaction) and the owner-confirmation burden (``metrics.confirmation_burden``) for
corridor-on cells (0 when off). Corridor-on cells also carry the violations of the corridor bounds summed over their
runs (``n_corridor_runs``, ``n_bound_violations``, ``n_headroom_violations``, ``n_block_violations``, from the audit's
corridor meta, as in ``corridor_sweep``) and their effective corridor parameters with a SHA-256.

Resumable: the JSON is rewritten (atomically) after every cell and every friction entry; a rerun skips a cell whose
config hash (model, τ, reaction, corridor, threat model, gate ensemble, seed) is unchanged and reruns it otherwise.
Cells already in the JSON but not requested are kept (the R19 cells and the grid share one file).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import corridor as corridor_mod
from .sweep_cells import check_own_tau, model_at_tau, model_target_fpr, run_cell, tau_at, val_scores, write_json

FPR_GRID = (0.005, 0.01, 0.02)
REACTIONS = ("md1", "md3", "md3+esc")
R19_REACTIONS = ("md2", "mdinf")
CORRIDORS = ("off", "on")
ESC_FPR = 0.05
FB_WINDOW_DAYS = 30
SWEEP_FILE = "policy-sweep.json"
SCORES_FILE = "legit-scores.npz"


def parse_reaction(name: str) -> dict:
    """``md<N>`` | ``mdinf`` with an optional ``+esc`` → ``{"max_declines": N | None, "esc": bool}``."""
    n = name.strip().lower()
    esc = n.endswith("+esc")
    base = n[:-4] if esc else n
    if base == "mdinf":
        md = None
    elif base.startswith("md") and base[2:].isdigit() and int(base[2:]) >= 1:
        md = int(base[2:])
    else:
        raise ValueError(f"bad reaction {name!r}: expected md<N> (N >= 1) or mdinf, optionally +esc")
    if esc and md == 1:
        raise ValueError("escalation has no effect with md1 (the first decline blocks the card)")
    return {"max_declines": md, "esc": esc}


def cell_key(fpr: float, reaction: str, corridor: str, variant: tuple | None = None) -> str:
    """``fpr=…|<reaction>|corridor=<off|on>``; a corridor variant other than the file's adds ``|kappa=…|q=…``."""
    k = f"fpr={float(fpr):g}|{reaction}|corridor={corridor}"
    if variant is not None:
        kappa, q = variant
        k += ("" if kappa is None else f"|kappa={float(kappa):g}") + f"|q={float(q):g}"
    return k


def corridor_variants(file_corridor, kappas=None, qs=None) -> list[tuple]:
    """(κ, q) of every corridor variant: ``kappas`` × ``qs``, each defaulting to the file's value. ValueError when κ
    is given for a corridor without the V1 weekly split."""
    if kappas and file_corridor.daily_mode != "weekly_split":
        raise ValueError("corridor kappa variants need the V1 daily corridor (daily: {mode: weekly_split})")
    ks = [float(k) for k in kappas] if kappas else [file_corridor.kappa]
    qq = [float(q) for q in qs] if qs else [float(file_corridor.q)]
    out = []
    for k in ks:
        for q in qq:
            if (k, q) not in out:
                out.append((k, q))
    return out


def variant_id(file_corridor, kappa, q) -> tuple | None:
    """None for the file's own (κ, q) (today's key and hash), else ``(κ, q)``."""
    fk = None if file_corridor.kappa is None else float(file_corridor.kappa)
    k = None if kappa is None else float(kappa)
    return None if (k, float(q)) == (fk, float(file_corridor.q)) else (k, float(q))


def params_sha256(public: dict | None) -> str | None:
    return None if public is None else hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def usd_uplift(rec: dict, confidence: float, seed: int, one_sided: bool, n_boot: int) -> None:
    """Adds per run ``usd_uplift_mean`` / ``usd_uplift_upper``: the paired bootstrap of (VER_adv,i − VER_clean,i) · A_i
    (the gate's ``uplift_ci`` on dollars, same seed and level), and ``worst_usd_uplift_upper`` / ``worst_run_usd``."""
    from .metrics import uplift_ci
    a = np.asarray(rec["amounts"], dtype=np.float64)
    for r in rec["runs"].values():
        m, _, hi = uplift_ci(np.asarray(r["ver_adv"]) * a, np.asarray(r["ver_clean"]) * a, confidence, n_boot, seed,
                             one_sided=one_sided)
        r["usd_uplift_mean"], r["usd_uplift_upper"] = float(m), float(hi)
    bad = [k for k, r in rec["runs"].items() if not math.isfinite(r["usd_uplift_upper"])]
    if bad:
        raise RuntimeError(f"non-finite dollar uplift upper bound in runs {bad}")
    w = max(rec["runs"], key=lambda k: rec["runs"][k]["usd_uplift_upper"])
    rec["worst_run_usd"], rec["worst_usd_uplift_upper"] = w, rec["runs"][w]["usd_uplift_upper"]


def _threat_without_policy(tm) -> dict:
    d = dataclasses.asdict(dataclasses.replace(tm, corridor=None))
    for k in ("max_declines", "tau_after_decline", "corridor"):
        d.pop(k, None)
    return d


def _sha256(path) -> str | None:
    return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()


def machine_info() -> dict:
    """The machine of a run (R17: the whole grid on one machine). ``HOST_MACHINE`` (set by the chain script on the
    host) names the physical machine when the run is inside a container."""
    return {"system": platform.system(), "machine": platform.machine(), "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "host_machine": os.environ.get("HOST_MACHINE")}


def provenance() -> dict:
    """Code version and machine of one computed cell or friction entry (recorded per entry, ruling R17)."""
    from .cli import git_info
    m = machine_info()
    return {**git_info(), "host_machine": m["host_machine"], "platform": m["platform"], "cpu_count": m["cpu_count"]}


PROVENANCE_KEYS = ("git_sha", "git_dirty", "host_machine")


def check_reuse(prov: dict | None, cur: dict, what: str) -> None:
    """ValueError unless a finished entry may be reused by this run: same git SHA, dirty flag and host machine, and not
    computed from a dirty tree (its code cannot be identified). No silent mixing of code versions or machines."""
    old = {k: (prov or {}).get(k) for k in PROVENANCE_KEYS}
    new = {k: cur.get(k) for k in PROVENANCE_KEYS}
    if old != new or old["git_dirty"] is True:
        raise ValueError(f"{what} was computed with {old}, this run is {new}: refusing to reuse it (results of "
                         "different code or machines must not be mixed); use another --out or remove the entry")


def load_sweep(path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# --- legitimate-stream score cache --------------------------------------------------------------------------------


def save_scores(path, scores: dict, model_sha: str) -> None:
    cards = np.array(sorted(scores), dtype=np.int64)
    lens = np.array([len(scores[c]) for c in cards], dtype=np.int64)
    p = np.concatenate([scores[c] for c in cards]) if len(cards) else np.empty(0)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, cards=cards, lens=lens, p=p.astype(np.float64), model_sha=np.array(model_sha))
    tmp.replace(path)


def load_scores(path, model_sha: str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        if str(z["model_sha"]) != model_sha:
            return None
        cards, lens, vals = z["cards"], z["lens"], z["p"]
    out, i = {}, 0
    for c, n in zip(cards.tolist(), lens.tolist()):
        out[int(c)] = vals[i:i + n].copy()
        i += n
    return out


# --- the sweep ----------------------------------------------------------------------------------------------------


def policy_sweep(df, model, model_dir, label: str, tm, corridor_tm, cfg, *, fprs=FPR_GRID, reactions=REACTIONS,
                 corridors=CORRIDORS, kappas=None, qs=None, esc_fpr: float = ESC_FPR, r19: bool = False, seed: int = 42,
                 n_jobs: int | None = None, out_dir, ensemble: bool = True, friction: bool = True,
                 role: str = "grid", threat_path=None, corridor_path=None, gate_path=None, log=None) -> dict:
    """Run (or resume) the cells of one model; returns the sweep dict also written to ``out_dir/SWEEP_FILE``.
    ``kappas`` / ``qs``: corridor variants of the corridor-on cells (module docstring); None = the file's value."""
    from .audit import N_BOOT, _versions
    from .cli import git_info
    say = log or (lambda m: None)
    if role not in ("grid", "confirmation"):
        raise ValueError(f"role must be grid or confirmation, got {role!r}")
    specs = [(r, parse_reaction(r)) for r in reactions]
    for c in corridors:
        if c not in CORRIDORS:
            raise ValueError(f"corridor must be off or on, got {c!r}")
    if "on" in corridors:
        if corridor_tm is None or corridor_tm.corridor is None:
            raise ValueError("corridor 'on' needs a corridor threat model with a corridor section")
        if _threat_without_policy(corridor_tm) != _threat_without_policy(tm):
            raise ValueError("the corridor threat model differs from the threat model outside the corridor and the "
                             "bank reaction: the corridor cells would not be comparable")
    if (kappas or qs) and "on" not in corridors:
        raise ValueError("corridor variants (kappa / q) need corridor 'on' cells")
    variants = corridor_variants(corridor_tm.corridor, kappas, qs) if "on" in corridors else []
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SWEEP_FILE
    model_sha = hashlib.sha256((Path(model_dir) / "model.json").read_bytes()).hexdigest()
    old = load_sweep(path)
    if old is not None and old.get("model_json_sha256") != model_sha:
        raise ValueError(f"{path} belongs to another model (model.json SHA-256 {old.get('model_json_sha256')}); "
                         "use another --out or --label")
    if old is not None and old.get("role", "grid") != role:
        raise ValueError(f"{path} has role {old.get('role')!r}, this run {role!r}")
    prov = provenance()

    own_fpr = model_target_fpr(model_dir)
    if own_fpr is None:
        raise ValueError(f"{model_dir}/model.json has no meta.target_fpr: the tau reproduction check cannot run")
    y_val, p_val = val_scores(df, model)
    taus = {float(f): tau_at(y_val, p_val, f) for f in sorted({*map(float, fprs), own_fpr})}
    check_own_tau(model_dir, model, taus[own_fpr], own_fpr)
    tau_esc = tau_at(y_val, p_val, esc_fpr)
    grid_taus = [taus[float(f)] for f in fprs]
    if any(sp["esc"] for _, sp in specs) and not (0 < tau_esc < min(grid_taus)):
        raise ValueError(f"escalation threshold {tau_esc} (val FPR {esc_fpr}) must be in (0, every grid tau); grid "
                         f"taus {grid_taus}")
    say(f"{label}: own target FPR {own_fpr}, tau {model.tau:.6g}; grid taus "
        + ", ".join(f"{f:g}: {t:.6g}" for f, t in taus.items()) + f"; escalation tau {tau_esc:.6g}")

    tm_on = None
    if "on" in corridors or (old and any(c["corridor"] == "on" for c in old.get("cells", []))):
        if corridor_tm is not None and corridor_tm.corridor is not None:
            tm_on = corridor_mod.prepare(dataclasses.replace(tm, corridor=corridor_tm.corridor), df)

    from .features.columns import feature_set_name
    gate = {"attacks": list(cfg.attacks), "attack_seeds": list(cfg.attack_seeds), "target_fpr": cfg.target_fpr,
            "confidence": cfg.confidence, "one_sided": cfg.one_sided, "spares_period": cfg.spares_period,
            "delta": cfg.delta, "recall_clean_min": cfg.recall_clean_min, "n_boot": N_BOOT}
    if old is not None and old.get("gate") != gate:
        raise ValueError(f"{path} was made with the gate {old.get('gate')}, this run has {gate}: use the same --gate "
                         "or another --out")
    data = old or {"cells": [], "friction": {"false_blocks": {}, "burden": {}}}
    data.update(label=label, model_dir=str(model_dir), model_json_sha256=model_sha, role=role,
                feature_set=feature_set_name(model.feature_columns), n_features=len(model.feature_columns),
                own_target_fpr=own_fpr, own_tau=float(model.tau), esc_fpr=float(esc_fpr), tau_esc=tau_esc,
                taus={**(old or {}).get("taus", {}), **{f"{f:g}": t for f, t in taus.items()}}, seed=int(seed),
                gate=gate,
                corridor_params=None if tm_on is None else tm_on.corridor.public())
    cells = {c["key"]: c for c in data["cells"]}
    file_c = None if corridor_tm is None else corridor_tm.corridor
    prepared: dict = {}

    def tm_variant(var):
        """The prepared corridor threat model of a variant id (None = the file's corridor, ``tm_on`` itself)."""
        if var is None:
            return tm_on
        if var not in prepared:
            k, q = var
            c = dataclasses.replace(tm_on.corridor, q=float(q), **({} if k is None else {"kappa": float(k)}))
            prepared[var] = corridor_mod.prepare(dataclasses.replace(tm_on, corridor=c), df)  # floor refitted per q
        return prepared[var]

    def corridor_of(c):
        """The prepared corridor of a corridor-on cell (its stored κ / q; the file's for records without them)."""
        if c.get("corridor_q") is None:
            return tm_on.corridor
        cor = tm_variant(variant_id(file_c, c.get("corridor_kappa"), c["corridor_q"])).corridor
        if c.get("corridor_params") is not None and cor.public() != c["corridor_params"]:
            raise ValueError(f"cell {c['key']}: its corridor parameters differ from the corridor rebuilt from "
                             "--corridor-threat and its kappa / q")
        return cor

    def save():
        data["cells"] = sorted(cells.values(), key=lambda c: (c["target_fpr"], c["reaction"], c["corridor"], c["key"]))
        amounts = next((c["amounts"] for c in data["cells"] if c.get("amounts")), None)
        if amounts:
            data["mean_amount"] = float(np.mean(amounts))
            data["delta_usd"] = None if cfg.delta is None else float(cfg.delta * np.mean(amounts))
        _attach_friction(data)
        data["meta"] = {"seed": int(seed), **git_info(), "machine": machine_info(), "versions": _versions(),
                        "threat_file": None if threat_path is None else str(threat_path),
                        "threat_sha256": _sha256(threat_path),
                        "corridor_threat_file": None if corridor_path is None else str(corridor_path),
                        "corridor_threat_sha256": _sha256(corridor_path),
                        "gate_file": None if gate_path is None else str(gate_path), "gate_sha256": _sha256(gate_path),
                        "n_jobs": n_jobs, "wall_seconds_last_invocation": round(time.perf_counter() - t0, 1)}
        write_json(path, data)

    todo = [(float(f), r, sp, c, variant_id(file_c, *v) if c == "on" else None)
            for f in fprs for r, sp in specs for c in corridors for v in (variants if c == "on" else [None])]
    if r19:
        todo += [(own_fpr, r, parse_reaction(r), "off", None) for r in R19_REACTIONS]
    seen = set()
    todo = [t for t in todo if not (cell_key(t[0], t[1], t[3], t[4]) in seen
                                    or seen.add(cell_key(t[0], t[1], t[3], t[4])))]

    for fpr, rname, sp, corr, var in (todo if ensemble else []):
        key = cell_key(fpr, rname, corr, var)
        tau = taus[fpr]
        tau2 = tau_esc if sp["esc"] else None
        base = tm_variant(var) if corr == "on" else dataclasses.replace(tm, corridor=None)
        tm_c = dataclasses.replace(base, max_declines=sp["max_declines"], tau_after_decline=tau2)
        cpub = None if corr == "off" else tm_c.corridor.public()
        config = {"model_json_sha256": model_sha, "target_fpr": fpr, "tau": tau, "max_declines": sp["max_declines"],
                  "tau_after_decline": tau2, "corridor": cpub,
                  "threat": _threat_without_policy(tm), "gate": data["gate"], "seed": int(seed)}
        h = config_hash(config)
        if key in cells and cells[key].get("config_hash") == h:
            check_reuse(cells[key].get("provenance"), prov, f"{path} cell {key}")
            say(f"{label} {key}: done, skipped")
            continue
        t1 = time.perf_counter()
        mdir, mdl = model_at_tau(model_dir, model, tau, fpr, out / "models" / f"fpr-{fpr:g}",
                                 "policy-sweep: same weights, tau at target_fpr on val")
        rec, runs = run_cell(df, mdl, mdir, tm_c, cfg, seed, n_jobs=n_jobs, threat_path=threat_path,
                             per_target=True)
        usd_uplift(rec, cfg.confidence, seed, cfg.one_sided, N_BOOT)
        cell = {"key": key, "config_hash": h, "target_fpr": fpr, "tau": tau, "reaction": rname,
                "max_declines": sp["max_declines"], "tau_after_decline": tau2, "corridor": corr, **rec,
                "corridor_kappa": None if cpub is None else cpub["daily"].get("kappa"),
                "corridor_q": None if cpub is None else float(cpub["q"]),
                "corridor_params": cpub, "corridor_params_sha256": params_sha256(cpub),
                **violation_counts(runs, corr == "on"),
                "U": rec["worst_uplift_upper"], "U_usd": rec["worst_usd_uplift_upper"],
                "wall_seconds": round(time.perf_counter() - t1, 1), "provenance": prov}
        cells[key] = cell
        say(f"{label} {key}: U {100 * cell['U']:.1f} % ({cell['worst_run']}), U$ {cell['U_usd']:.0f} $, mean uplift "
            f"{100 * cell['mean_uplift']:.1f} %, test recall {cell['test_recall']:.4f}, {cell['wall_seconds']} s")
        save()

    if friction and cells:
        _friction(df, model, model_dir, data, cells, corridor_of, out, model_sha, n_jobs, say, save, prov)
    save()
    return data


VIOLATION_KEYS = ("n_bound_violations", "n_headroom_violations", "n_block_violations")


def violation_counts(runs: dict, corridor_on: bool) -> dict:
    """Corridor-bound violations summed over the runs of a cell from each run's ``meta["corridor"]`` (as in
    ``corridor_sweep``): ``n_corridor_runs`` and the three counters; for a corridor-off cell 0 runs and None."""
    if not corridor_on:
        return {"n_corridor_runs": 0, **{k: None for k in VIOLATION_KEYS}}
    metas = [r.meta.get("corridor") for r in runs.values()]
    if any(m is None for m in metas):
        raise RuntimeError("a corridor-on run has no corridor meta")
    return {"n_corridor_runs": len(metas), **{k: int(sum(m.get(k, 0) for m in metas)) for k in VIOLATION_KEYS}}


def _fb_key(tau: float, md, tau2) -> str:
    return json.dumps({"tau": tau, "max_declines": md, "tau_after_decline": tau2, "window_days": FB_WINDOW_DAYS},
                      sort_keys=True)


def _burden_key(tau: float, corridor_public: dict) -> str:
    return json.dumps({"tau": tau, "corridor": corridor_public}, sort_keys=True, default=str)


def _cell_corridor_params(c: dict, data: dict) -> dict | None:
    """The corridor parameters of a corridor-on cell (its own; the sweep's for records without them)."""
    return c.get("corridor_params") or data.get("corridor_params")


def _friction(df, model, model_dir, data, cells, corridor_of, out, model_sha, n_jobs, say, save, prov) -> None:
    from .metrics import confirmation_burden, false_block_rate, legit_test_scores
    fb, bu = data["friction"]["false_blocks"], data["friction"]["burden"]
    need_fb = {_fb_key(c["tau"], c["max_declines"], c["tau_after_decline"]): c for c in cells.values()}
    on = [c for c in cells.values() if c["corridor"] == "on"]
    if on and data.get("corridor_params") is None:
        raise ValueError("corridor-on cells need --corridor-threat for their confirmation burden")
    need_bu = {_burden_key(c["tau"], _cell_corridor_params(c, data)): c for c in on}
    for k in need_fb:
        if k in fb:
            check_reuse(fb[k].get("provenance"), prov, f"false-block entry {k}")
    for k in need_bu:
        if k in bu:
            check_reuse(bu[k].get("provenance"), prov, f"burden entry {k}")
    missing_fb = [k for k in need_fb if k not in fb]
    missing_bu = [k for k in need_bu if k not in bu]
    if not (missing_fb or missing_bu):
        return
    t0 = time.perf_counter()
    scores = load_scores(out / SCORES_FILE, model_sha)
    if scores is None:
        scores = legit_test_scores(df, model, n_jobs=n_jobs, model_dir=model_dir)
        save_scores(out / SCORES_FILE, scores, model_sha)
        say(f"{data['label']}: legit test scores {sum(len(v) for v in scores.values())} ops, "
            f"{time.perf_counter() - t0:.0f} s")
    for k in missing_fb:
        c = need_fb[k]
        t1 = time.perf_counter()
        r = false_block_rate(df, model, c["tau"], c["max_declines"], FB_WINDOW_DAYS,
                             tau_after_decline=c["tau_after_decline"], n_jobs=n_jobs, scores=scores)
        fb[k] = dict(dataclasses.asdict(r), wall_seconds=round(time.perf_counter() - t1, 1), provenance=prov)
        say(f"{data['label']}: false blocks tau {c['tau']:.6g} {c['reaction']}: "
            f"{100 * r.share_blocked_card_windows:.2f} % of card-months, {fb[k]['wall_seconds']} s")
        save()
    for k in missing_bu:
        c = need_bu[k]
        t1 = time.perf_counter()
        cor = corridor_of(c)
        r = confirmation_burden(df, scores, c["tau"], cor, n_jobs=n_jobs)
        bu[k] = dict(dataclasses.asdict(r), wall_seconds=round(time.perf_counter() - t1, 1), provenance=prov)
        say(f"{data['label']}: burden tau {c['tau']:.6g} corridor {corridor_label(c, data)}: "
            f"{100 * r.share_confirm_card_days:.2f} % of card-days, "
            f"{bu[k]['wall_seconds']} s")
        save()


def _attach_friction(data: dict) -> None:
    """Per cell ``friction``: the false-block and burden entries of its (τ, reaction) and (τ, corridor), or None."""
    fb, bu = data["friction"]["false_blocks"], data["friction"]["burden"]
    for c in data["cells"]:
        f = fb.get(_fb_key(c["tau"], c["max_declines"], c["tau_after_decline"]))
        if c["corridor"] == "on":
            cp = _cell_corridor_params(c, data)
            b = None if cp is None else bu.get(_burden_key(c["tau"], cp))
        else:
            b = {"share_confirm_ops": 0.0, "share_confirm_card_days": 0.0}
        c["friction"] = None if f is None or b is None else {
            "share_blocked_card_windows": f["share_blocked_card_windows"],
            "share_blocked_cards": f["share_blocked_cards"], "decline_rate": f["decline_rate"],
            "share_confirm_ops": b["share_confirm_ops"], "share_confirm_card_days": b["share_confirm_card_days"]}


def corridor_label(c: dict, data: dict) -> str:
    """``off`` or ``on κ … q …`` (the cell's effective corridor strength)."""
    if c["corridor"] != "on":
        return "off"
    p = _cell_corridor_params(c, data) or {}
    k = (p.get("daily") or {}).get("kappa")
    return "on" + ("" if k is None else f" κ {k:g}") + ("" if p.get("q") is None else f" q {p['q']:g}")


def _viol(c: dict) -> str:
    if c.get("n_corridor_runs") is None:
        return "n/a" if c["corridor"] == "on" else "–"
    if not c["n_corridor_runs"]:
        return "–"
    return " / ".join(str(c[k]) for k in VIOLATION_KEYS) + f" ({c['n_corridor_runs']} runs)"


def _pct(x, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f} %"


def sweep_markdown(data: dict) -> str:
    g = data["gate"]
    lines = [f"# Bank-policy sweep: {data['label']} ({data['role']})", "",
             f"Model `{data['model_dir']}` ({data['feature_set']}, {data['n_features']} features), own target FPR "
             f"{data['own_target_fpr']:g}, τ {data['own_tau']:.6g}; escalation τ' {data['tau_esc']:.6g} (val FPR "
             f"{data['esc_fpr']:g}). Ensemble: {', '.join(g['attacks'])} × seeds {g['attack_seeds']}; δ {g['delta']}"
             + (f", δ$ = δ · mean A = {data['delta_usd']:.0f} $ (mean A {data['mean_amount']:.2f} $)"
                if data.get("delta_usd") is not None else "") + ".", "",
             "U = worst one-sided upper bound of the paired uplift VER_adv − VER_clean over the runs; U$ = the same "
             "statistic on (VER_adv,i − VER_clean,i) · A_i. Friction on the legitimate test stream: false-blocked "
             "card-months (30-day windows) and card-days with an owner confirmation.", "",
             "| target FPR | τ | reaction | corridor | U (run) | U$ | mean uplift | test recall | test FPR | "
             "blocked card-months | blocked cards | confirm card-days | corridor violations bound / headroom / "
             "B(h) | time |",
             "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for c in data["cells"]:
        f = c.get("friction") or {}
        lines.append(f"| {c['target_fpr']:g} | {c['tau']:.4g} | {c['reaction']} | {corridor_label(c, data)} | "
                     f"{_pct(c['U'])} ({c['worst_run']}) | {c['U_usd']:.0f} $ | {_pct(c['mean_uplift'])} | "
                     f"{c['test_recall']:.4f} | {c['test_fpr']:.4f} | {_pct(f.get('share_blocked_card_windows'), 2)} | "
                     f"{_pct(f.get('share_blocked_cards'), 2)} | {_pct(f.get('share_confirm_card_days'), 2)} | "
                     f"{_viol(c)} | {c['wall_seconds']} s |")
    lines += ["", "## Runs (one-sided upper bound of the uplift per attack/seed)", "",
              "| target FPR | reaction | corridor | runs |", "|---:|---|---|---|"]
    for c in data["cells"]:
        lines.append(f"| {c['target_fpr']:g} | {c['reaction']} | {corridor_label(c, data)} | "
                     + ", ".join(f"{k} {_pct(r['uplift_upper'])}" for k, r in c["runs"].items()) + " |")
    m = data.get("meta", {})
    mach = m.get("machine", {})
    lines += ["", f"git `{m.get('git_sha')}` (uncommitted changes: {m.get('git_dirty')}); machine "
              f"{mach.get('host_machine') or mach.get('platform')} ({mach.get('cpu_count')} CPUs visible); seed "
              f"{data['seed']}.", ""]
    return "\n".join(lines)


def write_markdown(data: dict, out_dir) -> None:
    (Path(out_dir) / "policy-sweep.md").write_text(sweep_markdown(data), encoding="utf-8")
