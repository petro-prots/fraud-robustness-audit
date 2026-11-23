"""``fraudaudit policy-summary``: the Task 19 front, the pre-registered hypotheses and the R19 bias experiment from the
``policy-sweep.json`` of every model (rulings R18, R19, R30; rules fixed in the brief before any run).

* **Fronts** ``front-blocks.png`` / ``front-confirm.png``: U (left axis) and U$ (right axis, scaled so that δ and
  δ$ = δ · mean A lie on one line) against false-blocked card-months / confirmation card-days, one marker per model,
  the per-model Pareto set (both objectives minimised) filled.
* **T1 (dependence)**, per axis × model and pooled: every one-step pair softer → stricter on one axis (others fixed);
  axes ``tau`` (target FPR up = τ down), ``escalation`` (md3 → md3+esc), ``max_declines`` (md3 → md1), ``corridor``
  (off → on). ΔU = U_stricter − U_softer, Δfriction likewise with the axis' own friction (false-blocked card-months
  for the three blocking axes, confirmation card-days for the corridor axis; both medians are reported). Supported
  if median ΔU < −R and median Δfriction > 0; refuted if median ΔU ≥ 0; otherwise inconclusive.
* **T2 (reachability)**: a green cell has every run's upper bound ≤ δ. Per model the green cell with the lowest
  false-blocked card-month share (ties: lowest confirmation card-days) is confirmed with fresh seeds (a confirmation
  sweep, role ``confirmation``); T2 holds for the model if the confirmation is green too. ``confirmations.json`` /
  ``confirmations.tsv`` list the cells still to confirm (read by the chain script).
* **Policy vs model (R30)**: policy span per model = U(softest) − U(strictest); model span per cell = max − min of U
  over the models; seed spread = max − min of the run upper bounds over seeds within one attack and cell. Supported if
  the median policy span exceeds both the median model span and the median seed spread. An interaction is described,
  not tested (R32): a table of the median ΔU per axis on the base vs the hardened model of each family.
* **R19**: for every model with the cells ``mdinf``, ``md1``, ``md2``, ``md3`` at its own τ without the corridor, the
  relative overestimation (VER_∞ − VER_md) / VER_md of the ES runs (VER per target averaged over the ES seeds) and of
  the clean leg, with a paired percentile bootstrap over targets (also the difference VER_∞ − VER_md in pp).

Task 19b (ruling R32; rules fixed in its brief before any run), ``--confirm-rule front``:

* the sweeps of one label from several files (Task 19 and Task 19b) are merged (``merge_sweeps``; refused unless the
  gate, threat and corridor threat files, the model and its τ agree); cells of several commits only with
  ``merge_commits`` naming every commit (justified by the reproduction check of the brief);
* T1 also on the corridor-strength axes ``kappa`` (κ of the V1 daily split down) and ``q`` (corridor quantile down),
  one-step pairs at fixed τ, reaction and the other corridor parameter; friction = confirmation card-days;
* the front: per model the Pareto set (U on the gate seeds, false-blocked card-months; strict dominance, ties kept)
  of every policy cell (the R19 extras md2 / mdinf excluded), its confirmation on fresh seeds (per seed U_s = the
  worst attack's upper bound; median, worst; x_c = worst − U_c), the margin X = max x_c (also P90, per model);
* the risk-appetite curve δ′ = 0.08 … 0.50: (A) the cheapest confirmed front cell passing at δ′ (median_s U_s ≤ δ′
  and worst_s U_s ≤ δ′ + 0.04), (B) the cheapest policy cell with U ≤ δ′ − X; each with and without md1;
* corridor-bound violations summed over the cells that carry the counters (Task 19b grid and confirmations).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .policy_sweep import R19_REACTIONS, _pct, parse_reaction  # noqa: F401
from .sweep_cells import write_json

RESOLUTION = 0.036  # R: calibrated resolution of the gate on M3 (pp of A), the T1 threshold
UNRELIABLE = 0.08  # the shaded band of the fronts: U below it is within the resolution zone
GRID_REACTIONS = ("md1", "md3", "md3+esc")
SOFTEST = ("md3", "off")  # with the lowest target FPR
STRICTEST = ("md1", "on")  # with the highest target FPR
AXES = ("tau", "escalation", "max_declines", "corridor")
CORRIDOR_AXES = ("kappa", "q")  # Task 19b: corridor strength, stricter = smaller κ / lower q
AXIS_FRICTION = {"tau": "share_blocked_card_windows", "escalation": "share_blocked_card_windows",
                 "max_declines": "share_blocked_card_windows", "corridor": "share_confirm_card_days",
                 "kappa": "share_confirm_card_days", "q": "share_confirm_card_days"}
N_BOOT = 2000
CONFIRM_RULES = ("t2", "front")
DELTA_PRIMES = tuple(round(0.08 + i / 100, 2) for i in range(43))  # δ′ = 0.08 … 0.50, step 1 pp (R32)
WORST_SLACK = 0.04  # passing rule (variant B): worst_s U_s ≤ δ′ + 0.04
EPS = 1e-9  # float tolerance of the passing / margin comparisons (δ′ + 0.04 is not exact in binary)
MD1_NOTE = "impractical (md1: most cards blocked at least once)"


def grid_cells(sweep: dict) -> list[dict]:
    """The policy cells (Task 19's 3 × 3 × 2 and the Task 19b corridor variants; the R19 extra reactions excluded)."""
    return [c for c in sweep["cells"] if c["reaction"] in GRID_REACTIONS]


def base_cells(sweep: dict) -> list[dict]:
    """The policy cells with the file's corridor (no κ / q variant): the Task 19 grid, one cell per (τ, reaction,
    corridor)."""
    return [c for c in grid_cells(sweep) if not c.get("corridor_variant")]


def _index(cells) -> dict:
    return {(round(c["target_fpr"], 12), c["reaction"], c["corridor"]): c for c in cells}


def one_step_pairs(cells: list[dict], axis: str) -> list[tuple[dict, dict]]:
    """(softer, stricter) cell pairs that differ by one step on ``axis`` only."""
    idx = _index(cells)
    fprs = sorted({k[0] for k in idx})
    out = []
    for (f, r, c), cell in idx.items():
        if axis == "tau":
            i = fprs.index(f)
            nxt = None if i + 1 >= len(fprs) else idx.get((fprs[i + 1], r, c))
        elif axis == "escalation":
            nxt = idx.get((f, "md3+esc", c)) if r == "md3" else None
        elif axis == "max_declines":
            nxt = idx.get((f, "md1", c)) if r == "md3" else None
        elif axis == "corridor":
            nxt = idx.get((f, r, "on")) if c == "off" else None
        else:
            raise ValueError(f"unknown axis {axis!r}")
        if nxt is not None:
            out.append((cell, nxt))
    return out


def _fr(c: dict, key: str) -> float:
    f = c.get("friction")
    if f is None:
        raise ValueError(f"cell {c['key']} has no friction (run policy-sweep --friction-only first)")
    return float(f[key])


def t1_axis(pairs: list[tuple[dict, dict]], axis: str, resolution: float = RESOLUTION) -> dict:
    """Median ΔU, median Δfriction (axis friction and both kinds) and the verdict over ``pairs``."""
    if not pairs:
        return {"n_pairs": 0, "median_dU": None, "median_dfriction": None, "verdict": "no pairs"}
    du = [s["U"] - w["U"] for w, s in pairs]
    key = AXIS_FRICTION[axis]
    dfr = [_fr(s, key) - _fr(w, key) for w, s in pairs]
    dfb = [_fr(s, "share_blocked_card_windows") - _fr(w, "share_blocked_card_windows") for w, s in pairs]
    dcf = [_fr(s, "share_confirm_card_days") - _fr(w, "share_confirm_card_days") for w, s in pairs]
    mdu, mdf = float(np.median(du)), float(np.median(dfr))
    if mdu < -resolution and mdf > 0:
        verdict = "supported"
    elif mdu >= 0:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return {"n_pairs": len(pairs), "median_dU": mdu, "median_dfriction": mdf, "friction": key,
            "median_d_blocked_card_months": float(np.median(dfb)), "median_d_confirm_card_days": float(np.median(dcf)),
            "verdict": verdict}


def variant_pairs(cells: list[dict], axis: str) -> list[tuple[dict, dict]]:
    """(softer, stricter) corridor-on cell pairs one step apart on ``axis`` (``kappa``: the next smaller κ, ``q``: the
    next lower q, over the levels present among the cells) at the same τ, reaction and other corridor parameter; only
    pairs whose both cells exist."""
    if axis not in CORRIDOR_AXES:
        raise ValueError(f"unknown corridor axis {axis!r}")
    on = [c for c in cells if c["corridor"] == "on" and c.get("corridor_q") is not None]
    idx = {(round(c["target_fpr"], 12), c["reaction"], c.get("corridor_kappa"), c["corridor_q"]): c for c in on}
    if len(idx) != len(on):
        raise ValueError("two corridor-on cells share target FPR, reaction, kappa and q")
    pos = 2 if axis == "kappa" else 3
    levels = sorted({k[pos] for k in idx if k[pos] is not None})
    out = []
    for k, cell in sorted(idx.items(), key=lambda kv: kv[1]["key"]):
        if k[pos] is None or levels.index(k[pos]) == 0:
            continue
        nk = list(k)
        nk[pos] = levels[levels.index(k[pos]) - 1]
        nxt = idx.get(tuple(nk))
        if nxt is not None:
            out.append((cell, nxt))
    return out


def t1(sweeps: dict, resolution: float = RESOLUTION) -> dict:
    """``{axis: {label: t1_axis, "pooled": t1_axis}}``: the Task 19 axes over the cells with the file's corridor and,
    when any model has corridor variants, the corridor-strength axes over every corridor-on cell."""
    out = {}
    for axis in AXES:
        per = {lab: one_step_pairs(base_cells(s), axis) for lab, s in sweeps.items()}
        out[axis] = {lab: t1_axis(p, axis, resolution) for lab, p in per.items()}
        out[axis]["pooled"] = t1_axis([x for p in per.values() for x in p], axis, resolution)
    if any(c.get("corridor_variant") for s in sweeps.values() for c in s["cells"]):
        for axis in CORRIDOR_AXES:
            per = {lab: variant_pairs(grid_cells(s), axis) for lab, s in sweeps.items()}
            out[axis] = {lab: t1_axis(p, axis, resolution) for lab, p in per.items()}
            out[axis]["pooled"] = t1_axis([x for p in per.values() for x in p], axis, resolution)
    return out


def model_families(labels) -> dict:
    """``{family: (base label, hardened label)}`` from labels ``<family>-base`` / ``<family>-hard`` (both present)."""
    out = {}
    for lab in labels:
        if lab.endswith("-base") and f"{lab[:-5]}-hard" in labels:
            out[lab[:-5]] = (lab, f"{lab[:-5]}-hard")
    return dict(sorted(out.items()))


def corridor_relative_effect(sweep: dict) -> dict:
    """Median over the Task 19 corridor pairs (off → on, the file's corridor) of U_on / U_off; pairs with U_off ≤ 0
    have no ratio and are counted as skipped."""
    pairs = one_step_pairs(base_cells(sweep), "corridor")
    r = [s["U"] / w["U"] for w, s in pairs if w["U"] > 0]
    return {"median_ratio": float(np.median(r)) if r else None, "n_pairs": len(pairs),
            "n_skipped": len(pairs) - len(r)}


def interaction_table(t1_res: dict, sweeps: dict) -> dict:
    """Descriptive (R32, no verdict): per axis and model family the median ΔU on the base and on the hardened model,
    their difference (hardened − base) and both T1 verdicts; per model the corridor's relative effect."""
    fams = model_families(list(sweeps))
    rows = []
    for axis, by in t1_res.items():
        for fam, (b, h) in fams.items():
            rb, rh = by.get(b), by.get(h)
            if rb is None or rh is None:
                continue
            mb, mh = rb["median_dU"], rh["median_dU"]
            rows.append({"axis": axis, "family": fam, "base": b, "hardened": h, "median_dU_base": mb,
                         "median_dU_hardened": mh, "difference": None if mb is None or mh is None else mh - mb,
                         "verdict_base": rb["verdict"], "verdict_hardened": rh["verdict"]})
    return {"rows": rows, "corridor_relative_effect": {lab: corridor_relative_effect(s) for lab, s in sweeps.items()},
            "families": {f: list(v) for f, v in fams.items()}}


def _extreme(cells, fpr_pick, reaction, corridor):
    idx = _index(cells)
    fprs = sorted({k[0] for k in idx})
    return idx.get((fpr_pick(fprs), reaction, corridor)) if fprs else None


def _model_spans(sweeps: dict, keep=lambda c: True) -> dict:
    """``{cell: max − min U over the models}`` for the cells (kept by ``keep``) that at least two models have."""
    by_cell: dict = {}
    for lab, s in sweeps.items():
        for k, c in _index(c for c in base_cells(s) if keep(c)).items():
            by_cell.setdefault(k, {})[lab] = c["U"]
    return {f"fpr={k[0]:g}|{k[1]}|corridor={k[2]}": float(max(v.values()) - min(v.values()))
            for k, v in sorted(by_cell.items()) if len(v) >= 2}


def _seed_spreads(sweeps: dict, keep=lambda c: True) -> list[float]:
    """max − min of the run upper bounds over the seeds of one attack, per (model, cell kept by ``keep``, attack)."""
    out = []
    for s in sweeps.values():
        for c in base_cells(s):
            if not keep(c):
                continue
            by_attack: dict = {}
            for rk, r in c["runs"].items():
                by_attack.setdefault(rk.rsplit("/", 1)[0], []).append(r["uplift_upper"])
            out += [max(v) - min(v) for v in by_attack.values() if len(v) >= 2]
    return out


def _median(xs) -> float | None:
    xs = list(xs)
    return float(np.median(xs)) if xs else None


def policy_vs_model(sweeps: dict, resolution: float = RESOLUTION) -> dict:
    spans = {}
    for lab, s in sweeps.items():
        g = base_cells(s)
        soft, strict = _extreme(g, min, *SOFTEST), _extreme(g, max, *STRICTEST)
        spans[lab] = None if soft is None or strict is None else float(soft["U"] - strict["U"])
    model_spans = _model_spans(sweeps)
    seed_spreads = _seed_spreads(sweeps)
    med_p = _median(v for v in spans.values() if v is not None)
    med_m = _median(model_spans.values())
    med_s = _median(seed_spreads)
    if med_p is None or med_m is None or med_s is None:
        verdict = "not computable"
    else:
        verdict = "supported" if med_p > med_m and med_p > med_s else "not supported"
    return {"policy_span": spans, "median_policy_span": med_p, "model_span": model_spans,
            "median_model_span": med_m, "median_seed_spread": med_s, "n_seed_spreads": len(seed_spreads),
            "verdict": verdict}


PRACTICAL_EXCLUDED = ("md1",)  # R33: blocking after the first decline is impractical (row 5.4)


def policy_vs_model_practical(sweeps: dict, exclude_reactions=PRACTICAL_EXCLUDED) -> dict:
    """Descriptive companion of ``policy_vs_model`` (R47 (2), definition fixed before the computation): the practical
    policies are the grid cells whose reaction is not in ``exclude_reactions``; the policy span of a model is max − min
    U over them; the model span and the seed spread are the R30 ones over the same cells. No verdict: only whether the
    medians compare the same way as the R30 rule (``None`` when a median is missing)."""
    excl = set(exclude_reactions)
    keep = lambda c: c["reaction"] not in excl  # noqa: E731
    spans, n_cells = {}, {}
    for lab, s in sweeps.items():
        us = [c["U"] for c in base_cells(s) if keep(c)]
        n_cells[lab] = len(us)
        spans[lab] = float(max(us) - min(us)) if us else None
    model_spans = _model_spans(sweeps, keep)
    seed_spreads = _seed_spreads(sweeps, keep)
    med_p = _median(v for v in spans.values() if v is not None)
    med_m = _median(model_spans.values())
    med_s = _median(seed_spreads)
    return {"excluded_reactions": sorted(excl), "n_cells": max(n_cells.values(), default=0),
            "n_cells_per_model": n_cells, "policy_span": spans, "median_policy_span": med_p,
            "model_span": model_spans, "median_model_span": med_m, "median_seed_spread": med_s,
            "n_seed_spreads": len(seed_spreads),
            "policy_gt_model": None if med_p is None or med_m is None else med_p > med_m,
            "policy_gt_seed": None if med_p is None or med_s is None else med_p > med_s}


def green(c: dict, delta: float) -> bool:
    return all(r["uplift_upper"] <= delta for r in c["runs"].values())


def t2(sweeps: dict, confirmations: dict, delta: float) -> dict:
    """Per model: green cells, the cell picked for confirmation and the confirmation verdict."""
    out = {}
    for lab, s in sweeps.items():
        gs = [c for c in base_cells(s) if green(c, delta)]
        pick = None
        if gs:
            pick = min(gs, key=lambda c: (_fr(c, "share_blocked_card_windows"), _fr(c, "share_confirm_card_days"),
                                          c["key"]))
        conf = None
        if pick is not None and lab in confirmations:
            cc = next((c for c in confirmations[lab]["cells"] if c["key"] == pick["key"]), None)
            if cc is not None:
                check_confirmation(s, pick, confirmations[lab], cc)
                conf = {"key": cc["key"], "U": cc["U"], "green": green(cc, delta),
                        "attack_seeds": confirmations[lab]["gate"]["attack_seeds"]}
        rmin = s["gate"].get("recall_clean_min")
        out[lab] = {"green_cells": [{"key": c["key"], "U": c["U"], "test_recall": c["test_recall"],
                                     "recall_ok": None if rmin is None else c["test_recall"] >= rmin} for c in gs],
                    "picked": None if pick is None else pick["key"], "confirmation": conf,
                    "t2": "no green cell" if pick is None else ("pending confirmation" if conf is None else
                                                              ("confirmed" if conf["green"] else "not confirmed"))}
    return out


def check_confirmation(grid: dict, cell: dict, conf: dict, conf_cell: dict) -> None:
    """ValueError unless the confirmation sweep is a valid fresh-seed rerun of the picked cell: same model (model.json
    SHA-256), same base seed, same gate apart from the attack seeds (attacks, δ, confidence, …), attack seeds disjoint
    from the grid's, same threat and corridor-threat files (SHA-256), same τ, reaction and corridor (κ, q and every
    other corridor parameter)."""
    g, c = grid["gate"], conf["gate"]
    errs = []
    if conf.get("model_json_sha256") != grid.get("model_json_sha256"):
        errs.append("another model (model.json SHA-256)")
    if list(c["attacks"]) != list(g["attacks"]):
        errs.append(f"attacks {c['attacks']} != {g['attacks']}")
    if set(c["attack_seeds"]) & set(g["attack_seeds"]):
        errs.append(f"attack seeds {c['attack_seeds']} overlap the grid's {g['attack_seeds']}")
    gd = sorted(k for k in set(g) | set(c) if k not in ("attack_seeds", "attacks") and g.get(k) != c.get(k))
    if gd:
        errs.append(f"gate fields {gd} differ")
    if conf.get("seed") != grid.get("seed"):
        errs.append(f"seed {conf.get('seed')} != {grid.get('seed')}")
    for k in MERGE_META[1:]:
        if (conf.get("meta") or {}).get(k) != (grid.get("meta") or {}).get(k):
            errs.append(f"meta.{k} differs")
    for k in ("tau", "max_declines", "tau_after_decline", "corridor", "reaction", "corridor_kappa", "corridor_q",
              "corridor_params"):
        if conf_cell.get(k) != cell.get(k):
            errs.append(f"{k} {conf_cell.get(k)!r} != {cell.get(k)!r}")
    if errs:
        raise ValueError(f"confirmation of {grid['label']} {cell['key']} is not valid: " + "; ".join(errs))


def confirmation_plan(sweeps: dict, t2_res: dict) -> list[dict]:
    plan = []
    for lab, r in t2_res.items():
        if r["picked"] is None or r["confirmation"] is not None:
            continue
        c = next(c for c in sweeps[lab]["cells"] if c["key"] == r["picked"])
        plan.append({"label": lab, "model_dir": sweeps[lab]["model_dir"], "target_fpr": c["target_fpr"],
                     "reaction": c["reaction"], "corridor": c["corridor"], "key": c["key"]})
    return plan


# --- R19 --------------------------------------------------------------------------------------------------------


def _ver_matrix(cell: dict, attack: str | None) -> np.ndarray:
    """Per target VER averaged over the runs of ``attack`` (``None`` = the clean leg, identical in every run)."""
    if attack is None:
        return np.asarray(next(iter(cell["runs"].values()))["ver_clean"], dtype=np.float64)
    rows = [r["ver_adv"] for k, r in cell["runs"].items() if k.rsplit("/", 1)[0] == attack]
    if not rows:
        raise ValueError(f"cell {cell['key']} has no {attack} runs")
    return np.mean(np.asarray(rows, dtype=np.float64), axis=0)


def rel_overestimation(v_inf: np.ndarray, v_md: np.ndarray, confidence: float = 0.95, n_boot: int = N_BOOT,
                       seed: int = 42) -> dict:
    """(mean VER_∞ − mean VER_md) / mean VER_md with a paired percentile bootstrap over targets (two-sided)."""
    a, b = np.asarray(v_inf, dtype=np.float64), np.asarray(v_md, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired arrays of equal non-zero length expected")
    est = float((a.mean() - b.mean()) / b.mean()) if b.mean() > 0 else float("nan")
    idx = np.random.default_rng(seed).integers(0, a.size, size=(n_boot, a.size))
    ma, mb = a[idx].mean(axis=1), b[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (ma - mb) / mb
    n_dropped = int((~np.isfinite(rs)).sum())  # resamples with mean VER_md = 0: the ratio is undefined
    rs = rs[np.isfinite(rs)]
    alpha = 1 - confidence
    lo, hi = (np.quantile(rs, [alpha / 2, 1 - alpha / 2]) if rs.size else (float("nan"), float("nan")))
    dlo, dhi = np.quantile(ma - mb, [alpha / 2, 1 - alpha / 2])  # the absolute difference, same resamples
    return {"ver_inf": float(a.mean()), "ver_md": float(b.mean()), "rel": est, "lower": float(lo), "upper": float(hi),
            "n_boot": int(n_boot), "n_dropped": n_dropped, "diff": float(a.mean() - b.mean()),
            "diff_lower": float(dlo), "diff_upper": float(dhi)}


def r19(sweeps: dict, attack: str = "es", seed: int = 42) -> dict:
    out = {}
    for lab, s in sweeps.items():
        own = round(float(s["own_target_fpr"]), 12)
        idx = _index([c for c in s["cells"] if not c.get("corridor_variant")])
        cells = {r: idx.get((own, r, "off")) for r in ("mdinf", "md1", "md2", "md3")}
        if any(v is None for v in cells.values()):
            continue
        keys = {json.dumps(c.get("targets")) for c in cells.values()}
        if len(keys) != 1:
            raise ValueError(f"{lab}: R19 cells have different targets")
        res = {}
        for leg, att in (("adv_" + attack, attack), ("clean", None)):
            v_inf = _ver_matrix(cells["mdinf"], att)
            res[leg] = {md: rel_overestimation(v_inf, _ver_matrix(cells[md], att), seed=seed)
                        for md in ("md1", "md2", "md3")}
        out[lab] = res
    return out


R19_TASK24_HYPOTHESIS = (
    "Fixed before XGBoost / GRU are run (ruling R32): «the relative overestimation of the loss without the bank's "
    "reaction is larger for a more robust model (lower VER_md)» – Spearman rank correlation between VER_md and the "
    "relative overestimation (VER_∞ − VER_md) / VER_md over the 8 models of Task 24, p < 0.05 (negative "
    "correlation); separately the same test for the absolute overestimation VER_∞ − VER_md in pp. This table (4 "
    "models) gives no correlation verdict.")


def r19_markdown(res: dict, attack: str = "es") -> str:
    lines = ["# R19: loss-estimate bias without the bank's reaction", "",
             "Own τ, no corridor. VER per target averaged over the ES runs (seeds of the gate) for the adversarial "
             "leg; the clean leg replays the real operations. Relative overestimation = (VER_∞ − VER_md) / VER_md, "
             "VER_∞ = never blocked; 95 % paired percentile bootstrap over targets.", "",
             "| model | leg | max_declines | VER_∞ | VER_md | overestimation [95 % CI] | difference VER_∞ − VER_md, "
             "pp [95 % CI] | resamples dropped (VER_md = 0) |",
             "|---|---|---:|---:|---:|---|---|---:|"]
    for lab, legs in res.items():
        for leg, by in legs.items():
            for md, r in by.items():
                diff = ("–" if r.get("diff") is None else
                        f"{100 * r['diff']:.2f} [{100 * r['diff_lower']:.2f}; {100 * r['diff_upper']:.2f}]")
                lines.append(f"| {lab} | {leg.replace('adv_', 'adversarial ')} | {md[2:]} | {_pct(r['ver_inf'])} | "
                             f"{_pct(r['ver_md'])} | {_pct(r['rel'])} [{_pct(r['lower'])}; {_pct(r['upper'])}] | "
                             f"{diff} | {r['n_dropped']} of {r['n_boot']} |")
    lines += ["", "## Pre-registered for Task 24 (recorded here, not tested)", "", R19_TASK24_HYPOTHESIS]
    return "\n".join(lines) + "\n"


def r19_plot(res: dict, path: Path, attack: str = "es") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(res)
    mds = ("md1", "md2", "md3")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    palette = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a4fd1", "#c23b6b"]
    w = 0.8 / max(1, len(labs))
    for ax, leg, title in ((axes[0], "adv_" + attack, f"adversarial ({attack.upper()} runs)"),
                           (axes[1], "clean", "clean leg (real operations)")):
        for i, lab in enumerate(labs):
            r = [res[lab][leg][m] for m in mds]
            x = np.arange(len(mds)) + (i - (len(labs) - 1) / 2) * w
            y = [100 * v["rel"] for v in r]
            err = [[100 * (v["rel"] - v["lower"]) for v in r], [100 * (v["upper"] - v["rel"]) for v in r]]
            ax.bar(x, y, width=w, color=palette[i % 5], label=lab, yerr=err, capsize=3)
        ax.set_xticks(np.arange(len(mds)), [f"max_declines = {m[2:]}" for m in mds])
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("overestimation of VER without the bank's reaction (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Task 19b: front, confirmation on fresh seeds, margin X, risk-appetite curve ---------------------------------


def cell_corridor_text(c: dict) -> str:
    if c["corridor"] != "on":
        return "off"
    k = c.get("corridor_kappa")
    return "on" + ("" if k is None else f" κ {k:g}") + ("" if c.get("corridor_q") is None else f" q {c['corridor_q']:g}")


def policy_name(c: dict) -> str:
    return f"FPR {c['target_fpr']:g}, {c['reaction']}, corridor {cell_corridor_text(c)}"


def cell_summary(c: dict) -> dict:
    """The numbers of one policy cell reported on the front and the risk-appetite curve."""
    return {"key": c["key"], "policy": policy_name(c), "target_fpr": c["target_fpr"], "tau": c["tau"],
            "reaction": c["reaction"], "corridor": c["corridor"], "corridor_kappa": c.get("corridor_kappa"),
            "corridor_q": c.get("corridor_q"), "U": c["U"], "U_usd": c.get("U_usd"),
            "blocked_card_months": _fr(c, "share_blocked_card_windows"),
            "blocked_cards": _fr(c, "share_blocked_cards"),
            "confirm_card_days": _fr(c, "share_confirm_card_days"), "md1": c["reaction"] == "md1"}


def front_cells(sweep: dict) -> list[dict]:
    """The Pareto set of the policy cells (R19 extras excluded): U on the gate seeds and false-blocked card-months,
    both minimised, strict dominance (exact ties stay)."""
    g = grid_cells(sweep)
    missing = [c["key"] for c in g if not c.get("friction")]
    if missing:
        raise ValueError(f"{sweep['label']}: cells without friction {missing} (run policy-sweep --friction-only)")
    flags = pareto_flags([(c["U"], _fr(c, "share_blocked_card_windows")) for c in g])
    return [c for c, f in zip(g, flags) if f]


CONFIRM_SEEDS = (4, 5, 6, 7, 8, 9)  # the pre-registered fresh seeds of the front confirmation (gate-seeds-4-9.yaml)


def confirmation_stats(conf_cell: dict, u_grid: float, attacks=None, expected_seeds=None) -> dict:
    """Per seed s: U_s = max over the attacks of the run upper bounds; median and worst over the seeds; x_c = worst −
    the cell's U on the gate seeds; all run upper bounds. ValueError unless every seed has every attack and (with
    ``expected_seeds``) the seeds are exactly those and the run count is attacks × seeds."""
    by_seed: dict = {}
    for rk, r in conf_cell["runs"].items():
        a, sd = rk.rsplit("/", 1)
        by_seed.setdefault(int(sd), {})[a] = float(r["uplift_upper"])
    if not by_seed:
        raise ValueError(f"confirmation of {conf_cell['key']} has no runs")
    if attacks is not None:
        bad = {sd: sorted(v) for sd, v in by_seed.items() if sorted(v) != sorted(attacks)}
        if bad:
            raise ValueError(f"confirmation of {conf_cell['key']}: seeds without every attack {sorted(attacks)}: {bad}")
    if expected_seeds is not None:
        want = sorted(int(x) for x in expected_seeds)
        if sorted(by_seed) != want:
            raise ValueError(f"confirmation of {conf_cell['key']}: seeds {sorted(by_seed)}, the rule needs {want}")
        if attacks is not None and len(conf_cell["runs"]) != len(attacks) * len(want):
            raise ValueError(f"confirmation of {conf_cell['key']}: {len(conf_cell['runs'])} runs, the rule needs "
                             f"{len(attacks) * len(want)}")
    u_s = {sd: max(v.values()) for sd, v in sorted(by_seed.items())}
    vals = list(u_s.values())
    worst = float(max(vals))
    return {"run_upper": {k: float(r["uplift_upper"]) for k, r in sorted(conf_cell["runs"].items())},
            "U_s": {str(sd): u for sd, u in u_s.items()}, "median": float(np.median(vals)), "worst": worst,
            "x_c": worst - float(u_grid), "n_runs": len(conf_cell["runs"]), "seeds": sorted(u_s)}


def passes(median: float, worst: float, delta_p: float) -> bool:
    """Passing rule at appetite δ′ (variant B, R32): median_s U_s ≤ δ′ and worst_s U_s ≤ δ′ + 0.04."""
    return median <= delta_p + EPS and worst <= delta_p + WORST_SLACK + EPS


def margin(xs_by_label: dict) -> dict:
    """X = max x_c over the confirmed front cells of every model; the 90th percentile (numpy linear interpolation)
    and the per-model max for information."""
    allx = [x for v in xs_by_label.values() for x in v]
    return {"X": float(max(allx)) if allx else None, "p90": float(np.percentile(allx, 90)) if allx else None,
            "per_model": {lab: (float(max(v)) if v else None) for lab, v in xs_by_label.items()},
            "n_confirmed": len(allx)}


def front_analysis(grid: dict, conf: dict, expected_seeds=CONFIRM_SEEDS) -> dict:
    """Per model the front cells with their confirmation (or None), the margin X and the plan of the front cells
    still to confirm (``confirmations-front.tsv``). A confirmation must use exactly ``expected_seeds`` (R32: 4–9)."""
    front, plan = {}, []
    for lab, s in grid.items():
        cs = conf.get(lab)
        rows = []
        for c in front_cells(s):
            row = cell_summary(c)
            cc = None if cs is None else next((x for x in cs["cells"] if x["key"] == c["key"]), None)
            row["confirmation"] = None
            if cc is not None:
                check_confirmation(s, c, cs, cc)
                if sorted(cs["gate"]["attack_seeds"]) != sorted(expected_seeds):
                    raise ValueError(f"confirmation of {lab} {c['key']}: attack seeds {cs['gate']['attack_seeds']}, "
                                     f"the front rule needs {list(expected_seeds)} (gate-seeds-4-9.yaml)")
                row["confirmation"] = dict(confirmation_stats(cc, c["U"], cs["gate"]["attacks"], expected_seeds),
                                           attack_seeds=list(cs["gate"]["attack_seeds"]))
            else:
                plan.append({"label": lab, "model_dir": s["model_dir"], "target_fpr": c["target_fpr"],
                             "reaction": c["reaction"], "corridor": c["corridor"],
                             "kappa": c.get("corridor_kappa") if c["corridor"] == "on" else None,
                             "q": c.get("corridor_q") if c["corridor"] == "on" else None, "key": c["key"]})
            rows.append(row)
        front[lab] = sorted(rows, key=lambda r: (r["blocked_card_months"], r["U"], r["key"]))
    xs = {lab: [r["confirmation"]["x_c"] for r in rows if r["confirmation"]] for lab, rows in front.items()}
    m = margin(xs)
    m.update(n_front=sum(len(r) for r in front.values()), n_pending=len(plan), provisional=bool(plan))
    return {"front": front, "margin": m, "plan": plan}


def _cheapest(rows: list[dict]) -> dict | None:
    """The lowest false-blocked card-months (ties: lowest confirmation card-days, then the key)."""
    return min(rows, key=lambda r: (r["blocked_card_months"], r["confirm_card_days"], r["key"])) if rows else None


def _brief(r: dict | None) -> dict | None:
    if r is None:
        return None
    out = {k: r[k] for k in ("key", "policy", "reaction", "U", "U_usd", "blocked_card_months", "blocked_cards",
                             "confirm_card_days", "md1")}
    if r.get("confirmation"):
        out.update(median=r["confirmation"]["median"], worst=r["confirmation"]["worst"])
    return out


def appetite(front: dict, grid: dict, x_margin: float | None, deltas=DELTA_PRIMES) -> dict:
    """Per model and δ′: (A) the cheapest confirmed front cell passing at δ′; (B) the cheapest policy cell of the model
    with U (gate seeds) ≤ δ′ − X (None without X); each with and without md1 cells."""
    out = {}
    for lab, rows in front.items():
        cells = [cell_summary(c) for c in grid_cells(grid[lab])]
        pts = []
        for d in deltas:
            e = {"delta": float(d)}
            for tag, keep in (("with_md1", lambda r: True), ("without_md1", lambda r: not r["md1"])):
                e[f"A_{tag}"] = _brief(_cheapest([r for r in rows if keep(r) and r["confirmation"] and passes(
                    r["confirmation"]["median"], r["confirmation"]["worst"], d)]))
                e[f"B_{tag}"] = None if x_margin is None else _brief(_cheapest(
                    [r for r in cells if keep(r) and r["U"] <= d - x_margin + EPS]))
            pts.append(e)
        out[lab] = pts
    return out


def violation_totals(sweeps) -> dict:
    """Corridor-bound violations summed over the corridor-on cells that carry the counters (Task 19b grid and
    confirmations); corridor-on cells without them (Task 19 records) are counted, not recomputed."""
    from .policy_sweep import VIOLATION_KEYS
    out = {"n_cells": 0, "n_corridor_runs": 0, **{k: 0 for k in VIOLATION_KEYS}, "n_on_cells_without_counters": 0}
    for s in sweeps:
        for c in s["cells"]:
            if c["corridor"] != "on":
                continue
            if c.get("n_corridor_runs") is None:
                out["n_on_cells_without_counters"] += 1
                continue
            out["n_cells"] += 1
            out["n_corridor_runs"] += int(c["n_corridor_runs"])
            for k in VIOLATION_KEYS:
                out[k] += int(c[k] or 0)
    out["any"] = any(out[k] > 0 for k in VIOLATION_KEYS)
    return out


# --- fronts -----------------------------------------------------------------------------------------------------


ROW_FRICTION = {"share_blocked_card_windows": "blocked_card_months", "share_confirm_card_days": "confirm_card_days"}


def pareto_flags(points: list[tuple]) -> list[bool]:
    from .corridor_sweep import pareto
    return pareto(points)


def front_plot(sweeps: dict, path: Path, friction_key: str, xlabel: str, delta: float,
               confirmed: dict | None = None) -> None:
    """U / U$ against the friction ``friction_key``; ``confirmed`` (``front_analysis()["front"]``) adds for every
    confirmed front cell its median U_s on the fresh seeds (bar marker) with a whisker to the worst U_s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a4fd1", "#c23b6b"]
    markers = ["o", "s", "^", "D", "v"]
    mean_a = float(np.mean([s["mean_amount"] for s in sweeps.values() if s.get("mean_amount")]))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax2 = ax.twinx()
    ymax, ymin = 0.0, 0.0
    for i, (lab, s) in enumerate(sweeps.items()):
        g = [c for c in grid_cells(s) if c.get("friction")]
        if not g:
            continue
        x = [100 * c["friction"][friction_key] for c in g]
        y = [100 * c["U"] for c in g]
        fl = pareto_flags([(c["friction"][friction_key], c["U"]) for c in g])
        col, mk = palette[i % 5], markers[i % 5]
        ax.scatter([a for a, f in zip(x, fl) if not f], [b for b, f in zip(y, fl) if not f], s=34,
                   facecolors="none", edgecolors=col, marker=mk, lw=1.2)
        ax.scatter([a for a, f in zip(x, fl) if f], [b for b, f in zip(y, fl) if f], s=46, color=col, marker=mk,
                   label=f"{lab} (filled = Pareto set)")
        front = sorted((a, b) for a, b, f in zip(x, y, fl) if f)
        if len(front) > 1:
            ax.plot(*zip(*front), color=col, lw=1.0, alpha=0.7)
        ax2.scatter(x, [c["U_usd"] for c in g], s=10, color=col, marker="x", alpha=0.6)
        ys = y + [100 * c["U_usd"] / mean_a for c in g]
        for r in (confirmed or {}).get(lab, []):
            cf = r.get("confirmation")
            if not cf:
                continue
            xr = 100 * r[ROW_FRICTION[friction_key]]
            ax.plot([xr, xr], [100 * cf["median"], 100 * cf["worst"]], color=col, lw=1.0, alpha=0.8)
            ax.scatter([xr], [100 * cf["median"]], s=70, color=col, marker="_", lw=2.0)
            ys += [100 * cf["worst"]]
        ymax, ymin = max(ymax, max(ys)), min(ymin, min(ys))
    top = max(ymax * 1.08, 100 * delta * 1.5)
    bottom = 1.08 * ymin  # U can be negative (the attack below the clean leg)
    ax.set_ylim(bottom, top)
    ax2.set_ylim(bottom / 100 * mean_a, top / 100 * mean_a)
    ax.axhspan(bottom, 100 * UNRELIABLE, color="#bbbbbb", alpha=0.25, lw=0)
    ax.text(ax.get_xlim()[1], 100 * UNRELIABLE / 2, "statistically unreliable zone (resolution) ", ha="right",
            va="center", fontsize=8, color="#555555")
    ax.axhline(100 * delta, color="#c0392b", lw=1.2, ls="--")
    ax.text(ax.get_xlim()[1], 100 * delta, f"bank risk appetite δ′ = {delta:g} (δ\\$ = {delta * mean_a:.0f} \\$) – "
            "cheapest passing policy lies on this line ", ha="right", va="bottom", fontsize=8, color="#c0392b")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("U: worst one-sided upper bound of the uplift (% of A)")
    ax2.set_ylabel("U\\$ (×): the same statistic on dollars per episode (\\$)")
    ax.grid(color="#e5e5e5", lw=0.8)
    ax.spines[["top"]].set_visible(False)
    if any(r.get("confirmation") for rows in (confirmed or {}).values() for r in rows):
        ax.plot([], [], color="#555555", marker="_", ms=10, lw=1.0,
                label="confirmed front cell: median U_s on fresh seeds (bar), whisker to the worst")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- summary ----------------------------------------------------------------------------------------------------

MERGE_FIELDS = ("model_json_sha256", "own_target_fpr", "own_tau", "esc_fpr", "tau_esc", "seed", "gate", "role")
MERGE_META = ("gate_sha256", "threat_sha256", "corridor_threat_sha256")


def normalize(sweep: dict, source: str) -> dict:
    """``sweep`` with every cell carrying ``corridor_kappa`` / ``corridor_q`` / ``corridor_params`` (records without
    them – Task 19 – take the sweep's corridor for corridor-on cells), ``corridor_variant`` (κ, q other than the
    corridor threat file's) and ``source``; ``sources`` keeps every input file with its own cells and friction (for
    the provenance check after a merge)."""
    s = dict(sweep)
    cp = s.get("corridor_params")
    file_kq = None if cp is None else ((cp.get("daily") or {}).get("kappa"), float(cp["q"]))
    cells = []
    for c in s["cells"]:
        c = dict(c)
        if c["corridor"] == "on":
            pp = c.get("corridor_params") or cp
            if pp is None:
                raise ValueError(f"{source}: corridor-on cell {c['key']} without corridor parameters")
            c["corridor_params"] = pp
            c["corridor_kappa"] = (pp.get("daily") or {}).get("kappa")
            c["corridor_q"] = float(pp["q"])
            c["corridor_variant"] = (c["corridor_kappa"], c["corridor_q"]) != file_kq
        else:
            c["corridor_kappa"], c["corridor_q"], c["corridor_variant"] = None, None, False
        c["source"] = source
        cells.append(c)
    s["cells"] = cells
    s["sources"] = [{"file": source, "cells": cells, "friction": s.get("friction") or {},
                     "meta": s.get("meta") or {}}]
    return s


def merge_sweeps(sweeps: list[dict]) -> dict:
    """One sweep per label from the normalised sweeps of several files (Task 19 + Task 19b). ValueError unless the
    model (model.json SHA-256), its τ (own, escalation, and every τ of a common target FPR), the gate (ensemble and
    file SHA-256), the threat and corridor threat files (SHA-256; the corridor variants are applied in code), the seed
    and the role agree, or when one cell key is in more than one file."""
    first = sweeps[0]
    if len(sweeps) == 1:
        return first
    lab = first["label"]
    files = [x["file"] for s in sweeps for x in s["sources"]]
    for s in sweeps[1:]:
        errs = [k for k in MERGE_FIELDS if s.get(k) != first.get(k)]
        errs += [f"meta.{k}" for k in MERGE_META if (s.get("meta") or {}).get(k) != (first.get("meta") or {}).get(k)]
        if s.get("corridor_params") is not None and first.get("corridor_params") is not None \
                and s["corridor_params"] != first["corridor_params"]:
            errs.append("corridor_params (the corridor threat file's corridor)")
        common = set(s.get("taus", {})) & set(first.get("taus", {}))
        errs += [f"taus[{f}]" for f in sorted(common) if s["taus"][f] != first["taus"][f]]
        if s.get("mean_amount") is not None and first.get("mean_amount") is not None \
                and s["mean_amount"] != first["mean_amount"]:
            errs.append("mean_amount (the targets)")
        if errs:
            raise ValueError(f"refusing to merge the sweeps of {lab!r} ({', '.join(files)}): they differ in "
                             + ", ".join(errs))
    for kind in ("false_blocks", "burden"):  # I4: one friction entry, one value (a free check of the recomputation)
        seen: dict = {}
        for src in (x for s in sweeps for x in s["sources"]):
            for k, e in ((src.get("friction") or {}).get(kind) or {}).items():
                v = {f: x for f, x in e.items() if f not in ("wall_seconds", "provenance")}
                if k in seen and seen[k][1] != v:
                    raise ValueError(f"refusing to merge the sweeps of {lab!r}: the {kind} friction entry {k} differs "
                                     f"between {seen[k][0]} and {src['file']}")
                seen.setdefault(k, (src["file"], v))
    keys = [c["key"] for s in sweeps for c in s["cells"]]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"refusing to merge the sweeps of {lab!r}: cells {dup} are in more than one file")
    merged = dict(first)
    merged["cells"] = [c for s in sweeps for c in s["cells"]]
    merged["taus"] = {k: v for s in sweeps for k, v in s.get("taus", {}).items()}
    merged["corridor_params"] = next((s["corridor_params"] for s in sweeps if s.get("corridor_params")), None)
    merged["mean_amount"] = next((s["mean_amount"] for s in sweeps if s.get("mean_amount") is not None), None)
    merged["delta_usd"] = next((s["delta_usd"] for s in sweeps if s.get("delta_usd") is not None), None)
    merged["sources"] = [x for s in sweeps for x in s["sources"]]
    return merged


def load(paths) -> tuple[dict, dict]:
    """(grid sweeps, confirmation sweeps) by label from ``policy-sweep.json`` files; several files of one label and
    role are merged (``merge_sweeps``)."""
    grid, conf = {}, {}
    for p in paths:
        s = normalize(json.loads(Path(p).read_text(encoding="utf-8")), str(p))
        (conf if s.get("role") == "confirmation" else grid).setdefault(s["label"], []).append(s)
    return ({lab: merge_sweeps(v) for lab, v in grid.items()}, {lab: merge_sweeps(v) for lab, v in conf.items()})


def provenances(sweeps) -> list[dict]:
    """Distinct (git SHA, dirty, host machine) of every cell and friction entry of ``sweeps`` (of every input file
    of a merged sweep)."""
    from .policy_sweep import PROVENANCE_KEYS
    seen = []
    for s in sweeps:
        for src in s.get("sources") or [s]:
            fr = src.get("friction") or {}
            entries = (list(src["cells"]) + list(fr.get("false_blocks", {}).values())
                       + list(fr.get("burden", {}).values()))
            for e in entries:
                p = {k: (e.get("provenance") or {}).get(k) for k in PROVENANCE_KEYS}
                if p not in seen:
                    seen.append(p)
    return seen


def check_provenance(prov: list[dict], merge_commits=None) -> None:
    """ValueError unless every entry is from a clean tree on one machine and from one commit – or from several commits
    that ``merge_commits`` names explicitly (full SHA or a prefix of ≥ 7 characters; Task 19b merges the Task 19 cells
    after the reproduction check of its brief)."""
    machines = {p["host_machine"] for p in prov}
    shas = {p["git_sha"] for p in prov}
    dirty = any(p["git_dirty"] is True for p in prov)
    allowed = [m for m in (merge_commits or []) if m and len(m) >= 7]

    def named(sha):
        return sha is not None and any(sha.startswith(m) for m in allowed)
    if dirty or len(machines) != 1 or (len(shas) != 1 and not all(named(x) for x in shas)):
        raise ValueError(f"the sweeps were computed with more than one code version / machine or from a dirty tree: "
                         f"{prov}; rerun the affected cells with one clean commit on one machine (several commits "
                         "only when every one is named in --merge-commits)")


def summarize(paths, out_dir, resolution: float = RESOLUTION, confirm_rule: str = "t2",
              merge_commits=None, check_only: bool = False) -> dict:
    """The summary of the policy sweeps. ``confirm_rule`` ``t2`` (Task 19: the T2 green-cell confirmation plan
    ``confirmations.tsv``) or ``front`` (Task 19b: the front confirmed on fresh seeds, margin X, risk-appetite curve,
    plan ``confirmations-front.tsv``). ``check_only``: only load, merge and check provenance and δ (a cheap preflight
    of the chain); nothing is written."""
    if confirm_rule not in CONFIRM_RULES:
        raise ValueError(f"confirm_rule must be one of {CONFIRM_RULES}, got {confirm_rule!r}")
    grid, conf = load(paths)
    if not grid:
        raise ValueError("no grid sweeps given")
    prov = provenances([*grid.values(), *conf.values()])
    check_provenance(prov, merge_commits)
    deltas = {s["gate"]["delta"] for s in grid.values()}
    if len(deltas) != 1 or None in deltas:
        raise ValueError(f"the sweeps disagree on δ or δ is null: {sorted(map(str, deltas))}")
    delta = deltas.pop()
    if check_only:
        return {"check": "ok", "labels": list(grid), "provenances": prov,
                "n_cells": {lab: len(s["cells"]) for lab, s in grid.items()}}
    t1_res = t1(grid, resolution)
    res = {"delta": delta, "resolution": resolution, "confirm_rule": confirm_rule, "t1": t1_res,
           "policy_vs_model": policy_vs_model(grid, resolution), "interaction": interaction_table(t1_res, grid),
           "r19": r19(grid), "labels": list(grid), "provenance": prov[0] if len(prov) == 1 else None,
           "provenances": prov,
           "sources": {lab: [{"file": x["file"], "git_sha": (x["meta"] or {}).get("git_sha"),
                              "n_cells": len(x["cells"])} for x in s["sources"]]
                       for lab, s in [*grid.items(), *((f"{k} (confirmation)", v) for k, v in conf.items())]},
           "mean_amount": {lab: s.get("mean_amount") for lab, s in grid.items()},
           "delta_usd": {lab: s.get("delta_usd") for lab, s in grid.items()},
           "violations": violation_totals([*grid.values(), *conf.values()])}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    confirmed = None
    if confirm_rule == "t2":
        res["t2"] = t2(grid, conf, delta)
        res["confirmation_plan"] = confirmation_plan(grid, res["t2"])
        write_json(out / "confirmations.json", res["confirmation_plan"])
        (out / "confirmations.tsv").write_text("".join(
            f"{p['label']}\t{p['model_dir']}\t{p['target_fpr']:g}\t{p['reaction']}\t{p['corridor']}\n"
            for p in res["confirmation_plan"]), encoding="utf-8")
    else:
        fa = front_analysis(grid, conf)
        confirmed = fa["front"]
        res.update(front=fa["front"], margin=fa["margin"], confirmation_plan=fa["plan"],
                   appetite=appetite(fa["front"], grid, fa["margin"]["X"]))
        (out / "confirmations-front.tsv").write_text("".join(
            f"{p['label']}\t{p['model_dir']}\t{p['target_fpr']:g}\t{p['reaction']}\t{p['corridor']}\t"
            f"{'-' if p['kappa'] is None else format(p['kappa'], 'g')}\t{'-' if p['q'] is None else format(p['q'], 'g')}"
            f"\t{p['key']}\n" for p in fa["plan"]), encoding="utf-8")
        write_json(out / "front.json", {"front": fa["front"], "margin": fa["margin"], "plan": fa["plan"],
                                        "delta": delta})
        (out / "front.md").write_text(front_markdown(res), encoding="utf-8")
        write_json(out / "appetite.json", {"deltas": list(DELTA_PRIMES), "worst_slack": WORST_SLACK,
                                           "X": fa["margin"]["X"], "appetite": res["appetite"]})
        (out / "appetite.md").write_text(appetite_markdown(res), encoding="utf-8")
        appetite_plot(res["appetite"], out / "appetite.png", delta)
    write_json(out / "summary.json", res)
    (out / "hypotheses.md").write_text(hypotheses_markdown(res, grid), encoding="utf-8")
    if all(c.get("friction") for s in grid.values() for c in grid_cells(s)):
        front_plot(grid, out / "front-blocks.png", "share_blocked_card_windows",
                   "false-blocked card-months of legitimate cards (%)", delta, confirmed)
        front_plot(grid, out / "front-confirm.png", "share_confirm_card_days",
                   "card-days with an owner confirmation (%)", delta, confirmed)
    if res["r19"]:
        (out / "r19.md").write_text(r19_markdown(res["r19"]), encoding="utf-8")
        r19_plot(res["r19"], out / "r19.png")
    return res


def _prov_text(res: dict) -> str:
    return "; ".join(f"git `{p['git_sha']}` (dirty: {p['git_dirty']}), machine {p['host_machine']}"
                     for p in res["provenances"])


def violations_line(v: dict) -> str:
    from .policy_sweep import VIOLATION_KEYS  # noqa: F401
    text = (f"{v['n_bound_violations']} violations of the corridor bound in {v['n_corridor_runs']} corridor runs "
            f"({v['n_cells']} corridor-on cells with counters); headroom-bound violations {v['n_headroom_violations']}"
            f", B(h) block violations {v['n_block_violations']}.")
    if v["n_on_cells_without_counters"]:
        text += (f" {v['n_on_cells_without_counters']} corridor-on cells (Task 19 records) carry no counters and were "
                 "not recomputed.")
    return f"**{text}**" if v["any"] else text


def hypotheses_markdown(res: dict, grid: dict) -> str:
    R = res["resolution"]
    v = res["violations"]
    title = "Task 19b" if res.get("confirm_rule") == "front" else "Task 19"
    lines = [f"# {title}: pre-registered hypotheses", ""]
    if v["any"]:
        lines += [violations_line(v), ""]
    lines += [f"δ = {res['delta']:g}; R (resolution) = {100 * R:.1f} pp. Cells and friction entries: {_prov_text(res)}.",
              "", "Corridor bounds: " + violations_line(v), "",
              "## T1: a stricter policy lowers U at the price of friction", "",
              "One-step pairs softer → stricter on one axis, the other axes fixed. Supported: median ΔU < −R and median "
              "Δfriction > 0 (axis friction: false-blocked card-months for tau / escalation / max_declines, "
              "confirmation card-days for the corridor, kappa and q). Refuted: median ΔU ≥ 0. Otherwise inconclusive. "
              "Axes kappa / q (corridor strength): stricter = smaller κ of the V1 daily split / lower corridor "
              "quantile q, at the same τ, reaction and other corridor parameter.", "",
              "| axis | model | pairs | median ΔU | median Δ blocked card-months | median Δ confirm card-days | verdict |",
              "|---|---|---:|---:|---:|---:|---|"]
    for axis, by in res["t1"].items():
        for lab, r in by.items():
            if r["n_pairs"] == 0:
                lines.append(f"| {axis} | {lab} | 0 | – | – | – | no pairs |")
                continue
            lines.append(f"| {axis} | {lab} | {r['n_pairs']} | {_pct(r['median_dU'])} | "
                         f"{_pct(r['median_d_blocked_card_months'], 2)} | {_pct(r['median_d_confirm_card_days'], 2)} | "
                         f"{r['verdict']} |")
    if "t2" in res:
        lines += ["", "## T2: some policy passes the gate (confirmed with fresh seeds)", "",
                  "| model | green cells (U) | picked (lowest false blocks) | confirmation | T2 |", "|---|---|---|---|---|"]
        for lab, r in res["t2"].items():
            gc = ", ".join(f"`{g['key']}` ({_pct(g['U'])}, recall {g['test_recall']:.3f}"
                           + ("" if g["recall_ok"] is not False else " < recall_clean_min") + ")"
                           for g in r["green_cells"]) or "none"
            cf = "–" if r["confirmation"] is None else (f"seeds {r['confirmation']['attack_seeds']}: U "
                                                        f"{_pct(r['confirmation']['U'])}")
            lines.append(f"| {lab} | {gc} | {r['picked'] or '–'} | {cf} | {r['t2']} |")
    else:
        lines += ["", "T2 («a green cell exists», Task 19) stays in the Task 19 outputs; here the front is confirmed "
                  "on fresh seeds instead (`front.md`, `appetite.md`)."]
    pv = res["policy_vs_model"]
    lines += ["", "## Policy vs model (R30)", "",
              "Policy span = U(softest: lowest target FPR, md3, corridor off) − U(strictest: highest target FPR, md1, "
              "corridor on); model span = max − min of U over the models per cell; seed spread = max − min of the run "
              "upper bounds over seeds within one attack and cell (cells with the corridor threat file's corridor).",
              "", "| model | policy span |", "|---|---:|"]
    lines += [f"| {lab} | {_pct(x)} |" for lab, x in pv["policy_span"].items()]
    lines += ["", f"- median policy span {_pct(pv['median_policy_span'])}, median model span "
              f"{_pct(pv['median_model_span'])} ({len(pv['model_span'])} cells), median seed spread "
              f"{_pct(pv['median_seed_spread'])} ({pv['n_seed_spreads']} attack × cell groups)",
              f"- verdict «more from policy than from model»: **{pv['verdict']}**", ""]
    lines += interaction_markdown(res["interaction"])
    lines += ["## Cells (U per model)", "", "| cell | " + " | ".join(grid) + " |", "|---|" + "---:|" * len(grid)]
    keys = sorted({c["key"] for s in grid.values() for c in grid_cells(s)})
    for k in keys:
        row = []
        for s in grid.values():
            c = next((c for c in s["cells"] if c["key"] == k), None)
            row.append("–" if c is None else _pct(c["U"]) + (" ✓" if green(c, res["delta"]) else ""))
        lines.append(f"| `{k}` | " + " | ".join(row) + " |")
    lines += ["", "✓ = green (every run ≤ δ).", ""]
    return "\n".join(lines)


def interaction_markdown(it: dict) -> list[str]:
    lines = ["## Policy × model: descriptive interaction table (R32)", "",
             "Descriptive only, no test and no verdict (R32): per axis and model family the median ΔU of the T1 "
             "pairs on the base and on the hardened model, their difference (hardened − base) and the two T1 "
             "verdicts side by side.", ""]
    if it["rows"]:
        lines += ["| axis | family | median ΔU base | median ΔU hardened | difference (hardened − base) | "
                  "T1 base | T1 hardened |", "|---|---|---:|---:|---:|---|---|"]
        for r in it["rows"]:
            lines.append(f"| {r['axis']} | {r['family']} ({r['base']} / {r['hardened']}) | {_pct(r['median_dU_base'])}"
                         f" | {_pct(r['median_dU_hardened'])} | {_pct(r['difference'])} | {r['verdict_base']} | "
                         f"{r['verdict_hardened']} |")
    else:
        lines.append("No model family with both a `<family>-base` and a `<family>-hard` label.")
    lines += ["", "Corridor's relative effect per model: median over the Task 19 corridor pairs (off → on, the "
              "corridor threat file's corridor) of U_on / U_off.", "", "| model | median U_on / U_off | pairs | "
              "skipped (U_off ≤ 0) |", "|---|---:|---:|---:|"]
    for lab, e in it["corridor_relative_effect"].items():
        m = "–" if e["median_ratio"] is None else f"{e['median_ratio']:.2f}"
        lines.append(f"| {lab} | {m} | {e['n_pairs']} | {e['n_skipped']} |")
    return lines + [""]


def front_markdown(res: dict) -> str:
    m = res["margin"]
    lines = ["# Task 19b: Pareto front confirmed on fresh seeds", "",
             "Front per model: the Pareto set of every policy cell (Task 19 grid + Task 19b corridor variants; the R19 "
             "extras md2 / mdinf excluded) on (U on the gate seeds, false-blocked card-months), both minimised, strict "
             "dominance, exact ties kept. Confirmation: the same cell, all attacks × fresh seeds; per seed U_s = max "
             "over the attacks of the one-sided upper bound; x_c = worst_s U_s − U (gate seeds).", "",
             (f"**PROVISIONAL – {m['n_pending']} of {m['n_front']} front cells are not confirmed yet.** "
              if m.get("provisional") else "")
             + f"**Margin X = {_pct(m['X'], 2)}** (max x_c over {m['n_confirmed']} confirmed of {m.get('n_front', '?')} "
             "front cells of all models); "
             f"90th percentile of x_c {_pct(m['p90'], 2)}; per model max: "
             + ", ".join(f"{lab} {_pct(x, 2)}" for lab, x in m["per_model"].items()) + ".",
             "Recommendation for a bank that runs only the 3-seed gate: take a policy with U (gate seeds) ≤ δ′ − X.",
             ""]
    for lab, rows in res["front"].items():
        lines += [f"## {lab}", "", "| policy | U (gate seeds) | U$ | blocked card-months | cards ever blocked | "
                  "confirm card-days | U_s per seed | median U_s | worst U_s | x_c | note |",
                  "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
        for r in rows:
            cf = r["confirmation"]
            us = "pending" if cf is None else ", ".join(f"{sd}: {_pct(u)}" for sd, u in cf["U_s"].items())
            lines.append(f"| {r['policy']} | {_pct(r['U'])} | {r['U_usd']:.0f} $ | {_pct(r['blocked_card_months'], 2)} | "
                         f"{_pct(r['blocked_cards'], 2)} | {_pct(r['confirm_card_days'], 2)} | {us} | "
                         f"{'–' if cf is None else _pct(cf['median'])} | {'–' if cf is None else _pct(cf['worst'])} | "
                         f"{'–' if cf is None else _pct(cf['x_c'], 2)} | {MD1_NOTE if r['md1'] else ''} |")
        lines += ["", "Run upper bounds of the confirmations:", ""]
        for r in rows:
            if r["confirmation"]:
                lines.append(f"- {r['policy']}: " + ", ".join(f"{k} {_pct(u)}"
                                                             for k, u in r["confirmation"]["run_upper"].items()))
        lines.append("")
    return "\n".join(lines)


def _choice(e: dict | None, confirmed: bool) -> list[str]:
    if e is None:
        return ["none"] + ["–"] * (6 if confirmed else 5)
    out = [e["policy"] + (f" – {MD1_NOTE}" if e["md1"] else ""), _pct(e["U"])]
    if confirmed:
        out.append(f"{_pct(e.get('median'))} / {_pct(e.get('worst'))}")
    out += [_pct(e["blocked_card_months"], 2), _pct(e["blocked_cards"], 2), _pct(e["confirm_card_days"], 2),
            f"{e['U_usd']:.0f} $"]
    return out


def appetite_markdown(res: dict) -> str:
    X = res["margin"]["X"]
    prov = res["margin"].get("provisional")
    lines = ["# Task 19b: bank risk appetite δ′ → cheapest passing policy → its price for customers", ""]
    if prov:
        lines += [f"**PROVISIONAL: {res['margin']['n_pending']} of {res['margin']['n_front']} front cells are not "
                  "confirmed yet; X and the curves change when they are.**", ""]
    lines += [
             f"δ′ = {DELTA_PRIMES[0]:.2f} … {DELTA_PRIMES[-1]:.2f}, step 1 pp (below 0.08 the gate's resolution, 3.6–4.1 pp, "
             "makes the curve unreliable). (A) cheapest confirmed front cell that passes at δ′: median_s U_s ≤ δ′ and "
             f"worst_s U_s ≤ δ′ + {WORST_SLACK:g} on the fresh seeds. (B) recommended with margin X = {_pct(X, 2)}: "
             "the cheapest policy cell of the model with U (gate seeds) ≤ δ′ − X. Cheapest = lowest false-blocked "
             "card-months, ties → lowest confirmation card-days. A ≥ 90 % probability statement would need ≈ 29 runs "
             f"per cell (not done). The row δ′ = {res['delta']:g} is the gate δ = {100 * res['delta']:g} %.", ""]
    hdr_a = "(A) policy | U | median / worst on fresh seeds | blocked card-months | cards ever blocked | confirm card-days | U$"
    hdr_b = "(B) policy | U | blocked card-months | cards ever blocked | confirm card-days | U$"
    for lab, pts in res["appetite"].items():
        for tag, title in (("with_md1", "all policies"), ("without_md1", "practical policies (md3, md3+esc)")):
            lines += [f"## {lab}: {title}", "", f"| δ′ | {hdr_a} | {hdr_b} |", "|---:|" + "---|" * 13]
            for e in pts:
                d = f"**{e['delta']:.2f} (gate δ = {100 * res['delta']:g} %)**" \
                    if abs(e["delta"] - res["delta"]) < EPS else f"{e['delta']:.2f}"
                lines.append(f"| {d} | " + " | ".join(_choice(e[f"A_{tag}"], True) + _choice(e[f"B_{tag}"], False))
                             + " |")
            lines.append("")
    return "\n".join(lines)


def appetite_plot(app: dict, path: Path, delta: float) -> None:
    """x = δ′, y = false-blocked card-months of the chosen policy; step lines per model, (A) solid, (B) dashed; left
    all policies (md1 choices grey), right without md1; the gate's δ as a vertical line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a4fd1", "#c23b6b"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, tag, title in ((axes[0], "with_md1", "all policies (md1 grey: impractical)"),
                           (axes[1], "without_md1", "practical policies (md3, md3+esc)")):
        for i, (lab, pts) in enumerate(app.items()):
            col = palette[i % 5]
            x = [e["delta"] * 100 for e in pts]
            for kind, ls, name in (("A", "-", "(A) confirmed"), ("B", "--", "(B) with margin X")):
                ch = [e[f"{kind}_{tag}"] for e in pts]
                y = [np.nan if c is None else 100 * c["blocked_card_months"] for c in ch]
                ax.step(x, y, where="post", color=col, ls=ls, lw=1.4, label=f"{lab} {name}")
                g = [(a, b) for a, b, c in zip(x, y, ch) if c is not None and c["md1"]]
                if g:
                    ax.scatter(*zip(*g), s=14, color="#9a9a9a", zorder=3)
        ax.set_xlim(100 * DELTA_PRIMES[0] - 1, 100 * DELTA_PRIMES[-1] + 1)
        if not any(e[f"{k}_{tag}"] for pts in app.values() for e in pts for k in ("A", "B")):
            ax.set_ylim(0, 100)
            ax.text(29, 50, "no policy chosen at any δ′ (no confirmed front cell yet?)", ha="center", fontsize=9,
                    color="#555555")
        ax.axvline(100 * delta, color="#c0392b", lw=1.2, ls=":")
        ax.text(100 * delta, ax.get_ylim()[1], f" gate δ = {100 * delta:g} %", color="#c0392b", fontsize=8,
                va="top", ha="left")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("bank risk appetite δ′ (% of A)")
        ax.grid(color="#e5e5e5", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("false-blocked card-months of the chosen policy (%)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
