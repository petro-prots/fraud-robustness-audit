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
  the median policy span exceeds both the median model span and the median seed spread. Models whose policy span is
  above R while another's is not are reported as an interaction.
* **R19**: for every model with the cells ``mdinf``, ``md1``, ``md2``, ``md3`` at its own τ without the corridor, the
  relative overestimation (VER_∞ − VER_md) / VER_md of the ES runs (VER per target averaged over the ES seeds) and of
  the clean leg, with a paired percentile bootstrap over targets.
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
AXIS_FRICTION = {"tau": "share_blocked_card_windows", "escalation": "share_blocked_card_windows",
                 "max_declines": "share_blocked_card_windows", "corridor": "share_confirm_card_days"}
N_BOOT = 2000


def grid_cells(sweep: dict) -> list[dict]:
    """The 3 × 3 × 2 policy cells (the R19 extra reactions excluded)."""
    return [c for c in sweep["cells"] if c["reaction"] in GRID_REACTIONS]


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


def t1(sweeps: dict, resolution: float = RESOLUTION) -> dict:
    """``{axis: {label: t1_axis, "pooled": t1_axis}}`` over the grid cells of every model."""
    out = {}
    for axis in AXES:
        per = {lab: one_step_pairs(grid_cells(s), axis) for lab, s in sweeps.items()}
        out[axis] = {lab: t1_axis(p, axis, resolution) for lab, p in per.items()}
        out[axis]["pooled"] = t1_axis([x for p in per.values() for x in p], axis, resolution)
    return out


def _extreme(cells, fpr_pick, reaction, corridor):
    idx = _index(cells)
    fprs = sorted({k[0] for k in idx})
    return idx.get((fpr_pick(fprs), reaction, corridor)) if fprs else None


def policy_vs_model(sweeps: dict, resolution: float = RESOLUTION) -> dict:
    spans = {}
    for lab, s in sweeps.items():
        g = grid_cells(s)
        soft, strict = _extreme(g, min, *SOFTEST), _extreme(g, max, *STRICTEST)
        spans[lab] = None if soft is None or strict is None else float(soft["U"] - strict["U"])
    by_cell: dict = {}
    for lab, s in sweeps.items():
        for k, c in _index(grid_cells(s)).items():
            by_cell.setdefault(k, {})[lab] = c["U"]
    model_spans = {f"fpr={k[0]:g}|{k[1]}|corridor={k[2]}": float(max(v.values()) - min(v.values()))
                   for k, v in sorted(by_cell.items()) if len(v) >= 2}
    seed_spreads = []
    for s in sweeps.values():
        for c in grid_cells(s):
            by_attack: dict = {}
            for rk, r in c["runs"].items():
                by_attack.setdefault(rk.rsplit("/", 1)[0], []).append(r["uplift_upper"])
            seed_spreads += [max(v) - min(v) for v in by_attack.values() if len(v) >= 2]
    ps = [v for v in spans.values() if v is not None]
    med_p = float(np.median(ps)) if ps else None
    med_m = float(np.median(list(model_spans.values()))) if model_spans else None
    med_s = float(np.median(seed_spreads)) if seed_spreads else None
    if med_p is None or med_m is None or med_s is None:
        verdict = "not computable"
    else:
        verdict = "supported" if med_p > med_m and med_p > med_s else "not supported"
    above = sorted(lab for lab, v in spans.items() if v is not None and v > resolution)
    below = sorted(lab for lab, v in spans.items() if v is not None and v <= resolution)
    interaction = (f"policy span above R ({100 * resolution:.1f} pp) only on {', '.join(above)}; not on "
                   f"{', '.join(below)}" if above and below else "none")
    return {"policy_span": spans, "median_policy_span": med_p, "model_span": model_spans,
            "median_model_span": med_m, "median_seed_spread": med_s, "n_seed_spreads": len(seed_spreads),
            "verdict": verdict, "interaction": interaction}


def green(c: dict, delta: float) -> bool:
    return all(r["uplift_upper"] <= delta for r in c["runs"].values())


def t2(sweeps: dict, confirmations: dict, delta: float) -> dict:
    """Per model: green cells, the cell picked for confirmation and the confirmation verdict."""
    out = {}
    for lab, s in sweeps.items():
        gs = [c for c in grid_cells(s) if green(c, delta)]
        pick = None
        if gs:
            pick = min(gs, key=lambda c: (_fr(c, "share_blocked_card_windows"), _fr(c, "share_confirm_card_days"),
                                          c["key"]))
        conf = None
        if pick is not None and lab in confirmations:
            cc = next((c for c in confirmations[lab]["cells"] if c["key"] == pick["key"]), None)
            if cc is not None:
                conf = {"key": cc["key"], "U": cc["U"], "green": green(cc, delta),
                        "attack_seeds": confirmations[lab]["gate"]["attack_seeds"]}
        out[lab] = {"green_cells": [{"key": c["key"], "U": c["U"]} for c in gs],
                    "picked": None if pick is None else pick["key"], "confirmation": conf,
                    "t2": "no green cell" if pick is None else ("pending confirmation" if conf is None else
                                                              ("confirmed" if conf["green"] else "not confirmed"))}
    return out


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
    rs = rs[np.isfinite(rs)]
    alpha = 1 - confidence
    lo, hi = (np.quantile(rs, [alpha / 2, 1 - alpha / 2]) if rs.size else (float("nan"), float("nan")))
    return {"ver_inf": float(a.mean()), "ver_md": float(b.mean()), "rel": est, "lower": float(lo), "upper": float(hi)}


def r19(sweeps: dict, attack: str = "es", seed: int = 42) -> dict:
    out = {}
    for lab, s in sweeps.items():
        own = round(float(s["own_target_fpr"]), 12)
        idx = _index(s["cells"])
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


def r19_markdown(res: dict, attack: str = "es") -> str:
    lines = ["# R19: loss-estimate bias without the bank's reaction", "",
             "Own τ, no corridor. VER per target averaged over the ES runs (seeds of the gate) for the adversarial "
             "leg; the clean leg replays the real operations. Relative overestimation = (VER_∞ − VER_md) / VER_md, "
             "VER_∞ = never blocked; 95 % paired percentile bootstrap over targets.", "",
             "| model | leg | max_declines | VER_∞ | VER_md | overestimation [95 % CI] |", "|---|---|---:|---:|---:|---|"]
    for lab, legs in res.items():
        for leg, by in legs.items():
            for md, r in by.items():
                lines.append(f"| {lab} | {leg.replace('adv_', 'adversarial ')} | {md[2:]} | {_pct(r['ver_inf'])} | "
                             f"{_pct(r['ver_md'])} | {_pct(r['rel'])} [{_pct(r['lower'])}; {_pct(r['upper'])}] |")
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


# --- fronts -----------------------------------------------------------------------------------------------------


def pareto_flags(points: list[tuple]) -> list[bool]:
    from .corridor_sweep import pareto
    return pareto(points)


def front_plot(sweeps: dict, path: Path, friction_key: str, xlabel: str, delta: float) -> None:
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
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- summary ----------------------------------------------------------------------------------------------------


def load(paths) -> tuple[dict, dict]:
    """(grid sweeps, confirmation sweeps) by label from ``policy-sweep.json`` files."""
    grid, conf = {}, {}
    for p in paths:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        target = conf if s.get("role") == "confirmation" else grid
        if s["label"] in target:
            raise ValueError(f"label {s['label']!r} appears twice among the {s.get('role', 'grid')} sweeps")
        target[s["label"]] = s
    return grid, conf


def summarize(paths, out_dir, resolution: float = RESOLUTION) -> dict:
    grid, conf = load(paths)
    if not grid:
        raise ValueError("no grid sweeps given")
    deltas = {s["gate"]["delta"] for s in grid.values()}
    if len(deltas) != 1 or None in deltas:
        raise ValueError(f"the sweeps disagree on δ or δ is null: {sorted(map(str, deltas))}")
    delta = deltas.pop()
    machines = {json.dumps((s.get("meta", {}).get("machine") or {}).get("host_machine")) for s in grid.values()}
    res = {"delta": delta, "resolution": resolution, "t1": t1(grid, resolution),
           "policy_vs_model": policy_vs_model(grid, resolution), "t2": t2(grid, conf, delta), "r19": r19(grid),
           "labels": list(grid), "machines": sorted(machines),
           "mean_amount": {lab: s.get("mean_amount") for lab, s in grid.items()},
           "delta_usd": {lab: s.get("delta_usd") for lab, s in grid.items()}}
    res["confirmation_plan"] = confirmation_plan(grid, res["t2"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", res)
    write_json(out / "confirmations.json", res["confirmation_plan"])
    (out / "confirmations.tsv").write_text("".join(
        f"{p['label']}\t{p['model_dir']}\t{p['target_fpr']:g}\t{p['reaction']}\t{p['corridor']}\n"
        for p in res["confirmation_plan"]), encoding="utf-8")
    (out / "hypotheses.md").write_text(hypotheses_markdown(res, grid), encoding="utf-8")
    if all(c.get("friction") for s in grid.values() for c in grid_cells(s)):
        front_plot(grid, out / "front-blocks.png", "share_blocked_card_windows",
                   "false-blocked card-months of legitimate cards (%)", delta)
        front_plot(grid, out / "front-confirm.png", "share_confirm_card_days",
                   "card-days with an owner confirmation (%)", delta)
    if res["r19"]:
        (out / "r19.md").write_text(r19_markdown(res["r19"]), encoding="utf-8")
        r19_plot(res["r19"], out / "r19.png")
    return res


def hypotheses_markdown(res: dict, grid: dict) -> str:
    R = res["resolution"]
    lines = ["# Task 19: pre-registered hypotheses", "",
             f"δ = {res['delta']:g}; R (resolution) = {100 * R:.1f} pp; machines: {', '.join(res['machines'])}.", "",
             "## T1: a stricter policy lowers U at the price of friction", "",
             "One-step pairs softer → stricter on one axis, the other axes fixed. Supported: median ΔU < −R and median "
             "Δfriction > 0 (axis friction: false-blocked card-months for tau / escalation / max_declines, "
             "confirmation card-days for the corridor). Refuted: median ΔU ≥ 0. Otherwise inconclusive.", "",
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
    lines += ["", "## T2: some policy passes the gate (confirmed with fresh seeds)", "",
              "| model | green cells (U) | picked (lowest false blocks) | confirmation | T2 |", "|---|---|---|---|---|"]
    for lab, r in res["t2"].items():
        gc = ", ".join(f"`{g['key']}` ({_pct(g['U'])})" for g in r["green_cells"]) or "none"
        cf = "–" if r["confirmation"] is None else (f"seeds {r['confirmation']['attack_seeds']}: U "
                                                    f"{_pct(r['confirmation']['U'])}")
        lines.append(f"| {lab} | {gc} | {r['picked'] or '–'} | {cf} | {r['t2']} |")
    pv = res["policy_vs_model"]
    lines += ["", "## Policy vs model (R30)", "",
              "Policy span = U(softest: lowest target FPR, md3, corridor off) − U(strictest: highest target FPR, md1, "
              "corridor on); model span = max − min of U over the models per cell; seed spread = max − min of the run "
              "upper bounds over seeds within one attack and cell.", "",
              "| model | policy span |", "|---|---:|"]
    lines += [f"| {lab} | {_pct(v)} |" for lab, v in pv["policy_span"].items()]
    lines += ["", f"- median policy span {_pct(pv['median_policy_span'])}, median model span "
              f"{_pct(pv['median_model_span'])} ({len(pv['model_span'])} cells), median seed spread "
              f"{_pct(pv['median_seed_spread'])} ({pv['n_seed_spreads']} attack × cell groups)",
              f"- verdict «more from policy than from model»: **{pv['verdict']}**",
              f"- interaction: {pv['interaction']}", ""]
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
