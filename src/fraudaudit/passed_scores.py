"""Task 27d (R39 order 5, R48, R50): scores of the adversarial operations that passed, xgb-base vs xgb-hard.

Descriptive check of the escalation hypothesis on the grid cell FPR 0.01 md3 corridor off (ES, seeds 1–3, the stored
targets). The stored records keep no operations, so the cell is rerun per model with ``policy-sweep --trace``
(``gate-27d-es-1-3.yaml``); ``p`` is the score of the bank's decision (``trace.py``).

τ is the threshold the bank decided with (the trace's ``tau_decision``: the model.json τ when the cell's τ is within rel
1e-6 of it, ``sweep_cells.model_at_tau``); the cell's τ is kept as ``cell_tau``. From the trace: every **passed**
adversarial operation (corridor off, md3, no escalation – so p < τ; a passed operation
with p ≥ τ is refused as an inconsistent trace). Per model, per seed and pooled over the seeds: the number of passed
operations (and of model-declined ones, for context); quantiles (min, P10, P25, median, P75, P90, max; numpy linear
interpolation) of p and of p / τ; the **share of passed operations with p ≥ τ′** (τ′ = ``tau_esc`` of the model, val
FPR 0.05: they would be declined after an escalation) and per target the share of targets (pooled: target-runs) with at
least one passed operation with p ≥ τ′. Figure ``passed-scores-{uk,en}.png``: the distributions of log10(p / τ) of the
two models with vertical lines at τ (0) and τ′ (per model); operations with p = 0 cannot be drawn and are counted.

**Cross-check (exact):** each run ``es/s`` of the new cell equals run ``es/s`` of the same key in the grid file:
``uplift_mean``, ``uplift_upper``, ``ver_adv_mean``, ``ver_clean_mean``, ``asr`` and the per-target ``ver_adv`` /
``ver_clean`` lists, compared with ``==``. A difference is reported in the table; the outputs are still written and
the command exits non-zero (``all_equal`` false).

    fraudaudit passed-scores --cell LABEL,SWEEP,KEY,GRID_SWEEP ... --out DIR
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .md_study import _cell, _load, check_runs, check_same_setup, check_trace, provenance
from .trace import load_trace, trace_path

SEEDS = (1, 2, 3)
ATTACK = "es"
QUANTILES = (("min", 0), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("max", 100))
CROSS_FIELDS = ("uplift_mean", "uplift_upper", "ver_adv_mean", "ver_clean_mean", "asr", "ver_adv", "ver_clean")


def quantiles(xs) -> dict | None:
    a = np.asarray(list(xs), dtype=np.float64)
    if a.size == 0:
        return None
    return {name: float(np.percentile(a, q)) for name, q in QUANTILES}


def score_stats(per_target: list[dict], tau: float, tau_esc: float) -> dict:
    """Statistics of the passed operations of the given per-target trace rows (one run or several pooled)."""
    passed = [o for t in per_target for o in t["ops"] if o["passed"]]
    n_declined = sum(o["declined"] for t in per_target for o in t["ops"])
    bad = [o["p"] for o in passed if o["p"] >= tau]
    if bad:
        raise ValueError(f"{len(bad)} passed operations with p >= tau {tau} (corridor off, md3, no escalation): "
                         "the trace is inconsistent")
    p = [o["p"] for o in passed]
    hit = [any(o["passed"] and o["p"] >= tau_esc for o in t["ops"]) for t in per_target]
    return {"n_targets": len(per_target), "n_passed": len(passed), "n_declined": int(n_declined),
            "n_zero_p": int(sum(x <= 0 for x in p)),
            "p": quantiles(p), "p_over_tau": quantiles(x / tau for x in p),
            "n_passed_ge_tau_esc": int(sum(x >= tau_esc for x in p)),
            "share_passed_ge_tau_esc": float(np.mean([x >= tau_esc for x in p])) if p else None,
            "share_targets_with_passed_ge_tau_esc": float(np.mean(hit)) if hit else None}


def cross_check(cell: dict, grid_cell: dict, seeds=SEEDS) -> dict:
    rows = []
    for s in seeds:
        rk = f"{ATTACK}/{int(s)}"
        g = grid_cell["runs"].get(rk)
        if g is None:
            raise ValueError(f"grid cell {grid_cell['key']} has no run {rk}")
        for f in CROSS_FIELDS:
            if f not in g:
                raise ValueError(f"grid cell {grid_cell['key']} run {rk} has no {f}")
            a, b = cell["runs"][rk][f], g[f]
            rows.append({"run": rk, "field": f, "equal": a == b,
                         "new": a if not isinstance(a, list) else None, "grid": b if not isinstance(b, list) else None,
                         "n_differing": None if not isinstance(a, list) else
                         (abs(len(a) - len(b)) + sum(x != y for x, y in zip(a, b)))})
    return {"rows": rows, "all_equal": all(r["equal"] for r in rows)}


def passed_model(label: str, sweep_p, key: str, grid_p, seeds=SEEDS) -> dict:
    d, g = _load(sweep_p), _load(grid_p)
    if d.get("model_json_sha256") != g.get("model_json_sha256"):
        raise ValueError(f"{label}: model_json_sha256 of {sweep_p} differs from the grid {grid_p}")
    c, gc = _cell(d, key, sweep_p), _cell(g, key, grid_p)
    if c.get("corridor") != "off" or c.get("tau_after_decline") is not None:
        raise ValueError(f"{label} {key}: the cell must be corridor off without escalation")
    check_same_setup(c, d, gc, g, f"{label} {key} (grid)")
    check_runs(c, seeds, label)
    p = trace_path(Path(sweep_p).parent, key)
    if not p.is_file():
        raise ValueError(f"{label} {key}: no trace file {p}")
    tr = load_trace(p)
    check_trace(tr, c, f"{label} {key}")
    cell_tau, tau_esc = float(tr["tau"]), float(tr["tau_esc"])
    if cell_tau != float(c["tau"]) or tau_esc != float(d["tau_esc"]):
        raise ValueError(f"{label} {key}: tau / tau_esc of the trace differ from the sweep")
    if tr.get("tau_decision") is None:
        raise ValueError(f"{label} {key}: the trace has no tau_decision (the threshold the bank decided with)")
    tau = float(tr["tau_decision"])  # p < τ check and p / τ use the threshold of the decisions
    if not 0 < tau_esc < tau:
        raise ValueError(f"{label}: tau_esc {tau_esc} must be in (0, tau {tau})")
    per_seed, pooled = {}, []
    for s in seeds:
        rows = tr["runs"][f"{ATTACK}/{int(s)}"]["per_target"]
        per_seed[str(int(s))] = score_stats(rows, tau, tau_esc)
        pooled += rows
    return {"label": label, "key": key, "tau": tau, "cell_tau": cell_tau, "tau_esc": tau_esc, "esc_fpr": d.get("esc_fpr"),
            "seeds": [int(s) for s in seeds], "per_seed": per_seed, "pooled": score_stats(pooled, tau, tau_esc),
            "log10_p_over_tau": [math.log10(o["p"] / tau) for t in pooled for o in t["ops"]
                                 if o["passed"] and o["p"] > 0],
            "cross_check": cross_check(c, gc, seeds),
            "inputs": {"sweep": provenance(sweep_p, d), "grid": provenance(grid_p, g), "trace": str(p)}}


def passed_scores(cells, seeds=SEEDS) -> dict:
    """``cells``: iterable of (label, sweep path, key, grid path)."""
    rows = [passed_model(lab, sp, k, gp, seeds) for lab, sp, k, gp in cells]
    if len({r["label"] for r in rows}) != len(rows):
        raise ValueError("a label is given twice")
    return {"models": rows, "seeds": [int(s) for s in seeds], "attack": ATTACK,
            "all_equal": all(r["cross_check"]["all_equal"] for r in rows)}


# --- markdown and figure -----------------------------------------------------------------------------------------


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def _q(q: dict | None, fmt) -> str:
    return " | ".join("–" for _ in QUANTILES) if q is None else " | ".join(fmt(q[n]) for n, _ in QUANTILES)


def markdown(res: dict) -> str:
    lines = ["# Scores of the passed adversarial operations (Task 27d)", "",
             f"ES runs, seeds {', '.join(map(str, res['seeds']))}; grid cell md3, corridor off, no escalation (every "
             "passed operation has p < τ). τ′ = the escalation threshold of the model (val FPR 0.05). Descriptive: no "
             "verdict.", "",
             "| model | seed | passed ops | declined ops | passed with p ≥ τ′ | targets with a passed op ≥ τ′ | "
             "p = 0 |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in res["models"]:
        for s, st in list(m["per_seed"].items()) + [("pooled", m["pooled"])]:
            lines.append(f"| {m['label']} | {s} | {st['n_passed']} | {st['n_declined']} | "
                         f"{st['n_passed_ge_tau_esc']} ({_pct(st['share_passed_ge_tau_esc'])}) | "
                         f"{_pct(st['share_targets_with_passed_ge_tau_esc'])} | {st['n_zero_p']} |")
    head = " | ".join(n for n, _ in QUANTILES)
    lines += ["", f"| model | τ | τ′ | quantity | seed | {head} |", "|---|---:|---:|---|---|" + "---:|" * len(QUANTILES)]
    for m in res["models"]:
        for s, st in list(m["per_seed"].items()) + [("pooled", m["pooled"])]:
            lines.append(f"| {m['label']} | {m['tau']:.4g} | {m['tau_esc']:.4g} | p | {s} | "
                         f"{_q(st['p'], lambda x: f'{x:.3g}')} |")
            lines.append(f"| {m['label']} | {m['tau']:.4g} | {m['tau_esc']:.4g} | p / τ | {s} | "
                         f"{_q(st['p_over_tau'], lambda x: f'{100 * x:.1f} %')} |")
    lines += ["", "## Cross-check with the grid (exact, `==`)", "", "| model | run | field | new | grid | equal |",
              "|---|---|---|---|---|---|"]
    for m in res["models"]:
        for r in m["cross_check"]["rows"]:
            new = f"{r['n_differing']} differing" if r["n_differing"] is not None else repr(r["new"])
            grid = "list" if r["n_differing"] is not None else repr(r["grid"])
            lines.append(f"| {m['label']} | {r['run']} | {r['field']} | {new} | {grid} | "
                         f"{'equal' if r['equal'] else 'DIFFERS'} |")
    lines += ["", f"All runs equal to the grid: {'yes' if res['all_equal'] else 'NO'}.", "", "## Inputs", ""]
    for m in res["models"]:
        i = m["inputs"]
        lines.append(f"- {m['label']}: sweep `{i['sweep']['path']}` ({i['sweep']['git_sha']}), grid "
                     f"`{i['grid']['path']}` ({i['grid']['git_sha']}); model `{i['sweep']['model_json_sha256']}`")
    return "\n".join(lines) + "\n"


TEXT = {
    "en": {"x": "log10(p / τ) of the passed adversarial operations", "y": "share of passed operations",
           "tau": "τ (decline threshold)", "esc": "τ′ {lab} (after an escalation)"},
    "uk": {"x": "log10(p / τ) пропущених змагальних операцій", "y": "частка пропущених операцій",
           "tau": "τ (поріг відхилення)", "esc": "τ′ {lab} (після ескалації)"},
}
COLOURS = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a4fd1", "#c23b6b"]


def _num(x: float, lang: str, d: int = 1) -> str:
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if lang == "uk" else s


def plot(res: dict, path, lang: str = "en") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    allv = [v for m in res["models"] for v in m["log10_p_over_tau"]]
    lo = min(allv + [math.log10(m["tau_esc"] / m["tau"]) for m in res["models"]] + [-1.0])
    bins = np.linspace(math.floor(lo * 4) / 4, 0.0, 41)
    for i, m in enumerate(res["models"]):
        c = COLOURS[i % len(COLOURS)]
        v = np.asarray(m["log10_p_over_tau"], dtype=np.float64)
        if v.size:
            ax.hist(v, bins=bins, weights=np.full(v.size, 1.0 / v.size), histtype="step", lw=1.6, color=c,
                    label=f"{m['label']} (n = {v.size})")
        ax.axvline(math.log10(m["tau_esc"] / m["tau"]), color=c, ls=":", lw=1.4, label=t["esc"].format(lab=m["label"]))
    ax.axvline(0.0, color="#333333", ls="--", lw=1.0, label=t["tau"])
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, lang, 1)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, lang, 2)))
    ax.grid(color="#e5e5e5", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write(res: dict, out_dir) -> None:
    from .sweep_cells import write_json
    out = Path(out_dir)
    write_json(out / "passed-scores.json", res)
    (out / "passed-scores.md").write_text(markdown(res), encoding="utf-8")
    for lang in ("en", "uk"):
        plot(res, out / f"passed-scores-{lang}.png", lang)
