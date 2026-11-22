"""``fraudaudit models-summary``: the summary over the 8 models of Tasks 19, 19b and 24 (Task 25; rulings R30, R33,
R37, R38, R41 (5), (8), (10); every rule fixed in the brief before this step). Computation only, from finished
outputs: the per-family ``summary.json`` files of ``policy-summary`` and the grid ``policy-sweep.json`` files they list.
R19 bootstraps, fronts and risk-appetite curves are taken from the summaries as they are, never recomputed.

* **(a) Spearman** (unit = model, n = 8, exact permutation p): VER_md against the relative (H_rel) and the absolute
  (H_abs) overestimation of the adversarial ES leg at md3 – the two main tests (no multiplicity correction); md1 for
  information only. Supported iff p < 0.05 and ρ < 0.
* **(b) Policy vs model** on 8 models (R30 rule unchanged), next to the three family results; the model span split into
  within-family (base vs hardened) and across-family (four bases, four hardened) spans, descriptive.
* **(c) Strict Pareto front** of the 8 models: test recall at the own τ (maximised) × the first δ′ of curve (A) without
  md1 (minimised); dominance pairs with Δδ′, Δrecall and Δ test fraud operations; the δ′ gaps below the resolution.
* **(d) Cost of hardening** per family (descriptive), checked against the register values.
* **(e) Tables**: the 18 Task 19 grid cells × 8 models (U) and the R19 table of the 8 models.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import policy_summary as ps
from .spearman import critical_rho, leave_one_out, spearman_exact
from .sweep_cells import write_json

RESOLUTION = ps.RESOLUTION  # 0.036, register К.2
ALPHA = 0.05
OWN_REACTION = ("md3", "off")  # the own-τ cell: md3, corridor off
NO_PASS = "no passing cell up to 0.50"
FAMILY_NAMES = {"m3": "M3", "m2": "M2", "xgb": "XGB", "gru": "GRU"}

# Register values the cost table must reproduce (brief Task 25 §2 (d)); recall compared at the third decimal, FPR at
# the precision the register gives (a fraction at the fourth decimal = % with two decimals).
REGISTER = {
    "m3": {"recall": (0.977, 0.935), "fpr": (0.0091, 0.0087), "row": "3.6 / К.2"},
    "m2": {"recall": (0.974, 0.936), "fpr": (0.0090, 0.0089), "row": "3.6 / К.2"},
    "xgb": {"recall": (0.9888, 0.9389), "fpr": (0.0084, 0.0079), "row": "5.17"},
    "gru": {"recall": (0.9650, 0.9096), "fpr": (0.0086, 0.0095), "row": "5.28"},
}

DEPENDENCE_NOTE = ("The 8 models are not independent: M2 uses a subset of the M3 features; the base and the hardened "
                   "model of one family share data and architecture; all models share the targets and seeds. n = 8 is "
                   "the number of models, not of independent observations.")
PARETO_CAVEATS = ("One training per model (the spread over retrainings is not measured); the result holds in these "
                  "configurations only; post-hoc analysis: the axes are quantities of the method, the δ′ tolerance "
                  "is the resolution of register К.2 (3.6 pp), not chosen now.")


# --- (a) Spearman ----------------------------------------------------------------------------------------------------


def _spearman_test(points: list[tuple[str, float, float]], main: bool) -> dict:
    labs = [p[0] for p in points]
    x = [p[1] for p in points]
    y = [p[2] for p in points]
    r = spearman_exact(x, y, alpha=ALPHA)
    loo = leave_one_out(x, y)
    finite = [v for v in loo if not math.isnan(v)]
    expected = r.rho < 0
    out = {"points": [{"label": a, "x": b, "y": c} for a, b, c in points], "rho": r.rho, "p": r.p, "n": r.n,
           "n_permutations": r.n_permutations, "critical_rho": r.critical_rho, "significant": r.significant,
           "leave_one_out": dict(zip(labs, loo)), "loo_min": min(finite) if finite else None,
           "loo_max": max(finite) if finite else None, "expected_sign": bool(expected), "main": main,
           "significant_wrong_sign": bool(r.significant and r.rho > 0)}
    out["verdict"] = (("supported" if r.significant and expected else "not supported") if main else None)
    return out


def spearman_section(r19: dict, labels: list[str], attack: str = "es") -> dict:
    """The two main tests (md3) and their md1 versions (information) over ``labels``."""
    leg = "adv_" + attack
    missing = [lab for lab in labels if lab not in r19 or leg not in r19[lab]]
    if missing:
        raise ValueError(f"R19 results missing for {missing}")
    tests = {}
    for name, md, ycol, main in (("H_rel", "md3", "rel", True), ("H_abs", "md3", "diff", True),
                                 ("H_rel_md1", "md1", "rel", False), ("H_abs_md1", "md1", "diff", False)):
        pts = [(lab, float(r19[lab][leg][md]["ver_md"]), float(r19[lab][leg][md][ycol])) for lab in labels]
        tests[name] = dict(_spearman_test(pts, main), md=md, y=ycol)
    n = len(labels)
    return {"n": n, "critical_rho": critical_rho(n, ALPHA), "alpha": ALPHA, "leg": leg, "tests": tests,
            "note": DEPENDENCE_NOTE}


def r19_nonpositive(r19: dict, labels: list[str], attack: str = "es") -> dict:
    """The adversarial R19 cells whose relative overestimation is not positive with a CI excluding 0 (rel ≤ 0 or
    lower ≤ 0); a count, no verdict."""
    leg = "adv_" + attack
    cells = []
    for lab in labels:
        for md in ("md1", "md2", "md3"):
            e = r19[lab][leg][md]
            if not (e["rel"] > 0 and e["lower"] > 0):
                cells.append({"label": lab, "md": md, "rel": e["rel"], "lower": e["lower"], "upper": e["upper"]})
    return {"cells": cells, "k": len(cells), "n_cells": 3 * len(labels)}


# --- (b) policy vs model -------------------------------------------------------------------------------------------


def policy_vs_model_8(grid: dict, family_summaries: dict) -> dict:
    """``policy_vs_model`` over the 8 grids (R30 rule) and the family results of the input summaries."""
    eight = ps.policy_vs_model(grid)
    fams = []
    for src, s in family_summaries.items():
        p = s["policy_vs_model"]
        fams.append({"source": src, "labels": list(s["labels"]), "median_policy_span": p["median_policy_span"],
                     "median_model_span": p["median_model_span"], "median_seed_spread": p["median_seed_spread"],
                     "verdict": p["verdict"], "differs_from_eight": p["verdict"] != eight["verdict"]})
    return {"eight": eight, "families": fams}


def _u_by_cell(grid: dict) -> dict:
    by: dict = {}
    for lab, s in grid.items():
        for k, c in ps._index(ps.base_cells(s)).items():
            by.setdefault(k, {})[lab] = float(c["U"])
    return dict(sorted(by.items()))


def family_spans(grid: dict) -> dict:
    """Descriptive split of the model span: within a family (|U_base − U_hard| per cell, median over cells, per family
    and over all cell × family values) and across families (max − min of the four bases / four hardened per cell,
    median over cells)."""
    fams = ps.model_families(list(grid))
    fams = {f: fams[f] for f in dict.fromkeys(lab[:-5] for lab in grid if lab[:-5] in fams)}
    by = {k: v for k, v in _u_by_cell(grid).items() if len(v) == len(grid)}
    within = {f: float(np.median([abs(v[b] - v[h]) for v in by.values()])) for f, (b, h) in fams.items()}
    all_w = [abs(v[b] - v[h]) for v in by.values() for b, h in fams.values()]
    bases = [b for b, _ in fams.values()]
    hards = [h for _, h in fams.values()]
    return {"families": {f: list(v) for f, v in fams.items()}, "within": within,
            "within_all": float(np.median(all_w)) if all_w else None,
            "across_base": float(np.median([max(v[x] for x in bases) - min(v[x] for x in bases)
                                            for v in by.values()])),
            "across_hard": float(np.median([max(v[x] for x in hards) - min(v[x] for x in hards)
                                            for v in by.values()])),
            "n_cells": len(by)}


# --- shared cells ----------------------------------------------------------------------------------------------------


def check_shared(grid: dict) -> None:
    """ValueError unless every grid has the same gate block and own target FPR, and every Task 19 grid cell shared by
    the models has the same targets in every model."""
    labs = list(grid)
    first = grid[labs[0]]
    bad = [lab for lab in labs[1:] if grid[lab]["gate"] != first["gate"]]
    if bad:
        raise ValueError(f"the gate block of {bad} differs from {labs[0]}'s")
    bad = [lab for lab in labs[1:] if grid[lab]["own_target_fpr"] != first["own_target_fpr"]]
    if bad:
        raise ValueError(f"the own target FPR of {bad} differs from {labs[0]}'s")
    by: dict = {}
    for lab, s in grid.items():
        for k, c in ps._index(ps.base_cells(s)).items():
            by.setdefault(k, {})[lab] = json.dumps(c.get("targets"))
    for k, v in sorted(by.items()):
        if len(set(v.values())) != 1:
            raise ValueError(f"the targets of cell {k} differ between the models {sorted(v)}")


def shared_cells_table(grid: dict) -> dict:
    rows = []
    for (f, r, c), v in _u_by_cell(grid).items():
        rows.append({"key": f"fpr={f:g}|{r}|corridor={c}", "U": v,
                     "model_span": float(max(v.values()) - min(v.values())) if len(v) >= 2 else None})
    return {"labels": list(grid), "rows": rows}


# --- (c) Pareto ------------------------------------------------------------------------------------------------------


def first_delta(appetite: list[dict], key: str) -> float | None:
    """The smallest δ′ whose ``key`` entry (``A_without_md1`` / ``A_with_md1``) is not null."""
    ds = [float(e["delta"]) for e in appetite if e.get(key) is not None]
    return min(ds) if ds else None


def _dominates(a: dict, b: dict) -> bool:
    return (a["recall"] >= b["recall"] and a["delta_A"] <= b["delta_A"]
            and (a["recall"] > b["recall"] or a["delta_A"] < b["delta_A"]))


def pareto_front(models: dict, n_fraud: int, resolution: float = RESOLUTION) -> dict:
    """Strict Pareto front over ``{label: {recall, delta_A, delta_A_md1}}`` (recall maximised, δ′ minimised; exact ties
    stay). A model without δ′ (no passing cell up to 0.50) is kept off the comparison."""
    comp = [lab for lab, m in models.items() if m["delta_A"] is not None]
    out, pairs = {}, []
    for lab, m in models.items():
        row = {"recall": m["recall"], "delta_A": m["delta_A"], "delta_A_md1": m.get("delta_A_md1")}
        if m["delta_A"] is None:
            row.update(on_front=None, dominated_by=[], status=NO_PASS)
        else:
            dom = [a for a in comp if a != lab and _dominates(models[a], m)]
            row.update(on_front=not dom, dominated_by=dom, status="on the front" if not dom else "dominated")
        out[lab] = row
    for b in comp:
        for a in out[b]["dominated_by"]:
            dd = models[b]["delta_A"] - models[a]["delta_A"]
            dr = models[a]["recall"] - models[b]["recall"]
            pairs.append({"a": a, "b": b, "d_delta": dd, "d_delta_pp": 100 * dd, "d_recall": dr,
                          "d_recall_pp": 100 * dr, "d_ops": dr * n_fraud,
                          "delta_gap_below_resolution": round(dd, 9) < resolution})
    companion = [lab for lab in comp
                 if not any(p["b"] == lab and not p["delta_gap_below_resolution"] for p in pairs)]
    return {"models": out, "pairs": pairs, "front": [lab for lab in comp if out[lab]["on_front"]],
            "front_ignoring_gaps_below_resolution": companion,
            "no_passing_cell": [lab for lab in models if models[lab]["delta_A"] is None],
            "n_fraud_test": n_fraud, "resolution": resolution, "caveats": PARETO_CAVEATS}


def n_fraud_test(grid: dict, root: Path = Path(".")) -> dict:
    """The number of fraud operations in the test period from every model's ``model.json`` (``meta.test.n_fraud``
    where present, else ``meta.fraud_rows.test``); ValueError unless the file's SHA-256 is the sweep's and the number
    is the same for every model."""
    per = {}
    for lab, s in grid.items():
        p = Path(root) / s["model_dir"] / "model.json"
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != s["model_json_sha256"]:
            raise ValueError(f"{lab}: {p} SHA-256 differs from the sweep's model_json_sha256")
        meta = json.loads(raw).get("meta") or {}
        if (meta.get("test") or {}).get("n_fraud") is not None:
            per[lab] = {"n_fraud": int(meta["test"]["n_fraud"]), "field": "meta.test.n_fraud", "file": str(p)}
        elif (meta.get("fraud_rows") or {}).get("test") is not None:
            per[lab] = {"n_fraud": int(meta["fraud_rows"]["test"]), "field": "meta.fraud_rows.test", "file": str(p)}
        else:
            raise ValueError(f"{lab}: {p} has neither meta.test.n_fraud nor meta.fraud_rows.test")
    vals = {v["n_fraud"] for v in per.values()}
    if len(vals) != 1:
        raise ValueError(f"the test n_fraud differs between the models: { {k: v['n_fraud'] for k, v in per.items()} }")
    return {"n_fraud": vals.pop(), "per_model": per}


def own_cells(grid: dict) -> dict:
    """Per model the own-τ cell (own target FPR, md3, corridor off): key, test recall and test FPR."""
    out = {}
    for lab, s in grid.items():
        k = (round(float(s["own_target_fpr"]), 12), *OWN_REACTION)
        c = ps._index(ps.base_cells(s)).get(k)
        if c is None:
            raise ValueError(f"{lab}: no own-τ cell {k}")
        out[lab] = {"key": c["key"], "recall": float(c["test_recall"]), "fpr": float(c["test_fpr"])}
    return out


# --- (d) cost of hardening ------------------------------------------------------------------------------------------


def cost_of_hardening(own: dict, families: dict, n_fraud: int) -> dict:
    out = {}
    for f, (b, h) in families.items():
        rb, rh, fb, fh = own[b]["recall"], own[h]["recall"], own[b]["fpr"], own[h]["fpr"]
        mb, mh = 1 - rb, 1 - rh
        out[f] = {"base": b, "hardened": h, "recall_base": rb, "recall_hard": rh, "fpr_base": fb, "fpr_hard": fh,
                  "d_recall_pp": 100 * (rh - rb), "d_fpr_pp": 100 * (fh - fb), "missed_base": mb, "missed_hard": mh,
                  "missed_ratio": mh / mb if mb > 0 else None, "missed_ops_base": mb * n_fraud,
                  "missed_ops_hard": mh * n_fraud, "n_fraud_test": n_fraud}
    return out


def check_register(cost: dict, register: dict) -> list[dict]:
    """Compare the cost table with the register (recall at the third decimal, FPR as a fraction at the fourth);
    ValueError on any mismatch (the brief: stop and report)."""
    rows, bad = [], []
    for f, reg in register.items():
        if f not in cost:
            raise ValueError(f"register check: family {f} is not in the cost table")
        c = cost[f]
        vals = {"recall": (c["recall_base"], c["recall_hard"]), "fpr": (c["fpr_base"], c["fpr_hard"])}
        ok = True
        for what, (eb, eh) in (("recall", reg["recall"]), ("fpr", reg["fpr"])):
            for got, exp in zip(vals[what], (eb, eh)):
                d = 3 if what == "recall" else 4
                if round(got, d) != round(exp, d):
                    ok = False
                    bad.append(f"{f} {what}: {got} vs register {exp} (row {reg['row']})")
        rows.append({"family": f, "row": reg["row"], "recall_register": list(reg["recall"]),
                     "fpr_register": list(reg["fpr"]), "recall": list(vals["recall"]), "fpr": list(vals["fpr"]),
                     "ok": ok})
    if bad:
        raise ValueError("the cost table differs from the register: " + "; ".join(bad))
    return rows


# --- inputs ----------------------------------------------------------------------------------------------------------


def load_inputs(summary_paths) -> tuple[dict, dict, list[str], list[str]]:
    """(summaries by path, merged grid sweeps by label, labels in summary order, every git SHA of the summaries'
    sources). The grid files of each label are merged as ``policy_summary.load`` does; confirmation sweeps skipped."""
    summaries, files, labels, shas = {}, [], [], []
    for p in summary_paths:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        summaries[str(p)] = s
        for lab in s["labels"]:
            if lab in labels:
                raise ValueError(f"label {lab!r} is in more than one summary")
            labels.append(lab)
        for lab, src in s["sources"].items():
            shas += [x["git_sha"] for x in src if x.get("git_sha")]
            if lab.endswith(" (confirmation)"):
                continue
            files += [x["file"] for x in src]
    grid, conf = ps.load(files)
    if conf:
        raise ValueError(f"confirmation sweeps among the grid sources: {sorted(conf)}")
    if sorted(grid) != sorted(labels):
        raise ValueError(f"the grid sources give labels {sorted(grid)}, the summaries {sorted(labels)}")
    for path, s in summaries.items():  # the sources are the files the summary was made from
        for lab in s["labels"]:
            want = {x["file"]: x["n_cells"] for x in s["sources"][lab]}
            got = {x["file"]: len(x["cells"]) for x in grid[lab]["sources"]}
            if want != got:
                raise ValueError(f"{path}: {lab} sources {want} differ from the loaded files {got}")
    return summaries, {lab: grid[lab] for lab in labels}, labels, list(dict.fromkeys(shas))


def summarize8(summary_paths, out_dir, resolution: float = RESOLUTION, register: dict | None = REGISTER,
               root: Path = Path("."), meta: dict | None = None) -> dict:
    """The 8-model summary; ``meta`` (code version, platform, versions of this computation) is stored as given."""
    summaries, grid, labels, shas = load_inputs(summary_paths)
    prov = ps.provenances(grid.values())
    ps.check_provenance(prov, shas)
    check_shared(grid)
    r19 = {lab: s["r19"][lab] for s in summaries.values() for lab in s["labels"] if lab in s["r19"]}
    app = {lab: s["appetite"][lab] for s in summaries.values() for lab in s["labels"] if lab in s.get("appetite", {})}
    miss = [lab for lab in labels if lab not in app]
    if miss:
        raise ValueError(f"risk-appetite curves missing for {miss}")
    for path, s in summaries.items():  # the family policy-vs-model must be what the loaded grids give
        mine = ps.policy_vs_model({lab: grid[lab] for lab in s["labels"]})
        for k in ("median_policy_span", "median_model_span", "median_seed_spread"):
            if not math.isclose(mine[k], s["policy_vs_model"][k], rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"{path}: policy_vs_model {k} {s['policy_vs_model'][k]} differs from the loaded "
                                 f"grids ({mine[k]})")
    nf = n_fraud_test(grid, root)
    own = own_cells(grid)
    fams = ps.model_families(labels)  # sorted by name; reported in the order of the labels
    families = {f: fams[f] for f in dict.fromkeys(lab[:-5] for lab in labels if lab[:-5] in fams)}
    models = {lab: {"recall": own[lab]["recall"], "delta_A": first_delta(app[lab], "A_without_md1"),
                    "delta_A_md1": first_delta(app[lab], "A_with_md1")} for lab in labels}
    cost = cost_of_hardening(own, families, nf["n_fraud"])
    res = {"labels": labels, "provenances": prov, "commits": sorted({p["git_sha"] for p in prov}),
           "machines": sorted({p["host_machine"] for p in prov}), "inputs": list(summaries), "resolution": resolution,
           "spearman": spearman_section(r19, labels), "r19_nonpositive": r19_nonpositive(r19, labels),
           "policy_vs_model": policy_vs_model_8(grid, summaries), "family_spans": family_spans(grid),
           "own_cells": own, "n_fraud_test": nf, "pareto": pareto_front(models, nf["n_fraud"], resolution),
           "families": {f: list(v) for f, v in families.items()}, "cost": cost,
           "register_check": None if register is None else check_register(cost, register),
           "shared_cells": shared_cells_table(grid), "r19": r19, "meta": meta}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary8.json", res)
    (out / "summary8.md").write_text(markdown8(res), encoding="utf-8")
    for lang in ("en", "uk"):
        pareto_plot(res, out / f"pareto-8-{lang}.png", lang)
    return res


# --- markdown ----------------------------------------------------------------------------------------------------------


def _p1(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f} %"


def _pp(x) -> str:
    return "–" if x is None else f"{100 * x:+.1f} pp"


def _f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def _r4(x) -> str:
    return "–" if x is None else f"{x:.4f}"


def _dp(x) -> str:
    return NO_PASS if x is None else f"{x:.2f}"


TEST_TEXT = {"H_rel": "H_rel (main): VER_md vs relative overestimation, md3",
             "H_abs": "H_abs (main, second test): VER_md vs absolute overestimation (pp), md3",
             "H_rel_md1": "H_rel at md1: VER_md vs relative overestimation",
             "H_abs_md1": "H_abs at md1: VER_md vs absolute overestimation (pp)"}


def spearman_markdown(sp: dict) -> str:
    lines = [f"Unit = model, n = {sp['n']}; threshold |ρ| ≥ {sp['critical_rho']:.3f} for p < {sp['alpha']:g} "
             "(critical_rho(n), untied data); exact two-sided permutation p. Adversarial ES leg, own τ, corridor off "
             "(R19 values of the input summaries). Two main tests (H_rel, H_abs), no multiplicity correction.", "",
             f"Note: {sp['note']}", ""]
    for name, t in sp["tests"].items():
        ytext = "rel" if t["y"] == "rel" else "diff, pp"
        lines += [f"### {TEST_TEXT[name]}", "",
                  f"ρ = {t['rho']:.3f}, p = {t['p']:.3f}, n = {t['n']}, permutations = {t['n_permutations']}; "
                  f"expected sign (negative): {'yes' if t['expected_sign'] else 'no'}; leave-one-model-out ρ "
                  f"{_f3(t['loo_min'])} … {_f3(t['loo_max'])}.", ""]
        if t["main"]:
            v = f"**{t['verdict']}**"
            if t["significant_wrong_sign"]:
                v += " – significant with the wrong sign (ρ > 0)"
            lines.append(f"Verdict: {v} (pre-registered rule: supported iff p < 0.05 and ρ < 0).")
        else:
            lines.append("md1: information, no verdict.")
        lines += ["", f"| model | VER_md | y ({ytext}) | ρ without this model |", "|---|---:|---:|---:|"]
        for pt in t["points"]:
            y = _p1(pt["y"]) if t["y"] == "rel" else f"{100 * pt['y']:.1f}"
            lines.append(f"| {pt['label']} | {_p1(pt['x'])} | {y} | {_f3(t['leave_one_out'][pt['label']])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def pareto_markdown(pa: dict) -> str:
    lines = ["Axes: test recall at the own τ (cell `fpr=0.01|md3|corridor=off`, maximised) × the first δ′ of curve (A) "
             "without md1 (minimised). Strict dominance, exact ties kept. "
             f"n_fraud_test = {pa['n_fraud_test']}; resolution R = {100 * pa['resolution']:.1f} pp (register К.2).", "",
             "| model | recall | first δ′ (A, without md1) | first δ′ (A, with md1; information) | on front | "
             "dominated by |", "|---|---:|---:|---:|---|---|"]
    for lab, m in pa["models"].items():
        on = "–" if m["on_front"] is None else ("yes" if m["on_front"] else "no")
        lines.append(f"| {lab} | {_r4(m['recall'])} | {_dp(m['delta_A'])} | {_dp(m['delta_A_md1'])} | {on} | "
                     f"{', '.join(m['dominated_by']) or '–'} |")
    lines += ["", f"Front: {', '.join(pa['front']) or '–'}.", ""]
    if pa["no_passing_cell"]:
        lines += [f"Kept off the front comparison ({NO_PASS}): {', '.join(pa['no_passing_cell'])}.", ""]
    lines += ["| dominating | dominated | Δδ′, pp | Δrecall, pp | Δ test fraud operations | Δδ′ < R |",
              "|---|---|---:|---:|---:|---|"]
    for p in pa["pairs"]:
        lines.append(f"| {p['a']} | {p['b']} | {p['d_delta_pp']:.1f} | {p['d_recall_pp']:.1f} | "
                     f"{p['d_ops']:.0f} | {'yes' if p['delta_gap_below_resolution'] else 'no'} |")
    if not pa["pairs"]:
        lines.append("| – | – | – | – | – | – |")
    lines += ["", "Descriptive (no verdict) – front when δ′ gaps below the resolution are not counted (models not "
              f"dominated by any model with Δδ′ ≥ R): {', '.join(pa['front_ignoring_gaps_below_resolution']) or '–'}.",
              "", f"Caveats: {pa['caveats']}", ""]
    return "\n".join(lines) + "\n"


def markdown8(res: dict) -> str:
    prov = "; ".join(f"git `{p['git_sha']}` (dirty: {p['git_dirty']}), machine {p['host_machine']}"
                     for p in res["provenances"])
    lines = ["# Summary over the 8 models (Task 25)", "",
             f"Provenance: {len(res['commits'])} commits, {len(res['machines'])} machine – {prov}.",
             f"Inputs: {', '.join(f'`{x}`' for x in res['inputs'])}. Models: {', '.join(res['labels'])}."]
    if res.get("meta"):
        m = res["meta"]
        lines.append(f"This summary: code git `{m.get('git_sha')}` (dirty: {m.get('git_dirty')}), "
                     f"{m.get('platform', '–')}.")
    lines.append("")
    # (a)
    sp = res["spearman"]
    nz = res["r19_nonpositive"]
    lines += ["## (a) Spearman: VER_md vs the overestimation without the bank's reaction (R33, R37, R38)", "",
              spearman_markdown(sp),
              f"### R19 adversarial cells without a positive relative overestimation (CI excluding 0)", "",
              f"{nz['k']} of {nz['n_cells']} adversarial cells (count, no verdict):", ""]
    lines += [f"- {c['label']} {c['md']}: {_p1(c['rel'])} [{_p1(c['lower'])}; {_p1(c['upper'])}]" for c in nz["cells"]]
    lines.append("")
    # (b)
    pv = res["policy_vs_model"]
    e = pv["eight"]
    fs = res["family_spans"]
    lines += ["## (b) Policy vs model on 8 models (R30 rule, unchanged)", "",
              f"Median policy span {_p1(e['median_policy_span'])}, median model span {_p1(e['median_model_span'])} "
              f"({len(e['model_span'])} shared cells, max − min over all 8 models), median seed spread "
              f"{_p1(e['median_seed_spread'])} ({e['n_seed_spreads']} spreads).", "",
              f"Verdict (8 models): **{e['verdict']}** (pre-registered rule: supported iff the median policy span > "
              "the median model span and > the median seed spread).", "",
              "| input (family) | models | median policy span | median model span | median seed spread | verdict |",
              "|---|---|---:|---:|---:|---|"]
    for f in pv["families"]:
        lines.append(f"| `{f['source']}` | {', '.join(f['labels'])} | {_p1(f['median_policy_span'])} | "
                     f"{_p1(f['median_model_span'])} | {_p1(f['median_seed_spread'])} | {f['verdict']} |")
    diff = [f for f in pv["families"] if f["differs_from_eight"]]
    lines.append("")
    if diff:
        lines += [f"The 8-model verdict ({e['verdict']}) differs from the family verdict of "
                  + ", ".join(f"{', '.join(f['labels'])} ({f['verdict']})" for f in diff)
                  + "; both are printed, the rule is not changed.", ""]
    lines += ["Policy span per model: " + ", ".join(f"{k} {_p1(v)}" for k, v in e["policy_span"].items()) + ".", "",
              f"Descriptive (no verdict), model span split over {fs['n_cells']} cells: within families (median over "
              "cells of |U_base − U_hard|) " + ", ".join(f"{FAMILY_NAMES.get(f, f)} {_p1(v)}"
                                                         for f, v in fs["within"].items())
              + f", all families together {_p1(fs['within_all'])}; across families (median over cells of max − min) "
              f"the four bases {_p1(fs['across_base'])}, the four hardened {_p1(fs['across_hard'])}.", ""]
    # (c)
    lines += ["## (c) Strict Pareto front of the 8 models (R41 (5))", "", pareto_markdown(res["pareto"])]
    # (d)
    nfr = res["n_fraud_test"]["n_fraud"]
    lines += ["## (d) Cost of hardening (R41 (5)), descriptive", "",
              f"Cell `fpr=0.01|md3|corridor=off` (own τ), test period, n_fraud_test = {nfr}. FPR with two decimals "
              "(the register's precision).", "",
              "| family | recall base → hard | Δrecall, pp | FPR base → hard | ΔFPR, pp | missed share base → hard | "
              "missed_hard / missed_base | missed test fraud operations base → hard |",
              "|---|---|---:|---|---:|---|---:|---|"]
    for f, c in res["cost"].items():
        lines.append(f"| {FAMILY_NAMES.get(f, f)} | {c['recall_base']:.4f} → {c['recall_hard']:.4f} | "
                     f"{c['d_recall_pp']:.1f} | {100 * c['fpr_base']:.2f} % → {100 * c['fpr_hard']:.2f} % | "
                     f"{c['d_fpr_pp']:.2f} | {_p1(c['missed_base'])} → {_p1(c['missed_hard'])} | "
                     f"{'–' if c['missed_ratio'] is None else format(c['missed_ratio'], '.2f')} | "
                     f"{c['missed_ops_base']:.0f} → {c['missed_ops_hard']:.0f} of {nfr} |")
    lines.append("")
    if res["register_check"] is not None:
        lines += ["Register check (recall at the third decimal, FPR as a fraction at the fourth decimal):", ""]
        for r in res["register_check"]:
            lines.append(f"- {FAMILY_NAMES.get(r['family'], r['family'])} (row {r['row']}): recall "
                         f"{r['recall'][0]:.4f} → {r['recall'][1]:.4f} vs register {r['recall_register'][0]:g} → "
                         f"{r['recall_register'][1]:g}; FPR {100 * r['fpr'][0]:.2f} % → {100 * r['fpr'][1]:.2f} % "
                         f"vs register {100 * r['fpr_register'][0]:.2f} % → {100 * r['fpr_register'][1]:.2f} % – "
                         f"{'agrees' if r['ok'] else 'DIFFERS'}")
        lines.append("")
    # (e)
    sc = res["shared_cells"]
    labs = sc["labels"]
    lines += ["## (e) Shared cells and R19 over 8 models", "", "### U (% of A) in the 18 Task 19 grid cells", "",
              "| cell | " + " | ".join(labs) + " | model span |", "|---|" + "---:|" * (len(labs) + 1)]
    for r in sc["rows"]:
        lines.append(f"| `{r['key']}` | " + " | ".join(_p1(r["U"].get(lab)) for lab in labs)
                     + f" | {_p1(r['model_span'])} |")
    for leg, title in (("adv_es", "adversarial leg (ES runs)"), ("clean", "clean leg (real operations)")):
        lines += ["", f"### R19, {title}: own τ, corridor off", "",
                  "| model | max_declines | VER_∞ | VER_md | rel [95 % CI] | diff, pp [95 % CI] |",
                  "|---|---:|---:|---:|---|---|"]
        for lab in labs:
            for md in ("md1", "md2", "md3"):
                x = res["r19"][lab][leg][md]
                lines.append(f"| {lab} | {md[2:]} | {_p1(x['ver_inf'])} | {_p1(x['ver_md'])} | {_p1(x['rel'])} "
                             f"[{_p1(x['lower'])}; {_p1(x['upper'])}] | {100 * x['diff']:.1f} "
                             f"[{100 * x['diff_lower']:.1f}; {100 * x['diff_upper']:.1f}] |")
    return "\n".join(lines) + "\n"


# --- figure ------------------------------------------------------------------------------------------------------------


TEXT = {
    "en": {"x": "first δ′ of curve (A), without md1 (%)", "y": "test recall at the own τ",
           "base": "base", "hard": "hardened", "front": "Pareto front",
           "band": "δ′ resolution ±{r} pp (R = {R} pp, К.2)", "nopass": "not shown ({x}): no passing cell up to 0.50"},
    "uk": {"x": "перше δ′ кривої (A), без md1 (%)", "y": "recall на тесті при власному τ",
           "base": "базова", "hard": "донавчена", "front": "фронт Парето",
           "band": "роздільна здатність δ′ ±{r} п. п. (R = {R} п. п., К.2)",
           "nopass": "не показано ({x}): немає прохідної клітинки до 0,50"},
}
FAMILY_COLOURS = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a4fd1", "#c23b6b"]


def _num(x: float, lang: str, d: int = 1) -> str:
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if lang == "uk" else s


def pareto_plot(res: dict, path: Path, lang: str = "en") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    t = TEXT[lang]
    pa = res["pareto"]
    fams = res["families"]
    fam_of = {lab: f for f, pair in fams.items() for lab in pair}
    order = list(dict.fromkeys(fam_of[lab] for lab in res["labels"] if lab in fam_of))
    col = {f: FAMILY_COLOURS[i % len(FAMILY_COLOURS)] for i, f in enumerate(order)}
    half = 100 * res["resolution"] / 2
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for lab in res["labels"]:
        m = pa["models"][lab]
        if m["delta_A"] is None:
            continue
        f = fam_of.get(lab)
        c = col.get(f, "#555555")
        hard = lab.endswith("-hard")
        name = f"{FAMILY_NAMES.get(f, f)} {t['hard'] if hard else t['base']}" if f else lab
        x, y = 100 * m["delta_A"], m["recall"]
        ax.errorbar([x], [y], xerr=[[half], [half]], fmt="none", ecolor=c, alpha=0.35, lw=6, capsize=0)
        ax.scatter([x], [y], s=70, marker="o", facecolors="white" if hard else c, edgecolors=c, lw=1.6, zorder=3,
                   label=name)
    front = sorted((100 * pa["models"][lab]["delta_A"], pa["models"][lab]["recall"]) for lab in pa["front"])
    if front:
        ax.plot(*zip(*front), color="#333333", lw=1.0, ls="--", zorder=2, label=t["front"])
    ax.plot([], [], color="#999999", alpha=0.35, lw=6,
            label=t["band"].format(r=_num(half, lang), R=_num(2 * half, lang)))
    if pa["no_passing_cell"]:
        ax.text(0.99, 0.01, t["nopass"].format(x=", ".join(pa["no_passing_cell"])), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#555555")
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, lang, 0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, lang, 3)))
    ax.grid(color="#e5e5e5", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
