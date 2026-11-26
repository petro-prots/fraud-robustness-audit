"""Task 27c (R33, R35, R39 order 4, R50): m2-hard md3 vs md3+esc on 12 fresh seeds – descriptive, no verdict.

The 19b pair whose escalation effect was within the seed spread (m2-hard, own τ, corridor κ 2 q 0.99 = the corridor
file's), rerun on seeds 4–15 with all three attacks (``gate-seeds-4-15.yaml``, role ``confirmation``). Per seed s and
reaction: ``U_s`` = max over the 3 attacks of the run upper bounds at seed s (the definition of R49 (1),
``policy_summary.confirmation_stats``) and the attack giving it (the first in sorted order on a tie); the paired
difference ``d_s = U_s(b) − U_s(a)`` (b = md3+esc, a = md3). Distributions: median, P25, P75 (numpy linear
interpolation), min, max of U_s per reaction and of d_s, plus the counts d_s < 0, = 0, > 0. The gate seeds 1–3 of the
same keys (the grid file) are printed beside, not pooled (other seeds).

Checks (``ValueError``): both cells in the sweep; the same model (``model_json_sha256``) in the sweep and the reference;
each cell has the setup of the reference cell of the same key (``md_study.check_same_setup``: τ, max_declines,
escalation threshold, corridor parameters – the sweep's for the Task 19 records without per-cell ones –, target FPR,
threat files, budget where recorded); the two cells have the same τ and corridor parameters; each cell has exactly
3 attacks × the 12 seeds.

    fraudaudit reaction-pairs --sweep S --a KEY --b KEY --reference GRID --out DIR
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .md_study import _cell, _load, check_same_setup, effective, provenance
from .policy_summary import confirmation_stats

SEEDS = tuple(range(4, 16))
ATTACKS = ("es", "random", "stepwise")


def distribution(xs) -> dict:
    a = np.asarray(list(xs), dtype=np.float64)
    if a.size == 0:
        raise ValueError("empty distribution")
    p25, med, p75 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "median": float(med), "p25": float(p25), "p75": float(p75), "min": float(a.min()),
            "max": float(a.max())}


def u_per_seed(cell: dict, attacks=ATTACKS, seeds=None) -> dict:
    """{seed: {"U_s", "attack"}} via ``confirmation_stats`` (U_s = max over the attacks); ``seeds`` = the exact seed
    set required (None: whatever the cell has, every seed with every attack)."""
    st = confirmation_stats(cell, float(cell.get("U", 0.0)), list(attacks), seeds)
    out = {}
    for sd, u in st["U_s"].items():
        ups = {a: st["run_upper"][f"{a}/{sd}"] for a in sorted(attacks)}
        out[sd] = {"U_s": float(u), "attack": max(ups, key=ups.get)}
    return out


def pair_stats(ua: dict, ub: dict) -> dict:
    if sorted(ua, key=int) != sorted(ub, key=int):
        raise ValueError(f"the two cells have different seeds: {sorted(ua, key=int)} vs {sorted(ub, key=int)}")
    d = {sd: ub[sd]["U_s"] - ua[sd]["U_s"] for sd in sorted(ua, key=int)}
    vals = list(d.values())
    return {"d_s": d, "distribution": distribution(vals), "n_negative": int(sum(v < 0 for v in vals)),
            "n_zero": int(sum(v == 0 for v in vals)), "n_positive": int(sum(v > 0 for v in vals))}


def reaction_pairs(sweep_p, key_a: str, key_b: str, reference_p, seeds=SEEDS, attacks=ATTACKS) -> dict:
    s, g = _load(sweep_p), _load(reference_p)
    if s.get("model_json_sha256") != g.get("model_json_sha256"):
        raise ValueError(f"model_json_sha256 of {sweep_p} differs from the reference {reference_p}")
    a, b = _cell(s, key_a, sweep_p), _cell(s, key_b, sweep_p)
    ga, gb = _cell(g, key_a, reference_p), _cell(g, key_b, reference_p)
    for c, gc in ((a, ga), (b, gb)):
        check_same_setup(c, s, gc, g, c["key"])
    ea, eb = effective(a, s), effective(b, s)
    for k in ("tau", "corridor_params"):
        if ea[k] != eb[k]:
            raise ValueError(f"{k} of {key_a} and {key_b} differ")
    ua, ub = u_per_seed(a, attacks, seeds), u_per_seed(b, attacks, seeds)
    ra, rb = u_per_seed(ga, attacks), u_per_seed(gb, attacks)
    return {"a": key_a, "b": key_b, "label": s.get("label"), "tau": a["tau"], "seeds": [int(x) for x in seeds],
            "attacks": list(attacks),
            "cells": {"a": {"U_s": ua, "distribution": distribution(v["U_s"] for v in ua.values())},
                      "b": {"U_s": ub, "distribution": distribution(v["U_s"] for v in ub.values())}},
            "pairs": pair_stats(ua, ub),
            "reference": {"seeds": sorted(int(x) for x in ra), "a": ra, "b": rb, "pairs": pair_stats(ra, rb),
                          "U_a": float(ga["U"]), "U_b": float(gb["U"])},
            "inputs": {"sweep": provenance(sweep_p, s), "reference": provenance(reference_p, g)}}


def _pct(x) -> str:
    return f"{100 * x:.1f} %"


def _dist(d: dict) -> str:
    return (f"{_pct(d['median'])} | {_pct(d['p25'])} | {_pct(d['p75'])} | {_pct(d['min'])} | {_pct(d['max'])}")


def markdown(res: dict) -> str:
    a, b = res["a"], res["b"]
    lines = [f"# md3 vs md3+esc on fresh seeds (Task 27c): {res['label']}", "",
             f"a = `{a}`, b = `{b}`; seeds {res['seeds'][0]}–{res['seeds'][-1]}, attacks {', '.join(res['attacks'])}. "
             "U_s = max over the attacks of the one-sided upper bound of the uplift at seed s; d_s = U_s(b) − U_s(a), "
             "in pp. Seeds 1–3 (the grid) beside, not pooled. Descriptive: no verdict.", "",
             "| seed | U_s(a) (attack) | U_s(b) (attack) | d_s, pp |", "|---:|---:|---:|---:|"]
    for sd, d in res["pairs"]["d_s"].items():
        ua, ub = res["cells"]["a"]["U_s"][sd], res["cells"]["b"]["U_s"][sd]
        lines.append(f"| {sd} | {_pct(ua['U_s'])} ({ua['attack']}) | {_pct(ub['U_s'])} ({ub['attack']}) | "
                     f"{100 * d:.1f} |")
    r = res["reference"]
    for sd, d in r["pairs"]["d_s"].items():
        lines.append(f"| grid {sd} | {_pct(r['a'][sd]['U_s'])} ({r['a'][sd]['attack']}) | "
                     f"{_pct(r['b'][sd]['U_s'])} ({r['b'][sd]['attack']}) | {100 * d:.1f} |")
    p = res["pairs"]
    lines += ["", "| distribution | median | P25 | P75 | min | max |", "|---|---:|---:|---:|---:|---:|",
              f"| U_s(a) | {_dist(res['cells']['a']['distribution'])} |",
              f"| U_s(b) | {_dist(res['cells']['b']['distribution'])} |",
              f"| d_s | {_dist(p['distribution'])} |",
              f"| d_s, grid seeds | {_dist(r['pairs']['distribution'])} |", "",
              f"d_s < 0: {p['n_negative']}, d_s = 0: {p['n_zero']}, d_s > 0: {p['n_positive']} (of "
              f"{p['distribution']['n']} seeds). Grid U (seeds 1–3): a {_pct(r['U_a'])}, b {_pct(r['U_b'])}.", "",
              f"Inputs: sweep `{res['inputs']['sweep']['path']}` ({res['inputs']['sweep']['git_sha']}), reference "
              f"`{res['inputs']['reference']['path']}` ({res['inputs']['reference']['git_sha']}); model "
              f"`{res['inputs']['sweep']['model_json_sha256']}`.", ""]
    return "\n".join(lines)


def write(res: dict, out_dir) -> None:
    from .sweep_cells import write_json
    out = Path(out_dir)
    write_json(out / "reaction-pairs.json", res)
    (out / "reaction-pairs.md").write_text(markdown(res), encoding="utf-8")

