"""Sensitivity of the gate's verdict to the attacker's reconnaissance budget (R33, Task 19c).

The main numbers use ``recon_budget_ops: 2000``; the hardening generation used 20 000 per unit. This compares U (the
worst one-sided upper bound of the paired uplift over attack x seed) of the same cells at budgets 2 000, 8 000 and
20 000. Nothing else changes (delta, thresholds, policy, threat model, seeds).

The reference is a budget-2000 run at the SAME commit and machine as the others (``out/t19c/b2000``); the Task 19b
cells (earlier commit) are only a cross-check. Pre-registered rule (fixed before the run, R33): hypothesis H – the losses grow by less than the spread over seeds
(``SEED_SPREAD`` = 9.6 pp, the median spread over seeds of the grid, Task 19 ``hypotheses.md``). Per cell
``dU(B) = U(B) - U(2000)`` in pp. H is **supported** when ``dU(20 000) < 9.6`` pp in every cell, **not supported**
when it is >= 9.6 pp in at least one cell; ``dU(8 000)`` is descriptive (no verdict). The comparison is descriptive: U is
a maximum over 9 runs, so a rise below the spread cannot be told from noise and is not claimed to be zero.

    fraudaudit budget-sensitivity --reference out/t19b --budget 8000=out/t19c/b8000 --budget 20000=out/t19c/b20000 \
        --labels m3-hard,m2-hard --out out/t19c/summary
"""
from __future__ import annotations

import json
from pathlib import Path

SEED_SPREAD = 0.096
REFERENCE_BUDGET = 2000
HYPOTHESIS_BUDGET = 20000


def _cells(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {c["key"]: c for c in d["cells"]}, d


def _same(a, b, what: str) -> None:
    if a != b:
        raise ValueError(f"{what} differs between the reference and a budget run: {a!r} vs {b!r}")


def compare(reference: Path, budgets: dict[int, Path], labels: list[str]) -> dict:
    """``{label: {cell key: {budget: {U, U_usd, worst_run, runs, dU}}}}`` plus the verdict on budget 20 000.
    ``reference`` holds ``<label>/policy-sweep.json`` at budget 2 000, ``budgets[B]`` at budget B. Every cell carries
    its own ``recon_budget_ops`` (checked); model, gate, seed, τ, threat and corridor parameters of each common cell must
    be equal to the reference's (only the budget may differ). 20 000 must be among the budgets."""
    if HYPOTHESIS_BUDGET not in budgets:
        raise ValueError(f"budget {HYPOTHESIS_BUDGET} is required: H is decided on it")
    out: dict = {"seed_spread": SEED_SPREAD, "reference_budget": REFERENCE_BUDGET, "budgets": sorted(budgets),
                 "labels": {}}
    dus = []
    for label in labels:
        ref, ref_d = _cells(reference / label / "policy-sweep.json")
        per_b = {}
        for b, p in budgets.items():
            cells, d = _cells(p / label / "policy-sweep.json")
            for what in ("model_json_sha256", "gate", "seed", "own_tau"):
                _same(ref_d.get(what), d.get(what), f"{label}: {what}")
            for what in ("threat_sha256", "corridor_threat_sha256"):
                _same(ref_d["meta"].get(what), d["meta"].get(what), f"{label}: meta.{what}")
            per_b[b] = cells
        common = [k for k in ref if all(k in c for c in per_b.values())]
        if not common:
            raise ValueError(f"{label}: no cell present at every budget")
        out["labels"][label] = {}
        for k in common:
            if ref[k].get("recon_budget_ops") != REFERENCE_BUDGET:
                raise ValueError(f"{label} {k}: reference cell budget {ref[k].get('recon_budget_ops')} != {REFERENCE_BUDGET}")
            row = {REFERENCE_BUDGET: _brief(ref[k])}
            for b, cells in per_b.items():
                if cells[k].get("recon_budget_ops") != b:
                    raise ValueError(f"{label} {k}: cell budget {cells[k].get('recon_budget_ops')} != {b}")
                for what in ("tau", "max_declines", "tau_after_decline", "corridor_params", "target_fpr"):
                    _same(ref[k].get(what), cells[k].get(what), f"{label} {k}: {what}")
                row[b] = _brief(cells[k])
                row[b]["dU"] = row[b]["U"] - row[REFERENCE_BUDGET]["U"]
            dus.append(row[HYPOTHESIS_BUDGET]["dU"])
            out["labels"][label][k] = row
    out["verdict"] = {"budget": HYPOTHESIS_BUDGET, "max_dU": max(dus), "cells": len(dus),
                      "H": "supported" if max(dus) < SEED_SPREAD else "not supported"}
    return out


def _brief(c: dict) -> dict:
    runs = {k: v["uplift_upper"] for k, v in c["runs"].items()}
    return {"U": float(c["U"]), "U_usd": float(c["U_usd"]), "worst_run": c["worst_run"], "runs": runs}


def markdown(res: dict) -> str:
    pp = lambda x: f"{100 * x:.1f}"  # noqa: E731
    lines = ["# Budget sensitivity of the gate (R33)", "",
             f"U = worst one-sided upper bound of the paired uplift over attacks x seeds 1-3, in % of the episode amount; "
             f"dU in pp against budget {res['reference_budget']}. Seed spread {pp(res['seed_spread'])} pp.", ""]
    bs = [res["reference_budget"], *res["budgets"]]
    head = "| model | cell | " + " | ".join(f"U {b} (worst run)" for b in bs) + " | " + " | ".join(
        f"dU {b}" for b in res["budgets"]) + " |"
    lines += [head, "|---|---|" + "---:|" * (len(bs) + len(res["budgets"]))]
    for label, cells in res["labels"].items():
        for k, row in cells.items():
            lines.append(f"| {label} | `{k}` | " + " | ".join(f"{pp(row[b]['U'])} ({row[b]['worst_run']})" for b in bs)
                         + " | " + " | ".join(f"{100 * row[b]['dU']:+.1f}" for b in res["budgets"]) + " |")
    lines += ["", "## Hypothesis H (dU(B) < seed spread in every cell)", ""]
    v = res["verdict"]
    lines.append(f"- budget {v['budget']}: max dU = {100 * v['max_dU']:+.1f} pp over {v['cells']} cells -> **{v['H']}**")
    lines.append("- the other budgets are descriptive (no verdict).")
    lines += ["", "Per attack (upper bound of the worst seed, %):", "", "| model | cell | budget | es | random | stepwise |",
              "|---|---|---:|---:|---:|---:|"]
    for label, cells in res["labels"].items():
        for k, row in cells.items():
            for b in bs:
                w = {a: max((v for r, v in row[b]["runs"].items() if r.startswith(a + "/")), default=None)
                     for a in ("es", "random", "stepwise")}
                f = lambda x: "–" if x is None else pp(x)  # noqa: E731
                lines.append(f"| {label} | `{k}` | {b} | {f(w['es'])} | {f(w['random'])} | {f(w['stepwise'])} |")
    return "\n".join(lines) + "\n"
