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

Task 27f (R46): the same rule on budget 50 000 (H50, per family, seeds 1-3) – ``hypothesis_budget=50000``. Task 27e
(R45 (4), R46): the GRU cells on seeds 4-9, descriptive – ``hypothesis_budget=None`` (no verdict). Always also per seed
(descriptive): ``U_s`` = max over the attacks of the run upper bounds of seed s, ``dU_s = U_s(B) - U_s(2000)`` and the
seeds with ``dU_s >= SEED_SPREAD`` (with the attack that gives ``U_s(B)``) – the R45 (5) counter.
"""
from __future__ import annotations

import json
from pathlib import Path

SEED_SPREAD = 0.096
EPS = 1e-9  # a difference equal to the spread up to float noise counts as reaching it (">= 9.6 pp")
REFERENCE_BUDGET = 2000
HYPOTHESIS_BUDGET = 20000


def _cells(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {c["key"]: c for c in d["cells"]}, d


def _same(a, b, what: str) -> None:
    if a != b:
        raise ValueError(f"{what} differs between the reference and a budget run: {a!r} vs {b!r}")


def _by_seed(runs: dict) -> dict:
    """``{seed: (U_s, attack/seed of the max)}`` – U_s = max over the attacks of seed s (first run on ties)."""
    out: dict = {}
    for rk in sorted(runs, key=lambda k: (int(k.rsplit("/", 1)[1]), k)):
        sd = int(rk.rsplit("/", 1)[1])
        if sd not in out or runs[rk] > out[sd][0]:
            out[sd] = (runs[rk], rk)
    return out


def _per_seed(row: dict, b: int) -> None:
    ref = _by_seed(row[REFERENCE_BUDGET]["runs"])
    cur = _by_seed(row[b]["runs"])
    if sorted(ref) != sorted(cur):
        raise ValueError(f"seeds differ between the reference {sorted(ref)} and budget {b} {sorted(cur)}")
    row[REFERENCE_BUDGET]["U_s"] = {str(sd): u for sd, (u, _) in ref.items()}
    row[b]["U_s"] = {str(sd): u for sd, (u, _) in cur.items()}
    row[b]["dU_s"] = {str(sd): cur[sd][0] - ref[sd][0] for sd in cur}
    hits = [{"seed": sd, "dU": cur[sd][0] - ref[sd][0], "attack": cur[sd][1].rsplit("/", 1)[0]}
            for sd in sorted(cur) if cur[sd][0] - ref[sd][0] >= SEED_SPREAD - EPS]
    row[b]["seeds_dU_ge_spread"] = hits
    row[b]["n_seeds_dU_ge_spread"] = len(hits)
    row[b]["n_seeds"] = len(cur)


def compare(reference: Path, budgets: dict[int, Path], labels: list[str],
            hypothesis_budget: int | None = HYPOTHESIS_BUDGET) -> dict:
    """``{label: {cell key: {budget: {U, U_usd, worst_run, runs, dU}}}}`` plus the verdict on budget 20 000.
    ``reference`` holds ``<label>/policy-sweep.json`` at budget 2 000, ``budgets[B]`` at budget B. Every cell carries
    its own ``recon_budget_ops`` (checked); model, gate, seed, τ, threat and corridor parameters of each common cell must
    be equal to the reference's (only the budget may differ). ``hypothesis_budget`` (default 20 000) must be among the
    budgets; ``None`` = descriptive, no verdict."""
    if hypothesis_budget is not None and hypothesis_budget not in budgets:
        raise ValueError(f"budget {hypothesis_budget} is required: H is decided on it")
    out: dict = {"seed_spread": SEED_SPREAD, "reference_budget": REFERENCE_BUDGET, "budgets": sorted(budgets),
                 "hypothesis_budget": hypothesis_budget, "labels": {}}
    seeds = set()
    dus = []
    for label in labels:
        ref, ref_d = _cells(reference / label / "policy-sweep.json")
        seeds.add(tuple((ref_d.get("gate") or {}).get("attack_seeds") or ()))
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
                _per_seed(row, b)
            if hypothesis_budget is not None:
                dus.append(row[hypothesis_budget]["dU"])
            out["labels"][label][k] = row
    out["attack_seeds"] = sorted(seeds)[0] if len(seeds) == 1 else None
    out["verdict"] = None if hypothesis_budget is None else {
        "budget": hypothesis_budget, "max_dU": max(dus), "cells": len(dus),
        "H": "supported" if max(dus) < SEED_SPREAD - EPS else "not supported"}
    return out


def _seeds_text(seeds) -> str:
    if not seeds:
        return "the gate seeds"
    s = sorted(seeds)
    if s == list(range(s[0], s[-1] + 1)) and len(s) > 1:
        return f"seeds {s[0]}-{s[-1]}"
    return "seeds " + ", ".join(map(str, s))


def _brief(c: dict) -> dict:
    runs = {k: v["uplift_upper"] for k, v in c["runs"].items()}
    return {"U": float(c["U"]), "U_usd": float(c["U_usd"]), "worst_run": c["worst_run"], "runs": runs}


def markdown(res: dict) -> str:
    pp = lambda x: f"{100 * x:.1f}"  # noqa: E731
    lines = ["# Budget sensitivity of the gate (R33)", "",
             f"U = worst one-sided upper bound of the paired uplift over attacks x {_seeds_text(res.get('attack_seeds', (1, 2, 3)))}, in % of the episode amount; "
             f"dU in pp against budget {res['reference_budget']}. Seed spread {pp(res['seed_spread'])} pp.", ""]
    bs = [res["reference_budget"], *res["budgets"]]
    head = "| model | cell | " + " | ".join(f"U {b} (worst run)" for b in bs) + " | " + " | ".join(
        f"dU {b}" for b in res["budgets"]) + " |"
    lines += [head, "|---|---|" + "---:|" * (len(bs) + len(res["budgets"]))]
    for label, cells in res["labels"].items():
        for k, row in cells.items():
            lines.append(f"| {label} | `{k}` | " + " | ".join(f"{pp(row[b]['U'])} ({row[b]['worst_run']})" for b in bs)
                         + " | " + " | ".join(f"{100 * row[b]['dU']:+.1f}" for b in res["budgets"]) + " |")
    v = res["verdict"]
    if v is None:
        lines += ["", "Descriptive only: no hypothesis and no verdict on these runs.", ""]
    else:
        lines += ["", "## Hypothesis H (dU(B) < seed spread in every cell)", ""]
        lines.append(f"- budget {v['budget']}: max dU = {100 * v['max_dU']:+.2f} pp over {v['cells']} cells -> **{v['H']}**")
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
    lines += ["", f"Per seed (descriptive): U_s = max over the attacks of seed s, %; dU_s = U_s(B) - U_s({res['reference_budget']}), pp; "
              f"seeds with dU_s >= {pp(res['seed_spread'])} pp (the attack of U_s(B)).", "",
              "| model | cell | budget | U_s per seed | U_s per seed at the reference | dU_s per seed | seeds with dU_s >= spread |",
              "|---|---|---:|---|---|---|---|"]
    for label, cells in res["labels"].items():
        for k, row in cells.items():
            for b in res["budgets"]:
                r = row[b]
                us = ", ".join(f"{sd}: {pp(u)}" for sd, u in r["U_s"].items())
                rs = ", ".join(f"{sd}: {pp(u)}" for sd, u in row[res["reference_budget"]]["U_s"].items())
                ds = ", ".join(f"{sd}: {100 * d:+.2f}" for sd, d in r["dU_s"].items())
                hits = "; ".join(f"{h['seed']} ({h['attack']}, {100 * h['dU']:+.2f})" for h in r["seeds_dU_ge_spread"])
                lines.append(f"| {label} | `{k}` | {b} | {us} | {rs} | {ds} | {r['n_seeds_dU_ge_spread']} of {r['n_seeds']}"
                             + (f": {hits}" if hits else "") + " |")
    return "\n".join(lines) + "\n"
