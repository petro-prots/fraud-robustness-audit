"""Task 27a (R39 order 2, R41 (9) item 2): the front cells with x_c > 5.98 on fresh seeds 10-15.

Question: does the large x_c of a confirmed front cell repeat on fresh seeds, or was it one of the seeds 4-9? For every
cell: U on the gate seeds (grid sweep), and for the earlier (4-9) and the fresh (10-15) confirmation seeds the
statistics of ``policy_summary.confirmation_stats`` – U_s per seed (max over the attacks), median, worst,
x_c = worst - U(gate seeds). Descriptive: no new margin X, no verdict (X stays per family as registered).

The fresh runs are a new commit; the grid and the earlier confirmations are the Task 24 / 19b ones. The cell parameters
(τ, reaction, corridor, budget) and the model (``model_json_sha256``) must be equal in the three files; the seed sets
must be disjoint and complete (every attack on every seed).

    fraudaudit seed-extension --cell xgb-hard,<key>,<grid>.json,<confirm 4-9>.json,<confirm 10-15>.json ... --out DIR
"""
from __future__ import annotations

import json
from pathlib import Path

from fraudaudit.policy_summary import confirmation_stats

SAME_CELL = ("tau", "max_declines", "tau_after_decline", "corridor_params", "target_fpr", "recon_budget_ops")


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cell(d: dict, key: str, path) -> dict:
    for c in d["cells"]:
        if c["key"] == key:
            return c
    raise ValueError(f"{path}: no cell {key!r}")


def _attacks(c: dict) -> list[str]:
    return sorted({rk.rsplit("/", 1)[0] for rk in c["runs"]})


def compare_seeds(cells) -> dict:
    """``cells``: iterable of (label, key, grid path, earlier confirmation path, fresh confirmation path)."""
    rows = []
    for label, key, grid_p, earlier_p, fresh_p in cells:
        g, e, f = _load(grid_p), _load(earlier_p), _load(fresh_p)
        for what, d, want in (("grid", g, "grid"), ("earlier", e, "confirmation"), ("fresh", f, "confirmation")):
            if d.get("role", "grid") != want:
                raise ValueError(f"{label} {key}: the {what} file has role {d.get('role')!r}, expected {want!r}")
        for what, d in (("earlier", e), ("fresh", f)):
            if d.get("model_json_sha256") != g.get("model_json_sha256"):
                raise ValueError(f"{label} {key}: model_json_sha256 of the {what} file differs from the grid")
        for what, d in (("earlier", e), ("fresh", f)):  # the same threat model, gate (apart from the seeds) and seed
            for k in ("threat_sha256", "corridor_threat_sha256"):
                if (d.get("meta") or {}).get(k) != (g.get("meta") or {}).get(k):
                    raise ValueError(f"{label} {key}: meta.{k} of the {what} file differs from the grid")
            if d.get("seed") != g.get("seed"):
                raise ValueError(f"{label} {key}: seed of the {what} file {d.get('seed')!r} != grid {g.get('seed')!r}")
            strip = lambda x: {k: v for k, v in (x or {}).items() if k != "attack_seeds"}  # noqa: E731
            if strip(d.get("gate")) != strip(g.get("gate")):
                raise ValueError(f"{label} {key}: gate of the {what} file differs from the grid (apart from the seeds)")
        gc, ec, fc = _cell(g, key, grid_p), _cell(e, key, earlier_p), _cell(f, key, fresh_p)
        for what, c in (("earlier", ec), ("fresh", fc)):
            for k in SAME_CELL:
                if c.get(k) != gc.get(k):
                    raise ValueError(f"{label} {key}: {k} of the {what} cell {c.get(k)!r} != grid {gc.get(k)!r}")
        attacks = _attacks(gc)
        e_seeds = (e.get("gate") or {}).get("attack_seeds")
        f_seeds = (f.get("gate") or {}).get("attack_seeds")
        g_seeds = (g.get("gate") or {}).get("attack_seeds") or []
        if not e_seeds or not f_seeds:
            raise ValueError(f"{label} {key}: the confirmation files must name their attack seeds")
        if set(e_seeds) & set(g_seeds):
            raise ValueError(f"{label} {key}: earlier seeds {e_seeds} overlap the gate seeds {g_seeds}")
        if set(f_seeds) & (set(e_seeds) | set(g_seeds)):
            raise ValueError(f"{label} {key}: fresh seeds {f_seeds} overlap the earlier seeds {e_seeds} / gate "
                             f"seeds {g_seeds}")
        es = confirmation_stats(ec, gc["U"], attacks, e_seeds)
        fs = confirmation_stats(fc, gc["U"], attacks, f_seeds)
        worst_run = max(sorted(fc["runs"]), key=lambda rk: fc["runs"][rk]["uplift_upper"])
        rows.append({"label": label, "key": key, "U_gate": float(gc["U"]), "gate_seeds": list(g_seeds),
                     "earlier": es, "fresh": fs, "fresh_worst_attack": worst_run,
                     "commits": {"grid": (g.get("meta") or {}).get("git_sha"),
                                 "earlier": (e.get("meta") or {}).get("git_sha"),
                                 "fresh": (f.get("meta") or {}).get("git_sha")}})
    return {"cells": rows}


def markdown(res: dict) -> str:
    pct = lambda x: f"{100 * x:.1f} %"  # noqa: E731
    lines = ["# Front cells on fresh seeds (Task 27a)", "",
             "U = worst one-sided upper bound of the paired uplift over attacks x seeds; U_s = max over the attacks of "
             "seed s; x_c = worst_s U_s - U (gate seeds). Descriptive: no new margin X, no verdict.", "",
             "| model | cell | U (gate seeds) | seeds | U_s per seed | median U_s | worst U_s (run) | x_c |",
             "|---|---|---:|---|---|---:|---:|---:|"]
    for r in res["cells"]:
        for what in ("earlier", "fresh"):
            s = r[what]
            us = ", ".join(f"{sd}: {pct(u)}" for sd, u in s["U_s"].items())
            seeds = f"{s['seeds'][0]}-{s['seeds'][-1]}"
            run = f" ({r['fresh_worst_attack']})" if what == "fresh" else ""
            lines.append(f"| {r['label']} | `{r['key']}` | {pct(r['U_gate'])} | {seeds} | {us} | {pct(s['median'])} | "
                         f"{pct(s['worst'])}{run} | {pct(s['x_c'])} |")
    lines += ["", "Commits (grid / earlier / fresh): "
              + "; ".join(f"{r['label']} `{r['key']}`: {r['commits']['grid']} / {r['commits']['earlier']} / "
                          f"{r['commits']['fresh']}" for r in res["cells"]), ""]
    return "\n".join(lines) + "\n"
