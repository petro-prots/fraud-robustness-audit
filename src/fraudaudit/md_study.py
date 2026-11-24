"""Task 27b (R41 (6), R48, R50): why md2 on the base models – R19 on 10 fresh ES seeds and the blocking trace.

Cells (fixed in the Task 27 brief, not chosen after a result): xgb-base md2, xgb-hard md2, gru-base md2 (own τ, corridor
off) and m2-base md3 for comparison, each with its mdinf reference; ES only, seeds 4–13, ``policy-sweep --trace``.

(i) **Relative overestimation** (VER_∞ − VER_md) / VER_md by ``policy_summary.rel_overestimation`` (paired percentile
bootstrap over targets, n_boot 2000, seed 42 – unchanged): per seed s on the per-target VERs of run ``es/s`` of the two
cells, and pooled on the per-target VER averaged over the ES seeds (the R19 definition, ``policy_summary._ver_matrix``).
Beside it the gate-seed (1–3) value from the stored R19 grid cells computed by the same code, compared with the
``r19`` entry of ``summary8.json`` (equal / differs). Per model the counts of seeds with rel > 0 and lower bound > 0,
with upper bound < 0, and the rest (counts only), and the absolute difference VER_∞ − VER_md as ``rel_overestimation``
returns it.

(ii) **Blocking trace** (``trace.py``), per model × cell × seed and pooled over the seeds: share of targets blocked; the
distribution of ``block_op`` (for mdinf – never blocked – the distribution of the declines is the informative one; both
are given for both cells); median executed operations; mean declines; mean VER_adv. **ES strategy parameters** per run
(from ``best``): k, u, the amount profile (shares of A of the planned operations in time order), spacing (median and
minimum gap between consecutive planned operations and span first → last, hours) and the categories (count per
category). Descriptive: no verdict.

Checks (``ValueError``): both cells in the new sweep have the same targets and τ, are corridor off, have exactly the ES
runs of the expected seeds; the reference is mdinf; the trace belongs to the cell (config hash, runs, targets, per-target
VER_adv); the grid file(s) have the same model and both keys with the same targets. The grid may be given as several
policy-sweep.json files of one model joined by ``+`` (m2-base: md3 is a Task 19 cell, md2 / mdinf are Task 19b cells in
another file); a key found in two of them is refused.

    fraudaudit md-study --cell LABEL,SWEEP,MD_KEY,INF_KEY,GRID_SWEEP ... [--r19-reference summary8.json] --out DIR
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .policy_summary import _ver_matrix, rel_overestimation
from .trace import load_trace, trace_path

SEEDS = tuple(range(4, 14))
ATTACK = "es"
BOOT_SEED = 42


def _load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cell(d: dict, key: str, path) -> dict:
    for c in d["cells"]:
        if c["key"] == key:
            return c
    raise ValueError(f"{path}: no cell {key!r}")


def load_grid(spec: str) -> tuple[dict, list[dict], dict]:
    """``spec`` = one policy-sweep.json or several joined by ``+`` (one model). Returns ({model SHA, key: cell}, the
    per-file provenance, {key: the sweep dict it came from}). ValueError on mixed models or a key in two files."""
    cells: dict = {}
    owner: dict = {}
    prov = []
    sha = None
    for p in [x for x in spec.split("+") if x]:
        d = _load(p)
        if sha is not None and d.get("model_json_sha256") != sha:
            raise ValueError(f"grid {spec}: {p} belongs to another model (model_json_sha256 differs)")
        sha = d.get("model_json_sha256")
        for c in d["cells"]:
            if c["key"] in cells:
                raise ValueError(f"grid {spec}: cell {c['key']!r} is in more than one file")
            cells[c["key"]] = c
            owner[c["key"]] = d
        prov.append(provenance(p, d))
    if sha is None:
        raise ValueError(f"grid {spec!r}: no file")
    return {"model_json_sha256": sha, "cells": cells}, prov, owner


def provenance(path, d: dict) -> dict:
    m = d.get("meta") or {}
    return {"path": str(path), "git_sha": m.get("git_sha"), "git_dirty": m.get("git_dirty"),
            "model_json_sha256": d.get("model_json_sha256"), "model_dir": d.get("model_dir"), "role": d.get("role"),
            "attack_seeds": (d.get("gate") or {}).get("attack_seeds")}


def run_keys(seeds, attack: str = ATTACK) -> list[str]:
    return [f"{attack}/{int(s)}" for s in seeds]


def check_runs(cell: dict, seeds, what: str, attack: str = ATTACK) -> None:
    want = sorted(run_keys(seeds, attack))
    if sorted(cell["runs"]) != want:
        raise ValueError(f"{what} {cell['key']}: runs {sorted(cell['runs'])}, expected exactly {want}")


def check_trace(tr: dict, cell: dict, what: str) -> None:
    """ValueError unless the trace belongs to ``cell``: config hash, run keys, targets and per-target VER_adv."""
    if tr.get("key") != cell["key"] or tr.get("config_hash") != cell.get("config_hash"):
        raise ValueError(f"{what}: the trace belongs to {tr.get('key')!r} / {tr.get('config_hash')}, the cell is "
                         f"{cell['key']!r} / {cell.get('config_hash')}")
    if sorted(tr["runs"]) != sorted(cell["runs"]):
        raise ValueError(f"{what}: trace runs {sorted(tr['runs'])} != cell runs {sorted(cell['runs'])}")
    targets = [[int(c), s] for c, s in cell["targets"]]
    for rk, r in tr["runs"].items():
        if [[int(t["card"]), t["start"]] for t in r["per_target"]] != targets:
            raise ValueError(f"{what} {rk}: trace targets differ from the cell's")
        if [float(t["ver_adv"]) for t in r["per_target"]] != [float(x) for x in cell["runs"][rk]["ver_adv"]]:
            raise ValueError(f"{what} {rk}: trace VER_adv differs from the cell's")


# --- (i) relative overestimation -------------------------------------------------------------------------------------


def classify(r: dict) -> str:
    """``positive`` (rel > 0 and lower > 0), ``negative`` (upper < 0) or ``rest``."""
    if r["rel"] > 0 and r["lower"] > 0:
        return "positive"
    if r["upper"] < 0:
        return "negative"
    return "rest"


def overestimation(md: dict, inf: dict, seeds, attack: str = ATTACK) -> dict:
    per_seed = {}
    for s in seeds:
        rk = f"{attack}/{int(s)}"
        per_seed[str(int(s))] = rel_overestimation(np.asarray(inf["runs"][rk]["ver_adv"], dtype=np.float64),
                                                   np.asarray(md["runs"][rk]["ver_adv"], dtype=np.float64),
                                                   seed=BOOT_SEED)
    pooled = rel_overestimation(_ver_matrix(inf, attack), _ver_matrix(md, attack), seed=BOOT_SEED)
    counts = Counter(classify(r) for r in per_seed.values())
    return {"per_seed": per_seed, "pooled": pooled,
            "counts": {k: int(counts.get(k, 0)) for k in ("positive", "negative", "rest")}}


# --- (ii) blocking trace -------------------------------------------------------------------------------------------


def _counts(xs) -> dict:
    return {str(k): int(v) for k, v in sorted(Counter(int(x) for x in xs).items())}


def block_stats(targets: list[dict]) -> dict:
    """Per-target trace rows (one run or several pooled) → the descriptive blocking statistics."""
    if not targets:
        raise ValueError("no targets in the trace")
    return {"n": len(targets), "share_blocked": float(np.mean([bool(t["blocked"]) for t in targets])),
            "block_op_counts": _counts(t["block_op"] for t in targets if t["block_op"] is not None),
            "n_declines_counts": _counts(t["n_declines"] for t in targets),
            "median_n_ops": float(np.median([t["n_ops"] for t in targets])),
            "mean_n_declines": float(np.mean([t["n_declines"] for t in targets])),
            "mean_ver_adv": float(np.mean([t["ver_adv"] for t in targets]))}


def strategy_params(best: dict) -> dict:
    """k, u, amount profile, spacing and categories of an ES strategy (``audit.strategy_dict`` with merchants: its
    ``ops`` are the planned operations in time order)."""
    if best.get("type") is not None or "ops" not in best:
        raise ValueError(f"not an ES strategy: {best.get('type')!r}")
    ops = sorted(best["ops"], key=lambda o: o["offset_hours"])
    off = [float(o["offset_hours"]) for o in ops]
    gaps = np.diff(off) if len(off) > 1 else np.empty(0)
    return {"k": int(best["k"]), "u": float(best["u"]), "amount_profile": [float(o["share_of_A"]) for o in ops],
            "offsets_h": off, "median_gap_h": float(np.median(gaps)) if gaps.size else None,
            "min_gap_h": float(gaps.min()) if gaps.size else None, "span_h": float(off[-1] - off[0]) if off else None,
            "categories": dict(sorted(Counter(str(o.get("category")) for o in ops).items()))}


def trace_section(tr: dict, seeds, attack: str = ATTACK) -> dict:
    per_seed, pooled, strat = {}, [], {}
    for s in seeds:
        rk = f"{attack}/{int(s)}"
        rows = tr["runs"][rk]["per_target"]
        per_seed[str(int(s))] = block_stats(rows)
        pooled += rows
        strat[rk] = strategy_params(tr["runs"][rk]["best"])
    return {"per_seed": per_seed, "pooled": block_stats(pooled), "strategy": strat}


# --- the study ---------------------------------------------------------------------------------------------------------


def _same(a: dict, b: dict, k: str) -> bool:
    x, y = a.get(k), b.get(k)
    if isinstance(x, float) and isinstance(y, float):
        return math.isclose(x, y, rel_tol=1e-9)
    return x == y


def effective(cell: dict, sweep: dict) -> dict:
    """The cell's setup with the fallbacks of older records (Task 19 cells carry neither ``corridor_params`` nor
    ``recon_budget_ops``): the sweep's corridor for a corridor-on cell (as ``policy_summary.normalize``), the budget of
    the sweep's meta; None when not recorded at all."""
    c = {k: cell.get(k) for k in ("tau", "max_declines", "tau_after_decline", "corridor", "target_fpr")}
    c["corridor_params"] = (cell.get("corridor_params") or sweep.get("corridor_params")) if cell.get("corridor") == "on" \
        else None
    c["recon_budget_ops"] = cell.get("recon_budget_ops") or (sweep.get("meta") or {}).get("recon_budget_ops")
    return c


def check_same_setup(new: dict, new_sweep: dict, ref: dict, ref_sweep: dict, what: str) -> None:
    """ValueError unless ``new`` and the reference cell ``ref`` have the same setup: τ and the escalation threshold
    (relative 1e-9), max_declines, corridor and its parameters, target FPR, the threat files (SHA-256 in the sweeps'
    meta) and the reconnaissance budget (compared when both records carry it; older records do not, their budget is
    the one of the same threat file)."""
    a, b = effective(new, new_sweep), effective(ref, ref_sweep)
    for k in ("tau", "max_declines", "tau_after_decline", "corridor", "target_fpr", "corridor_params"):
        if not _same(a, b, k):
            raise ValueError(f"{what}: {k} {a.get(k)!r} != reference {b.get(k)!r}")
    if a["recon_budget_ops"] is not None and b["recon_budget_ops"] is not None \
            and a["recon_budget_ops"] != b["recon_budget_ops"]:
        raise ValueError(f"{what}: recon_budget_ops {a['recon_budget_ops']} != reference {b['recon_budget_ops']}")
    for k in ("threat_sha256", "corridor_threat_sha256"):
        x, y = (new_sweep.get("meta") or {}).get(k), (ref_sweep.get("meta") or {}).get(k)
        if x is not None and y is not None and x != y:
            raise ValueError(f"{what}: meta.{k} differs from the reference")


def study_model(label: str, sweep_p, md_key: str, inf_key: str, grid_spec: str, r19_ref: dict | None = None,
                seeds=SEEDS) -> dict:
    d = _load(sweep_p)
    md, inf = _cell(d, md_key, sweep_p), _cell(d, inf_key, sweep_p)
    for c in (md, inf):
        if c.get("corridor") != "off":
            raise ValueError(f"{label} {c['key']}: corridor must be off")
        check_runs(c, seeds, f"{label}")
    if inf.get("max_declines") is not None or inf.get("reaction") != "mdinf":
        raise ValueError(f"{label} {inf_key}: the reference must be mdinf")
    if md.get("max_declines") is None:
        raise ValueError(f"{label} {md_key}: the md cell must block (max_declines)")
    if md.get("targets") != inf.get("targets"):
        raise ValueError(f"{label}: the md and mdinf cells have different targets")
    if md.get("tau") != inf.get("tau"):
        raise ValueError(f"{label}: tau of {md_key} {md.get('tau')} != {inf_key} {inf.get('tau')}")
    grid, gprov, gsweeps = load_grid(grid_spec)
    if grid["model_json_sha256"] != d.get("model_json_sha256"):
        raise ValueError(f"{label}: model_json_sha256 of the grid {grid_spec} differs from {sweep_p}")
    gmd, ginf = (grid["cells"].get(k) for k in (md_key, inf_key))
    if gmd is None or ginf is None:
        raise ValueError(f"{label}: the grid {grid_spec} lacks {md_key!r} or {inf_key!r}")
    for g, c in ((gmd, md), (ginf, inf)):
        check_same_setup(c, d, g, gsweeps[c["key"]], f"{label} {c['key']} (grid)")
        if g.get("targets") != c.get("targets"):
            raise ValueError(f"{label} {c['key']}: the grid cell has other targets")
    traces = {}
    for name, c in (("md", md), ("inf", inf)):
        p = trace_path(Path(sweep_p).parent, c["key"])
        if not p.is_file():
            raise ValueError(f"{label} {c['key']}: no trace file {p}")
        tr = load_trace(p)
        check_trace(tr, c, f"{label} {c['key']}")
        traces[name] = (p, tr)

    over = overestimation(md, inf, seeds)
    g = rel_overestimation(_ver_matrix(ginf, ATTACK), _ver_matrix(gmd, ATTACK), seed=BOOT_SEED)
    reaction = md["reaction"]
    ref = None if r19_ref is None else ((r19_ref.get(label) or {}).get(f"adv_{ATTACK}") or {}).get(reaction)
    if r19_ref is not None and ref is None:
        raise ValueError(f"{label}: the R19 reference has no entry for {label} adv_{ATTACK} {reaction}")
    grid_row = {"seeds": sorted({int(rk.rsplit("/", 1)[1]) for rk in gmd["runs"] if rk.startswith(ATTACK + "/")}),
                **g, "summary8": ref,
                "summary8_check": None if ref is None else ("equal" if ref == g else "differs")}
    return {"label": label, "md_key": md_key, "inf_key": inf_key, "reaction": reaction, "tau": md["tau"],
            "n_targets": len(md["targets"]), "seeds": [int(s) for s in seeds], "overestimation": over,
            "grid": grid_row,
            "trace": {"md": trace_section(traces["md"][1], seeds), "inf": trace_section(traces["inf"][1], seeds)},
            "inputs": {"sweep": provenance(sweep_p, d), "grid": gprov,
                       "traces": {k: str(v[0]) for k, v in traces.items()}}}


def md_study(cells, r19_reference=None, seeds=SEEDS) -> dict:
    """``cells``: iterable of (label, sweep path, md key, mdinf key, grid spec). ``r19_reference``: summary8.json (its
    ``r19``) for the equal / differs check of the gate-seed values, or None."""
    ref = None if r19_reference is None else _load(r19_reference).get("r19")
    if r19_reference is not None and ref is None:
        raise ValueError(f"{r19_reference}: no r19 entry")
    rows = [study_model(lab, sp, mk, ik, gs, ref, seeds) for lab, sp, mk, ik, gs in cells]
    if len({r["label"] for r in rows}) != len(rows):
        raise ValueError("a label is given twice")
    n_boot = rows[0]["overestimation"]["pooled"]["n_boot"] if rows else None
    return {"models": rows, "seeds": [int(s) for s in seeds], "attack": ATTACK, "n_boot": n_boot,
            "bootstrap_seed": BOOT_SEED, "r19_reference": None if r19_reference is None else str(r19_reference)}


# --- markdown ---------------------------------------------------------------------------------------------------------


def _pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f} %"


def _pp(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def _h(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def _rel(r: dict) -> str:
    return (f"{_pct(r['ver_inf'])} | {_pct(r['ver_md'])} | {_pct(r['rel'])} [{_pct(r['lower'])}; {_pct(r['upper'])}] | "
            f"{_pp(r['diff'])} [{_pp(r['diff_lower'])}; {_pp(r['diff_upper'])}]")


def _dist(d: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items()) or "–"


def markdown(res: dict) -> str:
    lines = ["# Why md2 on the base models (Task 27b)", "",
             f"ES only, seeds {res['seeds'][0]}–{res['seeds'][-1]}; own τ, corridor off. Relative overestimation = "
             "(VER_∞ − VER_md) / VER_md, 95 % paired percentile bootstrap over targets (n_boot "
             f"{res['n_boot']}, seed {res['bootstrap_seed']}); pooled = per-target VER averaged over the ES seeds (the "
             "R19 definition); seeds 1–3 = the stored R19 grid cells, same code. Difference in pp. Descriptive: no "
             "verdict.", "",
             "## (i) Relative overestimation", "",
             "| model | cell | seed(s) | VER_∞ | VER_md | overestimation [95 % CI] | VER_∞ − VER_md, pp [95 % CI] |",
             "|---|---|---|---:|---:|---|---|"]
    for m in res["models"]:
        o = m["overestimation"]
        for s, r in o["per_seed"].items():
            lines.append(f"| {m['label']} | {m['reaction']} | {s} | {_rel(r)} |")
        lines.append(f"| {m['label']} | {m['reaction']} | pooled {m['seeds'][0]}–{m['seeds'][-1]} | {_rel(o['pooled'])} |")
        g = m["grid"]
        chk = "" if g["summary8_check"] is None else f" (summary8: {g['summary8_check']})"
        lines.append(f"| {m['label']} | {m['reaction']} | grid {', '.join(map(str, g['seeds']))}{chk} | {_rel(g)} |")
    lines += ["", "Seeds per model: rel > 0 and lower bound > 0 / upper bound < 0 / the rest.", "",
              "| model | cell | rel > 0, lower > 0 | upper < 0 | rest |", "|---|---|---:|---:|---:|"]
    for m in res["models"]:
        c = m["overestimation"]["counts"]
        lines.append(f"| {m['label']} | {m['reaction']} | {c['positive']} | {c['negative']} | {c['rest']} |")
    lines += ["", "## (ii) Blocking trace", "",
              "block_op = 1-based executed operation on which the card was blocked (counts); declines = declines per "
              "target (counts); pooled over the seeds = every target-run.", "",
              "| model | cell | seed | blocked | block_op | declines | median ops | mean declines | mean VER_adv |",
              "|---|---|---|---:|---|---|---:|---:|---:|"]
    for m in res["models"]:
        for name, key in (("md", m["md_key"]), ("inf", m["inf_key"])):
            t = m["trace"][name]
            rows = list(t["per_seed"].items()) + [("pooled", t["pooled"])]
            cell = m["reaction"] if name == "md" else "mdinf"
            for s, b in rows:
                lines.append(f"| {m['label']} | {cell} | {s} | {_pct(b['share_blocked'])} | "
                             f"{_dist(b['block_op_counts'])} | {_dist(b['n_declines_counts'])} | "
                             f"{b['median_n_ops']:g} | {b['mean_n_declines']:.2f} | {_pct(b['mean_ver_adv'])} |")
    lines += ["", "## ES strategy parameters per run", "",
              "Amount profile = shares of A of the planned operations in time order; gaps and span between planned "
              "operations in hours.", "",
              "| model | cell | run | k | u | amount profile | median gap, h | min gap, h | span, h | categories |",
              "|---|---|---|---:|---:|---|---:|---:|---:|---|"]
    for m in res["models"]:
        for name in ("md", "inf"):
            cell = m["reaction"] if name == "md" else "mdinf"
            for rk, s in m["trace"][name]["strategy"].items():
                prof = ", ".join(_pct(x) for x in s["amount_profile"])
                lines.append(f"| {m['label']} | {cell} | {rk} | {s['k']} | {_pct(s['u'])} | {prof} | "
                             f"{_h(s['median_gap_h'])} | {_h(s['min_gap_h'])} | {_h(s['span_h'])} | "
                             f"{_dist(s['categories'])} |")
    lines += ["", "## Inputs", ""]
    for m in res["models"]:
        i = m["inputs"]
        grid = "; ".join(f"`{g['path']}` ({g['git_sha']})" for g in i["grid"])
        lines.append(f"- {m['label']}: sweep `{i['sweep']['path']}` ({i['sweep']['git_sha']}), grid {grid}; model "
                     f"`{i['sweep']['model_json_sha256']}`")
    return "\n".join(lines) + "\n"
