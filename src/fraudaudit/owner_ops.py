"""Task 28 (ruling R54): the owner's real operations inside the attack window – a descriptive sensitivity check, no
verdict.

The audit replays an episode from the card state at its start; the owner's own operations inside the 7-day window are
not replayed (``audit.REPLAY``) in the adversarial or the clean leg. Here every target of a cell is executed two more
times with them interleaved (``episode`` module docstring; ``audit.owner_window_txs``, ``audit._owner_legs``), in both
legs, with the same best candidate and random stream:

* ``count`` – the owner's decline is an ordinary decline (counts to ``max_declines``, starts the escalation);
* ``nocount`` – it neither counts nor escalates (the bank resolves it with the owner); the attempt enters the state.

The attacker's reconnaissance is unchanged (he does not see the owner's operations), so the plain leg is exactly the
stored record of the cell. ``policy-sweep --owner-ops`` writes per cell ``<out>/<label>/owner/<safe key>.json.gz``:
the cell key, its config hash, the target keys and per run per target the ``owner`` dict of ``audit._owner_legs``.
Like the trace it changes no number and no config hash.

Summary (``fraudaudit owner-ops``): per cell and variant the gate's statistic on the owner legs – per run the paired
one-sided upper bound of VER_adv − VER_clean (``metrics.uplift_ci``, the sweep's confidence, ``one_sided``, seed and
``N_BOOT``), U = max over the runs, ΔU = U(variant) − U(plain), the worst run, the mean uplift and the mean VERs – and
the owner statistics (operations in the window, executed, declined, sent to confirmation, lost to a block, cards
blocked by the owner's decline). Cross-check (``all_equal``): the plain record of the new sweep equals the stored
reference record of the same key exactly (config hash, targets, every run's ``uplift_upper``, ``uplift_mean`` and
per-target VERs), and the
upper bound recomputed from its per-target VERs equals the stored one (the statistic is the gate's). Descriptive only.

    fraudaudit owner-ops --cell LABEL,SWEEP,KEY,REFERENCE … --out DIR
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .trace import load_trace, safe_key, write_trace

OWNER_DIR = "owner"
VARIANTS = ("count", "nocount")
VARIANT_TEXT = {"count": "owner's decline counts to max_declines", "nocount": "owner's decline does not count"}


def owner_path(sweep_dir, key: str) -> Path:
    return Path(sweep_dir) / OWNER_DIR / f"{safe_key(key)}.json.gz"


def owner_ok(path, config_hash: str) -> bool:
    """True iff the owner file exists, can be read and belongs to the cell configuration ``config_hash``."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        return load_trace(p).get("config_hash") == config_hash
    except (OSError, ValueError, EOFError):
        return False


def cell_owner(runs: dict) -> dict:
    """{"targets": [[card, start], …], "runs": {run_key: [owner dict per target]}} from ``{run_key: AuditResult}``
    audited with ``owner_ops``."""
    first = next(iter(runs.values()))
    keys = [[int(t["card"]), t["start"]] for t in first.per_target]
    out = {}
    for k, r in runs.items():
        if [[int(t["card"]), t["start"]] for t in r.per_target] != keys:
            raise RuntimeError(f"run {k}: targets differ from the first run")
        if any("owner" not in t for t in r.per_target):
            raise RuntimeError(f"run {k}: audited without the owner operations")
        out[k] = [t["owner"] for t in r.per_target]
    return {"targets": keys, "runs": out}


def write_owner(path, data: dict) -> None:
    write_trace(path, data)


def _load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cell(d: dict, key: str, path) -> dict:
    for c in d.get("cells", []):
        if c["key"] == key:
            return c
    raise ValueError(f"{path}: no cell {key!r}")


def _upper(va, vc, gate: dict, seed: int) -> tuple[float, float]:
    from .metrics import uplift_ci
    m, _, hi = uplift_ci(np.asarray(va, dtype=np.float64), np.asarray(vc, dtype=np.float64),
                         float(gate["confidence"]), int(gate["n_boot"]), int(seed), one_sided=bool(gate["one_sided"]))
    return float(m), float(hi)


def run_order(gate: dict, keys) -> list[str]:
    """The run keys in the gate's ensemble order (attacks × attack seeds, as ``gate.run_ensemble`` runs them), so that
    a tie of the worst run is broken as in the record (``sweep_cells.run_cell``: the first maximum)."""
    keys = set(keys)
    order = [f"{a}/{int(s)}" for a in gate["attacks"] for s in gate["attack_seeds"] if f"{a}/{int(s)}" in keys]
    if set(order) != keys:
        raise ValueError(f"runs {sorted(keys - set(order))} are not in the gate's ensemble")
    return order


def leg_stats(runs: dict, gate: dict, seed: int) -> dict:
    """U statistics of one leg: ``runs`` = {run_key: (ver_adv list, ver_clean list)}; the worst run is the first
    maximum in the gate's ensemble order."""
    per = {}
    for k in run_order(gate, runs):
        va, vc = runs[k]
        m, hi = _upper(va, vc, gate, seed)
        per[k] = {"uplift_mean": m, "uplift_upper": hi, "ver_adv_mean": float(np.mean(va)),
                  "ver_clean_mean": float(np.mean(vc))}
    worst = max(per, key=lambda k: per[k]["uplift_upper"])
    return {"runs": per, "U": per[worst]["uplift_upper"], "worst_run": worst,
            "mean_uplift": float(np.mean([v["uplift_mean"] for v in per.values()])),
            "ver_adv_mean": float(np.mean([v["ver_adv_mean"] for v in per.values()])),
            "ver_clean_mean": float(np.mean([v["ver_clean_mean"] for v in per.values()]))}


def owner_counts(own_runs: dict, variant: str) -> dict:
    """Owner statistics of a variant summed over runs × targets (``n`` = runs × targets)."""
    items = [o[variant] for lst in own_runs.values() for o in lst]
    keys = ("owner_ops", "owner_declined", "owner_confirms", "owner_lost", "clean_owner_ops", "clean_owner_declined",
            "clean_owner_confirms", "clean_owner_lost")
    out = {k: int(sum(o[k] for o in items)) for k in keys}
    out.update(n=len(items), blocked=int(sum(o["blocked"] for o in items)),
               blocked_by_owner=int(sum(o["blocked_by_owner"] for o in items)),
               clean_blocked=int(sum(o["clean_blocked"] for o in items)),
               clean_blocked_by_owner=int(sum(o["clean_blocked_by_owner"] for o in items)),
               # the brief's quantity: the card was blocked and an owner's decline is among the executed operations (in
               # ``count`` every owner decline before the block counted; ``blocked_by_owner`` = it was the last one)
               blocked_with_owner_decline=int(sum(o["blocked"] and o["owner_declined"] > 0 for o in items)),
               clean_blocked_with_owner_decline=int(sum(o["clean_blocked"] and o["clean_owner_declined"] > 0
                                                        for o in items)))
    return out


def window_counts(own_runs: dict) -> dict:
    """Owner operations in the window per target (the same in every run; checked)."""
    lists = [[(o["n_window_ops"], o["n_other_fraud_in_window"]) for o in lst] for lst in own_runs.values()]
    if any(lst != lists[0] for lst in lists[1:]):
        raise ValueError("the owner operations of the window differ between runs")
    n = np.array([a for a, _ in lists[0]], dtype=np.int64)
    return {"n_targets": int(n.size), "n_targets_with_owner_ops": int((n > 0).sum()), "total": int(n.sum()),
            "median": float(np.median(n)), "max": int(n.max()) if n.size else 0,
            "n_other_fraud_in_window": int(sum(b for _, b in lists[0]))}


def cross_check(rec: dict, ref: dict, gate: dict, seed: int) -> dict:
    """Plain record of the new sweep vs the stored reference record (exact), and the statistic recomputed."""
    diffs = []
    if sorted(rec["runs"]) != sorted(ref["runs"]):
        diffs.append(f"run keys {sorted(rec['runs'])} vs {sorted(ref['runs'])}")
    for k in sorted(set(rec["runs"]) & set(ref["runs"])):
        a, b = rec["runs"][k], ref["runs"][k]
        for f in ("uplift_upper", "uplift_mean", "ver_adv", "ver_clean"):
            if a.get(f) != b.get(f):
                diffs.append(f"{k} {f}")
        if a.get("ver_adv") is not None:
            _, hi = _upper(a["ver_adv"], a["ver_clean"], gate, seed)
            if hi != a["uplift_upper"]:
                diffs.append(f"{k} recomputed upper {hi} != stored {a['uplift_upper']}")
    if rec.get("targets") != ref.get("targets"):
        diffs.append("targets")
    if rec.get("config_hash") != ref.get("config_hash"):
        diffs.append(f"config hash {rec.get('config_hash')} vs {ref.get('config_hash')}")
    return {"equal": not diffs, "differences": diffs}


def cell_summary(label: str, sweep_path, key: str, ref_path) -> dict:
    sweep_path, ref_path = Path(sweep_path), Path(ref_path)
    d, ref_d = _load(sweep_path), _load(ref_path)
    if d.get("model_json_sha256") != ref_d.get("model_json_sha256"):
        raise ValueError(f"{label} {key}: the sweep and the reference belong to different models")
    rec, ref = _cell(d, key, sweep_path), _cell(ref_d, key, ref_path)
    path = owner_path(sweep_path.parent, key)
    if not owner_ok(path, rec["config_hash"]):
        raise ValueError(f"{label} {key}: no owner file of this configuration ({path})")
    own = load_trace(path)
    if (own.get("provenance") or {}).get("git_sha") != (rec.get("provenance") or {}).get("git_sha"):
        raise ValueError(f"{label} {key}: the owner file was written by another commit than the record")
    if own["targets"] != rec["targets"]:
        raise ValueError(f"{label} {key}: owner file targets differ from the record")
    if sorted(own["runs"]) != sorted(rec["runs"]):
        raise ValueError(f"{label} {key}: owner file runs {sorted(own['runs'])} != record {sorted(rec['runs'])}")
    gate, seed = d["gate"], int(d["seed"])
    plain = leg_stats({k: (r["ver_adv"], r["ver_clean"]) for k, r in rec["runs"].items()}, gate, seed)
    out = {"label": label, "key": key, "sweep": str(sweep_path), "reference": str(ref_path),
           "reference_git_sha": (ref_d.get("meta") or {}).get("git_sha"),
           "sweep_git_sha": (d.get("meta") or {}).get("git_sha"),
           "plain": plain, "cross_check": cross_check(rec, ref, gate, seed), "window": window_counts(own["runs"])}
    for v in VARIANTS:
        st = leg_stats({k: ([o[v]["ver_adv"] for o in lst], [o[v]["ver_clean"] for o in lst])
                        for k, lst in own["runs"].items()}, gate, seed)
        st["dU"] = st["U"] - plain["U"]
        st["d_runs"] = {k: st["runs"][k]["uplift_upper"] - plain["runs"][k]["uplift_upper"] for k in plain["runs"]}
        st["owner"] = owner_counts(own["runs"], v)
        out[v] = st
    return out


def owner_summary(cells: list[tuple]) -> dict:
    """``cells`` = [(label, sweep policy-sweep.json, key, reference policy-sweep.json)]."""
    res = [cell_summary(*c) for c in cells]
    return {"cells": res, "all_equal": all(c["cross_check"]["equal"] for c in res), "variants": dict(VARIANT_TEXT)}


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f} %"


def _pp(x: float) -> str:
    return f"{100 * x:+.1f} pp"


def markdown(res: dict) -> str:
    lines = ["# Task 28: the owner's operations inside the attack window (R54) – descriptive, no verdict", "",
             "Every target of a cell executed three times with the same best candidate and random stream: **plain** "
             "(the stored record: the owner's in-window operations not replayed), **count** (the owner's real operations "
             "of the 7-day window interleaved in both legs; the owner's decline counts to max_declines and starts the "
             "escalation) and **nocount** (interleaved; the owner's decline neither counts nor escalates). The attacker's "
             "reconnaissance is unchanged. U = max over the runs of the one-sided upper bound of the paired uplift "
             "VER_adv − VER_clean (the gate's statistic); ΔU = U(variant) − U(plain).", "",
             "## U per cell", "",
             "| model | cell | U plain (run) | U count (run) | ΔU count | U nocount (run) | ΔU nocount | "
             "mean uplift plain / count / nocount | VER_adv plain / count / nocount | "
             "VER_clean plain / count / nocount |",
             "|---|---|---|---|---:|---|---:|---|---|---|"]
    for c in res["cells"]:
        p, a, b = c["plain"], c["count"], c["nocount"]
        lines.append(f"| {c['label']} | `{c['key']}` | {_pct(p['U'])} ({p['worst_run']}) | {_pct(a['U'])} "
                     f"({a['worst_run']}) | {_pp(a['dU'])} | {_pct(b['U'])} ({b['worst_run']}) | {_pp(b['dU'])} | "
                     f"{_pct(p['mean_uplift'])} / {_pct(a['mean_uplift'])} / {_pct(b['mean_uplift'])} | "
                     f"{_pct(p['ver_adv_mean'])} / {_pct(a['ver_adv_mean'])} / {_pct(b['ver_adv_mean'])} | "
                     f"{_pct(p['ver_clean_mean'], 2)} / {_pct(a['ver_clean_mean'], 2)} / "
                     f"{_pct(b['ver_clean_mean'], 2)} |")
    lines += ["", "## The owner's operations", "",
              "Window: per target, the owner's real operations in [start; start + 7 d) (the same in every run). "
              "Executed / declined / sent to confirmation / lost to a block / cards blocked: sums over runs × targets of the "
              "adversarial leg (clean leg in brackets). Cards blocked: all / with an owner's decline among the executed "
              "operations / the owner's decline was the blocking one (in `nocount` an owner's decline never counts).", "",
              "| model | cell | targets with owner ops | ops per target median / max | other fraud rows in window | "
              "variant | executed | declined | confirmations | lost to a block | cards blocked: all / with an owner "
              "decline / by it |",
              "|---|---|---|---|---:|---|---|---|---|---|---|"]
    for c in res["cells"]:
        w = c["window"]
        for v in VARIANTS:
            o = c[v]["owner"]
            lines.append(f"| {c['label']} | `{c['key']}` | {w['n_targets_with_owner_ops']} of {w['n_targets']} | "
                         f"{w['median']:g} / {w['max']} | {w['n_other_fraud_in_window']} | {v} | "
                         f"{o['owner_ops']} ({o['clean_owner_ops']}) | {o['owner_declined']} "
                         f"({o['clean_owner_declined']}) | {o['owner_confirms']} ({o['clean_owner_confirms']}) | "
                         f"{o['owner_lost']} ({o['clean_owner_lost']}) | {o['blocked']} / "
                         f"{o['blocked_with_owner_decline']} / {o['blocked_by_owner']} of {o['n']} "
                         f"({o['clean_blocked']} / {o['clean_blocked_with_owner_decline']} / "
                         f"{o['clean_blocked_by_owner']}) |")
    lines += ["", "## Runs (one-sided upper bound per attack/seed: plain → count / nocount)", ""]
    for c in res["cells"]:
        lines.append(f"- {c['label']} `{c['key']}`: " + ", ".join(
            f"{k} {_pct(c['plain']['runs'][k]['uplift_upper'])} → {_pct(c['count']['runs'][k]['uplift_upper'])} / "
            f"{_pct(c['nocount']['runs'][k]['uplift_upper'])}" for k in c["plain"]["runs"]))
    lines += ["", "## Cross-check: plain leg = stored record", "",
              "| model | cell | reference (commit) | result |", "|---|---|---|---|"]
    for c in res["cells"]:
        x = c["cross_check"]
        lines.append(f"| {c['label']} | `{c['key']}` | {c['reference']} ({(c['reference_git_sha'] or '?')[:7]}) | "
                     + ("equal" if x["equal"] else "**differs**: " + "; ".join(x["differences"][:10])) + " |")
    return "\n".join(lines) + "\n"


def write(res: dict, out_dir) -> None:
    from .sweep_cells import write_json
    out = Path(out_dir)
    write_json(out / "owner-ops.json", res)
    (out / "owner-ops.md").write_text(markdown(res), encoding="utf-8")
