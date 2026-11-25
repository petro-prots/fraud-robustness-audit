"""Task 28: reproduction check of a stored run – an ``audit.json`` or ``skew.json`` of the same command at another commit.

Every value is compared exactly (floats bitwise, lists element by element), recursively; keys that describe the run
rather than its result are skipped wherever they occur (``VOLATILE``: commit, time, platform, library versions, worker
count, input file paths) – their differing values are only listed (``volatile_differences``: e.g. a library
version that changed). A key present on one side only is listed separately (``only_reference`` / ``only_new``: a
field added or removed by a later commit), not counted as a value difference. ``equal`` = no value difference.
Information only: the comparison never fails a chain; the reviewer reads ``repro.md``.

    fraudaudit audit-diff --reference OLD --new NEW --label NAME --out DIR   (appends to DIR/repro.json / repro.md)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

VOLATILE = frozenset({"git_sha", "git_dirty", "wall_seconds", "wall_seconds_with_loading", "platform", "versions",
                      "n_jobs", "skew_report", "file", "dir", "host_machine", "machine"})
MAX_LISTED = 50


def _walk(a, b, path: str, out: dict) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in VOLATILE:
                if k in a and k in b and a[k] != b[k] and k not in ("git_sha", "git_dirty") \
                        and not k.startswith("wall_seconds"):
                    v = f"{k}: {json.dumps(a[k], sort_keys=True)} -> {json.dumps(b[k], sort_keys=True)}"
                    if v not in out["volatile"]:
                        out["volatile"].append(v)
                continue
            p = f"{path}.{k}" if path else str(k)
            if k not in b:
                out["only_reference"].append(p)
            elif k not in a:
                out["only_new"].append(p)
            else:
                _walk(a[k], b[k], p, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["diffs"].append({"path": path, "reference": f"list of {len(a)}", "new": f"list of {len(b)}"})
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", out)
        return
    if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        return
    d = {"path": path, "reference": a, "new": b}
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        d["abs_diff"] = abs(float(a) - float(b))
    out["diffs"].append(d)


def compare(ref: dict, new: dict) -> dict:
    out = {"diffs": [], "only_reference": [], "only_new": [], "volatile": []}
    _walk(ref, new, "", out)
    num = [d["abs_diff"] for d in out["diffs"] if "abs_diff" in d]
    return {"equal": not out["diffs"], "n_value_differences": len(out["diffs"]),
            "max_abs_diff": max(num) if num else None, "differences": out["diffs"][:MAX_LISTED],
            "only_reference": out["only_reference"], "only_new": out["only_new"],
            "volatile_differences": out["volatile"][:MAX_LISTED]}


def _sha(d: dict) -> str | None:
    return d.get("git_sha") or (d.get("meta") or {}).get("git_sha")


def add_comparison(reference, new, label: str, out_dir) -> dict:
    ref_d = json.loads(Path(reference).read_text(encoding="utf-8"))
    new_d = json.loads(Path(new).read_text(encoding="utf-8"))
    res = {"label": label, "reference": str(reference), "new": str(new), "reference_git_sha": _sha(ref_d),
           "new_git_sha": _sha(new_d), **compare(ref_d, new_d)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / "repro.json"
    allr = json.loads(jp.read_text(encoding="utf-8")) if jp.is_file() else {"comparisons": []}
    allr["comparisons"] = [c for c in allr["comparisons"] if c["label"] != label] + [res]
    from .sweep_cells import write_json
    write_json(jp, allr)
    (out / "repro.md").write_text(markdown(allr), encoding="utf-8")
    return res


def _short(v) -> str:
    s = json.dumps(v)
    return s if len(s) <= 60 else s[:57] + "..."


def markdown(allr: dict) -> str:
    lines = ["# Task 28: reproduction of the stored M3 runs at the current commit", "",
             "Exact comparison (floats bitwise) of every result value; skipped keys: " + ", ".join(sorted(VOLATILE))
             + ". Keys on one side only = fields added / removed by later commits (not value differences).", "",
             "| comparison | reference (commit) | new (commit) | result | value differences | max abs diff | "
             "keys only in reference / new |", "|---|---|---|---|---:|---:|---|"]
    for c in allr["comparisons"]:
        mad = "" if c["max_abs_diff"] is None else f"{c['max_abs_diff']:.3g}"
        lines.append(f"| {c['label']} | {c['reference']} ({(c['reference_git_sha'] or '?')[:7]}) | {c['new']} "
                     f"({(c['new_git_sha'] or '?')[:7]}) | {'equal' if c['equal'] else '**differs**'} | "
                     f"{c['n_value_differences']} | {mad} | {len(c['only_reference'])} / {len(c['only_new'])} |")
    for c in allr["comparisons"]:
        if c["differences"] or c["only_reference"] or c["only_new"] or c.get("volatile_differences"):
            lines += ["", f"## {c['label']}", ""]
            for d in c["differences"]:
                lines.append(f"- `{d['path']}`: {_short(d['reference'])} → {_short(d['new'])}")
            if c["n_value_differences"] > len(c["differences"]):
                lines.append(f"- … {c['n_value_differences'] - len(c['differences'])} more (repro.json lists "
                             f"{len(c['differences'])})")
            for side in ("only_reference", "only_new"):
                keys = sorted({k.split("[")[0] for k in c[side]})
                if keys:
                    lines.append(f"- {side.replace('_', ' ')}: " + ", ".join(f"`{k}`" for k in keys[:20])
                                 + (f" … ({len(keys)} distinct)" if len(keys) > 20 else ""))
            for v in c.get("volatile_differences", []):
                lines.append(f"- run metadata (not compared): {_short(v) if len(v) > 200 else v}")
    return "\n".join(lines) + "\n"
