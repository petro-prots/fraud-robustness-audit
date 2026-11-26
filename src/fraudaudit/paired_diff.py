"""R57 (5): paired difference of two legs of one stored ``audit.json`` – an attack run or the conventional audit's
replay – with the same paired bootstrap as the «Paired difference» line of ``summary.md`` (``baseline_comparison``:
``uplift_ci``, two-sided, confidence / n_boot / seed of the primary baseline rule), on the same target episodes.

A leg is ``run:<attack>/<seed>`` (per-target ``ver_adv``) or ``replay:<rule>`` (per-target ``ver_replay``).
``worst_run`` applies the gate's rule (max upper bound of the uplift, first in run order on a tie) to a subset of
attacks. Before anything else the stored comparison is recomputed (worst run of the full ensemble − primary replay)
and must equal ``baseline.comparison.adv_minus_replay`` exactly – otherwise the bootstrap here is not the stored one.

Descriptive only; the subset (e.g. random + es) is chosen after the fact and is labelled so in the report.

    fraudaudit paired-diff --audit base out/…/base/audit.json --audit hard out/…/hard/audit.json \
        --attacks random es --out out/t28/m3/paired-random-es   (writes .json and .md)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fraudaudit.metrics import uplift_ci


def _targets(per_target: list[dict]) -> list[tuple]:
    return [(t["card"], t["start"]) for t in per_target]


def leg(audit: dict, ref: str) -> tuple[list[tuple], np.ndarray]:
    """Target keys and per-target VER of one leg: ``run:<key>`` or ``replay:<rule>``."""
    kind, _, name = ref.partition(":")
    if kind == "run":
        pt = audit["runs"][name]["per_target"]
        return _targets(pt), np.array([t["ver_adv"] for t in pt], dtype=np.float64)
    if kind == "replay":
        pt = audit["baseline"]["rules"][name]["per_target"]
        return _targets(pt), np.array([t["ver_replay"] for t in pt], dtype=np.float64)
    raise ValueError(f"leg must be run:<key> or replay:<rule>, got {ref!r}")


def boot_params(audit: dict) -> tuple[float, int, int]:
    """(confidence, n_boot, seed) of the primary baseline rule – what ``baseline_comparison`` uses."""
    b = audit["baseline"]
    metas = {r: x["meta"] for r, x in b["rules"].items()}
    params = {r: (float(m["confidence"]), int(m["n_boot"]), int(m["seed"])) for r, m in metas.items()}
    if len(set(params.values())) != 1:
        raise ValueError(f"baseline rules differ in bootstrap parameters: {params}")
    return params[b["primary"]]


def paired_diff(audit: dict, a: str, b: str) -> dict:
    """mean, lower, upper of ``VER_a,i − VER_b,i`` over the shared target episodes (two-sided)."""
    ka, va = leg(audit, a)
    kb, vb = leg(audit, b)
    if ka != kb:
        raise ValueError(f"{a} and {b} have different target episodes")
    conf, n_boot, seed = boot_params(audit)
    m, lo, hi = uplift_ci(va, vb, conf, n_boot, seed)
    return {"a": a, "b": b, "mean": m, "lower": lo, "upper": hi, "confidence": conf, "n_boot": n_boot,
            "seed": seed, "n_targets": len(ka)}


def worst_run(audit: dict, attacks: list[str] | None = None) -> str:
    """The gate's worst run (max ``uplift.upper``, first in run order on a tie), optionally among ``attacks``."""
    keys = [k for k in audit["runs"] if attacks is None or k.split("/")[0] in attacks]
    if not keys:
        raise ValueError(f"no runs of attacks {attacks}")
    return max(keys, key=lambda k: audit["runs"][k]["uplift"]["upper"])


def check_stored(audit: dict) -> None:
    """Recompute the stored «Paired difference» (worst run − primary replay); it must match exactly."""
    cmp = audit["baseline"]["comparison"]
    w = worst_run(audit)
    if w != cmp["worst_run"] or w != audit["worst_run"]:
        raise ValueError(f"worst run {w} differs from the stored {cmp['worst_run']} / {audit['worst_run']}")
    d = paired_diff(audit, f"run:{w}", f"replay:{audit['baseline']['primary']}")
    got, stored = (d["mean"], d["lower"], d["upper"]), tuple(cmp["adv_minus_replay"])
    if got != stored:
        raise ValueError(f"recomputed paired difference {got} != stored {stored}")


def word(d: dict, above: str, level: str = "on par") -> str:
    """Wording rule fixed before the computation (R57 (5)): interval above zero – ``above``; containing zero – on par;
    below zero – reported as such (not foreseen by the rule)."""
    if d["lower"] > 0:
        return above
    if d["upper"] < 0:
        return "below"
    return level


def subset_comparison(audit: dict, attacks: list[str]) -> dict:
    """(i) VER_adv(worst run among ``attacks``) − VER of the primary replay; (ii) VER of the ``max`` replay − the
    same worst run. The stored comparison is checked first."""
    check_stored(audit)
    primary = audit["baseline"]["primary"]
    w = worst_run(audit, attacks)
    run = audit["runs"][w]
    i = paired_diff(audit, f"run:{w}", f"replay:{primary}")
    ii = paired_diff(audit, "replay:max", f"run:{w}")
    i["word"] = word(i, "ahead")
    ii["word"] = word(ii, "exceeds")
    rules = audit["baseline"]["rules"]
    return {"attacks": list(attacks), "worst_run": w,
            "worst_ver_adv": [run["ver_adv"][k] for k in ("mean", "lower", "upper")],
            "worst_uplift": [run["uplift"][k] for k in ("mean", "lower", "upper")],
            "replay_ver": {r: [x["replay_ver"][k] for k in ("mean", "lower", "upper")] for r, x in rules.items()},
            "primary_rule": primary, "adv_minus_replay": i, "replay_max_minus_adv": ii,
            "stored_check": "recomputed worst run − primary replay equals the stored Paired difference exactly",
            "git_sha": audit["meta"].get("git_sha")}


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _d(d: dict) -> str:
    return f"{100 * d['mean']:+.1f} pp [{100 * d['lower']:+.1f}; {100 * d['upper']:+.1f}]"


def markdown(res: dict) -> str:
    att = " + ".join(res["attacks"])
    lines = [f"# Paired difference: {att} only (R57 (5))", "",
             f"Descriptive. **{att} only; the subset was chosen after the fact** – for comparability with the old "
             "ensemble without stepwise. The same paired bootstrap as the «Paired difference» line of `summary.md` "
             "(two-sided, confidence / n_boot / seed of the primary baseline rule); for each audit the stored line was "
             "recomputed first and matched exactly.", "",
             f"Wording rule (fixed before the computation): (i) interval above zero – «ahead», containing zero – "
             f"«on par»; (ii) above zero – «exceeds», containing zero – «on par».", "",
             f"| audit | worst run among {att} | VER_adv [CI] | VER of replay `first` / `max` | (i) VER_adv − replay "
             "`first` | word (i) | (ii) replay `max` − VER_adv | word (ii) | n targets |",
             "|---|---|---|---|---|---|---|---|---:|"]
    for label, r in res["audits"].items():
        va, rv = r["worst_ver_adv"], r["replay_ver"]
        i, ii = r["adv_minus_replay"], r["replay_max_minus_adv"]
        lines.append(f"| {label} | `{r['worst_run']}` | {_pct(va[0])} [{_pct(va[1])}; {_pct(va[2])}] "
                     f"| {_pct(rv['first'][0])} / {_pct(rv['max'][0])} | {_d(i)} | {i['word']} | {_d(ii)} "
                     f"| {ii['word']} | {i['n_targets']} |")
    lines += ["", "Bootstrap: " + "; ".join(
        f"{label} – confidence {r['adv_minus_replay']['confidence']}, n_boot {r['adv_minus_replay']['n_boot']}, "
        f"seed {r['adv_minus_replay']['seed']}, audit commit {r['git_sha']}" for label, r in res["audits"].items()),
        "", "Sources: " + ", ".join(f"`{p}`" for p in res["sources"].values()),
        "", f"Code of this comparison: {(res.get('code') or {}).get('git_sha', 'unknown')}"
        f" (dirty: {(res.get('code') or {}).get('git_dirty', 'unknown')}).", ""]
    return "\n".join(lines)


def write(audits: list[tuple[str, Path]], attacks: list[str], out: Path, code: dict | None = None) -> dict:
    res = {"attacks": list(attacks), "audits": {}, "sources": {}, "code": code}
    for label, path in audits:
        res["audits"][label] = subset_comparison(json.loads(Path(path).read_text(encoding="utf-8")), attacks)
        res["sources"][label] = str(path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    return res
