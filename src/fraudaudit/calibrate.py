"""Calibration of the gate's resolution (ruling R14): how small a δ the gate can tell apart from sampling noise.

``fraudaudit calibrate`` runs the same ensemble as the gate (every attack × attack seed of ``gate.yaml``, same
targets, spares, ``confidence`` and ``one_sided``) on the baseline model and reports, per run, the uplift mean,
its upper bound, the bootstrap SE, and two resolution figures:

* **sampling resolution** = max over runs of (upper bound − mean): how far the gate's upper bound sits above
  the measured mean only because the targets are a finite sample of episodes;
* **seed spread** = max − min of the uplift means over the attack seeds (per attack and over all runs): how
  much the measured uplift moves when only the attack's random stream changes.

For a model whose true uplift is 0 the measured mean is ≈ 0, so its upper bound is ≈ the sampling
resolution: a δ below ``max(upper − mean)`` cannot be certified even for such a model. The «verdict vs δ»
table applies the gate rule (worst run's upper bound ≤ δ) to the baseline model for each δ. It never changes
``gate.yaml``: the author fixes δ from these numbers.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .audit import _num, _pct
from .gate import GateConfig, parse_run_key

DEFAULT_DELTAS = (0.02, 0.05, 0.10, 0.15, 0.20)
PASSED, FAILED = "PASSED", "FAILED"
_RUN_META = ("seed", "git_sha", "git_dirty", "spares_period", "target_periods", "n_target_cards", "n_jobs",
             "n_boot", "recon_budget_ops", "median_amount", "platform", "versions")


def calibration(results: dict, cfg: GateConfig, deltas, meta: dict | None = None) -> dict:
    """Calibration numbers of ``results`` (``{run_key: AuditResult}`` of ``run_ensemble``) for ``deltas``."""
    if not results:
        raise ValueError("no audit runs")
    runs = {}
    for key, r in results.items():
        attack, seed = parse_run_key(key)
        mean, lower, upper = (float(x) for x in r.uplift)
        runs[key] = {"attack": attack, "attack_seed": seed, "mean": mean, "upper": upper,
                     "upper_minus_mean": upper - mean, "se": float(r.meta["uplift_se"]), "lower": lower,
                     "ver_adv_mean": float(r.ver_adv[0]), "ver_clean_mean": float(r.ver_clean[0]),
                     "n_targets": int(r.n_targets)}
    res_key = max(runs, key=lambda k: runs[k]["upper_minus_mean"])
    by_attack: dict[str, list[float]] = {}
    for v in runs.values():
        by_attack.setdefault(v["attack"], []).append(v["mean"])
    spread = {a: max(m) - min(m) for a, m in by_attack.items()}
    means = [v["mean"] for v in runs.values()]
    resolution = runs[res_key]["upper_minus_mean"]
    worst = max(runs, key=lambda k: runs[k]["upper"])
    w_up = runs[worst]["upper"]
    table = [{"delta": float(d), "baseline_verdict": PASSED if w_up <= d else FAILED, "margin": float(d) - w_up,
              "runs_passing": sum(v["upper"] <= d for v in runs.values()), "n_runs": len(runs),
              "zero_uplift_model_passes": bool(d >= resolution)} for d in deltas]
    first = next(iter(results.values())).meta
    m = {k: first.get(k) for k in _RUN_META}
    m.update(model=first.get("model"), threat_model={k: (first.get("threat_model") or {}).get(k)
                                                     for k in ("file", "sha256")},
             n_spare_episodes=(first.get("recon") or {}).get("n_spare_episodes"),
             run_wall_seconds={k: r.meta.get("wall_seconds") for k, r in results.items()})
    m.update(meta or {})
    return {"one_sided": cfg.one_sided, "confidence": cfg.confidence, "runs": runs,
            "resolution": {"max_upper_minus_mean": resolution, "max_upper_minus_mean_run": res_key,
                           "max_se": max(v["se"] for v in runs.values()),
                           "seed_spread": spread, "seed_spread_max": max(spread.values()),
                           "mean_range_all_runs": max(means) - min(means),
                           "min_certifiable_delta": resolution},
            "worst_run": worst, "worst_upper": w_up, "verdict_vs_delta": table,
            "gate": dataclasses.asdict(cfg), "meta": m}


def calibration_markdown(data: dict) -> str:
    """English Markdown report of ``calibration`` output."""
    m, r, g = data["meta"], data["resolution"], data["gate"]
    conf = int(round(100 * data["confidence"]))
    side = "one-sided" if data["one_sided"] else "two-sided"
    med = m.get("median_amount")

    def usd(x):
        return f"~{_num(x * med, 0)} $" if med else "–"

    spares = ("the fraud episodes of the S half of the fit period (card split, ruling R20)"
              if m.get("spares_period") == "fit"
              else "a 50/50 card split of the target episodes")
    lines = [
        "# Gate resolution calibration (baseline model)",
        "",
        f"The same ensemble as the gate (`attacks` {g['attacks']} × `attack_seeds` {g['attack_seeds']}) was run on "
        f"the model `{(m.get('model') or {}).get('dir')}`. Targets: {(data['runs'][data['worst_run']])['n_targets']} "
        f"fraud episodes of the period(s) {', '.join(m.get('target_periods') or [])}; reconnaissance spares: "
        f"{spares}. Uplift = VER_adv − VER_clean (paired bootstrap over target episodes), upper bound {side} at "
        f"{conf} %. `gate.yaml` is not changed by this command: δ is fixed by the author.",
        "",
        "## Runs",
        "",
        f"| attack | attack seed | uplift mean | {side} upper {conf} % bound | upper − mean | bootstrap SE "
        "| VER_adv | VER_clean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, v in data["runs"].items():
        mark = " **(worst)**" if key == data["worst_run"] else ""
        lines.append(f"| {v['attack']} | {v['attack_seed']} | {_pct(v['mean'])} | {_pct(v['upper'])}{mark} "
                     f"| {_pct(v['upper_minus_mean'])} | {_pct(v['se'])} | {_pct(v['ver_adv_mean'])} "
                     f"| {_pct(v['ver_clean_mean'])} |")
    spread = ", ".join(f"{a} {_pct(x)}" for a, x in r["seed_spread"].items())
    lines += [
        "",
        "## Resolution",
        "",
        f"- **Sampling resolution: {_pct(r['max_upper_minus_mean'])}** (max over runs of upper − mean, run "
        f"`{r['max_upper_minus_mean_run']}`; largest bootstrap SE {_pct(r['max_se'])}). This is how far the gate's "
        "upper bound lies above the measured mean only because the target episodes are a finite sample: with "
        "other episodes of the same kind the mean would move by about one SE. It does not depend on δ.",
        f"- **Seed spread: {_pct(r['seed_spread_max'])}** (max − min of the uplift means over the attack seeds; "
        f"per attack: {spread}; over all runs: {_pct(r['mean_range_all_runs'])}). This is how much the measured "
        "uplift moves when only the attack's random stream changes (the search finds a better or worse strategy). "
        "The bootstrap bound does not contain it; the gate covers it by taking the worst run.",
        "",
        f"**Minimal δ the gate can certify ≈ {_pct(r['min_certifiable_delta'])}** ({usd(r['min_certifiable_delta'])} "
        "per episode at the median episode amount). Reasoning: for a model whose true uplift is 0 the measured "
        "mean is ≈ 0, so its upper bound is ≈ 0 + (upper − mean). The gate passes only if the upper bound ≤ δ, so "
        "a δ below max(upper − mean) fails even such a model. At δ equal to this value a zero-uplift model passes "
        "a run only when its measured mean is ≤ 0, about half of the time; a δ of about twice the sampling "
        "resolution lets it pass most runs. So δ should be at least the sampling resolution, and a value close "
        "to it is a strict gate.",
        "",
        "## Verdict as a function of δ",
        "",
        f"Baseline model: the gate rule applied to the worst run (`{data['worst_run']}`, upper bound "
        f"{_pct(data['worst_upper'])}). Zero-uplift model: whether δ is at least the sampling resolution.",
        "",
        "| δ | δ in $ (median A) | baseline model verdict | δ − worst upper | runs with upper ≤ δ "
        "| zero-uplift model can pass |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for row in data["verdict_vs_delta"]:
        lines.append(f"| {_pct(row['delta'])} | {usd(row['delta'])} | {row['baseline_verdict']} "
                     f"| {_pct(row['margin'])} | {row['runs_passing']} of {row['n_runs']} "
                     f"| {'yes' if row['zero_uplift_model_passes'] else 'no'} |")
    walls = m.get("run_wall_seconds") or {}
    lines += [
        "",
        "## Metadata",
        "",
        f"- seed {m.get('seed')}; git `{m.get('git_sha')}` (uncommitted changes: {m.get('git_dirty')})",
        f"- current `delta` in `gate.yaml`: {g['delta']} (not changed)",
        f"- model τ {(m.get('model') or {}).get('tau')}, SHA-256 `model.json` "
        f"`{(m.get('model') or {}).get('model_json_sha256')}`",
        f"- threat model `{m['threat_model'].get('file')}`, SHA-256 `{m['threat_model'].get('sha256')}`",
        f"- spare episodes {m.get('n_spare_episodes')}; reconnaissance budget {m.get('recon_budget_ops')} "
        f"operations per run; bootstrap resamples {m.get('n_boot')}",
        f"- processes {m.get('n_jobs')}; run times: " + ", ".join(f"{k} {v} s" for k, v in walls.items())
        + (f"; total {m['wall_seconds']} s" if "wall_seconds" in m else "")
        + (f" (with data and model loading {m['wall_seconds_with_loading']} s)"
           if "wall_seconds_with_loading" in m else ""),
    ]
    if m.get("platform"):
        lines.append(f"- platform: {m['platform'].get('platform')}")
    if m.get("versions"):
        lines.append("- versions: " + ", ".join(f"{k} {v}" for k, v in m["versions"].items()))
    lines.append("")
    return "\n".join(lines)


def write_calibration(out_dir: Path | str, data: dict) -> str:
    """``out_dir/calibration.json`` and ``out_dir/calibration.md``; returns the Markdown."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = calibration_markdown(data)
    (out / "calibration.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "calibration.md").write_text(md, encoding="utf-8")
    return md
