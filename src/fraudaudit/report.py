"""Gate report: ``out/audit.json`` (every run, every check, the verdict) and the English Markdown summary
for ``$GITHUB_STEP_SUMMARY`` / ``out/summary.md``."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from .audit import (REPLAY_TEXT, BaselineResult, corridor_line, false_blocks_line, _num, _pct, baseline_to_json, candidate_dict, policy_markdown, spares_text,
                    to_json)
from .metrics import uplift_ci
from .gate import HELD_OUT_ATTACKS, Check, GateConfig, parse_run_key, passed

VERDICT_OK, VERDICT_FAIL = "PASSED", "FAILED"

_TITLE = {
    "ver_uplift": "VER uplift under attack – {side} upper {conf} % bound of VER_adv − VER_clean (paired "
                  "bootstrap), worst attack × seed",
    "recall_clean": "recall on clean test operations at τ",
    "feasibility": "share of executed adversarial operations within the threat model (minimum over runs)",
    "skew": "decision mismatches between the training and serving feature paths",
}


def _value(c: Check) -> str:
    if c.value is None:
        return "–"
    if c.name == "feasibility" and c.value != 1.0:
        return _pct_exact(c.value)
    if c.name in ("ver_uplift", "feasibility"):
        return _pct(c.value)
    if c.name == "recall_clean":
        return _num(c.value, 4)
    return str(int(c.value))


def _pct_exact(x: float) -> str:
    """Percent with as many decimals as needed so that a value below 1 never prints as 100 %."""
    if not math.isfinite(x):
        return f"{x} %"
    for d in range(1, 13):
        s = f"{100 * x:.{d}f}"
        if float(s) != 100.0:
            break
    return f"{s} %"


def _threshold(c: Check) -> str:
    if c.threshold is None:
        return "not set (δ = null)"
    return {"ver_uplift": f"≤ {_pct(c.threshold)}", "recall_clean": f"≥ {_num(c.threshold, 2)}",
            "feasibility": "= 100.0 %", "skew": "= 0 (and all features within tolerance)"}[c.name]


def _ci(t) -> str:
    return f"{_pct(t[0])} [{_pct(t[1])}; {_pct(t[2])}]"


def _side(r) -> str:
    return "one-sided" if r.meta.get("one_sided") else "two-sided"


def _uplift(r) -> str:
    """The uplift of run ``r``: mean and its upper bound (one-sided) or the two-sided interval."""
    if r.meta.get("one_sided"):
        return f"{_pct(r.uplift[0])} (upper {_pct(r.uplift[2])})"
    return _ci(r.uplift)


def baseline_comparison(b: BaselineResult, results: dict, worst_key: str) -> dict:
    """The episode side of the comparison: the worst run (gate's ``ver_uplift`` worst) and the mean and range of
    the uplift means over every attack × seed run, plus the paired bootstrap of ``ver_adv_i − ver_replay_i``
    (worst run vs the conventional audit's replay) on the same target episodes."""
    w = results[worst_key]
    if [(t["card"], t["start"]) for t in w.per_target] != [(t["card"], t["start"]) for t in b.per_target]:
        raise ValueError("the baseline and the episode audit have different target episodes")
    conf = float(b.meta["confidence"])
    diff = uplift_ci(np.array([t["ver_adv"] for t in w.per_target]),
                     np.array([t["ver_replay"] for t in b.per_target]), conf, int(b.meta["n_boot"]),
                     int(b.meta["seed"]))
    ups = [float(r.uplift[0]) for r in results.values()]
    return {"worst_run": worst_key, "worst_ver_adv": tuple(w.ver_adv), "worst_uplift": tuple(w.uplift),
            "uplift_mean_runs": float(np.mean(ups)), "uplift_min_runs": float(min(ups)),
            "uplift_max_runs": float(max(ups)), "n_runs": len(ups), "adv_minus_replay": tuple(diff)}


def baseline_interpretation(b: BaselineResult, cmp: dict) -> str:
    """One plain English sentence in two parts, both decided by intervals.

    (a) The per-operation evaded amount share vs the replay VER interval: above its upper bound, the
    conventional figure is not a loss estimate. (b) The paired ``ver_adv_i − ver_replay_i`` interval: lower
    bound > 0 – the attack the conventional audit found is weaker than the episode attack; upper < 0 –
    stronger; else no significant difference."""
    claim, rep = b.evaded_amount_share, b.replay_ver
    d = cmp["adv_minus_replay"]
    if claim > rep[2]:
        a = (f"The per-operation evasion rate is not a loss estimate: under the conventional audit, operations "
             f"worth {_pct(claim)} of the fraud amount evade, but the same operations executed as an episode "
             f"yield only {_pct(rep[0])}")
    elif claim < rep[1]:
        a = (f"The per-operation evasion rate ({_pct(claim)} of the fraud amount) is lower than what the same "
             f"operations yield as an episode ({_pct(rep[0])})")
    else:
        a = (f"The per-operation evasion rate ({_pct(claim)} of the fraud amount) is consistent with what the same "
             f"operations yield as an episode ({_pct(rep[0])})")
    ru, wu = _pct(b.replay_uplift[0]), _pct(cmp["worst_uplift"][0])
    gap = f"VER difference {_pct(d[0])} [{_pct(d[1])}; {_pct(d[2])}]"
    if d[1] > 0:
        part = f"the attack it finds is weaker than the episode attack: uplift {ru} vs {wu} ({gap})"
    elif d[2] < 0:
        part = f"the attack it finds is stronger than the episode attack: uplift {ru} vs {wu} ({gap})"
    else:
        part = f"the attack it finds is statistically indistinguishable from the episode attack: uplift {ru} vs {wu} ({gap})"
    return f"{a}; {part}."


_RULE_NAME = {"first": "first non-declined candidate", "max": "largest amount among non-declined candidates"}


def _baseline_section(baselines: dict, results: dict, worst_key: str, conf: int,
                      error: str | None = None) -> list[str]:
    lines = ["", "## Conventional audit (single-transaction attack) vs episode audit", "",
             "Informational comparison: does not affect the gate verdict."]
    if error is not None:
        return lines + ["", f"The baseline could not be computed because of an error (verdict and exit code unchanged): `{error}`"]
    b = next(iter(baselines.values()))  # primary: the rule of single-transaction.yaml
    cmp = baseline_comparison(b, results, worst_key)
    tms = b.meta["single_threat_model"]["params"]
    ep = b.meta["episode_params"]
    lines += ["", f"| amount rule | evaded operations | evaded amount share (original amounts) "
              f"| amount of found operations / amount A | VER of replay as episode [{conf} %] | uplift of replay over "
              f"VER_clean [{conf} %] | queries |", "|---|---:|---:|---:|---|---|---:|"]
    for rule, x in baselines.items():
        mark = " (primary)" if x is b else ""
        lines.append(f"| `{rule}` – {_RULE_NAME[rule]}{mark} | {_pct(x.evaded_tx_share)} ({x.n_evaded} of {x.n_ops}) "
                     f"| {_pct(x.evaded_amount_share)} | {_num(x.chosen_amount_ratio, 3)} "
                     f"| {_pct(x.replay_ver[0])} [{_pct(x.replay_ver[1])}; {_pct(x.replay_ver[2])}] "
                     f"| {_pct(x.replay_uplift[0])} [{_pct(x.replay_uplift[1])}; {_pct(x.replay_uplift[2])}] "
                     f"| {x.meta['queries']} |")
    va, d = cmp["worst_ver_adv"], cmp["adv_minus_replay"]
    lines += [
        "",
        f"Episode audit: worst run `{worst_key}` – VER_adv {_pct(va[0])} [{_pct(va[1])}; {_pct(va[2])}], "
        f"uplift {_uplift(results[worst_key])}; mean uplift over {cmp['n_runs']} "
        f"attack × seed runs {_pct(cmp['uplift_mean_runs'])} (from {_pct(cmp['uplift_min_runs'])} to "
        f"{_pct(cmp['uplift_max_runs'])}). Paired difference VER_adv (`{worst_key}`) − VER of replay (`{b.amount_rule}`): "
        f"{_pct(d[0])} [{_pct(d[1])}; {_pct(d[2])}].",
        "",
        baseline_interpretation(b, cmp),
        "",
        f"- Conventional audit: each real fraud operation of the target episodes is attacked separately by random "
        f"search (up to {tms['single_tx_budget']} queries per operation, model decisions only): amount "
        f"{_num(tms['amount_min'], 2)}–{_num(tms['amount_max'], 2)} $, time shift up to {tms['max_shift_days']} days "
        f"(wrapped into the {ep['window_days']}-day episode window in whole days, so a candidate can land "
        f"earlier than the original operation: {b.meta['n_wrapped_earlier_than_original']} of {b.meta['queries']} "
        "queries), merchant. The card state is the recorded stream: the history before the episode and all other "
        "real operations of the episode up to the candidate's time; aggregates are not recomputed and there is no "
        "bank reaction. If there is no non-declined candidate, the original operation is kept (not evaded).",
        "- Advantage of the conventional audit: it queries the model on the state of the target card itself, "
        "whereas the episode attacker reconnoitres on spare cards only.",
        f"- Replay: the found operations of each episode are ordered by time (at least "
        f"{ep['min_gap_secs']} s apart) and executed as an episode with aggregates recomputed, "
        f"`max_declines` = {ep['max_declines']} and ρ = {_num(ep['rho'], 2)}; for rule `{b.amount_rule}` the "
        f"ρ·A threshold was reached in {b.meta['n_replay_success']} of {b.n_targets} episodes, cards blocked: "
        f"{b.meta['n_replay_blocked']}.",
        f"- The same {b.n_targets} target episodes as in the episode audit (seed {b.meta['seed']}); "
        "time: " + ", ".join(f"`{r}` {x.meta['wall_seconds']} s" for r, x in baselines.items()) + ".",
    ]
    return lines


def summary_markdown(checks: list[Check], results: dict, meta: dict, merchants=None,
                     baseline: dict | None = None, baseline_error: str | None = None) -> str:
    """English Markdown: verdict, checks, recommendations, attacks × seeds, baseline, the conventional
    (single-transaction) audit comparison when ``baseline`` (``{amount_rule: BaselineResult}``, the first
    entry is the primary rule) or ``baseline_error`` is given (informational), clean quality,
    the worst run's strategy, metadata. ``results`` is ``{run_key: AuditResult}``; ``meta`` holds the
    gate-level metadata (``gate``, ``gate_file``, ``wall_seconds``, …); per-run metadata comes from the runs."""
    ok = passed(checks)
    by = {c.name: c for c in checks}
    up = by["ver_uplift"]
    worst_key = up.worst
    w = results[worst_key]
    wm = w.meta
    conf = int(round(100 * wm["confidence"]))
    mean_a = float(wm["mean_amount"])  # δ$ = δ · mean A of the targets (R32)
    w_attack, w_seed = parse_run_key(worst_key)
    delta = up.threshold
    failed = [c for c in checks if not c.passed]

    lines = [f"# {'✅ ' + VERDICT_OK if ok else '❌ ' + VERDICT_FAIL} – episode-attack robustness gate", ""]
    if meta.get("policy") or meta.get("delta_source") == "--delta":
        lines += ["**Bank-chosen policy / bar – a demonstration of the gate parameters, not the gate of record.** The bar "
                  "δ′ was set by the bank (in the demonstration from the uplift measured for this policy and model, "
                  "plus the margin), so a pass confirms the measurement, it does not certify the model at δ = 10 %.", ""]
    if (meta.get("gate") or {}).get("positive_control"):
        lines += ["**Positive control (`replay`), not a gate verdict on the model:** it only checks the decision chain "
                  "(paired bootstrap of the uplift → worst run → verdict) on an uplift that is 0 by construction.", ""]
    if ok:
        lines.append("All gate checks passed.")
    else:
        lines.append("Failed: " + ", ".join(f"`{c.name}`" for c in failed) + ". What to do – see the "
                     "\"Recommendations\" section.")
    lines += ["",
              f"Worst case: attack `{w_attack}`"
              + (" (held-out, never generates hardening data)" if w_attack in HELD_OUT_ATTACKS else "")
              + f", attack seed {w_seed}; {_side(w)} upper {conf} % bound of the "
              f"VER_adv − VER_clean uplift {_pct(up.value)} against "
              + (f"δ = {_pct(delta)}." if delta is not None else "an unset δ."),
              ""]
    per_card = int(wm["n_target_cards"]) == int(w.n_targets)
    unit = ("per compromised card (one target episode per card)" if per_card else "per episode")
    head = (f"In dollars (approximately: share × mean episode amount A of the targets = {_num(mean_a, 2)} $; "
            f"δ$ = δ · mean A; median A {_num(float(wm['median_amount']), 2)} $ for information): ")
    if up.value is None or not math.isfinite(up.value):
        lines.append(head + "not computed – the VER uplift of the worst run is not finite.")
    elif delta is not None:
        lines.append(head + f"the attack gives the fraudster up to ~{_num(up.value * mean_a, 0)} $ {unit} against an "
                     f"allowed ~{_num(delta * mean_a, 0)} $.")
    else:
        lines.append(head + f"the attack gives the fraudster up to ~{_num(up.value * mean_a, 0)} $ {unit}; the allowed "
                     "amount is not set.")
    pol = meta.get("policy")
    if pol or meta.get("delta_source") == "--delta":
        what = []
        if pol:
            cor = ("corridor off" if pol["corridor"] == "off"
                   else f"corridor κ {pol['corridor']['kappa']:g}, q {pol['corridor']['q']:g}")
            what.append(f"bank policy **{pol['name']}** (`{pol['file']}`, SHA-256 `{pol['sha256']}`): "
                        f"max_declines {pol['max_declines']}, {cor}")
        if meta.get("delta_source") == "--delta" and delta is not None:
            what.append(f"bar δ′ = {_pct(delta)} set by the bank on the command line (`--delta`)")
        lines += ["", "Chosen by the bank, not by the gate: " + "; ".join(what) + "."]
    lines += ["", "## Checks", "", "| check | value | threshold | result |", "|---|---:|---|---|"]
    for c in checks:
        title = _TITLE[c.name].format(conf=conf, side=_side(w))
        show = c.name == "ver_uplift" or (not c.passed and c.name == "feasibility")
        where = f" (run `{c.worst}`)" if show and c.worst and len(results) > 1 else ""
        lines.append(f"| `{c.name}` – {title}{where} | {_value(c)} | {_threshold(c)} "
                     f"| {'✅ yes' if c.passed else '❌ no'} |")
    if "skew" not in by:
        lines += ["", "No feature-path skew report (`--skew-report`) was passed: the `skew` check was not run."]

    if failed:
        lines += ["", "## Recommendations", ""]
        for c in failed:
            lines.append(f"- `{c.name}`: {c.recommendation}")

    lines += ["", "## Attacks × seed", "",
              f"The target and spare episodes are the same for all runs (seed {wm['seed']}); only the "
              "attack's random stream changes. The gate takes the worst run."
              + (" The stepwise attack is held-out: a closed-loop policy that chooses each next operation after "
                 "the bank's decision on the previous one; it never generated hardening data, so it checks whether "
                 "robustness transfers to an attacker with a different search principle. With `max_declines = 1` "
                 "the first decline blocks the card, so the policy is effectively open-loop there: its strength "
                 "comes from a low-dimensional structured search space (geometric amounts, a fixed cadence, one "
                 "merchant category), not from adapting to declines."
                 if any(parse_run_key(k)[0] in HELD_OUT_ATTACKS for k in results) else ""), "",
              f"| attack | attack seed | VER_adv [{conf} %, two-sided] | uplift "
              + (f"(one-sided upper {conf} % bound)" if w.meta.get("one_sided") else f"[{conf} %]")
              + " | ASR | recon operations "
              "| strategy evaluations | |",
              "|---|---:|---|---|---|---:|---:|---|"]
    for key, r in results.items():
        a, s = parse_run_key(key)
        mark = "**⚠ worst**" if key == worst_key else ""
        if a in HELD_OUT_ATTACKS:
            a = f"{a} (held-out, never generates hardening data)"
        lines.append(f"| {a} | {s} | {_ci(r.ver_adv)} | {_uplift(r)} | {_pct(r.asr)} "
                     f"({r.meta['n_success']} of {r.n_targets}) | {r.recon_ops} | {r.meta['recon']['evals']} | {mark} |")

    lines += ["", "## Baseline and assumptions", "",
              f"- VER_clean (replay of the real fraud operations of the target episodes): {_ci(w.ver_clean)}; "
              f"without an attack the ρ·A threshold was reached in {wm['n_clean_success']} of {w.n_targets} episodes.",
              "- VER_clean replays the real fraud operations of each target episode without the card's legitimate "
              "operations inside the episode window (same for the attack): both runs (VER_clean and VER_adv) start "
              "from the card state before the episode begins, so VER_clean is a replay of the fraud operations "
              "only, not the bank's historical outcome.",
              f"- VER > 1 (because of the lower bound of the operation amount) in the worst run: adversarial episodes "
              f"{wm['n_ver_adv_gt_1']} (max {_num(wm['max_ver_adv'])}), real {wm['n_ver_clean_gt_1']} "
              f"(max {_num(wm['max_ver_clean'])}); values are not clipped.",
              f"- Target episodes: {w.n_targets} (cards: {wm['n_target_cards']}); {spares_text(wm, w.spare_cards)} "
              f"Reconnaissance budget {wm['recon_budget_ops']} operations per run.",
              f"- Intervals: percentile bootstrap over episodes ({wm['n_boot']} resamples); VER intervals two-sided; "
              + (f"uplift paired, one-sided upper {conf} % bound (the gate tests only uplift ≤ δ)"
                 if w.meta.get("one_sided") else "uplift paired, two-sided")
              + "; ASR bound – Clopper–Pearson."]

    if w.meta.get("corridor"):
        lines += ["", "## Spending corridor (guaranteed loss bound)", "", f"- {corridor_line(w.meta['corridor'])}",
                  "- Decision order: model decline → decline; model approval within the 24 h and 7-day corridors → "
                  "auto-approve; otherwise owner confirmation, which the fraudster fails (no second factor). Both "
                  "the adversarial and the clean leg run under the corridor. The bound (max over the episode window "
                  "of the 7-day corridor) holds for any attack strategy when the episode lasts at most 7 days; the "
                  "corridor profile is lagged by 7 days and built from passed operations without fraud labels, so the "
                  "bound is computed before the episode and no attacker operation changes it. B(h) bounds the "
                  "amount after h days (Proposition 2); with confirmed_extends (V2) the bounds are post-hoc, from "
                  "the owner's confirmations only.",
                  "- Bound violations per run: " + ", ".join(
                      f"{k} {r.meta['corridor']['n_bound_violations']}" for k, r in results.items()
                      if r.meta.get("corridor")) + "; headroom-bound violations per run: " + ", ".join(
                      f"{k} {r.meta['corridor'].get('n_headroom_violations', 0)}" for k, r in results.items()
                      if r.meta.get("corridor")) + "; B(h) block violations per run: " + ", ".join(
                      f"{k} {r.meta['corridor'].get('n_block_violations', 0)}" for k, r in results.items()
                      if r.meta.get("corridor")) + "."]

    if w.false_blocks:
        lines += ["", "## False blocks of legitimate cards", "", f"- {false_blocks_line(w.false_blocks)}"]

    if baseline or baseline_error is not None:
        lines += _baseline_section(baseline or {}, results, worst_key, conf, baseline_error)

    gate = meta.get("gate") or {}
    lines += ["", "## Model quality on clean data (at τ)", ""]
    if "target_fpr" in gate:
        lines += [f"Target per-operation FPR (`gate.yaml`): {_pct(gate['target_fpr'])}; τ = {wm['model']['tau']}.", ""]
    lines += ["| period | recall | FPR | precision | operations | fraud |", "|---|---:|---:|---:|---:|---:|"]
    for p in ("val", "test"):
        q = w.clean_quality[p]
        lines.append(f"| {p} | {_num(q['recall'], 4)} | {_num(q['fpr'], 4)} | {_num(q['precision'], 4)} "
                     f"| {q['n']} | {q['n_fraud']} |")

    b = candidate_dict(w.best, merchants)
    lines += ["", f"## Strategy of the worst run (`{worst_key}`)", ""]
    if b.get("type") == "replay":
        lines += [REPLAY_TEXT]
    elif b.get("type") == "stepwise_policy":
        lines += policy_markdown(b)
    else:
        lines += [f"Operations k = {b['k']}, total ≈ {_pct(b['u'])} of A; plan in order of planned time (day counted "
                  "from midnight of the episode start day), before wrapping into the window and spacing the "
                  "operations:", "",
                  "| # | share of A | day | hour | category |", "|---:|---:|---:|---:|---|"]
        for i, op in enumerate(b["ops"], 1):
            lines.append(f"| {i} | {_pct(op['share_of_A'])} | {op['day_index']} | {op['time_of_day']} "
                         f"| {op.get('category', '–')} |")

    run_walls = ", ".join(f"{k} {r.meta['wall_seconds']} s" for k, r in results.items())
    lines += ["", "## Metadata", "",
              f"- split seed {wm['seed']}; attack seeds: "
              + ", ".join(str(s) for s in sorted({parse_run_key(k)[1] for k in results}))
              + f"; git `{wm['git_sha']}` (uncommitted changes: {wm['git_dirty']})",
              f"- platform: {wm['platform']['platform']}; processes: {wm['n_jobs']}"
              + (f"; gate time {meta['wall_seconds']} s" if "wall_seconds" in meta else "")
              + (f" (including data and model loading {meta['wall_seconds_with_loading']} s)"
                 if "wall_seconds_with_loading" in meta else ""),
              f"- run times: {run_walls}",
              f"- model: `{wm['model']['dir']}`, τ = {wm['model']['tau']}, SHA-256 `model.json`: "
              f"`{wm['model']['model_json_sha256']}`",
              f"- threat model: `{wm['threat_model']['file']}`, SHA-256 `{wm['threat_model']['sha256']}`"]
    if meta.get("gate_file"):
        lines.append(f"- gate: `{meta['gate_file']}`, SHA-256 `{meta.get('gate_sha256')}`")
    if meta.get("delta_source") == "--delta":
        lines.append("- δ of the gate file is replaced by `--delta`")
    if meta.get("policy"):
        lines.append(f"- bank policy: `{meta['policy']['file']}`, SHA-256 `{meta['policy']['sha256']}`; it overrides "
                     "max_declines and the corridor κ and q of the threat file")
    if "model_target_fpr" in meta:
        mt = meta["model_target_fpr"]
        lines.append("- the model's target_fpr is not given – the check was skipped" if mt is None
                     else f"- the model's target_fpr (`model.json`): {mt}, matches `gate.yaml`")
    if gate:
        lines.append("- gate parameters: " + ", ".join(f"{k} = {v}" for k, v in gate.items()))
    lines += ["- versions: " + ", ".join(f"{k} {v}" for k, v in wm["versions"].items()), ""]
    return "\n".join(lines)


def error_summary(message: str) -> str:
    """Minimal summary when the audit could not run (configuration or artifact error): CI must show why."""
    return (f"# ❌ {VERDICT_FAIL} – configuration/artifact error\n\n"
            "The audit did not run, so the gate is closed (exit 2). Reason:\n\n"
            f"```\n{message}\n```\n\n"
            "Fix the configuration (`gate.yaml`, threat model, skew report) or the model artifacts (`models/`, "
            "SHA-256 check in `manifest.json`) and rerun the audit; do not bypass the integrity check.\n")


def write_error_outputs(out_dir: Path | str, message: str) -> str:
    """``out_dir/audit.json`` (verdict, error) and ``out_dir/summary.md`` for a run that could not start."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = error_summary(message)
    (out / "audit.json").write_text(json.dumps({"verdict": VERDICT_FAIL, "passed": False, "error": message},
                                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(md, encoding="utf-8")
    return md


def _baseline_json(baseline: dict | None, results: dict, worst: str, error: str | None):
    if error is not None:
        return {"error": error}
    if not baseline:
        return None
    primary = next(iter(baseline))
    cmp = baseline_comparison(baseline[primary], results, worst)
    return {"primary": primary, "rules": {k: baseline_to_json(v) for k, v in baseline.items()},
            "comparison": cmp}


def write_gate_outputs(out_dir: Path | str, cfg: GateConfig, checks: list[Check], results: dict, meta: dict,
                       merchants=None, skew: dict | None = None, baseline: dict | None = None,
                       baseline_error: str | None = None) -> dict:
    """``out_dir/audit.json`` (verdict, checks, gate, every run via ``to_json``, skew, the single-transaction
    ``baseline`` ``{"primary", "rules": {rule: …}, "comparison"}``, ``{"error": …}`` or null) and
    ``out_dir/summary.md``. The baseline never changes the verdict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = passed(checks)
    meta = dict(meta, gate=dataclasses.asdict(cfg))
    worst = next(c.worst for c in checks if c.name == "ver_uplift")
    try:  # informational: any failure of the comparison becomes a note, never a changed verdict
        baseline_json = _baseline_json(baseline, results, worst, baseline_error)
        summary = summary_markdown(checks, results, meta, merchants, baseline, baseline_error)
    except Exception as e:  # noqa: BLE001
        baseline_error = f"{type(e).__name__}: {e}"
        baseline_json = _baseline_json(None, results, worst, baseline_error)
        summary = summary_markdown(checks, results, meta, merchants, None, baseline_error)
    data = {"verdict": VERDICT_OK if ok else VERDICT_FAIL, "passed": ok,
            "checks": [dataclasses.asdict(c) for c in checks], "gate": dataclasses.asdict(cfg),
            "worst_run": worst, "runs": {k: to_json(r, merchants) for k, r in results.items()},
            "skew": skew, "baseline": baseline_json, "meta": meta}
    (out / "audit.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary, encoding="utf-8")
    return data
