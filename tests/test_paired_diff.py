"""R57 (5): paired difference of two legs of a stored audit.json (fraudaudit.paired_diff)."""
import copy
import json

import numpy as np
import pytest

from fraudaudit.cli import main
from fraudaudit.metrics import uplift_ci
from fraudaudit.paired_diff import paired_diff, subset_comparison, word, worst_run

N = 40
META = {"confidence": 0.95, "n_boot": 500, "seed": 7}


def _audit() -> dict:
    rng = np.random.default_rng(0)
    keys = [(f"c{i}", 1000 + i) for i in range(N)]
    vers = {"random/1": 0.20, "es/1": 0.45, "es/2": 0.30, "stepwise/1": 0.80}
    uppers = {"random/1": 0.25, "es/1": 0.50, "es/2": 0.35, "stepwise/1": 0.85}
    runs = {}
    for k, v in vers.items():
        ver = np.clip(v + rng.normal(0, 0.05, N), 0, 1)
        runs[k] = {"ver_adv": {"mean": float(ver.mean()), "lower": 0.0, "upper": 1.0},
                   "uplift": {"mean": v, "lower": 0.0, "upper": uppers[k]},
                   "per_target": [{"card": c, "start": s, "ver_adv": float(x)} for (c, s), x in zip(keys, ver)]}
    rules = {}
    for r, v in (("first", 0.40), ("max", 0.60)):
        rep = np.clip(v + rng.normal(0, 0.05, N), 0, 1)
        rules[r] = {"replay_ver": {"mean": float(rep.mean()), "lower": 0.0, "upper": 1.0}, "meta": dict(META),
                    "per_target": [{"card": c, "start": s, "ver_replay": float(x)} for (c, s), x in zip(keys, rep)]}
    a = np.array([t["ver_adv"] for t in runs["stepwise/1"]["per_target"]])
    f = np.array([t["ver_replay"] for t in rules["first"]["per_target"]])
    stored = uplift_ci(a, f, META["confidence"], META["n_boot"], META["seed"])
    return {"worst_run": "stepwise/1", "runs": runs, "meta": {"git_sha": "abc"},
            "baseline": {"primary": "first", "rules": rules,
                         "comparison": {"worst_run": "stepwise/1", "adv_minus_replay": list(stored)}}}


def test_worst_run_follows_the_gate_rule_within_the_subset():
    a = _audit()
    assert worst_run(a) == "stepwise/1"
    assert worst_run(a, ["random", "es"]) == "es/1"
    a["runs"]["es/2"]["uplift"]["upper"] = 0.50  # tie: first in run order
    assert worst_run(a, ["random", "es"]) == "es/1"
    with pytest.raises(ValueError):
        worst_run(a, ["nope"])


def test_paired_diff_is_the_stored_bootstrap_and_sign_follows_the_order():
    a = _audit()
    d = paired_diff(a, "run:stepwise/1", "replay:first")
    assert (d["mean"], d["lower"], d["upper"]) == tuple(a["baseline"]["comparison"]["adv_minus_replay"])
    r = paired_diff(a, "replay:first", "run:stepwise/1")
    assert r["mean"] == pytest.approx(-d["mean"])
    assert d["n_targets"] == N and d["n_boot"] == META["n_boot"]


def test_subset_comparison_both_differences():
    a = _audit()
    res = subset_comparison(a, ["random", "es"])
    assert res["worst_run"] == "es/1"
    i, ii = res["adv_minus_replay"], res["replay_max_minus_adv"]
    assert (i["a"], i["b"]) == ("run:es/1", "replay:first")
    assert (ii["a"], ii["b"]) == ("replay:max", "run:es/1")
    assert i["mean"] == pytest.approx(0.05, abs=0.03) and ii["mean"] == pytest.approx(0.15, abs=0.03)
    assert ii["word"] == "exceeds"


def test_stored_mismatch_or_other_targets_refused():
    a = _audit()
    bad = copy.deepcopy(a)
    bad["baseline"]["comparison"]["adv_minus_replay"][0] += 1e-12
    with pytest.raises(ValueError, match="stored"):
        subset_comparison(bad, ["random", "es"])
    bad = copy.deepcopy(a)
    bad["runs"]["es/1"]["per_target"][3]["card"] = "other"
    with pytest.raises(ValueError, match="different target episodes"):
        subset_comparison(bad, ["random", "es"])
    bad = copy.deepcopy(a)
    bad["baseline"]["rules"]["max"]["meta"]["n_boot"] = 1
    with pytest.raises(ValueError, match="bootstrap parameters"):
        subset_comparison(bad, ["random", "es"])


def test_word_rule():
    assert word({"lower": 0.01, "upper": 0.1}, "exceeds") == "exceeds"
    assert word({"lower": -0.01, "upper": 0.1}, "exceeds") == "on par"
    assert word({"lower": -0.1, "upper": -0.01}, "exceeds") == "below"


def test_cli_writes_json_and_md(tmp_path):
    p = tmp_path / "audit.json"
    p.write_text(json.dumps(_audit()))
    out = tmp_path / "res" / "paired"
    assert main(["paired-diff", "--audit", "base", str(p), "--attacks", "random", "es", "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    assert res["audits"]["base"]["worst_run"] == "es/1"
    md = out.with_suffix(".md").read_text()
    assert "chosen after the fact" in md and "`es/1`" in md
