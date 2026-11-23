import json

import pytest

from fraudaudit.budget_sensitivity import SEED_SPREAD, compare, markdown

KEY = "fpr=0.01|md3|corridor=on|kappa=1.25|q=0.98"


def _write(d, label, budget, u, key=KEY, worst="stepwise/3", sha="abc"):
    runs = {f"{a}/{s}": {"uplift_upper": u / 2} for a in ("es", "random", "stepwise") for s in (1, 2, 3)}
    runs[worst] = {"uplift_upper": u}
    p = d / label
    p.mkdir(parents=True)
    (p / "policy-sweep.json").write_text(json.dumps({
        "model_json_sha256": sha, "gate": {"delta": 0.1}, "seed": 42, "own_tau": 0.5,
        "meta": {"recon_budget_ops": budget, "threat_sha256": "t", "corridor_threat_sha256": "c"},
        "cells": [{"key": key, "U": u, "U_usd": u * 5000, "worst_run": worst, "runs": runs, "recon_budget_ops": budget,
                   "tau": 0.5, "max_declines": 3, "tau_after_decline": None, "corridor_params": {"q": 0.98},
                   "target_fpr": 0.01}]}))


def _setup(tmp_path, u8, u20):
    _write(tmp_path / "r", "m", 2000, 0.20)
    _write(tmp_path / "b8", "m", 8000, u8)
    _write(tmp_path / "b20", "m", 20000, u20)
    return tmp_path / "r", {8000: tmp_path / "b8", 20000: tmp_path / "b20"}


def test_h_supported_below_the_seed_spread(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.29)
    res = compare(ref, b, ["m"])
    assert res["verdict"]["H"] == "supported"
    assert res["verdict"]["max_dU"] == pytest.approx(0.09) and res["labels"]["m"][KEY][20000]["dU"] == pytest.approx(0.09)


def test_h_not_supported_at_or_above_the_seed_spread(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.20 + SEED_SPREAD)
    res = compare(ref, b, ["m"])
    assert res["verdict"]["H"] == "not supported"


def test_a_decrease_counts_as_supported(tmp_path):
    ref, b = _setup(tmp_path, 0.15, 0.10)
    assert compare(ref, b, ["m"])["verdict"]["H"] == "supported"


def test_budget_mismatch_in_a_cell_is_refused(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.29)
    _write(tmp_path / "wrong", "m", 2000, 0.3)
    with pytest.raises(ValueError, match="cell budget"):
        compare(ref, {8000: tmp_path / "wrong", 20000: b[20000]}, ["m"])


def test_reference_cell_must_be_2000(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.29)
    _write(tmp_path / "r5", "m", 5000, 0.2)
    with pytest.raises(ValueError, match="reference cell budget"):
        compare(tmp_path / "r5", b, ["m"])


def test_other_model_between_reference_and_budget_run_is_refused(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.29)
    _write(tmp_path / "b20x", "m", 20000, 0.29, sha="other")
    with pytest.raises(ValueError, match="model_json_sha256"):
        compare(ref, {8000: b[8000], 20000: tmp_path / "b20x"}, ["m"])


def test_20000_is_required(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.29)
    with pytest.raises(ValueError, match="20000"):
        compare(ref, {8000: b[8000]}, ["m"])


def test_two_labels(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.29)
    for d, bud, u in ((ref, 2000, 0.2), (b[8000], 8000, 0.22), (b[20000], 20000, 0.5)):
        _write(d, "m2", bud, u)
    res = compare(ref, b, ["m", "m2"])
    assert res["verdict"]["H"] == "not supported" and res["verdict"]["cells"] == 2


def test_no_common_cell_is_refused(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.29)
    _write(tmp_path / "other", "m", 8000, 0.3, key="other")
    with pytest.raises(ValueError, match="no cell"):
        compare(ref, {8000: tmp_path / "other", 20000: b[20000]}, ["m"])


def test_markdown_mentions_verdict_and_attacks(tmp_path):
    ref, b = _setup(tmp_path, 0.25, 0.29)
    md = markdown(compare(ref, b, ["m"]))
    assert "supported" in md and "stepwise/3" in md and "| m |" in md and "+9.0" in md


def test_cli_budget_sensitivity(tmp_path, capsys):
    from fraudaudit.cli import main
    ref, b = _setup(tmp_path, 0.25, 0.29)
    out = tmp_path / "o"
    code = main(["budget-sensitivity", "--reference", str(ref), "--budget", f"8000={b[8000]}",
                 "--budget", f"20000={b[20000]}", "--labels", "m", "--out", str(out)])
    assert code == 0 and json.loads((out / "budget-sensitivity.json").read_text())["verdict"]["H"] == "supported"
    assert (out / "budget-sensitivity.md").exists()
    assert main(["budget-sensitivity", "--reference", str(ref), "--budget", "oops", "--labels", "m", "--out", str(out)]) == 2
    assert "B=DIR" in capsys.readouterr().err


def test_cli_duplicate_budget_refused(tmp_path, capsys):
    from fraudaudit.cli import main
    ref, b = _setup(tmp_path, 0.25, 0.29)
    code = main(["budget-sensitivity", "--reference", str(ref), "--budget", f"20000={b[20000]}",
                 "--budget", f"20000={b[20000]}", "--labels", "m", "--out", str(tmp_path / "o")])
    assert code == 2 and "twice" in capsys.readouterr().err


# --- Task 27f (R46: H50 on budget 50 000) and 27e (R45 (4): seeds 4-9, descriptive) --------------------------------


def _write_seeds(d, label, budget, ups, seeds, key=KEY, sha="abc"):
    """``ups``: {(attack, seed): upper bound}; missing runs get 0.01."""
    runs = {f"{a}/{s}": {"uplift_upper": ups.get((a, s), 0.01)} for a in ("es", "random", "stepwise") for s in seeds}
    worst = max(runs, key=lambda k: runs[k]["uplift_upper"])
    u = runs[worst]["uplift_upper"]
    p = d / label
    p.mkdir(parents=True)
    (p / "policy-sweep.json").write_text(json.dumps({
        "model_json_sha256": sha, "gate": {"delta": 0.1, "attack_seeds": list(seeds)}, "seed": 42, "own_tau": 0.5,
        "meta": {"threat_sha256": "t", "corridor_threat_sha256": "c"},
        "cells": [{"key": key, "U": u, "U_usd": u * 5000, "worst_run": worst, "runs": runs, "recon_budget_ops": budget,
                   "tau": 0.5, "max_declines": 3, "tau_after_decline": None, "corridor_params": {"q": 0.98},
                   "target_fpr": 0.01}]}))


def test_hypothesis_budget_50000_decides_h50_and_needs_that_budget(tmp_path):
    seeds = (1, 2, 3)
    _write_seeds(tmp_path / "r", "m", 2000, {("stepwise", 1): 0.20}, seeds)
    _write_seeds(tmp_path / "b50", "m", 50000, {("stepwise", 2): 0.20 + SEED_SPREAD + 1e-9}, seeds)
    res = compare(tmp_path / "r", {50000: tmp_path / "b50"}, ["m"], hypothesis_budget=50000)
    assert res["verdict"]["budget"] == 50000 and res["verdict"]["H"] == "not supported"
    assert res["verdict"]["max_dU"] == pytest.approx(SEED_SPREAD)
    md = markdown(res)
    assert "budget 50000: max dU" in md and "seeds 1-3" in md
    with pytest.raises(ValueError, match="budget 20000 is required"):  # H on a budget that was not run
        compare(tmp_path / "r", {50000: tmp_path / "b50"}, ["m"], hypothesis_budget=20000)


def test_descriptive_mode_has_no_verdict_and_counts_seeds(tmp_path):
    seeds = (4, 5, 6, 7, 8, 9)
    ref = {("stepwise", s): 0.20 for s in seeds}
    b20 = {("stepwise", s): 0.20 for s in seeds}
    b20[("es", 5)] = 0.20 + SEED_SPREAD + 1e-9  # seed 5: at the spread -> counted (>=)
    b20[("random", 7)] = 0.20 + SEED_SPREAD - 0.001  # seed 7: just below -> not counted
    b20[("stepwise", 9)] = 0.05  # seed 9: a drop
    _write_seeds(tmp_path / "r", "g", 2000, ref, seeds)
    _write_seeds(tmp_path / "b20", "g", 20000, b20, seeds)
    res = compare(tmp_path / "r", {20000: tmp_path / "b20"}, ["g"], hypothesis_budget=None)
    assert res["verdict"] is None
    row = res["labels"]["g"][KEY]
    assert row[2000]["U_s"] == {str(s): pytest.approx(0.20) for s in seeds}
    assert row[20000]["dU_s"]["5"] == pytest.approx(SEED_SPREAD) and row[20000]["dU_s"]["9"] == pytest.approx(-0.15)
    assert row[20000]["seeds_dU_ge_spread"] == [{"seed": 5, "dU": pytest.approx(SEED_SPREAD), "attack": "es"}]
    assert row[20000]["n_seeds_dU_ge_spread"] == 1 and row[20000]["n_seeds"] == 6
    md = markdown(res)
    assert "supported" not in md and "descriptive" in md and "seeds 4-9" in md
    assert "1 of 6" in md


def test_seeds_must_match_between_reference_and_budget(tmp_path):
    _write_seeds(tmp_path / "r", "g", 2000, {}, (4, 5, 6))
    _write_seeds(tmp_path / "b20", "g", 20000, {}, (1, 2, 3))
    with pytest.raises(ValueError, match="gate"):
        compare(tmp_path / "r", {20000: tmp_path / "b20"}, ["g"], hypothesis_budget=None)


def test_cli_budget_sensitivity_descriptive_end_to_end(tmp_path):
    from fraudaudit.cli import main
    seeds = (4, 5, 6, 7, 8, 9)
    _write_seeds(tmp_path / "r", "g", 2000, {}, seeds)
    _write_seeds(tmp_path / "b20", "g", 20000, {("stepwise", 4): 0.5}, seeds)
    rc = main(["budget-sensitivity", "--reference", str(tmp_path / "r"), "--budget", f"20000={tmp_path / 'b20'}",
               "--labels", "g", "--hypothesis-budget", "none", "--out", str(tmp_path / "o")])
    assert rc == 0
    j = json.loads((tmp_path / "o" / "budget-sensitivity.json").read_text())
    assert j["verdict"] is None and j["labels"]["g"][KEY]["20000"]["n_seeds_dU_ge_spread"] == 1
    md = (tmp_path / "o" / "budget-sensitivity.md").read_text()
    assert "Descriptive only" in md and "supported" not in md
