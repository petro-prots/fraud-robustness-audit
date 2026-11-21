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
