import json
from pathlib import Path

import pytest

from fraudaudit import cli
from fraudaudit.budget_figure import cell_name, collect, figure, markdown, positions

KEY = "fpr=0.01|md3|corridor=on|kappa=1.25|q=0.98"
KEY_ESC = "fpr=0.01|md3+esc|corridor=on|kappa=1.25|q=0.98"


def _row(ref, us):
    row = {"2000": {"U": ref}}
    for b, u in us.items():
        row[str(b)] = {"U": u, "dU": u - ref}
    return row


def _write(path, labels, budgets, seeds=(1, 2, 3), verdict=None, with_seeds=True, spread=0.096):
    d = {"seed_spread": spread, "reference_budget": 2000, "budgets": budgets, "labels": labels, "verdict": verdict}
    if with_seeds:
        d["attack_seeds"] = None if seeds is None else list(seeds)
    path.write_text(json.dumps(d))
    return path


def _files(tmp_path):
    g19 = _write(tmp_path / "g19.json", {"m3-hard": {KEY: _row(0.219, {8000: 0.305, 20000: 0.202})}},
                 [8000, 20000], with_seeds=False)  # Task 19c files carry no attack_seeds
    g24 = _write(tmp_path / "g24.json", {"xgb-hard": {KEY: _row(0.388, {8000: 0.306, 20000: 0.392})},
                                         "gru-hard": {KEY: _row(0.211, {8000: 0.241, 20000: 0.191}),
                                                      KEY_ESC: _row(0.179, {8000: 0.139, 20000: 0.354})}},
                 [8000, 20000], with_seeds=False)
    g50g = _write(tmp_path / "g50g.json", {"gru-hard": {KEY: _row(0.211, {50000: 0.278}),
                                                        KEY_ESC: _row(0.179, {50000: 0.258})}}, [50000])
    g50x = _write(tmp_path / "g50x.json", {"xgb-hard": {KEY: _row(0.388, {50000: 0.375})}}, [50000])
    g50m = _write(tmp_path / "g50m.json", {"m3-hard": {KEY: _row(0.219, {50000: 0.298})}}, [50000])
    fresh = _write(tmp_path / "fresh.json", {"gru-hard": {KEY: _row(0.190, {20000: 0.298, 50000: 0.296}),
                                                          KEY_ESC: _row(0.195, {20000: 0.250, 50000: 0.289})}},
                   [20000, 50000], seeds=range(4, 10))
    return [g24, g19, g50g, g50x, g50m], [fresh]


def test_collect_groups_by_family_and_keeps_every_budget(tmp_path):
    gate, fresh = _files(tmp_path)
    res = collect(gate, fresh)
    assert [(c["family"], c["label"], c["key"]) for c in res["cells"]] == [
        ("MLP", "m3-hard", KEY), ("XGB", "xgb-hard", KEY), ("GRU", "gru-hard", KEY), ("GRU", "gru-hard", KEY_ESC)]
    gru_esc = res["cells"][3]
    assert gru_esc["dU"] == pytest.approx({8000: -0.040, 20000: 0.175, 50000: 0.079})
    assert gru_esc["fresh_dU"] == pytest.approx({20000: 0.055, 50000: 0.094})
    assert res["cells"][1]["dU"] == pytest.approx({8000: -0.082, 20000: 0.004, 50000: -0.013})
    assert res["cells"][1]["fresh_dU"] == {} and "ref_runs" not in res["cells"][1]
    assert res["budgets"] == [8000, 20000, 50000] and res["fresh_budgets"] == [20000, 50000]
    assert res["fresh_seeds"] == [4, 5, 6, 7, 8, 9] and res["other_commit_budgets"] == [8000]


def test_reference_must_be_bitwise_equal_across_gate_files(tmp_path):
    gate, fresh = _files(tmp_path)
    _write(gate[4], {"m3-hard": {KEY: _row(0.2190000001, {50000: 0.298})}}, [50000])
    with pytest.raises(ValueError, match="reference \\(U 0.2190000001"):
        collect(gate, fresh)


def test_same_budget_twice_is_refused(tmp_path):
    gate, fresh = _files(tmp_path)
    with pytest.raises(ValueError, match="given twice"):
        collect(gate + [gate[4]], fresh)


def test_stored_du_must_match_the_us(tmp_path):
    gate, fresh = _files(tmp_path)
    row = _row(0.219, {50000: 0.298})
    row["50000"]["dU"] = 0.1
    _write(gate[4], {"m3-hard": {KEY: row}}, [50000])
    with pytest.raises(ValueError, match="stored dU"):
        collect(gate, fresh)


def test_gate_file_on_other_seeds_is_refused(tmp_path):
    gate, fresh = _files(tmp_path)
    with pytest.raises(ValueError, match="gate file has seeds"):
        collect(gate + fresh, [])


@pytest.mark.parametrize("seeds, verdict, match", [((3, 4, 5), None, "disjoint"), ((4, 5), {"H": "supported"}, "descriptive")])
def test_fresh_file_must_be_on_fresh_seeds_without_verdict(tmp_path, seeds, verdict, match):
    gate, _ = _files(tmp_path)
    bad = _write(tmp_path / "bad.json", {"gru-hard": {KEY: _row(0.19, {20000: 0.3})}}, [20000], seeds=seeds,
                 verdict=verdict)
    with pytest.raises(ValueError, match=match):
        collect(gate, [bad])


def test_fresh_cell_must_be_a_gate_cell(tmp_path):
    gate, _ = _files(tmp_path)
    bad = _write(tmp_path / "bad.json", {"gru-base": {KEY: _row(0.38, {20000: 0.45})}}, [20000], seeds=range(4, 10))
    with pytest.raises(ValueError, match="not among the gate cells"):
        collect(gate, [bad])


def test_positions_leave_a_gap_between_families(tmp_path):
    res = collect(*_files(tmp_path))
    xs, spans = positions(res)
    assert xs == pytest.approx([0.0, 1.8, 3.6, 4.6])
    assert [s[0] for s in spans] == ["MLP", "XGB", "GRU"] and spans[2][1:] == pytest.approx((3.6, 4.6))


def test_cell_name_is_label_and_reaction_when_the_rest_is_common(tmp_path):
    res = collect(*_files(tmp_path))
    assert [cell_name(res, c) for c in res["cells"]] == ["m3-hard md3", "xgb-hard md3", "gru-hard md3",
                                                         "gru-hard md3+esc"]


def test_markdown_marks_the_other_commit_and_missing_values(tmp_path):
    md = markdown(collect(*_files(tmp_path)))
    assert "dU 8000 (other commit)" in md and "dU 20000, seeds 4-9" in md
    assert f"| GRU | `gru-hard {KEY_ESC}` | -4.0 | +17.5 | +7.9 | +5.5 | +9.4 |" in md
    assert f"| XGB | `xgb-hard {KEY}` | -8.2 | +0.4 | -1.3 | – | – |" in md
    assert md.startswith("# dU of the budget sensitivity, 4 cells")


def test_figure_writes_json_markdown_and_both_languages(tmp_path):
    gate, fresh = _files(tmp_path)
    out = tmp_path / "out"
    res = figure(gate, fresh, out, meta={"git_sha": "x"})
    for f in ("budget-du.json", "budget-du.md", "budget-du-en.png", "budget-du-uk.png"):
        assert (out / f).stat().st_size > 0
    assert json.loads((out / "budget-du.json").read_text())["meta"] == {"git_sha": "x"} and len(res["cells"]) == 4


def test_cli_budget_figure(tmp_path, capsys):
    gate, fresh = _files(tmp_path)
    out = tmp_path / "cli"
    argv = ["budget-figure", *sum((["--gate", str(p)] for p in gate), []), "--fresh", str(fresh[0]), "--out", str(out)]
    assert cli.main(argv) == 0
    assert "# dU of the budget sensitivity" in capsys.readouterr().out
    assert (out / "budget-du-uk.png").exists()
    assert cli.main(["budget-figure", "--gate", str(gate[0]), "--gate", str(gate[0]), "--out", str(out)]) == 2


def test_a_missing_gate_file_is_refused_not_drawn_incomplete(tmp_path):
    gate, fresh = _files(tmp_path)
    with pytest.raises(ValueError, match="a gate file missing"):
        collect([g for g in gate if g.name != "g50x.json"], fresh)


def test_gate_file_with_null_seeds_is_refused(tmp_path):
    gate, fresh = _files(tmp_path)
    _write(gate[4], {"m3-hard": {KEY: _row(0.219, {50000: 0.298})}}, [50000], seeds=None)
    with pytest.raises(ValueError, match="gate file has seeds None"):
        collect(gate, fresh)


def test_reference_runs_must_be_equal_too(tmp_path):
    gate, fresh = _files(tmp_path)
    row = _row(0.219, {50000: 0.298})
    row["2000"]["runs"] = {"stepwise/1": 0.219}
    _write(gate[4], {"m3-hard": {KEY: row}}, [50000])
    with pytest.raises(ValueError, match="reference"):
        collect(gate, fresh)


def test_fresh_files_must_share_their_seeds(tmp_path):
    gate, fresh = _files(tmp_path)
    other = _write(tmp_path / "f2.json", {"gru-base": {}}, [8000], seeds=range(10, 16))
    with pytest.raises(ValueError, match="share their seeds"):
        collect(gate, fresh + [other])


def test_unknown_family_prefix_and_other_seed_spread_are_refused(tmp_path):
    gate, fresh = _files(tmp_path)
    bad = _write(tmp_path / "bad.json", {"lgbm-hard": {KEY: _row(0.3, {8000: 0.3, 20000: 0.3, 50000: 0.3})}},
                 [8000, 20000, 50000])
    with pytest.raises(ValueError, match="unknown family prefix"):
        collect(gate + [bad], fresh)
    _write(gate[4], {"m3-hard": {KEY: _row(0.219, {50000: 0.298})}}, [50000], spread=0.1)
    with pytest.raises(ValueError, match="seed spread"):
        collect(gate, fresh)


REAL = Path("out")
REAL_GATE = ["t19c/summary", "t19c-t24/summary-xgb", "t19c-t24/summary-gru", "t27-chain1/t27f/summary-mlp",
             "t27-chain1/t27f/summary-xgb", "t27-chain1/t27f/summary-gru"]
REAL_FRESH = "t27-chain1/t27e/summary"


@pytest.mark.skipif(not all((REAL / d / "budget-sensitivity.json").exists() for d in [*REAL_GATE, REAL_FRESH]),
                    reason="local run outputs absent")
def test_real_outputs_pass_the_checks():
    res = collect([REAL / d / "budget-sensitivity.json" for d in REAL_GATE],
                  [REAL / REAL_FRESH / "budget-sensitivity.json"])
    assert len(res["cells"]) == 12 and res["budgets"] == [8000, 20000, 50000]
    assert [c["family"] for c in res["cells"]] == ["MLP"] * 4 + ["XGB"] * 4 + ["GRU"] * 4
    assert sum(1 for c in res["cells"] if c["fresh_dU"]) == 4
