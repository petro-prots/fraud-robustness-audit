"""Task 25: the 8-model summary (Spearman, policy vs model on 8 models, Pareto front, cost of hardening)."""
import hashlib
import json
from pathlib import Path

import pytest

from fraudaudit import models_summary as msum
from fraudaudit import policy_summary as ps
from fraudaudit.policy_sweep import cell_key
from fraudaudit.spearman import critical_rho, leave_one_out, spearman_exact

LABELS = ["m3-base", "m3-hard", "m2-base", "m2-hard", "xgb-base", "xgb-hard", "gru-base", "gru-hard"]


# --- synthetic inputs ---------------------------------------------------------------------------------------------


def _r19_entry(ver_md, rel, diff, lower=None):
    lo = rel - 0.01 if lower is None else lower
    return {"ver_inf": ver_md * (1 + rel), "ver_md": ver_md, "rel": rel, "lower": lo, "upper": rel + 0.02,
            "diff": diff, "diff_lower": diff - 0.01, "diff_upper": diff + 0.01, "n_boot": 2000, "n_dropped": 0}


def _r19(md3, md1=None):
    """``{label: {adv_es: {md1, md2, md3}, clean: ...}}`` from ``md3`` = {label: (ver_md, rel, diff)}."""
    out = {}
    for lab, (v, r, d) in md3.items():
        v1, r1, d1 = (md1 or {}).get(lab, (v, r, d))
        legs = {"md1": _r19_entry(v1, r1, d1), "md2": _r19_entry(v, r, d), "md3": _r19_entry(v, r, d)}
        out[lab] = {"adv_es": legs, "clean": {k: dict(e) for k, e in legs.items()}}
    return out


def _cell(fpr, reaction, corridor, U, recall=0.9, test_fpr=0.009, targets=None, git="abc1234567"):
    return {"key": cell_key(fpr, reaction, corridor), "target_fpr": fpr, "reaction": reaction, "corridor": corridor,
            "tau": 1 - fpr, "U": U, "test_recall": recall, "test_fpr": test_fpr,
            "targets": list(range(5)) if targets is None else targets,
            "runs": {"es/1": {"uplift_upper": U}, "es/2": {"uplift_upper": U - 0.02}},
            "provenance": {"git_sha": git, "git_dirty": False, "host_machine": "box"}}


def toy_sweep(label, offset=0.0, recall=0.9, test_fpr=0.009, targets=None, git="abc1234567", model_dir="m"):
    cells = []
    for i, f in enumerate((0.005, 0.01, 0.02)):
        for r in ("md1", "md3", "md3+esc"):
            for c in ("off", "on"):
                U = 0.6 + offset - 0.05 * i - (0.01 if c == "on" else 0) - (0.04 if r == "md3+esc" else 0)
                cells.append(_cell(f, r, c, U, recall if f == 0.01 else recall + 0.001, test_fpr, targets, git))
    for r in ("mdinf", "md2"):
        cells.append(_cell(0.01, r, "off", 0.7 + offset, recall, test_fpr, targets, git))
    return {"label": label, "role": "grid", "model_dir": model_dir, "model_json_sha256": f"sha-{label}",
            "own_target_fpr": 0.01, "own_tau": 0.5, "esc_fpr": 0.05, "tau_esc": 0.1, "seed": 42,
            "taus": {"0.01": 0.5}, "gate": {"delta": 0.1, "attack_seeds": [1, 2], "attacks": ["es"]},
            "corridor_params": {"q": 0.99, "daily": {"kappa": 2.0}}, "mean_amount": 100.0, "cells": cells,
            "friction": {"false_blocks": {}, "burden": {}},
            "meta": {"gate_sha256": "g", "threat_sha256": "t", "corridor_threat_sha256": "c", "git_sha": git}}


def toy_grid8(**over):
    grid = {}
    for i, lab in enumerate(LABELS):
        kw = dict(offset=0.01 * i, recall=0.95 - 0.01 * i)
        kw.update(over.get(lab, {}))
        grid[lab] = ps.normalize(toy_sweep(lab, **kw), f"{lab}.json")
    return grid


# --- (a) Spearman ---------------------------------------------------------------------------------------------------


def test_spearman_section_picks_md3_rel_diff_and_ver_md():
    md3 = {lab: (0.5 + 0.05 * i, 0.30 - 0.03 * i, 0.01 * ((i * 3) % 8)) for i, lab in enumerate(LABELS)}
    md1 = {lab: (0.2 + 0.01 * i, 0.10 + 0.02 * i, 0.05) for i, lab in enumerate(LABELS)}
    md1[LABELS[0]] = (0.2, 0.10, 0.06)  # not constant
    r19 = _r19(md3, md1)
    res = msum.spearman_section(r19, LABELS)
    assert res["n"] == 8 and res["critical_rho"] == pytest.approx(critical_rho(8))
    assert res["critical_rho"] == pytest.approx(0.738, abs=5e-4)
    h = res["tests"]["H_rel"]
    xs = [0.5 + 0.05 * i for i in range(8)]
    ys = [0.30 - 0.03 * i for i in range(8)]
    assert [p["label"] for p in h["points"]] == LABELS
    assert [p["x"] for p in h["points"]] == pytest.approx(xs) and [p["y"] for p in h["points"]] == pytest.approx(ys)
    ref = spearman_exact(xs, ys)
    assert h["rho"] == pytest.approx(-1.0) and h["p"] == pytest.approx(ref.p) and h["n_permutations"] == 40320
    assert h["leave_one_out"] == {lab: pytest.approx(v) for lab, v in zip(LABELS, leave_one_out(xs, ys))}
    assert h["loo_min"] == pytest.approx(-1.0) and h["loo_max"] == pytest.approx(-1.0)
    assert h["expected_sign"] is True and h["main"] is True and h["verdict"] == "supported"
    ya = [0.01 * ((i * 3) % 8) for i in range(8)]
    assert [p["y"] for p in res["tests"]["H_abs"]["points"]] == pytest.approx(ya)
    assert res["tests"]["H_abs"]["rho"] == pytest.approx(spearman_exact(xs, ya).rho)
    # md1: information only
    for t in ("H_rel_md1", "H_abs_md1"):
        assert res["tests"][t]["main"] is False and res["tests"][t]["verdict"] is None
    assert res["tests"]["H_rel_md1"]["points"][1]["x"] == pytest.approx(0.21)
    md = msum.spearman_markdown(res)
    assert "n = 8" in md and "0.738" in md and "information, no verdict" in md
    assert "no multiplicity correction" in md and "not independent" in md


def test_spearman_wrong_sign_significant_is_not_supported():
    md3 = {lab: (0.5 + 0.05 * i, 0.10 + 0.03 * i, 0.02 + 0.01 * i) for i, lab in enumerate(LABELS)}
    res = msum.spearman_section(_r19(md3), LABELS)
    h = res["tests"]["H_rel"]
    assert h["rho"] == pytest.approx(1.0) and h["p"] < 0.05
    assert h["verdict"] == "not supported" and h["expected_sign"] is False and h["significant_wrong_sign"] is True
    assert "significant with the wrong sign" in msum.spearman_markdown(res)


def test_spearman_not_significant_is_not_supported():
    ranks = [4, 7, 2, 8, 1, 5, 3, 6]  # weak association with x
    md3 = {lab: (0.5 + 0.05 * i, 0.01 * ranks[i], 0.05 + 0.001 * i) for i, lab in enumerate(LABELS)}
    res = msum.spearman_section(_r19(md3), LABELS)
    h = res["tests"]["H_rel"]
    assert h["p"] >= 0.05 and h["verdict"] == "not supported" and h["significant_wrong_sign"] is False


def test_r19_nonpositive_cells_counted_of_24():
    md3 = {lab: (0.5, 0.1, 0.02) for lab in LABELS}
    r19 = _r19(md3)
    r19["xgb-base"]["adv_es"]["md2"]["lower"] = -0.01  # CI includes 0
    r19["gru-base"]["adv_es"]["md2"].update(rel=-0.02, lower=-0.03, upper=-0.01)  # negative, CI excluding 0
    r19["m2-base"]["adv_es"]["md3"]["lower"] = 0.0  # CI touches 0: not excluding it
    r19["m3-base"]["clean"]["md3"]["lower"] = -0.5  # the clean leg is not counted
    res = msum.r19_nonpositive(r19, LABELS)
    assert res["n_cells"] == 24 and res["k"] == 3
    assert [(c["label"], c["md"]) for c in res["cells"]] == [("m2-base", "md3"), ("xgb-base", "md2"),
                                                             ("gru-base", "md2")]


# --- (b) policy vs model ----------------------------------------------------------------------------------------------


def test_policy_vs_model_8_equals_policy_vs_model_on_the_merged_dict():
    grid = toy_grid8()
    fam = {"a.json": {"labels": LABELS[:4], "policy_vs_model": {"median_policy_span": 0.1, "median_model_span": 0.2,
                                                                "median_seed_spread": 0.01, "verdict": "x"}}}
    res = msum.policy_vs_model_8(grid, fam)
    ref = ps.policy_vs_model(grid)
    for k in ("median_policy_span", "median_model_span", "median_seed_spread", "verdict", "model_span",
              "policy_span"):
        assert res["eight"][k] == ref[k]
    assert res["families"][0]["source"] == "a.json" and res["families"][0]["verdict"] == "x"
    assert res["families"][0]["differs_from_eight"] == (ref["verdict"] != "x")


def test_family_spans_within_and_across():
    grid = toy_grid8()  # offsets 0.01 * i: base/hard of one family differ by 0.01 in every cell
    res = msum.family_spans(grid)
    assert set(res["within"]) == {"m3", "m2", "xgb", "gru"}
    for f in res["within"]:
        assert res["within"][f] == pytest.approx(0.01)
    assert res["within_all"] == pytest.approx(0.01)
    # bases offsets 0, 0.02, 0.04, 0.06 → 0.06; hardened 0.01, 0.03, 0.05, 0.07 → 0.06
    assert res["across_base"] == pytest.approx(0.06) and res["across_hard"] == pytest.approx(0.06)
    assert res["n_cells"] == 18


# --- shared cells -------------------------------------------------------------------------------------------------


def test_check_shared_fails_when_targets_differ():
    grid = toy_grid8(**{"gru-hard": {"targets": [0, 1, 2, 3, 99]}})
    with pytest.raises(ValueError, match="targets"):
        msum.check_shared(grid)


def test_check_shared_fails_when_gate_differs():
    grid = toy_grid8()
    grid["xgb-base"] = dict(grid["xgb-base"], gate=dict(grid["xgb-base"]["gate"], delta=0.2))
    with pytest.raises(ValueError, match="gate"):
        msum.check_shared(grid)
    msum.check_shared(toy_grid8())


def test_shared_cells_table():
    grid = toy_grid8()
    t = msum.shared_cells_table(grid)
    assert len(t["rows"]) == 18
    r = next(r for r in t["rows"] if r["key"] == "fpr=0.005|md3|corridor=off")
    assert r["U"]["m3-base"] == pytest.approx(0.6) and r["U"]["gru-hard"] == pytest.approx(0.67)
    assert r["model_span"] == pytest.approx(0.07)


# --- (c) Pareto ---------------------------------------------------------------------------------------------------


def test_first_delta_from_appetite():
    app = [{"delta": 0.08, "A_with_md1": None, "A_without_md1": None},
           {"delta": 0.09, "A_with_md1": {"key": "x"}, "A_without_md1": None},
           {"delta": 0.1, "A_with_md1": {"key": "x"}, "A_without_md1": {"key": "y"}}]
    assert msum.first_delta(app, "A_without_md1") == 0.1 and msum.first_delta(app, "A_with_md1") == 0.09
    assert msum.first_delta(app[:1], "A_without_md1") is None


def test_pareto_strict_dominance_ties_and_resolution_flag():
    models = {
        "a": {"recall": 0.98, "delta_A": 0.10, "delta_A_md1": 0.09},
        "b": {"recall": 0.97, "delta_A": 0.136, "delta_A_md1": None},  # dominated by a, Δδ′ exactly 0.036
        "c": {"recall": 0.98, "delta_A": 0.10, "delta_A_md1": None},  # exact tie with a: both stay
        "d": {"recall": 0.99, "delta_A": 0.20, "delta_A_md1": None},  # on the front (best recall)
        "e": {"recall": 0.96, "delta_A": 0.1359, "delta_A_md1": None},  # dominated by a, c, gap just below 0.036
    }
    res = msum.pareto_front(models, n_fraud=2000)
    m = res["models"]
    assert m["a"]["on_front"] and m["c"]["on_front"] and m["d"]["on_front"]
    assert not m["b"]["on_front"] and m["b"]["dominated_by"] == ["a", "c"]
    assert m["e"]["dominated_by"] == ["a", "c"]  # b: δ′ 0.136 > 0.1359, no dominance
    pairs = {(p["a"], p["b"]): p for p in res["pairs"]}
    assert set(pairs) == {("a", "b"), ("c", "b"), ("a", "e"), ("c", "e")}
    ab = pairs[("a", "b")]
    assert ab["d_delta_pp"] == pytest.approx(3.6) and ab["d_recall_pp"] == pytest.approx(1.0)
    assert ab["d_ops"] == pytest.approx(20.0) and ab["delta_gap_below_resolution"] is False  # exactly 0.036
    ae = pairs[("a", "e")]
    assert ae["delta_gap_below_resolution"] is True and ae["d_ops"] == pytest.approx(40.0)  # 0.0359
    # b is dominated with a gap ≥ R: off the companion front; e only with gaps < R: on it
    assert res["front"] == ["a", "c", "d"]
    assert res["front_ignoring_gaps_below_resolution"] == ["a", "c", "d", "e"]


def test_pareto_resolution_flag_on_the_delta_grid():
    # 0.36 - 0.32 is 0.03999… in floating point: a 4 pp gap must not be flagged as below the 3.6 pp resolution
    models = {"a": {"recall": 0.98, "delta_A": 0.32, "delta_A_md1": None},
              "b": {"recall": 0.97, "delta_A": 0.36, "delta_A_md1": None},
              "c": {"recall": 0.96, "delta_A": 0.35, "delta_A_md1": None}}
    pairs = {(p["a"], p["b"]): p for p in msum.pareto_front(models, n_fraud=2000)["pairs"]}
    assert pairs[("a", "b")]["delta_gap_below_resolution"] is False
    assert pairs[("a", "c")]["delta_gap_below_resolution"] is True  # 3 pp


def test_check_shared_fails_when_targets_missing():
    grid = {lab: toy_sweep(lab) for lab in LABELS}
    for c in grid["xgb-base"]["cells"]:
        c.pop("targets", None)
    with pytest.raises(ValueError, match="no targets"):
        msum.check_shared(grid)


def test_pareto_model_without_passing_cell_is_kept_off():
    models = {"a": {"recall": 0.98, "delta_A": 0.10, "delta_A_md1": 0.10},
              "z": {"recall": 0.999, "delta_A": None, "delta_A_md1": None}}
    res = msum.pareto_front(models, n_fraud=2145)
    z = res["models"]["z"]
    assert z["on_front"] is None and z["dominated_by"] == [] and z["status"] == "no passing cell up to 0.50"
    assert res["models"]["a"]["on_front"] is True and res["front"] == ["a"] and res["pairs"] == []
    assert res["no_passing_cell"] == ["z"]
    md = msum.pareto_markdown(res)
    assert "no passing cell up to 0.50" in md


def test_n_fraud_test_from_model_json(tmp_path):
    grid = {}
    for lab, meta in (("a", {"test": {"n_fraud": 2145}}), ("b", {"fraud_rows": {"test": 2145}})):
        d = tmp_path / lab
        d.mkdir()
        (d / "model.json").write_text(json.dumps({"meta": meta}), encoding="utf-8")
        grid[lab] = {"model_dir": lab, "model_json_sha256": hashlib.sha256((d / "model.json").read_bytes()).hexdigest()}
    res = msum.n_fraud_test(grid, root=tmp_path)
    assert res["n_fraud"] == 2145 and res["per_model"]["a"]["field"] == "meta.test.n_fraud"
    assert res["per_model"]["b"]["field"] == "meta.fraud_rows.test"
    (tmp_path / "b" / "model.json").write_text(json.dumps({"meta": {"fraud_rows": {"test": 2000}}}), encoding="utf-8")
    grid["b"]["model_json_sha256"] = hashlib.sha256((tmp_path / "b" / "model.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="n_fraud"):
        msum.n_fraud_test(grid, root=tmp_path)
    grid["b"]["model_json_sha256"] = "other"
    with pytest.raises(ValueError, match="SHA-256"):
        msum.n_fraud_test(grid, root=tmp_path)


# --- (d) cost of hardening ----------------------------------------------------------------------------------------


def test_cost_of_hardening_arithmetic():
    own = {"m3-base": {"recall": 0.98, "fpr": 0.0091}, "m3-hard": {"recall": 0.94, "fpr": 0.0087}}
    res = msum.cost_of_hardening(own, {"m3": ("m3-base", "m3-hard")}, n_fraud=2000)
    r = res["m3"]
    assert r["d_recall_pp"] == pytest.approx(-4.0) and r["d_fpr_pp"] == pytest.approx(-0.04)
    assert r["missed_base"] == pytest.approx(0.02) and r["missed_hard"] == pytest.approx(0.06)
    assert r["missed_ratio"] == pytest.approx(3.0)
    assert r["missed_ops_base"] == pytest.approx(40.0) and r["missed_ops_hard"] == pytest.approx(120.0)


def test_register_check():
    cost = {"m3": {"recall_base": 0.97669, "recall_hard": 0.93473, "fpr_base": 0.009067, "fpr_hard": 0.00871}}
    reg = {"m3": {"recall": (0.977, 0.935), "fpr": (0.0091, 0.0087), "row": "3.6"}}
    rows = msum.check_register(cost, reg)
    assert rows[0]["family"] == "m3" and rows[0]["ok"] is True
    with pytest.raises(ValueError, match="register"):
        msum.check_register(cost, {"m3": dict(reg["m3"], recall=(0.977, 0.936))})


# --- end to end -------------------------------------------------------------------------------------------------


def _write_family(tmp_path, name, labels, grid8, recall_by_label, appetite_first):
    files = {}
    for lab in labels:
        s = toy_sweep(lab, offset=0.01 * LABELS.index(lab), recall=recall_by_label[lab],
                      model_dir=f"models/{lab}")
        md = tmp_path / "models" / lab
        md.mkdir(parents=True)
        (md / "model.json").write_text(json.dumps({"meta": {"fraud_rows": {"test": 2000}}}), encoding="utf-8")
        s["model_json_sha256"] = hashlib.sha256((md / "model.json").read_bytes()).hexdigest()
        p = tmp_path / name / lab / "policy-sweep.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(s), encoding="utf-8")
        files[lab] = str(p)
    loaded, _ = ps.load(files.values())
    pvm = ps.policy_vs_model(loaded)
    md3 = {lab: (0.5 + 0.05 * LABELS.index(lab), 0.3 - 0.03 * LABELS.index(lab), 0.01 * LABELS.index(lab))
           for lab in labels}
    app = {}
    for lab in labels:
        first = appetite_first[lab]
        app[lab] = [{"delta": d, "A_with_md1": None if first is None or d < first else {"key": "x"},
                     "A_without_md1": None if first is None or d < first else {"key": "x"}}
                    for d in ps.DELTA_PRIMES]
    summ = {"labels": labels, "sources": {lab: [{"file": f, "git_sha": "abc1234567", "n_cells": 20}]
                                          for lab, f in files.items()},
            "r19": _r19(md3), "appetite": app, "policy_vs_model": pvm}
    sp = tmp_path / name / "summary.json"
    sp.write_text(json.dumps(summ), encoding="utf-8")
    return str(sp)


def test_summarize8_end_to_end(tmp_path):
    recall = {lab: 0.95 - 0.01 * i for i, lab in enumerate(LABELS)}
    first = {lab: 0.10 + 0.01 * i for i, lab in enumerate(LABELS)}
    first["gru-hard"] = None
    paths = [_write_family(tmp_path, "m", LABELS[:4], None, recall, first),
             _write_family(tmp_path, "x", LABELS[4:6], None, recall, first),
             _write_family(tmp_path, "g", LABELS[6:], None, recall, first)]
    out = tmp_path / "out"
    res = msum.summarize8(paths, out, register=None, root=tmp_path)
    for f in ("summary8.json", "summary8.md", "pareto-8-en.png", "pareto-8-uk.png"):
        assert (out / f).exists()
    assert res["labels"] == LABELS and res["n_fraud_test"]["n_fraud"] == 2000
    assert res["pareto"]["front"] == ["m3-base"] and res["pareto"]["no_passing_cell"] == ["gru-hard"]
    assert res["spearman"]["tests"]["H_rel"]["verdict"] == "supported"
    assert res["cost"]["m3"]["d_recall_pp"] == pytest.approx(-1.0)
    md = (out / "summary8.md").read_text(encoding="utf-8")
    heads = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    pos = [md.index(f"## {h}") for h in heads]
    assert pos == sorted(pos) and md.index("abc1234567") < pos[0]
    assert "(pre-registered rule:" in md
    j = json.loads((out / "summary8.json").read_text(encoding="utf-8"))
    assert j["policy_vs_model"]["eight"]["verdict"] in ("supported", "not supported")
    # a label in two summaries is refused
    with pytest.raises(ValueError, match="more than one summary"):
        msum.summarize8([paths[0], paths[0]], out, register=None, root=tmp_path)


def test_cli_models_summary_parser():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["models-summary", "--summary", "a.json", "--summary", "b.json", "--out", "o"])
    assert a.command == "models-summary" and [str(x) for x in a.summary] == ["a.json", "b.json"]
    assert str(a.out) == "o"


def _three_families(tmp_path):
    recall = {lab: 0.95 - 0.01 * i for i, lab in enumerate(LABELS)}
    first = {lab: 0.10 + 0.01 * i for i, lab in enumerate(LABELS)}
    return [_write_family(tmp_path, "m", LABELS[:4], None, recall, first),
            _write_family(tmp_path, "x", LABELS[4:6], None, recall, first),
            _write_family(tmp_path, "g", LABELS[6:], None, recall, first)]


def test_summarize8_register_mismatch_writes_nothing(tmp_path):
    paths = _three_families(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="register"):
        msum.summarize8(paths, out, root=tmp_path)  # the default REGISTER does not match the toy data
    assert not out.exists() or not any(out.iterdir())


def test_summarize8_refuses_a_commit_not_named_in_the_sources(tmp_path):
    paths = _three_families(tmp_path)
    f = tmp_path / "x" / "xgb-hard" / "policy-sweep.json"
    f.write_text(f.read_text(encoding="utf-8").replace("abc1234567", "fff0000000"), encoding="utf-8")
    with pytest.raises(ValueError, match="code version"):
        msum.summarize8(paths, tmp_path / "out", register=None, root=tmp_path)
