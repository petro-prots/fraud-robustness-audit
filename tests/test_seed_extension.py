"""Task 27a: fresh seeds 10-15 for the front cells with x_c > 5.98 (descriptive, no verdict)."""
import json

import pytest

from fraudaudit.seed_extension import compare_seeds, markdown

KEY = "fpr=0.01|md3+esc|corridor=on|kappa=1.25|q=0.98"


def _sweep(path, role, seeds, ups, key=KEY, sha="sha", tau=0.5, u=None, budget=2000):
    runs = {f"{a}/{s}": {"uplift_upper": ups.get((a, s), 0.01)} for a in ("es", "random", "stepwise") for s in seeds}
    cell = {"key": key, "U": u if u is not None else max(r["uplift_upper"] for r in runs.values()), "runs": runs,
            "tau": tau, "max_declines": 3, "tau_after_decline": 0.1, "corridor_params": {"q": 0.98},
            "target_fpr": 0.01, "recon_budget_ops": budget}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"label": "xgb-hard", "role": role, "model_json_sha256": sha, "own_tau": 0.5,
                                "gate": {"attack_seeds": list(seeds)}, "cells": [cell]}))
    return path


def test_stats_per_seed_set(tmp_path):
    g = _sweep(tmp_path / "g.json", "grid", (1, 2, 3), {("stepwise", 1): 0.23})
    e = _sweep(tmp_path / "e.json", "confirmation", (4, 5, 6, 7, 8, 9), {("stepwise", 6): 0.352, ("es", 4): 0.196})
    f = _sweep(tmp_path / "f.json", "confirmation", (10, 11, 12, 13, 14, 15), {("random", 12): 0.30})
    res = compare_seeds([("xgb-hard", KEY, g, e, f)])
    row = res["cells"][0]
    assert row["U_gate"] == pytest.approx(0.23)
    assert row["earlier"]["worst"] == pytest.approx(0.352) and row["earlier"]["x_c"] == pytest.approx(0.122)
    assert row["fresh"]["seeds"] == [10, 11, 12, 13, 14, 15]
    assert row["fresh"]["worst"] == pytest.approx(0.30) and row["fresh"]["x_c"] == pytest.approx(0.07)
    assert row["fresh"]["U_s"]["12"] == pytest.approx(0.30)
    assert row["fresh_worst_attack"] == "random/12"
    md = markdown(res)
    assert "verdict" not in md.lower() or "no verdict" in md.lower()
    assert "12: 30.0 %" in md


def test_mismatches_are_refused(tmp_path):
    g = _sweep(tmp_path / "g.json", "grid", (1, 2, 3), {})
    e = _sweep(tmp_path / "e.json", "confirmation", (4, 5, 6, 7, 8, 9), {})
    bad_seeds = _sweep(tmp_path / "f1.json", "confirmation", (4, 5, 6, 7, 8, 9), {})
    with pytest.raises(ValueError, match="seeds"):
        compare_seeds([("xgb-hard", KEY, g, e, bad_seeds)])
    other_model = _sweep(tmp_path / "f2.json", "confirmation", (10, 11, 12, 13, 14, 15), {}, sha="other")
    with pytest.raises(ValueError, match="model_json_sha256"):
        compare_seeds([("xgb-hard", KEY, g, e, other_model)])
    other_tau = _sweep(tmp_path / "f3.json", "confirmation", (10, 11, 12, 13, 14, 15), {}, tau=0.6)
    with pytest.raises(ValueError, match="tau"):
        compare_seeds([("xgb-hard", KEY, g, e, other_tau)])
    grid_role = _sweep(tmp_path / "f4.json", "grid", (10, 11, 12, 13, 14, 15), {})
    with pytest.raises(ValueError, match="role"):
        compare_seeds([("xgb-hard", KEY, g, e, grid_role)])
    with pytest.raises(ValueError, match="no cell"):
        compare_seeds([("xgb-hard", "fpr=0.02|md1|corridor=on", g, e, other_tau)])


def test_cli_parser():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["seed-extension", "--cell", f"xgb-hard,{KEY},g.json,e.json,f.json", "--out", "o"])
    assert a.command == "seed-extension" and a.cell == [f"xgb-hard,{KEY},g.json,e.json,f.json"]
    b = build_parser().parse_args(["budget-sensitivity", "--reference", "r", "--budget", "50000=d", "--labels", "m",
                                   "--out", "o", "--hypothesis-budget", "50000"])
    assert b.hypothesis_budget == "50000"
    c = build_parser().parse_args(["budget-sensitivity", "--reference", "r", "--budget", "20000=d", "--labels", "m",
                                   "--out", "o", "--hypothesis-budget", "none"])
    assert c.hypothesis_budget == "none"


def test_threat_gate_and_earlier_overlap_are_refused(tmp_path):
    g = _sweep(tmp_path / "g.json", "grid", (1, 2, 3), {})
    e = _sweep(tmp_path / "e.json", "confirmation", (4, 5, 6, 7, 8, 9), {})
    f = _sweep(tmp_path / "f.json", "confirmation", (10, 11, 12, 13, 14, 15), {})
    d = json.loads(f.read_text())
    d["meta"] = {"threat_sha256": "changed"}
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="threat_sha256"):
        compare_seeds([("xgb-hard", KEY, g, e, f)])
    d["meta"] = {}
    d["gate"]["delta"] = 0.2
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="gate"):
        compare_seeds([("xgb-hard", KEY, g, e, f)])
    e2 = _sweep(tmp_path / "e2.json", "confirmation", (3, 4, 5, 6, 7, 8), {})
    f2 = _sweep(tmp_path / "f2.json", "confirmation", (10, 11, 12, 13, 14, 15), {})
    with pytest.raises(ValueError, match="earlier seeds"):
        compare_seeds([("xgb-hard", KEY, g, e2, f2)])


def test_cli_seed_extension_end_to_end(tmp_path):
    from fraudaudit.cli import main
    g = _sweep(tmp_path / "g.json", "grid", (1, 2, 3), {("stepwise", 1): 0.23})
    e = _sweep(tmp_path / "e.json", "confirmation", (4, 5, 6, 7, 8, 9), {})
    f = _sweep(tmp_path / "f.json", "confirmation", (10, 11, 12, 13, 14, 15), {})
    assert main(["seed-extension", "--cell", f"xgb-hard,{KEY},{g},{e},{f}", "--out", str(tmp_path / "o")]) == 0
    assert json.loads((tmp_path / "o" / "seed-extension.json").read_text())["cells"][0]["U_gate"] == pytest.approx(0.23)
    assert main(["seed-extension", "--cell", "a,b", "--out", str(tmp_path / "o2")]) == 2
