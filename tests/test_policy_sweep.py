"""Task 19: bank-policy grid (policy-sweep / policy-summary), cached friction, U$, T1 / T2 / policy-vs-model, R19."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_sparkov_like
from fraudaudit import policy_summary as ps
from fraudaudit.policy_sweep import cell_key, parse_reaction, usd_uplift
from fraudaudit.threat import ThreatModel

ROOT = Path(__file__).resolve().parents[1]
TM_V1V2 = ROOT / "threat-models" / "realistic-corridor-v1v2.yaml"


# ---- pure parts ---------------------------------------------------------------------------------------------------

def test_parse_reaction():
    assert parse_reaction("md1") == {"max_declines": 1, "esc": False}
    assert parse_reaction("md3+esc") == {"max_declines": 3, "esc": True}
    assert parse_reaction("mdinf") == {"max_declines": None, "esc": False}
    for bad in ("md0", "mdx", "md1+esc", "esc", "md3+esc+esc"):
        with pytest.raises(ValueError):
            parse_reaction(bad)


def test_usd_uplift_is_the_paired_bootstrap_on_dollars():
    from fraudaudit.metrics import uplift_ci
    a = np.array([100.0, 200.0, 300.0, 400.0])
    rec = {"amounts": a.tolist(), "runs": {
        "es/1": {"ver_adv": [0.5, 0.5, 0.5, 0.5], "ver_clean": [0.4, 0.4, 0.4, 0.4]},
        "es/2": {"ver_adv": [0.9, 0.1, 0.3, 0.6], "ver_clean": [0.1, 0.1, 0.1, 0.1]}}}
    usd_uplift(rec, 0.95, 42, True, 2000)
    r1 = rec["runs"]["es/1"]
    assert r1["usd_uplift_mean"] == pytest.approx(0.1 * a.mean()) and r1["usd_uplift_upper"] >= r1["usd_uplift_mean"]
    want = uplift_ci(np.array([0.9, 0.1, 0.3, 0.6]) * a, np.full(4, 0.1) * a, 0.95, 2000, 42, one_sided=True)
    assert rec["runs"]["es/2"]["usd_uplift_upper"] == pytest.approx(want[2])
    assert rec["worst_run_usd"] == max(rec["runs"], key=lambda k: rec["runs"][k]["usd_uplift_upper"])
    # not U × mean A: the dollar statistic weighs the targets by their amounts
    from fraudaudit.metrics import uplift_ci as u
    share = u(np.array([0.9, 0.1, 0.3, 0.6]), np.full(4, 0.1), 0.95, 2000, 42, one_sided=True)
    assert rec["runs"]["es/2"]["usd_uplift_upper"] != pytest.approx(share[2] * a.mean())


def _cell(fpr, reaction, corridor, U, fb, cf, runs=None):
    return {"key": cell_key(fpr, reaction, corridor), "target_fpr": fpr, "reaction": reaction, "corridor": corridor,
            "U": U, "friction": {"share_blocked_card_windows": fb, "share_confirm_card_days": cf},
            "runs": runs or {"es/1": {"uplift_upper": U}, "es/2": {"uplift_upper": U - 0.02}}}


def toy_grid(offset=0.0, tau_step=-0.05, corr_step=-0.01, md1_step=0.0, esc_step=-0.04):
    """18 cells with known axis effects on U: tau one step −0.05, corridor −0.01, md3→md1 0, md3→esc −0.04."""
    cells = []
    for i, f in enumerate((0.005, 0.01, 0.02)):
        for r in ("md1", "md3", "md3+esc"):
            for c in ("off", "on"):
                U = 0.6 + offset + i * tau_step + (corr_step if c == "on" else 0) \
                    + (md1_step if r == "md1" else 0) + (esc_step if r == "md3+esc" else 0)
                fb = 0.1 * (i + 1) + (0.2 if r == "md1" else 0) + (0.05 if r == "md3+esc" else 0)
                cf = 0.03 if c == "on" else 0.0
                cells.append(_cell(f, r, c, U, fb, cf))
    return {"cells": cells, "own_target_fpr": 0.01, "model_dir": "m", "gate": {"delta": 0.1, "attack_seeds": [1, 2]}}


def test_t1_toy_grid_known_verdicts():
    res = ps.t1({"a": toy_grid()}, resolution=0.036)
    assert res["tau"]["a"]["n_pairs"] == 12 and res["tau"]["a"]["median_dU"] == pytest.approx(-0.05)
    assert res["tau"]["a"]["verdict"] == "supported"
    assert res["escalation"]["a"]["median_dU"] == pytest.approx(-0.04)
    assert res["escalation"]["a"]["verdict"] == "supported" and res["escalation"]["a"]["n_pairs"] == 6
    assert res["max_declines"]["a"]["median_dU"] == pytest.approx(0.0)
    assert res["max_declines"]["a"]["verdict"] == "refuted"  # the attacker compensates
    assert res["corridor"]["a"]["median_dU"] == pytest.approx(-0.01)
    assert res["corridor"]["a"]["verdict"] == "inconclusive"  # below the resolution
    assert res["corridor"]["a"]["median_dfriction"] == pytest.approx(0.03)
    assert res["tau"]["pooled"]["n_pairs"] == 12


def test_policy_vs_model_toy():
    a, b = toy_grid(), toy_grid(offset=0.02)
    res = ps.policy_vs_model({"a": a, "b": b}, resolution=0.036)  # seed spread 0.02 everywhere (runs U, U − 0.02)
    # softest (0.005, md3, off) U 0.6; strictest (0.02, md1, on) U 0.6 − 0.10 − 0.01 = 0.49
    assert res["policy_span"]["a"] == pytest.approx(0.11) and res["policy_span"]["b"] == pytest.approx(0.11)
    assert res["median_model_span"] == pytest.approx(0.02) and res["median_seed_spread"] == pytest.approx(0.02)
    assert res["verdict"] == "supported" and res["interaction"] == "none"
    flat = toy_grid(tau_step=0.0, corr_step=0.0, esc_step=0.0)
    res2 = ps.policy_vs_model({"a": a, "flat": flat}, resolution=0.036)
    assert res2["policy_span"]["flat"] == pytest.approx(0.0) and "only on a" in res2["interaction"]


def test_t2_pick_and_confirmation():
    g = toy_grid(offset=-0.45)  # U from 0.15 down to 0.04: some cells green at δ 0.1
    res = ps.t2({"a": g}, {}, 0.10)["a"]
    greens = {c["key"] for c in g["cells"] if all(r["uplift_upper"] <= 0.10 for r in c["runs"].values())}
    assert {x["key"] for x in res["green_cells"]} == greens and res["t2"] == "pending confirmation"
    picked = next(c for c in g["cells"] if c["key"] == res["picked"])
    assert picked["friction"]["share_blocked_card_windows"] == min(
        c["friction"]["share_blocked_card_windows"] for c in g["cells"] if c["key"] in greens)
    plan = ps.confirmation_plan({"a": g}, {"a": res})
    assert plan == [{"label": "a", "model_dir": "m", "target_fpr": picked["target_fpr"], "reaction": picked["reaction"],
                     "corridor": picked["corridor"], "key": picked["key"]}]
    conf = {"a": {"gate": {"attack_seeds": [4, 5, 6]}, "cells": [dict(picked, U=0.2, runs={"es/4": {"uplift_upper": 0.2}})]}}
    res2 = ps.t2({"a": g}, conf, 0.10)["a"]
    assert res2["t2"] == "not confirmed" and ps.confirmation_plan({"a": g}, {"a": res2}) == []
    assert ps.t2({"a": toy_grid()}, {}, 0.10)["a"]["t2"] == "no green cell"


def test_rel_overestimation_toy():
    v_md = np.array([0.2, 0.4, 0.6, 0.8])
    r = ps.rel_overestimation(v_md * 1.5, v_md)
    assert r["rel"] == pytest.approx(0.5) and r["lower"] == pytest.approx(0.5) and r["upper"] == pytest.approx(0.5)
    r2 = ps.rel_overestimation(np.array([0.4, 0.4, 0.9, 0.8]), v_md)
    assert r2["rel"] == pytest.approx((2.5 / 4 - 0.5) / 0.5) and r2["lower"] <= r2["rel"] <= r2["upper"]


# ---- friction: cached p equals the live replay ------------------------------------------------------------------

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel, train_model
    df = make_sparkov_like(n_cards=120, days=200, start=pd.Timestamp("2019-11-01"), seed=3)
    model, _ = train_model(df, seed=0, epochs=1)
    d = tmp_path_factory.mktemp("model")
    model.save(d, meta={"seed": 0, "target_fpr": 0.01})
    return df, FraudModel.from_artifacts(load_artifacts(d)), d


def test_false_blocks_cached_equals_live(env):
    from fraudaudit.metrics import false_block_rate, legit_test_scores
    df, model, d = env
    scores = legit_test_scores(df, model, n_jobs=1)
    taus = sorted({float(np.quantile(np.concatenate(list(scores.values())), q)) for q in (0.5, 0.8)})
    n_blocked = 0
    for tau in taus:
        for md, tau2 in ((1, None), (3, None), (3, tau / 2), (None, None), (2, tau / 3)):
            live = false_block_rate(df, model, tau, md, 30, tau_after_decline=tau2, n_jobs=1)
            cached = false_block_rate(df, None, tau, md, 30, tau_after_decline=tau2, n_jobs=1, scores=scores)
            par = false_block_rate(df, None, tau, md, 30, tau_after_decline=tau2, n_jobs=2, scores=scores)
            assert live == cached == par
            n_blocked += live.n_blocked_card_windows
    assert n_blocked > 0  # blocked windows exercised (skipped scores)


# ---- policy-sweep end to end on synthetic data --------------------------------------------------------------------

def _setup(env):
    from fraudaudit.gate import GateConfig
    ctm = ThreatModel.from_yaml(TM_V1V2)
    ctm = dataclasses.replace(ctm, recon_budget_ops=100, corridor=dataclasses.replace(ctm.corridor, lookback_days=60))
    tm = dataclasses.replace(ctm, corridor=None)
    g = GateConfig(target_fpr=0.01, recall_clean_min=0.0, delta=0.1, attacks=["random", "es"], attack_seeds=[1],
                   confidence=0.95, one_sided=True, spares_period="fit")
    return tm, ctm, g


@pytest.fixture(scope="module")
def sweep(env, tmp_path_factory):
    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    out = tmp_path_factory.mktemp("sweep")
    data = policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01, 0.05), reactions=("md1", "md3+esc"),
                        corridors=("off", "on"), esc_fpr=0.2, r19=True, seed=1, n_jobs=1, out_dir=out / "m")
    return data, out


CELL_KEYS = {"key", "config_hash", "target_fpr", "tau", "reaction", "max_declines", "tau_after_decline", "corridor",
             "runs", "worst_run", "worst_uplift_upper", "mean_uplift", "max_ver_adv", "test_recall", "test_fpr",
             "amounts", "targets", "U", "U_usd", "worst_run_usd", "worst_usd_uplift_upper", "wall_seconds",
             "ensemble_seconds", "friction"}


def test_policy_sweep_cell_schema(env, sweep):
    data, out = sweep
    _, model, _ = env
    assert len(data["cells"]) == 2 * 2 * 2 + 2  # grid + the R19 cells md2, mdinf
    keys = {c["key"] for c in data["cells"]}
    assert cell_key(0.01, "mdinf", "off") in keys and cell_key(0.01, "md2", "off") in keys
    for c in data["cells"]:
        assert set(c) == CELL_KEYS
        assert set(c["runs"]) == {"random/1", "es/1"}
        assert c["U"] == max(r["uplift_upper"] for r in c["runs"].values())
        assert len(c["amounts"]) == len(c["runs"]["es/1"]["ver_adv"]) == len(c["targets"])
        f = c["friction"]
        assert f is not None and 0 <= f["share_blocked_card_windows"] <= 1
        assert (f["share_confirm_card_days"] == 0.0) == (c["corridor"] == "off")
        if c["reaction"] == "mdinf":
            assert f["share_blocked_card_windows"] == 0.0
    own = next(c for c in data["cells"] if c["key"] == cell_key(0.01, "md1", "off"))
    assert own["tau"] == pytest.approx(model.tau, rel=1e-6)
    esc = next(c for c in data["cells"] if c["reaction"] == "md3+esc")
    assert esc["tau_after_decline"] == data["tau_esc"] < min(c["tau"] for c in data["cells"])
    assert data["delta_usd"] == pytest.approx(0.1 * np.mean(own["amounts"]))
    assert data["meta"]["machine"]["cpu_count"] and "versions" in data["meta"]
    saved = json.loads((out / "m" / "policy-sweep.json").read_text())
    assert saved["cells"] == json.loads(json.dumps(data["cells"]))


def test_policy_sweep_resumes_and_reruns_changed_config(env, sweep, monkeypatch, tmp_path):
    import shutil

    from fraudaudit import policy_sweep as mod
    df, model, d = env
    tm, ctm, g = _setup(env)
    data, out = sweep
    work = tmp_path / "m"
    shutil.copytree(out / "m", work)
    calls = []
    real = mod.run_cell
    monkeypatch.setattr(mod, "run_cell", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    again = mod.policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01, 0.05), reactions=("md1", "md3+esc"),
                             corridors=("off", "on"), esc_fpr=0.2, r19=True, seed=1, n_jobs=1, out_dir=work)
    assert calls == [] and [c["U"] for c in again["cells"]] == [c["U"] for c in data["cells"]]
    # another escalation threshold changes only the md3+esc cells' config: those run again
    mod.policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01, 0.05), reactions=("md1", "md3+esc"),
                     corridors=("off", "on"), esc_fpr=0.3, seed=1, n_jobs=1, out_dir=work, friction=False)
    assert len(calls) == 4


def test_escalation_threshold_must_be_below_tau(env, tmp_path):
    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    with pytest.raises(ValueError, match="escalation threshold"):
        policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01, 0.05), reactions=("md3+esc",), corridors=("off",),
                     esc_fpr=0.02, seed=1, n_jobs=1, out_dir=tmp_path / "x")


def test_corridor_threat_must_match(env, tmp_path):
    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    with pytest.raises(ValueError, match="differs from the threat model"):
        policy_sweep(df, model, d, "m", dataclasses.replace(tm, k_max=5), ctm, g, fprs=(0.01,), reactions=("md1",),
                     corridors=("on",), seed=1, n_jobs=1, out_dir=tmp_path / "x")


def test_policy_sweep_n_jobs_identical(env, sweep, tmp_path):
    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    data, _ = sweep
    par = policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.05,), reactions=("md3+esc",), corridors=("on",),
                       esc_fpr=0.2, seed=1, n_jobs=2, out_dir=tmp_path / "p")
    a = next(c for c in data["cells"] if c["key"] == cell_key(0.05, "md3+esc", "on"))
    b = par["cells"][0]
    for k in ("U", "U_usd", "runs", "friction", "test_recall", "tau", "tau_after_decline"):
        assert a[k] == b[k], k


def test_policy_summary_end_to_end(env, sweep, tmp_path):
    import shutil

    from fraudaudit.cli import main
    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    _, out0 = sweep
    out = tmp_path / "sw"
    shutil.copytree(out0 / "m", out / "m")
    # the md3 cell completes R19 (md1 / md2 / md3 vs mdinf); a run with another grid keeps the cells already there
    data = policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01,), reactions=("md3",), corridors=("off",),
                        seed=1, n_jobs=1, out_dir=out / "m")
    assert len(data["cells"]) == 11
    other = json.loads(json.dumps(data))
    other["label"] = "m-hard"
    for c in other["cells"]:
        c["U"] = c["worst_uplift_upper"] = c["U"] * 0.5
        for r in c["runs"].values():
            r["uplift_upper"] *= 0.5
    p2 = tmp_path / "m-hard" / "policy-sweep.json"
    p2.parent.mkdir()
    p2.write_text(json.dumps(other))
    code = main(["policy-summary", str(out / "m" / "policy-sweep.json"), str(p2), "--out", str(tmp_path / "s")])
    assert code == 0
    s = json.loads((tmp_path / "s" / "summary.json").read_text())
    assert set(s["t1"]) == set(ps.AXES) and set(s["t1"]["tau"]) == {"m", "m-hard", "pooled"}
    assert set(s["r19"]) == {"m", "m-hard"} and set(s["r19"]["m"]) == {"adv_es", "clean"}
    for name in ("hypotheses.md", "front-blocks.png", "front-confirm.png", "r19.md", "r19.png", "confirmations.tsv"):
        assert (tmp_path / "s" / name).stat().st_size > 0 or name == "confirmations.tsv"
    md = (tmp_path / "s" / "hypotheses.md").read_text()
    assert "## T1" in md and "## T2" in md and "Policy vs model" in md


def test_cli_policy_sweep_parser():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["policy-sweep", "--model", "models", "--label", "m3-base", "--model", "x",
                                   "--label", "x", "--r19", "--no-friction"])
    assert a.model == [Path("models"), Path("x")] and a.label == ["m3-base", "x"] and a.r19 and a.no_friction
    assert a.target_fpr == [0.005, 0.01, 0.02] and a.reactions == ["md1", "md3", "md3+esc"]
    assert a.corridor == ["off", "on"] and a.esc_fpr == 0.05 and a.role == "grid"
    assert str(a.corridor_threat).endswith("realistic-corridor-v1v2.yaml")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["policy-sweep", "--model", "m", "--label", "l", "--no-friction", "--friction-only"])
