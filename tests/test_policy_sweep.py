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
            "tau": 1 - fpr, "max_declines": 1 if reaction == "md1" else 3,
            "tau_after_decline": 0.1 if reaction.endswith("+esc") else None, "test_recall": 0.9,
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
    return {"cells": cells, "own_target_fpr": 0.01, "model_dir": "m", "label": "a", "model_json_sha256": "sha",
            "gate": {"delta": 0.1, "attack_seeds": [1, 2], "attacks": ["es"], "recall_clean_min": 0.5}}


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
    assert res["verdict"] == "supported" and "interaction" not in res  # R32: described in its own table, no verdict
    flat = toy_grid(tau_step=0.0, corr_step=0.0, esc_step=0.0)
    res2 = ps.policy_vs_model({"a": a, "flat": flat}, resolution=0.036)
    assert res2["policy_span"]["flat"] == pytest.approx(0.0) and res2["policy_span"]["a"] == pytest.approx(0.11)


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
    conf = {"a": {"gate": {"attack_seeds": [4, 5, 6], "attacks": ["es"]}, "model_json_sha256": "sha",
                  "cells": [dict(picked, U=0.2, runs={"es/4": {"uplift_upper": 0.2}})]}}
    res2 = ps.t2({"a": g}, conf, 0.10)["a"]
    assert res2["t2"] == "not confirmed" and ps.confirmation_plan({"a": g}, {"a": res2}) == []
    assert ps.t2({"a": toy_grid()}, {}, 0.10)["a"]["t2"] == "no green cell"
    ok = ps.t2({"a": g}, {"a": dict(conf["a"], cells=[dict(picked, runs={"es/4": {"uplift_upper": 0.05}})])}, 0.10)
    assert ok["a"]["t2"] == "confirmed"
    # M5: an invalid confirmation is refused, never counted
    bad = [dict(conf["a"], gate={"attack_seeds": [1, 4], "attacks": ["es"]}),  # seeds overlap the grid's
           dict(conf["a"], gate={"attack_seeds": [4], "attacks": ["random"]}),  # other attacks
           dict(conf["a"], model_json_sha256="other"),
           dict(conf["a"], cells=[dict(picked, tau=0.5, runs={"es/4": {"uplift_upper": 0.05}})]),
           dict(conf["a"], cells=[dict(picked, corridor="x", runs={"es/4": {"uplift_upper": 0.05}})])]
    for b in bad:
        with pytest.raises(ValueError, match="confirmation"):
            ps.t2({"a": g}, {"a": b}, 0.10)


def test_rel_overestimation_toy():
    v_md = np.array([0.2, 0.4, 0.6, 0.8])
    r = ps.rel_overestimation(v_md * 1.5, v_md)
    assert r["rel"] == pytest.approx(0.5) and r["lower"] == pytest.approx(0.5) and r["upper"] == pytest.approx(0.5)
    r2 = ps.rel_overestimation(np.array([0.4, 0.4, 0.9, 0.8]), v_md)
    assert r2["rel"] == pytest.approx((2.5 / 4 - 0.5) / 0.5) and r2["lower"] <= r2["rel"] <= r2["upper"]
    assert r2["n_dropped"] == 0
    r3 = ps.rel_overestimation(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.3]))  # resamples with VER_md = 0
    assert 0 < r3["n_dropped"] < r3["n_boot"]


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
             "ensemble_seconds", "friction", "provenance", "corridor_kappa", "corridor_q", "corridor_params",
             "corridor_params_sha256", "n_corridor_runs", "n_bound_violations", "n_headroom_violations",
             "n_block_violations"}


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
    assert a.corridor_kappa is None and a.corridor_q is None
    v = build_parser().parse_args(["policy-sweep", "--model", "m", "--label", "l", "--corridor-kappa", "1.25,1.5",
                                   "--corridor-q", "0.98,0.99"])
    assert v.corridor_kappa == [1.25, 1.5] and v.corridor_q == [0.98, 0.99]
    sm = build_parser().parse_args(["policy-summary", "x.json", "--confirm-rule", "front", "--merge-commits",
                                    "abc1234,def5678"])
    assert sm.confirm_rule == "front" and sm.merge_commits == ["abc1234", "def5678"]
    assert build_parser().parse_args(["policy-summary", "x.json"]).confirm_rule == "t2"


def test_cli_policy_sweep_refuses_variants_without_corridor_on(capsys):
    from fraudaudit.cli import main
    code = main(["policy-sweep", "--model", "models", "--label", "x", "--corridor", "off", "--corridor-kappa", "1.5",
                 "--data", "/nonexistent"])
    assert code == 2 and "need --corridor on" in capsys.readouterr().err


# ---- review fixes: wiring of the cell policy, provenance ---------------------------------------------------------

def test_cell_policy_reaches_the_threat_model(env, tmp_path, monkeypatch):
    """I2: the ThreatModel handed to run_cell carries the cell's reaction and corridor (not the YAML's policy)."""
    from fraudaudit import policy_sweep as mod
    df, model, d = env
    tm, ctm, g = _setup(env)
    seen = {}
    real = mod.run_cell

    def spy(df_, mdl, mdir, tm_c, *a, **k):
        seen[(round(float(mdl.tau), 12), tm_c.max_declines, tm_c.tau_after_decline,
              None if tm_c.corridor is None else json.dumps(tm_c.corridor.public(), sort_keys=True))] = True
        return real(df_, mdl, mdir, tm_c, *a, **k)

    monkeypatch.setattr(mod, "run_cell", spy)
    data = mod.policy_sweep(df, model, d, "m", dataclasses.replace(tm, max_declines=7), ctm, g, fprs=(0.01, 0.05),
                            reactions=("md1", "md3", "md3+esc"), corridors=("off", "on"), esc_fpr=0.2, r19=True,
                            seed=1, n_jobs=1, out_dir=tmp_path / "w", friction=False)
    on = data["corridor_params"]
    assert on is not None and on["q"] == ctm.corridor.q  # the prepared corridor of --corridor-threat
    want = set()
    for c in data["cells"]:
        md = {"md1": 1, "md2": 2, "md3": 3, "md3+esc": 3, "mdinf": None}[c["reaction"]]
        t2 = data["tau_esc"] if c["reaction"] == "md3+esc" else None
        want.add((round(c["tau"], 12), md, t2, json.dumps(on, sort_keys=True) if c["corridor"] == "on" else None))
    assert set(seen) == want and len(seen) == len(data["cells"]) == 2 * 3 * 2 + 2
    assert any(k[1] is None for k in seen) and any(k[1] == 2 for k in seen)  # the R19 cells mdinf, md2


def test_reuse_refused_for_other_code_or_machine(env, sweep, monkeypatch, tmp_path):
    """I1: a finished cell / friction entry is never reused by a run of another git SHA, machine or a dirty tree."""
    import shutil

    from fraudaudit import policy_sweep as mod
    df, model, d = env
    tm, ctm, g = _setup(env)
    data, out = sweep
    assert all(c["provenance"]["git_sha"] for c in data["cells"])
    assert all(e["provenance"] for e in data["friction"]["false_blocks"].values())
    real = mod.provenance()
    for change in ({"git_sha": "other"}, {"host_machine": "another box"}, {"git_dirty": True}):
        work = tmp_path / str(len(list(tmp_path.iterdir())))
        shutil.copytree(out / "m", work)
        monkeypatch.setattr(mod, "provenance", lambda change=change: dict(real, **change))
        with pytest.raises(ValueError, match="refusing to reuse"):
            mod.policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01,), reactions=("md1",), corridors=("off",),
                             seed=1, n_jobs=1, out_dir=work, friction=False)
        with pytest.raises(ValueError, match="refusing to reuse"):
            mod.policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01,), reactions=("md1",), corridors=("off",),
                             seed=1, n_jobs=1, out_dir=work, ensemble=False)


def test_other_gate_refused_on_resume(env, sweep, tmp_path):
    import shutil

    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    _, out = sweep
    shutil.copytree(out / "m", tmp_path / "m")
    with pytest.raises(ValueError, match="gate"):
        policy_sweep(df, model, d, "m", tm, ctm, dataclasses.replace(g, delta=0.2), fprs=(0.01,), reactions=("md1",),
                     corridors=("off",), seed=1, n_jobs=1, out_dir=tmp_path / "m", ensemble=False)


def test_summary_refuses_mixed_provenance(sweep, tmp_path):
    data, out = sweep
    other = json.loads(json.dumps(data))
    other["label"] = "x"
    other["cells"][0]["provenance"]["git_sha"] = "different"
    p2 = tmp_path / "x.json"
    p2.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="more than one code version"):
        ps.summarize([out / "m" / "policy-sweep.json", p2], tmp_path / "s")


# ---- Task 19b: corridor strength, merge, front confirmation, margin X, risk-appetite curve (R32) -----------------

FILE_KQ = (2.0, 0.99)


def _cp(kappa, q):
    return {"q": q, "daily": {"mode": "weekly_split", "kappa": kappa}, "confirmed_extends": True}


def _ccell(fpr, reaction, corridor, U, fb, cf, kappa=None, q=None, runs=None):
    """A toy cell in the normalised form of policy_summary (corridor κ / q, variant flag, cards ever blocked)."""
    var = None if corridor == "off" or (kappa, q) == FILE_KQ else (kappa, q)
    c = _cell(fpr, reaction, corridor, U, fb, cf, runs)
    c["key"] = cell_key(fpr, reaction, corridor, var)
    c["friction"]["share_blocked_cards"] = min(1.0, 2 * fb)
    c["U_usd"] = 1000 * U
    c.update(corridor_kappa=kappa if corridor == "on" else None, corridor_q=q if corridor == "on" else None,
             corridor_variant=var is not None)
    if reaction in ("md2", "mdinf"):
        c["max_declines"] = None if reaction == "mdinf" else 2
    return c


def test_cell_key_of_the_file_variant_is_todays_key():
    from fraudaudit.policy_sweep import corridor_variants, variant_id
    ctm = ThreatModel.from_yaml(TM_V1V2)
    assert variant_id(ctm.corridor, 2, 0.99) is None and variant_id(ctm.corridor, 1.5, 0.99) == (1.5, 0.99)
    assert cell_key(0.01, "md3", "on", None) == "fpr=0.01|md3|corridor=on"
    assert cell_key(0.01, "md3", "on", (1.25, 0.98)) == "fpr=0.01|md3|corridor=on|kappa=1.25|q=0.98"
    assert corridor_variants(ctm.corridor, [1.25, 1.5], [0.98, 0.99]) == [(1.25, 0.98), (1.25, 0.99), (1.5, 0.98),
                                                                          (1.5, 0.99)]
    assert corridor_variants(ctm.corridor) == [(2.0, 0.99)]
    with pytest.raises(ValueError, match="weekly_split"):
        corridor_variants(dataclasses.replace(ctm.corridor, daily_mode="quantile", kappa=None), [1.5], None)


def test_t1_corridor_strength_pairs_only_existing():
    """κ axis: q 0.99: κ 2 → 1.5, 1.5 → 1.25; q 0.98: 1.5 → 1.25 (κ 2 / q 0.98 absent) – 3 × 2 reactions; q axis:
    0.99 → 0.98 at κ 1.5 and 1.25 – 2 × 2. Other τ have only the file's corridor: no pairs there."""
    levels = (1.25, 1.5, 2.0)
    cells = []
    for f in (0.005, 0.01, 0.02):
        for r in ("md1", "md3", "md3+esc"):
            cells.append(_ccell(f, r, "off", 0.6, 0.1, 0.0))
            cells.append(_ccell(f, r, "on", 0.2 + 0.08 * 2 + 0.05, 0.1, 0.06, *FILE_KQ))
    for r in ("md3", "md3+esc"):
        for k in (1.25, 1.5):
            for q in (0.98, 0.99):
                U = 0.2 + 0.08 * levels.index(k) + (0.05 if q == 0.99 else 0)
                cf = 0.1 - 0.02 * levels.index(k) - (0.01 if q == 0.99 else 0)
                cells.append(_ccell(0.01, r, "on", U, 0.1, cf, k, q))
    res = ps.t1({"a": {"cells": cells}}, resolution=0.036)
    assert res["kappa"]["a"]["n_pairs"] == 6 and res["q"]["a"]["n_pairs"] == 4
    assert res["kappa"]["a"]["median_dU"] == pytest.approx(-0.08) and res["kappa"]["a"]["verdict"] == "supported"
    assert res["q"]["a"]["median_dU"] == pytest.approx(-0.05) and res["q"]["a"]["verdict"] == "supported"
    assert res["kappa"]["a"]["median_dfriction"] == pytest.approx(0.02)
    pairs = ps.variant_pairs(cells, "kappa")
    assert {(w["corridor_kappa"], w["corridor_q"], s["corridor_kappa"]) for w, s in pairs} == {
        (2.0, 0.99, 1.5), (1.5, 0.99, 1.25), (1.5, 0.98, 1.25)}
    # the Task 19 axes still see exactly the 18 cells with the file's corridor
    assert res["corridor"]["a"]["n_pairs"] == 9
    assert "kappa" not in ps.t1({"a": toy_grid()})  # no variants: Task 19 output unchanged


def test_front_toy_strict_dominance_ties_kept_r19_excluded():
    cells = [_ccell(0.01, "md3", "off", 0.30, 0.10, 0.0),  # A front
             _ccell(0.02, "md3", "off", 0.20, 0.20, 0.0),  # B front
             _ccell(0.02, "md3", "on", 0.20, 0.20, 0.05, *FILE_KQ),  # C: exact tie with B, kept
             _ccell(0.02, "md3+esc", "off", 0.25, 0.20, 0.0),  # D dominated by B
             _ccell(0.02, "md1", "off", 0.30, 0.30, 0.0),  # E dominated
             _ccell(0.01, "md2", "off", 0.05, 0.0, 0.0), _ccell(0.01, "mdinf", "off", 0.01, 0.0, 0.0)]  # R19: out
    front = ps.front_cells({"label": "a", "cells": cells})
    assert {c["key"] for c in front} == {cells[0]["key"], cells[1]["key"], cells[2]["key"]}


def _conf_runs(u_s: dict, attacks=("es", "random", "stepwise")) -> dict:
    """Runs whose per-seed max over the attacks is ``u_s[seed]`` (the other attacks below it)."""
    runs = {}
    for sd, u in u_s.items():
        for i, a in enumerate(attacks):
            runs[f"{a}/{sd}"] = {"uplift_upper": u - 0.01 * i}
    return runs


def test_confirmation_stats_and_margin():
    u_s = {4: 0.10, 5: 0.12, 6: 0.08, 7: 0.15, 8: 0.11, 9: 0.09}
    st = ps.confirmation_stats({"key": "k", "runs": _conf_runs(u_s)}, 0.12, ["es", "random", "stepwise"])
    assert st["U_s"] == {str(k): pytest.approx(v) for k, v in u_s.items()} and st["n_runs"] == 18
    assert st["median"] == pytest.approx(0.105) and st["worst"] == pytest.approx(0.15)
    assert st["x_c"] == pytest.approx(0.03) and len(st["run_upper"]) == 18
    with pytest.raises(ValueError, match="every attack"):
        ps.confirmation_stats({"key": "k", "runs": _conf_runs(u_s, ("es", "random"))}, 0.1, ["es", "random", "stepwise"])
    m = ps.margin({"a": [0.03, -0.01], "b": [0.05], "c": []})
    assert m["X"] == pytest.approx(0.05) and m["p90"] == pytest.approx(0.046) and m["n_confirmed"] == 3
    assert m["per_model"] == {"a": pytest.approx(0.03), "b": pytest.approx(0.05), "c": None}
    assert ps.margin({"a": []})["X"] is None


def test_passing_rule_boundaries():
    assert ps.passes(0.10, 0.14, 0.10)  # median = δ′ and worst = δ′ + 0.04 pass
    assert not ps.passes(0.10 + 1e-6, 0.12, 0.10) and not ps.passes(0.09, 0.14 + 1e-6, 0.10)
    for d in ps.DELTA_PRIMES:  # the boundary holds for every δ′ despite binary floats (0.30 + 0.04 < 0.34)
        assert ps.passes(d, round(d + 0.04, 2), d) and not ps.passes(d, round(d + 0.04, 2) + 1e-6, d)
    assert len(ps.DELTA_PRIMES) == 43 and ps.DELTA_PRIMES[0] == 0.08 and ps.DELTA_PRIMES[-1] == 0.5


def _toy_19b():
    """A model whose front is c1 (md1), c2 (md3+esc, corridor κ 1.25), c3 (md3, off); c4 dominated by c2. c1 and c2
    are confirmed on seeds 4–9, c3 is pending."""
    c1 = _ccell(0.02, "md1", "off", 0.08, 0.40, 0.0)
    c2 = _ccell(0.01, "md3+esc", "on", 0.12, 0.10, 0.07, 1.25, 0.98)
    c3 = _ccell(0.005, "md3", "off", 0.30, 0.05, 0.0)
    c4 = _ccell(0.01, "md3", "on", 0.15, 0.10, 0.06, 1.25, 0.98)
    gate = {"delta": 0.1, "attack_seeds": [1, 2, 3], "attacks": ["es", "random", "stepwise"]}
    grid = {"label": "a", "model_dir": "m", "model_json_sha256": "sha", "gate": gate, "cells": [c1, c2, c3, c4]}
    conf = {"label": "a", "model_json_sha256": "sha", "gate": dict(gate, attack_seeds=[4, 5, 6, 7, 8, 9]),
            "cells": [dict(c1, runs=_conf_runs({s: 0.09 for s in range(4, 10)})),
                      dict(c2, runs=_conf_runs(dict(zip(range(4, 10), (0.12, 0.13, 0.14, 0.15, 0.16, 0.20)))))]}
    return grid, conf, (c1, c2, c3, c4)


def test_front_analysis_and_appetite_curve():
    grid, conf, (c1, c2, c3, c4) = _toy_19b()
    fa = ps.front_analysis({"a": grid}, {"a": conf})
    rows = {r["key"]: r for r in fa["front"]["a"]}
    assert set(rows) == {c1["key"], c2["key"], c3["key"]}
    assert rows[c2["key"]]["confirmation"]["median"] == pytest.approx(0.145)
    assert rows[c2["key"]]["confirmation"]["x_c"] == pytest.approx(0.08)
    assert rows[c3["key"]]["confirmation"] is None and rows[c1["key"]]["md1"]
    assert fa["margin"]["X"] == pytest.approx(0.08)
    assert [p["key"] for p in fa["plan"]] == [c3["key"]] and fa["plan"][0]["kappa"] is None
    app = {round(e["delta"], 2): e for e in ps.appetite(fa["front"], {"a": grid}, fa["margin"]["X"])["a"]}
    key = lambda e: None if e is None else e["key"]  # noqa: E731
    assert key(app[0.08]["A_with_md1"]) is None  # nothing passes: «none»
    assert key(app[0.09]["A_with_md1"]) == c1["key"] and key(app[0.09]["A_without_md1"]) is None  # md1 excluded
    assert key(app[0.15]["A_without_md1"]) is None  # c2: median 14.5 % ≤ 15 % but worst 20 % > 19 %
    assert key(app[0.16]["A_with_md1"]) == c2["key"] == key(app[0.16]["A_without_md1"])  # fewer false blocks than c1
    assert app[0.16]["A_with_md1"]["median"] == pytest.approx(0.145)
    assert key(app[0.16]["B_with_md1"]) == c1["key"] and key(app[0.16]["B_without_md1"]) is None  # U ≤ 0.08
    assert key(app[0.20]["B_with_md1"]) == c2["key"]  # U ≤ 0.12: c2 is cheaper than c1
    assert key(app[0.38]["B_without_md1"]) == c3["key"]  # U ≤ 0.30: c3 has the fewest false blocks
    none_x = ps.appetite(fa["front"], {"a": grid}, None)["a"][0]
    assert none_x["B_with_md1"] is None and none_x["B_without_md1"] is None
    md = ps.appetite_markdown({"appetite": {"a": list(app.values())}, "margin": fa["margin"], "delta": 0.1})
    assert "gate δ = 10 %" in md and ps.MD1_NOTE in md and "| none |" in md


def test_confirmation_must_match_the_corridor_variant():
    grid, conf, (c1, c2, c3, c4) = _toy_19b()
    bad = dict(conf, cells=[dict(c2, corridor_kappa=1.5, runs=conf["cells"][1]["runs"])])
    with pytest.raises(ValueError, match="corridor_kappa"):
        ps.front_analysis({"a": grid}, {"a": bad})


def _sweep_file(label, cells, *, sha="m", own_tau=0.9, gate_sha="g", threat_sha="t", ctm_sha="c", git="abc1234567",
                cp=None):
    for c in cells:
        c["provenance"] = {"git_sha": git, "git_dirty": False, "host_machine": "box"}
    return {"label": label, "role": "grid", "model_dir": "m", "model_json_sha256": sha, "own_target_fpr": 0.01,
            "own_tau": own_tau, "esc_fpr": 0.05, "tau_esc": 0.1, "seed": 42, "taus": {"0.01": own_tau},
            "gate": {"delta": 0.1, "attack_seeds": [1], "attacks": ["es"]}, "corridor_params": cp or _cp(*FILE_KQ),
            "mean_amount": 100.0, "cells": cells, "friction": {"false_blocks": {}, "burden": {}},
            "meta": {"gate_sha256": gate_sha, "threat_sha256": threat_sha, "corridor_threat_sha256": ctm_sha,
                     "git_sha": git}}


def test_merge_task19_and_19b_sweeps_and_refusals():
    old = [_cell(0.01, "md3", "on", 0.2, 0.1, 0.05), _cell(0.01, "md3", "off", 0.3, 0.1, 0.0)]  # Task 19 records
    new = [dict(_cell(0.01, "md3", "on", 0.15, 0.1, 0.07), key=cell_key(0.01, "md3", "on", (1.25, 0.98)),
                corridor_params=_cp(1.25, 0.98), corridor_kappa=1.25, corridor_q=0.98)]
    a = ps.normalize(_sweep_file("x", old), "t19.json")
    b = ps.normalize(_sweep_file("x", new, git="def7654321"), "t19b.json")
    on = next(c for c in a["cells"] if c["corridor"] == "on")
    assert (on["corridor_kappa"], on["corridor_q"], on["corridor_variant"]) == (2.0, 0.99, False)
    m = ps.merge_sweeps([a, b])
    assert len(m["cells"]) == 3 and sum(c["corridor_variant"] for c in m["cells"]) == 1
    assert [x["file"] for x in m["sources"]] == ["t19.json", "t19b.json"]
    for kw, what in (({"sha": "other"}, "model_json_sha256"), ({"own_tau": 0.8}, "own_tau"),
                     ({"gate_sha": "g2"}, "gate_sha256"), ({"threat_sha": "t2"}, "threat_sha256"),
                     ({"ctm_sha": "c2"}, "corridor_threat_sha256"), ({"cp": _cp(2.0, 0.98)}, "corridor_params")):
        bad = ps.normalize(_sweep_file("x", json.loads(json.dumps(new)), **kw), "bad.json")
        with pytest.raises(ValueError, match=f"refusing to merge.*{what}"):
            ps.merge_sweeps([a, bad])
    with pytest.raises(ValueError, match="more than one file"):
        ps.merge_sweeps([a, ps.normalize(_sweep_file("x", json.loads(json.dumps(old))), "dup.json")])
    # two commits only when both are named
    prov = ps.provenances([m])
    assert {p["git_sha"] for p in prov} == {"abc1234567", "def7654321"}
    with pytest.raises(ValueError, match="more than one code version"):
        ps.check_provenance(prov)
    with pytest.raises(ValueError, match="more than one code version"):
        ps.check_provenance(prov, ["abc1234"])
    ps.check_provenance(prov, ["abc1234", "def7654321"])
    with pytest.raises(ValueError, match="more than one code version"):  # a dirty tree is never merged
        ps.check_provenance([dict(prov[0], git_dirty=True)], ["abc1234"])


def test_interaction_table_toy():
    base, hard = toy_grid(), toy_grid(tau_step=-0.02)
    t1_res = ps.t1({"m3-base": base, "m3-hard": hard, "solo": toy_grid()})
    it = ps.interaction_table(t1_res, {"m3-base": base, "m3-hard": hard, "solo": toy_grid()})
    assert it["families"] == {"m3": ["m3-base", "m3-hard"]}
    tau = next(r for r in it["rows"] if r["axis"] == "tau")
    assert tau["median_dU_base"] == pytest.approx(-0.05) and tau["median_dU_hardened"] == pytest.approx(-0.02)
    assert tau["difference"] == pytest.approx(0.03)
    assert (tau["verdict_base"], tau["verdict_hardened"]) == ("supported", "inconclusive")
    assert len(it["rows"]) == len(ps.AXES)
    ratios = [(c["U"] - 0.01) / c["U"] for c in base["cells"] if c["corridor"] == "off"]
    e = it["corridor_relative_effect"]["m3-base"]
    assert e["n_pairs"] == 9 and e["median_ratio"] == pytest.approx(float(np.median(ratios))) and e["n_skipped"] == 0
    md = "\n".join(ps.interaction_markdown(it))
    assert "Descriptive only" in md and "supported" in md and "interaction: none" not in md


def test_violation_counts_from_run_meta():
    from types import SimpleNamespace

    from fraudaudit.policy_sweep import violation_counts
    runs = {"es/1": SimpleNamespace(meta={"corridor": {"n_bound_violations": 1, "n_headroom_violations": 2,
                                                       "n_block_violations": 0}}),
            "es/2": SimpleNamespace(meta={"corridor": {"n_bound_violations": 0, "n_headroom_violations": 1}})}
    assert violation_counts(runs, True) == {"n_corridor_runs": 2, "n_bound_violations": 1, "n_headroom_violations": 3,
                                            "n_block_violations": 0}
    assert violation_counts(runs, False)["n_corridor_runs"] == 0
    with pytest.raises(RuntimeError):
        violation_counts({"es/1": SimpleNamespace(meta={})}, True)
    cells = [dict(_cell(0.01, "md3", "on", 0.1, 0.1, 0.1), n_corridor_runs=9, n_bound_violations=1,
                  n_headroom_violations=0, n_block_violations=2), _cell(0.01, "md1", "on", 0.1, 0.1, 0.1),
             dict(_cell(0.01, "md3", "off", 0.1, 0.1, 0.0), n_corridor_runs=0, n_bound_violations=None)]
    v = ps.violation_totals([{"cells": cells}])
    assert (v["n_cells"], v["n_corridor_runs"], v["n_bound_violations"], v["n_block_violations"]) == (1, 9, 1, 2)
    assert v["n_on_cells_without_counters"] == 1 and v["any"]
    line = ps.violations_line(v)
    assert line.startswith("**1 violations of the corridor bound in 9 corridor runs") and "not recomputed" in line


def test_r19_reports_the_difference_in_pp():
    r = ps.rel_overestimation(np.array([0.3, 0.5, 0.7]), np.array([0.2, 0.4, 0.6]))
    assert r["diff"] == pytest.approx(0.1) and r["diff_lower"] == pytest.approx(0.1) == r["diff_upper"]
    md = ps.r19_markdown({"m2-base": {"adv_es": {"md1": r}}})
    assert "10.00 [10.00; 10.00]" in md and "Spearman" in md and "Task 24" in md


@pytest.fixture(scope="module")
def vsweep(env, tmp_path_factory):
    """Two corridor-variant cells (κ 1.25 / 1.5, q 0.98) at the own τ, md3, run with a spy on run_cell."""
    from fraudaudit import policy_sweep as mod
    df, model, d = env
    tm, ctm, g = _setup(env)
    out = tmp_path_factory.mktemp("vsweep")
    seen, real = [], mod.run_cell

    def spy(df_, mdl, mdir, tm_c, *a, **k):
        c = tm_c.corridor
        seen.append({"kappa": c.kappa, "q": c.q, "floor_q": c.floor_fit.q, "split": c.split is not None,
                     "extends": c.confirmed_extends, "lookback": c.lookback_days, "md": tm_c.max_declines,
                     "public": c.public()})
        return real(df_, mdl, mdir, tm_c, *a, **k)

    mod.run_cell = spy
    try:
        data = mod.policy_sweep(df, model, d, "m", dataclasses.replace(tm, max_declines=7), ctm, g, fprs=(0.01,),
                                reactions=("md3",), corridors=("on",), kappas=(1.25, 1.5), qs=(0.98,), esc_fpr=0.2,
                                seed=1, n_jobs=1, out_dir=out / "m")
    finally:
        mod.run_cell = real
    return data, out, seen


def test_corridor_variant_reaches_the_cell(env, vsweep):
    from fraudaudit.policy_sweep import params_sha256
    data, _, seen = vsweep
    _, ctm, _ = _setup(env)
    assert sorted((s["kappa"], s["q"], s["floor_q"]) for s in seen) == [(1.25, 0.98, 0.98), (1.5, 0.98, 0.98)]
    assert all(s["split"] and s["extends"] and s["lookback"] == 60 and s["md"] == 3 for s in seen)
    assert {c["key"] for c in data["cells"]} == {cell_key(0.01, "md3", "on", (1.25, 0.98)),
                                                 cell_key(0.01, "md3", "on", (1.5, 0.98))}
    pub = {json.dumps(s["public"], sort_keys=True) for s in seen}
    for c in data["cells"]:
        assert set(c) == CELL_KEYS
        assert json.dumps(c["corridor_params"], sort_keys=True) in pub
        assert c["corridor_params_sha256"] == params_sha256(c["corridor_params"])
        assert (c["corridor_kappa"], c["corridor_q"]) == (c["corridor_params"]["daily"]["kappa"], 0.98)
        assert c["n_corridor_runs"] == len(c["runs"]) and isinstance(c["n_bound_violations"], int)
        assert c["friction"]["share_confirm_card_days"] >= 0  # the burden of the variant's own corridor
    # every other corridor parameter as in the file; the file's own corridor stays the sweep-level one
    other = {k: v for k, v in seen[0]["public"].items() if k not in ("q", "daily")}
    assert other == {k: v for k, v in data["corridor_params"].items() if k not in ("q", "daily")}
    assert data["corridor_params"]["q"] == ctm.corridor.q
    assert len(data["friction"]["burden"]) == 2  # one burden entry per variant


def test_corridor_variant_n_jobs_identical(env, vsweep, tmp_path):
    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    data, _, _ = vsweep
    par = policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01,), reactions=("md3",), corridors=("on",),
                       kappas=(1.25,), qs=(0.98,), esc_fpr=0.2, seed=1, n_jobs=2, out_dir=tmp_path / "p")
    a = next(c for c in data["cells"] if c["key"] == cell_key(0.01, "md3", "on", (1.25, 0.98)))
    b = par["cells"][0]
    for k in ("key", "config_hash", "U", "U_usd", "runs", "friction", "corridor_params", "n_bound_violations",
              "n_headroom_violations", "n_block_violations"):
        assert a[k] == b[k], k


def test_task19_records_resume_without_variant_fields(env, sweep, monkeypatch, tmp_path):
    """Records of Task 19 (no κ / q / violation fields) are skipped by a rerun with the same grid – with or without
    an explicit file variant – and their friction is reused."""
    from fraudaudit import policy_sweep as mod
    df, model, d = env
    tm, ctm, g = _setup(env)
    _, out = sweep
    raw = json.loads((out / "m" / "policy-sweep.json").read_text())
    for c in raw["cells"]:
        for k in ("corridor_kappa", "corridor_q", "corridor_params", "corridor_params_sha256", "n_corridor_runs",
                  "n_bound_violations", "n_headroom_violations", "n_block_violations"):
            c.pop(k)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "policy-sweep.json").write_text(json.dumps(raw))
    import shutil
    shutil.copy(out / "m" / mod.SCORES_FILE, tmp_path / "m" / mod.SCORES_FILE)
    calls = []
    real = mod.run_cell
    monkeypatch.setattr(mod, "run_cell", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for kw in ({}, {"kappas": (ctm.corridor.kappa,), "qs": (ctm.corridor.q,)}):
        again = mod.policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01, 0.05), reactions=("md1", "md3+esc"),
                                 corridors=("off", "on"), esc_fpr=0.2, r19=True, seed=1, n_jobs=1,
                                 out_dir=tmp_path / "m", **kw)
        assert calls == [] and {c["key"] for c in again["cells"]} == {c["key"] for c in raw["cells"]}
        assert all(c["friction"] is not None for c in again["cells"])
    assert mod.sweep_markdown(again)  # renders old records (no counters: n/a)


def test_policy_summary_front_mode_end_to_end(env, sweep, vsweep, tmp_path):
    """Task 19-like and Task 19b-like files of one label merged; the front confirmed by a confirmation file; every
    Task 19b output written."""
    from fraudaudit.cli import main
    _, out0 = sweep
    _, outv, _ = vsweep
    first = main(["policy-summary", str(out0 / "m" / "policy-sweep.json"), str(outv / "m" / "policy-sweep.json"),
                  "--out", str(tmp_path / "s0"), "--confirm-rule", "front"])
    assert first == 0
    plan = (tmp_path / "s0" / "confirmations-front.tsv").read_text().splitlines()
    front = json.loads((tmp_path / "s0" / "front.json").read_text())
    assert len(plan) == len(front["front"]["m"]) >= 1 and front["margin"]["X"] is None
    for line in plan:
        f = line.split("\t")
        assert len(f) == 8 and f[0] == "m" and (f[5] == "-") == (f[4] == "off")
    # a confirmation file for every front cell (fabricated runs on seed 2, disjoint from the grid's seed 1)
    grid = {c["key"]: c for p in (out0, outv) for c in json.loads((p / "m" / "policy-sweep.json").read_text())["cells"]}
    conf = json.loads((outv / "m" / "policy-sweep.json").read_text())
    conf["role"], conf["gate"] = "confirmation", dict(conf["gate"], attack_seeds=[2])
    conf["cells"] = [dict(grid[ln.split("\t")[7]], runs={k.replace("/1", "/2"): dict(r, uplift_upper=r["uplift_upper"]
                                                                                       + 0.01)
                                                         for k, r in grid[ln.split("\t")[7]]["runs"].items()})
                     for ln in plan]
    pc = tmp_path / "conf.json"
    pc.write_text(json.dumps(conf))
    code = main(["policy-summary", str(out0 / "m" / "policy-sweep.json"), str(outv / "m" / "policy-sweep.json"),
                 str(pc), "--out", str(tmp_path / "s"), "--confirm-rule", "front"])
    assert code == 0
    s = tmp_path / "s"
    for name in ("hypotheses.md", "summary.json", "front.md", "front.json", "appetite.md", "appetite.json",
                 "appetite.png", "front-blocks.png", "front-confirm.png"):
        assert (s / name).stat().st_size > 0, name
    assert (s / "confirmations-front.tsv").read_text() == ""
    res = json.loads((s / "summary.json").read_text())
    assert res["margin"]["X"] is not None and res["margin"]["n_confirmed"] == len(plan)
    assert "kappa" in res["t1"] and "t2" not in res and len(res["appetite"]["m"]) == 43
    n_on = sum(c["corridor"] == "on" for c in [*grid.values(), *conf["cells"]])
    assert res["violations"]["n_cells"] == n_on and res["violations"]["n_corridor_runs"] == 2 * n_on
    md = (s / "hypotheses.md").read_text()
    assert "Task 19b" in md and "corridor runs" in md and "Descriptive only" in md
