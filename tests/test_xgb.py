"""XGBoost model kind (Task 24, R35): artifact format, trainer, selection on clean val, shared interface, hardening."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_sparkov_like
from fraudaudit import harden as hmod
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.gate import GateConfig
from fraudaudit.model import xgb as xmod
from fraudaudit.model.artifacts import ArtifactIntegrityError, derive_with_tau, load_artifacts
from fraudaudit.model.pipeline import FraudModel, load_model
from fraudaudit.model.xgb import XGBFraudModel, pick_best, train_xgb_model
from fraudaudit.skew import skew_check
from fraudaudit.threat import ThreatModel

ROOT = Path(__file__).resolve().parents[1]
TINY = {"max_depth": 2, "n_estimators": 8}


@pytest.fixture(scope="module")
def df():
    return make_sparkov_like(n_cards=160, days=500, seed=3, start=pd.Timestamp("2019-04-01"))


@pytest.fixture(scope="module")
def trained(df):
    return train_xgb_model(df, seed=0, params=TINY)


@pytest.fixture(scope="module")
def model_dir(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("xgb")
    trained[0].save(d, meta=dict(trained[1], seed=0))
    return d


def test_manifest_and_files(model_dir):
    m = json.loads((model_dir / "manifest.json").read_text())
    assert m["kind"] == "xgb" and m["format"] == 1
    assert set(m["sha256"]) == {"xgb.json", "encoder.json", "model.json"}
    assert json.loads((model_dir / "model.json").read_text())["kind"] == "xgb"
    assert not any(p.suffix in {".pkl", ".pickle", ".joblib"} for p in model_dir.iterdir())
    json.loads((model_dir / "xgb.json").read_text())  # the model is plain JSON, not a pickle
    info = json.loads((model_dir / "model.json").read_text())
    assert "xgboost" in info["versions"]


def test_roundtrip_same_predictions(trained, model_dir, df):
    model, report = trained
    a = load_artifacts(model_dir)
    assert a.kind == "xgb" and a.mlp is None and a.scaler is None and a.iforest is None
    loaded = load_model(model_dir)
    assert isinstance(loaded, XGBFraudModel) and loaded.kind == "xgb"
    F = np.random.default_rng(0).normal(size=(50, len(FEATURE_COLUMNS))).astype("float32")
    np.testing.assert_array_equal(model.predict_proba(F), loaded.predict_proba(F))
    assert loaded.tau == model.tau and loaded.params == model.params
    assert loaded.params == {**xmod.FIXED_PARAMS, **TINY}
    assert report["xgb"]["params"] == loaded.params


def test_history_is_ignored_and_shape_checked(trained):
    model, _ = trained
    F = np.random.default_rng(1).normal(size=(5, len(FEATURE_COLUMNS))).astype("float32")
    np.testing.assert_array_equal(model.predict_proba(F), model.predict_proba(F, history=np.zeros((5, 3, 38))))
    with pytest.raises(ValueError):
        model.predict_proba(F[:, :10])
    p = model.predict_proba(F)
    assert p.dtype == np.float64 and p.shape == (5,) and ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("name", ["xgb.json", "encoder.json", "model.json"])
def test_tampered_file_rejected_before_deserializing(trained, tmp_path, monkeypatch, name):
    trained[0].save(tmp_path, meta={"seed": 0})
    p = tmp_path / name
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))

    def boom(*a, **k):
        raise AssertionError("deserializer called before integrity check")

    import xgboost
    monkeypatch.setattr(xgboost.Booster, "load_model", boom)
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


def test_manifest_kind_must_match_model_json(trained, tmp_path):
    trained[0].save(tmp_path, meta={"seed": 0})
    m = json.loads((tmp_path / "manifest.json").read_text())
    for bad in ({"kind": "mlp"}, {"kind": "gru"}):
        (tmp_path / "manifest.json").write_text(json.dumps({**m, **bad}))
        with pytest.raises(ArtifactIntegrityError):
            load_artifacts(tmp_path)
    del m["kind"]  # an xgb manifest without kind = an MLP manifest listing the wrong files
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


def test_committed_mlp_model_loads_unchanged():
    d = ROOT / "models"
    m = json.loads((d / "manifest.json").read_text())
    assert "kind" not in m and set(m) == {"format", "sha256"}  # the baseline artifact format is untouched
    assert "kind" not in json.loads((d / "model.json").read_text())
    assert type(load_model(d)) is FraudModel and load_artifacts(d).kind == "mlp"


def test_params_validated_on_load(trained, tmp_path):
    model, _ = trained
    keep = model.params
    model.params = {**keep, "max_depth": "deep"}
    try:
        model.save(tmp_path, meta={})
    finally:
        model.params = keep
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


def test_derive_with_tau_keeps_kind(model_dir, tmp_path):
    derive_with_tau(model_dir, tmp_path / "d", 0.123, {"target_fpr": 0.02, "note": "t"})
    a = load_artifacts(tmp_path / "d")
    assert a.kind == "xgb" and a.tau == 0.123 and a.meta["target_fpr"] == 0.02
    assert (tmp_path / "d" / "xgb.json").read_bytes() == (model_dir / "xgb.json").read_bytes()
    assert load_model(tmp_path / "d").tau == 0.123


def test_train_report_and_selection_inputs(trained, df):
    model, report = trained
    assert report["feature_set"] == "bachelor+hour-calendar+behav" and report["model_kind"] == "xgb"
    for name in ("val", "test"):
        assert {"recall", "fpr", "precision", "n", "n_fraud"} <= set(report[name])
    assert 0.0 <= model.tau <= 1.0 and report["target_fpr"] == 0.01
    assert report["xgb"]["selection"] is None  # params were given: no grid search
    assert "scaler" not in report and report["xgb"]["n_threads"] >= 1


def test_selection_runs_grid_on_val_only_and_records_it(df):
    grid = ((2, 5), (2, 9))
    model, report = train_xgb_model(df, seed=0, grid=grid)
    sel = report["xgb"]["selection"]
    assert [(s["max_depth"], s["n_estimators"]) for s in sel] == list(grid)
    assert all({"recall", "fpr", "tau"} <= set(s) for s in sel)
    assert report["xgb"]["selected"] in [{"max_depth": d, "n_estimators": n} for d, n in grid]
    assert report["xgb"]["selection_data"] == "val (clean)"


def test_pick_best_rule():
    rows = [{"max_depth": 6, "n_estimators": 400, "recall": 0.80},
            {"max_depth": 4, "n_estimators": 200, "recall": 0.78},
            {"max_depth": 8, "n_estimators": 200, "recall": 0.80},
            {"max_depth": 4, "n_estimators": 400, "recall": 0.80}]
    best = pick_best(rows)  # highest recall; the tie goes to the smaller model (fewest leaves: n_est * 2**depth)
    assert (best["max_depth"], best["n_estimators"]) == (4, 400)
    assert pick_best(rows[:2])["max_depth"] == 6
    assert pick_best([{"max_depth": 8, "n_estimators": 200, "recall": 0.5},
                      {"max_depth": 6, "n_estimators": 400, "recall": 0.5}])["max_depth"] == 6
    assert xmod.XGB_GRID == tuple((d, n) for d in (4, 6, 8) for n in (200, 400))


def test_fixed_hyperparameters():
    assert xmod.FIXED_PARAMS == {"learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                                 "tree_method": "hist", "objective": "binary:logistic"}


def test_training_uses_balanced_fit_rows_only(df, monkeypatch):
    seen = {}
    real = xmod._fit_booster

    def spy(X, y, params, seed, n_threads):
        seen["n"], seen["pos"], seen["cols"] = len(y), int(y.sum()), X.shape[1]
        return real(X, y, params, seed, n_threads)

    monkeypatch.setattr(xmod, "_fit_booster", spy)
    train_xgb_model(df, seed=0, params=TINY)
    assert seen["cols"] == 38 and seen["pos"] * 2 == seen["n"]  # balanced, 38 FEATURE_COLUMNS, no flag column


def test_same_seed_same_model_bytes(df, tmp_path):
    a, _ = train_xgb_model(df, seed=3, params=TINY)
    b, _ = train_xgb_model(df, seed=3, params=TINY)
    a.save(tmp_path / "a", meta={})
    b.save(tmp_path / "b", meta={})
    assert (tmp_path / "a" / "xgb.json").read_bytes() == (tmp_path / "b" / "xgb.json").read_bytes()


def test_skew_serving_equals_batch_and_n_jobs_identical(model_dir, df):
    m = load_model(model_dir)
    a = skew_check(df, m, n_cards=6, seed=7, n_jobs=1, model_dir=model_dir)
    b = skew_check(df, m, n_cards=6, seed=7, n_jobs=2, model_dir=model_dir)
    assert a == b and a.passed and a.decision_mismatches == 0 and a.max_abs_p_diff == 0.0


def test_harden_round_with_xgb(df, model_dir, tmp_path):
    from fraudaudit.model.pipeline import load_model as lm
    model = lm(model_dir)
    tm = dataclasses.replace(ThreatModel.from_yaml(ROOT / "threat-models" / "realistic.yaml"),
                             recon_budget_ops=40, cards_per_eval=2)
    cfg = GateConfig(target_fpr=0.01, recall_clean_min=0.5, delta=None, attacks=["random", "es", "stepwise"],
                     attack_seeds=[1, 2, 3], confidence=0.95, one_sided=True, spares_period="fit")
    final, reports, info = hmod.harden(df, model, tm, cfg, rounds=2, seed=7, model_dir=model_dir, out_dir=tmp_path,
                                       n_jobs=1, budget_ops=40, gen_seeds=(1,), gen_max_declines=(1, 3),
                                       mini_audit_seeds=(1,), baseline_audit=False)
    assert isinstance(final, XGBFraudModel) and [r.round for r in reports] == [1, 2]
    assert final.params == model.params  # hyperparameters stay the clean-val choice, no re-selection on attacked data
    a = load_artifacts(tmp_path / "round-2")
    assert a.kind == "xgb" and a.meta["harden"]["round"] == 2 and "test" not in a.meta
    assert a.meta["xgb"]["selection"] is None
    assert a.meta["fraud_rows"]["fit"] > int(((hmod.period(df) == "fit") & (df["is_fraud"] == 1)).sum())


def test_unknown_kinds_are_refused(trained, tmp_path):
    trained[0].save(tmp_path, meta={})
    m = json.loads((tmp_path / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps({**m, "kind": ["xgb"]}))
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


def test_xgb_params_cli_validation():
    from fraudaudit import cli
    ok = cli.build_parser().parse_args(["train", "--model-kind", "xgb", "--xgb-params", "6,200"])
    assert ok.xgb_params == {"max_depth": 6, "n_estimators": 200}
    for bad in ("6", "6,200,1", "0,200", "a,b"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["train", "--model-kind", "xgb", "--xgb-params", bad])


def test_non_finite_grid_tau_is_stored_as_null(df, monkeypatch):
    real = xmod.threshold_at_fpr
    calls = {"n": 0}

    def flaky(y, p, fpr):
        calls["n"] += 1
        return float("inf") if calls["n"] == 1 else real(y, p, fpr)

    monkeypatch.setattr(xmod, "threshold_at_fpr", flaky)
    _, rep = train_xgb_model(df, seed=0, grid=((2, 5), (2, 9)))
    assert rep["xgb"]["selection"][0]["tau"] is None
    json.dumps(rep, allow_nan=False)
