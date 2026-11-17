"""Task 19: models trained on a registered feature subset (the 34-feature M2 baseline) load and score again."""
import hashlib
import json

import keras
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import (AGG_COLUMNS, BASE_COLUMNS, FEATURE_COLUMNS, FEATURE_SETS, column_index,
                                         feature_set_name)
from fraudaudit.model.artifacts import ArtifactIntegrityError, load_artifacts, save_artifacts
from fraudaudit.model.pipeline import FraudModel, build_datasets, build_mlp

M2 = FEATURE_SETS["bachelor+hour-calendar"]


def test_registry_sets_are_ordered_subsets():
    assert len(M2) == 34 and M2 == BASE_COLUMNS + AGG_COLUMNS[:7]
    assert FEATURE_SETS["bachelor+hour-calendar+behav"] == FEATURE_COLUMNS
    for cols in FEATURE_SETS.values():
        idx = column_index(cols)
        assert list(idx) == sorted(idx) and [FEATURE_COLUMNS[i] for i in idx] == list(cols)
    assert feature_set_name(list(M2)) == "bachelor+hour-calendar"
    with pytest.raises(ValueError, match="registered"):
        feature_set_name(list(reversed(M2)))
    with pytest.raises(ValueError, match="registered"):
        feature_set_name(list(M2[:-1]) + ["cnt_1h"])


@pytest.fixture
def m2_parts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(M2))).astype("float32")
    mlp = keras.Sequential([keras.Input(shape=(len(M2) + 1,)), keras.layers.Dense(1, activation="sigmoid")])
    scaler = StandardScaler().fit(X)
    iforest = IsolationForest(n_estimators=5, random_state=0).fit(scaler.transform(X))
    df = pd.DataFrame({"merchant": ["fraud_a", "fraud_b"], "state": ["NY", "CA"],
                       "category": ["travel", "home"], "gender": ["F", "M"]})
    enc = BaseEncoder().fit(df)
    return dict(mlp=mlp, scaler=scaler, iforest=iforest, encoder=enc, tau=0.37, meta={"seed": 42},
                feature_columns=M2)


def _edit_and_rehash(d, **changes):
    info = json.loads((d / "model.json").read_text())
    info.update(changes)
    (d / "model.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    m = json.loads((d / "manifest.json").read_text())
    m["sha256"]["model.json"] = hashlib.sha256((d / "model.json").read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def test_34_column_model_loads_and_selects_by_index(tmp_path, m2_parts):
    save_artifacts(tmp_path, **m2_parts)
    a = load_artifacts(tmp_path)
    assert a.feature_columns == M2
    model = FraudModel.from_artifacts(a)
    F = np.random.default_rng(1).normal(size=(17, len(FEATURE_COLUMNS))).astype("float32")
    sub = F[:, list(column_index(M2))]
    Xs = m2_parts["scaler"].transform(sub)
    flag = (m2_parts["iforest"].predict(Xs) > -1).astype("float32")[:, None]
    want = m2_parts["mlp"](np.hstack([Xs, flag]).astype("float32"), training=False).numpy().reshape(-1)
    assert np.allclose(model.predict_proba(F), want, atol=1e-6)
    with pytest.raises(ValueError, match="expected shape"):
        model.predict_proba(sub)  # the model takes the 38 computed features, not its own 34


@pytest.mark.parametrize("cols", [list(reversed(M2)), list(M2[:-1]) + ["unknown"], list(M2) + ["cnt_1h"],
                                  list(M2[1:])])
def test_reordered_or_unknown_columns_rejected(tmp_path, m2_parts, cols):
    save_artifacts(tmp_path, **m2_parts)
    _edit_and_rehash(tmp_path, feature_columns=cols)
    with pytest.raises(ArtifactIntegrityError, match="feature_columns"):
        load_artifacts(tmp_path)


def test_unregistered_set_cannot_be_saved(tmp_path, m2_parts):
    with pytest.raises(ValueError, match="registered"):
        save_artifacts(tmp_path, **dict(m2_parts, feature_columns=tuple(reversed(M2))))


@pytest.mark.parametrize("name", ["mlp.keras", "model.json", "scaler.skops"])
def test_tampered_subset_model_rejected_before_deserialisation(tmp_path, m2_parts, monkeypatch, name):
    save_artifacts(tmp_path, **m2_parts)
    p = tmp_path / name
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))

    def boom(*a, **k):
        raise AssertionError("deserializer called before integrity check")

    monkeypatch.setattr("skops.io.load", boom)
    monkeypatch.setattr("keras.models.load_model", boom)
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_artifacts(tmp_path)


@pytest.fixture
def m2_model_dir(sparkov_like, tmp_path):
    """A 34-feature model fitted on the synthetic set (scaler/IsolationForest on the M2 columns, untrained MLP)."""
    keras.utils.set_random_seed(0)
    ds = build_datasets(sparkov_like, seed=0)
    idx = list(column_index(M2))
    Ffit = ds.F["fit"][:, idx]
    scaler = StandardScaler().fit(Ffit)
    iforest = IsolationForest(n_estimators=10, random_state=0).fit(scaler.transform(Ffit))
    save_artifacts(tmp_path / "m2", mlp=build_mlp(len(M2) + 1), scaler=scaler, iforest=iforest,
                   encoder=ds.encoder, tau=0.5, meta={"seed": 0, "target_fpr": 0.01}, feature_columns=M2)
    return tmp_path / "m2"


def test_batch_and_serving_parity_for_subset_model(sparkov_like, m2_model_dir):
    from fraudaudit.skew import skew_check
    model = FraudModel.from_artifacts(load_artifacts(m2_model_dir))
    assert len(model.feature_columns) == 34
    rep = skew_check(sparkov_like, model, n_cards=8, seed=0, n_jobs=1)
    assert rep.passed and rep.decision_mismatches == 0 and rep.max_abs_p_diff <= 1e-4


def test_harden_refuses_subset_model(sparkov_like, m2_model_dir, tmp_path):
    from fraudaudit.gate import GateConfig
    from fraudaudit.harden import harden
    from fraudaudit.threat import ThreatModel
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    model = FraudModel.from_artifacts(load_artifacts(m2_model_dir))
    with pytest.raises(ValueError, match="feature set"):
        harden(sparkov_like, model, ThreatModel.from_yaml(root / "threat-models/realistic.yaml"),
               GateConfig.from_yaml(root / "gate.yaml"), model_dir=m2_model_dir, out_dir=tmp_path / "h", n_jobs=1)
