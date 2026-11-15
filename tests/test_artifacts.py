import json

import keras
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.model import artifacts
from fraudaudit.model.artifacts import ArtifactIntegrityError, load_artifacts, save_artifacts


@pytest.fixture
def parts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 35)).astype("float32")
    mlp = keras.Sequential([keras.Input(shape=(35,)), keras.layers.Dense(1, activation="sigmoid")])
    scaler = StandardScaler().fit(X)
    iforest = IsolationForest(n_estimators=5, random_state=0).fit(X)
    df = pd.DataFrame({"merchant": ["fraud_a", "fraud_b"], "state": ["NY", "CA"],
                       "category": ["travel", "home"], "gender": ["F", "M"]})
    enc = BaseEncoder().fit(df)
    return X, dict(mlp=mlp, scaler=scaler, iforest=iforest, encoder=enc, tau=0.37, meta={"seed": 42})


def test_roundtrip_same_predictions(tmp_path, parts):
    X, kw = parts
    save_artifacts(tmp_path, **kw)
    a = load_artifacts(tmp_path)
    assert np.allclose(kw["mlp"].predict(X, verbose=0), a.mlp.predict(X, verbose=0), atol=1e-6)
    assert np.allclose(kw["scaler"].transform(X), a.scaler.transform(X), atol=1e-6)
    assert np.allclose(kw["iforest"].score_samples(X), a.iforest.score_samples(X), atol=1e-9)
    assert a.encoder.to_json() == kw["encoder"].to_json()
    assert a.tau == 0.37 and a.meta == {"seed": 42}
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["feature_columns"] == list(FEATURE_COLUMNS)
    assert set(m["sha256"]) == {"mlp.keras", "scaler.skops", "iforest.skops", "encoder.json"}
    assert {"python", "tensorflow", "keras", "numpy", "scikit-learn", "skops", "pandas"} <= set(m["versions"])
    assert a.manifest == m


def test_manifest_is_stable(tmp_path, parts):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    text = (tmp_path / "manifest.json").read_text()
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("name", ["iforest.skops", "mlp.keras", "scaler.skops", "encoder.json"])
def test_tampered_file_rejected(tmp_path, parts, monkeypatch, name):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    p = tmp_path / name
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))

    def boom(*a, **k):
        raise AssertionError("deserializer called before integrity check")

    monkeypatch.setattr("skops.io.load", boom)
    monkeypatch.setattr("skops.io.get_untrusted_types", boom)
    monkeypatch.setattr("keras.models.load_model", boom)
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


def test_missing_file_rejected(tmp_path, parts):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    (tmp_path / "scaler.skops").unlink()
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


def test_unexpected_type_rejected(tmp_path, parts, monkeypatch):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    monkeypatch.setattr("skops.io.get_untrusted_types", lambda file: ["os.system"])
    with pytest.raises(ArtifactIntegrityError, match="os.system"):
        load_artifacts(tmp_path)


def test_no_pickle_files(tmp_path, parts):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    names = {p.name for p in tmp_path.iterdir()}
    assert not [n for n in names if n.endswith((".pkl", ".pickle", ".joblib"))]
    assert names == {"mlp.keras", "scaler.skops", "iforest.skops", "encoder.json", "manifest.json"}
    assert artifacts.TRUSTED_SKOPS_TYPES is not True
