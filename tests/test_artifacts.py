import hashlib
import json

import keras
import numpy as np
import pandas as pd
import pytest
import skops.io
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
    assert set(m) == {"format", "sha256"}
    assert set(m["sha256"]) == {"mlp.keras", "scaler.skops", "iforest.skops", "encoder.json", "model.json"}
    info = json.loads((tmp_path / "model.json").read_text())
    assert info["feature_columns"] == list(FEATURE_COLUMNS)
    assert {"python", "tensorflow", "keras", "numpy", "scikit-learn", "scipy", "skops", "pandas"} <= set(info["versions"])
    assert a.manifest == m


def test_manifest_is_stable(tmp_path, parts):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    text = (tmp_path / "manifest.json").read_text()
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("name", ["iforest.skops", "mlp.keras", "scaler.skops", "encoder.json", "model.json"])
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
    assert names == {"mlp.keras", "scaler.skops", "iforest.skops", "encoder.json", "model.json",
                     "manifest.json"}


def test_skops_load_gets_explicit_list(tmp_path, parts, monkeypatch):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    seen = []
    real = skops.io.load
    monkeypatch.setattr("skops.io.load", lambda p, trusted=None: (seen.append(trusted), real(p, trusted=trusted))[1])
    load_artifacts(tmp_path)
    assert seen == [list(artifacts.TRUSTED_SKOPS_TYPES)] * 2
    assert all(isinstance(x, list) for x in seen)


def _rehash(d, *names):
    m = json.loads((d / "manifest.json").read_text())
    for n in names:
        m["sha256"][n] = hashlib.sha256((d / n).read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def _edit_model(d, **changes):
    info = json.loads((d / "model.json").read_text())
    info.update(changes)
    (d / "model.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def test_tau_tamper_rejected(tmp_path, parts):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    _edit_model(tmp_path, tau=0.99)  # hash not updated
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_artifacts(tmp_path)


def test_feature_columns_mismatch_rejected(tmp_path, parts):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    _edit_model(tmp_path, feature_columns=list(reversed(FEATURE_COLUMNS)))
    _rehash(tmp_path, "model.json")
    with pytest.raises(ArtifactIntegrityError, match="feature_columns"):
        load_artifacts(tmp_path)


class Custom:
    def __init__(self):
        self.x = 1


def _save_untrusted_iforest(d, kw):
    save_artifacts(d, **kw)
    bad = StandardScaler().fit(np.zeros((3, 2)))
    bad.extra_ = Custom()
    skops.io.dump(bad, d / "iforest.skops")
    _rehash(d, "iforest.skops")
    return skops.io.get_untrusted_types(file=d / "iforest.skops")


def test_genuinely_untrusted_type_rejected(tmp_path, parts, monkeypatch):
    _, kw = parts
    types = _save_untrusted_iforest(tmp_path, kw)
    assert any(t.endswith("Custom") for t in types)
    real = skops.io.load

    def boom(*a, **k):
        raise AssertionError("skops.io.load called for untrusted type")

    # scaler.skops is fine and loaded first; only the iforest load must not happen
    calls = []
    monkeypatch.setattr("skops.io.load", lambda p, trusted=None: (calls.append(str(p)), real(p, trusted=trusted))[1])
    with pytest.raises(ArtifactIntegrityError, match="Custom"):
        load_artifacts(tmp_path)
    assert not any(c.endswith("iforest.skops") for c in calls)


def test_allow_list_is_honoured(tmp_path, parts, monkeypatch):
    _, kw = parts
    types = _save_untrusted_iforest(tmp_path, kw)
    monkeypatch.setattr(artifacts, "TRUSTED_SKOPS_TYPES", tuple(types))
    a = load_artifacts(tmp_path)
    assert isinstance(a.iforest.extra_, Custom)


GOOD_SHA = "0" * 64


@pytest.mark.parametrize("manifest", [
    "[]", "not json", "{}", '{"format": 2, "sha256": {}}', '{"format": 1}',
    '{"format": 1, "sha256": []}', '{"format": 1, "sha256": {"mlp.keras": 1}}',
])
def test_malformed_manifest_rejected(tmp_path, parts, manifest):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    (tmp_path / "manifest.json").write_text(manifest)
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


@pytest.mark.parametrize("changes", [
    {"tau": "x"}, {"tau": None}, {"tau": 1.5}, {"tau": -0.1}, {"tau": True},
    {"tau": float("nan")}, {"tau": float("inf")}, {"meta": []}, {"versions": "x"},
    {"feature_columns": None},
])
def test_malformed_model_json_rejected(tmp_path, parts, changes):
    _, kw = parts
    save_artifacts(tmp_path, **kw)
    _edit_model(tmp_path, **changes)
    _rehash(tmp_path, "model.json")
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


@pytest.mark.parametrize("content", ["[]", "{}", "garbage"])
def test_malformed_model_and_encoder_rejected(tmp_path, parts, content):
    _, kw = parts
    for name in ("model.json", "encoder.json"):
        save_artifacts(tmp_path, **kw)
        (tmp_path / name).write_text(content)
        _rehash(tmp_path, name)
        with pytest.raises(ArtifactIntegrityError):
            load_artifacts(tmp_path)
