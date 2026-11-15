"""Save and load the audited model without pickle.

Keras goes to ``.keras`` (loaded with ``safe_mode=True``), scikit-learn objects to
``.skops`` (loaded with an explicit allow-list of types), the rest to JSON. A SHA-256
manifest is checked on the raw bytes BEFORE any deserializer runs, because the files
are committed to git and loaded in CI.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import skops
import skops.io as sio
import sklearn
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..features.base import BaseEncoder
from ..features.columns import FEATURE_COLUMNS

MLP_FILE = "mlp.keras"
SCALER_FILE = "scaler.skops"
IFOREST_FILE = "iforest.skops"
ENCODER_FILE = "encoder.json"
MANIFEST_FILE = "manifest.json"
HASHED_FILES = (MLP_FILE, SCALER_FILE, IFOREST_FILE, ENCODER_FILE)

# Types outside skops' built-in safe set that StandardScaler and IsolationForest
# (with its ExtraTreeRegressor estimators and sklearn.tree._tree.Tree) may need.
# Derived with skops.io.get_untrusted_types on freshly saved objects (skops 0.10.0,
# scikit-learn 1.5.2): the list is empty, as skops already treats these sklearn and
# numpy types as trusted. Anything NOT listed here makes loading fail.
TRUSTED_SKOPS_TYPES: tuple[str, ...] = ()


class ArtifactIntegrityError(RuntimeError):
    """An artifact is missing, altered, or contains a type outside the allow-list."""


@dataclass
class Artifacts:
    mlp: keras.Model
    scaler: StandardScaler
    iforest: IsolationForest
    encoder: BaseEncoder
    tau: float
    meta: dict
    manifest: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "skops": skops.__version__,
        "pandas": pd.__version__,
    }


def save_artifacts(out_dir: Path, *, mlp: keras.Model, scaler: StandardScaler,
                   iforest: IsolationForest, encoder: BaseEncoder, tau: float, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp.save(out_dir / MLP_FILE)
    sio.dump(scaler, out_dir / SCALER_FILE)
    sio.dump(iforest, out_dir / IFOREST_FILE)
    (out_dir / ENCODER_FILE).write_text(
        json.dumps(encoder.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "tau": float(tau),
        "feature_columns": list(FEATURE_COLUMNS),
        "versions": _versions(),
        "meta": meta,
        "sha256": {name: _sha256(out_dir / name) for name in HASHED_FILES},
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify(model_dir: Path, manifest: dict) -> None:
    expected = manifest.get("sha256", {})
    if set(expected) != set(HASHED_FILES):
        raise ArtifactIntegrityError(f"manifest lists {sorted(expected)}, expected {sorted(HASHED_FILES)}")
    for name in HASHED_FILES:
        path = model_dir / name
        if not path.is_file():
            raise ArtifactIntegrityError(f"missing artifact: {name}")
        if _sha256(path) != expected[name]:
            raise ArtifactIntegrityError(f"SHA-256 mismatch for {name}")


def _load_skops(path: Path):
    extra = set(sio.get_untrusted_types(file=path)) - set(TRUSTED_SKOPS_TYPES)
    if extra:
        raise ArtifactIntegrityError(f"{path.name} contains untrusted types: {sorted(extra)}")
    return sio.load(path, trusted=list(TRUSTED_SKOPS_TYPES))


def load_artifacts(model_dir: Path) -> Artifacts:
    model_dir = Path(model_dir)
    try:
        manifest = json.loads((model_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ArtifactIntegrityError(f"cannot read {MANIFEST_FILE}: {e}") from e
    _verify(model_dir, manifest)  # raw bytes first, nothing is deserialized before this
    mlp = keras.models.load_model(model_dir / MLP_FILE, safe_mode=True, compile=False)
    scaler = _load_skops(model_dir / SCALER_FILE)
    iforest = _load_skops(model_dir / IFOREST_FILE)
    encoder = BaseEncoder.from_json(
        json.loads((model_dir / ENCODER_FILE).read_text(encoding="utf-8")))
    return Artifacts(mlp=mlp, scaler=scaler, iforest=iforest, encoder=encoder,
                     tau=float(manifest["tau"]), meta=manifest["meta"], manifest=manifest)
