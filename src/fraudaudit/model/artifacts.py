"""Save and load the audited model without pickle.

Keras goes to ``.keras`` (loaded with ``safe_mode=True``), scikit-learn objects to
``.skops`` (loaded with an explicit allow-list of types), the rest to JSON. A SHA-256
manifest is checked on the raw bytes BEFORE any deserializer runs, because the files
are committed to git and loaded in CI.

Hashes protect against corruption and partial file swaps. Authenticity of a committed
change comes from PR review and the audit gate itself, not from the manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import scipy
import skops
import skops.io as sio
import sklearn
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..features.base import BaseEncoder
from ..features.columns import FEATURE_COLUMNS, feature_set_name

MLP_FILE = "mlp.keras"
SCALER_FILE = "scaler.skops"
IFOREST_FILE = "iforest.skops"
ENCODER_FILE = "encoder.json"
XGB_FILE = "xgb.json"  # XGBoost's own JSON model format: plain text, no pickle
GRU_FILE = "gru.keras"
MODEL_FILE = "model.json"  # tau, feature_columns, versions, meta: hashed, unlike manifest.json
MANIFEST_FILE = "manifest.json"
HASHED_FILES = (MLP_FILE, SCALER_FILE, IFOREST_FILE, ENCODER_FILE, MODEL_FILE)  # the MLP kind (the default)
MANIFEST_FORMAT = 1
# Model kinds (Task 24). The MLP kind is the original format and carries NO ``kind`` field (the committed ``models/``
# stays byte for byte as it is); every other kind writes ``kind`` into manifest.json AND model.json, and the two must
# agree. The manifest is read first (to know which files to hash), model.json is hash-checked and must confirm it.
KIND_FILES: dict[str, tuple[str, ...]] = {
    "mlp": HASHED_FILES,
    "xgb": (XGB_FILE, ENCODER_FILE, MODEL_FILE),
    "gru": (GRU_FILE, SCALER_FILE, ENCODER_FILE, MODEL_FILE),
}

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
    mlp: keras.Model | None  # None unless kind == "mlp"
    scaler: StandardScaler | None
    iforest: IsolationForest | None
    encoder: BaseEncoder
    tau: float
    meta: dict
    manifest: dict
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS  # a registered feature set (``columns.FEATURE_SETS``)
    kind: str = "mlp"
    xgb: object | None = None  # xgboost.Booster when kind == "xgb"
    xgb_params: dict | None = None
    gru: keras.Model | None = None  # when kind == "gru" (the scaler is in ``scaler``)
    gru_params: dict | None = None


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
        "scipy": scipy.__version__,
        "skops": skops.__version__,
        "pandas": pd.__version__,
    }


def _versions_xgb() -> dict:
    import xgboost
    return {**_versions(), "xgboost": xgboost.__version__}


def save_artifacts(out_dir: Path, *, mlp: keras.Model, scaler: StandardScaler,
                   iforest: IsolationForest, encoder: BaseEncoder, tau: float, meta: dict,
                   feature_columns=FEATURE_COLUMNS) -> None:
    feature_set_name(feature_columns)  # only a registered feature set can be saved
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp.save(out_dir / MLP_FILE)
    sio.dump(scaler, out_dir / SCALER_FILE)
    sio.dump(iforest, out_dir / IFOREST_FILE)
    (out_dir / ENCODER_FILE).write_text(
        json.dumps(encoder.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_info = {
        "tau": float(tau),
        "feature_columns": list(feature_columns),
        "versions": _versions(),
        "meta": meta,
    }
    (out_dir / MODEL_FILE).write_text(
        json.dumps(model_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "format": MANIFEST_FORMAT,
        "sha256": {name: _sha256(out_dir / name) for name in HASHED_FILES},
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_xgb_artifacts(out_dir: Path, *, booster, encoder: BaseEncoder, tau: float, meta: dict, params: dict,
                       feature_columns=FEATURE_COLUMNS) -> None:
    """XGBoost kind: ``xgb.json`` (the booster), ``encoder.json``, ``model.json`` (with ``kind``, the chosen
    hyperparameters ``xgb_params``) and a manifest with ``kind``. No scaler, no IsolationForest flag."""
    feature_set_name(feature_columns)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / XGB_FILE))
    (out_dir / ENCODER_FILE).write_text(
        json.dumps(encoder.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_info = {
        "kind": "xgb",
        "tau": float(tau),
        "feature_columns": list(feature_columns),
        "xgb_params": dict(params),
        "versions": _versions_xgb(),
        "meta": meta,
    }
    (out_dir / MODEL_FILE).write_text(
        json.dumps(model_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"format": MANIFEST_FORMAT, "kind": "xgb",
                "sha256": {name: _sha256(out_dir / name) for name in KIND_FILES["xgb"]}}
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_gru_artifacts(out_dir: Path, *, net: keras.Model, scaler: StandardScaler, encoder: BaseEncoder, tau: float,
                       meta: dict, params: dict, feature_columns=FEATURE_COLUMNS) -> None:
    """GRU kind: ``gru.keras`` (loaded with ``safe_mode=True``), ``scaler.skops``, ``encoder.json``, ``model.json``
    (with ``kind`` and the chosen window/size ``gru_params`` = ``{"k", "hidden"}``) and a manifest with ``kind``. No
    IsolationForest flag."""
    feature_set_name(feature_columns)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net.save(out_dir / GRU_FILE)
    sio.dump(scaler, out_dir / SCALER_FILE)
    (out_dir / ENCODER_FILE).write_text(
        json.dumps(encoder.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_info = {
        "kind": "gru",
        "tau": float(tau),
        "feature_columns": list(feature_columns),
        "gru_params": {"k": int(params["k"]), "hidden": int(params["hidden"])},
        "versions": _versions(),
        "meta": meta,
    }
    (out_dir / MODEL_FILE).write_text(
        json.dumps(model_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"format": MANIFEST_FORMAT, "kind": "gru",
                "sha256": {name: _sha256(out_dir / name) for name in KIND_FILES["gru"]}}
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ArtifactIntegrityError(f"cannot read {path.name}: {e}") from e
    if not isinstance(data, dict):
        raise ArtifactIntegrityError(f"{path.name} is not a JSON object")
    return data


def _check_manifest(manifest: dict) -> tuple[str, dict]:
    """(kind, expected hashes). No ``kind`` in the manifest = the original MLP format; an explicit ``"mlp"`` is
    refused (the MLP kind never writes it, so it would be a hand-edited manifest)."""
    expected = manifest.get("sha256")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ArtifactIntegrityError(f"unsupported manifest format: {manifest.get('format')!r}")
    kind = manifest.get("kind", "mlp")
    if not isinstance(kind, str) or kind not in KIND_FILES or ("kind" in manifest and kind == "mlp"):
        raise ArtifactIntegrityError(f"unsupported model kind in the manifest: {manifest.get('kind')!r}")
    files = KIND_FILES[kind]
    if not isinstance(expected, dict) or set(expected) != set(files):
        raise ArtifactIntegrityError(f"manifest sha256 must list exactly {sorted(files)}")
    if not all(isinstance(v, str) for v in expected.values()):
        raise ArtifactIntegrityError("manifest sha256 values must be strings")
    return kind, expected


def _check_model_info(info: dict, kind: str = "mlp") -> None:
    if info.get("kind", "mlp") != kind or (kind == "mlp" and "kind" in info):
        raise ArtifactIntegrityError(f"model.json kind {info.get('kind')!r} does not match the manifest ({kind!r})")
    if kind == "xgb":
        p = info.get("xgb_params")
        if not isinstance(p, dict) or any(isinstance(p.get(k), bool) or not isinstance(p.get(k), int)
                                          for k in ("max_depth", "n_estimators")):
            raise ArtifactIntegrityError("xgb_params must hold integer max_depth and n_estimators")
    if kind == "gru":
        p = info.get("gru_params")
        if not isinstance(p, dict) or any(isinstance(p.get(k), bool) or not isinstance(p.get(k), int) or p.get(k) < 1
                                          for k in ("k", "hidden")) or not 2 <= p["k"] <= 20:
            raise ArtifactIntegrityError("gru_params must hold integer 2 <= k <= 20 and hidden >= 1")
    tau = info.get("tau")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) \
            or not 0.0 <= tau <= 1.0:
        raise ArtifactIntegrityError(f"tau must be a finite number in [0, 1], got {tau!r}")
    cols = info.get("feature_columns")
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ArtifactIntegrityError("feature_columns differ from FEATURE_COLUMNS and from every registered set")
    try:
        feature_set_name(cols)
    except ValueError as e:
        raise ArtifactIntegrityError(f"feature_columns differ from FEATURE_COLUMNS: {e}") from None
    if not isinstance(info.get("meta"), dict):
        raise ArtifactIntegrityError("meta must be an object")
    if not isinstance(info.get("versions"), dict):
        raise ArtifactIntegrityError("versions must be an object")


def _verify(model_dir: Path, expected: dict) -> None:
    for name in expected:
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
    manifest = _read_json(model_dir / MANIFEST_FILE)
    kind, expected = _check_manifest(manifest)
    _verify(model_dir, expected)  # raw bytes first, nothing deserialized before this
    info = _read_json(model_dir / MODEL_FILE)
    _check_model_info(info, kind)
    enc = _read_json(model_dir / ENCODER_FILE)
    try:
        encoder = BaseEncoder.from_json(enc)
    except (KeyError, TypeError) as e:
        raise ArtifactIntegrityError(f"malformed {ENCODER_FILE}: {e}") from e
    if kind == "xgb":
        import xgboost
        booster = xgboost.Booster()
        booster.load_model(str(model_dir / XGB_FILE))
        return Artifacts(mlp=None, scaler=None, iforest=None, encoder=encoder, tau=float(info["tau"]),
                         meta=info["meta"], manifest=manifest, feature_columns=tuple(info["feature_columns"]),
                         kind="xgb", xgb=booster, xgb_params=dict(info["xgb_params"]))
    if kind == "gru":
        from .gru import configure_tf  # before the first TensorFlow op: fixed threads, deterministic ops
        configure_tf()
        net = keras.models.load_model(model_dir / GRU_FILE, safe_mode=True, compile=False)
        return Artifacts(mlp=None, scaler=_load_skops(model_dir / SCALER_FILE), iforest=None, encoder=encoder,
                         tau=float(info["tau"]), meta=info["meta"], manifest=manifest,
                         feature_columns=tuple(info["feature_columns"]), kind="gru", gru=net,
                         gru_params=dict(info["gru_params"]))
    if kind != "mlp":
        raise ArtifactIntegrityError(f"no loader for model kind {kind!r}")
    mlp = keras.models.load_model(model_dir / MLP_FILE, safe_mode=True, compile=False)
    scaler = _load_skops(model_dir / SCALER_FILE)
    iforest = _load_skops(model_dir / IFOREST_FILE)
    return Artifacts(mlp=mlp, scaler=scaler, iforest=iforest, encoder=encoder,
                     tau=float(info["tau"]), meta=info["meta"], manifest=manifest,
                     feature_columns=tuple(info["feature_columns"]))


def derive_with_tau(src_dir: Path, dst_dir: Path, tau: float, meta_update: dict) -> None:
    """Copy the verified model of ``src_dir`` to ``dst_dir`` with another decision threshold ``tau`` (same weights,
    another operating point): the artifact files are copied byte for byte, ``model.json`` gets ``tau`` and
    ``meta`` updated with ``meta_update`` plus ``derived_from`` (the source ``model.json`` SHA-256), and a new
    manifest is written. The source is verified (``load_artifacts``) first."""
    import shutil
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    load_artifacts(src_dir)  # integrity of the source before anything is copied
    if not (isinstance(tau, (int, float)) and math.isfinite(tau) and 0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be a finite number in [0, 1], got {tau!r}")
    kind = json.loads((src_dir / MANIFEST_FILE).read_text(encoding="utf-8")).get("kind", "mlp")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in KIND_FILES[kind]:
        if name != MODEL_FILE:
            shutil.copyfile(src_dir / name, dst_dir / name)
    info = _read_json(src_dir / MODEL_FILE)
    info["tau"] = float(tau)
    info["meta"] = dict(info["meta"], **meta_update, derived_from={"dir": str(src_dir),
                                                                   "model_json_sha256": _sha256(src_dir / MODEL_FILE)})
    (dst_dir / MODEL_FILE).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"format": MANIFEST_FORMAT, "sha256": {name: _sha256(dst_dir / name) for name in KIND_FILES[kind]}}
    if kind != "mlp":
        manifest["kind"] = kind
    (dst_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
