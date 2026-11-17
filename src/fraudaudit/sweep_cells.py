"""Shared cell runner of the grid sweeps (``corridor_sweep``, ``policy_sweep``): the model's τ on clean val, a model
directory at another τ (same weights), the gate's ensemble on one cell and its record, the partial JSON.

One cell = one threat model (bank reaction, corridor) at one τ; the record holds per run (attack × attack seed of
``gate.yaml``) the uplift mean and one-sided upper bound, the mean VERs and the ASR, the worst run, the mean uplift,
the largest VER_adv and the clean test recall/FPR at the cell's τ.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


def val_scores(df: pd.DataFrame, model) -> tuple[np.ndarray, np.ndarray]:
    """(labels, p) of the clean val rows: features on the WHOLE stream (the windowed aggregates need the history
    before val), as in training."""
    from .data import period
    from .model.pipeline import features
    m = (period(df) == "val").to_numpy()
    p = np.asarray(model.predict_proba(features(df, model.encoder)[m]), dtype=np.float64)
    return df.loc[m, "is_fraud"].to_numpy(dtype=np.int64), p


def tau_at(y: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    """τ at ``target_fpr`` on clean val (``threshold_at_fpr``, as in training)."""
    from .metrics import threshold_at_fpr
    return float(threshold_at_fpr(y, p, target_fpr))


def val_tau(df: pd.DataFrame, model, target_fpr: float) -> float:
    """τ at ``target_fpr`` on clean val (one call of ``val_scores``)."""
    return tau_at(*val_scores(df, model), target_fpr)


def model_target_fpr(model_dir) -> float | None:
    meta = json.loads((Path(model_dir) / "model.json").read_text(encoding="utf-8")).get("meta", {})
    v = meta.get("target_fpr")
    return None if v is None else float(v)


def check_own_tau(model_dir, model, tau: float, fpr: float) -> None:
    """ValueError unless τ recomputed at the model's own target FPR reproduces ``model.json`` τ (rel 1e-6)."""
    if not math.isclose(tau, float(model.tau), rel_tol=1e-6):
        raise ValueError(f"tau recomputed at the model's own target_fpr {fpr} is {tau}, {model_dir}/model.json has "
                         f"{model.tau}: the val scores differ from training")


def model_at_tau(model_dir, model, tau: float, fpr: float, derived_dir, note: str):
    """(directory, model) at threshold ``tau``: the model itself when ``tau`` equals its own τ, else a derived
    directory ``derived_dir`` (``derive_with_tau``: same weights, ``meta.target_fpr = fpr``)."""
    from .model.artifacts import derive_with_tau, load_artifacts
    from .model.pipeline import FraudModel
    if math.isclose(tau, float(model.tau), rel_tol=1e-6):
        return Path(model_dir), model
    derive_with_tau(model_dir, derived_dir, tau, {"target_fpr": float(fpr), "note": note})
    return Path(derived_dir), FraudModel.from_artifacts(load_artifacts(derived_dir))


def run_cell(df, model, model_dir, tm, cfg, seed: int, *, n_jobs=None, threat_path=None,
             per_target: bool = False) -> tuple[dict, dict]:
    """The gate's ensemble (``gate.run_ensemble``) on one cell; returns (record, ``{run_key: AuditResult}``).
    ``per_target`` adds per run the per-target VERs in target order (``ver_adv``, ``ver_clean``) and once the
    episode amounts (``amounts``)."""
    from .gate import run_ensemble
    t0 = time.perf_counter()
    runs = run_ensemble(df, model, tm, cfg, seed, n_jobs=n_jobs, model_dir=model_dir, threat_path=threat_path)
    rec = {"runs": {k: {"uplift_mean": r.uplift[0], "uplift_upper": r.uplift[2], "ver_adv_mean": r.ver_adv[0],
                        "ver_clean_mean": r.ver_clean[0], "asr": r.asr} for k, r in runs.items()}}
    bad = [k for k, v in rec["runs"].items() if not math.isfinite(v["uplift_upper"])]
    if bad:  # max() over NaN is order-dependent (gate.decide fails such runs as a tool error)
        raise RuntimeError(f"non-finite uplift upper bound in runs {bad}")
    worst = max(rec["runs"], key=lambda k: rec["runs"][k]["uplift_upper"])
    first = next(iter(runs.values()))
    rec.update(worst_run=worst, worst_uplift_upper=rec["runs"][worst]["uplift_upper"],
               mean_uplift=float(np.mean([v["uplift_mean"] for v in rec["runs"].values()])),
               max_ver_adv=float(max(t["ver_adv"] for r in runs.values() for t in r.per_target)),
               test_recall=first.clean_quality["test"]["recall"], test_fpr=first.clean_quality["test"]["fpr"])
    if per_target:
        keys = [(t["card"], t["start"]) for t in first.per_target]
        for k, r in runs.items():
            if [(t["card"], t["start"]) for t in r.per_target] != keys:
                raise RuntimeError(f"run {k}: targets differ from the first run")
            rec["runs"][k]["ver_adv"] = [float(t["ver_adv"]) for t in r.per_target]
            rec["runs"][k]["ver_clean"] = [float(t["ver_clean"]) for t in r.per_target]
        rec["amounts"] = [float(t["amount"]) for t in first.per_target]
        rec["targets"] = [[int(c), s] for c, s in keys]
    rec["ensemble_seconds"] = round(time.perf_counter() - t0, 1)
    return rec, runs


def write_json(path, data) -> None:
    """Write ``data`` as JSON atomically (temporary file + rename), so an interrupted run never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
