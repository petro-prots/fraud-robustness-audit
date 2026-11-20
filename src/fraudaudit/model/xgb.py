"""Gradient-boosted trees on the 38 ``FEATURE_COLUMNS`` (Task 24, rulings R30/R32/R35).

Differences from the MLP model, all fixed before any run:

* no scaler and no IsolationForest flag (trees do not need scaling; the flag is an MLP input only);
* the hyperparameters ``max_depth`` and ``n_estimators`` are chosen on CLEAN ``val`` only, before any gate or attack:
  grid ``XGB_GRID``, criterion = recall at FPR 0.01 (τ from ``val`` itself, as everywhere), tie → the smaller model
  (fewest leaves: ``n_estimators * 2**max_depth``). Everything else is ``FIXED_PARAMS``;
* hardening retrains with the SAME chosen hyperparameters (no re-selection on attacked data).

Training and scoring use the same balanced ``fit`` rows as the MLP (``pipeline.split_and_balance``), the same
τ rule (``threshold_at_fpr`` at 0.01 on clean ``val``).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.columns import FEATURE_COLUMNS
from ..metrics import threshold_at_fpr
from .artifacts import Artifacts, save_xgb_artifacts
from .pipeline import FEATURE_SET, PERIODS, TARGET_FPR, quality, split_and_balance

XGB_GRID: tuple[tuple[int, int], ...] = tuple((d, n) for d in (4, 6, 8) for n in (200, 400))  # (max_depth, n_estimators)
FIXED_PARAMS = {"learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                "objective": "binary:logistic"}
N_THREADS = 4  # training threads (recorded in the report); scoring is single-threaded and thread-independent


def _fit_booster(X: np.ndarray, y: np.ndarray, params: dict, seed: int, n_threads: int):
    import xgboost
    clf = xgboost.XGBClassifier(**params, random_state=seed, n_jobs=n_threads, verbosity=0)
    clf.fit(X, y)
    booster = clf.get_booster()
    booster.set_param({"nthread": 1})
    return booster


class XGBFraudModel:
    """Booster on the 38 FEATURE_COLUMNS; ``predict_proba(F, history=None)`` as every model (``history`` ignored)."""

    kind = "xgb"

    def __init__(self, *, booster, encoder, tau: float, params: dict, feature_columns=FEATURE_COLUMNS) -> None:
        self.booster = booster
        self.encoder = encoder
        self.tau = float(tau)
        self.params = dict(params)
        self.feature_columns = tuple(feature_columns)
        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError("the XGBoost model takes the 38 FEATURE_COLUMNS")
        self.booster.set_param({"nthread": 1})

    @classmethod
    def from_artifacts(cls, a: Artifacts) -> "XGBFraudModel":
        return cls(booster=a.xgb, encoder=a.encoder, tau=a.tau, params=a.xgb_params,
                   feature_columns=a.feature_columns)

    def predict_proba(self, F: np.ndarray, history=None) -> np.ndarray:
        F = np.asarray(F, dtype=np.float32)
        if F.ndim != 2 or F.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"expected shape (n, {len(FEATURE_COLUMNS)}), got {F.shape}")
        return np.asarray(self.booster.inplace_predict(F, validate_features=False), dtype=np.float64).reshape(-1)

    def save(self, out_dir: Path, meta: dict) -> None:
        save_xgb_artifacts(out_dir, booster=self.booster, encoder=self.encoder, tau=self.tau, meta=meta,
                           params=self.params, feature_columns=self.feature_columns)


def pick_best(rows: list[dict]) -> dict:
    """The grid row with the highest ``recall``; a tie goes to the smaller model (fewest leaves)."""
    return min(rows, key=lambda r: (-r["recall"], r["n_estimators"] * 2 ** r["max_depth"]))


def train_xgb_model(df: pd.DataFrame, seed: int, params: dict | None = None, grid=XGB_GRID,
                    sample_frac: float | None = None, n_threads: int = N_THREADS) -> tuple[XGBFraudModel, dict]:
    """Train on the balanced ``fit`` rows; return the model and the same kind of report as ``train_model``.

    ``params`` = ``{"max_depth", "n_estimators"}`` given (hardening rounds, tests): no search. Otherwise every
    ``grid`` entry is trained and scored on clean ``val`` and the winner (``pick_best``) is kept; the whole grid is
    in ``report["xgb"]["selection"]``. ``test`` is only reported, never used."""
    t0 = time.perf_counter()
    encoder, F, y, rows, idx = split_and_balance(df, seed, sample_frac)
    X, yb = F["fit"][idx], y["fit"][idx]
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    selection = None
    if params is None:
        selection, boosters = [], {}
        for depth, n_est in grid:
            booster = _fit_booster(X, yb, {**FIXED_PARAMS, "max_depth": depth, "n_estimators": n_est}, seed, n_threads)
            m = XGBFraudModel(booster=booster, encoder=encoder, tau=0.5, params={})
            p_val = m.predict_proba(F["val"])
            tau = threshold_at_fpr(y["val"], p_val, TARGET_FPR)
            q = quality(y["val"], p_val, tau)
            selection.append({"max_depth": depth, "n_estimators": n_est, "tau": float(tau),
                              "recall": q["recall"], "fpr": q["fpr"]})
            boosters[(depth, n_est)] = booster
        best = pick_best(selection)
        chosen = {"max_depth": best["max_depth"], "n_estimators": best["n_estimators"]}
        booster = boosters[(best["max_depth"], best["n_estimators"])]
    else:
        chosen = {"max_depth": int(params["max_depth"]), "n_estimators": int(params["n_estimators"])}
        booster = _fit_booster(X, yb, {**FIXED_PARAMS, **chosen}, seed, n_threads)
    t_fit = time.perf_counter() - t1

    full = {**FIXED_PARAMS, **chosen}
    model = XGBFraudModel(booster=booster, encoder=encoder, tau=0.5, params=full)
    p_val = model.predict_proba(F["val"])
    tau = threshold_at_fpr(y["val"], p_val, TARGET_FPR)
    if not math.isfinite(tau):
        raise ValueError(f"threshold_at_fpr returned a non-finite tau ({tau!r}): no threshold in "
                         f"[0, 1] keeps FPR on val within {TARGET_FPR}; refusing to save the model")
    model.tau = tau
    p_test = model.predict_proba(F["test"])

    report = {
        "model_kind": "xgb",
        "seed": seed,
        "sample_frac": sample_frac,
        "target_fpr": TARGET_FPR,
        "tau": model.tau,
        "rows": {name: int(len(y[name])) for name in PERIODS},
        "fraud_rows": {name: int(y[name].sum()) for name in PERIODS},
        "train_balanced_rows": int(len(yb)),
        "feature_set": FEATURE_SET,
        "xgb": {"params": full, "selected": chosen, "selection": selection,
                "selection_data": "val (clean)" if selection is not None else "given",
                "n_threads": int(n_threads)},
        "val": quality(y["val"], p_val, model.tau),
        "test": quality(y["test"], p_test, model.tau),
        "seconds": {"prepare": round(t_prep, 1), "fit": round(t_fit, 1),
                    "total": round(time.perf_counter() - t0, 1)},
    }
    return model, report
