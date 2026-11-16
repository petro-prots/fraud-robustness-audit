"""Training and inference of the bachelor fraud-detection model.

Architecture and hyperparameters follow the bachelor listings exactly: StandardScaler ->
IsolationForest flag (``predict(X) > -1``) as the last (39th) input -> MLP
(Dense 55 relu, BN, Dense 26 sigmoid, BN, Dense 12 sigmoid, BN, Dense 1 sigmoid),
Adamax(0.01), BinaryCrossentropy, 20 epochs, batch 32 (Keras default).

Feature set ``bachelor+hour-calendar+behav`` (rulings R16, R23, Task 23): the 27 bachelor base features
without the calendar features ``trans_month``/``trans_year`` (25), the time of day ``hour_sin``/``hour_cos``,
7 windowed aggregates and 4 behavioural features ``cnt_1h``, ``sum_1h``, ``cnt_cat_24h``, ``amt_ratio_30d``
(38 features).

Data-handling corrections of the approved design:

* features are computed on the whole time-ordered stream (aggregates see the history);
* encoder, scaler and IsolationForest are fit on ``fit`` rows only;
* balancing resamples ``fit`` fraud rows with replacement up to the number of ``fit``
  legit rows; ``val`` and ``test`` stay unbalanced;
* tau = ``threshold_at_fpr(y_val, p_val, 0.01)`` on clean ``val``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from ..data import period
from ..features.aggregates import batch_aggregates
from ..features.base import BaseEncoder
from ..features.columns import BASE_COLUMNS, FEATURE_COLUMNS
from ..metrics import threshold_at_fpr
from .artifacts import Artifacts, save_artifacts

TARGET_FPR = 0.01
EPOCHS = 20
BATCH_SIZE = 32  # Keras default, as in the listing (mlp.fit without batch_size)
N_INPUTS = len(FEATURE_COLUMNS) + 1  # 38 features + IsolationForest flag
FEATURE_SET = "bachelor+hour-calendar+behav"  # 25 bachelor base + hour_sin/hour_cos + 7 aggregates + 4 behavioural
PERIODS = ("fit", "val", "test")


def build_mlp(n_inputs: int = N_INPUTS) -> keras.Model:
    """The bachelor MLP (listing 3.6), compiled as in the listing."""
    mlp = keras.Sequential([
        keras.Input(shape=(n_inputs,)),
        keras.layers.Dense(55, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(26, activation="sigmoid"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(12, activation="sigmoid"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    mlp.compile(
        optimizer=keras.optimizers.Adamax(learning_rate=0.01),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()],
    )
    return mlp


def iforest_flag(iforest: IsolationForest, Xs: np.ndarray) -> np.ndarray:
    """Listing: ``np.vectorize(lambda v: v > -1)(iso_forest.predict(X))`` as a float32 column."""
    return (iforest.predict(Xs) > -1).astype(np.float32)[:, np.newaxis]


class FraudModel:
    """scaler -> IsolationForest flag -> MLP; ``predict_proba`` takes the 38 FEATURE_COLUMNS."""

    def __init__(self, *, mlp: keras.Model, scaler: StandardScaler, iforest: IsolationForest,
                 encoder: BaseEncoder, tau: float) -> None:
        self.mlp = mlp
        self.scaler = scaler
        self.iforest = iforest
        self.encoder = encoder
        self.tau = float(tau)
        # BatchNorm in inference mode; fixed signature, so one trace serves batches and single rows.
        self._forward = tf.function(
            lambda x: self.mlp(x, training=False),
            input_signature=[tf.TensorSpec(shape=(None, N_INPUTS), dtype=tf.float32)],
        )

    @classmethod
    def from_artifacts(cls, a: Artifacts) -> "FraudModel":
        return cls(mlp=a.mlp, scaler=a.scaler, iforest=a.iforest, encoder=a.encoder, tau=a.tau)

    def mlp_inputs(self, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=np.float32)
        if F.ndim != 2 or F.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"expected shape (n, {len(FEATURE_COLUMNS)}), got {F.shape}")
        Xs = self.scaler.transform(F)
        return np.hstack([Xs, iforest_flag(self.iforest, Xs)]).astype(np.float32)

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        X = self.mlp_inputs(F)
        return self._forward(tf.constant(X)).numpy().reshape(-1).astype(np.float64)

    def save(self, out_dir: Path, meta: dict) -> None:
        save_artifacts(out_dir, mlp=self.mlp, scaler=self.scaler, iforest=self.iforest,
                       encoder=self.encoder, tau=self.tau, meta=meta)


@dataclass
class Datasets:
    encoder: BaseEncoder
    scaler: StandardScaler
    iforest: IsolationForest
    F: dict[str, np.ndarray]  # period -> (n, 38) float32 features, unbalanced
    y: dict[str, np.ndarray]
    rows: dict[str, np.ndarray]  # period -> source `row` ids
    X_train: np.ndarray  # balanced fit rows, scaled + flag, (n, 39)
    y_train: np.ndarray
    train_rows: np.ndarray


def features(df: pd.DataFrame, encoder: BaseEncoder) -> np.ndarray:
    """FEATURE_COLUMNS for every row of the (whole, time-ordered) stream."""
    base = encoder.transform(df)
    agg = batch_aggregates(df)
    return pd.concat([base, agg], axis=1)[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)


def build_datasets(df: pd.DataFrame, seed: int, sample_frac: float | None = None) -> Datasets:
    """Features on the whole stream; encoder/scaler/IsolationForest fit and balancing on ``fit`` only.

    ``sample_frac`` (timing probes only) subsamples ``fit`` rows after the features are computed;
    the encoder still sees all ``fit`` rows, ``val`` and ``test`` stay complete.
    """
    per = period(df).to_numpy()
    fit_mask = per == "fit"
    encoder = BaseEncoder().fit(df[fit_mask])
    F_all = features(df, encoder)
    y_all = df["is_fraud"].to_numpy(dtype=np.int64)
    row_all = df["row"].to_numpy(dtype=np.int64)

    F, y, rows = {}, {}, {}
    for name in PERIODS:
        m = per == name
        F[name], y[name], rows[name] = F_all[m], y_all[m], row_all[m]
    if sample_frac is not None:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(F["fit"]), size=max(1, int(round(len(F["fit"]) * sample_frac))),
                                  replace=False))
        F["fit"], y["fit"], rows["fit"] = F["fit"][keep], y["fit"][keep], rows["fit"][keep]

    scaler = StandardScaler().fit(F["fit"])
    Xs_fit = scaler.transform(F["fit"])
    iforest = IsolationForest(random_state=seed).fit(Xs_fit)
    X_fit = np.hstack([Xs_fit, iforest_flag(iforest, Xs_fit)]).astype(np.float32)

    legit, fraud = y["fit"] == 0, y["fit"] == 1
    up = resample(np.flatnonzero(fraud), replace=True, n_samples=int(legit.sum()), random_state=seed)
    idx = np.concatenate([np.flatnonzero(legit), up])
    return Datasets(encoder=encoder, scaler=scaler, iforest=iforest, F=F, y=y, rows=rows,
                    X_train=X_fit[idx], y_train=y["fit"][idx], train_rows=rows["fit"][idx])


def quality(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    pred = p >= tau
    pos, neg = y == 1, y == 0
    return {
        "n": int(len(y)),
        "n_fraud": int(pos.sum()),
        "recall": float(pred[pos].mean()) if pos.any() else 0.0,
        "fpr": float(pred[neg].mean()) if neg.any() else 0.0,
        "precision": float(y[pred].mean()) if pred.any() else 0.0,
        "tp": int((pred & pos).sum()),
        "fp": int((pred & neg).sum()),
    }


def train_model(df: pd.DataFrame, seed: int, epochs: int = EPOCHS,
                sample_frac: float | None = None, verbose: int = 0) -> tuple[FraudModel, dict]:
    """Train the bachelor model; return it and recall/FPR/precision on ``val`` and ``test`` at tau."""
    keras.utils.set_random_seed(seed)  # python, numpy, tensorflow
    t0 = time.perf_counter()
    ds = build_datasets(df, seed, sample_frac)
    t_prep = time.perf_counter() - t0

    mlp = build_mlp()
    t1 = time.perf_counter()
    mlp.fit(ds.X_train, ds.y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=verbose)
    t_fit = time.perf_counter() - t1

    model = FraudModel(mlp=mlp, scaler=ds.scaler, iforest=ds.iforest, encoder=ds.encoder, tau=0.5)
    p_val = model.predict_proba(ds.F["val"])
    tau = threshold_at_fpr(ds.y["val"], p_val, TARGET_FPR)
    if not math.isfinite(tau):
        raise ValueError(f"threshold_at_fpr returned a non-finite tau ({tau!r}): no threshold in "
                         f"[0, 1] keeps FPR on val within {TARGET_FPR}; refusing to save the model")
    model.tau = tau
    p_test = model.predict_proba(ds.F["test"])

    report = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "sample_frac": sample_frac,
        "target_fpr": TARGET_FPR,
        "tau": model.tau,
        "rows": {name: int(len(ds.y[name])) for name in PERIODS},
        "fraud_rows": {name: int(ds.y[name].sum()) for name in PERIODS},
        "train_balanced_rows": int(len(ds.y_train)),
        "iforest_flag_share_train_balanced": float(ds.X_train[:, -1].mean()),
        "feature_set": FEATURE_SET,
        "n_base_features": len(BASE_COLUMNS),
        "val": quality(ds.y["val"], p_val, model.tau),
        "test": quality(ds.y["test"], p_test, model.tau),
        "seconds": {"prepare": round(t_prep, 1), "fit": round(t_fit, 1),
                    "total": round(time.perf_counter() - t0, 1)},
    }
    return model, report
