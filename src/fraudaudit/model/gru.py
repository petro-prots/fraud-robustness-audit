"""GRU over the card's last ``k`` operations (Task 24, rulings R30/R32/R35/R37).

Decisions fixed before any run (brief §1, R37, R38):

* input = the 38 ``FEATURE_COLUMNS`` rows of the card's last ``k`` operations INCLUDING the current one, standardized with
  the scaler fit on ``fit`` rows, left-padded with zeros and an explicit boolean mask for a card with fewer operations;
  operations of the owner and of the attacker enter alike (every operation enters the card state). Hence the GRU sees the
  card's history as ``k`` raw feature rows, while the MLP and XGBoost see it compressed in the windowed aggregates:
  a difference of the GRU is "architecture together with the history window", not recurrence alone;
* one GRU layer (``hidden`` units) and a Dense sigmoid head; Adamax 0.01, binary cross-entropy, 20 epochs, batch 32
  (as the MLP); no IsolationForest flag;
* ``k`` and ``hidden`` are chosen on CLEAN ``val`` only, before any gate or attack: grid ``GRU_GRID`` = k {5, 10, 20} x
  hidden {32, 64}, criterion = recall at FPR 0.01 (τ from ``val`` itself), tie → the smaller model: fewest trainable
  parameters, then the shorter window (``k`` does not change the parameter count) – ``pick_best``;
* hardening retrains with the SAME ``k``/``hidden`` (no re-selection on attacked data);
* training history = the true history of the original row in the whole time-ordered stream (inserted adversarial rows
  included); the balanced (resampled) fraud rows are looked up by position, never resampled sequences;
* strict reproducibility (R37 addendum): fixed seed, TensorFlow deterministic ops, ONE intra-op and ONE inter-op thread
  for all GRU work (training and scoring; recorded in ``meta``), own seeded shuffling of the batches.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from ..features.columns import FEATURE_COLUMNS
from ..metrics import threshold_at_fpr
from ..serving.state import HISTORY_ROWS
from .artifacts import Artifacts, save_gru_artifacts
from .history import StreamHistory, previous_index, score_rows
from .pipeline import BATCH_SIZE, EPOCHS, FEATURE_SET, PERIODS, TARGET_FPR, quality, split_and_balance

GRU_GRID: tuple[tuple[int, int], ...] = tuple((k, h) for k in (5, 10, 20) for h in (32, 64))  # (k, hidden)
N_THREADS = 1  # intra- and inter-op threads of TensorFlow for every GRU training and scoring
N_FEATURES = len(FEATURE_COLUMNS)
SCORE_CHUNK = 4096  # rows per forward pass when scoring a stream

_state: dict = {"configured": False}


def configure_tf() -> dict:
    """Strict mode (R37): deterministic TensorFlow ops, one intra-op and one inter-op thread. Must run before the
    first TensorFlow operation of the process (the loaders and trainers call it first); idempotent. Returns what is
    in force – ``strict`` is False if the thread counts could not be set because TensorFlow already ran with others
    (a test process; never a CLI run, which is checked by ``tests/test_gru.py``)."""
    if not _state["configured"]:
        tf.config.experimental.enable_op_determinism()
        try:
            tf.config.threading.set_intra_op_parallelism_threads(N_THREADS)
            tf.config.threading.set_inter_op_parallelism_threads(N_THREADS)
        except RuntimeError:
            pass
        _state["configured"] = True
    return tf_settings()


def tf_settings() -> dict:
    intra = tf.config.threading.get_intra_op_parallelism_threads()
    inter = tf.config.threading.get_inter_op_parallelism_threads()
    from tensorflow.python.framework import config as _tf_config  # no public getter in TensorFlow 2.18
    det = bool(_tf_config.is_op_determinism_enabled())
    return {"intra_op_threads": int(intra), "inter_op_threads": int(inter), "op_determinism": det,
            "strict": det and intra == N_THREADS and inter == N_THREADS}


def build_gru(k: int, hidden: int) -> keras.Model:
    """Inputs ``seq`` (k, 38) float32 and ``mask`` (k,) bool (False = padding), one GRU layer, Dense sigmoid."""
    seq = keras.Input(shape=(k, N_FEATURES), dtype="float32", name="seq")
    mask = keras.Input(shape=(k,), dtype="bool", name="mask")
    h = keras.layers.GRU(hidden)(seq, mask=mask)
    out = keras.layers.Dense(1, activation="sigmoid")(h)
    net = keras.Model([seq, mask], out)
    net.compile(
        optimizer=keras.optimizers.Adamax(learning_rate=0.01),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()],
    )
    return net


def n_parameters(k: int, hidden: int) -> int:
    """Trainable parameters of ``build_gru`` (``k`` does not matter): GRU with ``reset_after`` + Dense head."""
    return 3 * (hidden * (N_FEATURES + hidden) + 2 * hidden) + hidden + 1


class GRUFraudModel:
    """scaler -> GRU over the last ``k`` rows -> sigmoid. ``predict_proba(F, history)``: ``F`` the 38 features of the
    operations to score, ``history`` (``model.history``) their previous rows; ``history_rows = k - 1``."""

    kind = "gru"

    def __init__(self, *, net: keras.Model, scaler: StandardScaler, encoder, tau: float, params: dict,
                 feature_columns=FEATURE_COLUMNS) -> None:
        configure_tf()
        self.net, self.scaler, self.encoder = net, scaler, encoder
        self.tau = float(tau)
        self.params = {"k": int(params["k"]), "hidden": int(params["hidden"])}
        self.feature_columns = tuple(feature_columns)
        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError("the GRU model takes the 38 FEATURE_COLUMNS")
        k = self.params["k"]
        if k - 1 > HISTORY_ROWS:
            raise ValueError(f"window k = {k} needs more history than the card state keeps ({HISTORY_ROWS} + 1)")
        self.history_rows = k - 1
        self._forward = tf.function(
            lambda seq, mask: self.net([seq, mask], training=False),
            input_signature=[tf.TensorSpec(shape=(None, k, N_FEATURES), dtype=tf.float32),
                             tf.TensorSpec(shape=(None, k), dtype=tf.bool)],
        )

    @classmethod
    def from_artifacts(cls, a: Artifacts) -> "GRUFraudModel":
        return cls(net=a.gru, scaler=a.scaler, encoder=a.encoder, tau=a.tau, params=a.gru_params,
                   feature_columns=a.feature_columns)

    def _sequences(self, F: np.ndarray, history) -> tuple[np.ndarray, np.ndarray]:
        """(scaled sequences (n, k, 38) float32, mask (n, k)) of the rows ``F`` with their history."""
        r = self.history_rows
        rows, valid = history.last(r)
        seq = np.concatenate([rows, F[:, None, :]], axis=1)
        mask = np.concatenate([valid, np.ones((len(F), 1), dtype=bool)], axis=1)
        flat = self.scaler.transform(seq.reshape(-1, N_FEATURES)).reshape(seq.shape).astype(np.float32)
        flat[~mask] = 0.0
        return flat, mask

    def predict_proba(self, F: np.ndarray, history=None) -> np.ndarray:
        F = np.asarray(F, dtype=np.float32)
        if F.ndim != 2 or F.shape[1] != N_FEATURES:
            raise ValueError(f"expected shape (n, {N_FEATURES}), got {F.shape}")
        if history is None:
            raise ValueError(f"the GRU needs the card history of every row (history.last({self.history_rows}))")
        if len(history) != len(F):
            raise ValueError(f"history of {len(history)} rows for {len(F)} operations")
        out = np.empty(len(F), dtype=np.float64)
        for lo in range(0, len(F), SCORE_CHUNK):
            sl = slice(lo, lo + SCORE_CHUNK)
            sub = history.subset(sl) if isinstance(history, StreamHistory) else _slice_array_history(history, sl)
            seq, mask = self._sequences(F[sl], sub)
            out[sl] = self._forward(tf.constant(seq), tf.constant(mask)).numpy().reshape(-1)
        return out

    def save(self, out_dir: Path, meta: dict) -> None:
        save_gru_artifacts(out_dir, net=self.net, scaler=self.scaler, encoder=self.encoder, tau=self.tau, meta=meta,
                           params=self.params, feature_columns=self.feature_columns)


def _slice_array_history(h, sl):
    from .history import ArrayHistory
    return ArrayHistory(h.rows[sl], h.valid[sl])


def pick_best(rows: list[dict]) -> dict:
    """The grid row with the highest ``recall``; a tie (exactly equal recall: discrete counts) goes to the smaller
    model – fewest trainable parameters, then the shorter window."""
    return min(rows, key=lambda r: (-r["recall"], n_parameters(r["k"], r["hidden"]), r["k"]))


class _Batches(keras.utils.PyDataset):
    """Balanced training rows as batches of (sequence, mask), reshuffled every epoch from ``seed`` (own seeded
    permutation: nothing depends on TensorFlow's global generator or on threads)."""

    def __init__(self, Fs: np.ndarray, prev: np.ndarray, pos: np.ndarray, y: np.ndarray, k: int, batch_size: int,
                 seed: int) -> None:
        super().__init__(workers=0, use_multiprocessing=False)
        self.Fs, self.prev, self.pos, self.y, self.k, self.bs, self.seed = Fs, prev, pos, y, k, batch_size, seed
        self.epoch = 0
        self._perm = np.random.default_rng([seed, 0]).permutation(len(pos))

    def __len__(self) -> int:
        return math.ceil(len(self.pos) / self.bs)

    def __getitem__(self, i: int):
        sel = self._perm[i * self.bs:(i + 1) * self.bs]
        p = self.pos[sel]
        rows, valid = StreamHistory(self.Fs, self.prev[p]).last(self.k - 1)
        seq = np.concatenate([rows, self.Fs[p][:, None, :]], axis=1)
        mask = np.concatenate([valid, np.ones((len(p), 1), dtype=bool)], axis=1)
        return {"seq": seq, "mask": mask}, self.y[sel].astype(np.float32)

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._perm = np.random.default_rng([self.seed, self.epoch]).permutation(len(self.pos))


def _fit_net(Fs, prev, pos_train, y_train, k, hidden, seed, epochs, verbose=0) -> keras.Model:
    configure_tf()
    keras.utils.set_random_seed(seed)  # python, numpy, tensorflow (the initial weights)
    net = build_gru(k, hidden)
    net.fit(_Batches(Fs, prev, pos_train, y_train, k, BATCH_SIZE, seed), epochs=epochs, verbose=verbose, shuffle=False)
    return net


def train_gru_model(df: pd.DataFrame, seed: int, params: dict | None = None, grid=GRU_GRID, epochs: int = EPOCHS,
                    sample_frac: float | None = None, verbose: int = 0) -> tuple[GRUFraudModel, dict]:
    """Train on the balanced ``fit`` rows; return the model and the same kind of report as ``train_model``.

    ``params`` = ``{"k", "hidden"}`` given (hardening rounds, tests): no search. Otherwise every ``grid`` entry is
    trained and scored on clean ``val`` and the winner (``pick_best``) is kept; the whole grid is in
    ``report["gru"]["selection"]``. ``test`` is only reported, never used."""
    cfg = configure_tf()
    t0 = time.perf_counter()
    encoder, F, y, rows, idx, F_all, pos = split_and_balance(df, seed, sample_frac, with_stream=True)
    scaler = StandardScaler().fit(F["fit"])
    Fs = scaler.transform(F_all).astype(np.float32)  # the whole stream, scaled once: history rows are looked up here
    ts = df["ts"].to_numpy(dtype="datetime64[ns]").view("int64")
    k_max = max(k for k, _ in grid) if params is None else int(params["k"])
    prev = previous_index(df["cc_num"].to_numpy(), ts, k_max - 1)
    pos_train, y_train = pos["fit"][idx], y["fit"][idx]  # balanced rows by stream position (true history)
    t_prep = time.perf_counter() - t0

    def scored(net, k, hidden, tau=0.5) -> tuple[GRUFraudModel, np.ndarray]:
        m = GRUFraudModel(net=net, scaler=scaler, encoder=encoder, tau=tau, params={"k": k, "hidden": hidden})
        return m, score_rows(m, df, F_all, mask=pos["val"])

    t1 = time.perf_counter()
    selection = None
    if params is None:
        selection, nets = [], {}
        for k, hidden in grid:
            net = _fit_net(Fs, prev[:, :k - 1], pos_train, y_train, k, hidden, seed, epochs, verbose)
            _, p_val = scored(net, k, hidden)
            tau = threshold_at_fpr(y["val"], p_val, TARGET_FPR)
            q = quality(y["val"], p_val, tau)
            selection.append({"k": k, "hidden": hidden, "n_parameters": n_parameters(k, hidden),
                              "tau": float(tau) if math.isfinite(tau) else None,
                              "recall": q["recall"], "fpr": q["fpr"]})
            nets[(k, hidden)] = net
        best = pick_best(selection)
        chosen = {"k": best["k"], "hidden": best["hidden"]}
        net = nets[(best["k"], best["hidden"])]
    else:
        chosen = {"k": int(params["k"]), "hidden": int(params["hidden"])}
        net = _fit_net(Fs, prev[:, :chosen["k"] - 1], pos_train, y_train, chosen["k"], chosen["hidden"], seed, epochs,
                       verbose)
    t_fit = time.perf_counter() - t1

    model, p_val = scored(net, chosen["k"], chosen["hidden"])
    tau = threshold_at_fpr(y["val"], p_val, TARGET_FPR)
    if not math.isfinite(tau):
        raise ValueError(f"threshold_at_fpr returned a non-finite tau ({tau!r}): no threshold in "
                         f"[0, 1] keeps FPR on val within {TARGET_FPR}; refusing to save the model")
    model.tau = float(tau)
    p_test = score_rows(model, df, F_all, mask=pos["test"])

    report = {
        "model_kind": "gru",
        "seed": seed,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "sample_frac": sample_frac,
        "target_fpr": TARGET_FPR,
        "tau": model.tau,
        "rows": {name: int(len(y[name])) for name in PERIODS},
        "fraud_rows": {name: int(y[name].sum()) for name in PERIODS},
        "train_balanced_rows": int(len(y_train)),
        "feature_set": FEATURE_SET,
        "gru": {"params": chosen, "selected": chosen, "selection": selection,
                "selection_data": "val (clean)" if selection is not None else "given",
                "history_rows_kept": HISTORY_ROWS, "tf": cfg, "seed_scheme": "keras.utils.set_random_seed(seed) + "
                "default_rng([seed, epoch]) batch order"},
        "val": quality(y["val"], p_val, model.tau),
        "test": quality(y["test"], p_test, model.tau),
        "seconds": {"prepare": round(t_prep, 1), "fit": round(t_fit, 1),
                    "total": round(time.perf_counter() - t0, 1)},
    }
    return model, report
