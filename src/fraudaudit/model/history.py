"""Card history for sequence models (Task 24, GRU): ONE interface for the batch/audit path and the serving path.

A model with ``history_rows = r > 0`` (the GRU with window ``k`` has ``r = k - 1``) is scored as
``model.predict_proba(F, history=h)`` where ``F`` are the 38 feature rows of the operations to score and ``h.last(r)``
returns ``(rows, valid)``: the feature rows of each operation's previous ``r`` operations of the SAME card (oldest
first, shape ``(n, r, 38)``) and a boolean mask ``(n, r)`` that is False for the left padding of a card with fewer than
``r`` earlier operations. MLP and XGBoost have ``history_rows = 0`` and are never given a history.

"Previous" means: processed earlier in the stream, of any age, whoever sent it (owner or attacker), passed or
declined – every operation enters the card state (``CardState`` keeps the last ``HISTORY_ROWS``). Two implementations:

* ``StreamHistory`` – batch path: the whole time-ordered stream, history of row i = the rows before i of its card.
  Training with balanced (resampled) rows gathers the TRUE history of the original row, never a resampled one.
* ``ArrayHistory`` – serving path: built by ``Serving`` from ``CardState`` for one operation.

``score_rows`` is the single batch entry point (clean quality, validation scores, hardening diagnostics, skew check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..serving.state import HISTORY_ROWS


class ArrayHistory:
    """History of ``n`` operations as arrays: ``rows`` (n, L, 38) float32 oldest first, ``valid`` (n, L) bool."""

    def __init__(self, rows: np.ndarray, valid: np.ndarray) -> None:
        if rows.ndim != 3 or valid.shape != rows.shape[:2]:
            raise ValueError(f"rows (n, L, d) and valid (n, L) expected, got {rows.shape} and {valid.shape}")
        self.rows, self.valid = rows, valid

    def __len__(self) -> int:
        return self.rows.shape[0]

    @property
    def length(self) -> int:
        return self.rows.shape[1]

    def last(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 < m <= self.length:
            raise ValueError(f"history holds {self.length} previous rows per operation, {m} requested")
        return self.rows[:, self.length - m:], self.valid[:, self.length - m:]


class StreamHistory:
    """History of the rows of a stream by index: ``prev`` (n, L) int32, column ``l - 1`` = the row ``l`` operations
    earlier on the same card (most recent first), -1 where the card has no such operation; ``F`` (N, 38) = the
    feature rows the indices point into (the whole stream, already scaled when the caller wants scaled rows).
    ``subset`` selects operations (a mask or index array) without copying ``F``."""

    def __init__(self, F: np.ndarray, prev: np.ndarray) -> None:
        self.F, self.prev = F, prev

    def __len__(self) -> int:
        return self.prev.shape[0]

    @property
    def length(self) -> int:
        return self.prev.shape[1]

    def subset(self, sel) -> "StreamHistory":
        return StreamHistory(self.F, self.prev[sel])

    def last(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 < m <= self.length:
            raise ValueError(f"history holds {self.length} previous rows per operation, {m} requested")
        idx = self.prev[:, :m][:, ::-1]  # oldest first
        valid = idx >= 0
        rows = self.F[np.where(valid, idx, 0)]
        rows[~valid] = 0.0
        return rows, valid


def previous_index(cards: np.ndarray, ts: np.ndarray | None = None, length: int = HISTORY_ROWS) -> np.ndarray:
    """``prev`` of ``StreamHistory`` for a stream in processing order: ``cards`` (n,) card ids; the previous
    operations of row i are the rows of the same card before i in stream order. ``ts`` (optional, int64 ns) is used
    to check that every card's operations are in time order (``ValueError`` otherwise: a history built from an
    unsorted stream would silently differ from the serving state)."""
    n = len(cards)
    order = np.argsort(cards, kind="stable")  # stream order kept within a card
    sc = cards[order]
    pos = np.arange(n)
    start = np.zeros(n, dtype=np.int64)  # sorted position of the first operation of the card
    if n:
        new = np.r_[True, sc[1:] != sc[:-1]]
        start = np.maximum.accumulate(np.where(new, pos, 0))
    if ts is not None and n > 1:
        t = np.asarray(ts)[order]
        if ((t[1:] < t[:-1]) & (sc[1:] == sc[:-1])).any():
            raise ValueError("the stream is not in time order within a card")
    prev = np.full((n, length), -1, dtype=np.int32)
    for lag in range(1, length + 1):
        ok = pos - lag >= start
        prev[order[ok], lag - 1] = order[pos[ok] - lag]
    return prev


def stream_history(df: pd.DataFrame, F: np.ndarray, length: int = HISTORY_ROWS) -> StreamHistory:
    """``StreamHistory`` of every row of ``df`` (the whole time-ordered stream) over its feature rows ``F``."""
    ts = df["ts"].to_numpy(dtype="datetime64[ns]").view("int64")
    return StreamHistory(F, previous_index(df["cc_num"].to_numpy(), ts, length))


def history_rows_of(model) -> int:
    return int(getattr(model, "history_rows", 0))


def score_rows(model, df: pd.DataFrame, F: np.ndarray, mask=None) -> np.ndarray:
    """``predict_proba`` of the rows of the time-ordered stream ``df`` selected by ``mask`` (all rows when None),
    ``F`` = the features of the WHOLE stream. A sequence model also gets the true history of every row, taken from the
    whole stream (a card's earlier operations in an earlier period count); other models are called as before."""
    sel = slice(None) if mask is None else mask
    if history_rows_of(model) == 0:
        return np.asarray(model.predict_proba(F[sel]), dtype=np.float64)
    h = stream_history(df, F, max(HISTORY_ROWS, history_rows_of(model))).subset(sel)
    return np.asarray(model.predict_proba(F[sel], history=h), dtype=np.float64)
