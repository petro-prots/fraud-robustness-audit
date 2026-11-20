"""Serving path: the 38 features of one operation from the card state, and the decision.

The 27 base features come from a numpy path for a single operation that reproduces
``BaseEncoder.transform`` value for value (pinned by ``tests/test_serving.py``); building a
one-row DataFrame per operation would dominate the time of ``decide``. A model trained on a registered feature
subset (``columns.FEATURE_SETS``, e.g. the 34-feature M2 baseline) gets the same 38 features and selects its own
columns inside ``FraudModel.predict_proba``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from ..features.base import PREFIX, hour_features
from ..features.columns import BASE_COLUMNS, FEATURE_COLUMNS
from ..model.history import ArrayHistory
from .state import CardState, Tx

_N_BASE = len(BASE_COLUMNS)
_COL = {c: i for i, c in enumerate(BASE_COLUMNS)}


@dataclass(frozen=True)
class Decision:
    p: float
    declined: bool


class Serving:
    """Per-operation features and decisions of a fitted ``FraudModel`` (or any object with
    ``encoder``, ``tau`` and ``predict_proba``)."""

    def __init__(self, model) -> None:
        enc = model.encoder
        if enc.merchants is None:
            raise ValueError("BaseEncoder is not fitted")
        self.model = model
        self.tau = float(model.tau)
        self._merchant = {m: float(i) for i, m in enumerate(enc.merchants)}
        self._state = {s: float(i) for i, s in enumerate(enc.states)}
        self._category = {c: i for i, c in enumerate(enc.categories)}  # index 0 = dropped baseline
        self._gender = set(enc.genders)
        self._cat0 = _COL[f"category_{enc.categories[1]}"]
        self._history_rows = int(getattr(model, "history_rows", 0))  # > 0: a sequence model (GRU) needs the history
        self._token = uuid.uuid4().hex  # marks the history rows cached in ``HistoryEntry`` as built by this instance

    def _base(self, state: CardState, tx: Tx) -> np.ndarray:
        prof = state.profile
        out = np.zeros(_N_BASE, dtype=np.float64)
        try:
            out[_COL["merchant"]] = self._merchant[str(tx.merchant).removeprefix(PREFIX)]
        except KeyError:
            raise ValueError(f"unknown merchant in transform: {tx.merchant!r}") from None
        try:
            out[_COL["state"]] = self._state[str(prof.state)]
        except KeyError:
            raise ValueError(f"unknown state in transform: {prof.state!r}") from None
        cat = self._category.get(str(tx.category))
        if cat is None:
            raise ValueError(f"unknown category in transform: {tx.category!r}")
        gender = str(prof.gender)
        if gender not in self._gender:
            raise ValueError(f"unknown gender in transform: {gender!r}")

        out[_COL["amt"]] = tx.amt
        out[_COL["lat"]] = prof.lat
        out[_COL["long"]] = prof.long
        out[_COL["city_pop"]] = prof.city_pop
        out[_COL["merch_lat"]] = tx.merch_lat
        out[_COL["merch_long"]] = tx.merch_long
        trans_date = tx.ts.normalize()
        out[_COL["age"]] = np.round((trans_date - prof.dob).days / 365.25)  # half-to-even, as transform
        out[_COL["latitudinal_distance"]] = abs(np.round(np.float64(tx.merch_lat) - np.float64(prof.lat), 3))
        out[_COL["longitudinal_distance"]] = abs(np.round(np.float64(tx.merch_long) - np.float64(prof.long), 3))
        if cat > 0:
            out[self._cat0 + cat - 1] = 1.0
        out[_COL["gender_M"]] = gender == "M"
        hs, hc = hour_features([tx.ts.hour], [tx.ts.minute], [tx.ts.second])
        out[_COL["hour_sin"]] = hs[0]
        out[_COL["hour_cos"]] = hc[0]
        return out

    def features(self, state: CardState, tx: Tx) -> np.ndarray:
        """FEATURE_COLUMNS (float32, length 38) of ``tx`` given the card history in ``state``."""
        f = np.empty(len(FEATURE_COLUMNS), dtype=np.float32)
        f[:_N_BASE] = self._base(state, tx)
        f[_N_BASE:] = state.aggregates(tx.ts, tx.amt, tx.category)
        if not np.isfinite(f).all():
            bad = [FEATURE_COLUMNS[i] for i in np.flatnonzero(~np.isfinite(f))]
            raise ValueError(f"non-finite features: {bad}")
        return f

    def history(self, state: CardState, rows: int | None = None) -> ArrayHistory:
        """The card's previous ``rows`` operations (default: what the model needs) as feature rows, oldest first,
        left-padded with a False mask for a card with fewer: the serving side of ``model.history``."""
        r = self._history_rows if rows is None else int(rows)
        if not state._hx_known:
            raise ValueError("this CardState was built from a frame without merchant/merch_lat/merch_long columns: "
                             "it cannot feed a sequence model")
        entries = state._hx[-r:] if r else ()
        out = np.zeros((1, r, len(FEATURE_COLUMNS)), dtype=np.float32)
        valid = np.zeros((1, r), dtype=bool)
        off = r - len(entries)
        for i, e in enumerate(entries):
            if e.token != self._token:
                f = np.empty(len(FEATURE_COLUMNS), dtype=np.float32)
                f[:_N_BASE] = self._base(state, e.tx)
                f[_N_BASE:] = e.agg
                if not np.isfinite(f).all():
                    raise ValueError("non-finite history features")
                e.row, e.token = f, self._token
            out[0, off + i] = e.row
            valid[0, off + i] = True
        return ArrayHistory(out, valid)

    def decide(self, state: CardState, tx: Tx) -> Decision:
        """One operation, one call of the compiled ``predict_proba``; declined iff ``p >= tau``. A sequence model
        also gets the card's previous rows (``history``): the same call, the same state, no other path."""
        f = self.features(state, tx)[np.newaxis, :]
        if self._history_rows:
            p = float(self.model.predict_proba(f, history=self.history(state))[0])
        else:
            p = float(self.model.predict_proba(f)[0])
        return Decision(p=p, declined=p >= self.tau)
