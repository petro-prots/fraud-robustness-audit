"""Serving path: the 34 features of one operation from the card state, and the decision.

The 27 base features come from a numpy path for a single operation that reproduces
``BaseEncoder.transform`` value for value (pinned by ``tests/test_serving.py``); building a
one-row DataFrame per operation would dominate the time of ``decide``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.base import PREFIX
from ..features.columns import BASE_COLUMNS, FEATURE_COLUMNS
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
        out[_COL["trans_month"]] = trans_date.month
        out[_COL["trans_year"]] = trans_date.year
        out[_COL["latitudinal_distance"]] = abs(np.round(np.float64(tx.merch_lat) - np.float64(prof.lat), 3))
        out[_COL["longitudinal_distance"]] = abs(np.round(np.float64(tx.merch_long) - np.float64(prof.long), 3))
        if cat > 0:
            out[self._cat0 + cat - 1] = 1.0
        out[_COL["gender_M"]] = gender == "M"
        return out

    def features(self, state: CardState, tx: Tx) -> np.ndarray:
        """FEATURE_COLUMNS (float32, length 34) of ``tx`` given the card history in ``state``."""
        f = np.empty(len(FEATURE_COLUMNS), dtype=np.float32)
        f[:_N_BASE] = self._base(state, tx)
        f[_N_BASE:] = state.aggregates(tx.ts)
        if not np.isfinite(f).all():
            bad = [FEATURE_COLUMNS[i] for i in np.flatnonzero(~np.isfinite(f))]
            raise ValueError(f"non-finite features: {bad}")
        return f

    def decide(self, state: CardState, tx: Tx) -> Decision:
        """One operation, one call of the compiled ``predict_proba``; declined iff ``p >= tau``."""
        p = float(self.model.predict_proba(self.features(state, tx)[np.newaxis, :])[0])
        return Decision(p=p, declined=p >= self.tau)
