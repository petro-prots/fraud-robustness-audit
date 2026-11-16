"""Per-card state of the serving path: owner profile and incremental windowed aggregates.

Same contract as the batch path (``features/aggregates.py``): an earlier operation j counts for a
query at time t iff ``t_j < t`` (same-second operations never see each other, even if already
pushed) and ``t_j >= t - w``; ``secs_since_prev`` is capped at ``NO_PREV_SECS``. The behavioural
features (Task 23) follow the same rule: ``cnt_1h``/``sum_1h`` over ``[t - 1 h, t)``, ``cnt_cat_24h`` over the
operations of the queried operation's category in ``[t - 24 h, t)`` (the state keeps every operation's
category), ``amt_ratio_30d`` from the queried amount and ``mean_amt_30d``.

Time only moves forward. ``now`` is the latest of ``at`` (``from_history``) and every pushed
timestamp; queries and pushes before ``now`` raise ``ValueError``. Because of that, ``push`` may
drop every operation with ``t_j < now - 30 days``: no later query can still need it (it is outside
the 30-day window and its gap to any later query is > 30 days, i.e. capped anyway).
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.aggregates import CAT_WINDOW_SECS, HOUR_SECS, NO_PREV_SECS, WINDOW_DAYS, amt_ratio
from ..features.columns import AGG_COLUMNS

_NS = 1_000_000_000
_DAY_NS = 86_400 * _NS
_WINDOWS_NS = tuple(d * _DAY_NS for d in WINDOW_DAYS)
_KEEP_NS = max(_WINDOWS_NS)  # 30 days
_NO_PREV_NS = NO_PREV_SECS * _NS
_HOUR_NS = HOUR_SECS * _NS
_CAT_NS = CAT_WINDOW_SECS * _NS
_COL = {c: i for i, c in enumerate(AGG_COLUMNS)}


@dataclass(frozen=True)
class CardProfile:
    """Owner fields of a card that do not change between its operations."""
    cc_num: int
    gender: str
    dob: pd.Timestamp
    lat: float
    long: float
    city_pop: int
    state: str


@dataclass(frozen=True)
class Tx:
    """One operation as the attacker controls it; ``merchant`` is the raw name (``fraud_`` prefix kept)."""
    ts: pd.Timestamp
    amt: float
    merchant: str
    category: str
    merch_lat: float
    merch_long: float


OWNER_FIELDS = ("gender", "dob", "lat", "long", "city_pop", "state")


def profile_of(df: pd.DataFrame, cc_num: int) -> CardProfile:
    """Owner profile of a card; ``ValueError`` if an owner field is not constant over its operations.

    The batch path computes base features per row, the serving path from one profile per card, so a
    card with varying owner fields would make the two paths diverge silently.
    """
    g = df[df["cc_num"] == cc_num]
    if g.empty:
        raise ValueError(f"no operations of card {cc_num}")
    varying = [c for c in OWNER_FIELDS if g[c].nunique(dropna=False) > 1]
    if varying:
        raise ValueError(f"owner fields of card {cc_num} are not constant: {varying}")
    r = g.iloc[0]
    return CardProfile(cc_num=int(cc_num), gender=str(r["gender"]), dob=pd.Timestamp(r["dob"]),
                       lat=float(r["lat"]), long=float(r["long"]), city_pop=int(r["city_pop"]),
                       state=str(r["state"]))


class CardState:
    """Owner profile + operations of the last 30 days as time-ordered parallel lists ``ts_ns``, ``amt``,
    ``category``."""

    def __init__(self, profile: CardProfile) -> None:
        self.profile = profile
        self._t: list[int] = []  # ns, non-decreasing
        self._a: list[float] = []
        self._c: list[str] = []
        self._now: int | None = None  # ns; None = no time seen yet

    @classmethod
    def from_history(cls, profile: CardProfile, history: pd.DataFrame, at: pd.Timestamp) -> "CardState":
        """State right before ``at``: the card's operations with ``at - 30d <= ts < at``."""
        s = cls(profile)
        at_ns = pd.Timestamp(at).value
        h = history
        if "cc_num" in h:
            h = h[h["cc_num"] == profile.cc_num]
        t = h["ts"].to_numpy(dtype="datetime64[ns]").view("int64")
        m = (t < at_ns) & (t >= at_ns - _KEEP_NS)
        t, amt = t[m], h["amt"].to_numpy(dtype=np.float64)[m]
        cat = h["category"].astype(str).to_numpy()[m]
        order = np.argsort(t, kind="stable")
        s._t = t[order].tolist()
        s._a = amt[order].tolist()
        s._c = cat[order].tolist()
        s._now = at_ns
        return s

    def __len__(self) -> int:
        return len(self._t)

    def _check_time(self, ts_ns: int, what: str) -> None:
        if self._now is not None and ts_ns < self._now:
            raise ValueError(f"{what} at {pd.Timestamp(ts_ns)} is before the state time "
                             f"{pd.Timestamp(self._now)}; time only moves forward")

    def aggregates(self, ts: pd.Timestamp, amt: float, category: str) -> np.ndarray:
        """AGG_COLUMNS for an operation of ``amt`` in ``category`` at ``ts`` (float64, length 11); the state
        is not changed."""
        t = pd.Timestamp(ts).value
        self._check_time(t, "query")
        times, amts = self._t, self._a
        hi = bisect_left(times, t)  # ops strictly before t: [0, hi)
        out = np.zeros(len(AGG_COLUMNS), dtype=np.float64)
        for k, w in enumerate(_WINDOWS_NS):
            lo = bisect_left(times, t - w, 0, hi)  # first t_j >= t - w
            n = hi - lo
            if n > 0:
                out[k] = n
                out[3 + k] = sum(amts[lo:hi]) / n
        gap = t - times[hi - 1] if hi > 0 else _NO_PREV_NS
        out[6] = min(gap, _NO_PREV_NS) / _NS
        lo = bisect_left(times, t - _HOUR_NS, 0, hi)
        out[_COL["cnt_1h"]] = hi - lo
        out[_COL["sum_1h"]] = sum(amts[lo:hi])
        lo = bisect_left(times, t - _CAT_NS, 0, hi)
        category = str(category)
        out[_COL["cnt_cat_24h"]] = sum(1 for c in self._c[lo:hi] if c == category)
        out[_COL["amt_ratio_30d"]] = amt_ratio(float(amt), out[_COL["mean_amt_30d"]])
        return out

    def push(self, tx: Tx) -> None:
        """Record ``tx`` as history for later operations; drop operations older than 30 days."""
        t = pd.Timestamp(tx.ts).value
        self._check_time(t, "push")
        self._t.append(t)
        self._a.append(float(tx.amt))
        self._c.append(str(tx.category))
        self._now = t
        drop = bisect_left(self._t, t - _KEEP_NS)  # keep t_j >= now - 30d
        if drop:
            del self._t[:drop], self._a[:drop], self._c[:drop]

    def copy(self) -> "CardState":
        c = CardState(self.profile)
        c._t, c._a, c._c = list(self._t), list(self._a), list(self._c)
        c._now = self._now
        return c
