"""Per-card state of the serving path: owner profile and incremental windowed aggregates.

Same contract as the batch path (``features/aggregates.py``): an earlier operation j counts for a
query at time t iff ``t_j < t`` (same-second operations never see each other, even if already
pushed) and ``t_j >= t - w``; ``secs_since_prev`` is capped at ``NO_PREV_SECS``.

Time only moves forward. ``now`` is the latest of ``at`` (``from_history``) and every pushed
timestamp; queries and pushes before ``now`` raise ``ValueError``. Because of that, ``push`` may
drop every operation with ``t_j < now - 30 days``: no later query can still need it (it is outside
the 30-day window and its gap to any later query is > 30 days, i.e. capped anyway).
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.aggregates import NO_PREV_SECS, WINDOW_DAYS
from ..features.columns import AGG_COLUMNS

_NS = 1_000_000_000
_DAY_NS = 86_400 * _NS
_WINDOWS_NS = tuple(d * _DAY_NS for d in WINDOW_DAYS)
_KEEP_NS = max(_WINDOWS_NS)  # 30 days
_NO_PREV_NS = NO_PREV_SECS * _NS


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


def profile_of(df: pd.DataFrame, cc_num: int) -> CardProfile:
    """Profile from the card's first operation in (ts, row) order."""
    g = df[df["cc_num"] == cc_num]
    if g.empty:
        raise ValueError(f"no operations of card {cc_num}")
    r = g.sort_values(["ts", "row"] if "row" in g else ["ts"], kind="stable").iloc[0]
    return CardProfile(cc_num=int(cc_num), gender=str(r["gender"]), dob=pd.Timestamp(r["dob"]),
                       lat=float(r["lat"]), long=float(r["long"]), city_pop=int(r["city_pop"]),
                       state=str(r["state"]))


class CardState:
    """Owner profile + operations of the last 30 days as a time-ordered deque of ``(ts_ns, amt)``."""

    def __init__(self, profile: CardProfile) -> None:
        self.profile = profile
        self._ops: deque[tuple[int, float]] = deque()
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
        order = np.argsort(t, kind="stable")
        s._ops.extend(zip(t[order].tolist(), amt[order].tolist()))
        s._now = at_ns
        return s

    def __len__(self) -> int:
        return len(self._ops)

    def _check_time(self, ts_ns: int, what: str) -> None:
        if self._now is not None and ts_ns < self._now:
            raise ValueError(f"{what} at {pd.Timestamp(ts_ns)} is before the state time "
                             f"{pd.Timestamp(self._now)}; time only moves forward")

    def aggregates(self, ts: pd.Timestamp) -> np.ndarray:
        """AGG_COLUMNS for an operation at ``ts`` (float64, length 7); the state is not changed."""
        t = pd.Timestamp(ts).value
        self._check_time(t, "query")
        times = [o[0] for o in self._ops]
        amts = [o[1] for o in self._ops]
        hi = bisect_left(times, t)  # ops strictly before t: [0, hi)
        out = np.zeros(len(AGG_COLUMNS), dtype=np.float64)
        for k, w in enumerate(_WINDOWS_NS):
            lo = bisect_left(times, t - w)  # first t_j >= t - w
            n = hi - lo
            if n > 0:
                out[k] = n
                out[3 + k] = sum(amts[lo:hi]) / n
        gap = t - times[hi - 1] if hi > 0 else _NO_PREV_NS
        out[6] = min(gap, _NO_PREV_NS) / _NS
        return out

    def push(self, tx: Tx) -> None:
        """Record ``tx`` as history for later operations; drop operations older than 30 days."""
        t = pd.Timestamp(tx.ts).value
        self._check_time(t, "push")
        if self._ops and t < self._ops[-1][0]:  # cannot happen when _now is set, kept as a guard
            raise ValueError("operations must be pushed in time order")
        self._ops.append((t, float(tx.amt)))
        self._now = t
        floor = t - _KEEP_NS
        while self._ops[0][0] < floor:
            self._ops.popleft()

    def copy(self) -> "CardState":
        c = CardState(self.profile)
        c._ops = deque(self._ops)
        c._now = self._now
        return c
