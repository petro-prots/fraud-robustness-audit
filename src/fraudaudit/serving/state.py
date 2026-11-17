"""Per-card state of the serving path: owner profile and incremental windowed aggregates.

Same contract as the batch path (``features/aggregates.py``): an earlier operation j counts for a
query at time t iff ``t_j < t`` (same-second operations never see each other, even if already
pushed) and ``t_j >= t - w``; ``secs_since_prev`` is capped at ``NO_PREV_SECS``. The behavioural
features (Task 23) follow the same rule: ``cnt_1h``/``sum_1h`` over ``[t - 1 h, t)``, ``cnt_cat_24h`` over the
operations of the queried operation's category in ``[t - 24 h, t)`` (the state keeps every operation's
category), ``amt_ratio_30d`` from the queried amount and ``mean_amt_30d``.

Corridor state (Task 22, ``fraudaudit.corridor``): every operation in the state carries its AUTO-APPROVED amount
in whole cents (``auto_sum_cents``: the sum over ``(t - W, t]``, same-second operations already pushed included –
the rolling window of the corridor rule, unlike the model's strictly-earlier aggregates); history operations count
as auto-approved. ``spend`` (``SpendHistory``) holds every operation of the card that PASSED (all of them, not only
30 days) – the input of the corridor profile (ruling R26): the real history before the state time (it happened, no
fraud label is consulted) and every later operation that was auto-approved or confirmed by the owner, whoever sent
it; model declines and failed confirmations never enter it. The corridor reads it with a lag of 7 days, which is
what makes the corridor of an episode independent of the attacker's operations.

Time only moves forward. ``now`` is the latest of ``at`` (``from_history``) and every pushed
timestamp; queries and pushes before ``now`` raise ``ValueError``. Because of that, ``push`` may
drop every operation with ``t_j < now - 30 days``: no later query can still need it (it is outside
the 30-day window and its gap to any later query is > 30 days, i.e. capped anyway).
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
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


class _SpendBuffer:
    """Shared growable storage of ``SpendHistory`` objects: ``n`` slots are used; a history owns the slots before its
    own length and only the history whose length equals ``n`` may write the next slot (append-only, never
    overwritten)."""
    __slots__ = ("t", "a", "cs", "n")

    def __init__(self, t: np.ndarray, a: np.ndarray, cs: np.ndarray, capacity: int) -> None:
        n = t.size
        self.t = np.empty(capacity, dtype=np.int64)
        self.a = np.empty(capacity, dtype=np.float64)
        self.cs = np.empty(capacity + 1, dtype=np.float64)
        self.t[:n], self.a[:n], self.cs[:n + 1] = t, a, cs
        self.n = n


class SpendHistory:
    """The card's passed operations (time-ordered ns ``t``, amounts ``a``): the input of the corridor profile.

    Persistent: ``append`` returns a new object, so copies of a ``CardState`` can share one history. The storage is
    an amortised append-only buffer shared by the histories that extend one another (``_SpendBuffer``): appending to
    the newest history writes one slot, appending to an older one (a branch) copies its prefix first, so ``append``
    is O(1) amortised and no history ever sees another's later operations.
    ``window_quantile(day, L, W, q, lag)`` is the q-quantile (``np.quantile``, linear) of the rolling-W sums
    ``V_j = sum{a_i : t_j - W < t_i <= t_j}`` of every operation j with ``day - L <= t_j < day - lag``. ``V_j`` sums
    over time, not stream order: operations with the same timestamp as j all count (duplicates of one second give
    equal values), and so may operations up to ``t_j`` outside the lookback. With ``lag >= 0`` every input is
    before ``day - lag``, so a value for a day never changes once ``day - lag`` has passed; 0.0 without such
    operations. Values are cached per (day, L, W, q, lag); an append at ``t`` keeps every cached value with
    ``t >= day - lag`` (the new operation is not among its inputs, and the sums of its inputs end before it)."""

    __slots__ = ("_buf", "_n", "_cache")

    def __init__(self, t: np.ndarray, a: np.ndarray) -> None:
        t = np.asarray(t, dtype=np.int64)
        a = np.asarray(a, dtype=np.float64)
        if t.shape != a.shape or t.ndim != 1 or (t.size > 1 and (np.diff(t) < 0).any()):
            raise ValueError("SpendHistory needs time-ordered t and a of the same length")
        self._buf = _SpendBuffer(t, a, np.concatenate([[0.0], np.cumsum(a)]), max(16, t.size))
        self._n = int(t.size)
        self._cache: dict = {}

    @property
    def t(self) -> np.ndarray:
        return self._buf.t[:self._n]

    @property
    def a(self) -> np.ndarray:
        return self._buf.a[:self._n]

    @property
    def _cs(self) -> np.ndarray:
        return self._buf.cs[:self._n + 1]

    @classmethod
    def empty(cls) -> "SpendHistory":
        return cls(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))

    def __len__(self) -> int:
        return self._n

    def append(self, t_ns: int, amt: float) -> "SpendHistory":
        n, buf = self._n, self._buf
        if n and t_ns < buf.t[n - 1]:
            raise ValueError("SpendHistory.append: time only moves forward")
        if buf.n != n or n == buf.t.size:  # a branch of an older history, or full: copy the prefix
            buf = _SpendBuffer(buf.t[:n], buf.a[:n], buf.cs[:n + 1], max(16, 2 * (n + 1)))
        buf.t[n] = np.int64(t_ns)
        buf.a[n] = float(amt)
        buf.cs[n + 1] = buf.cs[n] + float(amt)
        buf.n = n + 1
        h = SpendHistory.__new__(SpendHistory)
        h._buf, h._n = buf, n + 1
        h._cache = {k: v for k, v in self._cache.items() if t_ns >= k[0] - k[4]}
        return h

    def window_quantile(self, day_ns: int, lookback_ns: int, w_ns: int, q: float, lag_ns: int) -> float:
        if lag_ns < 0:
            raise ValueError("lag must be >= 0")
        key = (int(day_ns), int(lookback_ns), int(w_ns), float(q), int(lag_ns))
        v = self._cache.get(key)
        if v is None:
            t = self.t
            lo = int(np.searchsorted(t, day_ns - lookback_ns, "left"))
            hi = int(np.searchsorted(t, day_ns - lag_ns, "left"))
            if hi <= lo:
                v = 0.0
            else:
                cs = self._cs
                tj = t[lo:hi]
                vals = cs[np.searchsorted(t, tj, "right")] - cs[np.searchsorted(t, tj - w_ns, "right")]
                v = float(np.quantile(vals, q))
            self._cache[key] = v
        return v


def _cents(a: float) -> int:
    return int(round(float(a) * 100))


class CardState:
    """Owner profile + operations of the last 30 days as time-ordered parallel lists ``ts_ns``, ``amt``,
    ``category`` and auto-approved cents; plus the card's passed operations ``spend`` (corridor profile)."""

    def __init__(self, profile: CardProfile) -> None:
        self.profile = profile
        self._t: list[int] = []  # ns, non-decreasing
        self._a: list[float] = []
        self._c: list[str] = []
        self._u: list[int] = []  # auto-approved amount in cents (0 if the operation was not auto-approved)
        self._now: int | None = None  # ns; None = no time seen yet
        self.spend: SpendHistory = SpendHistory.empty()

    @classmethod
    def from_history(cls, profile: CardProfile, history: pd.DataFrame, at: pd.Timestamp) -> "CardState":
        """State right before ``at``: the card's operations with ``at - 30d <= ts < at`` (all counted as
        auto-approved), and ``spend`` = every operation with ``ts < at`` (the real history passed; no fraud label
        is used, an ``is_fraud`` column is ignored)."""
        s = cls(profile)
        at_ns = pd.Timestamp(at).value
        h = history
        if "cc_num" in h:
            h = h[h["cc_num"] == profile.cc_num]
        t_all = h["ts"].to_numpy(dtype="datetime64[ns]").view("int64")
        amt_all = h["amt"].to_numpy(dtype=np.float64)
        m = (t_all < at_ns) & (t_all >= at_ns - _KEEP_NS)
        t, amt = t_all[m], amt_all[m]
        cat = h["category"].astype(str).to_numpy()[m]
        order = np.argsort(t, kind="stable")
        s._t = t[order].tolist()
        s._a = amt[order].tolist()
        s._c = cat[order].tolist()
        s._u = [_cents(a) for a in s._a]
        s._now = at_ns
        lm = t_all < at_ns
        lo = np.argsort(t_all[lm], kind="stable")
        s.spend = SpendHistory(t_all[lm][lo], amt_all[lm][lo])
        return s

    @property
    def now(self) -> int | None:
        """The state time in ns (``at`` of ``from_history`` or the latest pushed timestamp); None if unset."""
        return self._now

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

    def auto_sum_cents(self, t_ns: int, w_ns: int) -> int:
        """Auto-approved cents of the operations in the state with ``t - W < t_j <= t`` (corridor window): every
        operation already pushed with the same timestamp ``t`` counts – earlier ones of a same-second burst included,
        the queried operation itself only once pushed. A read only: no time check."""
        hi = bisect_right(self._t, t_ns)
        lo = bisect_right(self._t, t_ns - w_ns, 0, hi)
        return sum(self._u[lo:hi])

    def push(self, tx: Tx, *, auto_approved: bool = True, passed: bool = True) -> None:
        """Record ``tx`` as history for later operations; drop operations older than 30 days.

        ``auto_approved``: the amount counts in the corridor sums (history and callers without a corridor: True).
        ``passed``: the operation went through (auto-approved or confirmed), so it is appended to the corridor
        profile ``spend``; False for a model decline or a failed confirmation. No label of who sent it is used."""
        t = pd.Timestamp(tx.ts).value
        self._check_time(t, "push")
        self._t.append(t)
        self._a.append(float(tx.amt))
        self._c.append(str(tx.category))
        self._u.append(_cents(tx.amt) if auto_approved else 0)
        self._now = t
        if passed:
            self.spend = self.spend.append(t, float(tx.amt))
        drop = bisect_left(self._t, t - _KEEP_NS)  # keep t_j >= now - 30d
        if drop:
            del self._t[:drop], self._a[:drop], self._c[:drop], self._u[:drop]

    def copy(self) -> "CardState":
        c = CardState(self.profile)
        c._t, c._a, c._c, c._u = list(self._t), list(self._a), list(self._c), list(self._u)
        c._now = self._now
        c.spend = self.spend  # persistent: shared
        return c
