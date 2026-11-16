"""Eleven windowed per-card aggregates, batch (training) path.

Contract shared with the incremental serving path (``serving/state.py``):

* operation j of the same card counts for operation i iff ``t_j < t_i`` (strictly earlier:
  operations in the same second do not see each other) and ``t_j >= t_i - w``
  (inclusive lower bound: an operation exactly ``w`` earlier is in the window), w in {1, 7, 30} days;
* ``cnt_<w>`` = number of such operations, ``mean_amt_<w>`` = their mean ``amt``; empty window -> 0, 0.0;
* ``secs_since_prev`` = ``t_i - max{t_j : t_j < t_i}`` in seconds; no earlier operation, or a gap
  strictly greater than 30 days -> ``NO_PREV_SECS`` (2 592 000). A gap of exactly 30 days is
  2 592 000 either way, so the cap is continuous.

Behavioural features (Task 23, pre-registered; same strictly-earlier / inclusive-lower-bound rule):

* ``cnt_1h``, ``sum_1h`` = number and sum of ``amt`` of the operations with ``t_i - 1 h <= t_j < t_i``;
* ``cnt_cat_24h`` = number of operations of the SAME ``category`` as operation i with
  ``t_i - 24 h <= t_j < t_i``;
* ``amt_ratio_30d`` = ``log1p(amt_i / max(mean_amt_30d_i, 1.0))`` (amount of the operation itself against
  the card's 30-day mean; the floor 1.0 keeps a first operation finite: ``log1p(amt_i)``).

All values are computed in float64 and cast to float32 at the end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudaudit.features.columns import AGG_COLUMNS

WINDOW_DAYS = (1, 7, 30)
NO_PREV_SECS = 30 * 86_400  # 2 592 000
HOUR_SECS = 3_600
CAT_WINDOW_SECS = 86_400  # cnt_cat_24h
RATIO_FLOOR = 1.0  # amt_ratio_30d denominator floor
_NS = 1_000_000_000
_DAY_NS = 86_400 * _NS
_HOUR_NS = HOUR_SECS * _NS
_CAT_NS = CAT_WINDOW_SECS * _NS
_COL = {c: i for i, c in enumerate(AGG_COLUMNS)}


def amt_ratio(amt, mean_amt_30d):
    """``log1p(amt / max(mean_amt_30d, 1.0))`` in float64 (shared by the batch and serving paths)."""
    return np.log1p(np.asarray(amt, dtype=np.float64) / np.maximum(np.asarray(mean_amt_30d, dtype=np.float64),
                                                                   RATIO_FLOOR))


def batch_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """AGG_COLUMNS for every row of ``df`` (needs ``cc_num``, ``ts``, ``amt``, ``category``); index and row
    order kept.

    Vectorised per card (and per card x category for ``cnt_cat_24h``): ``np.searchsorted`` on the sorted
    int64 nanosecond timestamps and float64 cumulative sums of ``amt``; cast to float32 only at the end.
    """
    n = len(df)
    card = df["cc_num"].to_numpy()
    t = df["ts"].to_numpy(dtype="datetime64[ns]").view("int64")
    amt = df["amt"].to_numpy(dtype=np.float64)
    cat = pd.factorize(df["category"].astype(str))[0] if n else np.zeros(0, dtype=np.int64)

    res = np.empty((n, len(AGG_COLUMNS)), dtype=np.float64)

    order = np.lexsort((t, card))  # by card, then time (stable)
    card_s, t_s, amt_s = card[order], t[order], amt[order]
    bounds = np.flatnonzero(np.diff(card_s)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [n]))
    res_s = np.empty_like(res)
    for a, b in zip(starts, ends):
        if a == b:
            continue
        tc = t_s[a:b]
        csum = np.concatenate(([0.0], np.cumsum(amt_s[a:b])))
        hi = np.searchsorted(tc, tc, side="left")  # ops strictly before t_i: [0, hi)
        for k, d in enumerate(WINDOW_DAYS):
            lo = np.searchsorted(tc, tc - d * _DAY_NS, side="left")  # first t_j >= t_i - w
            cnt = hi - lo
            s = csum[hi] - csum[lo]
            res_s[a:b, k] = cnt
            res_s[a:b, 3 + k] = np.divide(s, cnt, out=np.zeros(b - a), where=cnt > 0)
        has_prev = hi > 0
        gap = np.full(b - a, float(NO_PREV_SECS))
        prev_t = tc[np.maximum(hi - 1, 0)]
        gap[has_prev] = (tc[has_prev] - prev_t[has_prev]) / _NS
        gap[gap > NO_PREV_SECS] = NO_PREV_SECS
        res_s[a:b, 6] = gap
        lo = np.searchsorted(tc, tc - _HOUR_NS, side="left")
        res_s[a:b, _COL["cnt_1h"]] = hi - lo
        res_s[a:b, _COL["sum_1h"]] = csum[hi] - csum[lo]
    res[order] = res_s

    # cnt_cat_24h: per card x category
    order = np.lexsort((t, cat, card))
    card_s, cat_s, t_s = card[order], cat[order], t[order]
    bounds = np.flatnonzero((np.diff(card_s) != 0) | (np.diff(cat_s) != 0)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [n]))
    cc = np.empty(n, dtype=np.float64)
    for a, b in zip(starts, ends):
        tc = t_s[a:b]
        cc[a:b] = np.searchsorted(tc, tc, side="left") - np.searchsorted(tc, tc - _CAT_NS, side="left")
    res[order, _COL["cnt_cat_24h"]] = cc

    res[:, _COL["amt_ratio_30d"]] = amt_ratio(amt, res[:, _COL["mean_amt_30d"]])
    return pd.DataFrame(res.astype(np.float32), index=df.index, columns=list(AGG_COLUMNS))
