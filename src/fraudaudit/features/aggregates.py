"""Seven windowed per-card aggregates, batch (training) path.

Contract shared with the incremental serving path (``serving/state.py``):

* operation j of the same card counts for operation i iff ``t_j < t_i`` (strictly earlier:
  operations in the same second do not see each other) and ``t_j >= t_i - w``
  (inclusive lower bound: an operation exactly ``w`` earlier is in the window), w in {1, 7, 30} days;
* ``cnt_<w>`` = number of such operations, ``mean_amt_<w>`` = their mean ``amt``; empty window -> 0, 0.0;
* ``secs_since_prev`` = ``t_i - max{t_j : t_j < t_i}`` in seconds; no earlier operation, or a gap
  strictly greater than 30 days -> ``NO_PREV_SECS`` (2 592 000). A gap of exactly 30 days is
  2 592 000 either way, so the cap is continuous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudaudit.features.columns import AGG_COLUMNS

WINDOW_DAYS = (1, 7, 30)
NO_PREV_SECS = 30 * 86_400  # 2 592 000
_NS = 1_000_000_000
_DAY_NS = 86_400 * _NS


def batch_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """AGG_COLUMNS for every row of ``df`` (needs ``cc_num``, ``ts``, ``amt``); index and row order kept.

    Vectorised per card: ``np.searchsorted`` on the card's sorted int64 nanosecond timestamps
    and float64 cumulative sums of ``amt``; cast to float32 only at the end.
    """
    n = len(df)
    card = df["cc_num"].to_numpy()
    t = df["ts"].to_numpy(dtype="datetime64[ns]").view("int64")
    amt = df["amt"].to_numpy(dtype=np.float64)

    order = np.lexsort((t, card))  # by card, then time (stable)
    card_s, t_s, amt_s = card[order], t[order], amt[order]
    bounds = np.flatnonzero(np.diff(card_s)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [n]))

    res = np.empty((n, len(AGG_COLUMNS)), dtype=np.float64)
    for a, b in zip(starts, ends):
        tc = t_s[a:b]
        csum = np.concatenate(([0.0], np.cumsum(amt_s[a:b])))
        hi = np.searchsorted(tc, tc, side="left")  # ops strictly before t_i: [0, hi)
        for k, d in enumerate(WINDOW_DAYS):
            lo = np.searchsorted(tc, tc - d * _DAY_NS, side="left")  # first t_j >= t_i - w
            cnt = hi - lo
            s = csum[hi] - csum[lo]
            res[a:b, k] = cnt
            res[a:b, 3 + k] = np.divide(s, cnt, out=np.zeros(b - a), where=cnt > 0)
        has_prev = hi > 0
        gap = np.full(b - a, float(NO_PREV_SECS))
        prev_t = tc[np.maximum(hi - 1, 0)]
        gap[has_prev] = (tc[has_prev] - prev_t[has_prev]) / _NS
        gap[gap > NO_PREV_SECS] = NO_PREV_SECS
        res[a:b, 6] = gap

    out = np.empty_like(res)
    out[order] = res
    return pd.DataFrame(out.astype(np.float32), index=df.index, columns=list(AGG_COLUMNS))
