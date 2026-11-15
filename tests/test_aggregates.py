import numpy as np
import pandas as pd
import pytest

from fraudaudit.features.aggregates import NO_PREV_SECS, batch_aggregates
from fraudaudit.features.columns import AGG_COLUMNS

T0 = pd.Timestamp("2020-02-01 10:00:00")
DAY = pd.Timedelta(days=1)
SEC = pd.Timedelta(seconds=1)


def _ops(*ops, card=1):
    """ops: (ts, amt) pairs of one card -> minimal frame sorted by ts like load_raw."""
    df = pd.DataFrame({
        "cc_num": np.int64(card),
        "ts": [pd.Timestamp(t) for t, _ in ops],
        "amt": [float(a) for _, a in ops],
    })
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def _naive(df: pd.DataFrame) -> pd.DataFrame:
    """O(n^2) reference of the contract: j counts for i iff same card, t_j < t_i, t_j >= t_i - w."""
    out = np.zeros((len(df), len(AGG_COLUMNS)), dtype=np.float64)
    card = df["cc_num"].to_numpy()
    t = df["ts"].to_numpy()
    amt = df["amt"].to_numpy(dtype=np.float64)
    for i in range(len(df)):
        prev = (card == card[i]) & (t < t[i])
        for k, d in enumerate((1, 7, 30)):
            m = prev & (t >= t[i] - np.timedelta64(d, "D"))
            out[i, k] = m.sum()
            out[i, 3 + k] = amt[m].mean() if m.any() else 0.0
        if prev.any():
            gap = (t[i] - t[prev].max()) / np.timedelta64(1, "s")
            out[i, 6] = NO_PREV_SECS if gap > NO_PREV_SECS else gap
        else:
            out[i, 6] = NO_PREV_SECS
    return pd.DataFrame(out, index=df.index, columns=list(AGG_COLUMNS))


def test_output_shape_and_dtypes(sparkov_like):
    out = batch_aggregates(sparkov_like)
    assert list(out.columns) == list(AGG_COLUMNS)
    assert out.index.equals(sparkov_like.index)
    assert (out.dtypes == "float32").all()
    assert not out.isna().any().any()


def test_window_excludes_same_second():
    df = _ops((T0 - 2 * 3600 * SEC, 10), (T0 - 3600 * SEC, 20), (T0, 30), (T0, 40))
    out = batch_aggregates(df)
    # both 10:00:00 ops see only the two earlier ops, not each other
    assert out["cnt_1d"].tolist() == [0, 1, 2, 2]
    assert out["mean_amt_1d"].iloc[2] == out["mean_amt_1d"].iloc[3] == pytest.approx(15.0)
    assert out["secs_since_prev"].iloc[2] == out["secs_since_prev"].iloc[3] == 3600


def test_same_second_first_ops_of_card():
    out = batch_aggregates(_ops((T0, 10), (T0, 20)))
    assert (out["cnt_30d"] == 0).all()
    assert (out["secs_since_prev"] == NO_PREV_SECS).all()


def test_window_boundaries():
    # exactly 7 days earlier is IN cnt_7d, 7 days + 1 s earlier is OUT
    out = batch_aggregates(_ops((T0 - 7 * DAY, 5), (T0, 1)))
    assert out["cnt_7d"].iloc[1] == 1 and out["cnt_1d"].iloc[1] == 0
    out = batch_aggregates(_ops((T0 - 7 * DAY - SEC, 5), (T0, 1)))
    assert out["cnt_7d"].iloc[1] == 0 and out["cnt_30d"].iloc[1] == 1
    assert out["mean_amt_7d"].iloc[1] == 0.0
    # same for 1 and 30 days
    out = batch_aggregates(_ops((T0 - DAY, 5), (T0, 1)))
    assert out["cnt_1d"].iloc[1] == 1
    out = batch_aggregates(_ops((T0 - DAY - SEC, 5), (T0, 1)))
    assert out["cnt_1d"].iloc[1] == 0 and out["cnt_7d"].iloc[1] == 1
    out = batch_aggregates(_ops((T0 - 30 * DAY, 5), (T0, 1)))
    assert out["cnt_30d"].iloc[1] == 1 and out["mean_amt_30d"].iloc[1] == 5.0
    out = batch_aggregates(_ops((T0 - 30 * DAY - SEC, 5), (T0, 1)))
    assert out["cnt_30d"].iloc[1] == 0 and out["mean_amt_30d"].iloc[1] == 0.0


def test_first_tx_defaults():
    out = batch_aggregates(_ops((T0, 42.5)))
    assert out.iloc[0].tolist() == [0, 0, 0, 0.0, 0.0, 0.0, 2_592_000]


def test_secs_since_prev_cap():
    out = batch_aggregates(_ops((T0 - 3 * DAY, 1), (T0, 1)))
    assert out["secs_since_prev"].iloc[1] == 3 * 86400
    # exactly 30 days: in the 30d window, gap 2 592 000 (cap is "strictly greater", same value)
    out = batch_aggregates(_ops((T0 - 30 * DAY, 1), (T0, 1)))
    assert out["secs_since_prev"].iloc[1] == 2_592_000
    # 30 days - 1 s: real gap; 30 days + 1 s: capped
    out = batch_aggregates(_ops((T0 - 30 * DAY + SEC, 1), (T0, 1)))
    assert out["secs_since_prev"].iloc[1] == 2_592_000 - 1
    out = batch_aggregates(_ops((T0 - 30 * DAY - SEC, 1), (T0, 1)))
    assert out["secs_since_prev"].iloc[1] == 2_592_000
    # a gap far beyond 30 days is capped, not the raw gap
    out = batch_aggregates(_ops((T0 - 90 * DAY, 1), (T0, 1)))
    assert out["secs_since_prev"].iloc[1] == 2_592_000


def test_mean_amount():
    df = _ops((T0 - 20 * DAY, 100), (T0 - 3 * DAY, 10), (T0 - 2 * 3600 * SEC, 20), (T0, 7))
    out = batch_aggregates(df)
    last = out.iloc[3]
    assert last["cnt_1d"] == 1 and last["mean_amt_1d"] == pytest.approx(20.0)
    assert last["cnt_7d"] == 2 and last["mean_amt_7d"] == pytest.approx(15.0)
    assert last["cnt_30d"] == 3 and last["mean_amt_30d"] == pytest.approx(130 / 3, rel=1e-6)


def test_cards_do_not_mix():
    a = _ops((T0 - SEC * 60, 100), card=1)
    b = _ops((T0, 5), card=2)
    df = pd.concat([a, b]).sort_values("ts", kind="stable").reset_index(drop=True)
    out = batch_aggregates(df)
    assert (out["cnt_30d"] == 0).all()
    assert (out["secs_since_prev"] == NO_PREV_SECS).all()


def test_index_and_row_order_preserved():
    df = _ops((T0 - DAY / 2, 4), (T0, 8), (T0 - DAY / 4, 6))  # sorted by ts inside _ops
    df = df.iloc[[2, 0, 1]]  # shuffle rows
    df.index = [30, 10, 20]
    out = batch_aggregates(df)
    assert out.index.tolist() == [30, 10, 20]
    # index 30 = T0, 10 = T0 - 12 h, 20 = T0 - 6 h
    assert out["cnt_1d"].tolist() == [2, 0, 1]


def test_matches_naive_reference(sparkov_like):
    out = batch_aggregates(sparkov_like)
    ref = _naive(sparkov_like)
    for c in ("cnt_1d", "cnt_7d", "cnt_30d"):
        assert (out[c].to_numpy() == ref[c].to_numpy()).all(), c
    for c in ("mean_amt_1d", "mean_amt_7d", "mean_amt_30d"):
        np.testing.assert_allclose(out[c].to_numpy(), ref[c].to_numpy(), rtol=0, atol=1e-4)
    assert (out["secs_since_prev"].to_numpy() == ref["secs_since_prev"].to_numpy()).all()
    # the fixture exercises the interesting cases
    assert (ref["cnt_1d"] > 0).any() and (ref["secs_since_prev"] == NO_PREV_SECS).sum() > 40
