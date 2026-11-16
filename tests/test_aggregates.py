import numpy as np
import pandas as pd
import pytest

from fraudaudit.features.aggregates import NO_PREV_SECS, batch_aggregates
from fraudaudit.features.columns import AGG_COLUMNS, CATEGORIES

T0 = pd.Timestamp("2020-02-01 10:00:00")
DAY = pd.Timedelta(days=1)
SEC = pd.Timedelta(seconds=1)


HOUR = pd.Timedelta(hours=1)
INT_COLS = ("cnt_1d", "cnt_7d", "cnt_30d", "secs_since_prev", "cnt_1h", "cnt_cat_24h")
FLOAT_COLS = ("mean_amt_1d", "mean_amt_7d", "mean_amt_30d", "sum_1h", "amt_ratio_30d")


def _ops(*ops, card=1):
    """ops: (ts, amt) or (ts, amt, category) tuples of one card -> minimal frame sorted by ts like load_raw
    (category defaults to ``home``)."""
    df = pd.DataFrame({
        "cc_num": np.int64(card),
        "ts": [pd.Timestamp(o[0]) for o in ops],
        "amt": [float(o[1]) for o in ops],
        "category": [o[2] if len(o) > 2 else "home" for o in ops],
    })
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def _naive(df: pd.DataFrame) -> pd.DataFrame:
    """O(n^2) reference of the contract: j counts for i iff same card, t_j < t_i, t_j >= t_i - w
    (cnt_cat_24h: and same category; amt_ratio_30d = log1p(amt_i / max(mean_amt_30d_i, 1)))."""
    col = {c: i for i, c in enumerate(AGG_COLUMNS)}
    out = np.zeros((len(df), len(AGG_COLUMNS)), dtype=np.float64)
    card = df["cc_num"].to_numpy()
    t = df["ts"].to_numpy()
    amt = df["amt"].to_numpy(dtype=np.float64)
    cat = df["category"].astype(str).to_numpy()
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
        h = prev & (t >= t[i] - np.timedelta64(3600, "s"))
        out[i, col["cnt_1h"]] = h.sum()
        out[i, col["sum_1h"]] = amt[h].sum()
        out[i, col["cnt_cat_24h"]] = (prev & (t >= t[i] - np.timedelta64(86400, "s")) & (cat == cat[i])).sum()
        out[i, col["amt_ratio_30d"]] = np.log1p(amt[i] / max(out[i, col["mean_amt_30d"]], 1.0))
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
    assert out.iloc[0].tolist() == pytest.approx([0, 0, 0, 0.0, 0.0, 0.0, 2_592_000, 0, 0.0, 0,
                                                  np.log1p(42.5)])


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


def _assert_matches_naive(df):
    out = batch_aggregates(df)
    ref = _naive(df)
    for c in INT_COLS:
        assert (out[c].to_numpy() == ref[c].to_numpy()).all(), c
    for c in FLOAT_COLS:
        np.testing.assert_allclose(out[c].to_numpy(), ref[c].to_numpy(), rtol=1e-6, atol=1e-4, err_msg=c)
    return ref


def test_matches_naive_reference(sparkov_like):
    ref = _assert_matches_naive(sparkov_like)
    # the fixture exercises the interesting cases
    assert (ref["cnt_1d"] > 0).any() and (ref["secs_since_prev"] == NO_PREV_SECS).sum() > 40
    assert (ref["cnt_1h"] > 0).any()


def _dense(seed=0, n_cards=4, n=300):
    """Bursty random cards over 10 days with 3 categories: many 1 h / same-category-24 h neighbours,
    same-second duplicates and ops exactly 1 h / 24 h apart."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(n_cards):
        secs = rng.integers(0, 10 * 86400, n)
        secs[: n // 10] = secs[n // 10: 2 * (n // 10)] + rng.choice([0, 3600, 86400], n // 10)  # exact boundaries
        parts.append(pd.DataFrame({"cc_num": np.int64(c), "ts": T0 + pd.to_timedelta(secs, unit="s"),
                                   "amt": np.round(rng.lognormal(3.0, 1.2, n), 2).clip(0.01, None),
                                   "category": rng.choice(list(CATEGORIES[:3]), n)}))
    df = pd.concat(parts).sort_values("ts", kind="stable").reset_index(drop=True)
    return df


@pytest.mark.parametrize("seed", [0, 1])
def test_behavioural_match_naive_on_dense_cards(seed):
    df = _dense(seed)
    assert df.duplicated(["cc_num", "ts"]).any()
    ref = _assert_matches_naive(df)
    assert (ref["cnt_1h"] > 2).any() and (ref["cnt_cat_24h"] > 2).any()
    assert (ref["cnt_cat_24h"] < ref["cnt_1d"]).any()  # other categories are not counted


def test_one_hour_window():
    # exactly 1 h earlier is IN, 1 h + 1 s is OUT; same second is OUT
    df = _ops((T0 - 2 * HOUR, 100), (T0 - HOUR - SEC, 50), (T0 - HOUR, 10), (T0 - 60 * SEC, 5), (T0, 7), (T0, 8))
    out = batch_aggregates(df)
    assert out["cnt_1h"].tolist() == [0, 1, 2, 2, 2, 2]
    assert out["sum_1h"].tolist() == pytest.approx([0.0, 100.0, 150.0, 60.0, 15.0, 15.0])
    assert out["cnt_1d"].iloc[4] == 4


def test_same_category_24h():
    df = _ops((T0 - DAY - SEC, 1, "travel"), (T0 - DAY, 1, "travel"), (T0 - HOUR, 1, "home"),
              (T0 - 60 * SEC, 1, "travel"), (T0, 1, "travel"), (T0, 1, "travel"), (T0 + SEC, 1, "home"))
    out = batch_aggregates(df)
    # travel at T0: ops at T0 - 1 d (inclusive) and T0 - 60 s; not T0 - 1 d - 1 s, not home, not same second
    assert out["cnt_cat_24h"].tolist() == [0, 1, 0, 2, 2, 2, 1]


def test_amt_ratio_30d():
    df = _ops((T0 - 31 * DAY, 1000), (T0 - 2 * DAY, 20), (T0 - DAY, 40), (T0, 90), (T0 + 31 * DAY, 0.5))
    out = batch_aggregates(df)
    r = out["amt_ratio_30d"].to_numpy(dtype=np.float64)
    assert r[3] == pytest.approx(np.log1p(90 / 30), rel=1e-6)  # mean of the 2 ops in 30 d (not the 1000)
    assert r[0] == pytest.approx(np.log1p(1000.0), rel=1e-6)  # no history: floor 1.0
    assert r[4] == pytest.approx(np.log1p(0.5), rel=1e-6)  # empty 30-day window
    # a mean below 1 $ is floored at 1
    out = batch_aggregates(_ops((T0 - DAY, 0.2), (T0, 3)))
    assert out["amt_ratio_30d"].iloc[1] == pytest.approx(np.log1p(3.0), rel=1e-6)
