import dataclasses

import numpy as np
import pandas as pd
import pytest

from fraudaudit.features.aggregates import NO_PREV_SECS
from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import AGG_COLUMNS, BASE_COLUMNS, CATEGORIES, FEATURE_COLUMNS
from fraudaudit.model.pipeline import features
from fraudaudit.serving.core import Decision, Serving
from fraudaudit.serving.state import CardProfile, CardState, Tx, profile_of

T0 = pd.Timestamp("2020-02-01 10:00:00")
DAY = pd.Timedelta(days=1)
SEC = pd.Timedelta(seconds=1)
N_BASE = len(BASE_COLUMNS)
COUNT_COLS = [N_BASE + AGG_COLUMNS.index(c)
              for c in ("cnt_1d", "cnt_7d", "cnt_30d", "secs_since_prev", "cnt_1h", "cnt_cat_24h")]
MEAN_COLS = [N_BASE + AGG_COLUMNS.index(c)
             for c in ("mean_amt_1d", "mean_amt_7d", "mean_amt_30d", "sum_1h", "amt_ratio_30d")]
LOG1P_1 = float(np.log1p(1.0))


class StubModel:
    """Duck-typed FraudModel: fixed probability, real encoder."""

    def __init__(self, encoder, tau, p):
        self.encoder, self.tau, self.p = encoder, tau, p
        self.calls = 0

    def predict_proba(self, F):
        self.calls += 1
        assert F.shape == (1, len(FEATURE_COLUMNS))
        return np.full(len(F), self.p, dtype=np.float64)


def _row_profile(r) -> CardProfile:
    return CardProfile(cc_num=int(r.cc_num), gender=str(r.gender), dob=pd.Timestamp(r.dob),
                       lat=float(r.lat), long=float(r.long), city_pop=int(r.city_pop), state=str(r.state))


def _row_tx(r) -> Tx:
    return Tx(ts=pd.Timestamp(r.ts), amt=float(r.amt), merchant=str(r.merchant), category=str(r.category),
              merch_lat=float(r.merch_lat), merch_long=float(r.merch_long))


def _profile(cc=1) -> CardProfile:
    return CardProfile(cc_num=cc, gender="F", dob=pd.Timestamp("1980-05-05"), lat=40.0, long=-80.0,
                       city_pop=1000, state="NC")


def _replay(df: pd.DataFrame, serving: Serving) -> np.ndarray:
    """Serving-path features of every row: each card's state from ``profile_of``, replayed through
    ``CardState.push`` in (ts, row) order; the profile is never overridden."""
    out = np.empty((len(df), len(FEATURE_COLUMNS)), dtype=np.float32)
    df = df.sort_values(["ts", "row"], kind="stable")
    for card, g in df.groupby("cc_num", sort=False):
        state = CardState.from_history(profile_of(df, card), g.iloc[:0], at=g["ts"].iloc[0])
        for idx, r in zip(g.index, g.itertuples(index=False)):
            tx = _row_tx(r)
            out[idx] = serving.features(state, tx)
            state.push(tx)
    return out


def test_serving_matches_batch_features(sparkov_like):
    df = sparkov_like
    enc = BaseEncoder().fit(df)
    serving = Serving(StubModel(enc, tau=0.5, p=0.1))
    want = features(df, enc)
    got = _replay(df, serving)
    # the fixture has same-second ops of one card: they must not see each other on either path
    assert df.duplicated(["cc_num", "ts"]).sum() > 0
    np.testing.assert_array_equal(got[:, :N_BASE], want[:, :N_BASE])
    np.testing.assert_array_equal(got[:, COUNT_COLS], want[:, COUNT_COLS])
    np.testing.assert_allclose(got[:, MEAN_COLS], want[:, MEAN_COLS], rtol=1e-6, atol=1e-4)


def test_fast_base_path_matches_transform_all_categories_and_genders(sparkov_like):
    df = sparkov_like
    enc = BaseEncoder().fit(df)
    serving = Serving(StubModel(enc, tau=0.5, p=0.1))
    assert set(df["category"]) == set(CATEGORIES) and set(df["gender"]) == {"F", "M"}
    want = enc.transform(df).to_numpy()
    for i, r in enumerate(df.itertuples(index=False)):
        state = CardState(_row_profile(r))
        np.testing.assert_array_equal(serving.features(state, _row_tx(r))[:N_BASE], want[i])


@pytest.mark.parametrize("ts", ["2020-02-01 00:00:00", "2020-02-01 06:00:00", "2020-02-01 12:00:00",
                                "2020-02-01 23:59:00", "2020-02-01 23:59:59", "2020-02-01 03:20:15"])
def test_fast_path_hour_features_equal_transform(sparkov_like, ts):
    df = sparkov_like.iloc[[0]].copy()
    df["ts"] = pd.Timestamp(ts)
    enc = BaseEncoder().fit(sparkov_like)
    serving = Serving(StubModel(enc, tau=0.5, p=0.1))
    r = next(df.itertuples(index=False))
    got = serving.features(CardState(_row_profile(r)), _row_tx(r))[:N_BASE]
    want = enc.transform(df).to_numpy()[0]
    cols = [BASE_COLUMNS.index("hour_sin"), BASE_COLUMNS.index("hour_cos")]
    np.testing.assert_array_equal(got[cols], want[cols])
    np.testing.assert_array_equal(got, want)


def test_from_history_takes_only_last_30d_before_at(sparkov_like):
    df = sparkov_like
    card = df["cc_num"].iloc[0]
    g = df[df["cc_num"] == card]
    at = g["ts"].iloc[len(g) // 2]
    state = CardState.from_history(profile_of(df, card), df, at=at)
    want = g[(g["ts"] < at) & (g["ts"] >= at - 30 * DAY)]
    assert len(state) == len(want)
    agg = state.aggregates(at, 1.0, "home")
    assert agg[2] == len(want)
    assert agg[5] == pytest.approx(want["amt"].mean() if len(want) else 0.0)


def test_state_truncates_30d():
    s = CardState(_profile())
    s.push(Tx(T0, 10.0, "fraud_A", "home", 40.0, -80.0))
    s.push(Tx(T0 + 10 * DAY, 20.0, "fraud_A", "home", 40.0, -80.0))
    s.push(Tx(T0 + 30 * DAY, 30.0, "fraud_A", "home", 40.0, -80.0))
    assert len(s) == 3  # first op is exactly 30 days old: still needed
    s.push(Tx(T0 + 30 * DAY + SEC, 40.0, "fraud_A", "home", 40.0, -80.0))
    assert len(s) == 3  # now strictly older than 30 days: dropped
    agg = s.aggregates(T0 + 40 * DAY, 1.0, "home")
    assert agg[2] == 3 and agg[5] == pytest.approx(30.0)
    # a query going back in time cannot be answered from a truncated state
    with pytest.raises(ValueError):
        s.aggregates(T0 + 30 * DAY, 1.0, "home")
    with pytest.raises(ValueError):
        s.push(Tx(T0 + 29 * DAY, 1.0, "fraud_A", "home", 40.0, -80.0))


def test_profile_of_rejects_inconsistent_card(sparkov_like):
    df = sparkov_like.copy()
    card = df["cc_num"].iloc[0]
    assert profile_of(df, card).cc_num == card
    for col, value in (("gender", "X"), ("city_pop", 1), ("lat", 0.0), ("dob", "1900-01-01")):
        bad = df.copy()
        bad.loc[bad.index[bad["cc_num"] == card][-1], col] = value
        with pytest.raises(ValueError, match=col):
            profile_of(bad, card)
    with pytest.raises(ValueError):
        profile_of(df, -1)


@pytest.mark.parametrize("days, k", [(1, 0), (7, 1)])
def test_window_lower_bound_inclusive(days, k):
    t = T0 + 40 * DAY
    s = CardState(_profile())
    s.push(Tx(t - days * DAY - SEC, 100.0, "fraud_A", "home", 40.0, -80.0))  # just outside
    s.push(Tx(t - days * DAY, 10.0, "fraud_A", "home", 40.0, -80.0))  # exactly w earlier: inside
    agg = s.aggregates(t, 1.0, "home")
    assert agg[k] == 1 and agg[3 + k] == pytest.approx(10.0)
    assert agg[2] == 2  # both within 30 days
    assert agg[6] == days * 86_400


def test_same_second_ops_not_counted():
    s = CardState(_profile())
    s.push(Tx(T0 - 3600 * SEC, 10.0, "fraud_A", "home", 40.0, -80.0))
    s.push(Tx(T0, 30.0, "fraud_A", "home", 40.0, -80.0))
    agg = s.aggregates(T0, 1.0, "home")
    assert agg[0] == 1 and agg[3] == pytest.approx(10.0) and agg[6] == 3600


def test_first_tx_defaults():
    s = CardState(_profile())
    np.testing.assert_array_equal(s.aggregates(T0, 1.0, "home"), [0, 0, 0, 0.0, 0.0, 0.0, NO_PREV_SECS, 0, 0.0, 0,
                                                                  LOG1P_1])
    s.push(Tx(T0, 10.0, "fraud_A", "home", 40.0, -80.0))
    agg = s.aggregates(T0 + 31 * DAY, 1.0, "home")  # only op is older than 30 days
    np.testing.assert_array_equal(agg, [0, 0, 0, 0.0, 0.0, 0.0, NO_PREV_SECS, 0, 0.0, 0, LOG1P_1])


def test_copy_is_independent():
    s = CardState(_profile())
    s.push(Tx(T0, 10.0, "fraud_A", "home", 40.0, -80.0))
    c = s.copy()
    c.push(Tx(T0 + DAY, 20.0, "fraud_A", "home", 40.0, -80.0))
    assert len(s) == 1 and len(c) == 2
    assert s.aggregates(T0 + DAY, 1.0, "home")[2] == 1


def _serving_for(df, tau, p):
    enc = BaseEncoder().fit(df)
    model = StubModel(enc, tau=tau, p=p)
    return Serving(model), model


def test_decide_threshold(sparkov_like):
    r = next(sparkov_like.itertuples(index=False))
    tau = 0.04746
    for p, declined in ((tau, True), (np.nextafter(tau, 0.0), False), (0.9, True)):
        serving, model = _serving_for(sparkov_like, tau, p)
        d = serving.decide(CardState(_row_profile(r)), _row_tx(r))
        assert d == Decision(p=float(p), declined=declined)
        assert model.calls == 1


def test_features_reject_non_finite(sparkov_like):
    r = next(sparkov_like.itertuples(index=False))
    serving, _ = _serving_for(sparkov_like, 0.5, 0.1)
    bad = dataclasses.replace(_row_tx(r), amt=float("nan"))
    with pytest.raises(ValueError):
        serving.features(CardState(_row_profile(r)), bad)


def test_unknown_merchant_rejected(sparkov_like):
    r = next(sparkov_like.itertuples(index=False))
    serving, _ = _serving_for(sparkov_like, 0.5, 0.1)
    with pytest.raises(ValueError):
        serving.features(CardState(_row_profile(r)), dataclasses.replace(_row_tx(r), merchant="fraud_Nope"))


def test_decide_with_real_model(sparkov_like):
    import time

    from fraudaudit.model.pipeline import train_model

    df = sparkov_like
    model, _ = train_model(df, seed=0, epochs=1)
    serving = Serving(model)
    card = df["cc_num"].iloc[0]
    g = df[df["cc_num"] == card]
    state = CardState.from_history(profile_of(df, card), df, at=g["ts"].iloc[-1])
    tx = _row_tx(next(g.iloc[-1:].itertuples(index=False)))
    d = serving.decide(state, tx)
    p_ref = model.predict_proba(serving.features(state, tx)[np.newaxis, :])[0]
    assert isinstance(d.p, float) and 0.0 <= d.p <= 1.0
    assert d.p == p_ref
    assert d.declined == (d.p >= model.tau)
    times = []
    for _ in range(50):
        a = time.perf_counter()
        serving.decide(state, tx)
        times.append(time.perf_counter() - a)
    assert np.median(times) < 0.020  # loose smoke bound; ~1 ms expected on CPU


# ---- behavioural features (Task 23) ----------------------------------------------------------------

def _push(s, t, amt, cat):
    s.push(Tx(t, float(amt), "fraud_A", cat, 40.0, -80.0))


def test_behavioural_state_boundaries():
    from fraudaudit.features.columns import AGG_COLUMNS as A
    i = {c: A.index(c) for c in ("cnt_1h", "sum_1h", "cnt_cat_24h", "amt_ratio_30d", "mean_amt_30d")}
    s = CardState(_profile())
    _push(s, T0 - DAY - SEC, 1000, "travel")  # outside 24 h
    _push(s, T0 - DAY, 30, "travel")  # exactly 24 h: inside
    _push(s, T0 - 3600 * SEC - SEC, 50, "travel")  # outside 1 h
    _push(s, T0 - 3600 * SEC, 10, "home")  # exactly 1 h: inside
    _push(s, T0 - 60 * SEC, 5, "travel")
    _push(s, T0, 999, "travel")  # same second: never counted
    agg = s.aggregates(T0, 90.0, "travel")
    assert agg[i["cnt_1h"]] == 2 and agg[i["sum_1h"]] == pytest.approx(15.0)
    assert agg[i["cnt_cat_24h"]] == 3  # 30 (T0 - 1 d), 50, 5; not 1000, not home, not the same-second op
    assert agg[i["amt_ratio_30d"]] == pytest.approx(np.log1p(90.0 / agg[i["mean_amt_30d"]]))
    assert s.aggregates(T0, 90.0, "home")[i["cnt_cat_24h"]] == 1
    assert s.aggregates(T0, 90.0, "kids_pets")[i["cnt_cat_24h"]] == 0


def test_behavioural_categories_survive_truncation_copy_and_history(sparkov_like):
    s = CardState(_profile())
    _push(s, T0, 10, "travel")
    _push(s, T0 + 10 * DAY, 20, "home")
    _push(s, T0 + 31 * DAY, 30, "home")  # drops the first op
    k = AGG_COLUMNS.index("cnt_cat_24h")
    assert len(s) == 2 and s._c == ["home", "home"]
    c = s.copy()
    _push(c, T0 + 31 * DAY + SEC, 1, "home")
    assert s._c == ["home", "home"] and c._c == ["home", "home", "home"]
    assert c.aggregates(T0 + 31 * DAY + 2 * SEC, 1.0, "home")[k] == 2
    assert s.aggregates(T0 + 31 * DAY + 2 * SEC, 1.0, "home")[k] == 1
    # from_history keeps every operation's category
    df = sparkov_like
    card = df["cc_num"].iloc[0]
    g = df[df["cc_num"] == card]
    at = g["ts"].iloc[-1]
    st = CardState.from_history(profile_of(df, card), df, at=at)
    want = g[(g["ts"] < at) & (g["ts"] >= at - 30 * DAY)].sort_values("ts", kind="stable")
    assert st._c == want["category"].tolist()


@pytest.mark.parametrize("seed", [0, 1])
def test_serving_matches_batch_on_dense_cards(seed):
    """Dense bursty cards (many 1 h and same-category neighbours, exact boundaries, same-second ops):
    the incremental state equals the batch aggregates (counts exactly)."""
    from fraudaudit.features.aggregates import batch_aggregates
    from test_aggregates import _dense

    df = _dense(seed)
    want = batch_aggregates(df).to_numpy()
    got = np.empty_like(want)
    for card, g in df.sort_values("ts", kind="stable").groupby("cc_num", sort=False):
        s = CardState(_profile(int(card)))
        for idx, r in zip(g.index, g.itertuples(index=False)):
            got[idx] = s.aggregates(r.ts, r.amt, r.category).astype(np.float32)
            _push(s, r.ts, r.amt, r.category)
    cnt = [AGG_COLUMNS.index(c) for c in ("cnt_1d", "cnt_7d", "cnt_30d", "secs_since_prev", "cnt_1h", "cnt_cat_24h")]
    flt = [AGG_COLUMNS.index(c) for c in ("mean_amt_1d", "mean_amt_7d", "mean_amt_30d", "sum_1h", "amt_ratio_30d")]
    np.testing.assert_array_equal(got[:, cnt], want[:, cnt])
    np.testing.assert_allclose(got[:, flt], want[:, flt], rtol=1e-6, atol=1e-4)
    assert (want[:, AGG_COLUMNS.index("cnt_cat_24h")] > 2).any()
