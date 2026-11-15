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
COUNT_COLS = [N_BASE + AGG_COLUMNS.index(c) for c in ("cnt_1d", "cnt_7d", "cnt_30d", "secs_since_prev")]
MEAN_COLS = [N_BASE + AGG_COLUMNS.index(c) for c in ("mean_amt_1d", "mean_amt_7d", "mean_amt_30d")]


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
    """Serving-path features of every row: each card replayed through CardState.push in (ts, row) order."""
    out = np.empty((len(df), len(FEATURE_COLUMNS)), dtype=np.float32)
    df = df.sort_values(["ts", "row"], kind="stable")
    for _, g in df.groupby("cc_num", sort=False):
        state = None
        for idx, r in zip(g.index, g.itertuples(index=False)):
            prof = _row_profile(r)  # fixture owner fields vary per row; the base part must follow the row
            if state is None:
                state = CardState.from_history(prof, g.iloc[:0], at=pd.Timestamp(r.ts))
            state.profile = prof
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


def test_from_history_takes_only_last_30d_before_at(sparkov_like):
    df = sparkov_like
    card = df["cc_num"].iloc[0]
    g = df[df["cc_num"] == card]
    at = g["ts"].iloc[len(g) // 2]
    state = CardState.from_history(profile_of(df, card), df, at=at)
    want = g[(g["ts"] < at) & (g["ts"] >= at - 30 * DAY)]
    assert len(state) == len(want)
    agg = state.aggregates(at)
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
    agg = s.aggregates(T0 + 40 * DAY)
    assert agg[2] == 3 and agg[5] == pytest.approx(30.0)
    # a query going back in time cannot be answered from a truncated state
    with pytest.raises(ValueError):
        s.aggregates(T0 + 30 * DAY)
    with pytest.raises(ValueError):
        s.push(Tx(T0 + 29 * DAY, 1.0, "fraud_A", "home", 40.0, -80.0))


def test_same_second_ops_not_counted():
    s = CardState(_profile())
    s.push(Tx(T0 - 3600 * SEC, 10.0, "fraud_A", "home", 40.0, -80.0))
    s.push(Tx(T0, 30.0, "fraud_A", "home", 40.0, -80.0))
    agg = s.aggregates(T0)
    assert agg[0] == 1 and agg[3] == pytest.approx(10.0) and agg[6] == 3600


def test_first_tx_defaults():
    s = CardState(_profile())
    np.testing.assert_array_equal(s.aggregates(T0), [0, 0, 0, 0.0, 0.0, 0.0, NO_PREV_SECS])
    s.push(Tx(T0, 10.0, "fraud_A", "home", 40.0, -80.0))
    agg = s.aggregates(T0 + 31 * DAY)  # only op is older than 30 days
    np.testing.assert_array_equal(agg, [0, 0, 0, 0.0, 0.0, 0.0, NO_PREV_SECS])


def test_copy_is_independent():
    s = CardState(_profile())
    s.push(Tx(T0, 10.0, "fraud_A", "home", 40.0, -80.0))
    c = s.copy()
    c.push(Tx(T0 + DAY, 20.0, "fraud_A", "home", 40.0, -80.0))
    assert len(s) == 1 and len(c) == 2
    assert s.aggregates(T0 + DAY)[2] == 1


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
