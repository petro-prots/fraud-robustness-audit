import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudaudit.data import Episode, fraud_episodes
from fraudaudit.serving.state import OWNER_FIELDS, Tx, profile_of
from fraudaudit.threat import (
    EpisodeContext, MerchantTable, Strategy, ThreatModel, context_for, contexts_for, feasible, realize,
)

ROOT = Path(__file__).resolve().parents[1]
TM_PATH = ROOT / "threat-models" / "realistic.yaml"


@pytest.fixture
def tm():
    return ThreatModel.from_yaml(TM_PATH)


@pytest.fixture
def setup(sparkov_like, tm):
    df = sparkov_like
    merchants = MerchantTable.from_df(df)
    eps = fraud_episodes(df, {"fit", "val", "test"})
    return df, merchants, eps


def _random_txs(ctx, tm, merchants, seed):
    rng = np.random.default_rng(seed)
    s = Strategy.random(tm, len(merchants), rng)
    return s, realize(s, ctx, tm, merchants, rng)


# ---- ThreatModel / MerchantTable ---------------------------------------------------------------

def test_threat_model_from_yaml(tm):
    assert tm == ThreatModel(rho=0.8, k_max=20, amount_min=1.00, amount_max=1376.04, window_days=7,
                             max_declines=1, knowledge="decision", recon_budget_ops=2000,
                             cards_per_eval=5, merchant_coord_radius=1.0, min_gap_secs=60)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tm.rho = 0.5


def test_threat_model_rejects_unknown_or_missing_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(TM_PATH.read_text() + "extra: 1\n")
    with pytest.raises(ValueError, match="extra"):
        ThreatModel.from_yaml(p)
    p.write_text("rho: 0.8\n")
    with pytest.raises(ValueError, match="k_max"):
        ThreatModel.from_yaml(p)


def test_merchant_table_ordered_by_category_then_name(sparkov_like):
    df = sparkov_like
    mt = MerchantTable.from_df(df)
    first = df.drop_duplicates("merchant").set_index("merchant")["category"]
    assert mt.names == sorted(df["merchant"].unique(), key=lambda n: (first[n], n))
    assert all(n.startswith("fraud_") for n in mt.names)
    assert mt.category == [first[n] for n in mt.names]
    assert mt.category == sorted(mt.category)  # one contiguous block per category
    assert len(mt) == len(mt.names) == df["merchant"].nunique()


# ---- context ---------------------------------------------------------------------------------

def test_context_excludes_episode_and_later_ops(setup):
    df, _, eps = setup
    ep = eps[0]
    g = df[df["cc_num"] == ep.card]
    # leak bait: legit ops of the card after start and in the very second of start
    later = g[(g["ts"] > ep.start) & ~g["row"].isin(ep.rows)]
    assert len(later) > 0 and len(set(ep.rows)) > 1
    bait = g[g["row"] == ep.rows[0]].assign(is_fraud=0, row=df["row"].max() + 1, amt=999.99)
    df2 = pd.concat([df, bait], ignore_index=True).sort_values(["ts", "row"], kind="stable")
    for d in (df, df2):
        ctx = context_for(d, ep)
        assert isinstance(ctx, EpisodeContext)
        assert ctx.episode is ep
        assert ctx.profile == profile_of(d, ep.card)
        assert ctx.state.profile == ctx.profile
        assert max(ctx.state._t) < ep.start.value
        assert 999.99 not in ctx.state._a
        expected = g[(g["ts"] < ep.start) & (g["ts"] >= ep.start - pd.Timedelta(days=30))]
        assert not set(expected["row"]) & set(ep.rows)
        assert len(ctx.state) == len(expected)
        assert ctx.state._a == expected["amt"].tolist()


def test_contexts_for_equals_context_for(setup):
    df, _, eps = setup
    bulk = contexts_for(df, eps)
    assert [c.episode for c in bulk] == eps
    for b, e in zip(bulk, eps):
        c = context_for(df, e)
        assert b.profile == c.profile
        assert b.state._t == c.state._t and b.state._a == c.state._a and b.state._now == c.state._now
        np.testing.assert_array_equal(b.state.aggregates(e.start), c.state.aggregates(e.start))


# ---- realize / feasible ----------------------------------------------------------------------

def test_realize_respects_bounds(setup, tm):
    df, merchants, eps = setup
    for i, ep in enumerate(eps):
        ctx = context_for(df, ep)
        for seed in range(20):
            s, txs = _random_txs(ctx, tm, merchants, 100 * i + seed)
            assert 1 <= len(txs) <= tm.k_max and len(txs) == s.k
            for tx in txs:
                assert tm.amount_min <= tx.amt <= tm.amount_max
                assert round(tx.amt * 100) == pytest.approx(tx.amt * 100, abs=1e-6)
                assert ep.start <= tx.ts < ep.start + pd.Timedelta(days=tm.window_days)
                assert tx.ts == tx.ts.floor("s")
                j = merchants.names.index(tx.merchant)
                assert tx.category == merchants.category[j]
                assert abs(tx.merch_lat - ctx.profile.lat) <= tm.merchant_coord_radius
                assert abs(tx.merch_long - ctx.profile.long) <= tm.merchant_coord_radius
            ts = [t.ts for t in txs]
            assert ts == sorted(ts) and len(set(ts)) == len(ts)
            assert all(b - a >= pd.Timedelta(seconds=tm.min_gap_secs) for a, b in zip(ts, ts[1:]))
            assert feasible(txs, ctx, tm, merchants)


def _strategy(tm, day, hour, k=None):
    n = tm.k_max
    k = len(day) if k is None else k
    pad = lambda a, v: np.r_[np.asarray(a, dtype=float), np.full(n - len(a), v)]
    return Strategy(k=k, u=1.0, shares=np.ones(n), day=pad(day, 0.0), hour=pad(hour, 0.0),
                    merchant=np.zeros(n, dtype=int))


def _shifted_ctx(df, ep, tod):
    """Context of ``ep`` with the start moved to ``tod`` hours on its day."""
    ep2 = dataclasses.replace(ep, start=ep.start.normalize() + pd.Timedelta(hours=tod))
    return context_for(df, ep2), ep2.start


def test_realize_wraps_whole_days_keeping_time_of_day(setup, tm):
    df, merchants, eps = setup
    ctx, start = _shifted_ctx(df, eps[0], 10.5)  # start at 10:30
    d0 = start.normalize()
    s = _strategy(tm, day=[0.0, 6.0, 6.99], hour=[9.5, 23.0, 23.5])
    txs = realize(s, ctx, tm, merchants, np.random.default_rng(0))
    end = start + pd.Timedelta(days=tm.window_days)
    assert all(start <= t.ts < end for t in txs)
    raw = [d0 + pd.Timedelta(hours=9.5),                      # before start -> +1 day
           d0 + pd.Timedelta(days=6, hours=23),               # inside: unchanged
           (d0 + pd.Timedelta(days=6.99) + pd.Timedelta(hours=23.5)).round("s")]  # after end -> -1 day
    exp = sorted([raw[0] + pd.Timedelta(days=1), raw[1], raw[2] - pd.Timedelta(days=1)])
    assert [t.ts for t in txs] == exp
    tods = sorted((r - r.normalize()) for r in raw)
    assert sorted(t.ts - t.ts.normalize() for t in txs) == tods
    assert feasible(txs, ctx, tm, merchants)


def test_equal_times_spread_min_gap(setup, tm):
    df, merchants, eps = setup
    ctx, start = _shifted_ctx(df, eps[0], 10.5)
    k = tm.k_max
    txs = realize(_strategy(tm, [3.0] * k, [12.0] * k), ctx, tm, merchants, np.random.default_rng(0))
    ts = [t.ts for t in txs]
    assert len(set(ts)) == k
    assert ts[0] == start.normalize() + pd.Timedelta(days=3, hours=12)
    assert all(b - a == pd.Timedelta(seconds=tm.min_gap_secs) for a, b in zip(ts, ts[1:]))
    assert feasible(txs, ctx, tm, merchants)


def test_overflow_at_window_end_backward_pass(setup, tm):
    df, merchants, eps = setup
    ctx, start = _shifted_ctx(df, eps[0], 10.5)
    k = tm.k_max
    # day 6.999 + 10.4 h = day 7 10:22:34, ~7 min before the window end (day 7 10:30):
    # the forward pass would push the last ops past the end -> backward pass from end - 1 s
    txs = realize(_strategy(tm, [6.999] * k, [10.4] * k), ctx, tm, merchants, np.random.default_rng(0))
    ts = [t.ts for t in txs]
    end = start + pd.Timedelta(days=tm.window_days)
    assert ts[-1] == end - pd.Timedelta(seconds=1)
    assert all(b - a == pd.Timedelta(seconds=tm.min_gap_secs) for a, b in zip(ts, ts[1:]))
    assert all(start <= t < end for t in ts)
    assert feasible(txs, ctx, tm, merchants)


def test_gap_that_cannot_fit_raises(setup, tm):
    df, merchants, eps = setup
    ctx = context_for(df, eps[0])
    tm2 = dataclasses.replace(tm, min_gap_secs=86_400)
    with pytest.raises(ValueError, match="do not fit"):
        realize(_strategy(tm2, [0.0] * tm.k_max, [0.0] * tm.k_max), ctx, tm2, merchants, np.random.default_rng(0))


def test_rng_consumption_independent_of_k(setup, tm):
    df, merchants, eps = setup
    ctx = context_for(df, eps[0])
    s = Strategy.random(tm, len(merchants), np.random.default_rng(0))
    nxt = []
    for k in (1, 7, tm.k_max):
        rng = np.random.default_rng(9)
        realize(dataclasses.replace(s, k=k), ctx, tm, merchants, rng)
        nxt.append(rng.random())
    assert len(set(nxt)) == 1


def test_realize_amounts_follow_shares(setup, tm):
    df, merchants, eps = setup
    ctx = context_for(df, eps[0])
    k = 4
    shares = np.zeros(tm.k_max)
    shares[:k] = [1, 1, 2, 0]
    shares[k:] = 99.0  # ignored: only the first k shares count
    s = Strategy(k=k, u=0.9, shares=shares, day=np.zeros(tm.k_max), hour=np.linspace(0, 23, tm.k_max),
                 merchant=np.arange(tm.k_max) % len(merchants))
    txs = realize(s, ctx, tm, merchants, np.random.default_rng(0))
    a = 0.9 * eps[0].amount
    exp = np.clip(np.round(np.clip(a * np.array([0.25, 0.25, 0.5, 0.0]), tm.amount_min, tm.amount_max), 2),
                  tm.amount_min, tm.amount_max)
    assert sorted(t.amt for t in txs) == sorted(exp.tolist())
    # all-zero shares -> equal shares
    s0 = dataclasses.replace(s, shares=np.zeros(tm.k_max))
    txs0 = realize(s0, ctx, tm, merchants, np.random.default_rng(0))
    assert len({t.amt for t in txs0}) == 1


def test_owner_fields_unchanged(setup, tm):
    df, merchants, eps = setup
    ctx = context_for(df, eps[0])
    before = (ctx.profile, len(ctx.state), list(ctx.state._t))
    _, txs = _random_txs(ctx, tm, merchants, 3)
    assert (ctx.profile, len(ctx.state), list(ctx.state._t)) == before  # realize does not mutate ctx
    assert ctx.state.profile is ctx.profile
    tx_fields = {f.name for f in dataclasses.fields(Tx)}
    assert not tx_fields & set(OWNER_FIELDS)
    assert ctx.profile == profile_of(df, eps[0].card)
    assert feasible(txs, ctx, tm, merchants)


def test_unreachable_target_is_feasible_but_short(setup, tm):
    df, merchants, eps = setup
    ep = dataclasses.replace(eps[0], amount=100_000.0)
    ctx = context_for(df, ep)
    rng = np.random.default_rng(1)
    s = Strategy.random(tm, len(merchants), rng)
    s = dataclasses.replace(s, k=tm.k_max, u=1.0)
    txs = realize(s, ctx, tm, merchants, rng)
    assert feasible(txs, ctx, tm, merchants)
    assert sum(t.amt for t in txs) <= tm.k_max * tm.amount_max < tm.rho * ep.amount


def test_same_seed_same_txs(setup, tm):
    df, merchants, eps = setup
    ctx = context_for(df, eps[0])
    _, a = _random_txs(ctx, tm, merchants, 42)
    _, b = _random_txs(ctx, tm, merchants, 42)
    _, c = _random_txs(ctx, tm, merchants, 43)
    assert a == b
    assert a != c


def test_feasible_rejects_violations(setup, tm):
    df, merchants, eps = setup
    ep = eps[0]
    ctx = context_for(df, ep)
    _, txs = _random_txs(ctx, tm, merchants, 5)
    tx = txs[0]
    bad = [
        dataclasses.replace(tx, amt=0.5),
        dataclasses.replace(tx, amt=tm.amount_max + 0.01),
        dataclasses.replace(tx, amt=10.005),
        dataclasses.replace(tx, ts=ep.start - pd.Timedelta(seconds=1)),
        dataclasses.replace(tx, ts=ep.start + pd.Timedelta(days=tm.window_days)),
        dataclasses.replace(tx, merchant="fraud_Nowhere"),
        dataclasses.replace(tx, category="no_such_category"),
        dataclasses.replace(tx, merch_lat=ctx.profile.lat + 1.5),
        dataclasses.replace(tx, merch_long=ctx.profile.long - 1.5),
    ]
    for b in bad:
        assert not feasible([b] + txs[1:], ctx, tm, merchants), b
    assert not feasible([], ctx, tm, merchants)
    two = realize(_strategy(tm, [1.0, 2.0], [12.0, 12.0]), ctx, tm, merchants, np.random.default_rng(0))
    assert feasible(two, ctx, tm, merchants)
    close = [two[0], dataclasses.replace(two[1], ts=two[0].ts + pd.Timedelta(seconds=30))]
    assert not feasible(close, ctx, tm, merchants)  # 30 s < min_gap_secs
    assert not feasible([tx] * (tm.k_max + 1), ctx, tm, merchants)
    other = dataclasses.replace(ctx, profile=dataclasses.replace(ctx.profile, gender="X"))
    assert not feasible(txs, other, tm, merchants)  # state profile must be the card's profile


# ---- Strategy vector / clip / random ---------------------------------------------------------

def test_vector_roundtrip_and_length(tm):
    rng = np.random.default_rng(0)
    s = Strategy.random(tm, 50, rng)
    v = s.to_vector()
    assert v.dtype == np.float64 and v.shape == (Strategy.vector_length(tm),) == (2 + 4 * tm.k_max,)
    s2 = Strategy.from_vector(v, tm, 50)
    np.testing.assert_array_equal(s2.to_vector(), v)


def test_random_and_clip_in_range(tm):
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = Strategy.random(tm, 50, rng)
        _check_ranges(s, tm, 50)
    v = rng.normal(0, 100, Strategy.vector_length(tm))
    s = Strategy.from_vector(v, tm, 50)
    _check_ranges(s, tm, 50)
    s = Strategy(k=0, u=5.0, shares=-np.ones(tm.k_max), day=np.full(tm.k_max, 7.0),
                 hour=np.full(tm.k_max, 24.0), merchant=np.full(tm.k_max, 50)).clip(tm, 50)
    _check_ranges(s, tm, 50)
    assert s.k == 1 and s.u == 1.0


def _check_ranges(s, tm, n):
    assert isinstance(s.k, int) and 1 <= s.k <= tm.k_max
    assert tm.rho <= s.u <= 1.0
    for a in (s.shares, s.day, s.hour, s.merchant):
        assert a.shape == (tm.k_max,)
    assert (s.shares >= 0).all()
    assert ((s.day >= 0) & (s.day < tm.window_days)).all()
    assert ((s.hour >= 0) & (s.hour < 24)).all()
    assert s.merchant.dtype.kind == "i" and ((s.merchant >= 0) & (s.merchant < n)).all()
