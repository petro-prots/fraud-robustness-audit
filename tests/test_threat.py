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
                             cards_per_eval=5, merchant_coord_radius=1.0)
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


def test_merchant_table_sorted_with_first_category(sparkov_like):
    df = sparkov_like
    mt = MerchantTable.from_df(df)
    assert mt.names == sorted(df["merchant"].unique())
    assert all(n.startswith("fraud_") for n in mt.names)
    first = df.drop_duplicates("merchant").set_index("merchant")["category"]
    assert mt.category == [first[n] for n in mt.names]
    assert len(mt) == len(mt.names) == df["merchant"].nunique()


# ---- context ---------------------------------------------------------------------------------

def test_context_excludes_episode_and_later_ops(setup):
    df, _, eps = setup
    ep = eps[0]
    ctx = context_for(df, ep)
    assert isinstance(ctx, EpisodeContext)
    assert ctx.episode is ep
    assert ctx.profile == profile_of(df, ep.card)
    assert ctx.state.profile == ctx.profile
    g = df[df["cc_num"] == ep.card]
    expected = g[(g["ts"] < ep.start) & (g["ts"] >= ep.start - pd.Timedelta(days=30))]
    assert not set(expected["row"]) & set(ep.rows)
    assert len(ctx.state) == len(expected)
    # same contexts in bulk (one index by card)
    bulk = contexts_for(df, eps)
    assert [c.episode for c in bulk] == eps
    assert len(bulk[0].state) == len(ctx.state) and bulk[0].profile == ctx.profile


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
            assert [t.ts for t in txs] == sorted(t.ts for t in txs)
            assert feasible(txs, ctx, tm, merchants)


def test_realize_extremes_clamped_to_window(setup, tm):
    df, merchants, eps = setup
    ctx = context_for(df, eps[0])
    k = tm.k_max
    s = Strategy(k=k, u=1.0, shares=np.ones(k), day=np.r_[np.zeros(k // 2), np.full(k - k // 2, 6.999999)],
                 hour=np.r_[np.zeros(k // 2), np.full(k - k // 2, 23.99999)], merchant=np.zeros(k, dtype=int))
    txs = realize(s, ctx, tm, merchants, np.random.default_rng(0))
    start = eps[0].start
    assert min(t.ts for t in txs) == start  # before start -> moved to start
    assert max(t.ts for t in txs) < start + pd.Timedelta(days=tm.window_days)
    assert feasible(txs, ctx, tm, merchants)


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
