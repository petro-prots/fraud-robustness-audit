"""Corridor variants (Task 22c, rulings R27–R29): V1 weekly-split daily corridor and Proposition 2 (B(h)), V2
adaptation on owner-confirmed operations, the statistical population floor (R28), the used-C_24h audit."""
import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import make_sparkov_like
from fraudaudit import corridor as cor
from fraudaudit.corridor import CorridorConfig, FloorFit, Multipliers, WeeklySplit
from fraudaudit.episode import _Bank, run_adaptive_episode, run_episode
from fraudaudit.serving.state import CardState, SpendHistory, Tx
from fraudaudit.threat import MerchantTable, Strategy, ThreatModel, realize
from test_corridor import (PROF, T0, StubServing, _burst_times, _card_frame, _fixture_contexts, _greedy_fill,
                           _hist, _random_multipliers, cfg, ctx_with, tx)

ROOT = Path(__file__).resolve().parents[1]
TM_PATH = ROOT / "threat-models" / "realistic.yaml"
DAY = pd.Timedelta(days=1)
DAY_NS = 86_400 * 10**9


def split(shares=(0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2)) -> WeeklySplit:
    return WeeklySplit(shares=tuple(float(x) for x in shares), n_days=(1,) * 7)


def v1(kappa=2.0, **kw) -> CorridorConfig:
    return cfg(daily_mode="weekly_split", kappa=kappa, split=kw.pop("split", split()), **kw)


# ---- weekly split s(j) ------------------------------------------------------------------------------------------

def test_weekly_split_estimated_from_daily_series():
    days = pd.date_range("2019-01-01", periods=7 * 30 + 3, freq="D")  # a partial week at the end
    pattern = np.array([7.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0])  # Monday .. Sunday
    rng = np.random.default_rng(0)
    v = pattern[days.weekday] * 100 * rng.uniform(0.99, 1.01, days.size)
    s = cor.estimate_weekly_split(pd.Series(v, index=days))
    assert sum(s.shares) == pytest.approx(1.0)
    assert s.shares == pytest.approx(tuple(pattern / pattern.sum()), rel=0.01)
    assert sum(s.n_days) == days.size
    # no absolute dates: the same weekday pattern a year later gives the same shares
    days2 = days + pd.Timedelta(days=364)
    s2 = cor.estimate_weekly_split(pd.Series(v, index=days2))
    assert s2.shares == pytest.approx(s.shares)
    # means per weekday, not sums: the partial week does not tilt the shares
    flat = cor.estimate_weekly_split(pd.Series(np.full(days.size, 5.0), index=days))
    assert flat.shares == pytest.approx((1 / 7,) * 7)


def test_weekly_split_from_df_uses_fit_period_legit_spend():
    df = make_sparkov_like(n_cards=20, seed=4)
    s = cor.estimate_weekly_split(cor.daily_population_spend(df, "fit"))
    assert len(s.shares) == 7 and sum(s.shares) == pytest.approx(1.0) and all(x > 0 for x in s.shares)


# ---- V1 formula ----------------------------------------------------------------------------------------------

def test_v1_daily_corridor_is_kappa_share_of_weekly():
    st = CardState(PROF)
    st.spend = _hist([(T0 - 20 * DAY, 400.0), (T0 - 19 * DAY, 300.0), (T0 - 12 * DAY, 2.0)])
    m = Multipliers(holiday={}, weekday={"0": 3.0}, day_of_month={"3": 5.0}, estimates=(), holiday_weeks=(-3, 1))
    sp = split()
    c = v1(kappa=1.5, q=1.0, floor_7d=1.0, multipliers=("weekday", "day_of_month"), fitted=m, split=sp)
    for k in range(7):
        d = (T0 + k * DAY).normalize()
        c7 = cor.limit_cents(c, st.spend, d.value, "7d")
        assert c7 == 70_000  # the 7-day corridor is unchanged (weekday/day-of-month factors never touch it)
        assert cor.limit_cents(c, st.spend, d.value, "24h") == math.floor(1.5 * sp.shares[d.weekday()] * c7 + 1e-6)
    # the default (quantile) mode keeps the independent 24 h quantile with its multipliers
    q = cfg(q=1.0, floor_24h=1.0, floor_7d=1.0, multipliers=("weekday", "day_of_month"), fitted=m)
    # T0 is Monday 2020-02-03: max 24 h sum 400 $ × weekday 0 (×3) × day of month 3 (×5)
    assert cor.limit_cents(q, st.spend, T0.normalize().value, "24h") == 400 * 3 * 5 * 100


def test_v1_config_validation():
    with pytest.raises(ValueError, match="kappa"):
        cfg(daily_mode="weekly_split", kappa=0.5)
    with pytest.raises(ValueError, match="kappa"):
        cfg(daily_mode="weekly_split", kappa=None)
    with pytest.raises(ValueError, match="daily"):
        cfg(daily_mode="nope")
    with pytest.raises(ValueError, match="split"):
        run_episode(StubServing(), ctx_with(CardState(PROF)), [tx(T0, 1)], 1, 0.8,
                    corridor=cfg(daily_mode="weekly_split", kappa=2.0))


# ---- YAML -----------------------------------------------------------------------------------------------------

def _corridor_yaml(tmp_path, **changes):
    raw = yaml.safe_load((ROOT / "threat-models" / "realistic.yaml").read_text())
    raw["corridor"] = {"q": 0.99, "lookback_days": 180, "floor": {"mode": "population_percentile", "percentile": 10},
                       "multipliers": ["holiday"], "holiday_weeks": [-3, 1], "n_boot": 100, "confidence": 0.95,
                       "confirm_fail_counts_as_decline": True}
    for k, v in changes.items():
        if v is None:
            raw["corridor"].pop(k, None)
        else:
            raw["corridor"][k] = v
    p = tmp_path / "tm.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_yaml_variants_load(tmp_path):
    c = ThreatModel.from_yaml(_corridor_yaml(tmp_path)).corridor
    assert (c.floor_mode, c.floor_percentile, c.daily_mode, c.kappa, c.confirmed_extends) == \
        ("population_percentile", 10.0, "quantile", None, False)
    c = ThreatModel.from_yaml(_corridor_yaml(tmp_path, daily={"mode": "weekly_split", "kappa": 2},
                                             confirmed_extends=True)).corridor
    assert (c.daily_mode, c.kappa, c.confirmed_extends) == ("weekly_split", 2.0, True)
    c = ThreatModel.from_yaml(_corridor_yaml(tmp_path, floor=None, floor_24h=100.0, floor_7d=300.0)).corridor
    assert (c.floor_mode, c.floor_24h, c.floor_7d) == ("fixed", 100.0, 300.0)  # legacy fixed floors
    c = ThreatModel.from_yaml(_corridor_yaml(tmp_path, floor={"mode": "fixed"}, floor_24h=100.0,
                                             floor_7d=300.0)).corridor
    assert c.floor_mode == "fixed"
    pub = ThreatModel.from_yaml(_corridor_yaml(tmp_path, daily={"mode": "weekly_split", "kappa": 2},
                                               confirmed_extends=True)).corridor.public()
    assert pub["floor"] == {"mode": "population_percentile", "percentile": 10.0}
    assert pub["daily"] == {"mode": "weekly_split", "kappa": 2.0} and pub["confirmed_extends"] is True


@pytest.mark.parametrize("change", [
    {"floor": {"mode": "population_percentile"}},  # percentile missing
    {"floor": {"mode": "population_percentile", "percentile": 120}},
    {"floor": {"mode": "weird"}},
    {"floor_24h": 100.0},  # fixed floors with the population mode
    {"floor": None},  # neither floor mode nor fixed floors
    {"floor": {"mode": "fixed"}},  # fixed mode without floor values
    {"daily": {"mode": "weekly_split"}},
    {"daily": {"mode": "weekly_split", "kappa": 0.9}},
    {"daily": {"mode": "quantile", "kappa": 2}},
    {"daily": "weekly_split"},
    {"confirmed_extends": "yes"},
])
def test_yaml_variants_strict(tmp_path, change):
    with pytest.raises(ValueError):
        ThreatModel.from_yaml(_corridor_yaml(tmp_path, **change))


# ---- population floor (R28) -------------------------------------------------------------------------------------

def _floor_df():
    """Fit period ends at VAL_START (2020-03-21). Cards 1..10 established (history from 2019-06), daily spend c·10 $;
    card 99 new (first op 30 days before the fit end) with a tiny spend; one fraud op (excluded)."""
    rows = []
    fit_end = pd.Timestamp("2020-03-21")
    for c in range(1, 11):
        for d in pd.date_range("2019-06-01", fit_end - DAY, freq="D"):
            rows.append((c, d + pd.Timedelta(hours=12), 10.0 * c, 0))
    for d in pd.date_range(fit_end - 30 * DAY, fit_end - DAY, freq="D"):
        rows.append((99, d + pd.Timedelta(hours=12), 0.5, 0))
    rows.append((1, fit_end - 3 * DAY + pd.Timedelta(hours=13), 9_999.0, 1))
    rows.append((1, fit_end + 5 * DAY, 9_999.0, 0))  # val period: ignored
    df = pd.DataFrame(rows, columns=["cc_num", "ts", "amt", "is_fraud"])
    df["part"] = "train"
    return df


def test_population_floor_estimate_by_hand():
    f = cor.estimate_floor(_floor_df(), q=0.99, lookback_days=180, percentile=10)
    # established cards 1..10: 24 h sums = 10c, 7-day sums = 70c (quantile of constant values); the new card is out
    assert f.n_cards == 10
    assert f.floor_24h == pytest.approx(np.percentile(10.0 * np.arange(1, 11), 10))
    assert f.floor_7d == pytest.approx(np.percentile(70.0 * np.arange(1, 11), 10))
    assert (f.q, f.lookback_days, f.percentile) == (0.99, 180, 10.0)
    with pytest.raises(ValueError, match="established"):
        cor.estimate_floor(_floor_df(), q=0.99, lookback_days=400, percentile=10)


def test_population_floor_applies_to_new_cards_only():
    ff = FloorFit(q=0.99, lookback_days=180, percentile=10.0, floor_24h=150.0, floor_7d=500.0, n_cards=5,
                  fit_end="2020-03-21")
    c = cfg(q=1.0, floor_mode="population_percentile", floor_percentile=10.0, floor_fit=ff, floor_24h=0.0,
            floor_7d=0.0)
    old = _hist([(T0 - 200 * DAY, 1.0), (T0 - 20 * DAY, 30.0)])  # established, tiny own corridor
    new = _hist([(T0 - 100 * DAY, 1.0), (T0 - 20 * DAY, 30.0)])  # 100 days of history < L = 180
    d = T0.normalize().value
    assert cor.limit_cents(c, old, d, "7d") == 3_000 and cor.limit_cents(c, old, d, "24h") == 3_000
    assert cor.limit_cents(c, new, d, "7d") == 50_000 and cor.limit_cents(c, new, d, "24h") == 15_000
    assert not cor.floor_active(c, old, d, "7d") and cor.floor_active(c, new, d, "7d")
    assert cor.limit_cents(c, SpendHistory.empty(), d, "7d") == 50_000  # no history at all: a new card
    # exactly L days of history counts as established (first op <= d - L)
    edge = _hist([(T0.normalize() - 180 * DAY, 1.0)])
    assert cor.limit_cents(c, edge, d, "7d") == 100
    # fixed mode (backward compatibility): the floor applies to every card
    fx = cfg(q=1.0, floor_24h=150.0, floor_7d=500.0)
    assert cor.limit_cents(fx, old, d, "7d") == 50_000 and cor.floor_active(fx, old, d, "7d")


def test_population_floor_must_match_q():
    ff = FloorFit(q=0.99, lookback_days=180, percentile=10.0, floor_24h=1.0, floor_7d=1.0, n_cards=5,
                  fit_end="2020-03-21")
    with pytest.raises(ValueError, match="floor"):
        cfg(q=0.95, floor_mode="population_percentile", floor_percentile=10.0, floor_fit=ff).check_ready()
    with pytest.raises(ValueError, match="floor"):
        cfg(q=0.99, floor_mode="population_percentile", floor_percentile=10.0).check_ready()


def test_prepare_fits_floor_split_and_refits_floor_for_another_q():
    df = make_sparkov_like(n_cards=30, days=600, start=pd.Timestamp("2019-01-01"), seed=2)
    c = cfg(q=0.99, floor_mode="population_percentile", floor_percentile=10.0, floor_24h=0.0, floor_7d=0.0,
            lookback_days=60, daily_mode="weekly_split", kappa=2.0, multipliers=("weekday",))
    tm = dataclasses.replace(ThreatModel.from_yaml(TM_PATH), corridor=c)
    p = cor.prepare(tm, df).corridor
    p.check_ready()
    assert p.floor_fit.q == 0.99 and p.split is not None and p.fitted is not None
    p2 = cor.prepare(dataclasses.replace(tm, corridor=dataclasses.replace(p, q=0.9)), df).corridor
    assert p2.floor_fit.q == 0.9 and p2.floor_fit.floor_7d <= p.floor_fit.floor_7d
    assert p2.fitted is p.fitted and p2.split is p.split  # the q-free fits are kept


@pytest.mark.skipif(not (ROOT / "data" / "fraudTrain.csv").exists(), reason="needs the Sparkov data in data/")
def test_population_floor_reproduces_thesis_reference_numbers():
    """R28 reference (thesis research/06 §5, q 0.99): P10 ≈ 1 419 $ (7 d) and ≈ 600 $ (24 h) over 908 cards."""
    from fraudaudit.data import load_raw
    f = cor.estimate_floor(load_raw(ROOT / "data"), q=0.99, lookback_days=180, percentile=10)
    assert f.n_cards == 908
    assert f.floor_7d == pytest.approx(1419, abs=5) and f.floor_24h == pytest.approx(600, abs=5)


# ---- the profile lag boundary (Task 22b review) ----------------------------------------------------------------

def test_profile_op_exactly_at_day_minus_lag_is_excluded():
    d = T0.normalize()
    h = _hist([(d - 10 * DAY, 10.0), (d - 7 * DAY, 900.0)])  # exactly d − 7 d: outside [d − L, d − Λ)
    assert h.window_quantile(d.value, 180 * DAY_NS, DAY_NS, 1.0, cor.PROFILE_LAG_NS) == pytest.approx(10.0)
    assert cor.limit_cents(cfg(q=1.0, floor_24h=0.0, floor_7d=0.0), h, d.value, "24h") == 1_000
    later = (d + pd.Timedelta(seconds=1)).value  # the same calendar day: still excluded
    assert cor.limit_cents(cfg(q=1.0, floor_24h=0.0, floor_7d=0.0), h, later, "24h") == 1_000
    nxt = (d + DAY).value  # the next day: d + 1 − 7 d < the op time? no: op at d − 7 d < (d + 1) − 7 d → included
    assert cor.limit_cents(cfg(q=1.0, floor_24h=0.0, floor_7d=0.0), h, nxt, "24h") == 90_000


# ---- used C_24h audit -------------------------------------------------------------------------------------------

def test_check_used_corridor_checks_c24_too():
    b = {"c7_by_day": {1: 500, 2: 700}, "c24_by_day": {1: 100, 2: 200}, "bound_cents": 700}
    cor.check_used_corridor(b, ((1, 500), (2, 700)), used24=((1, 100), (2, 200)))
    with pytest.raises(RuntimeError, match="24 h"):
        cor.check_used_corridor(b, ((1, 500),), used24=((1, 101),))
    with pytest.raises(RuntimeError, match="outside"):
        cor.check_used_corridor(b, (), used24=((3, 1),))
    cor.check_used_corridor(b, (), used24=((3, 1),), inside_only=True)


def test_bank_records_used_c24_and_auto_ops():
    st = CardState(PROF)
    st.spend = _hist([(T0 - 20 * DAY, 400.0)])
    c = v1(kappa=2.0, q=1.0, floor_7d=1.0)
    bank = _Bank(StubServing(), ctx_with(st), None, 0.8, None, c)
    bank.submit(tx(T0, 30.0))
    bank.submit(tx(T0 + DAY, 20.0))
    o = bank.outcome()
    b = cor.episode_bound(c, st, T0, 7)
    assert dict(o.corridor_c24) == {d: b["c24_by_day"][d] for d, _ in o.corridor_c24}
    assert len(o.corridor_c24) == 2
    assert o.auto_ops == ((T0.value, 3_000), ((T0 + DAY).value, 2_000))
    cor.check_used_corridor(b, o.corridor_c7, used24=o.corridor_c24)


# ---- Proposition 2: B(h) -----------------------------------------------------------------------------------------

def test_episode_bound_blocks_by_hand():
    st = CardState(PROF)
    st.spend = _hist([(T0 - 20 * DAY, 700.0)])  # C_7d = 700 $ every day of the window
    sp = split((0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2))
    c = v1(kappa=1.0, q=1.0, floor_7d=1.0, split=sp)
    b = cor.episode_bound(c, st, T0, 7)  # T0 Monday 10:00: block i spans weekdays i−1 and i (mod 7)
    c24 = [math.floor(sp.shares[j] * 70_000 + 1e-6) for j in range(7)]
    blocks = [max(c24[i % 7], c24[(i + 1) % 7]) for i in range(7)]
    assert b["block_c24_cents"] == blocks
    assert b["b_h_cents"] == [min(70_000, sum(blocks[:h])) for h in range(1, 8)]
    assert b["b_h_cents"][-1] == 70_000  # the weekly bound caps the sum
    b3 = cor.episode_bound(c, st, T0, 3)
    assert len(b3["b_h_cents"]) == 3 and b3["b_h_cents"] == b["b_h_cents"][:3]
    # midnight start: each block is one calendar day
    bm = cor.episode_bound(c, st, T0.normalize(), 7)
    assert bm["block_c24_cents"] == c24


def test_auto_by_block_cumulative():
    t0 = T0.value
    ops = ((t0, 100), (t0 + DAY_NS - 1, 50), (t0 + DAY_NS, 7), (t0 + 3 * DAY_NS + 5, 1))
    assert cor.auto_by_block(ops, t0, 4) == [150, 157, 157, 158]
    with pytest.raises(ValueError):
        cor.auto_by_block(((t0 - 1, 1),), t0, 4)


def _random_split(rng):
    s = rng.uniform(0.3, 3, 7)
    return WeeklySplit(shares=tuple(float(x) for x in s / s.sum()), n_days=(1,) * 7)


def _block_check(o, b, t0, window_days):
    cum = cor.auto_by_block(o.auto_ops, pd.Timestamp(t0).value, window_days)
    return [g - lim for g, lim in zip(cum, b["b_h_cents"])], cum


@pytest.mark.parametrize("window_days", [7, 3])
def test_property_proposition2_random_plans(window_days):
    """Executable Proposition 2: for any plan the auto-approved attacker amount in the first h blocks is <= B(h)
    (random operation lists, random strategies and closed-loop policies; random κ, s, q, floors, multipliers)."""
    _, ctxs, merchants = _fixture_contexts()
    tm = dataclasses.replace(ThreatModel.from_yaml(TM_PATH), window_days=window_days)
    rng = np.random.default_rng(2027)
    violations, positive, binding = 0, 0, 0
    for trial in range(450):
        ctx = ctxs[trial % len(ctxs)]
        c = v1(kappa=float(rng.choice([1.0, 1.5, 2.0, 3.0])), q=float(rng.choice([0.5, 0.9, 0.99, 1.0])),
               floor_7d=float(rng.uniform(1, 800)), lookback_days=int(rng.integers(7, 200)),
               multipliers=("holiday", "weekday", "day_of_month"), fitted=_random_multipliers(rng),
               split=_random_split(rng), confirm_fail_counts_as_decline=bool(rng.integers(0, 2)))
        md = [None, 1, 3][int(rng.integers(0, 3))]
        start = pd.Timestamp(ctx.episode.start)
        p = rng.uniform(0, 1, 64) * (rng.uniform() < 0.5)
        kind = trial % 3
        if kind == 0:
            k = int(rng.integers(1, 41))
            offs = np.sort(rng.integers(0, window_days * 86_400, k))
            if k > 2:
                offs[1] = offs[0]
            txs = [Tx(start + pd.Timedelta(seconds=int(o)), float(np.round(rng.uniform(0.01, 2000), 2)),
                      merchants.names[0], merchants.category[0], ctx.profile.lat, ctx.profile.long) for o in offs]
            o = run_episode(StubServing(p), ctx, txs, md, tm.rho, corridor=c)
        elif kind == 1:
            s = Strategy.random(tm, len(merchants), rng)
            o = run_episode(StubServing(p), ctx, realize(s, ctx, tm, merchants, rng), md, tm.rho, corridor=c)
        else:
            from fraudaudit.policy import StepwisePolicy
            pol = StepwisePolicy.random(tm, merchants.n_categories, rng)
            t2 = dataclasses.replace(tm, max_declines=md if md is not None else 1, corridor=c)
            o, _ = run_adaptive_episode(StubServing(p), ctx, pol, t2, merchants, rng, tau_after_decline=None)
        b = cor.episode_bound(c, ctx.state, start, window_days)
        cor.check_used_corridor(b, o.corridor_c7, used24=o.corridor_c24)
        diff, cum = _block_check(o, b, start, window_days)
        violations += max(diff) > 0
        assert cum[-1] == round(o.passed_amount * 100)
        assert all(x <= y for x, y in zip(b["b_h_cents"], b["b_h_cents"][1:])) and b["b_h_cents"][-1] <= b["bound_cents"]
        positive += cum[-1] > 0
        binding += o.n_confirms > 0
    assert violations == 0
    assert positive > 100 and binding > 100


@pytest.mark.parametrize("window_days", [7, 3])
def test_property_proposition2_greedy_filler(window_days):
    """Proposition 2 against the greedy same-second filler (always-approving model): bursts at t0, around every
    midnight, at every block boundary and at t0 + D − 1 s. 0 violations of any B(h); B(h) attained exactly for some
    h > 0 on some trials (the bound is not vacuous)."""
    _, ctxs, _ = _fixture_contexts()
    rng = np.random.default_rng(78)
    violations, attained, positive = 0, 0, 0
    for trial in range(120):
        ctx = ctxs[trial % len(ctxs)]
        mult = ("holiday", "weekday", "day_of_month") if rng.uniform() < 0.7 else ()
        c = v1(kappa=float(rng.choice([1.0, 1.5, 2.0, 3.0])), q=float(rng.choice([0.5, 0.9, 0.99, 1.0])),
               floor_7d=float(rng.uniform(1, 800)), lookback_days=int(rng.integers(7, 200)), multipliers=mult,
               fitted=_random_multipliers(rng) if mult else None, split=_random_split(rng),
               confirm_fail_counts_as_decline=bool(rng.integers(0, 2)))
        t0 = pd.Timestamp(ctx.episode.start)
        b = cor.episode_bound(c, ctx.state, t0, window_days)
        bank = _Bank(StubServing(), ctx_with(ctx.state, start=t0), None, 0.8, None, c)
        times = set(_burst_times(t0, window_days, rng, trial % 2 == 0))
        times |= {t0 + i * DAY + off for i in range(window_days) for off in (pd.Timedelta(0),
                                                                              DAY - pd.Timedelta(seconds=1))}
        for t in sorted(times):
            _greedy_fill(bank, t)
        o = bank.outcome()
        cor.check_used_corridor(b, o.corridor_c7, used24=o.corridor_c24)
        diff, cum = _block_check(o, b, t0, window_days)
        violations += max(diff) > 0
        attained += any(d == 0 and g > 0 for d, g in zip(diff, cum))
        positive += cum[-1] > 0
    assert violations == 0
    assert attained >= 10 and positive > 80  # non-vacuity floors (diagnostic run: D=7 and D=3 >= 94)


# ---- V2: adaptation on owner-confirmed operations ----------------------------------------------------------------

def test_v2_owner_confirmation_extends_the_corridor():
    c = cfg(confirmed_extends=True)  # floors 100 $ / 300 $
    bank = _Bank(StubServing(), ctx_with(CardState(PROF)), None, 0.8, None, c, owner=True)
    assert bank.submit(tx(T0, 80.0)) is False  # auto
    assert bank.submit(tx(T0 + pd.Timedelta(minutes=1), 150.0)) is False  # 230 > 100: confirmed by the owner
    assert bank.n_confirms == 1
    assert bank.state.confirmed_sum_cents((T0 + pd.Timedelta(minutes=1)).value, DAY_NS) == 15_000
    # C'_24h = 100 + 150 = 250 $ > 80 + 60 → auto (without V2 it would need a confirmation)
    assert bank.submit(tx(T0 + pd.Timedelta(minutes=2), 60.0)) is False and bank.n_confirms == 1
    base = _Bank(StubServing(), ctx_with(CardState(PROF)), None, 0.8, None, cfg(), owner=True)
    for i, a in enumerate([80.0, 150.0, 60.0]):
        base.submit(tx(T0 + pd.Timedelta(minutes=i), a))
    assert base.n_confirms == 2
    # the extension leaves the 24 h window after 24 h
    assert bank.state.confirmed_sum_cents((T0 + DAY + pd.Timedelta(minutes=1)).value, DAY_NS) == 0
    assert bank.state.confirmed_sum_cents((T0 + DAY + pd.Timedelta(minutes=1)).value, 7 * DAY_NS) == 15_000


def test_v2_never_extends_on_attacker_operations():
    """Attacker operations (auto-approved, model-declined, failed confirmations) never extend the corridor."""
    c = cfg(confirmed_extends=True, confirm_fail_counts_as_decline=False)
    bank = _Bank(StubServing([0, 0.9, 0, 0, 0]), ctx_with(CardState(PROF)), None, 0.8, None, c)
    out = [bank.submit(tx(T0 + pd.Timedelta(minutes=i), a)) for i, a in enumerate([60.0, 5.0, 500.0, 40.0, 1.0])]
    assert out == [False, True, True, False, True]  # auto, model decline, failed confirm, auto (100 = C), confirm
    st = bank.state
    t = (T0 + pd.Timedelta(minutes=4)).value
    assert st.confirmed_sum_cents(t, DAY_NS) == 0 and st.confirmed_sum_cents(t, 7 * DAY_NS) == 0
    assert cor.effective_limits_cents(c, st, t) == cor.limits_cents(c, st, t)
    assert bank.outcome().confirmed_ops == ()
    with pytest.raises(ValueError):
        st.push(tx(T0 + DAY, 5.0), auto_approved=True, passed=True, confirmed=True)  # confirmed ops are not auto
    with pytest.raises(ValueError):
        st.push(tx(T0 + DAY, 5.0), auto_approved=False, passed=False, confirmed=True)  # and they passed


def test_v2_posthoc_bound_by_hand():
    c7 = {(T0.normalize() + k * DAY).value: 30_000 for k in range(8)}
    c24 = {d: 10_000 for d in c7}
    b = {"c7_by_day": c7, "c24_by_day": c24, "bound_cents": 30_000, "headroom_cents": 25_000,
         "block_c24_cents": [10_000] * 7, "b_h_cents": [min(30_000, 10_000 * h) for h in range(1, 8)]}
    conf = ((T0 - 6 * DAY).value, 4_000), ((T0 + 2 * DAY).value, 5_000), ((T0 + 2 * DAY).value + 1, 1_000)
    p = cor.posthoc_bound(b, conf, T0, 7)
    # C'_7d max: at T0 + 2 d + 1 ns the 7-day window holds 5 000 + 1 000 (the T0 − 6 d op left at T0 + 1 d)
    assert p["bound_cents"] == 30_000 + 6_000
    # blocks: block 1 has the pre-episode op? no (it is > 24 h before T0); block 3 holds 6 000
    assert p["block_c24_cents"] == [10_000, 10_000, 16_000, 11_000, 10_000, 10_000, 10_000]  # 5 000 left at T0 + 3 d
    assert p["b_h_cents"] == [min(36_000, x) for x in np.cumsum(p["block_c24_cents"]).tolist()]
    assert p["headroom_cents"] == 25_000 + 4_000 + 5_000 + 1_000
    # without confirmations the post-hoc bound is the pre-episode bound
    q = cor.posthoc_bound(b, (), T0, 7)
    assert q["bound_cents"] == 30_000 and q["b_h_cents"] == b["b_h_cents"] and q["headroom_cents"] == 25_000


@pytest.mark.parametrize("variant", ["v2", "v1+v2"])
def test_property_v2_bound_with_interleaved_owner(variant):
    """V2 bound: with the owner's operations interleaved in the episode (the owner passes confirmations, which extend
    the corridor; the attacker fails them), the attacker's auto-approved amount is <= the post-hoc max_t C'_7d and,
    per block, <= the post-hoc B'(h); the C the bank used is the pre-episode base C."""
    _, ctxs, _ = _fixture_contexts()
    rng = np.random.default_rng(99)
    violations, extended, positive = 0, 0, 0
    for trial in range(200):
        ctx = ctxs[trial % len(ctxs)]
        kw = dict(q=float(rng.choice([0.5, 0.9, 0.99])), floor_7d=float(rng.uniform(1, 600)),
                  lookback_days=int(rng.integers(7, 200)), confirmed_extends=True,
                  confirm_fail_counts_as_decline=bool(rng.integers(0, 2)))
        c = (v1(kappa=float(rng.choice([1.0, 2.0, 3.0])), split=_random_split(rng), **kw) if variant == "v1+v2"
             else cfg(floor_24h=float(rng.uniform(1, 200)), **kw))
        t0 = pd.Timestamp(ctx.episode.start)
        b = cor.episode_bound(c, ctx.state, t0, 7)
        bank = _Bank(StubServing(), ctx_with(ctx.state, start=t0), None, 0.8, None, c)
        offs = np.sort(rng.integers(0, 7 * 86_400, 80))
        attacker_auto = []
        for o in offs:
            owner = rng.uniform() < 0.4
            a = float(np.round(rng.uniform(1, 600 if owner else 300), 2))
            t = t0 + pd.Timedelta(seconds=int(o))
            n_auto = len(bank.auto_ops)
            bank.submit(tx(t, a), owner=owner)
            if not owner and len(bank.auto_ops) > n_auto:
                attacker_auto.append(bank.auto_ops[-1])
        o = bank.outcome()
        cor.check_used_corridor(b, o.corridor_c7, used24=o.corridor_c24)
        p = cor.posthoc_bound(b, ctx.state.confirmed_ops() + o.confirmed_ops, t0, 7)
        cum = cor.auto_by_block(tuple(attacker_auto), t0.value, 7)
        violations += cum[-1] > p["bound_cents"] or any(g > lim for g, lim in zip(cum, p["b_h_cents"]))
        violations += cum[-1] > p["headroom_cents"]
        extended += p["bound_cents"] > b["bound_cents"]
        positive += cum[-1] > 0
    assert violations == 0
    assert extended > 50 and positive > 100


# ---- defaults unchanged, batch / serving parity ----------------------------------------------------------------------

def test_defaults_unchanged_identical_outcomes():
    """Without the new keys the corridor behaves exactly as before: explicit quantile mode, no V2, fixed floors give
    identical outcomes; V2 without any owner confirmation equals the base corridor."""
    _, ctxs, merchants = _fixture_contexts()
    rng = np.random.default_rng(5)
    for trial in range(60):
        ctx = ctxs[trial % len(ctxs)]
        start = pd.Timestamp(ctx.episode.start)
        offs = np.sort(rng.integers(0, 7 * 86_400, 30))
        txs = [Tx(start + pd.Timedelta(seconds=int(o)), float(np.round(rng.uniform(1, 900), 2)), merchants.names[0],
                  merchants.category[0], ctx.profile.lat, ctx.profile.long) for o in offs]
        p = rng.uniform(0, 1, 64) * (rng.uniform() < 0.5)
        kw = dict(q=0.9, floor_24h=float(rng.uniform(1, 200)), floor_7d=float(rng.uniform(1, 600)))
        a = run_episode(StubServing(p), ctx, txs, 3, 0.8, corridor=cfg(**kw))
        b = run_episode(StubServing(p), ctx, txs, 3, 0.8,
                        corridor=cfg(daily_mode="quantile", confirmed_extends=False, floor_mode="fixed", **kw))
        v2 = run_episode(StubServing(p), ctx, txs, 3, 0.8, corridor=cfg(confirmed_extends=True, **kw))
        assert a == b == v2


def test_parity_batch_vs_serving_variants():
    df = make_sparkov_like(n_cards=12, seed=9)
    rng = np.random.default_rng(1)
    ff = FloorFit(q=0.9, lookback_days=40, percentile=10.0, floor_24h=150.0, floor_7d=400.0, n_cards=1,
                  fit_end="2020-03-21")
    configs = [v1(kappa=2.0, q=0.9, lookback_days=40, floor_7d=60.0, confirmed_extends=True,
                  multipliers=("holiday", "weekday", "day_of_month"), fitted=_random_multipliers(rng),
                  split=_random_split(rng)),
               cfg(q=0.9, lookback_days=40, floor_mode="population_percentile", floor_percentile=10.0, floor_fit=ff,
                   floor_24h=0.0, floor_7d=0.0, confirmed_extends=True),
               v1(kappa=1.5, q=0.9, lookback_days=40, floor_mode="population_percentile", floor_percentile=10.0,
                  floor_fit=ff, floor_24h=0.0, floor_7d=0.0)]
    n_checked = 0
    for c in configs:
        for card, g in df.groupby("cc_num"):
            g = g.sort_values(["ts", "row"], kind="stable")
            auto = rng.uniform(size=len(g)) < 0.7
            confirmed = ~auto & (rng.uniform(size=len(g)) < 0.5)
            passed = auto | confirmed
            batch = cor.batch_state(g["ts"].to_numpy(dtype="datetime64[ns]").view("int64"), g["amt"].to_numpy(),
                                    auto, passed, c, confirmed=confirmed)
            st = CardState(PROF)
            for i, r in enumerate(g.itertuples(index=False)):
                t = pd.Timestamp(r.ts)
                s24 = st.auto_sum_cents(t.value, DAY_NS)
                s7 = st.auto_sum_cents(t.value, 7 * DAY_NS)
                c24, c7 = cor.effective_limits_cents(c, st, t)
                assert (s24, s7, c24, c7) == (batch["s24"][i], batch["s7"][i], batch["c24"][i], batch["c7"][i])
                st.push(tx(t, r.amt), auto_approved=bool(auto[i]), passed=bool(passed[i]),
                        confirmed=bool(confirmed[i]))
                n_checked += 1
    assert n_checked > 900


# ---- audit integration: per-target blocks, n_jobs identity ---------------------------------------------------------

@pytest.fixture(scope="module")
def audit_env(tmp_path_factory):
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel, train_model
    df = make_sparkov_like(n_cards=80, days=400, start=pd.Timestamp("2019-06-01"), seed=6)
    model, _ = train_model(df, seed=0, epochs=1)
    d = tmp_path_factory.mktemp("model")
    model.save(d, meta={"seed": 0})
    tm = dataclasses.replace(ThreatModel.from_yaml(TM_PATH), recon_budget_ops=200)
    return df, FraudModel.from_artifacts(load_artifacts(d)), d, tm


def _audit(env, corridor, n_jobs=1):
    from fraudaudit.attacks.random_search import RandomSearch
    from fraudaudit.audit import run_audit
    df, model, d, tm = env
    return run_audit(df, model, dataclasses.replace(tm, corridor=corridor), RandomSearch(), seed=7, n_jobs=n_jobs,
                     model_dir=d, target_periods=("fit", "val", "test"), attack_seed=1, one_sided=True,
                     spares_period="test")


def test_audit_v1_v2_population_floor_n_jobs_identity(audit_env):
    c = cfg(q=0.9, lookback_days=60, floor_mode="population_percentile", floor_percentile=10.0, floor_24h=0.0,
            floor_7d=0.0, multipliers=("holiday",), daily_mode="weekly_split", kappa=2.0, confirmed_extends=True)
    a = _audit(audit_env, c, n_jobs=1)
    b = _audit(audit_env, c, n_jobs=2)
    assert a.per_target == b.per_target and a.uplift == b.uplift
    m = a.meta["corridor"]
    assert m["n_bound_violations"] == 0 and m["n_block_violations"] == 0 and m["n_headroom_violations"] == 0
    assert m["params"]["daily"] == {"mode": "weekly_split", "kappa": 2.0} and m["params"]["confirmed_extends"]
    assert m["floor_fit"]["q"] == 0.9 and len(m["weekly_split"]["shares"]) == 7
    assert "n_floor_touched" in m and len(m["b_h"]["median"]) == 7
    for t in a.per_target:
        assert len(t["b_h"]) == 7 and len(t["auto_by_block"]) == 7
        assert all(round(g) <= round(lim * 100) for g, lim in zip(t["auto_by_block"], t["b_h"]))
        assert t["corridor_bound"] == t["corridor_bound_pre"]  # no owner confirmation in the replay
    from fraudaudit.audit import corridor_line
    assert "weekly split" in corridor_line(m)
