"""Soft per-card spending corridor (Task 22): corridor math, multipliers, decision order and CONFIRM semantics,
the guaranteed loss bound (property test of the Proposition), batch/serving parity, corridor null, n_jobs."""
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
from fraudaudit.corridor import CorridorConfig, Multipliers
from fraudaudit.data import fraud_episodes
from fraudaudit.episode import run_adaptive_episode, run_episode
from fraudaudit.serving.core import Decision
from fraudaudit.serving.state import CardProfile, CardState, SpendHistory, Tx
from fraudaudit.threat import MerchantTable, Strategy, ThreatModel, contexts_for, realize

ROOT = Path(__file__).resolve().parents[1]
TM_PATH = ROOT / "threat-models" / "realistic.yaml"
TM_CORRIDOR_PATH = ROOT / "threat-models" / "realistic-corridor.yaml"
DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2020-02-03 10:00:00")  # a Monday
PROF = CardProfile(cc_num=1, gender="F", dob=pd.Timestamp("1980-05-05"), lat=40.0, long=-80.0, city_pop=1000,
                   state="NC")


def cfg(**kw) -> CorridorConfig:
    d = dict(q=0.99, lookback_days=180, floor_24h=100.0, floor_7d=300.0, multipliers=(), holiday_weeks=(-3, 1),
             n_boot=200, confidence=0.95, confirm_fail_counts_as_decline=True)
    d.update(kw)
    return CorridorConfig(**d)


def tx(ts, amt, cat="misc_pos"):
    return Tx(pd.Timestamp(ts), float(amt), "fraud_A", cat, 40.0, -80.0)


class StubServing:
    """p from a list in call order (0 when the list is exhausted); declined iff p >= 0.5."""
    tau = 0.5

    def __init__(self, ps=()):
        self.ps, self.i = list(ps), 0

    def decide(self, state, t):
        p = self.ps[self.i] if self.i < len(self.ps) else 0.0
        self.i += 1
        return Decision(p=p, declined=p >= 0.5)


def ctx_with(state, amount=1000.0, start=T0):
    return SimpleNamespace(episode=SimpleNamespace(amount=amount, start=start), profile=PROF, state=state)


# ---- corridor math on a stub history -----------------------------------------------------------------------

def _hist(rows):
    t = np.array([pd.Timestamp(ts).value for ts, _ in rows], dtype=np.int64)
    a = np.array([a for _, a in rows], dtype=np.float64)
    return SpendHistory(t, a)


def test_window_quantile_by_hand():
    d1 = pd.Timestamp("2020-01-01")
    h = _hist([(d1 + pd.Timedelta(hours=10), 10.0), (d1 + pd.Timedelta(hours=12), 20.0),
               (d1 + DAY + pd.Timedelta(hours=9), 30.0), (d1 + 4 * DAY, 5.0)])
    day = (d1 + 5 * DAY).value
    w24 = 86_400 * 10**9
    # rolling 24 h sums right after every op: 10, 30, 60 (30 + both ops of day 1), 5
    assert h.window_quantile(day, 180 * w24, w24, 0.5, 0) == pytest.approx(np.quantile([10, 30, 60, 5], 0.5))
    assert h.window_quantile(day, 180 * w24, w24, 1.0, 0) == pytest.approx(60.0)
    assert h.window_quantile(day, 180 * w24, 7 * w24, 1.0, 0) == pytest.approx(65.0)
    # lag 0: only ops strictly before the day count; lookback [day - L, day)
    assert h.window_quantile((d1 + DAY).value, 180 * w24, w24, 1.0, 0) == pytest.approx(30.0)
    assert h.window_quantile(day, 2 * w24, w24, 1.0, 0) == pytest.approx(5.0)
    assert h.window_quantile(d1.value, 180 * w24, w24, 0.99, 0) == 0.0  # no history


def test_window_quantile_lag_excludes_recent_ops():
    """R26: with lag W the profile of day d uses only operations in [d - L, d - W)."""
    d1 = pd.Timestamp("2020-01-01")
    w24 = 86_400 * 10**9
    h = _hist([(d1 + pd.Timedelta(hours=10), 10.0), (d1 + 3 * DAY + pd.Timedelta(hours=9), 500.0)])
    day = (d1 + 10 * DAY).value  # d - 7 d = Jan 4 00:00: the 500 op (Jan 4 09:00) is inside the lag, excluded
    assert h.window_quantile(day, 180 * w24, w24, 1.0, 7 * w24) == pytest.approx(10.0)
    assert h.window_quantile(day, 180 * w24, w24, 1.0, 0) == pytest.approx(500.0)
    later = (d1 + 11 * DAY).value  # d - 7 d = Jan 5 00:00 > Jan 4 09:00: now it counts
    assert h.window_quantile(later, 180 * w24, w24, 1.0, 7 * w24) == pytest.approx(500.0)
    # the window sum of an op in the profile still counts earlier ops (all before d - W anyway)
    assert h.window_quantile(later, 180 * w24, 7 * w24, 1.0, 7 * w24) == pytest.approx(510.0)
    # lookback [d - L, d - W): L counted from d
    assert h.window_quantile(later, 9 * w24, w24, 0.0, 7 * w24) == pytest.approx(500.0)


def test_window_value_counts_same_second_ops_and_lookback_edge():
    d1 = pd.Timestamp("2020-01-01 08:00:00")
    h = _hist([(d1, 10.0), (d1, 15.0), (d1 + DAY, 1.0)])  # exactly 24 h later: the first two drop out
    day = (d1.normalize() + 3 * DAY).value
    w24 = 86_400 * 10**9
    assert h.window_quantile(day, 180 * w24, w24, 1.0, 0) == pytest.approx(25.0)
    assert h.window_quantile(day, 180 * w24, w24, 0.0, 0) == pytest.approx(1.0)
    # lookback starts at day - L inclusive
    assert h.window_quantile(day, day - d1.value, w24, 0.0, 0) == pytest.approx(1.0)


def test_spend_history_append_is_persistent():
    h0 = _hist([(T0 - 3 * DAY, 50.0)])
    h1 = h0.append((T0).value, 70.0)
    assert len(h0) == 1 and len(h1) == 2
    day = (T0.normalize() + DAY).value
    w = 86_400 * 10**9
    assert h0.window_quantile(day, 180 * w, w, 1.0, 0) == pytest.approx(50.0)
    assert h1.window_quantile(day, 180 * w, w, 1.0, 0) == pytest.approx(70.0)


def test_spend_history_amortised_append_parity():
    """The amortised (shared-buffer) append and the carried-over quantile cache give exactly what a history built
    from scratch gives, also when several branches append to one parent (persistence)."""
    rng = np.random.default_rng(5)
    w24 = 86_400 * 10**9
    base_t = pd.Timestamp("2020-01-01").value
    t = np.sort(rng.integers(0, 60 * 86_400, 200)) * 10**9 + base_t
    t[5] = t[4]  # a same-second pair
    a = np.round(rng.uniform(1, 500, t.size), 2)
    parent = SpendHistory(t[:50], a[:50])
    for i in range(50, 100):  # grown by appends: the shared buffer has spare slots, so branches must not share them
        parent = parent.append(int(t[i]), float(a[i]))
    assert parent._buf.t.size > 100
    branches = []
    for b in range(4):  # four branches from one parent, interleaved appends and queries
        h, rows = parent, list(zip(t[:100].tolist(), a[:100].tolist()))
        tt = int(t[99])
        for _ in range(60):
            tt += int(rng.integers(0, 3 * 86_400)) * 10**9 * int(rng.uniform() < 0.8)
            amt = float(np.round(rng.uniform(1, 900), 2))
            h = h.append(tt, amt)
            rows.append((tt, amt))
            branches.append((h, list(rows)))  # every intermediate history is checked again at the end
            for lag in (0, 7 * w24):
                for back in (0, 1, 3, 8, 9):
                    day = tt - tt % w24 + w24 - back * w24
                    for wl in (w24, 7 * w24):
                        got = h.window_quantile(day, 180 * w24, wl, 0.9, lag)
                        ref = SpendHistory(np.array([r[0] for r in rows], dtype=np.int64),
                                           np.array([r[1] for r in rows])).window_quantile(day, 180 * w24, wl, 0.9,
                                                                                           lag)
                        assert got == ref
    assert len(parent) == 100 and parent.t.tolist() == t[:100].tolist() and parent.a.tolist() == a[:100].tolist()
    for h, rows in branches:
        assert h.t.tolist() == [r[0] for r in rows] and h.a.tolist() == [r[1] for r in rows]


def test_limit_cents_floor_quantile_multiplier():
    st = CardState(PROF)
    c = cfg(q=1.0)
    assert cor.limit_cents(c, st.spend, T0.value, "24h") == 10_000  # floor
    assert cor.limit_cents(c, st.spend, T0.value, "7d") == 30_000
    st.spend = _hist([(T0 - 9 * DAY, 250.55), (T0 - 2 * DAY, 900.0)])  # the 900 op is inside the 7-day lag
    assert cor.limit_cents(c, st.spend, T0.value, "24h") == 25_055
    m = Multipliers(holiday={}, weekday={"0": 1.5}, day_of_month={}, estimates=(), holiday_weeks=(-3, 1))
    c2 = cfg(q=1.0, multipliers=("weekday",), fitted=m)
    assert cor.limit_cents(c2, st.spend, T0.value, "24h") == math.floor(250.55 * 1.5 * 100 + 1e-6)  # Monday
    assert cor.limit_cents(c2, st.spend, T0.value, "7d") == 30_000  # weekday factors only on the 24 h window
    assert cor.limit_cents(c2, st.spend, (T0 + DAY).value, "24h") == 25_055  # Tuesday: no factor


def test_auto_sum_window_is_half_open_and_counts_auto_only():
    st = CardState(PROF)
    st.push(tx(T0 - 7 * DAY, 1.00))  # exactly 7 d before: out of (t - W, t]
    st.push(tx(T0 - 7 * DAY + pd.Timedelta(seconds=1), 2.00))
    st.push(tx(T0 - pd.Timedelta(hours=1), 4.00), auto_approved=False)
    st.push(tx(T0, 8.00))  # same second as the query: already pushed, counts
    assert st.auto_sum_cents(T0.value, 7 * 86_400 * 10**9) == 1000
    assert st.auto_sum_cents(T0.value, 86_400 * 10**9) == 800
    c = st.copy()
    c.push(tx(T0 + pd.Timedelta(seconds=5), 16.0))
    assert st.auto_sum_cents((T0 + pd.Timedelta(seconds=5)).value, 86_400 * 10**9) == 800  # copy is independent


def test_from_history_profile_is_label_free_and_excludes_future():
    h = pd.DataFrame({"cc_num": 1, "ts": [T0 - 200 * DAY, T0 - 2 * DAY, T0 - DAY, T0],
                      "amt": [5.0, 10.0, 99.0, 7.0], "category": "misc_pos", "is_fraud": [0, 0, 1, 0]})
    st = CardState.from_history(PROF, h, at=T0)
    # R26: every operation of the real history before ``at`` passed and enters the profile, whatever its label
    assert len(st.spend) == 3 and st.spend.a.tolist() == [5.0, 10.0, 99.0]
    h2 = h.assign(is_fraud=[1, 1, 0, 1])
    assert CardState.from_history(PROF, h2, at=T0).spend.a.tolist() == st.spend.a.tolist()
    assert CardState.from_history(PROF, h.drop(columns="is_fraud"), at=T0).spend.a.tolist() == [5.0, 10.0, 99.0]
    assert len(st) == 2  # the model state keeps 30 days, fraud included
    assert st.auto_sum_cents(T0.value - 1, 7 * 86_400 * 10**9) == 10_900  # history counts as auto-approved


# ---- holiday calendar and multipliers --------------------------------------------------------------------

def test_holiday_dates_and_offsets():
    assert cor.holiday_dates(2019) == {"christmas": pd.Timestamp("2019-12-25"),
                                       "independence_day": pd.Timestamp("2019-07-04"),
                                       "thanksgiving": pd.Timestamp("2019-11-28")}
    assert cor.holiday_dates(2020)["thanksgiving"] == pd.Timestamp("2020-11-26")
    lab = lambda s: cor.holiday_label(pd.Timestamp(s).value, -3, 1)  # noqa: E731
    assert lab("2019-12-25") == "christmas:0" and lab("2019-12-24") == "christmas:-1"
    assert lab("2019-12-12") == "christmas:-2" and lab("2020-01-07") == "christmas:1"
    assert lab("2019-12-11") == "thanksgiving:1"  # nearer to Thanksgiving (13 days) than to Christmas (14)
    assert lab("2020-01-08") is None and lab("2019-12-04") == "thanksgiving:0"
    assert lab("2019-07-01") == "independence_day:-1" and lab("2019-09-01") is None


def _seasonal_daily(seed=0, xmas=2.0, saturday=1.5, noise=0.02):
    days = pd.date_range("2019-01-01", "2020-03-20", freq="D")
    rng = np.random.default_rng(seed)
    v = np.full(len(days), 1000.0) * (1 + noise * rng.standard_normal(len(days)))
    lab = [cor.holiday_label(d.value, -3, 1) for d in days]
    v *= np.where([lb in ("christmas:-2", "christmas:-1") for lb in lab], xmas, 1.0)
    v *= np.where(days.weekday == 5, saturday, 1.0)
    return pd.Series(v, index=days)


def test_multipliers_significant_vs_not():
    m = cor.estimate_from_daily(_seasonal_daily(), cfg(multipliers=("holiday", "weekday", "day_of_month")))
    assert m.holiday["christmas:-2"] == pytest.approx(2.0, rel=0.1)
    assert m.holiday["christmas:-1"] == pytest.approx(2.0, rel=0.1)
    assert "independence_day:0" not in m.holiday and "thanksgiving:0" not in m.holiday
    assert m.weekday["5"] == pytest.approx(1.5 / (1 + 0.5 / 7), rel=0.03)
    est = {(e["kind"], e["group"]): e for e in m.estimates}
    e = est[("holiday", "independence_day:0")]
    assert not e["significant"] and e["lower"] <= 1 <= e["upper"]
    assert est[("holiday", "christmas:-1")]["significant"]


def test_multipliers_weak_effect_with_noise_not_significant():
    m = cor.estimate_from_daily(_seasonal_daily(xmas=1.01, saturday=1.0, noise=0.3),
                                cfg(multipliers=("holiday",)))
    assert "christmas:-1" not in m.holiday and "christmas:-2" not in m.holiday
    assert m.weekday == {} and m.day_of_month == {}  # kinds that are not enabled are not estimated


def test_multipliers_deterministic():
    a = cor.estimate_from_daily(_seasonal_daily(), cfg(multipliers=("holiday", "weekday")))
    b = cor.estimate_from_daily(_seasonal_daily(), cfg(multipliers=("holiday", "weekday")))
    assert a == b


def test_estimate_multipliers_from_df_runs_on_fit_period(sparkov_like):
    m = cor.estimate_multipliers(sparkov_like, cfg(multipliers=("holiday", "weekday", "day_of_month")))
    assert m.estimates and all(e["n_days"] > 0 for e in m.estimates)
    kinds = {e["kind"] for e in m.estimates}
    assert {"weekday", "day_of_month"} <= kinds <= {"holiday", "weekday", "day_of_month"}
    assert all(e["significant"] == (e["group"] in getattr(m, e["kind"])) for e in m.estimates)


# ---- config (strict loader) ----------------------------------------------------------------------------------

def test_yaml_corridor_null_and_absent_mean_disabled(tmp_path):
    tm = ThreatModel.from_yaml(TM_PATH)
    assert tm.corridor is None
    raw = yaml.safe_load(TM_PATH.read_text())
    raw.pop("corridor", None)
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert ThreatModel.from_yaml(p) == tm


def test_yaml_corridor_section_loads():
    tm = ThreatModel.from_yaml(TM_CORRIDOR_PATH)
    c = tm.corridor
    assert (c.q, c.lookback_days, c.multipliers, c.holiday_weeks) == (0.99, 180, ("holiday", "weekday",
                                                                                  "day_of_month"), (-3, 1))
    assert c.confirm_fail_counts_as_decline is True and c.fitted is None
    base = ThreatModel.from_yaml(TM_PATH)
    assert dataclasses.replace(tm, corridor=None) == base  # the only difference


@pytest.mark.parametrize("change", [
    {"q": 1.5}, {"q": -0.1}, {"lookback_days": 0}, {"floor_24h": -1}, {"floor_7d": "x"}, {"multipliers": ["easter"]},
    {"holiday_weeks": [1, -3]}, {"n_boot": 0}, {"confidence": 1.0}, {"confirm_fail_counts_as_decline": "yes"},
    {"extra": 1}, {"q": "drop"}, {"multipliers": ["weekday", "weekday"]},
])
def test_yaml_corridor_strict(tmp_path, change):
    raw = yaml.safe_load(TM_CORRIDOR_PATH.read_text())
    for k, v in change.items():
        if v == "drop":
            raw["corridor"].pop(k)
        else:
            raw["corridor"][k] = v
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        ThreatModel.from_yaml(p)


def test_yaml_corridor_rejects_episode_longer_than_window(tmp_path):
    """M3: the guarantee needs D <= W = 7 d, so a corridor threat model with window_days > 7 is rejected at load."""
    raw = yaml.safe_load(TM_CORRIDOR_PATH.read_text())
    raw["window_days"] = 8
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="window_days"):
        ThreatModel.from_yaml(p)
    raw["corridor"] = None  # without a corridor longer episodes stay allowed
    p.write_text(yaml.safe_dump(raw))
    assert ThreatModel.from_yaml(p).window_days == 8


def test_unfitted_multipliers_rejected():
    with pytest.raises(ValueError, match="fitted"):
        run_episode(StubServing(), ctx_with(CardState(PROF)), [tx(T0, 1)], 1, 0.8,
                    corridor=cfg(multipliers=("weekday",)))


# ---- decision order and CONFIRM semantics -------------------------------------------------------------------

def _run(ps, amts, md=1, owner=False, **kw):
    from fraudaudit.episode import _Bank
    bank = _Bank(StubServing(ps), ctx_with(CardState(PROF)), md, 0.8, None, cfg(**kw), owner=owner)
    for i, a in enumerate(amts):
        if bank.blocked:
            break
        bank.submit(tx(T0 + pd.Timedelta(minutes=i), a))
    return bank


def test_model_decline_comes_first():
    b = _run([0.9], [5.0], md=None)
    assert b.n_declines == 1 and b.n_confirms == 0 and b.passed == 0.0
    assert b.state.auto_sum_cents(T0.value, 86_400 * 10**9) == 0  # a declined op is not auto-approved


def test_within_corridor_auto_approved():
    b = _run([0, 0], [60.0, 40.0], md=1)  # 100.00 = floor_24h exactly: allowed (<=)
    assert b.passed == pytest.approx(100.0) and b.n_declines == 0 and b.n_confirms == 0
    assert b.decisions == [False, False]


def test_attacker_confirm_fails_and_counts_as_decline():
    b = _run([0, 0, 0], [60.0, 40.01, 1.0], md=1)
    assert b.passed == pytest.approx(60.0) and b.n_confirms == 1 and b.n_declines == 1 and b.blocked
    assert b.decisions == [False, True]


def test_attacker_confirm_fail_not_counted_when_configured():
    b = _run([0, 0, 0], [60.0, 40.01, 40.0], md=1, confirm_fail_counts_as_decline=False)
    assert b.passed == pytest.approx(100.0) and b.n_confirms == 1 and b.n_declines == 0 and not b.blocked
    assert b.decisions == [False, True, False]


def test_owner_confirm_succeeds_and_is_not_auto():
    b = _run([0, 0, 0], [60.0, 50.0, 40.0], md=1, owner=True)
    assert b.passed == pytest.approx(150.0) and b.n_confirms == 1 and b.n_declines == 0 and not b.blocked
    assert b.decisions == [False, False, False]
    assert b.state.auto_sum_cents((T0 + pd.Timedelta(minutes=2)).value, 86_400 * 10**9) == 10_000
    assert len(b.state.spend) == 3  # passed operations (auto-approved or confirmed) feed the profile


def test_profile_takes_passed_operations_only():
    """R26: the profile gets every operation that passed (auto-approved or confirmed) and nothing else, whoever
    sent it: model declines and failed confirmations never enter it."""
    from fraudaudit.episode import _Bank
    # attacker: auto-approved 60 (enters), model decline (not), failed confirmation 50 (not), auto 30 (enters)
    bank = _Bank(StubServing([0, 0.9, 0, 0]), ctx_with(CardState(PROF)), None, 0.8, None,
                 cfg(confirm_fail_counts_as_decline=False))
    for i, a in enumerate([60.0, 5.0, 50.0, 30.0]):
        bank.submit(tx(T0 + pd.Timedelta(minutes=i), a))
    assert bank.state.spend.a.tolist() == [60.0, 30.0]
    # owner: model decline (not), confirmed 50 (enters: it passed)
    own = _Bank(StubServing([0, 0.9, 0]), ctx_with(CardState(PROF)), None, 0.8, None, cfg(), owner=True)
    for i, a in enumerate([60.0, 5.0, 50.0]):
        own.submit(tx(T0 + pd.Timedelta(minutes=i), a))
    assert own.n_confirms == 1 and own.state.spend.a.tolist() == [60.0, 50.0]


def test_seven_day_window_binds_after_24h_resets():
    amts = [100.0, 100.0, 100.0, 0.01]  # one per day: 24 h fits every time, 7 d is full after three
    from fraudaudit.episode import _Bank
    bank = _Bank(StubServing(), ctx_with(CardState(PROF)), None, 0.8, None, cfg())
    out = [bank.submit(tx(T0 + i * DAY, a)) for i, a in enumerate(amts)]
    assert out == [False, False, False, True] and bank.n_confirms == 1


def test_escalation_follows_counted_confirm_failure():
    from fraudaudit.episode import _Bank
    bank = _Bank(StubServing([0, 0, 0.45]), ctx_with(CardState(PROF)), None, 0.8, 0.4, cfg())
    assert [bank.submit(tx(T0 + pd.Timedelta(minutes=i), a)) for i, a in enumerate([100.0, 1.0, 0.5])] \
        == [False, True, True]
    assert bank.escalated and bank.n_declines == 2 and bank.n_confirms == 1


def test_corridor_null_identical_outcomes():
    for ps, amts in (([0, 1, 0], [10, 20, 30]), ([0, 0, 0, 0], [500, 600, 700, 800])):
        a = run_episode(StubServing(ps), ctx_with(CardState(PROF)), [tx(T0 + pd.Timedelta(minutes=i), x)
                                                                     for i, x in enumerate(amts)], None, 0.8)
        b = run_episode(StubServing(ps), ctx_with(CardState(PROF)), [tx(T0 + pd.Timedelta(minutes=i), x)
                                                                     for i, x in enumerate(amts)], None, 0.8,
                        corridor=None)
        assert a == b and a.n_confirms == 0


# ---- the guaranteed bound -----------------------------------------------------------------------------------

def test_episode_bound_max_over_days_and_headroom():
    st = CardState(PROF)
    st.push(tx(T0 - 2 * DAY, 120.0))  # owner op in the 7-day window at the start; drops out at T0 + 5 d
    m = Multipliers(holiday={}, weekday={"2": 2.0}, day_of_month={}, estimates=(), holiday_weeks=(-3, 1))
    c = cfg(multipliers=("weekday",), fitted=m)
    b = cor.episode_bound(c, st, T0, 7)
    assert b["bound_cents"] == 30_000  # weekday factors do not touch the 7-day corridor
    assert b["headroom_cents"] == 30_000  # the owner op drops out before the episode ends
    b3 = cor.episode_bound(c, st, T0, 3)
    assert b3["headroom_cents"] == 30_000 - 12_000
    with pytest.raises(ValueError, match="D <= W"):
        cor.episode_bound(c, st, T0, 8)
    days = sorted(b["c7_by_day"])
    assert days == [(T0.normalize() + k * DAY).value for k in range(8)]  # [T0, T0 + 7 d) touches 8 calendar days
    assert max(b["c7_by_day"].values()) == b["bound_cents"]


def test_episode_bound_needs_the_pre_episode_state():
    """The bound is computed from the state right before the episode only (no operation at or after t0)."""
    st = CardState(PROF)
    st.push(tx(T0 + pd.Timedelta(seconds=1), 5.0))
    with pytest.raises(ValueError, match="before the episode start"):
        cor.episode_bound(cfg(), st, T0, 7)


def test_check_used_corridor():
    b = {"c7_by_day": {1: 500, 2: 700}, "bound_cents": 700}
    cor.check_used_corridor(b, ((1, 500), (2, 700)))
    cor.check_used_corridor(b, ((1, 500), (3, 900)), inside_only=True)  # days outside the window are skipped
    with pytest.raises(RuntimeError, match="pre-episode"):
        cor.check_used_corridor(b, ((1, 501),))
    with pytest.raises(RuntimeError, match="outside"):
        cor.check_used_corridor(b, ((3, 900),))


def _card_frame(rows):
    return pd.DataFrame({"cc_num": 1, "ts": [pd.Timestamp(t) for t, _, _ in rows], "amt": [a for _, a, _ in rows],
                         "category": "misc_pos", "is_fraud": [f for _, _, f in rows],
                         "row": np.arange(len(rows))})


@pytest.mark.parametrize("window_days", [7, 3])
def test_corridor_label_independent_during_the_episode(window_days):
    """R26: C_W(c, t) for every t in [t0, t0 + D) is the same whether the attacker's operations are labelled fraud,
    legit, or absent from the data, and equals the pre-episode bound's per-day values (batch and serving paths)."""
    rng = np.random.default_rng(11)
    t0 = pd.Timestamp("2020-03-10 13:17:05")
    owner = [(t0 - pd.Timedelta(seconds=int(s)), float(np.round(rng.uniform(1, 400), 2)), 0)
             for s in np.sort(rng.integers(60, 120 * 86_400, 300))[::-1]]
    offs = np.sort(rng.integers(0, window_days * 86_400, 25))
    offs[1] = offs[0]  # same-second attacker ops
    attack = [(t0 + pd.Timedelta(seconds=int(o)), float(np.round(rng.uniform(50, 5000), 2)), 1) for o in offs]
    variants = {"fraud": owner + attack, "legit": owner + [(t, a, 0) for t, a, _ in attack], "absent": owner}
    m = _random_multipliers(np.random.default_rng(3))
    c = cfg(q=0.9, lookback_days=60, floor_24h=20.0, floor_7d=60.0,
            multipliers=("holiday", "weekday", "day_of_month"), fitted=m)
    pre = CardState.from_history(PROF, _card_frame(owner), at=t0)
    b = cor.episode_bound(c, pre, t0, window_days)
    times = [t0 + pd.Timedelta(minutes=37 * k) for k in range(window_days * 24 * 60 // 37)] + [t for t, _, _ in attack]
    times += [t0 + window_days * DAY - pd.Timedelta(seconds=1)]
    for t in sorted(times):
        assert t0 <= t < t0 + window_days * DAY
        lims = {name: cor.limits_cents(c, CardState.from_history(PROF, _card_frame(rows), at=t), t)
                for name, rows in variants.items()}
        assert lims["fraud"] == lims["legit"] == lims["absent"]
        assert lims["absent"][1] == b["c7_by_day"][t.normalize().value]
    # batch path over the whole stream (every operation passed, no labels): same C at the attacker's times
    allrows = sorted(variants["fraud"], key=lambda r: r[0])
    tt = np.array([pd.Timestamp(r[0]).value for r in allrows], dtype=np.int64)
    bs = cor.batch_state(tt, np.array([r[1] for r in allrows]), np.ones(len(allrows), bool),
                         np.ones(len(allrows), bool), c)
    for i, r in enumerate(allrows):
        if r[2] == 1:
            assert bs["c7"][i] == b["c7_by_day"][pd.Timestamp(r[0]).normalize().value]
    # serving path: the bank's C during the episode (attacker's auto-approved ops enter the profile) = the bound's
    from fraudaudit.episode import _Bank
    bank = _Bank(StubServing(), ctx_with(pre, start=t0), None, 0.8, None, c)
    for t, a, _ in attack:
        bank.submit(tx(t, a))
    assert len(bank.state.spend) > len(pre.spend)  # passed attacker ops did enter the (lagged) profile
    o = bank.outcome()
    assert o.corridor_c7 and all(b["c7_by_day"][d] == v for d, v in o.corridor_c7)


def _fixture_contexts():
    df = make_sparkov_like(n_cards=24, seed=5)
    eps = fraud_episodes(df, {"fit", "val", "test"})
    return df, contexts_for(df, eps), MerchantTable.from_df(df)


def _random_multipliers(rng):
    return Multipliers(holiday={f"christmas:{k}": float(rng.uniform(0.5, 2)) for k in range(-3, 2)},
                       weekday={str(d): float(rng.uniform(0.5, 2)) for d in range(7)},
                       day_of_month={str(d): float(rng.uniform(0.5, 2)) for d in range(1, 32)},
                       estimates=(), holiday_weeks=(-3, 1))


@pytest.mark.parametrize("window_days", [7, 3])
def test_property_guarantee_random_adversarial_plans(window_days):
    """Executable form of the Proposition: for any plan the auto-approved attacker amount is
    <= max_{t in [t0, t0 + D)} C_7d(c, t) (and <= the tighter headroom bound); D = 7 and D = 3 < W."""
    df, ctxs, merchants = _fixture_contexts()
    tm = dataclasses.replace(ThreatModel.from_yaml(TM_PATH), window_days=window_days)
    rng = np.random.default_rng(2026)
    violations, positive, confirms = 0, 0, 0
    for trial in range(600):
        ctx = ctxs[trial % len(ctxs)]
        c = cfg(q=float(rng.choice([0.5, 0.9, 0.99, 1.0])), floor_24h=float(rng.uniform(1, 300)),
                floor_7d=float(rng.uniform(1, 800)), lookback_days=int(rng.integers(7, 200)),
                multipliers=("holiday", "weekday", "day_of_month"), fitted=_random_multipliers(rng),
                confirm_fail_counts_as_decline=bool(rng.integers(0, 2)))
        md = [None, 1, 3][int(rng.integers(0, 3))]
        start = pd.Timestamp(ctx.episode.start)
        win_s = tm.window_days * 86_400
        kind = trial % 3
        p = rng.uniform(0, 1, 64) * (rng.uniform() < 0.5)  # half the plans face an always-approving model
        if kind == 0:  # arbitrary operation lists: any amounts, times anywhere in the window, same seconds
            k = int(rng.integers(1, 41))
            offs = np.sort(rng.integers(0, win_s, k))
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
        b = cor.episode_bound(c, ctx.state, start, tm.window_days)
        got = round(o.passed_amount * 100)
        if got > b["bound_cents"] or got > b["headroom_cents"]:
            violations += 1
        # the C the bank actually used is the pre-episode per-day value
        assert all(b["c7_by_day"][d] == v for d, v in o.corridor_c7)
        positive += got > 0
        confirms += o.n_confirms > 0
    assert violations == 0
    assert positive > 100 and confirms > 100  # not vacuous: plans pass and the corridor binds


def _greedy_fill(bank, ts, max_ops: int = 400) -> None:
    """Black-box greedy attacker at one second: amounts of 2^k cents from 2^21 down to 1 cent, each sent again until
    it does not pass (a same-second burst that fills the corridor to the cent). ``max_ops`` stops a burst that never
    hits the corridor (a broken rule) instead of looping forever; a correct rule needs about 2 operations per level."""
    k, n = 2**21, 0
    while k >= 1 and not bank.blocked and n < max_ops:
        while not bank.blocked and n < max_ops:
            n += 1
            if bank.submit(Tx(ts, k / 100, "fraud_A", "misc_pos", 40.0, -80.0)):
                break
        k //= 2


def _burst_times(t0, window_days, rng, full):
    """t0, both sides of every midnight inside the episode (d − 1 s, d) and t0 + D − 1 s; without ``full`` a random
    subset of the inner ones."""
    end = t0 + window_days * DAY
    inner = []
    d = t0.normalize() + DAY
    while d < end:
        inner += [d - pd.Timedelta(seconds=1), d]
        d += DAY
    if not full:
        inner = [t for t in inner if rng.uniform() < 0.5]
    return [t0] + inner + [end - pd.Timedelta(seconds=1)]


@pytest.mark.parametrize("window_days", [7, 3])
def test_property_guarantee_greedy_filler(window_days):
    """The Proposition against the strongest attacker the rule allows: an always-approving model, greedy same-second
    bursts filling the corridor to the cent at t0, around every midnight and at t0 + D − 1 s, and plans of many
    small operations with same-second duplicates. 0 violations; with bursts before every midnight and no 24 h
    limit the greedy attacker gets exactly the headroom, and on some trials exactly the bound; the bank's C during
    the episode is the pre-episode per-day value, and the bound is the max of the C it used."""
    from fraudaudit.episode import _Bank
    _, ctxs, _ = _fixture_contexts()
    rng = np.random.default_rng(77)
    violations, attained, tight, positive = 0, 0, 0, 0
    for trial in range(160):
        ctx = ctxs[trial % len(ctxs)]
        no24 = trial % 2 == 0
        mult = ("holiday", "weekday", "day_of_month") if rng.uniform() < 0.7 else ()
        c = cfg(q=float(rng.choice([0.5, 0.9, 0.99, 1.0])), floor_24h=1e7 if no24 else float(rng.uniform(1, 300)),
                floor_7d=float(rng.uniform(1, 800)), lookback_days=int(rng.integers(7, 200)), multipliers=mult,
                fitted=_random_multipliers(rng) if mult else None,
                confirm_fail_counts_as_decline=bool(rng.integers(0, 2)))
        t0 = pd.Timestamp(ctx.episode.start)
        b = cor.episode_bound(c, ctx.state, t0, window_days)
        bank = _Bank(StubServing(), ctx_with(ctx.state, start=t0), None, 0.8, None, c)
        full = trial % 4 < 2
        if trial % 8 == 7:  # many small operations at random seconds, same-second duplicates
            n = 300
            offs = np.sort(rng.integers(0, window_days * 86_400, n))
            offs[1::3] = offs[0::3][:len(offs[1::3])]
            offs = np.sort(np.append(offs, window_days * 86_400 - 1))
            for o in offs:
                bank.submit(Tx(t0 + pd.Timedelta(seconds=int(o)), float(np.round(rng.uniform(0.5, 40), 2)),
                               "fraud_A", "misc_pos", 40.0, -80.0))
            full = False
        else:
            for t in _burst_times(t0, window_days, rng, full):
                _greedy_fill(bank, t)
        o = bank.outcome()
        got = round(o.passed_amount * 100)
        used = dict(o.corridor_c7)
        assert all(b["c7_by_day"][d] == v for d, v in used.items())
        if full:
            assert set(used) == set(b["c7_by_day"]) and max(used.values()) == b["bound_cents"]
        violations += got > b["bound_cents"] or got > b["headroom_cents"]
        if full and no24:
            assert got == b["headroom_cents"]  # the headroom bound is tight
            tight += 1
        attained += got == b["bound_cents"] > 0
        positive += got > 0
    assert violations == 0
    assert tight >= 30 and attained >= 5 and positive > 100


def test_bound_reached_by_greedy_attacker():
    """The bound is not vacuous: an attacker splitting into small steps under an approving model gets close."""
    st = CardState(PROF)
    c = cfg(floor_24h=1e6, floor_7d=500.0, confirm_fail_counts_as_decline=False)
    txs = [tx(T0 + pd.Timedelta(hours=i), 60.0) for i in range(20)]
    o = run_episode(StubServing(), ctx_with(st), txs, 1, 0.8, corridor=c)
    b = cor.episode_bound(c, st, T0, 7)
    assert b["bound_cents"] == 50_000 and round(o.passed_amount * 100) == 48_000


# ---- batch / serving parity of the corridor state ---------------------------------------------------------

def test_parity_batch_vs_serving():
    df = make_sparkov_like(n_cards=12, seed=9)
    rng = np.random.default_rng(1)
    m = _random_multipliers(rng)
    c = cfg(q=0.9, lookback_days=40, floor_24h=20.0, floor_7d=60.0,
            multipliers=("holiday", "weekday", "day_of_month"), fitted=m)
    n_checked = 0
    for card, g in df.groupby("cc_num"):
        g = g.sort_values(["ts", "row"], kind="stable")
        auto = rng.uniform(size=len(g)) < 0.7
        passed = auto | (rng.uniform(size=len(g)) < 0.5)  # auto-approved ops passed; others confirmed or not
        batch = cor.batch_state(g["ts"].to_numpy(dtype="datetime64[ns]").view("int64"), g["amt"].to_numpy(),
                                auto, passed, c)
        st = CardState(PROF)
        for i, r in enumerate(g.itertuples(index=False)):
            t = pd.Timestamp(r.ts)
            s24 = st.auto_sum_cents(t.value, 86_400 * 10**9)
            s7 = st.auto_sum_cents(t.value, 7 * 86_400 * 10**9)
            c24, c7 = cor.limits_cents(c, st, t)
            assert (s24, s7, c24, c7) == (batch["s24"][i], batch["s7"][i], batch["c24"][i], batch["c7"][i])
            st.push(tx(t, r.amt), auto_approved=bool(auto[i]), passed=bool(passed[i]))
            n_checked += 1
    assert n_checked > 300


# ---- audit integration: corridor null, never-binding corridor, n_jobs identity -----------------------------

@pytest.fixture(scope="module")
def audit_env(tmp_path_factory):
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel, train_model
    df = make_sparkov_like(n_cards=80)
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


def test_audit_never_binding_corridor_equals_null(audit_env):
    a = _audit(audit_env, None)
    b = _audit(audit_env, cfg(floor_24h=1e9, floor_7d=1e9))
    keys = ("ver_adv", "ver_clean", "n_ops", "n_declines", "blocked")
    assert [{k: t[k] for k in keys} for t in a.per_target] == [{k: t[k] for k in keys} for t in b.per_target]
    assert a.uplift == b.uplift and "corridor" not in a.meta and b.meta["corridor"]["n_bound_violations"] == 0


def test_audit_corridor_meta_and_n_jobs_identity(audit_env):
    c = cfg(q=0.9, floor_24h=10.0, floor_7d=30.0, multipliers=("holiday", "weekday", "day_of_month"))
    a = _audit(audit_env, c, n_jobs=1)
    b = _audit(audit_env, c, n_jobs=2)
    assert a.per_target == b.per_target and a.uplift == b.uplift
    m = a.meta["corridor"]
    assert m["n_bound_violations"] == 0 and m["n_headroom_violations"] == 0 and m["n_targets"] == a.n_targets
    for t in a.per_target:
        assert round(t["ver_adv"] * t["amount"] * 100) <= round(t["corridor_bound"] * 100)
        assert round(t["ver_adv"] * t["amount"] * 100) <= round(t["corridor_headroom"] * 100)
        assert t["bound_violation"] is False and t["headroom_violation"] is False
        assert t["corridor_bound_share"] == pytest.approx(t["corridor_bound"] / t["amount"])
    assert set(m["bound_share"]) == {"mean", "median", "max", "q25", "q75", "q90"}
    assert "headroom" in cor_line(m)


def cor_line(m):
    from fraudaudit.audit import corridor_line
    return corridor_line(m)


def test_bound_stats_quantiles():
    pt = [{"amount": 100.0, "corridor_bound": float(b), "corridor_headroom": float(b) / 2} for b in range(1, 101)]
    s = cor.bound_stats(pt, 0.8)
    x = np.arange(1, 101) / 100
    assert s["bound_share"]["q25"] == pytest.approx(np.quantile(x, 0.25))
    assert s["bound_share"]["q75"] == pytest.approx(np.quantile(x, 0.75))
    assert s["bound_share"]["q90"] == pytest.approx(np.quantile(x, 0.9))
    assert s["bound_share"]["median"] == pytest.approx(np.median(x))
    assert s["share_targets_bound_below_rho_a"] == pytest.approx(0.79)


def test_multiplier_summary_reports_tests_and_expected_false_positives():
    m = cor.estimate_from_daily(_seasonal_daily(), cfg(multipliers=("holiday", "weekday", "day_of_month")))
    s = m.summary()
    assert s["n_tests"] == len(m.estimates) == s["n_estimates"]
    assert s["confidence"] == 0.95
    assert s["expected_false_positives"] == pytest.approx(len(m.estimates) * 0.05)


# ---- confirmation burden on the legitimate stream (Task 18 replay infrastructure) -------------------------

def _burden_df():
    from test_metrics import _fb_df
    return _fb_df(n_cards=3, n_windows=2, per_window=25)


def test_confirmation_burden_by_hand():
    from fraudaudit.metrics import confirmation_burden
    from test_metrics import _StubServing
    df = _burden_df()  # 25 hourly ops of amt 1..25 per window; one history op of 1.00
    r = confirmation_burden(df, _StubServing(), 0.5, cfg(floor_24h=100.0, floor_7d=1e6, q=0.0), n_jobs=1)
    # day 1 (hours 1..23): cumulative auto sum 1+2+…; ops 10 and 20 are declined by the model (not auto)
    assert r.n_ops == 150 and r.n_declines == 12 and r.n_confirms > 0
    assert 0 < r.share_confirm_ops < 1 and r.share_confirm_card_days == pytest.approx(
        r.n_confirm_card_days / r.n_card_days)
    r_off = confirmation_burden(df, _StubServing(), 0.5, cfg(floor_24h=1e9, floor_7d=1e9), n_jobs=1)
    assert r_off.n_confirms == 0 and r_off.share_confirm_card_days == 0.0


def test_confirmation_burden_cached_scores_equal_live(audit_env):
    from fraudaudit.metrics import confirmation_burden, legit_test_scores
    df, model, d, _ = audit_env
    c = cfg(q=0.9, floor_24h=30.0, floor_7d=80.0)
    live = confirmation_burden(df, model, model.tau, c, n_jobs=1)
    scores = legit_test_scores(df, model, n_jobs=1)
    par_scores = legit_test_scores(df, model, n_jobs=2, model_dir=d)
    assert par_scores.keys() == scores.keys() and all(np.array_equal(par_scores[k], v) for k, v in scores.items())
    cached = confirmation_burden(df, scores, model.tau, c, n_jobs=1)
    par = confirmation_burden(df, scores, model.tau, c, n_jobs=2)
    assert live == cached == par and live.n_confirms > 0


# ---- corridor-sweep ------------------------------------------------------------------------------------------

def test_pareto_flags():
    from fraudaudit.corridor_sweep import pareto
    assert pareto([(1, 1), (2, 2), (0, 3), (1, 1)]) == [True, False, True, True]
    assert pareto([(1, 2, 3), (1, 2, 2), (0, 5, 5)]) == [False, True, True]


def test_derive_with_tau(audit_env, tmp_path):
    from fraudaudit.model.artifacts import derive_with_tau, load_artifacts
    _, model, d, _ = audit_env
    derive_with_tau(d, tmp_path / "m", 0.25, {"target_fpr": 0.003})
    a = load_artifacts(tmp_path / "m")
    assert a.tau == 0.25 and a.meta["target_fpr"] == 0.003 and a.meta["derived_from"]["dir"] == str(d)
    assert load_artifacts(d).tau == model.tau  # the source is untouched


def test_corridor_sweep_end_to_end(tmp_path):
    from fraudaudit.cli import build_parser
    from fraudaudit.corridor_sweep import corridor_sweep, write_outputs
    from fraudaudit.gate import GateConfig
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel, train_model
    df = make_sparkov_like(n_cards=120, days=200, start=pd.Timestamp("2019-11-01"), seed=3)
    model, _ = train_model(df, seed=0, epochs=1)
    model.save(tmp_path / "nofpr", meta={"seed": 0})
    model.save(tmp_path / "model", meta={"seed": 0, "target_fpr": 0.01})
    model = FraudModel.from_artifacts(load_artifacts(tmp_path / "model"))
    tm = dataclasses.replace(ThreatModel.from_yaml(TM_CORRIDOR_PATH), recon_budget_ops=100)
    tm = dataclasses.replace(tm, corridor=dataclasses.replace(tm.corridor, lookback_days=60))  # short fixture history
    g = GateConfig(target_fpr=0.01, recall_clean_min=0.5, delta=0.1, attacks=["random"], attack_seeds=[1],
                   confidence=0.95, one_sided=True, spares_period="fit")
    with pytest.raises(ValueError, match="target_fpr"):  # M6: no silent skip of the tau reproduction check
        corridor_sweep(df, model, tmp_path / "nofpr", tm, g, qs=(0.9,), fprs=(0.05,), seed=1, n_jobs=1,
                       out_dir=tmp_path / "out0")
    with pytest.raises(ValueError, match="not in the FPR grid"):  # 22b review: error when 0.01 is not in the grid
        corridor_sweep(df, model, tmp_path / "model", tm, g, qs=(0.9,), fprs=(0.05,), seed=1, n_jobs=1,
                       out_dir=tmp_path / "out1")
    with pytest.raises(ValueError, match="variant"):
        corridor_sweep(df, model, tmp_path / "model", tm, g, qs=(0.9,), fprs=(0.01,), variants=("v3",), seed=1,
                       n_jobs=1, out_dir=tmp_path / "out2")
    data = corridor_sweep(df, model, tmp_path / "model", tm, g, qs=(0.9, 0.99), fprs=(0.01,),
                          variants=("base", "v1:2+v2"), seed=1, n_jobs=1, out_dir=tmp_path / "out")
    assert [(c["variant"], c["q"]) for c in data["cells"]] == [(None, None), ("base", 0.9), ("base", 0.99),
                                                                ("v1:2+v2", 0.9), ("v1:2+v2", 0.99)]
    grid = data["cells"][1:]
    assert all(c["n_bound_violations"] == 0 and c["n_headroom_violations"] == 0 and c["n_block_violations"] == 0
               and "pareto_3d" in c for c in grid)
    assert grid[0]["bound"]["bound"]["mean"] <= grid[1]["bound"]["bound"]["mean"]  # a higher q never lowers C
    assert grid[0]["floor_fit"]["q"] == 0.9 and grid[1]["floor_fit"]["q"] == 0.99  # the floor is refitted per q
    assert grid[2]["variant_label"] == "V1(κ=2)+V2" and len(grid[2]["bound"]["b_h"]["median"]) == 7
    assert grid[2]["empirical_blocks"]["n_over"] == 0
    assert all(c["burden"]["n_ops"] > 0 and 0 <= c["burden"]["share_confirm_card_days"] <= 1 for c in grid)
    write_outputs(data, tmp_path / "out")
    assert (tmp_path / "out" / "corridor-sweep.png").stat().st_size > 0
    md = (tmp_path / "out" / "corridor-sweep.md").read_text()
    assert "| target FPR |" in md and "M(c) P90 $" in md and "## B(h)" in md
    a = build_parser().parse_args(["corridor-sweep", "--q", "0.9,0.99", "--target-fpr", "0.01",
                                   "--variants", "base,v1:2"])
    assert a.q == [0.9, 0.99] and a.target_fpr == [0.01] and str(a.threat).endswith("realistic-corridor.yaml")
    assert a.variants == ["base", "v1:2"]
    d = build_parser().parse_args(["corridor-sweep"])
    assert d.q == [0.98, 0.99, 0.995] and d.target_fpr == [0.01] and len(d.variants) == 6


def test_report_mentions_corridor(audit_env, tmp_path):
    from fraudaudit.audit import summary_markdown, to_json
    from fraudaudit.gate import GateConfig, decide
    from fraudaudit.report import write_gate_outputs
    r = _audit(audit_env, cfg(q=0.9, floor_24h=10.0, floor_7d=30.0))
    assert "Spending corridor" in summary_markdown(to_json(r))
    g = GateConfig(target_fpr=0.01, recall_clean_min=0.0, delta=0.1, attacks=["random"], attack_seeds=[1],
                   confidence=0.95, one_sided=True, spares_period="test")
    res = {"random/1": r}
    write_gate_outputs(tmp_path, g, decide(g, res), res, {"seed": 7})
    md = (tmp_path / "summary.md").read_text()
    assert "## Spending corridor" in md and "random/1 0" in md


def test_val_tau_reproduces_training_tau(audit_env):
    from fraudaudit.corridor_sweep import val_tau
    df, model, _, _ = audit_env
    assert val_tau(df, model, 0.01) == pytest.approx(model.tau, rel=1e-6)  # train_model: tau at 1 % FPR on val
