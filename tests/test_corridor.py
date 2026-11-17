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
from fraudaudit.serving.state import CardProfile, CardState, LegitHistory, Tx
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
    return LegitHistory(t, a)


def test_window_quantile_by_hand():
    d1 = pd.Timestamp("2020-01-01")
    h = _hist([(d1 + pd.Timedelta(hours=10), 10.0), (d1 + pd.Timedelta(hours=12), 20.0),
               (d1 + DAY + pd.Timedelta(hours=9), 30.0), (d1 + 4 * DAY, 5.0)])
    day = (d1 + 5 * DAY).value
    w24 = 86_400 * 10**9
    # rolling 24 h sums right after every op: 10, 30, 60 (30 + both ops of day 1), 5
    assert h.window_quantile(day, 180 * w24, w24, 0.5) == pytest.approx(np.quantile([10, 30, 60, 5], 0.5))
    assert h.window_quantile(day, 180 * w24, w24, 1.0) == pytest.approx(60.0)
    assert h.window_quantile(day, 180 * w24, 7 * w24, 1.0) == pytest.approx(65.0)
    # only ops strictly before the day count; lookback [day - L, day)
    assert h.window_quantile((d1 + DAY).value, 180 * w24, w24, 1.0) == pytest.approx(30.0)
    assert h.window_quantile(day, 2 * w24, w24, 1.0) == pytest.approx(5.0)
    assert h.window_quantile(d1.value, 180 * w24, w24, 0.99) == 0.0  # no history


def test_window_value_counts_same_second_ops_and_lookback_edge():
    d1 = pd.Timestamp("2020-01-01 08:00:00")
    h = _hist([(d1, 10.0), (d1, 15.0), (d1 + DAY, 1.0)])  # exactly 24 h later: the first two drop out
    day = (d1.normalize() + 3 * DAY).value
    w24 = 86_400 * 10**9
    assert h.window_quantile(day, 180 * w24, w24, 1.0) == pytest.approx(25.0)
    assert h.window_quantile(day, 180 * w24, w24, 0.0) == pytest.approx(1.0)
    # lookback starts at day - L inclusive
    assert h.window_quantile(day, day - d1.value, w24, 0.0) == pytest.approx(1.0)


def test_legit_history_append_is_persistent():
    h0 = _hist([(T0 - 3 * DAY, 50.0)])
    h1 = h0.append((T0).value, 70.0)
    assert len(h0) == 1 and len(h1) == 2
    day = (T0.normalize() + DAY).value
    w = 86_400 * 10**9
    assert h0.window_quantile(day, 180 * w, w, 1.0) == pytest.approx(50.0)
    assert h1.window_quantile(day, 180 * w, w, 1.0) == pytest.approx(70.0)


def test_limit_cents_floor_quantile_multiplier():
    st = CardState(PROF)
    c = cfg(q=1.0)
    assert cor.limit_cents(c, st.legit, T0.value, "24h") == 10_000  # floor
    assert cor.limit_cents(c, st.legit, T0.value, "7d") == 30_000
    st.legit = _hist([(T0 - 2 * DAY, 250.55)])
    assert cor.limit_cents(c, st.legit, T0.value, "24h") == 25_055
    m = Multipliers(holiday={}, weekday={"0": 1.5}, day_of_month={}, estimates=(), holiday_weeks=(-3, 1))
    c2 = cfg(q=1.0, multipliers=("weekday",), fitted=m)
    assert cor.limit_cents(c2, st.legit, T0.value, "24h") == math.floor(250.55 * 1.5 * 100 + 1e-6)  # Monday
    assert cor.limit_cents(c2, st.legit, T0.value, "7d") == 30_000  # weekday factors only on the 24 h window
    assert cor.limit_cents(c2, st.legit, (T0 + DAY).value, "24h") == 25_055  # Tuesday: no factor


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


def test_from_history_legit_profile_excludes_fraud_and_future():
    h = pd.DataFrame({"cc_num": 1, "ts": [T0 - 200 * DAY, T0 - 2 * DAY, T0 - DAY, T0],
                      "amt": [5.0, 10.0, 99.0, 7.0], "category": "misc_pos", "is_fraud": [0, 0, 1, 0]})
    st = CardState.from_history(PROF, h, at=T0)
    assert len(st.legit) == 2  # the 200-day-old legit op is kept for the profile, fraud and ts >= at are not
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
    assert len(b.state.legit) == 3  # the owner's operations feed the profile


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


def _fixture_contexts():
    df = make_sparkov_like(n_cards=24, seed=5)
    eps = fraud_episodes(df, {"fit", "val", "test"})
    return df, contexts_for(df, eps), MerchantTable.from_df(df)


def _random_multipliers(rng):
    return Multipliers(holiday={f"christmas:{k}": float(rng.uniform(0.5, 2)) for k in range(-3, 2)},
                       weekday={str(d): float(rng.uniform(0.5, 2)) for d in range(7)},
                       day_of_month={str(d): float(rng.uniform(0.5, 2)) for d in range(1, 32)},
                       estimates=(), holiday_weeks=(-3, 1))


def test_property_guarantee_random_adversarial_plans():
    """Executable form of the Proposition: for any plan the auto-approved attacker amount is
    <= max_{t in [t0, t0 + D)} C_7d(c, t) (and <= the tighter headroom bound)."""
    df, ctxs, merchants = _fixture_contexts()
    tm = ThreatModel.from_yaml(TM_PATH)
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
        positive += got > 0
        confirms += o.n_confirms > 0
    assert violations == 0
    assert positive > 100 and confirms > 100  # not vacuous: plans pass and the corridor binds


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
        legit = (g["is_fraud"] == 0).to_numpy()
        batch = cor.batch_state(g["ts"].to_numpy(dtype="datetime64[ns]").view("int64"), g["amt"].to_numpy(),
                                auto, legit, c)
        st = CardState(PROF)
        for i, r in enumerate(g.itertuples(index=False)):
            t = pd.Timestamp(r.ts)
            s24 = st.auto_sum_cents(t.value, 86_400 * 10**9)
            s7 = st.auto_sum_cents(t.value, 7 * 86_400 * 10**9)
            c24, c7 = cor.limits_cents(c, st, t)
            assert (s24, s7, c24, c7) == (batch["s24"][i], batch["s7"][i], batch["c24"][i], batch["c7"][i])
            st.push(tx(t, r.amt), auto_approved=bool(auto[i]), legit=bool(legit[i]))
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
    assert m["n_bound_violations"] == 0 and m["n_targets"] == a.n_targets
    for t in a.per_target:
        assert round(t["ver_adv"] * t["amount"] * 100) <= round(t["corridor_bound"] * 100)
        assert t["corridor_bound_share"] == pytest.approx(t["corridor_bound"] / t["amount"])
    assert set(m["bound_share"]) == {"mean", "median", "max"}


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
    model.save(tmp_path / "model", meta={"seed": 0})
    model = FraudModel.from_artifacts(load_artifacts(tmp_path / "model"))
    tm = dataclasses.replace(ThreatModel.from_yaml(TM_CORRIDOR_PATH), recon_budget_ops=100)
    g = GateConfig(target_fpr=0.01, recall_clean_min=0.5, delta=0.1, attacks=["random"], attack_seeds=[1],
                   confidence=0.95, one_sided=True, spares_period="fit")
    data = corridor_sweep(df, model, tmp_path / "model", tm, g, qs=(0.9, 0.99), fprs=(0.05,), seed=1, n_jobs=1,
                          out_dir=tmp_path / "out")
    assert [c["q"] for c in data["cells"]] == [None, 0.9, 0.99]
    grid = data["cells"][1:]
    assert all(c["n_bound_violations"] == 0 and "pareto_3d" in c for c in grid)
    assert grid[0]["bound"]["bound"]["mean"] <= grid[1]["bound"]["bound"]["mean"]  # a higher q never lowers C
    assert all(c["burden"]["n_ops"] > 0 and 0 <= c["burden"]["share_confirm_card_days"] <= 1 for c in grid)
    write_outputs(data, tmp_path / "out")
    assert (tmp_path / "out" / "corridor-sweep.png").stat().st_size > 0
    assert "| target FPR |" in (tmp_path / "out" / "corridor-sweep.md").read_text()
    a = build_parser().parse_args(["corridor-sweep", "--q", "0.9,0.99", "--target-fpr", "0.01"])
    assert a.q == [0.9, 0.99] and a.target_fpr == [0.01] and str(a.threat).endswith("realistic-corridor.yaml")


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
