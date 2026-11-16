"""Stepwise closed-loop held-out attack: policy, adaptive runner, search, audit dispatch, gate, harden."""
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import make_sparkov_like
from fraudaudit.attacks.random_search import Attack
from fraudaudit.attacks.stepwise import StepwiseAttack
from fraudaudit.data import fraud_episodes
from fraudaudit.episode import EpisodeOutcome, run_adaptive_episode, run_episode
from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.oracle import ReconOracle
from fraudaudit.policy import POLICY_FIELDS, StepwisePolicy
from fraudaudit.serving.core import Decision, Serving
from fraudaudit.serving.state import CardProfile, CardState, Tx
from fraudaudit.threat import MerchantTable, ThreatModel, contexts_for, feasible_ops

TM_PATH = Path(__file__).resolve().parents[1] / "threat-models" / "realistic.yaml"
AMT = FEATURE_COLUMNS.index("amt")
T0 = pd.Timestamp("2020-02-01 10:00:00")


class LimitModel:
    """Duck-typed model: decline iff amt > limit."""

    def __init__(self, encoder, limit=500.0):
        self.encoder, self.tau, self.limit = encoder, 0.5, limit

    def predict_proba(self, F):
        return (np.asarray(F)[:, AMT] > self.limit).astype(np.float64)


class AmountBank:
    """Stub bank on the decision interface: declined iff amt > limit; records every amount it saw."""

    tau = 0.5

    def __init__(self, limit):
        self.limit, self.seen = limit, []

    def decide(self, state, tx):
        self.seen.append(float(tx.amt))
        p = 1.0 if tx.amt > self.limit else 0.0
        return Decision(p=p, declined=p >= 0.5)


class ListBank:
    """Stub bank: p from a list in call order."""

    tau = 0.5

    def __init__(self, ps):
        self.ps, self.n = list(ps), 0

    def decide(self, state, tx):
        p = self.ps[self.n]
        self.n += 1
        return Decision(p=p, declined=p >= 0.5)


class CountingServing:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def decide(self, state, tx):
        self.calls += 1
        return self.inner.decide(state, tx)


class ReplayPolicy:
    """A 'policy' that ignores the decisions and proposes a fixed operation list (equivalence test)."""

    def __init__(self, txs):
        self.txs = txs

    def act(self, ctx, tm, merchants, rng):
        for tx in self.txs:
            yield tx


@pytest.fixture(scope="module")
def df():
    return make_sparkov_like()


@pytest.fixture(scope="module")
def tm():
    return ThreatModel.from_yaml(TM_PATH)


@pytest.fixture(scope="module")
def merchants(df):
    return MerchantTable.from_df(df)


@pytest.fixture(scope="module")
def contexts(df):
    return contexts_for(df, fraud_episodes(df, {"fit", "val", "test"}))


def _ctx(amount=1000.0, start=T0):
    prof = CardProfile(cc_num=1, gender="F", dob=pd.Timestamp("1980-05-05"), lat=40.0, long=-80.0,
                       city_pop=1000, state="NC")
    state = CardState(prof)
    state.push(Tx(start - pd.Timedelta(days=1), 5.0, "fraud_A", "misc_pos", 40.0, -80.0))
    return SimpleNamespace(episode=SimpleNamespace(amount=amount, start=start, card=1), profile=prof, state=state)


def _policy(**kw):
    base = dict(s0=0.5, g=1.5, b=0.5, hour=12.0, hour_jitter=12.0, gap_h=1.0, cat_pref=0, switch_on_decline=0,
                u=1.0)
    base.update(kw)
    return StepwisePolicy(**base)


# ---- policy vector: ranges and clipping -------------------------------------------------------------------

def test_policy_vector_layout_and_roundtrip(tm, merchants):
    assert POLICY_FIELDS == ("s0", "g", "b", "hour", "hour_jitter", "gap_h", "cat_pref", "switch_on_decline", "u")
    rng = np.random.default_rng(0)
    n_cat = merchants.n_categories
    for _ in range(50):
        p = StepwisePolicy.random(tm, n_cat, rng)
        v = p.to_vector()
        assert v.shape == (len(POLICY_FIELDS),)
        assert StepwisePolicy.from_vector(v, tm, n_cat) == p
        assert p.clip(tm, n_cat) == p  # random policies are already in range


def test_policy_clip_ranges(tm, merchants):
    n_cat = merchants.n_categories
    lo = StepwisePolicy.from_vector([-5, -5, -5, -3, -5, -5, -5, -5, -5], tm, n_cat)
    hi = StepwisePolicy.from_vector([9, 9, 9, 25.5, 99, 999, 999, 9, 9], tm, n_cat)
    nan = StepwisePolicy.from_vector([np.nan] * 9, tm, n_cat)
    gmin = tm.min_gap_secs / 3600
    assert (lo.s0, lo.g, lo.b, lo.hour_jitter, lo.gap_h, lo.cat_pref, lo.switch_on_decline, lo.u) == \
        (0.02, 1.0, 0.1, 0.0, gmin, 0, 0, tm.rho)
    assert lo.hour == pytest.approx(21.0)  # the time of day is circular: -3 h is 21:00
    assert (hi.s0, hi.g, hi.b, hi.hour_jitter, hi.gap_h, hi.cat_pref, hi.switch_on_decline, hi.u) == \
        (1.0, 3.0, 0.9, 12.0, 48.0, n_cat - 1, 1, 1.0)
    assert hi.hour == pytest.approx(1.5)
    for p in (lo, hi, nan):
        assert 0.0 <= p.hour < 24.0
        assert isinstance(p.cat_pref, int) and isinstance(p.switch_on_decline, int)
    with pytest.raises(ValueError):
        StepwisePolicy.from_vector(np.zeros(8), tm, n_cat)


def test_policy_random_within_bounds(tm, merchants):
    n_cat = merchants.n_categories
    rng = np.random.default_rng(3)
    ps = [StepwisePolicy.random(tm, n_cat, rng) for _ in range(300)]
    for name, (a, b) in StepwisePolicy.bounds(tm, n_cat).items():
        vals = [getattr(p, name) for p in ps]
        assert min(vals) >= a and max(vals) <= b, name
    assert {p.switch_on_decline for p in ps} == {0, 1}
    assert len({p.cat_pref for p in ps}) > 1


# ---- the closed-loop runner -------------------------------------------------------------------------------

def test_next_amount_reacts_to_decline(tm, merchants):
    """Stub bank declining amt > 300: after a decline with b < 1 the next amount is smaller."""
    tm3 = dataclasses.replace(tm, max_declines=3)
    bank = AmountBank(300.0)
    p = _policy(s0=0.5, b=0.5, g=1.5)
    o, txs = run_adaptive_episode(bank, _ctx(), p, tm3, merchants, np.random.default_rng(0),
                                  tau_after_decline=None)
    amts = [t.amt for t in txs]
    assert amts[0] == pytest.approx(round(0.5 * tm.amount_max, 2))
    assert o.decisions[:3] == (True, True, False)
    assert amts[1] < amts[0] and amts[1] == pytest.approx(round(amts[0] * 0.5, 2), abs=0.011)
    assert amts[2] < amts[1]
    assert amts[3] > amts[2]  # after a pass the amount grows by g
    assert bank.seen == amts  # the bank saw exactly the executed operations


def test_policy_does_not_see_p(tm, merchants):
    """knowledge: decision – two banks with different p but identical decisions give identical episodes."""
    class PBank(AmountBank):
        def __init__(self, limit, p_pass, p_dec):
            super().__init__(limit)
            self.p_pass, self.p_dec = p_pass, p_dec

        def decide(self, state, tx):
            d = tx.amt > self.limit
            return Decision(p=self.p_dec if d else self.p_pass, declined=d)
    tm3 = dataclasses.replace(tm, max_declines=3)
    runs = [run_adaptive_episode(PBank(300.0, pp, pd_), _ctx(), _policy(), tm3, merchants,
                                 np.random.default_rng(0), tau_after_decline=None)[1]
            for pp, pd_ in ((0.0, 1.0), (0.49, 0.5))]
    assert runs[0] == runs[1]


def test_block_stops_policy(tm, merchants):
    bank = AmountBank(300.0)
    o, txs = run_adaptive_episode(bank, _ctx(), _policy(s0=0.5), tm, merchants, np.random.default_rng(0),
                                  tau_after_decline=None)
    assert tm.max_declines == 1 and o.blocked and o.n_ops == 1 and len(txs) == 1 and o.passed_amount == 0


def test_stops_at_target_and_k_max(tm, merchants):
    never = AmountBank(1e9)
    o, txs = run_adaptive_episode(never, _ctx(amount=1000.0), _policy(s0=0.1, g=1.0, u=0.9), tm, merchants,
                                  np.random.default_rng(0), tau_after_decline=None)
    assert o.passed_amount >= 900.0 and sum(t.amt for t in txs[:-1]) < 900.0  # stopped right after u·A
    assert txs[-1].amt <= 900.0 - sum(t.amt for t in txs[:-1]) + 0.005  # the last op takes only the rest
    o, txs = run_adaptive_episode(AmountBank(1e9), _ctx(amount=1e6), _policy(s0=0.02, g=1.0, gap_h=0.5), tm,
                                  merchants, np.random.default_rng(0), tau_after_decline=None)
    assert len(txs) == tm.k_max and o.n_ops == tm.k_max and not o.success


def test_time_band_gap_and_window(tm, merchants):
    """Ops at least gap apart, inside the daily band hour ± jitter, inside the window; stops at window end."""
    p = _policy(s0=0.02, g=1.0, hour=3.0, hour_jitter=1.0, gap_h=0.5)
    o, txs = run_adaptive_episode(AmountBank(1e9), _ctx(amount=1e6), p, tm, merchants, np.random.default_rng(0),
                                  tau_after_decline=None)
    ts = [t.ts for t in txs]
    assert all(b - a >= pd.Timedelta(minutes=30) for a, b in zip(ts, ts[1:]))
    for t in ts:
        h = t.hour + t.minute / 60 + t.second / 3600
        assert 2.0 <= h <= 4.0
    assert ts[0] >= T0 and ts[-1] < T0 + pd.Timedelta(days=tm.window_days)
    # 2 h band, 30 min gap: 5 ops per day fit (2:00, 2:30, ..., 4:00); 20 ops span 4 days
    assert len(txs) == tm.k_max
    long_gap = _policy(s0=0.02, g=1.0, hour=12.0, hour_jitter=12.0, gap_h=48.0)
    o, txs = run_adaptive_episode(AmountBank(1e9), _ctx(amount=1e6), long_gap, tm, merchants,
                                  np.random.default_rng(0), tau_after_decline=None)
    assert len(txs) == 4  # 0, 48, 96, 144 h; 192 h is outside the 7-day window


def test_switch_on_decline_changes_category(tm, merchants):
    tm5 = dataclasses.replace(tm, max_declines=5)
    always = AmountBank(-1.0)  # every op declined
    _, stay = run_adaptive_episode(always, _ctx(), _policy(cat_pref=2, switch_on_decline=0), tm5, merchants,
                                   np.random.default_rng(0), tau_after_decline=None)
    _, move = run_adaptive_episode(AmountBank(-1.0), _ctx(), _policy(cat_pref=2, switch_on_decline=1), tm5,
                                   merchants, np.random.default_rng(0), tau_after_decline=None)
    cats = merchants.categories
    assert {t.category for t in stay} == {cats[2]}
    assert move[0].category == cats[2]
    assert all(b.category != a.category for a, b in zip(move, move[1:]))


def test_equivalence_with_run_episode_on_fixed_sequence(tm, merchants):
    txs = [Tx(T0 + pd.Timedelta(minutes=2 * i), float(10 + i), "fraud_A", "misc_pos", 40.0, -80.0)
           for i in range(8)]
    ps = [0.1, 0.7, 0.2, 0.3, 0.9, 0.05, 0.6, 0.1]
    for md, tad in ((1, None), (2, None), (3, 0.25), (None, None), (None, 0.15)):
        tm_x = dataclasses.replace(tm, max_declines=md, rho=0.05)
        ref = run_episode(ListBank(ps), _ctx(amount=100.0), txs, md, tm_x.rho, tau_after_decline=tad)
        got, ex = run_adaptive_episode(ListBank(ps), _ctx(amount=100.0), ReplayPolicy(txs), tm_x, merchants,
                                       np.random.default_rng(0), tau_after_decline=tad)
        assert isinstance(got, EpisodeOutcome)
        assert got == ref, (md, tad)
        assert ex == txs[:ref.n_ops]


def test_runner_never_exceeds_k_max(tm, merchants):
    txs = [Tx(T0 + pd.Timedelta(minutes=2 * i), 5.0, "fraud_A", "misc_pos", 40.0, -80.0) for i in range(30)]
    o, ex = run_adaptive_episode(ListBank([0.0] * 30), _ctx(amount=1e4), ReplayPolicy(txs), tm, merchants,
                                 np.random.default_rng(0), tau_after_decline=None)
    assert o.n_ops == tm.k_max == len(ex)


def test_adaptive_ops_feasible_on_real_contexts(df, tm, merchants, contexts):
    serving = Serving(LimitModel(BaseEncoder().fit(df), limit=200.0))
    rng = np.random.default_rng(5)
    n_ops = 0
    for i in range(60):
        p = StepwisePolicy.random(tm, merchants.n_categories, rng)
        ctx = contexts[i % len(contexts)]
        tm_i = dataclasses.replace(tm, max_declines=1 + i % 3)
        o, txs = run_adaptive_episode(serving, ctx, p, tm_i, merchants, np.random.default_rng([1, i]),
                                      tau_after_decline=None)
        assert len(txs) == o.n_ops <= tm.k_max
        assert all(feasible_ops(txs, ctx, tm, merchants)) if txs else True
        n_ops += len(txs)
    assert n_ops > 60


def test_adaptive_deterministic_for_rng(tm, merchants, contexts, df):
    serving = Serving(LimitModel(BaseEncoder().fit(df)))
    p = StepwisePolicy.random(tm, merchants.n_categories, np.random.default_rng(1))
    a = run_adaptive_episode(serving, contexts[0], p, tm, merchants, np.random.default_rng([4, 2]),
                             tau_after_decline=None)
    b = run_adaptive_episode(serving, contexts[0], p, tm, merchants, np.random.default_rng([4, 2]),
                             tau_after_decline=None)
    assert a == b


# ---- search: StepwiseAttack on the recon oracle -----------------------------------------------------------

@pytest.fixture(scope="module")
def spares(df):
    eps = [dataclasses.replace(e, amount=3000.0) for e in fraud_episodes(df, {"fit", "val", "test"})]
    return contexts_for(df, eps)


@pytest.fixture(scope="module")
def serving(df):
    return Serving(LimitModel(BaseEncoder().fit(df)))


def test_stepwise_attack_protocol():
    assert isinstance(StepwiseAttack(), Attack)


def test_stepwise_budget_accounting(serving, spares, tm, merchants):
    cs = CountingServing(serving)
    with ReconOracle(cs, spares, tm, merchants, 700, np.random.default_rng(0)) as o:
        run = StepwiseAttack().run(o, tm, np.random.default_rng(1))
        assert run.ops_used == o.ops_used == cs.calls <= 700
        assert run.evals == o.evals >= 1
        # the worst case of a policy is cards_per_eval · k_max: the last evaluation needed that much room
        assert o.remaining < tm.cards_per_eval * tm.k_max
        assert isinstance(run.best, StepwisePolicy)
        assert run.best_fitness == max(f for _, f in o.history)


def test_stepwise_reserves_worst_case(serving, spares, tm, merchants):
    with ReconOracle(serving, spares, tm, merchants, tm.cards_per_eval * tm.k_max - 1,
                     np.random.default_rng(0)) as o:
        run = StepwiseAttack().run(o, tm, np.random.default_rng(1))
    assert run.best is None and run.evals == 0 and run.ops_used == 0


def test_stepwise_learns_on_stub(serving, spares, tm, merchants):
    """On a bank declining amt > 500 with max_declines 1, the best policy starts below the limit."""
    with ReconOracle(serving, spares, tm, merchants, 3000, np.random.default_rng(0)) as o:
        run = StepwiseAttack().run(o, tm, np.random.default_rng(2))
    assert run.best.s0 * tm.amount_max <= 500.0
    assert run.best_fitness > 0.1


def test_stepwise_deterministic(serving, spares, tm, merchants):
    runs = []
    for bs in (3, 8):  # the batch size does not change the candidate sequence
        with ReconOracle(serving, spares, tm, merchants, 900, np.random.default_rng(0)) as o:
            runs.append((StepwiseAttack(batch_size=bs).run(o, tm, np.random.default_rng(1)), o.history))
    (a, ha), (b, hb) = runs
    assert (a.best, a.best_fitness, a.evals, a.ops_used) == (b.best, b.best_fitness, b.evals, b.ops_used)
    assert ha == hb


@pytest.fixture(scope="module")
def real_model_dir(df, tmp_path_factory):
    from fraudaudit.model.pipeline import train_model
    model, _ = train_model(df, seed=0, epochs=1)
    d = tmp_path_factory.mktemp("model")
    model.save(d, meta={"seed": 0})
    return d


def test_stepwise_identical_for_n_jobs(df, tm, merchants, real_model_dir):
    sp = contexts_for(df, fraud_episodes(df, {"fit", "val", "test"}))
    runs = []
    for n_jobs in (1, 2):
        with ReconOracle(None, sp, tm, merchants, 800, np.random.default_rng(11), n_jobs=n_jobs,
                         model_dir=real_model_dir) as o:
            runs.append((StepwiseAttack().run(o, tm, np.random.default_rng(12)), o.history))
    (a, ha), (b, hb) = runs
    assert a.evals >= 4
    assert (a.best, a.best_fitness, a.evals, a.ops_used) == (b.best, b.best_fitness, b.evals, b.ops_used)
    assert ha == hb


# ---- gate and harden --------------------------------------------------------------------------------------

def test_gate_accepts_stepwise_and_makes_it(tm):
    from fraudaudit.gate import ATTACKS, HELD_OUT_ATTACKS, GateConfig, make_attack
    assert "stepwise" in ATTACKS and HELD_OUT_ATTACKS == ("stepwise",)
    assert isinstance(make_attack("stepwise", tm), StepwiseAttack)
    cfg = GateConfig.from_yaml(Path(__file__).resolve().parents[1] / "gate.yaml")
    assert cfg.attacks == ["random", "es", "stepwise"] and cfg.attack_seeds == [1, 2, 3]


def test_harden_rejects_stepwise_for_generation(df, tm):
    from fraudaudit import harden as hmod
    with pytest.raises(ValueError, match="held-out"):
        hmod.check_generation_attack("stepwise")
    with pytest.raises(ValueError, match="only"):
        hmod.check_generation_attack("random")
    hmod.check_generation_attack("es")
    with pytest.raises(ValueError, match="held-out"):  # before any data or model work
        hmod.harden(df, None, tm, None, rounds=1, model_dir="nowhere", out_dir="nowhere", gen_attack="stepwise")


def test_cli_harden_gen_attack_flag():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["harden", "--gen-attack", "stepwise"])
    assert a.gen_attack == "stepwise"
    assert build_parser().parse_args(["harden"]).gen_attack == "es"


# ---- audit dispatch -------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audit_df():
    return make_sparkov_like(n_cards=80)


@pytest.fixture(scope="module")
def audit_model_dir(audit_df, tmp_path_factory):
    from fraudaudit.model.pipeline import train_model
    model, _ = train_model(audit_df, seed=0, epochs=1)
    d = tmp_path_factory.mktemp("audit-model")
    model.save(d, meta={"seed": 0})
    return d


@pytest.fixture(scope="module")
def audit_model(audit_model_dir):
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    return FraudModel.from_artifacts(load_artifacts(audit_model_dir))


def _audit(audit_df, audit_model, tm, audit_model_dir, n_jobs):
    from fraudaudit.audit import run_audit
    return run_audit(audit_df, audit_model, dataclasses.replace(tm, recon_budget_ops=400), StepwiseAttack(), seed=7,
                     n_jobs=n_jobs, model_dir=audit_model_dir, target_periods=("fit", "val", "test"),
                     threat_path=TM_PATH, spares_period="test", one_sided=True, attack_seed=1)


def test_audit_dispatches_policy_to_adaptive_runner(audit_df, audit_model, tm, audit_model_dir, monkeypatch):
    from fraudaudit import audit as audit_mod
    from fraudaudit import episode as episode_mod
    calls = {"adaptive": 0, "fixed": 0}
    orig_a, orig_f = episode_mod.run_adaptive_episode, episode_mod.run_episode

    def spy_a(*a, **kw):
        calls["adaptive"] += 1
        return orig_a(*a, **kw)

    def spy_f(*a, **kw):
        calls["fixed"] += 1
        return orig_f(*a, **kw)
    monkeypatch.setattr(episode_mod, "run_adaptive_episode", spy_a)
    monkeypatch.setattr(episode_mod, "run_episode", spy_f)
    r = _audit(audit_df, audit_model, tm, audit_model_dir, n_jobs=1)
    assert isinstance(r.best, StepwisePolicy) and r.best == r.best.clip(tm, MerchantTable.from_df(audit_df)
                                                                        .n_categories)
    # every recon card and every target go through the closed-loop runner, never through run_episode
    assert calls["fixed"] == 0
    assert calls["adaptive"] == r.meta["recon"]["evals"] * tm.cards_per_eval + r.n_targets
    assert r.feasible_share == 1.0
    assert r.meta["n_executed_adv_ops"] == sum(t["n_ops"] for t in r.per_target) > 0
    assert r.meta["recon"]["attack"] == "StepwiseAttack" and r.meta["recon"]["held_out"] is True
    data = audit_mod.to_json(r, MerchantTable.from_df(audit_df))
    assert data["best"]["type"] == "stepwise_policy" and "category" in data["best"]
    assert "Closed-loop stepwise policy" in audit_mod.summary_markdown(data)


def test_audit_stepwise_identical_for_n_jobs(audit_df, audit_model, tm, audit_model_dir):
    r1 = _audit(audit_df, audit_model, tm, audit_model_dir, n_jobs=1)
    r2 = _audit(audit_df, audit_model, tm, audit_model_dir, n_jobs=2)
    assert r1.best == r2.best and r1.per_target == r2.per_target and r1.uplift == r2.uplift


def test_gate_summary_marks_stepwise_held_out(audit_df, audit_model, tm, audit_model_dir):
    from fraudaudit.gate import GateConfig, decide
    from fraudaudit.report import summary_markdown
    r = _audit(audit_df, audit_model, tm, audit_model_dir, n_jobs=1)
    cfg = GateConfig(target_fpr=0.01, recall_clean_min=0.0, delta=10.0, attacks=["stepwise"], attack_seeds=[1],
                     confidence=0.95, one_sided=True, spares_period="test")
    results = {"stepwise/1": r}
    md = summary_markdown(decide(cfg, results), results, {}, MerchantTable.from_df(audit_df))
    assert "stepwise (held-out, never used in hardening)" in md
    assert "Worst case: attack `stepwise` (held-out, never used in hardening)" in md
    assert "Closed-loop stepwise policy" in md


def test_cli_harden_refuses_stepwise_before_loading(tmp_path, capsys):
    from fraudaudit.cli import main
    rc = main(["harden", "--gen-attack", "stepwise", "--data", str(tmp_path / "none"), "--out", str(tmp_path / "o")])
    assert rc == 2 and "held-out" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
