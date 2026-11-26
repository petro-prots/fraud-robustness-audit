import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fraudaudit.episode import EpisodeOutcome, run_episode
from fraudaudit.data import fraud_episodes
from fraudaudit.features.base import BaseEncoder
from fraudaudit.serving.core import Decision, Serving
from fraudaudit.serving.state import CardProfile, CardState, Tx

T0 = pd.Timestamp("2020-02-01 10:00:00")


class StubServing:
    """Serving-compatible: p from a list in call order; declined iff p >= 0.5."""

    tau = 0.5

    def __init__(self, ps):
        self.ps, self.calls = list(ps), []

    def decide(self, state, tx):
        self.calls.append(len(state))
        p = self.ps[len(self.calls) - 1]
        return Decision(p=p, declined=p >= 0.5)


def _ctx(amount=100.0):
    prof = CardProfile(cc_num=1, gender="F", dob=pd.Timestamp("1980-05-05"), lat=40.0, long=-80.0,
                       city_pop=1000, state="NC")
    state = CardState(prof)
    state.push(Tx(T0 - pd.Timedelta(days=1), 5.0, "fraud_A", "misc_pos", 40.0, -80.0))
    return SimpleNamespace(episode=SimpleNamespace(amount=amount), profile=prof, state=state)


def _txs(amts):
    return [Tx(T0 + pd.Timedelta(minutes=i), a, "fraud_A", "misc_pos", 40.0, -80.0) for i, a in enumerate(amts)]


def test_block_after_first_decline():
    s = StubServing([0, 1, 0])
    o = run_episode(s, _ctx(), _txs([10, 20, 30]), 1, 0.8)
    assert (o.passed_amount, o.n_ops, o.n_declines, o.blocked) == (10.0, 2, 1, True)
    assert len(s.calls) == 2 and o.decisions == (False, True)
    assert o.ver == pytest.approx(0.1)


def test_unlimited_declines():
    o = run_episode(StubServing([0, 1, 0]), _ctx(), _txs([10, 20, 30]), None, 0.8)
    assert (o.passed_amount, o.n_ops, o.n_declines, o.blocked) == (40.0, 3, 1, False)


def test_max_declines_two():
    o = run_episode(StubServing([1, 0, 1, 0]), _ctx(), _txs([1, 2, 3, 4]), 2, 0.8)
    assert (o.passed_amount, o.n_ops, o.n_declines, o.blocked) == (2.0, 3, 2, True)
    assert o.decisions == (True, False, True) and o.ver == pytest.approx(0.02)


def test_declined_tx_updates_state():
    s = StubServing([1, 0, 0])
    ctx = _ctx()
    run_episode(s, ctx, _txs([10, 20, 30]), None, 0.8)
    assert s.calls == [1, 2, 3]  # the declined first attempt is history for the second


def test_context_state_not_mutated():
    ctx = _ctx()
    before = len(ctx.state)
    run_episode(StubServing([0, 0]), ctx, _txs([1, 2]), None, 0.8)
    assert len(ctx.state) == before


def test_success_flag():
    ctx = _ctx(100.0)
    assert run_episode(StubServing([0, 0]), ctx, _txs([40, 40]), None, 0.8).success  # 80 >= 80
    o = run_episode(StubServing([0, 0]), ctx, _txs([40, 39]), None, 0.8)
    assert not o.success and o.ver == pytest.approx(0.79)


def test_no_decline_no_block_and_frozen():
    o = run_episode(StubServing([0]), _ctx(), _txs([5]), 1, 0.8)
    assert not o.blocked and o.n_declines == 0
    assert isinstance(o, EpisodeOutcome)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.ver = 1.0


def test_cent_exact_success():
    ctx = _ctx(0.30)  # 0.7 * 0.30 is 0.21 only up to float noise
    assert run_episode(StubServing([0]), ctx, _txs([0.21]), None, 0.7).success
    assert not run_episode(StubServing([0]), ctx, _txs([0.20]), None, 0.7).success


def test_rejects_bad_amount_and_max_declines():
    for a in (0.0, -5.0):
        with pytest.raises(ValueError, match="amount"):
            run_episode(StubServing([0]), _ctx(a), _txs([1]), None, 0.8)
    with pytest.raises(ValueError, match="max_declines"):
        run_episode(StubServing([0]), _ctx(), _txs([1]), 0, 0.8)


def test_real_serving_and_context(sparkov_like):
    from fraudaudit.threat import MerchantTable, Strategy, ThreatModel, context_for, realize
    from pathlib import Path

    class ListModel:
        def __init__(self, encoder, ps):
            self.encoder, self.tau, self.ps, self.i = encoder, 0.5, ps, 0

        def predict_proba(self, F):
            p = self.ps[self.i]
            self.i += 1
            return np.full(len(F), p)

    df = sparkov_like
    tm = ThreatModel.from_yaml(Path(__file__).resolve().parents[1] / "threat-models" / "realistic.yaml")
    merchants = MerchantTable.from_df(df)
    ctx = context_for(df, fraud_episodes(df, {"fit", "val", "test"})[0])
    rng = np.random.default_rng(3)
    s = dataclasses.replace(Strategy.random(tm, len(merchants), rng), k=4)
    txs = realize(s, ctx, tm, merchants, rng)
    assert len(txs) == 4
    n0 = len(ctx.state)
    model = ListModel(BaseEncoder().fit(df), [0.1, 0.9, 0.1, 0.1])
    o = run_episode(Serving(model), ctx, txs, 1, tm.rho)
    assert (o.n_ops, o.n_declines, o.blocked, o.decisions) == (2, 1, True, (False, True))
    assert o.passed_amount == txs[0].amt and o.ver == pytest.approx(txs[0].amt / ctx.episode.amount)
    assert len(ctx.state) == n0
    model = ListModel(BaseEncoder().fit(df), [0.1, 0.9, 0.1, 0.1])
    o = run_episode(Serving(model), ctx, txs, None, tm.rho)
    assert o.n_ops == 4 and o.decisions == (False, True, False, False)
    assert o.passed_amount == pytest.approx(txs[0].amt + txs[2].amt + txs[3].amt)


# ---- card-level escalation (tau_after_decline) -------------------------------------------------

def test_escalation_sequence_blocks_under_second_threshold():
    s = StubServing([0.3, 0.6, 0.35, 0.2])
    o = run_episode(s, _ctx(), _txs([1, 2, 3, 4]), 2, 0.8, tau_after_decline=0.3)
    assert o.decisions == (False, True, True)  # op3: p=0.35 >= 0.3 declined under tau2
    assert (o.n_ops, o.n_declines, o.blocked, o.escalated) == (3, 2, True, True)
    assert len(s.calls) == 3  # op4 not executed


def test_escalation_changes_later_decisions_only():
    o = run_episode(StubServing([0.6, 0.35, 0.2]), _ctx(), _txs([1, 2, 3]), None, 0.8, tau_after_decline=0.3)
    assert o.decisions == (True, True, False) and o.escalated
    o = run_episode(StubServing([0.3, 0.35]), _ctx(), _txs([1, 2]), None, 0.8, tau_after_decline=0.3)
    assert o.decisions == (False, False) and not o.escalated  # no decline yet: tau still applies


def test_escalation_null_identical_to_before():
    for ps in ([0, 1, 0, 1], [0.3, 0.6, 0.35, 0.2], [0, 0, 0]):
        a = run_episode(StubServing(ps), _ctx(), _txs([1, 2, 3, 4][:len(ps)]), 2, 0.8)
        b = run_episode(StubServing(ps), _ctx(), _txs([1, 2, 3, 4][:len(ps)]), 2, 0.8, tau_after_decline=None)
        assert a == b and not a.escalated


def test_escalation_validation():
    for bad in (0.5, 0.7, 0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            run_episode(StubServing([0]), _ctx(), _txs([1]), 1, 0.8, tau_after_decline=bad)


def test_escalation_is_per_episode():
    ctx = _ctx()
    a = run_episode(StubServing([0.6, 0.35]), ctx, _txs([1, 2]), None, 0.8, tau_after_decline=0.3)
    b = run_episode(StubServing([0.35, 0.35]), ctx, _txs([1, 2]), None, 0.8, tau_after_decline=0.3)
    assert a.escalated and b.decisions == (False, False) and not b.escalated
