import dataclasses
from types import SimpleNamespace

import pandas as pd
import pytest

from fraudaudit.episode import EpisodeOutcome, run_episode
from fraudaudit.serving.core import Decision
from fraudaudit.serving.state import CardProfile, CardState, Tx

T0 = pd.Timestamp("2020-02-01 10:00:00")


class StubServing:
    """Serving-compatible: p from a list in call order; declined iff p >= 0.5."""

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
