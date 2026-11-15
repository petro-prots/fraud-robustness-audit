import dataclasses
from pathlib import Path

import numpy as np
import pytest

from conftest import make_sparkov_like
from fraudaudit.attacks.random_search import Attack, AttackRun, RandomSearch
from fraudaudit.data import fraud_episodes
from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.oracle import BudgetExhausted, ReconOracle
from fraudaudit.serving.core import Serving
from fraudaudit.threat import MerchantTable, Strategy, ThreatModel, contexts_for, realize

TM_PATH = Path(__file__).resolve().parents[1] / "threat-models" / "realistic.yaml"
AMT = FEATURE_COLUMNS.index("amt")
A_SPARE = 3000.0  # spare episodes with a large target: one big operation is always declined


class LimitModel:
    """Duck-typed model: decline iff amt > 500."""

    def __init__(self, encoder):
        self.encoder, self.tau = encoder, 0.5

    def predict_proba(self, F):
        return (np.asarray(F)[:, AMT] > 500).astype(np.float64)


class CountingServing:
    """Serving proxy counting every decision (= every executed operation)."""

    def __init__(self, serving):
        self.inner, self.calls = serving, 0

    def decide(self, state, tx):
        self.calls += 1
        return self.inner.decide(state, tx)


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
def spares(df):
    eps = [dataclasses.replace(e, amount=A_SPARE) for e in fraud_episodes(df, {"fit", "val", "test"})]
    return contexts_for(df, eps)


@pytest.fixture(scope="module")
def serving(df):
    return Serving(LimitModel(BaseEncoder().fit(df)))


def _oracle(serving, spares, tm, merchants, budget, seed=0, **kw):
    return ReconOracle(serving, spares, tm, merchants, budget, np.random.default_rng(seed), **kw)


def _strategy(tm, k, u=1.0):
    n = tm.k_max
    return Strategy(k=k, u=u, shares=np.ones(n), day=np.linspace(0, 6, n), hour=np.full(n, 12.0),
                    merchant=np.zeros(n, dtype=np.int64))


def test_budget_accounting(serving, spares, tm, merchants):
    cs = CountingServing(serving)
    o = _oracle(cs, spares, tm, merchants, budget=200)
    assert o.remaining == 200 and o.ops_used == 0
    # k=2 on A=3000: both amounts 1376.04 -> the first one is declined, the card is blocked after 1 op
    f = o.evaluate(_strategy(tm, 2))
    assert f == 0.0
    assert o.ops_used == cs.calls == tm.cards_per_eval  # not-executed ops are not charged
    # k=10 on A=3000: 300 each, never declined -> every op executed, ver = 1
    f = o.evaluate(_strategy(tm, 10))
    assert f == pytest.approx(1.0)
    assert o.ops_used == cs.calls == tm.cards_per_eval * 11
    rng = np.random.default_rng(5)
    while True:
        try:
            o.evaluate(Strategy.random(tm, len(merchants), rng))
        except BudgetExhausted:
            break
        assert o.ops_used == cs.calls <= 200
    assert o.ops_used == cs.calls <= 200
    assert o.remaining == 200 - o.ops_used
    assert o.evals == len(o.history)


def test_evaluate_raises_before_overspending(serving, spares, tm, merchants):
    cs = CountingServing(serving)
    o = _oracle(cs, spares, tm, merchants, budget=tm.cards_per_eval * 4 - 1)
    with pytest.raises(BudgetExhausted):
        o.evaluate(_strategy(tm, 4))  # worst case 5*4 = 20 > 19
    assert (o.ops_used, cs.calls, o.evals) == (0, 0, 0)
    o.evaluate(_strategy(tm, 3))  # worst case 15 fits; 800 each -> blocked after the first op
    assert o.ops_used == cs.calls == tm.cards_per_eval


def test_budget_exhausted_returns_best_so_far(serving, spares, tm, merchants):
    o = _oracle(serving, spares, tm, merchants, budget=150)
    run = RandomSearch().run(o, tm, np.random.default_rng(1))
    assert isinstance(run, AttackRun)
    assert run.evals >= 1 and run.evals == o.evals
    assert run.ops_used == o.ops_used <= 150
    fits = [f for _, f in o.history]
    assert run.best_fitness == max(fits)
    first_best = next(s for s, f in o.history if f == run.best_fitness)  # ties: the first one
    np.testing.assert_array_equal(run.best.to_vector(), first_best.to_vector())


def test_random_search_learns_small_amounts(serving, spares, tm, merchants):
    o = _oracle(serving, spares, tm, merchants, budget=3000)
    run = RandomSearch().run(o, tm, np.random.default_rng(2))
    assert run.best_fitness > 0.5
    rng = np.random.default_rng(0)
    for ctx in spares:
        assert all(tx.amt <= 500 for tx in realize(run.best, ctx, tm, merchants, rng))


def test_evaluate_many_matches_sequential(serving, spares, tm, merchants):
    rng = np.random.default_rng(9)
    cands = [Strategy.random(tm, len(merchants), rng) for _ in range(40)]
    seq = _oracle(serving, spares, tm, merchants, budget=400, seed=3)
    want = []
    with pytest.raises(BudgetExhausted):
        for s in cands:
            want.append(seq.evaluate(s))
    bat = _oracle(serving, spares, tm, merchants, budget=400, seed=3)
    got = []
    with pytest.raises(BudgetExhausted) as e:
        for i in range(0, len(cands), 7):
            got += bat.evaluate_many(cands[i:i + 7])
    got += e.value.fitness
    assert got == want
    assert (bat.ops_used, bat.evals) == (seq.ops_used, seq.evals)


def test_spare_choice_deterministic(serving, spares, tm, merchants):
    s = _strategy(tm, 6, u=0.9)
    a = _oracle(serving, spares, tm, merchants, budget=500, seed=4)
    b = _oracle(serving, spares, tm, merchants, budget=500, seed=4)
    assert [a.evaluate(s) for _ in range(3)] == [b.evaluate(s) for _ in range(3)]
    with pytest.raises(ValueError, match="spare"):
        _oracle(serving, spares[:tm.cards_per_eval - 1], tm, merchants, budget=10)


def test_attack_protocol():
    assert isinstance(RandomSearch(), Attack)


@pytest.fixture(scope="module")
def real_model_dir(df, tmp_path_factory):
    from fraudaudit.model.pipeline import train_model
    model, _ = train_model(df, seed=0, epochs=1)
    d = tmp_path_factory.mktemp("model")
    model.save(d, meta={"seed": 0})
    return d


def test_random_search_identical_for_n_jobs(df, tm, merchants, real_model_dir):
    spares = contexts_for(df, fraud_episodes(df, {"fit", "val", "test"}))
    runs = []
    for n_jobs in (1, 2):
        with ReconOracle(None, spares, tm, merchants, 400, np.random.default_rng(11), n_jobs=n_jobs,
                         model_dir=real_model_dir) as o:
            runs.append((RandomSearch().run(o, tm, np.random.default_rng(12)), o.history))
    (a, ha), (b, hb) = runs
    assert a.evals >= 4  # several candidates per pool round
    assert (a.best_fitness, a.evals, a.ops_used) == (b.best_fitness, b.evals, b.ops_used)
    np.testing.assert_array_equal(a.best.to_vector(), b.best.to_vector())
    assert [f for _, f in ha] == [f for _, f in hb]
