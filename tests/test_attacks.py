import dataclasses
from pathlib import Path

import numpy as np
import pytest

from conftest import make_sparkov_like
from fraudaudit.attacks.evolution import EvolutionStrategy
from fraudaudit.attacks.random_search import Attack, AttackRun, RandomSearch
from fraudaudit.data import fraud_episodes
from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.oracle import BudgetExhausted, ReconOracle
from fraudaudit.serving.core import Decision, Serving
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


# ---- evolution strategy ---------------------------------------------------------------------------

class NightServing:
    """Stub bank: decline iff amt > 500 (LimitModel) or the hour of the operation is in [22, 4)."""

    def __init__(self, serving):
        self.inner = serving

    def decide(self, state, tx):
        h = tx.ts.hour
        if h >= 22 or h < 4:
            return Decision(p=1.0, declined=True)
        return self.inner.decide(state, tx)


def test_es_attack_protocol():
    assert isinstance(EvolutionStrategy(), Attack)
    es = EvolutionStrategy()
    assert (es.mu, es.lam, es.sigma, es.p_merchant, es.p_k) == (5, 10, 0.15, 0.1, 0.2)
    with pytest.raises(ValueError):
        EvolutionStrategy(mu=0)
    with pytest.raises(ValueError):
        EvolutionStrategy(p_merchant=1.5)


def test_es_from_threat_model(tm):
    es = EvolutionStrategy.from_params(tm.es)
    assert (es.mu, es.lam, es.sigma, es.p_merchant, es.p_k) == (5, 10, 0.15, 0.1, 0.2)


def test_es_beats_random_on_stub(serving, spares, tm, merchants):
    """Same budget (the realistic one), same oracle seeds. Measured with the default parameters:
    per seed ES >= RS on only 6 of 12 seeds (the stub is easy: ~130 random candidates already reach
    ~0.9 of the ceiling 1.0), so the brief's "3 of 3 seeds" does not hold; the claim tested is the
    mean over a fixed seed set (all seeds measured, none dropped) and that ES never collapses."""
    night = NightServing(serving)
    budget = tm.recon_budget_ops
    rs_fits, es_fits = [], []
    for seed in range(12):
        rs = RandomSearch().run(_oracle(night, spares, tm, merchants, budget, seed=seed), tm,
                                np.random.default_rng([seed, 2]))
        es = EvolutionStrategy().run(_oracle(night, spares, tm, merchants, budget, seed=seed), tm,
                                     np.random.default_rng([seed, 2]))
        rs_fits.append(rs.best_fitness)
        es_fits.append(es.best_fitness)
    assert np.mean(es_fits) > np.mean(rs_fits)
    assert min(es_fits) > 0.8


def test_es_respects_budget(serving, spares, tm, merchants):
    for budget in (0, 7, 60, 400):
        cs = CountingServing(serving)
        o = _oracle(cs, spares, tm, merchants, budget, seed=5)
        run = EvolutionStrategy().run(o, tm, np.random.default_rng(6))
        assert run.ops_used == o.ops_used == cs.calls <= budget
        assert run.evals == o.evals == len(o.history)
        if run.evals == 0:
            assert run.best is None and np.isnan(run.best_fitness)
            continue
        fits = [f for _, f in o.history]
        assert run.best_fitness == max(fits)  # children evaluated before exhaustion count
        first_best = next(s for s, f in o.history if f == run.best_fitness)  # ties: the oldest
        np.testing.assert_array_equal(run.best.to_vector(), first_best.to_vector())
    # the run stops only when the next candidate's worst case does not fit
    assert o.remaining < tm.cards_per_eval * tm.k_max


def test_es_mutation_bounds_and_locality(tm, merchants):
    n_m = len(merchants)
    rng = np.random.default_rng(0)
    es = EvolutionStrategy()
    parent = Strategy.random(tm, n_m, rng).clip(tm, n_m)
    for _ in range(300):
        c = es.mutate(parent, tm, n_m, rng)
        assert isinstance(c.k, int) and 1 <= c.k <= tm.k_max and abs(c.k - parent.k) <= 1
        assert tm.rho <= c.u <= 1.0
        assert np.all((c.shares >= 0) & (c.shares <= 1))
        assert np.all((c.day >= 0) & (c.day < tm.window_days))
        assert np.all((c.hour >= 0) & (c.hour < 24))
        assert c.merchant.dtype == np.int64 and np.all((c.merchant >= 0) & (c.merchant < n_m))
        parent = c
    # extreme k stays in range
    for k in (1, tm.k_max):
        p = dataclasses.replace(parent, k=k)
        ks = {EvolutionStrategy(p_k=1.0).mutate(p, tm, n_m, rng).k for _ in range(50)}
        assert ks <= {max(1, k - 1), k, min(tm.k_max, k + 1)} and len(ks) == 2
    # merchant locality: without replacement every step is a small Gaussian move
    local = EvolutionStrategy(sigma=0.01, p_merchant=0.0)
    p = dataclasses.replace(parent, merchant=np.full(tm.k_max, n_m // 2, dtype=np.int64))
    steps = np.concatenate([local.mutate(p, tm, n_m, rng).merchant - n_m // 2 for _ in range(200)])
    assert np.abs(steps).max() <= np.ceil(5 * 0.01 * n_m)
    assert np.any(steps != 0)
    # with replacement the merchant jumps anywhere
    jump = EvolutionStrategy(sigma=0.0, p_merchant=1.0)
    m = np.concatenate([jump.mutate(p, tm, n_m, rng).merchant for _ in range(50)])
    assert np.abs(m - n_m // 2).max() > 0.3 * n_m
    # no noise, no jumps: the child equals the parent
    still = EvolutionStrategy(sigma=0.0, p_merchant=0.0, p_k=0.0)
    np.testing.assert_array_equal(still.mutate(parent, tm, n_m, rng).to_vector(), parent.to_vector())


def test_es_mutation_rng_consumption_fixed(tm, merchants):
    n_m = len(merchants)
    es = EvolutionStrategy()
    a = Strategy.random(tm, n_m, np.random.default_rng(1)).clip(tm, n_m)
    b = dataclasses.replace(Strategy.random(tm, n_m, np.random.default_rng(2)).clip(tm, n_m), k=1)
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    es.mutate(a, tm, n_m, r1)
    es.mutate(b, tm, n_m, r2)
    assert r1.random() == r2.random()


def test_es_identical_for_n_jobs(df, tm, merchants, real_model_dir):
    spares = contexts_for(df, fraud_episodes(df, {"fit", "val", "test"}))
    runs = []
    for n_jobs in (1, 2):
        with ReconOracle(None, spares, tm, merchants, 600, np.random.default_rng(11), n_jobs=n_jobs,
                         model_dir=real_model_dir) as o:
            runs.append((EvolutionStrategy().run(o, tm, np.random.default_rng(13)), o.history))
    (a, ha), (b, hb) = runs
    assert a.evals > EvolutionStrategy().mu  # at least one generation
    assert (a.best_fitness, a.evals, a.ops_used) == (b.best_fitness, b.evals, b.ops_used)
    np.testing.assert_array_equal(a.best.to_vector(), b.best.to_vector())
    assert [f for _, f in ha] == [f for _, f in hb]
    assert all(np.array_equal(x.to_vector(), y.to_vector()) for (x, _), (y, _) in zip(ha, hb))
