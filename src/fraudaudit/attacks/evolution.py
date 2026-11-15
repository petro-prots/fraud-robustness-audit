"""(μ+λ) evolution strategy: the main attack of the audit gate.

Same threat model and budget as ``RandomSearch``: every candidate is a ``Strategy`` evaluated by the
``ReconOracle`` (mean ``ver`` on spare cards), the run stops at ``BudgetExhausted``.

Algorithm:

1. initial population: ``mu`` ``Strategy.random`` candidates, evaluated with one ``evaluate_many``;
2. each generation: ``lam`` children, each a ``mutate`` of a parent drawn uniformly from the current
   ``mu`` parents; all children are evaluated with one ``evaluate_many`` call (parallel in the
   oracle's process pool);
3. selection: the best ``mu`` of parents ∪ children by fitness, ties broken by age (the earlier
   evaluated candidate first), so the result is deterministic and the best-so-far is never lost;
4. ``BudgetExhausted``: the children evaluated before the first one that did not fit still take part
   in the final selection; ``best`` is the best (oldest on ties) of everything ever evaluated.

Mutation (``mutate``), on the fields of ``Strategy.to_vector``:

* ``u``, ``shares``, ``day``, ``hour``: + N(0, (sigma · range)²) per entry, with the ranges of
  ``Strategy.random``: ``u`` ∈ [rho, 1], ``shares`` ∈ [0, 1], ``day`` ∈ [0, window_days),
  ``hour`` ∈ [0, 24); shares are also clipped to [0, 1] so the noise keeps its scale (shares are
  relative weights, the upper bound does not restrict the realised amounts; if all of the first
  ``k`` shares become 0, ``realize`` splits ``u·A`` equally). The time of day is circular:
  ``hour`` is wrapped modulo 24 (23:30 + 1 h is 00:30), not clipped, so the search does not pile
  up at 0 and 24;
* ``merchant[i]``: with probability ``p_merchant`` a uniform index in [0, n_merchants), otherwise
  ``rint(merchant[i] + N(0, (sigma · n_merchants)²))`` – a local move, as ``MerchantTable`` orders
  merchants by (category, name);
* ``k``: with probability ``p_k`` ± 1 (each sign with probability 1/2);
* then ``Strategy.clip``.

Randomness: everything comes from the ``rng`` given to ``run``, consumed in a fixed order (parent
choice, then a fixed block per child, whatever its values), and fitnesses do not depend on the
oracle's ``n_jobs``, so the run is identical for any ``n_jobs``.
"""
from __future__ import annotations

import numpy as np

from ..oracle import BudgetExhausted, ReconOracle
from ..threat import ESParams, Strategy, ThreatModel
from .random_search import AttackRun


class EvolutionStrategy:
    """(μ+λ) evolution strategy over relative episode strategies (see the module docstring)."""

    def __init__(self, mu: int = 5, lam: int = 10, sigma: float = 0.15, p_merchant: float = 0.1,
                 p_k: float = 0.2) -> None:
        p = ESParams(mu=mu, lam=lam, sigma=sigma, p_merchant=p_merchant, p_k=p_k)  # validates
        self.mu, self.lam, self.sigma, self.p_merchant, self.p_k = p.mu, p.lam, p.sigma, p.p_merchant, p.p_k

    @classmethod
    def from_params(cls, p: ESParams) -> "EvolutionStrategy":
        return cls(mu=p.mu, lam=p.lam, sigma=p.sigma, p_merchant=p.p_merchant, p_k=p.p_k)

    def mutate(self, parent: Strategy, tm: ThreatModel, n_merchants: int, rng: np.random.Generator) -> Strategy:
        """A clipped child of ``parent``. Always draws the same amount of randomness from ``rng``
        (``6·k_max + 3`` numbers), whatever ``parent`` is."""
        n, s = tm.k_max, self.sigma
        du = rng.normal(0.0, s * (1.0 - tm.rho))
        dshares = rng.normal(0.0, s * 1.0, n)
        dday = rng.normal(0.0, s * tm.window_days, n)
        dhour = rng.normal(0.0, s * 24.0, n)
        replace = rng.random(n) < self.p_merchant
        uniform_m = rng.integers(0, n_merchants, n)
        step_m = rng.normal(0.0, s * n_merchants, n)
        change_k = rng.random() < self.p_k
        sign_k = 1 if rng.random() < 0.5 else -1

        merchant = np.where(replace, uniform_m, np.rint(np.asarray(parent.merchant, dtype=np.float64) + step_m))
        child = Strategy(k=int(parent.k) + (sign_k if change_k else 0), u=float(parent.u) + du,
                         shares=np.clip(np.asarray(parent.shares, dtype=np.float64) + dshares, 0.0, 1.0),
                         day=np.asarray(parent.day, dtype=np.float64) + dday,
                         hour=np.mod(np.asarray(parent.hour, dtype=np.float64) + dhour, 24.0),
                         merchant=merchant)
        return child.clip(tm, n_merchants)

    def run(self, oracle: ReconOracle, tm: ThreatModel, rng: np.random.Generator) -> AttackRun:
        n_m = len(oracle.merchants)
        evals0, ops0 = oracle.evals, oracle.ops_used
        # population entries: (fitness, age, strategy); age = order of evaluation in this run
        pop: list[tuple[float, int, Strategy]] = []
        age = 0

        def evaluate(cands: list[Strategy]) -> tuple[list[tuple[float, int, Strategy]], bool]:
            nonlocal age
            try:
                fits, exhausted = oracle.evaluate_many(cands), False
            except BudgetExhausted as e:
                fits, exhausted = e.fitness, True
            out = [(float(f), age + i, s) for i, (s, f) in enumerate(zip(cands, fits))]
            age += len(out)
            return out, exhausted

        def select(entries):
            return sorted(entries, key=lambda e: (-e[0], e[1]))[:self.mu]

        init = [Strategy.random(tm, n_m, rng).clip(tm, n_m) for _ in range(self.mu)]
        evaluated, exhausted = evaluate(init)
        pop = select(evaluated)
        while not exhausted:
            parents = [pop[int(rng.integers(0, len(pop)))][2] for _ in range(self.lam)]
            children = [self.mutate(p, tm, n_m, rng) for p in parents]
            evaluated, exhausted = evaluate(children)
            pop = select(pop + evaluated)

        best_fit, _, best = pop[0] if pop else (float("nan"), 0, None)
        return AttackRun(best=best, best_fitness=best_fit, evals=oracle.evals - evals0,
                         ops_used=oracle.ops_used - ops0)
