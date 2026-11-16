"""Attack protocol and the random-search baseline attack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..oracle import BudgetExhausted, ReconOracle
from ..policy import StepwisePolicy
from ..threat import Strategy, ThreatModel


@dataclass(frozen=True)
class AttackRun:
    """Result of an attack: the best candidate found – a ``Strategy`` or, for the stepwise attack, a
    ``StepwisePolicy`` (``None`` and ``nan`` when the budget did not allow a single evaluation) –, its fitness
    (mean ``ver`` on spare cards), evaluations and ops spent."""
    best: "Strategy | StepwisePolicy | None"
    best_fitness: float
    evals: int
    ops_used: int


@runtime_checkable
class Attack(Protocol):
    def run(self, oracle: ReconOracle, tm: ThreatModel, rng: np.random.Generator) -> AttackRun: ...


class RandomSearch:
    """Evaluate ``Strategy.random`` candidates until ``BudgetExhausted``; keep the best (first on ties).

    Candidates are drawn from ``rng`` one by one and evaluated in batches of ``batch_size``
    (default ``4 · oracle.n_jobs``) via ``oracle.evaluate_many``; the candidate sequence and the
    evaluations do not depend on the batch size, so the result is the same for any ``n_jobs``.
    The search stops at the first candidate whose worst case no longer fits into the budget.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        self.batch_size = batch_size

    def run(self, oracle: ReconOracle, tm: ThreatModel, rng: np.random.Generator) -> AttackRun:
        n_m = len(oracle.merchants)
        return random_search(oracle, lambda: Strategy.random(tm, n_m, rng).clip(tm, n_m),
                             self.batch_size or 4 * oracle.n_jobs)


def random_search(oracle: ReconOracle, draw, batch_size: int) -> AttackRun:
    """Evaluate candidates ``draw()`` (drawn one by one, in batches of ``batch_size`` via ``evaluate_many``)
    until ``BudgetExhausted``; keep the best (first on ties). Shared by ``RandomSearch`` and ``StepwiseAttack``."""
    evals0, ops0 = oracle.evals, oracle.ops_used
    best, best_fit = None, float("nan")

    def take(cands, fits):
        nonlocal best, best_fit
        for s, f in zip(cands, fits):
            if best is None or f > best_fit:
                best, best_fit = s, f

    while True:
        cands = [draw() for _ in range(batch_size)]
        try:
            take(cands, oracle.evaluate_many(cands))
        except BudgetExhausted as e:
            take(cands, e.fitness)
            break
    return AttackRun(best=best, best_fitness=best_fit, evals=oracle.evals - evals0,
                     ops_used=oracle.ops_used - ops0)
