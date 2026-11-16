"""Stepwise closed-loop attack: the held-out attack of the gate (design §8). It is never used in hardening
(``harden.check_generation_attack`` rejects it), so it checks whether robustness gained by adversarial
retraining against the evolution strategy transfers to an attacker with a different search principle.

The candidate is a ``StepwisePolicy`` (``fraudaudit.policy``): a parametric closed-loop policy that chooses
every next operation after seeing the bank's decision on the previous one (``knowledge: decision``). The
policy vector is tuned by random search on the reconnaissance oracle (``random_search``): policies drawn
with ``StepwisePolicy.random`` one by one from ``rng``, evaluated in batches (default ``4 · oracle.n_jobs``)
by ``oracle.evaluate_many`` – same spare cards, same budget accounting (worst case ``cards_per_eval · k_max``
reserved, executed operations charged), stop at ``BudgetExhausted``, best (first on ties) kept. The candidate
sequence and the evaluations do not depend on the batch size or ``n_jobs``.
"""
from __future__ import annotations

import numpy as np

from ..oracle import ReconOracle
from ..policy import StepwisePolicy
from ..threat import ThreatModel
from .random_search import AttackRun, random_search


class StepwiseAttack:
    """Random search over ``StepwisePolicy`` vectors (see the module docstring)."""

    held_out = True  # never generates hardening data

    def __init__(self, batch_size: int | None = None) -> None:
        self.batch_size = batch_size

    def run(self, oracle: ReconOracle, tm: ThreatModel, rng: np.random.Generator) -> AttackRun:
        n_cat = oracle.merchants.n_categories
        return random_search(oracle, lambda: StepwisePolicy.random(tm, n_cat, rng),
                             self.batch_size or 4 * oracle.n_jobs)
