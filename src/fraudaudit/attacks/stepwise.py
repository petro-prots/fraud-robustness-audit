"""Stepwise closed-loop attack: the held-out attack of the gate (design §8). It never generates hardening data
(``harden.check_generation_attack`` rejects it; it only measures in the val mini-audit), so it checks whether
robustness gained by adversarial retraining against ES and random search transfers to an attacker with a
different search principle.

With ``max_declines = 1`` (the realistic threat model) the first decline blocks the card, so the policy never
gets to react to a decline: it is effectively open-loop. Its strength comes from its low-dimensional structured
search space (geometric amounts, a fixed cadence, one merchant category: 9 parameters against the 82 of a
``Strategy``), not from adaptation to declines; the closed-loop parameters (``b``, ``switch_on_decline``) only
matter for ``max_declines > 1``.

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
