"""Reconnaissance oracle: the attacker tries strategies on spare compromised cards under a budget.

Realistic profile: the attacker sees only which of his operations passed or were declined (so he
knows how much money passed). Every probe is a real, stateful attempt on a spare card: each
evaluation runs the strategy on ``tm.cards_per_eval`` distinct spare cards, each from its fresh
pre-episode state, with the bank's reaction (``max_declines``). A blocked card burns: its remaining
operations are not executed.

Budget (in operations): every executed operation (up to and including the blocking decline) is
charged; operations not executed after blocking are not. Before an evaluation the oracle checks
the worst case ``cards_per_eval · k`` against ``remaining`` and raises ``BudgetExhausted`` without
executing anything if it does not fit, so ``ops_used <= budget_ops`` always and an evaluation is
never cut in the middle.

Candidates are fixed ``Strategy`` plans or closed-loop ``StepwisePolicy`` objects (the held-out stepwise
attack): a policy's worst case is ``cards_per_eval · k_max`` (it decides how many operations to make
while it runs), and only its actually executed operations are charged.

Determinism: evaluation number ``i`` (0, 1, ... over the oracle's lifetime) picks its spare cards
with ``default_rng([seed, i])`` and realizes the strategy on card number ``j`` of that pick with
``default_rng([seed, i, j])``; ``seed`` is drawn once from the ``rng`` given to the constructor.
Results therefore do not depend on process scheduling: ``evaluate_many`` in a process pool
(``n_jobs > 1``, each worker loads its own model from ``model_dir``) gives exactly what
sequential ``evaluate`` calls give.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .episode import candidate_max_ops, clip_candidate, execute_candidate
from .serving.core import Serving
from .threat import EpisodeContext, MerchantTable, Strategy, ThreatModel


class BudgetExhausted(Exception):
    """The next evaluation does not fit into the remaining budget (worst case).

    ``fitness`` holds the fitness of the strategies of the same ``evaluate_many`` call that were
    evaluated before the one that did not fit (empty for ``evaluate``)."""

    def __init__(self, msg: str, fitness: list[float] | None = None) -> None:
        super().__init__(msg)
        self.fitness = list(fitness or [])


def _evaluate_cards(serving, spares: list[EpisodeContext], tm: ThreatModel, merchants: MerchantTable,
                    seed: int, eval_idx: int, strategy) -> list[tuple[float, int, bool]]:
    """(ver, executed ops, blocked) on every spare card of evaluation ``eval_idx`` (a ``Strategy`` or a
    closed-loop ``StepwisePolicy``, see ``episode.execute_candidate``)."""
    pick = np.random.default_rng([seed, eval_idx]).choice(len(spares), size=tm.cards_per_eval, replace=False)
    out = []
    for j, c in enumerate(pick):
        o, _ = execute_candidate(serving, spares[int(c)], strategy, tm, merchants,
                                 np.random.default_rng([seed, eval_idx, j]))
        out.append((o.ver, o.n_ops, o.blocked))
    return out


_WORKER: dict = {}  # per worker process: set once by _init_worker


def _load_serving(model_dir) -> Serving:
    from .model.artifacts import load_artifacts
    from .model.pipeline import FraudModel
    return Serving(FraudModel.from_artifacts(load_artifacts(Path(model_dir))))


def _init_worker(model_dir, spares, tm, merchants, seed) -> None:
    _WORKER.update(serving=_load_serving(model_dir), spares=spares, tm=tm, merchants=merchants, seed=seed)


def _eval_job(job: tuple[int, Strategy]):
    i, s = job
    w = _WORKER
    return _evaluate_cards(w["serving"], w["spares"], w["tm"], w["merchants"], w["seed"], i, s)


class ReconOracle:
    """Fitness of strategies on spare cards (mean ``ver``), charged against ``budget_ops``.

    ``serving`` is used in-process (stub models in tests); it may be ``None`` when ``model_dir`` is
    given (the model is then loaded from there). ``n_jobs > 1`` needs ``model_dir``: every worker
    loads its own model, as Keras models are not picklable. ``n_jobs=None`` means ``os.cpu_count()``
    with ``model_dir`` and 1 without. Use as a context manager (or call ``close``) to stop the pool.
    """

    def __init__(self, serving, spares: list[EpisodeContext], tm: ThreatModel, merchants: MerchantTable,
                 budget_ops: int, rng: np.random.Generator, *, n_jobs: int | None = None,
                 model_dir: Path | str | None = None) -> None:
        if len(spares) < tm.cards_per_eval:
            raise ValueError(f"{len(spares)} spare cards, need at least cards_per_eval={tm.cards_per_eval}")
        if budget_ops < 0:
            raise ValueError("budget_ops must be >= 0")
        if serving is None and model_dir is None:
            raise ValueError("need serving or model_dir")
        jobs = n_jobs if n_jobs is not None else ((os.cpu_count() or 1) if model_dir is not None else 1)
        if jobs > 1 and model_dir is None:
            raise ValueError("n_jobs > 1 needs model_dir: every worker loads its own model")
        self.spares, self.tm, self.merchants = list(spares), tm, merchants
        self.budget_ops = int(budget_ops)
        self.n_jobs = max(1, int(jobs))
        self.model_dir = None if model_dir is None else str(model_dir)
        self.seed = int(rng.integers(0, 2**63))
        self.ops_used = 0
        self.evals = 0
        self.cards_blocked = 0
        self.history: list[tuple[Strategy, float]] = []
        self._serving = serving
        self._pool: ProcessPoolExecutor | None = None

    @property
    def remaining(self) -> int:
        return self.budget_ops - self.ops_used

    def _worst(self, s) -> int:
        return self.tm.cards_per_eval * candidate_max_ops(s, self.tm)

    def _serving_local(self):
        if self._serving is None:
            self._serving = _load_serving(self.model_dir)
        return self._serving

    def _commit(self, s: Strategy, cards: list[tuple[float, int, bool]]) -> float:
        ops = sum(n for _, n, _ in cards)
        assert ops <= self._worst(s) <= self.remaining
        self.ops_used += ops
        self.evals += 1
        self.cards_blocked += sum(b for _, _, b in cards)
        fit = float(np.mean([v for v, _, _ in cards]))
        self.history.append((s, fit))
        return fit

    def evaluate(self, strategy: Strategy) -> float:
        """Mean ``ver`` of ``strategy`` on ``cards_per_eval`` spare cards; ``BudgetExhausted``
        (nothing executed, nothing charged) if the worst case does not fit."""
        s = clip_candidate(strategy, self.tm, self.merchants)
        if self._worst(s) > self.remaining:
            raise BudgetExhausted(f"worst case {self._worst(s)} ops > remaining {self.remaining}")
        cards = _evaluate_cards(self._serving_local(), self.spares, self.tm, self.merchants, self.seed,
                                self.evals, s)
        return self._commit(s, cards)

    def evaluate_many(self, strategies: list[Strategy]) -> list[float]:
        """``[evaluate(s) for s in strategies]``, with the evaluations in a process pool.

        Identical to the sequential calls: each round reserves the worst case of the candidates in
        list order against ``remaining`` and runs the prefix that fits in parallel, then charges the
        executed operations in list order; the next round starts from the actual ``remaining``.
        When the next candidate does not fit even alone, ``BudgetExhausted`` is raised with the
        fitness of the candidates evaluated so far in ``.fitness``."""
        todo = [clip_candidate(s, self.tm, self.merchants) for s in strategies]
        done: list[float] = []
        while todo:
            room, n = self.remaining, 0
            while n < len(todo) and self._worst(todo[n]) <= room:
                room -= self._worst(todo[n])
                n += 1
            if n == 0:
                raise BudgetExhausted(f"worst case {self._worst(todo[0])} ops > remaining {self.remaining}",
                                      fitness=done)
            batch, todo = todo[:n], todo[n:]
            jobs = [(self.evals + i, s) for i, s in enumerate(batch)]
            if self.n_jobs > 1 and n > 1:
                results = list(self._get_pool().map(_eval_job, jobs))
            else:
                serving = self._serving_local()
                results = [_evaluate_cards(serving, self.spares, self.tm, self.merchants, self.seed, i, s)
                           for i, s in jobs]
            done += [self._commit(s, r) for s, r in zip(batch, results)]
        return done

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_jobs, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                initargs=(self.model_dir, self.spares, self.tm, self.merchants, self.seed))
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __enter__(self) -> "ReconOracle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
