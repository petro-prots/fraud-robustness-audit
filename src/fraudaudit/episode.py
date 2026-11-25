"""Execution of one fraud episode against the bank's decisions, with the bank's reaction.

The attacker sends the operations in order. Every attempt (passed or declined) enters the card
state, because the bank sees the attempt. After ``max_declines`` declines the card is blocked and
the remaining operations are not executed. ``ver`` is the share of the target sum A withdrawn.

``run_episode`` executes a fixed operation list (a realised ``Strategy``); ``run_adaptive_episode``
executes a closed-loop policy that chooses every next operation after seeing the previous decision.
Both share the bank side (``_Bank``), and so do the legitimate replays of ``metrics`` (false blocks,
confirmation burden).

With a spending corridor (``corridor.CorridorConfig``, Task 22) the bank decides in this order: model decline →
decline; model approval within the corridor of both windows → auto-approve; else → owner confirmation, which the
owner passes (``owner=True``: friction, not auto-approved) and a fraudster fails (the operation does not pass; it
counts as a decline when ``confirm_fail_counts_as_decline``). ``passed_amount`` of an attacker episode is then the
auto-approved amount A_auto of the guarantee (``corridor.episode_bound``). Every operation that passed (auto-approved
or confirmed) enters the corridor profile, whoever sent it (ruling R26: no fraud labels); the profile is read with a
7-day lag, so it cannot change the corridor of an episode of at most 7 days. ``corridor_c7`` / ``corridor_c24`` record
the 7-day / 24 h corridor the bank used on each day (``corridor.check_used_corridor``); ``auto_ops`` and
``confirmed_ops`` the auto-approved and owner-confirmed operations (Proposition 2 per block; V2 post-hoc bound).
``scores``, ``model_declines`` and ``escalated_ops`` record per executed operation the score p of the bank's decision, the
model's decline at the active threshold and whether the threshold was already escalated (the trace of Task 27 chain 2,
``trace.py``); they never enter a result number.

The owner's operations inside the episode window (Task 28, ruling R54; ``owner_txs``): the card owner's real operations
are replayed interleaved with the episode's operations, by time – an owner operation with ``ts`` before the next episode
operation goes first, on equal ``ts`` the episode operation first. They pass through the same bank (model decision,
corridor; a confirmation is passed – the owner has the second factor) and enter the card state, but they are not part
of the episode: their amount never counts to ``passed_amount`` / ``ver``, they are not in ``decisions`` or the trace,
and an attacker policy is never told about them. ``owner_declines_count``: True – the owner's decline is an ordinary
decline (counts to ``max_declines``, starts the escalation); False – it neither counts nor escalates (the bank resolves
it with the owner), the declined attempt still enters the state. The episode ends with its last operation; owner
operations left when the card is blocked are ``owner_lost``. Without ``owner_txs`` nothing changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .policy import StepwisePolicy
from .serving.core import Serving
from .serving.state import Tx
from .threat import EpisodeContext, MerchantTable, ThreatModel, realize


@dataclass(frozen=True)
class EpisodeOutcome:
    passed_amount: float
    ver: float
    n_ops: int
    n_declines: int
    blocked: bool
    success: bool
    decisions: tuple[bool, ...]  # declined flag of every executed operation, in order
    escalated: bool = False  # the threshold was switched to tau_after_decline after the first decline
    n_confirms: int = 0  # operations sent to owner confirmation by the corridor
    corridor_c7: tuple = ()  # ((day ns, C_7d cents), …) the corridor checks used, by day in order
    corridor_c24: tuple = ()  # ((day ns, C_24h cents), …) likewise for the 24 h corridor
    auto_ops: tuple = ()  # ((t ns, cents), …) the auto-approved operations, in order
    confirmed_ops: tuple = ()  # ((t ns, cents), …) the owner-confirmed operations, in order
    # per executed operation, in order (the trace of Task 27 chain 2; never part of a result number):
    scores: tuple = ()  # p of the bank's decision (``serving.decide(...).p``), not a re-scoring
    model_declines: tuple = ()  # the model's decline at the active threshold (τ, after an escalation τ′)
    escalated_ops: tuple = ()  # the threshold was already escalated when this operation was decided
    # the owner's operations replayed inside the window (Task 28; all 0 without ``owner_txs``):
    owner_ops: int = 0  # executed owner operations
    owner_declined: int = 0  # of them declined by the model (whether or not the decline counted)
    owner_confirms: int = 0  # of them sent to (and passed) the owner confirmation by the corridor
    owner_lost: int = 0  # owner operations of the window not executed because the card was blocked
    blocked_by_owner: bool = False  # the decline that blocked the card was the owner's


def run_episode(serving: Serving, ctx: EpisodeContext, txs: list[Tx], max_declines: int | None,
                rho: float, *, tau_after_decline: float | None = None, corridor=None,
                owner_txs: list[Tx] | None = None, owner_declines_count: bool = True) -> EpisodeOutcome:
    """Run ``txs`` on a copy of ``ctx.state`` (the context is never changed).

    ``max_declines=None`` means the bank never blocks the card; otherwise it must be >= 1.
    ``success`` is ``passed_amount >= rho * A`` with A = ``ctx.episode.amount``, compared in whole
    cents so that float noise cannot flip it: ``round(passed*100) >= ceil(round(rho*A*100, 6))``.

    ``tau_after_decline`` (None = no escalation): after the FIRST decline on this card every later
    operation is declined iff ``p >= tau_after_decline`` (``p`` is the same single-row score as
    before, only the threshold changes). It must lie in (0, 1) and be below the model's ``tau`` when
    ``serving.tau`` is known (``ValueError`` otherwise). ``max_declines`` counts declines under
    whichever threshold was active.

    ``corridor`` (``corridor.CorridorConfig`` or None): the soft spending corridor (module docstring); the operations
    are the attacker's.

    ``owner_txs`` / ``owner_declines_count``: the owner's operations of the window interleaved (module docstring).
    """
    bank = _Bank(serving, ctx, max_declines, rho, tau_after_decline, corridor,
                 owner_declines_count=owner_declines_count)
    owner = _OwnerStream(owner_txs)
    for tx in txs:
        owner.before(bank, tx.ts)
        if bank.blocked:
            break
        bank.submit(tx)
        if bank.blocked:
            break
    owner.finish(bank)
    return bank.outcome()


class _OwnerStream:
    """The owner's operations of the window, fed to the bank before each episode operation (module docstring)."""

    def __init__(self, txs: list[Tx] | None) -> None:
        self.txs = list(txs or ())
        for a, b in zip(self.txs, self.txs[1:]):
            if pd.Timestamp(b.ts) < pd.Timestamp(a.ts):
                raise ValueError("owner operations must be in time order")
        self.i = 0

    def before(self, bank: "_Bank", ts) -> None:
        """Submit every not yet executed owner operation with ``ts`` strictly before ``ts`` (stops at a block)."""
        t = pd.Timestamp(ts)
        while self.i < len(self.txs) and not bank.blocked and pd.Timestamp(self.txs[self.i].ts) < t:
            bank.submit(self.txs[self.i], background=True)
            self.i += 1

    def finish(self, bank: "_Bank") -> None:
        """The episode ended: the owner operations left are lost if the card is blocked."""
        if bank.blocked:
            bank.owner_lost = len(self.txs) - self.i


class _Bank:
    """The bank's side of one episode, shared by ``run_episode`` and ``run_adaptive_episode``: every
    attempt enters the card state, declines are counted (with the ``tau_after_decline`` escalation)
    and the card is blocked after ``max_declines`` declines. With ``corridor`` the decision order of the module
    docstring applies; ``owner`` says whose operations these are (the owner passes a confirmation, a fraudster
    does not). Every operation that passed feeds the corridor profile (``CardState.push(passed=…)``)."""

    def __init__(self, serving, ctx: EpisodeContext, max_declines: int | None, rho: float,
                 tau_after_decline: float | None, corridor=None, *, owner: bool = False,
                 owner_declines_count: bool = True) -> None:
        if not ctx.episode.amount > 0:
            raise ValueError("episode amount must be > 0")
        if max_declines is not None and max_declines < 1:
            raise ValueError("max_declines must be None or >= 1")
        if tau_after_decline is not None:
            if not 0 < tau_after_decline < 1:
                raise ValueError("tau_after_decline must be None or in (0, 1)")
            tau = getattr(serving, "tau", None)
            if tau is not None and not tau_after_decline < tau:
                raise ValueError(f"tau_after_decline ({tau_after_decline}) must be < model tau ({tau})")
        if corridor is not None:
            corridor.check_ready()
        self.serving, self.max_declines, self.rho = serving, max_declines, rho
        self.tau_after_decline = tau_after_decline
        self.corridor, self.owner = corridor, bool(owner)
        self.n_confirms = 0
        self.c7_used: dict[int, int] = {}
        self.c24_used: dict[int, int] = {}
        self.auto_ops: list[tuple[int, int]] = []
        self.confirmed_ops: list[tuple[int, int]] = []
        self.state = ctx.state.copy()
        self.target = float(ctx.episode.amount)
        self.passed = 0.0
        self.n_declines = 0
        self.blocked = False
        self.escalated = False
        self.decisions: list[bool] = []
        self.scores: list[float] = []
        self.model_declines: list[bool] = []
        self.escalated_ops: list[bool] = []
        self.owner_declines_count = bool(owner_declines_count)
        self.owner_ops = self.owner_declined = self.owner_confirms = self.owner_lost = 0
        self.blocked_by_owner = False

    def submit(self, tx: Tx, *, owner: bool | None = None, background: bool = False) -> bool:
        """Decide ``tx`` and push it into the state; returns the «did not pass» flag (all the attacker sees; without
        a corridor it is the declined flag). ``owner`` overrides the bank's ``owner`` for this operation (episodes
        that interleave the owner's and the attacker's operations). ``background``: the owner's operation inside the
        window of another episode (module docstring, Task 28): decided and pushed as any operation, the owner passes a
        confirmation, but it is not part of the episode (no ``passed``, ``decisions``, trace or corridor lists) and
        its decline counts only with ``owner_declines_count``."""
        own = True if background else (self.owner if owner is None else bool(owner))
        if self.blocked:
            raise RuntimeError("the card is blocked")
        d = self.serving.decide(self.state, tx)
        declined = d.declined
        if self.escalated:
            declined = d.p >= self.tau_after_decline
        declined = bool(declined)
        if background:
            return self._submit_background(tx, declined)
        self.scores.append(float(d.p))
        self.model_declines.append(declined)
        self.escalated_ops.append(bool(self.escalated))
        auto = not declined
        confirm = False
        if auto and self.corridor is not None:
            from .corridor import DAY_NS, check
            ok, c24, c7 = check(self.corridor, self.state, tx.ts, tx.amt)
            t = int(pd.Timestamp(tx.ts).value)
            self.c7_used[t - t % DAY_NS] = c7  # constant per day: the same value on every check of the day
            self.c24_used[t - t % DAY_NS] = c24
            if not ok:
                auto, confirm = False, True
                self.n_confirms += 1
        passed = auto or (confirm and own)
        if confirm and not own and self.corridor.confirm_fail_counts_as_decline:
            declined = True  # the fraudster's failed confirmation counts as a decline
        self.decisions.append(not passed)
        if declined:
            self.n_declines += 1
        if passed:
            self.passed += float(tx.amt)
        if self.corridor is not None:
            t = int(pd.Timestamp(tx.ts).value)
            if auto:
                self.auto_ops.append((t, int(round(float(tx.amt) * 100))))
            elif confirm and own:
                self.confirmed_ops.append((t, int(round(float(tx.amt) * 100))))
        self.state.push(tx, auto_approved=auto, passed=passed, confirmed=confirm and own)
        if declined and self.tau_after_decline is not None:
            self.escalated = True
        if declined and self.max_declines is not None and self.n_declines >= self.max_declines:
            self.blocked = True
        return not passed

    def _submit_background(self, tx: Tx, declined: bool) -> bool:
        """The owner's operation of the window (``submit(background=True)``) after the model decision."""
        self.owner_ops += 1
        auto, confirm = not declined, False
        if auto and self.corridor is not None:
            from .corridor import DAY_NS, check
            ok, c24, c7 = check(self.corridor, self.state, tx.ts, tx.amt)
            t = int(pd.Timestamp(tx.ts).value)
            self.c7_used[t - t % DAY_NS] = c7
            self.c24_used[t - t % DAY_NS] = c24
            if not ok:
                auto, confirm = False, True
                self.owner_confirms += 1
        passed = auto or confirm  # the owner passes the confirmation
        self.state.push(tx, auto_approved=auto, passed=passed, confirmed=confirm)
        if declined:
            self.owner_declined += 1
            if self.owner_declines_count:
                self.n_declines += 1
                if self.tau_after_decline is not None:
                    self.escalated = True
                if self.max_declines is not None and self.n_declines >= self.max_declines:
                    self.blocked = True
                    self.blocked_by_owner = True
        return not passed

    def outcome(self) -> EpisodeOutcome:
        passed, target = self.passed, self.target
        return EpisodeOutcome(passed_amount=passed, ver=passed / target, n_ops=len(self.decisions),
                              n_declines=self.n_declines, blocked=self.blocked,
                              success=round(passed * 100) >= math.ceil(round(self.rho * target * 100, 6)),
                              decisions=tuple(self.decisions), escalated=self.escalated,
                              n_confirms=self.n_confirms, corridor_c7=tuple(sorted(self.c7_used.items())),
                              corridor_c24=tuple(sorted(self.c24_used.items())), auto_ops=tuple(self.auto_ops),
                              confirmed_ops=tuple(self.confirmed_ops), scores=tuple(self.scores),
                              model_declines=tuple(self.model_declines), escalated_ops=tuple(self.escalated_ops),
                              owner_ops=self.owner_ops, owner_declined=self.owner_declined,
                              owner_confirms=self.owner_confirms, owner_lost=self.owner_lost,
                              blocked_by_owner=self.blocked_by_owner)


def run_adaptive_episode(serving, ctx: EpisodeContext, policy, tm: ThreatModel, merchants: MerchantTable,
                         rng: np.random.Generator, *, tau_after_decline: float | None,
                         owner_txs: list[Tx] | None = None,
                         owner_declines_count: bool = True) -> tuple[EpisodeOutcome, list[Tx]]:
    """Closed-loop episode: ``policy`` decides every next operation AFTER seeing the decision on the
    previous one. Returns the outcome (as ``run_episode``) and the executed operations, in order.

    ``policy.act(ctx, tm, merchants, rng)`` is a generator: ``next`` gives the first operation, ``send(declined)``
    the next one, ``StopIteration`` ends the episode. Only the declined flag is sent (``knowledge: decision``;
    no score). The bank side is the same as in ``run_episode`` (shared ``_Bank``): every attempt enters the
    state, ``tm.max_declines`` blocks the card, ``tau_after_decline`` escalates the threshold after the first
    decline, success is ``>= tm.rho·A`` in whole cents. At most ``tm.k_max`` operations are executed. A policy
    that proposes a fixed list gives exactly ``run_episode`` on that list. ``tm.corridor`` is the spending corridor
    (as in ``run_episode``); the policy is sent the «did not pass» flag. ``owner_txs`` / ``owner_declines_count``: the
    owner's operations of the window interleaved (module docstring); the policy is told only about its own operations.
    """
    bank = _Bank(serving, ctx, tm.max_declines, tm.rho, tau_after_decline, getattr(tm, "corridor", None),
                 owner_declines_count=owner_declines_count)
    owner = _OwnerStream(owner_txs)
    executed: list[Tx] = []
    gen = policy.act(ctx, tm, merchants, rng)
    try:
        tx = next(gen, None)
        while tx is not None:
            owner.before(bank, tx.ts)
            if bank.blocked:
                break
            declined = bank.submit(tx)
            executed.append(tx)
            if bank.blocked or len(executed) >= tm.k_max:
                break
            try:
                tx = gen.send(declined)
            except StopIteration:
                tx = None
    finally:
        gen.close()
    owner.finish(bank)
    return bank.outcome(), executed


# --- candidates of the attacks: a fixed Strategy or a closed-loop StepwisePolicy -----------------------------


def clip_candidate(c, tm: ThreatModel, merchants: MerchantTable):
    """``c`` with every field in range (``Strategy.clip`` / ``StepwisePolicy.clip``); the replay candidate of the
    positive control has no fields."""
    from .attacks.replay import ReplayCandidate
    if isinstance(c, ReplayCandidate):
        return c
    if isinstance(c, StepwisePolicy):
        return c.clip(tm, merchants.n_categories)
    return c.clip(tm, len(merchants))


def candidate_max_ops(c, tm: ThreatModel) -> int:
    """Most operations one episode of ``c`` can execute: ``k`` of a strategy, ``k_max`` of a policy."""
    return int(tm.k_max) if isinstance(c, StepwisePolicy) else int(c.k)


def execute_candidate(serving, ctx: EpisodeContext, c, tm: ThreatModel, merchants: MerchantTable,
                      rng: np.random.Generator, *, owner_txs: list[Tx] | None = None,
                      owner_declines_count: bool = True) -> tuple[EpisodeOutcome, list[Tx]]:
    """One episode of candidate ``c`` on ``ctx`` with the bank's reaction of ``tm``; returns the outcome and
    the executed operations. A ``Strategy`` is realised in advance (``realize``) and run by ``run_episode``
    (the executed operations are the time-ordered prefix the bank saw); a ``StepwisePolicy`` runs closed-loop
    (``run_adaptive_episode``). ``rng`` is consumed by ``realize`` or by the policy. ``owner_txs`` /
    ``owner_declines_count``: the owner's operations of the window (module docstring); the realised operations of a
    ``Strategy`` do not depend on them."""
    if isinstance(c, StepwisePolicy):
        return run_adaptive_episode(serving, ctx, c, tm, merchants, rng, tau_after_decline=tm.tau_after_decline,
                                    owner_txs=owner_txs, owner_declines_count=owner_declines_count)
    txs = realize(c, ctx, tm, merchants, rng)
    o = run_episode(serving, ctx, txs, tm.max_declines, tm.rho, tau_after_decline=tm.tau_after_decline,
                    corridor=tm.corridor, owner_txs=owner_txs, owner_declines_count=owner_declines_count)
    return o, txs[:o.n_ops]
