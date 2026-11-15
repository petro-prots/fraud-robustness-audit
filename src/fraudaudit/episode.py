"""Execution of one fraud episode against the bank's decisions, with the bank's reaction.

The attacker sends the operations in order. Every attempt (passed or declined) enters the card
state, because the bank sees the attempt. After ``max_declines`` declines the card is blocked and
the remaining operations are not executed. ``ver`` is the share of the target sum A withdrawn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .serving.core import Serving
from .serving.state import Tx
from .threat import EpisodeContext


@dataclass(frozen=True)
class EpisodeOutcome:
    passed_amount: float
    ver: float
    n_ops: int
    n_declines: int
    blocked: bool
    success: bool
    decisions: tuple[bool, ...]  # declined flag of every executed operation, in order


def run_episode(serving: Serving, ctx: EpisodeContext, txs: list[Tx], max_declines: int | None,
                rho: float) -> EpisodeOutcome:
    """Run ``txs`` on a copy of ``ctx.state`` (the context is never changed).

    ``max_declines=None`` means the bank never blocks the card; otherwise it must be >= 1.
    ``success`` is ``passed_amount >= rho * A`` with A = ``ctx.episode.amount``, compared in whole
    cents so that float noise cannot flip it: ``round(passed*100) >= ceil(round(rho*A*100, 6))``.
    """
    if not ctx.episode.amount > 0:
        raise ValueError("episode amount must be > 0")
    if max_declines is not None and max_declines < 1:
        raise ValueError("max_declines must be None or >= 1")
    state = ctx.state.copy()
    target = float(ctx.episode.amount)
    passed = 0.0
    n_declines = 0
    blocked = False
    decisions: list[bool] = []
    for tx in txs:
        d = serving.decide(state, tx)
        decisions.append(d.declined)
        if d.declined:
            n_declines += 1
        else:
            passed += float(tx.amt)
        state.push(tx)
        if d.declined and max_declines is not None and n_declines >= max_declines:
            blocked = True
            break
    return EpisodeOutcome(passed_amount=passed, ver=passed / target, n_ops=len(decisions),
                          n_declines=n_declines, blocked=blocked, success=round(passed * 100) >= math.ceil(round(rho * target * 100, 6)),
                          decisions=tuple(decisions))
