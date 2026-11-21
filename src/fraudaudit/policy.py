"""Stepwise closed-loop attacker policy (the held-out attack of the gate, design §8).

The episode is a sequential decision problem (an MDP; strictly a POMDP, as the attacker does not see the
card state the bank sees). State: amount withdrawn so far, operations used, declines so far, time elapsed
and the outcome of the last operation. Action: the next operation (amount, delay, merchant). Transition:
the bank's decision. Reward: the amount that passed. ``StepwisePolicy`` is a small parametric policy for
this problem (not reinforcement learning); its parameters are relative, so a policy found on spare cards
transfers to any target card. The attacker knows only the decision (``knowledge: decision``): the policy
receives the declined flag of each of its operations and never the score ``p``.

Parameters (``POLICY_FIELDS``, vector layout of ``to_vector``) and their ranges (``bounds``):

* ``s0`` ∈ [0.02, 1] – the first amount as a share of ``amount_max``;
* ``g`` ∈ [1, 3] – amount multiplier after a passed operation;
* ``b`` ∈ [0.1, 0.9] – amount multiplier after a declined one (only matters when ``max_declines > 1``);
* ``hour`` ∈ [0, 24) and ``hour_jitter`` ∈ [0, 12] – the daily band of active hours
  ``[hour − hour_jitter, hour + hour_jitter]`` (circular; ``hour_jitter = 12`` is the whole day);
* ``gap_h`` ∈ [min_gap_secs / 3600, 48] – the delay between operations, in hours;
* ``cat_pref`` ∈ {0, …, n_categories − 1} – the preferred category block of ``MerchantTable``;
* ``switch_on_decline`` ∈ {0, 1} – move to another category after a decline;
* ``u`` ∈ [rho, 1] – the attacker stops once the withdrawn amount reaches ``u·A``.

``clip`` maps any vector into these ranges: NaN becomes the lower bound (``hour`` 0), ``hour`` is wrapped
modulo 24 (circular, not clipped), ``cat_pref`` and ``switch_on_decline`` are rounded to integers.

Behaviour (``act``, a generator that receives the declined flag of each proposed operation):

* time: the first operation is at the first second ≥ the episode start inside the active band; every next one
  at the first second ≥ previous + ``max(gap_h, min_gap_secs)`` inside the band (integer seconds); the policy
  stops when that time leaves the window ``[start, start + window_days)``;
* amount: a level starts at ``s0 · amount_max``, is multiplied by ``g`` after a pass and by ``b`` after a
  decline and kept within the amount bounds; the operation amount is ``min(level, u·A − withdrawn)`` clipped
  to ``[amount_min, amount_max]`` and rounded to cents. The policy stops when the withdrawn amount reaches
  ``u·A`` (compared in whole cents);
* merchant: a merchant drawn uniformly from the current category block; after a decline with
  ``switch_on_decline = 1`` the block changes to a uniformly drawn other one; merchant coordinates are
  home + U(−r, r) each, as in ``realize``;
* randomness: one ``(k_max, 4)`` block of uniforms is drawn from ``rng`` at the start (latitude, longitude,
  merchant pick, category switch of operation i), so ``rng`` consumption does not depend on the policy
  or on the decisions;
* every proposed operation passes the single-operation checks of the threat model (``threat._tx_ok``); one
  that does not is not proposed and the policy stops (cannot happen by construction, kept as a guard).

The bank's reaction (``max_declines``, ``tau_after_decline``) and the ``k_max`` limit are enforced by the
runner, ``episode.run_adaptive_episode``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from .serving.state import Tx
from .threat import MerchantTable, ThreatModel, _tx_ok

POLICY_FIELDS = ("s0", "g", "b", "hour", "hour_jitter", "gap_h", "cat_pref", "switch_on_decline", "u")
_DAY_S = 86_400
S0_RANGE = (0.02, 1.0)
G_RANGE = (1.0, 3.0)
B_RANGE = (0.1, 0.9)
JITTER_RANGE = (0.0, 12.0)
GAP_H_MAX = 48.0


@dataclass(frozen=True)
class StepwisePolicy:
    s0: float
    g: float
    b: float
    hour: float
    hour_jitter: float
    gap_h: float
    cat_pref: int
    switch_on_decline: int
    u: float

    @staticmethod
    def bounds(tm: ThreatModel, n_categories: int) -> dict[str, tuple[float, float]]:
        """Closed ranges of every field (``hour`` is half-open: [0, 24))."""
        return {"s0": S0_RANGE, "g": G_RANGE, "b": B_RANGE, "hour": (0.0, 24.0), "hour_jitter": JITTER_RANGE,
                "gap_h": (tm.min_gap_secs / 3600, GAP_H_MAX), "cat_pref": (0, n_categories - 1),
                "switch_on_decline": (0, 1), "u": (tm.rho, 1.0)}

    @classmethod
    def random(cls, tm: ThreatModel, n_categories: int, rng: np.random.Generator) -> "StepwisePolicy":
        """A uniform draw from the ranges; ``s0`` and ``gap_h`` are log-uniform (scale parameters).
        Always draws the same amount of randomness (9 numbers)."""
        gap_lo = max(tm.min_gap_secs / 3600, 1e-6)
        v = [math.exp(rng.uniform(math.log(S0_RANGE[0]), math.log(S0_RANGE[1]))),
             rng.uniform(*G_RANGE), rng.uniform(*B_RANGE), rng.uniform(0.0, 24.0), rng.uniform(*JITTER_RANGE),
             math.exp(rng.uniform(math.log(gap_lo), math.log(GAP_H_MAX))),
             rng.integers(0, n_categories), rng.integers(0, 2), rng.uniform(tm.rho, 1.0)]
        return cls.from_vector(np.asarray(v, dtype=np.float64), tm, n_categories)

    def clip(self, tm: ThreatModel, n_categories: int | None = None) -> "StepwisePolicy":
        """Every field in its range (module docstring); ``cat_pref`` only >= 0 when ``n_categories`` is None."""
        def c(x, lo, hi):
            x = float(x)
            return float(lo) if math.isnan(x) else float(min(max(x, lo), hi))
        h = float(self.hour)
        h = 0.0 if not math.isfinite(h) else float(np.mod(h, 24.0))
        if h >= 24.0:  # np.mod of a tiny negative number can round to 24.0
            h = 0.0
        cat = float(self.cat_pref)
        cat = 0 if math.isnan(cat) else int(np.clip(np.rint(cat), 0, np.inf if n_categories is None
                                                    else n_categories - 1))
        sw = float(self.switch_on_decline)
        sw = 0 if math.isnan(sw) else int(np.clip(np.rint(sw), 0, 1))
        return StepwisePolicy(s0=c(self.s0, *S0_RANGE), g=c(self.g, *G_RANGE), b=c(self.b, *B_RANGE), hour=h,
                              hour_jitter=c(self.hour_jitter, *JITTER_RANGE),
                              gap_h=c(self.gap_h, tm.min_gap_secs / 3600, GAP_H_MAX), cat_pref=cat,
                              switch_on_decline=sw, u=c(self.u, tm.rho, 1.0))

    def to_vector(self) -> np.ndarray:
        return np.array([float(getattr(self, f)) for f in POLICY_FIELDS], dtype=np.float64)

    @classmethod
    def from_vector(cls, v, tm: ThreatModel, n_categories: int | None = None) -> "StepwisePolicy":
        """Inverse of ``to_vector``, followed by ``clip`` (any real vector maps to a valid policy)."""
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (len(POLICY_FIELDS),):
            raise ValueError(f"expected a policy vector of length {len(POLICY_FIELDS)}, got {v.shape}")
        return cls(**{f: float(x) for f, x in zip(POLICY_FIELDS, v)}).clip(tm, n_categories)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def act(self, ctx, tm: ThreatModel, merchants: MerchantTable, rng: np.random.Generator):
        """Generator of the next operation; ``send`` it the declined flag of the previous one (module docstring)."""
        p = self.clip(tm, merchants.n_categories)
        draws = rng.uniform(0.0, 1.0, size=(tm.k_max, 4))
        n_cat = merchants.n_categories
        start = pd.Timestamp(ctx.episode.start)
        win = tm.window_days * _DAY_S
        lead = int((start - start.normalize()) / pd.Timedelta(seconds=1))
        band_lo = int(round((p.hour - p.hour_jitter) * 3600)) % _DAY_S
        band_w = int(round(2 * p.hour_jitter * 3600))
        gap = max(int(round(p.gap_h * 3600)), int(tm.min_gap_secs), 1)

        def in_band(x: int) -> int:  # first offset >= x whose time of day is inside the band
            if band_w >= _DAY_S:
                return x
            d = (lead + x - band_lo) % _DAY_S
            return x if d <= band_w else x + (_DAY_S - d)

        goal_c = math.ceil(round(p.u * float(ctx.episode.amount) * 100, 6))
        passed_c = 0
        level = p.s0 * tm.amount_max
        cat = p.cat_pref
        r = tm.merchant_coord_radius
        off = in_band(0)
        for i in range(tm.k_max):
            if off >= win:
                return
            rest = (goal_c - passed_c) / 100
            amt = float(np.round(np.clip(min(level, rest), tm.amount_min, tm.amount_max), 2))
            a, e = merchants.block(cat)
            j = a + min(int(draws[i, 2] * (e - a)), e - a - 1)
            tx = Tx(ts=start + pd.Timedelta(seconds=int(off)), amt=amt, merchant=merchants.names[j],
                    category=merchants.category[j], merch_lat=ctx.profile.lat + r * (2 * float(draws[i, 0]) - 1),
                    merch_long=ctx.profile.long + r * (2 * float(draws[i, 1]) - 1))
            if not _tx_ok(tx, ctx, tm, merchants):
                return
            declined = yield tx
            if declined:
                level *= p.b
                if p.switch_on_decline and n_cat > 1:
                    cat = (cat + 1 + min(int(draws[i, 3] * (n_cat - 1)), n_cat - 2)) % n_cat
            else:
                passed_c += int(round(amt * 100))
                level *= p.g
            level = min(max(level, tm.amount_min), tm.amount_max)
            if passed_c >= goal_c:
                return
            off = in_band(off + gap)
