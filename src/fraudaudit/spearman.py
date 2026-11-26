"""Exact permutation Spearman test (Task 24, brief §3, R33/R37): the unit is a model, n is small (8; 7 or 6 when a GRU
model is dropped, R37), so the p-value is the exact permutation p, never an asymptotic one.

``spearman_exact(x, y)``: ρ = Pearson correlation of the average ranks (ties handled), two-sided exact p = the share of
the n! permutations of ``y`` whose |ρ| is at least the observed |ρ| (the tied ranks of the data are permuted, so ties
are accounted for). ``critical_rho(n)`` = the smallest |ρ| with p < 0.05 for untied data: 0.738 for n = 8, 0.786 for
n = 7, 0.886 for n = 6 – the fixed fall-back thresholds of R37; the test runs for any n and prints n and the threshold.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

MAX_N = 9  # 9! = 362 880 permutations; the models of Task 24 are at most 8
_EPS = 1e-12


@dataclass(frozen=True)
class SpearmanResult:
    rho: float
    p: float  # exact two-sided permutation p
    n: int
    n_permutations: int
    critical_rho: float  # smallest |rho| with p < alpha for n untied points
    significant: bool  # p < alpha


def _perm_rhos(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """ρ of ``rx`` against every permutation of ``ry`` (ranks already)."""
    n = len(rx)
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.intp)
    xc = rx - rx.mean()
    yc = ry - ry.mean()
    return (yc[perms] @ xc) / np.sqrt((xc @ xc) * (yc @ yc))


def _check(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"x and y must be one-dimensional of the same length, got {x.shape} and {y.shape}")
    if not 3 <= len(x) <= MAX_N:
        raise ValueError(f"the exact test supports 3 to {MAX_N} points, got {len(x)}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must be finite")
    return x, y


def critical_rho(n: int, alpha: float = 0.05) -> float:
    """Smallest |ρ| for which the exact two-sided p of n untied points is below ``alpha``."""
    if not 3 <= n <= MAX_N:
        raise ValueError(f"the exact test supports 3 to {MAX_N} points, got {n}")
    r = np.arange(1, n + 1, dtype=np.float64)
    a = np.abs(_perm_rhos(r, r))
    for t in np.unique(a.round(12)):
        if (a >= t - _EPS).mean() < alpha:
            return float(t)
    return float("inf")  # no |rho| reaches alpha (n < 5 at alpha 0.05)


def spearman_exact(x, y, alpha: float = 0.05) -> SpearmanResult:
    x, y = _check(x, y)
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        raise ValueError("a constant input has no rank correlation")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rhos = _perm_rhos(rx, ry)
    p = float((np.abs(rhos) >= abs(rho) - _EPS).mean())
    return SpearmanResult(rho=rho, p=p, n=len(x), n_permutations=len(rhos), critical_rho=critical_rho(len(x), alpha),
                          significant=p < alpha)


def leave_one_out(x, y) -> list[float]:
    """ρ with each point left out in turn (the range printed next to the test, brief §3)."""
    x, y = _check(x, y)
    out = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        rx, ry = rankdata(x[keep]), rankdata(y[keep])
        out.append(float("nan") if np.ptp(rx) == 0 or np.ptp(ry) == 0 else float(np.corrcoef(rx, ry)[0, 1]))
    return out
