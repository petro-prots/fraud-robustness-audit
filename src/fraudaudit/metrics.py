"""Metrics: Clopper-Pearson upper bound, the threshold for a target FPR, bootstrap interval of VER.

``clopper_pearson_upper`` and ``threshold_at_fpr`` are ported unchanged from fraudgate
(``src/fraudgate/metrics.py``).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_curve


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided upper confidence bound for the proportion k/n."""
    if n == 0:
        return 1.0
    return 1.0 if k == n else float(beta.ppf(confidence, k + 1, n - k))


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold tau with FPR (decision ``p >= tau``) not above target_fpr."""
    fpr, _, thr = roc_curve(y, p)
    return float(thr[np.flatnonzero(fpr <= target_fpr)[-1]])


def ver_ci(vers: np.ndarray, confidence: float = 0.95, n_boot: int = 2000,
           seed: int = 42) -> tuple[float, float, float]:
    """(mean, lower, upper): mean VER over episodes and its two-sided percentile bootstrap
    interval. Episodes are resampled with replacement ``n_boot`` times (``default_rng(seed)``)."""
    v = np.asarray(vers, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("ver_ci needs at least one episode")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    idx = np.random.default_rng(seed).integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)
