"""Metrics: Clopper-Pearson upper bound, the threshold for a target FPR, bootstrap intervals of VER
and of the paired uplift VER_adv − VER_clean.

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
    return _bootstrap_mean(v, confidence, n_boot, seed)


def _bootstrap_mean(v: np.ndarray, confidence: float, n_boot: int, seed: int) -> tuple[float, float, float]:
    idx = np.random.default_rng(seed).integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def uplift_ci(ver_adv: np.ndarray, ver_clean: np.ndarray, confidence: float = 0.95, n_boot: int = 2000,
              seed: int = 42) -> tuple[float, float, float]:
    """(mean, lower, upper) of the per-episode difference ``ver_adv_i − ver_clean_i``: a paired
    percentile bootstrap (the same resampled episodes for both legs, ``default_rng(seed)``).
    The gate rule «upper bound of VER_adv − VER_clean ≤ δ» uses ``upper``."""
    a = np.asarray(ver_adv, dtype=np.float64).reshape(-1)
    c = np.asarray(ver_clean, dtype=np.float64).reshape(-1)
    if a.shape != c.shape:
        raise ValueError(f"paired arrays differ in length: {a.size} vs {c.size}")
    if a.size == 0:
        raise ValueError("uplift_ci needs at least one episode")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return _bootstrap_mean(a - c, confidence, n_boot, seed)
