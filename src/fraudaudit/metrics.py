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


def _boot_means(v: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Means of ``n_boot`` resamples of ``v`` with replacement (``default_rng(seed)``)."""
    idx = np.random.default_rng(seed).integers(0, v.size, size=(n_boot, v.size))
    return v[idx].mean(axis=1)


def _bootstrap_mean(v: np.ndarray, confidence: float, n_boot: int, seed: int,
                    one_sided: bool = False) -> tuple[float, float, float]:
    means = _boot_means(v, n_boot, seed)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    if one_sided:
        hi = np.quantile(means, confidence)
    return float(v.mean()), float(lo), float(hi)


def _paired(ver_adv, ver_clean) -> np.ndarray:
    a = np.asarray(ver_adv, dtype=np.float64).reshape(-1)
    c = np.asarray(ver_clean, dtype=np.float64).reshape(-1)
    if a.shape != c.shape:
        raise ValueError(f"paired arrays differ in length: {a.size} vs {c.size}")
    if a.size == 0:
        raise ValueError("uplift_ci needs at least one episode")
    return a - c


def uplift_ci(ver_adv: np.ndarray, ver_clean: np.ndarray, confidence: float = 0.95, n_boot: int = 2000,
              seed: int = 42, *, one_sided: bool = False) -> tuple[float, float, float]:
    """(mean, lower, upper) of the per-episode difference ``ver_adv_i − ver_clean_i``: a paired
    percentile bootstrap (the same resampled episodes for both legs, ``default_rng(seed)``).

    Two-sided (default): ``lower``/``upper`` are the (1 − c)/2 and (1 + c)/2 percentiles. ``one_sided=True``
    (the gate, ruling R14: it only tests «uplift ≤ δ»): ``upper`` is the one-sided upper bound at level c,
    the c-th percentile (95th for c = 0.95), so it is never above the two-sided upper bound; ``lower`` stays
    the two-sided lower bound (descriptive only, not part of the one-sided statement, kept finite for JSON).
    The gate rule «upper bound of VER_adv − VER_clean ≤ δ» uses ``upper``."""
    d = _paired(ver_adv, ver_clean)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return _bootstrap_mean(d, confidence, n_boot, seed, one_sided)


def uplift_se(ver_adv: np.ndarray, ver_clean: np.ndarray, n_boot: int = 2000, seed: int = 42) -> float:
    """Bootstrap standard error of the mean uplift: the standard deviation (ddof=1) of the same resampled
    means ``uplift_ci`` uses (same ``seed``)."""
    d = _paired(ver_adv, ver_clean)
    return float(_boot_means(d, n_boot, seed).std(ddof=1))
