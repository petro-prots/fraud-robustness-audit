"""Metrics: Clopper-Pearson upper bound and the threshold for a target FPR.

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
