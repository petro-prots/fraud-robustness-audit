import numpy as np
import pytest

from fraudaudit.metrics import clopper_pearson_upper, threshold_at_fpr, uplift_ci, ver_ci


def test_cp_upper():
    assert clopper_pearson_upper(0, 100) == pytest.approx(0.0295, abs=5e-5)
    assert clopper_pearson_upper(0, 300) == pytest.approx(1 - 0.05 ** (1 / 300), rel=1e-9)
    assert clopper_pearson_upper(300, 300) == 1.0
    assert 14 / 300 < clopper_pearson_upper(14, 300) < 0.08
    assert clopper_pearson_upper(0, 0) == 1.0


def test_threshold_at_fpr():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(10_000, dtype=int), np.ones(500, dtype=int)]
    p = np.r_[rng.uniform(0, 1, 10_000), rng.uniform(0.5, 1, 500)]
    tau = threshold_at_fpr(y, p, 0.01)
    pred = p >= tau
    fpr = pred[y == 0].mean()
    assert fpr <= 0.01
    # best recall among all thresholds that keep FPR within the target (brute force)
    best = max((p[y == 1] >= t).mean() for t in np.unique(p) if (p[y == 0] >= t).mean() <= 0.01)
    assert pred[y == 1].mean() == best

def test_threshold_at_fpr_separable():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    tau = threshold_at_fpr(y, p, 0.01)
    assert tau == pytest.approx(0.8)


def test_uplift_ci_paired():
    rng = np.random.default_rng(0)
    clean = rng.uniform(0, 1, 50)
    assert uplift_ci(clean, clean) == (0.0, 0.0, 0.0)
    assert uplift_ci(clean + 0.25, clean) == pytest.approx((0.25, 0.25, 0.25))
    adv = clean + rng.normal(0.1, 0.05, 50)
    mean, lo, hi = uplift_ci(adv, clean)
    assert mean == pytest.approx((adv - clean).mean())
    assert lo < mean < hi
    # pairing: much narrower than the difference of two unpaired intervals
    a, c = ver_ci(adv), ver_ci(clean)
    assert hi - lo < (a[2] - a[1]) + (c[2] - c[1])
    assert uplift_ci(adv, clean) == uplift_ci(adv, clean)
    with pytest.raises(ValueError):
        uplift_ci(adv, clean[:-1])
    with pytest.raises(ValueError):
        uplift_ci(np.array([]), np.array([]))
