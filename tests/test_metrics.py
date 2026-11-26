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


def test_uplift_ci_one_sided():
    from fraudaudit.metrics import uplift_se
    rng = np.random.default_rng(1)
    clean = rng.uniform(0, 1, 80)
    adv = clean + rng.normal(0.05, 0.3, 80)
    two = uplift_ci(adv, clean, 0.95, 2000, 3)
    one = uplift_ci(adv, clean, 0.95, 2000, 3, one_sided=True)
    assert one[0] == two[0]                    # same mean
    assert one[1] == two[1]                    # lower: the two-sided lower bound (documented)
    assert one[2] <= two[2] and one[2] < two[2]  # one-sided 95 % upper = 95th percentile < 97.5th
    d = adv - clean
    means = d[np.random.default_rng(3).integers(0, d.size, size=(2000, d.size))].mean(axis=1)
    assert one[2] == pytest.approx(float(np.quantile(means, 0.95)))
    assert uplift_se(adv, clean, 2000, 3) == pytest.approx(float(means.std(ddof=1)))
    assert uplift_ci(clean, clean, one_sided=True) == (0.0, 0.0, 0.0)
    assert uplift_se(clean, clean) == 0.0
    # the default is unchanged: two-sided
    assert uplift_ci(adv, clean, 0.95, 2000, 3) == two


# --- false blocks of legitimate cards ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from fraudaudit.metrics import false_block_rate  # noqa: E402
from fraudaudit.serving.core import Decision  # noqa: E402

_T0 = pd.Timestamp("2020-06-01 00:00:00")


class _StubServing:
    """amt encodes the position of the operation inside its window (1..25): p = 1 on every 10th, else 0.3."""
    tau = 0.5

    def decide(self, state, tx):
        p = 1.0 if int(round(tx.amt)) % 10 == 0 else 0.3
        return Decision(p=p, declined=p >= self.tau)


def _fb_df(n_cards=3, n_windows=2, per_window=25, with_fraud=True):
    rows, row = [], 0
    for c in range(n_cards):
        rows.append(dict(cc_num=c, ts=_T0 - pd.Timedelta(days=3), part="train", amt=1.0, is_fraud=0, row=row))
        row += 1
        for w in range(n_windows):
            for i in range(1, per_window + 1):
                ts = _T0 + pd.Timedelta(days=30 * w) + pd.Timedelta(hours=i)
                rows.append(dict(cc_num=c, ts=ts, part="test", amt=float(i), is_fraud=0, row=row))
                row += 1
        if with_fraud:  # a fraud operation of the test period must not be replayed
            rows.append(dict(cc_num=c, ts=_T0 + pd.Timedelta(hours=30), part="test", amt=10.0, is_fraud=1, row=row))
            row += 1
    df = pd.DataFrame(rows)
    df["merchant"], df["category"] = "fraud_m", "misc"
    df["merch_lat"], df["merch_long"] = 40.0, -75.0
    df["gender"], df["dob"] = "F", pd.Timestamp("1980-01-01")
    df["lat"], df["long"], df["city_pop"], df["state"] = 40.0, -75.0, 1000, "NC"
    return df.sort_values(["ts", "row"]).reset_index(drop=True)


def test_false_block_rate_stub():
    df = _fb_df()
    srv = _StubServing()
    r3 = false_block_rate(df, srv, 0.5, 3, n_jobs=1)
    assert r3.n_card_windows == 6 and r3.n_cards == 3 and r3.n_ops == 150
    assert r3.share_blocked_card_windows == 0.0 and r3.share_blocked_cards == 0.0
    assert r3.mean_declines_per_card_window == 2.0  # ops 10 and 20; the fraud row is not replayed
    assert r3.decline_rate == pytest.approx(2 / 25)
    r2 = false_block_rate(df, srv, 0.5, 2, n_jobs=1)
    assert r2.share_blocked_card_windows == 1.0 and r2.share_blocked_cards == 1.0
    assert r2.n_blocked_card_windows == 6
    assert r2.n_decided_ops == 6 * 20  # blocked at the 20th op: ops 21..25 are not decided


def test_false_block_rate_none_never_blocks():
    r = false_block_rate(_fb_df(), _StubServing(), 0.5, None, n_jobs=1)
    assert r.share_blocked_card_windows == 0.0 and r.n_blocked_card_windows == 0
    assert r.mean_declines_per_card_window == 2.0 and r.decline_rate == pytest.approx(2 / 25)
    assert r.max_declines is None


def test_false_block_rate_escalation():
    # after the first decline (op 10) the threshold drops to 0.2 < 0.3: every later op is declined
    r = false_block_rate(_fb_df(), _StubServing(), 0.5, 3, tau_after_decline=0.2, n_jobs=1)
    assert r.share_blocked_card_windows == 1.0  # declines 10, 11, 12 -> blocked at 3
    assert r.mean_declines_per_card_window == 3.0
    r_none = false_block_rate(_fb_df(), _StubServing(), 0.5, None, tau_after_decline=0.2, n_jobs=1)
    assert r_none.n_declines == 6 * 16 and r_none.share_blocked_card_windows == 0.0  # ops 10..25
    with pytest.raises(ValueError):
        false_block_rate(_fb_df(), _StubServing(), 0.5, 3, tau_after_decline=0.6, n_jobs=1)


def test_false_block_rate_windows_reset():
    df = _fb_df(n_cards=1, n_windows=3)
    r = false_block_rate(df, _StubServing(), 0.5, 2, n_jobs=1)
    assert r.n_card_windows == 3 and r.n_blocked_card_windows == 3 and r.n_cards == 1


@pytest.fixture(scope="module")
def real_model_dir(tmp_path_factory):
    from conftest import make_sparkov_like
    from fraudaudit.model.pipeline import train_model
    # 16 cards: with the 38-feature set a 1-epoch model on 12 cards saturates val scores (tau = inf)
    model, _ = train_model(make_sparkov_like(n_cards=16), seed=0, epochs=1)
    d = tmp_path_factory.mktemp("model")
    model.save(d, meta={"seed": 0})
    return d


def test_false_block_rate_identical_for_n_jobs(real_model_dir):
    from conftest import make_sparkov_like
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    df = make_sparkov_like(n_cards=16)
    model = FraudModel.from_artifacts(load_artifacts(real_model_dir))
    kw = dict(tau_after_decline=model.tau / 2, model_dir=real_model_dir)
    a = false_block_rate(df, model, model.tau, 2, n_jobs=1, **kw)
    b = false_block_rate(df, model, model.tau, 2, n_jobs=2, **kw)
    assert a == b and a.n_card_windows > 0


def test_false_blocks_line_and_attach():
    from fraudaudit.audit import false_blocks_line
    import dataclasses
    fb = dataclasses.asdict(false_block_rate(_fb_df(), _StubServing(), 0.5, 2, n_jobs=1))
    line = false_blocks_line(fb)
    assert "False blocks of legitimate cards: 100.0 % of card-months (6 of 6), 100.0 % of cards (3 of 3)" in line
    assert "no threshold" in line
