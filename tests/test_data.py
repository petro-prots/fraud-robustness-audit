import pandas as pd

from fraudaudit.data import Episode, VAL_START, fraud_episodes, period, split_cards

from conftest import make_sparkov_like


def _frame(rows):
    """rows: (ts, part, card, amt, fraud)"""
    return pd.DataFrame(rows, columns=["ts", "part", "cc_num", "amt", "is_fraud"])


def test_period_boundaries():
    df = _frame([
        ("2020-03-20 23:59:00", "train", 1, 1.0, 0),
        ("2020-03-21 00:00:00", "train", 1, 1.0, 0),
        ("2020-07-01 00:00:00", "test", 1, 1.0, 0),
    ])
    df["ts"] = pd.to_datetime(df["ts"])
    assert VAL_START == pd.Timestamp("2020-03-21")
    assert period(df).tolist() == ["fit", "val", "test"]


def test_episode_split_by_gap():
    df = _frame([
        ("2020-01-01 10:00:00", "train", 7, 10.0, 1),
        ("2020-01-02 10:00:00", "train", 7, 20.0, 1),
        ("2020-01-12 10:00:00", "train", 7, 5.0, 1),
        ("2020-01-05 10:00:00", "train", 7, 99.0, 0),
    ])
    df["ts"] = pd.to_datetime(df["ts"])
    df["row"] = range(len(df))
    eps = fraud_episodes(df, {"fit"})
    assert len(eps) == 2
    assert [e.amount for e in eps] == [30.0, 5.0]
    assert eps[0].start == pd.Timestamp("2020-01-01 10:00:00")
    assert eps[0].rows == (0, 1)
    assert eps[1].rows == (2,)


def test_split_cards_disjoint(sparkov_like):
    eps = fraud_episodes(sparkov_like, {"fit", "val", "test"})
    assert len(eps) >= 2
    a, b = split_cards(eps, seed=42)
    assert {e.card for e in a}.isdisjoint({e.card for e in b})
    assert sorted(a + b, key=lambda e: (e.card, e.start)) == sorted(eps, key=lambda e: (e.card, e.start))
    assert a and b
    assert split_cards(eps, seed=42) == (a, b)


def test_sorted_by_ts_then_row(sparkov_like):
    df = sparkov_like
    key = list(zip(df["ts"], df["row"]))
    assert key == sorted(key)
    assert set(df["part"]) == {"train", "test"}
    same = df.groupby(["cc_num", "ts"]).size()
    assert (same >= 2).any()


def test_sparkov_like_deterministic():
    a, b = make_sparkov_like(), make_sparkov_like()
    pd.testing.assert_frame_equal(a, b)
