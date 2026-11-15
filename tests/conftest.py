import numpy as np
import pandas as pd
import pytest

from fraudaudit.data import finalize

COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last",
    "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob",
    "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud",
]
from fraudaudit.features.columns import CATEGORIES
START = pd.Timestamp("2020-01-01")


def make_sparkov_like(n_cards: int = 40, days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Deterministic Sparkov-like set; same columns/dtypes as load_raw output."""
    rng = np.random.default_rng(seed)
    cut = START + pd.Timedelta(days=int(days * 0.8))  # train/test boundary
    rows = []
    states = ["NC", "SC", "TX", "NY", "CA"]
    home_lat = rng.uniform(30, 47, n_cards)
    home_long = rng.uniform(-120, -75, n_cards)
    home_zip = rng.integers(10000, 99999, n_cards)
    home_state = rng.choice(states, n_cards)
    dob = pd.Timestamp("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 50, n_cards), unit="D")
    for c in range(n_cards):
        card = 4_000_000_000_000_000 + c
        n = int(rng.integers(30, 60))
        secs = np.sort(rng.integers(0, days * 86400, size=n))
        secs[1] = secs[0]  # two ops of the same card in the same second
        for s in secs:
            rows.append((START + pd.Timedelta(seconds=int(s)), card, 0, c))
        if c % 4 == 0:  # fraud episode: ~6 ops within ~2 days
            t0 = int(rng.integers(5, days - 25)) * 86400 + int(rng.integers(0, 86400))
            for k in range(6):
                rows.append((START + pd.Timedelta(seconds=t0 + int(k * rng.integers(1000, 30000))), card, 1, c))
        if c % 8 == 0:  # second episode, >= 8 days after the first one ends (first lasts < 2 days)
            t1 = t0 + int(rng.integers(10, 20)) * 86400
            for k in range(4):
                rows.append((START + pd.Timedelta(seconds=t1 + k * 3600), card, 1, c))
    ts, card, fraud, ci = (list(x) for x in zip(*rows))
    ci = np.array(ci)
    n = len(ts)
    df = pd.DataFrame({
        "trans_date_trans_time": pd.Series(ts).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": np.array(card, dtype="int64"),
        "merchant": [f"fraud_Shop {i}" for i in rng.integers(0, 20, n)],
        "category": rng.choice(CATEGORIES, n),
        "amt": np.round(rng.lognormal(3.5, 1.0, n).clip(1.0, 1376.04), 2),
        "first": "Ann", "last": "Doe",
        "gender": rng.choice(["F", "M"], n),
        "street": "1 Main St", "city": "Town", "state": home_state[ci], "zip": home_zip[ci],
        "lat": home_lat[ci], "long": home_long[ci], "city_pop": rng.integers(100, 100000, n),
        "job": "Engineer", "dob": dob[ci].strftime("%Y-%m-%d"),
        "trans_num": [f"{i:032x}" for i in range(n)],
        "unix_time": (pd.Series(ts) - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64").to_numpy(),
        "merch_lat": home_lat[ci] + rng.uniform(-1, 1, n),
        "merch_long": home_long[ci] + rng.uniform(-1, 1, n),
        "is_fraud": np.array(fraud, dtype="int64"),
    })[COLUMNS]
    df["_ts"] = pd.to_datetime(df["trans_date_trans_time"])
    part = np.where(df["_ts"] >= cut, "test", "train")
    train = df[part == "train"].drop(columns="_ts").reset_index(drop=True)
    test = df[part == "test"].drop(columns="_ts").reset_index(drop=True)
    return finalize(train, test)


@pytest.fixture
def sparkov_like():
    return make_sparkov_like()
