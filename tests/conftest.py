import numpy as np
import pandas as pd
import pytest

from fraudaudit.data import finalize

COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last",
    "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob",
    "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud",
]
CATEGORIES = ["grocery_pos", "gas_transport", "shopping_net", "misc_net", "food_dining", "travel"]
START = pd.Timestamp("2020-01-01")


def make_sparkov_like(n_cards: int = 40, days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Deterministic Sparkov-like set; same columns/dtypes as load_raw output."""
    rng = np.random.default_rng(seed)
    cut = START + pd.Timedelta(days=int(days * 0.8))  # train/test boundary
    rows = []
    for c in range(n_cards):
        card = 4_000_000_000_000_000 + c
        n = int(rng.integers(30, 60))
        secs = np.sort(rng.integers(0, days * 86400, size=n))
        secs[1] = secs[0]  # two ops of the same card in the same second
        for s in secs:
            rows.append((START + pd.Timedelta(seconds=int(s)), card, 0))
        if c % 4 == 0:  # fraud episode: ~6 ops within ~2 days
            t0 = int(rng.integers(5, days - 10)) * 86400 + int(rng.integers(0, 86400))
            for k in range(6):
                rows.append((START + pd.Timedelta(seconds=t0 + int(k * rng.integers(1000, 30000))), card, 1))
        if c % 8 == 0:  # second episode, > 7 days later
            t1 = int(rng.integers(5, days - 10)) * 86400
            for k in range(4):
                rows.append((START + pd.Timedelta(seconds=t1 + k * 3600), card, 1))
    ts, card, fraud = (list(x) for x in zip(*rows))
    n = len(ts)
    df = pd.DataFrame({
        "trans_date_trans_time": pd.Series(ts).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": np.array(card, dtype="int64"),
        "merchant": [f"fraud_Shop {i}" for i in rng.integers(0, 20, n)],
        "category": rng.choice(CATEGORIES, n),
        "amt": np.round(rng.lognormal(3.5, 1.0, n).clip(1.0, 1376.04), 2),
        "first": "Ann", "last": "Doe",
        "gender": rng.choice(["F", "M"], n),
        "street": "1 Main St", "city": "Town", "state": "NC", "zip": 28654,
        "lat": 36.0, "long": -81.0, "city_pop": rng.integers(100, 100000, n),
        "job": "Engineer", "dob": "1980-01-01",
        "trans_num": [f"{i:032x}" for i in range(n)],
        "unix_time": 0,
        "merch_lat": 36.5, "merch_long": -81.5,
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
