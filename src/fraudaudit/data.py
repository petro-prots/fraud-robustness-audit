"""Data loading, time periods and fraud episodes (Sparkov)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VAL_START = pd.Timestamp("2020-03-21")
EPISODE_GAP = pd.Timedelta(days=7)
SPARES_PERIODS = ("fit", "test")  # where the reconnaissance spares come from (ruling R14), see audit.select_episodes
# Ruling R20: the fit-period fraud episodes are split by card into H (hardening generation, ``harden``) and S
# (reconnaissance spares of the gate). The split seed is a constant, NOT the audit ``--seed``: an audit with another
# seed must never reconnoitre on cards whose adversarial episodes were used to train the hardened model.
FIT_SPLIT_SEED = 42
_FIT_SPLIT_STREAM = 20


def finalize(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Concatenate train and test, add ts/row/part, sort stably by (ts, row)."""
    df = pd.concat([train.assign(part="train"), test.assign(part="test")], ignore_index=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df["row"] = np.arange(len(df), dtype="int64")
    df = df.sort_values(["ts", "row"], kind="stable").reset_index(drop=True)
    return df


def load_raw(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "fraudTrain.csv")
    test = pd.read_csv(data_dir / "fraudTest.csv")
    return finalize(train, test)


def period(df: pd.DataFrame) -> pd.Series:
    out = np.where(df["part"] == "test", "test", np.where(df["ts"] >= VAL_START, "val", "fit"))
    return pd.Series(out, index=df.index, name="period")


@dataclass(frozen=True)
class Episode:
    card: int
    start: pd.Timestamp
    amount: float
    rows: tuple[int, ...]


def fraud_episodes(df: pd.DataFrame, periods: set[str]) -> list[Episode]:
    """Fraud ops per card in the given periods, split where the gap exceeds 7 days."""
    fr = df[(df["is_fraud"] == 1) & period(df).isin(periods)]
    fr = fr.sort_values(["cc_num", "ts", "row"], kind="stable")
    episodes: list[Episode] = []
    for card, g in fr.groupby("cc_num", sort=True):
        ts = g["ts"].to_numpy()
        new = np.concatenate([[True], np.diff(ts) > EPISODE_GAP.to_timedelta64()])
        for _, e in g.groupby(np.cumsum(new)):
            episodes.append(Episode(
                card=int(card),
                start=pd.Timestamp(e["ts"].iloc[0]),
                amount=float(e["amt"].sum()),
                rows=tuple(int(r) for r in e["row"]),
            ))
    return episodes


def split_cards(episodes: list[Episode], seed: int) -> tuple[list[Episode], list[Episode]]:
    """Split by card 50/50: (target, spare). Episodes of one card stay together."""
    cards = sorted({e.card for e in episodes})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cards))
    n_target = (len(cards) + 1) // 2
    target = {cards[i] for i in perm[:n_target]}
    return ([e for e in episodes if e.card in target],
            [e for e in episodes if e.card not in target])


def fit_split(episodes: list[Episode], seed: int = FIT_SPLIT_SEED) -> tuple[list[Episode], list[Episode]]:
    """Split the fit-period episodes by card ~50/50 into (H, S) (ruling R20); episodes of one card stay together.

    H (``(n_cards + 1) // 2`` cards) feeds the adversarial generation of ``harden``, S is the gate's pool of
    reconnaissance spares (``spares_period: fit``). Deterministic: the cards are sorted and permuted with
    ``default_rng([seed, 20])`` (its own stream, unlike ``split_cards``), so the input order does not matter.
    Each list keeps the input order."""
    cards = sorted({e.card for e in episodes})
    perm = np.random.default_rng([int(seed), _FIT_SPLIT_STREAM]).permutation(len(cards))
    h = {cards[i] for i in perm[:(len(cards) + 1) // 2]}
    return [e for e in episodes if e.card in h], [e for e in episodes if e.card not in h]


def fit_split_meta(h: list[Episode], s: list[Episode], seed: int = FIT_SPLIT_SEED) -> dict:
    """Seed and sizes of a ``fit_split`` for reports."""
    return {"seed": int(seed), "n_h_cards": len({e.card for e in h}), "n_s_cards": len({e.card for e in s}),
            "n_h_episodes": len(h), "n_s_episodes": len(s)}
