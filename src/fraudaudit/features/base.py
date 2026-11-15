"""The 27 base features of the bachelor model and a JSON-serializable encoder.

Semantics follow Appendix B of the bachelor thesis. The encoder never looks at
other rows in ``transform``, so one-row frames (serving path) give the same
values as batches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import BASE_COLUMNS, CATEGORIES

PREFIX = "fraud_"
PASSTHROUGH = ("amt", "lat", "long", "city_pop", "merch_lat", "merch_long")


def _merchant_names(s: pd.Series) -> pd.Series:
    return s.astype(str).str.removeprefix(PREFIX)


def _unique(values) -> list[str]:
    return list(dict.fromkeys(values))  # order of first appearance


class BaseEncoder:
    def __init__(self) -> None:
        self.merchants: list[str] | None = None
        self.states: list[str] | None = None
        self.categories: list[str] | None = None  # fixed, incl. the dropped baseline
        self.genders: list[str] | None = None  # F is the dropped baseline

    def fit(self, df: pd.DataFrame) -> "BaseEncoder":
        self.merchants = _unique(_merchant_names(df["merchant"]))
        self.states = _unique(df["state"].astype(str))
        self.categories = list(CATEGORIES)
        unknown = set(df["category"].astype(str)) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown category in fit: {sorted(unknown)}")
        bad = set(df["gender"].astype(str)) - {"F", "M"}
        if bad:
            raise ValueError(f"unexpected genders {sorted(bad)}")
        self.genders = ["F", "M"]
        return self

    @staticmethod
    def _codes(values: pd.Series, known: list[str], what: str) -> np.ndarray:
        index = {v: i for i, v in enumerate(known)}
        codes = values.map(index)
        if codes.isna().any():
            bad = sorted(set(values[codes.isna()]))[:5]
            raise ValueError(f"unknown {what} in transform: {bad}")
        return codes.to_numpy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.merchants is None:
            raise ValueError("BaseEncoder is not fitted")
        out = pd.DataFrame(index=df.index)
        out["merchant"] = self._codes(_merchant_names(df["merchant"]), self.merchants, "merchant")
        out["state"] = self._codes(df["state"].astype(str), self.states, "state")
        for c in PASSTHROUGH:
            out[c] = df[c].to_numpy(dtype="float64")
        trans_date = pd.to_datetime(df["ts"]).dt.normalize()
        days = (trans_date - pd.to_datetime(df["dob"])).dt.days
        out["age"] = np.round(days / 365.25)  # half-to-even, as Python/numpy round
        out["trans_month"] = trans_date.dt.month
        out["trans_year"] = trans_date.dt.year
        out["latitudinal_distance"] = np.abs(np.round(df["merch_lat"] - df["lat"], 3))
        out["longitudinal_distance"] = np.abs(np.round(df["merch_long"] - df["long"], 3))
        cat = df["category"].astype(str)
        self._codes(cat, self.categories, "category")  # validation only
        for c in self.categories[1:]:
            out[f"category_{c}"] = (cat == c).to_numpy()
        gender = df["gender"].astype(str)
        self._codes(gender, self.genders, "gender")
        out["gender_M"] = (gender == "M").to_numpy()
        return out[list(BASE_COLUMNS)].astype("float32")

    def to_json(self) -> dict:
        if self.merchants is None:
            raise ValueError("BaseEncoder is not fitted")
        return {
            "version": 1,
            "merchants": list(self.merchants),
            "states": list(self.states),
            "categories": list(self.categories),
            "genders": list(self.genders),
        }

    @classmethod
    def from_json(cls, d: dict) -> "BaseEncoder":
        enc = cls()
        enc.merchants = list(d["merchants"])
        enc.states = list(d["states"])
        enc.categories = list(d["categories"])
        enc.genders = list(d["genders"])
        return enc
