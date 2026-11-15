"""Episode threat model, the relative attacker strategy and its realisation as operations.

A fraudster holding a compromised card plans an episode of at most ``k_max`` operations within
``window_days`` after the episode start, aiming to withdraw at least ``rho`` of the target sum A
(the real episode's amount). Strategies are learned on spare cards and transferred to target cards,
so a ``Strategy`` is *relative* to an episode: amounts are shares of ``u·A``, times are offsets from
the start day, merchant coordinates are offsets from the card's home.

Realisation rules (``realize``):

* amounts ``u·A·shares[:k] / sum(shares[:k])`` (equal shares when they are all zero), clipped to
  ``[amount_min, amount_max]`` and rounded to cents with ``np.round`` (round half to even on the
  binary value); the bounds are whole cents, so rounding keeps an amount in bounds;
* time ``start.normalize() + day days + hour hours``, rounded to whole seconds; an operation before
  ``start`` is moved to ``start`` and one at or after ``start + window_days`` is moved to the last
  second of the window (``day < window_days`` alone does not keep ``day + hour`` inside it);
* merchant ``merchant[i]`` is an index into ``MerchantTable`` (category from the data); merchant
  coordinates are the card's home ``(lat, long)`` plus ``U(-r, r)`` each, drawn from ``rng``.

``feasible`` checks the threat model only, not success (``sum >= rho·A``): a strategy whose target is
physically unreachable (``k_max·amount_max < rho·A``) is still a valid, merely unsuccessful episode.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import Episode
from .serving.state import CardProfile, CardState, Tx, profile_of

_SEC = pd.Timedelta(seconds=1)
_COORD_TOL = 1e-9  # home + U(-r, r) minus home may exceed r by float rounding


@dataclass(frozen=True)
class ThreatModel:
    rho: float
    k_max: int
    amount_min: float
    amount_max: float
    window_days: int
    max_declines: int
    knowledge: str
    recon_budget_ops: int
    cards_per_eval: int
    merchant_coord_radius: float

    @classmethod
    def from_yaml(cls, path: Path) -> "ThreatModel":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping")
        names = [f.name for f in dataclasses.fields(cls)]
        extra = sorted(set(raw) - set(names))
        missing = [n for n in names if n not in raw]
        if extra:
            raise ValueError(f"{path}: unknown keys {extra}")
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        tm = cls(rho=float(raw["rho"]), k_max=int(raw["k_max"]), amount_min=float(raw["amount_min"]),
                 amount_max=float(raw["amount_max"]), window_days=int(raw["window_days"]),
                 max_declines=int(raw["max_declines"]), knowledge=str(raw["knowledge"]),
                 recon_budget_ops=int(raw["recon_budget_ops"]), cards_per_eval=int(raw["cards_per_eval"]),
                 merchant_coord_radius=float(raw["merchant_coord_radius"]))
        if not (0 < tm.rho <= 1 and tm.k_max >= 1 and 0 < tm.amount_min <= tm.amount_max
                and tm.window_days >= 1 and tm.merchant_coord_radius >= 0):
            raise ValueError(f"{path}: inconsistent threat model {tm}")
        if tm.knowledge not in ("decision", "score"):
            raise ValueError(f"{path}: knowledge must be 'decision' or 'score', got {tm.knowledge!r}")
        return tm


@dataclass(frozen=True, eq=False)
class MerchantTable:
    """Merchants sorted by raw name (``fraud_`` prefix kept); category of the first occurrence in data."""
    names: list[str]
    category: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(self.names)})

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "MerchantTable":
        first = df.drop_duplicates("merchant", keep="first")
        cat = dict(zip(first["merchant"].astype(str), first["category"].astype(str)))
        names = sorted(cat)
        return cls(names=names, category=[cat[n] for n in names])

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int | None:
        return self._index.get(name)


@dataclass(frozen=True)
class EpisodeContext:
    """What the attacker works with on one card: the episode (start, target A), the owner profile and
    the card state right before ``episode.start`` (no operation of the episode, nothing after start)."""
    episode: Episode
    profile: CardProfile
    state: CardState


def _context(g: pd.DataFrame, profile: CardProfile, episode: Episode) -> EpisodeContext:
    hist = g[(g["ts"] < episode.start) & ~g["row"].isin(episode.rows)]
    return EpisodeContext(episode=episode, profile=profile,
                          state=CardState.from_history(profile, hist, at=episode.start))


def context_for(df: pd.DataFrame, episode: Episode) -> EpisodeContext:
    g = df[df["cc_num"] == episode.card]
    return _context(g, profile_of(g, episode.card), episode)


def contexts_for(df: pd.DataFrame, episodes: list[Episode]) -> list[EpisodeContext]:
    """``context_for`` of every episode, indexing ``df`` by card once (same order as ``episodes``)."""
    cards = {e.card for e in episodes}
    groups = dict(tuple(df[df["cc_num"].isin(cards)].groupby("cc_num", sort=False)))
    profiles = {c: profile_of(groups[c], c) for c in cards if c in groups}
    out = []
    for e in episodes:
        if e.card not in groups:
            raise ValueError(f"no operations of card {e.card}")
        out.append(_context(groups[e.card], profiles[e.card], e))
    return out


@dataclass
class Strategy:
    """Relative episode plan. Arrays have length ``k_max``; only the first ``k`` entries are used.

    Vector layout (``to_vector``, length ``2 + 4·k_max``):
    ``[k, u, shares[0:k_max], day[0:k_max], hour[0:k_max], merchant[0:k_max]]``.
    """
    k: int
    u: float
    shares: np.ndarray
    day: np.ndarray
    hour: np.ndarray
    merchant: np.ndarray

    @staticmethod
    def vector_length(tm: ThreatModel) -> int:
        return 2 + 4 * tm.k_max

    @classmethod
    def random(cls, tm: ThreatModel, n_merchants: int, rng: np.random.Generator) -> "Strategy":
        n = tm.k_max
        return cls(k=int(rng.integers(1, n + 1)), u=float(rng.uniform(tm.rho, 1.0)),
                   shares=rng.uniform(0.0, 1.0, n), day=rng.uniform(0.0, tm.window_days, n),
                   hour=rng.uniform(0.0, 24.0, n), merchant=rng.integers(0, n_merchants, n))

    def clip(self, tm: ThreatModel, n_merchants: int | None = None) -> "Strategy":
        """Every field in range: k integer in [1, k_max], u in [rho, 1], shares >= 0,
        day in [0, window_days), hour in [0, 24), merchant integer in [0, n_merchants)
        (only >= 0 when ``n_merchants`` is None). Non-finite values are treated as 0."""
        def arr(a):
            a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            if a.shape != (tm.k_max,):
                raise ValueError(f"expected arrays of length k_max={tm.k_max}, got {a.shape}")
            return a
        k = float(np.nan_to_num(self.k, nan=1.0))
        u = float(np.nan_to_num(self.u, nan=tm.rho))
        merch = np.rint(arr(self.merchant))
        merch = np.clip(merch, 0, None if n_merchants is None else n_merchants - 1).astype(np.int64)
        return Strategy(
            k=int(np.clip(np.rint(k), 1, tm.k_max)),
            u=float(np.clip(u, tm.rho, 1.0)),
            shares=np.maximum(arr(self.shares), 0.0),
            day=np.clip(arr(self.day), 0.0, np.nextafter(float(tm.window_days), 0.0)),
            hour=np.clip(arr(self.hour), 0.0, np.nextafter(24.0, 0.0)),
            merchant=merch,
        )

    def to_vector(self) -> np.ndarray:
        return np.concatenate([[float(self.k), float(self.u)],
                               np.asarray(self.shares, dtype=np.float64), np.asarray(self.day, dtype=np.float64),
                               np.asarray(self.hour, dtype=np.float64),
                               np.asarray(self.merchant, dtype=np.float64)])

    @classmethod
    def from_vector(cls, v, tm: ThreatModel, n_merchants: int | None = None) -> "Strategy":
        """Inverse of ``to_vector``, followed by ``clip`` (so any real vector maps to a valid strategy)."""
        v = np.asarray(v, dtype=np.float64)
        n = tm.k_max
        if v.shape != (cls.vector_length(tm),):
            raise ValueError(f"expected a vector of length {cls.vector_length(tm)}, got {v.shape}")
        return cls(k=v[0], u=v[1], shares=v[2:2 + n], day=v[2 + n:2 + 2 * n], hour=v[2 + 2 * n:2 + 3 * n],
                   merchant=v[2 + 3 * n:2 + 4 * n]).clip(tm, n_merchants)


def realize(strategy: Strategy, ctx: EpisodeContext, tm: ThreatModel, merchants: MerchantTable,
            rng: np.random.Generator) -> list[Tx]:
    """Concrete operations of ``strategy`` on the episode of ``ctx``, sorted by ``ts`` (stable).

    Deterministic for a given ``rng`` state; draws exactly ``2·k`` uniforms (merchant coordinates).
    """
    s = strategy.clip(tm, len(merchants))
    k = s.k
    ep = ctx.episode
    shares = s.shares[:k]
    total = shares.sum()
    w = shares / total if total > 0 else np.full(k, 1.0 / k)
    amts = np.round(np.clip(s.u * ep.amount * w, tm.amount_min, tm.amount_max), 2)

    start = pd.Timestamp(ep.start)
    lo, hi = start, start + pd.Timedelta(days=tm.window_days) - _SEC
    base = start.normalize()
    r = tm.merchant_coord_radius
    off = rng.uniform(-r, r, size=(k, 2))
    txs = []
    for i in range(k):
        ts = (base + pd.Timedelta(days=float(s.day[i])) + pd.Timedelta(hours=float(s.hour[i]))).round("s")
        ts = min(max(ts, lo), hi)
        j = int(s.merchant[i])
        txs.append(Tx(ts=ts, amt=float(amts[i]), merchant=merchants.names[j], category=merchants.category[j],
                      merch_lat=ctx.profile.lat + float(off[i, 0]), merch_long=ctx.profile.long + float(off[i, 1])))
    return sorted(txs, key=lambda t: t.ts)


def feasible(txs: list[Tx], ctx: EpisodeContext, tm: ThreatModel, merchants: MerchantTable | None = None) -> bool:
    """Do ``txs`` respect the threat model on the episode of ``ctx``? Does not check success.

    Checks: 1..k_max operations; each amount in bounds and a whole number of cents; each ts in
    ``[start, start + window_days)``; merchant coordinates within ``merchant_coord_radius`` of home;
    the state's owner profile is the card's profile (``Tx`` carries no owner fields). With
    ``merchants`` also: merchant in the table with its category.
    """
    if not 1 <= len(txs) <= tm.k_max:
        return False
    if ctx.state.profile != ctx.profile or ctx.profile.cc_num != ctx.episode.card:
        return False
    start = pd.Timestamp(ctx.episode.start)
    end = start + pd.Timedelta(days=tm.window_days)
    r = tm.merchant_coord_radius + _COORD_TOL
    for tx in txs:
        if not isinstance(tx, Tx):
            return False
        a = float(tx.amt)
        if not (np.isfinite(a) and tm.amount_min <= a <= tm.amount_max):
            return False
        if abs(a * 100 - round(a * 100)) > 1e-6:
            return False
        ts = pd.Timestamp(tx.ts)
        if not start <= ts < end:
            return False
        if not (abs(tx.merch_lat - ctx.profile.lat) <= r and abs(tx.merch_long - ctx.profile.long) <= r):
            return False
        if merchants is not None:
            j = merchants.index(tx.merchant)
            if j is None or merchants.category[j] != tx.category:
                return False
    return True
