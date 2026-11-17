"""Soft per-card spending corridor with a guaranteed loss bound (Task 22, rulings R24/R25).

Bank policy on top of the model, for every operation i of card c at time t_i with amount a_i:

1. the model declines (``p >= `` the active threshold) → decline (``max_declines``, ``tau_after_decline`` as before);
2. the model approves and, for BOTH windows W ∈ {24 h, 7 d}, ``S_W(c, t_i⁻) + a_i <= C_W(c, t_i)`` → AUTO-APPROVE;
3. the model approves but a window would be exceeded → CONFIRM (owner confirmation in the bank app). The owner
   confirms (friction, the operation passes, it is NOT auto-approved); a fraudster has the card details but not the
   owner's second factor (threat-model assumption), so the operation does not pass and – with
   ``confirm_fail_counts_as_decline`` – counts as a decline.

``S_W(c, t⁻)`` is the sum of the card's AUTO-APPROVED operations in the rolling window ``(t − W, t]`` without the
current one (``CardState.auto_sum_cents``). Everything is compared in whole cents: ``C_W`` is rounded down to a
cent, so ``S + a <= C`` holds exactly.

Corridor ``C_W(c, t) = max(floor_W, Q_W(c, d)) · m_W(d)``, d = the calendar day of t:

* ``Q_W(c, d)`` – the q-quantile of the card's own rolling-W spend over the lookback: the values
  ``V_j = sum{a : t_j − W < t <= t_j}`` right after every LEGITIMATE operation j of the card with
  ``d − lookback <= t_j < d`` (``LegitHistory.window_quantile``; only operations strictly before t, recomputed
  once per day). The profile is the owner's legitimate spend: every non-fraud operation of the history (the
  data were never decided by a corridor, so all count) and, in the legitimate replay, every owner operation
  whatever its decision; attacker operations never enter it, so C does not depend on the attack;
* ``m_W(d)`` – multipliers of statistically expected cycles, estimated on the fit period from the POPULATION
  (``estimate_multipliers``): weeks related to a major US holiday (Sparkov is US data) – the week offset
  ``k = floor((d − h) / 7 days)`` from the nearest of Christmas (Dec 25), Thanksgiving (4th Thursday of November)
  and Independence Day (Jul 4), ``k ∈ holiday_weeks`` – applied to both windows; weekday and day of month
  (salary days) – applied to the 24 h window only (a 7-day window always spans every weekday). A factor is the
  ratio of the mean daily population spend of its days to the mean daily spend of the baseline days (days
  outside every holiday week); it is used only when its percentile bootstrap interval (days resampled with
  replacement within the group and within the baseline) excludes 1, otherwise m = 1. Holiday dates come from
  their rules; the model never gets a calendar date.

Guarantee (Proposition, thesis ch. 2): if the episode lasts D <= W = 7 days, for ANY attack the auto-approved
attacker amount is ``A_auto <= max_{t ∈ [t0, t0 + D)} C_7d(c, t)`` (``episode_bound``: C is constant per day, so the
max over the days the window touches). ``headroom_cents`` is a tighter valid bound,
``max(0, max_t (C_7d(c, t) − S_owner(t)))`` with S_owner the owner's auto-approved operations still in the window
(no owner operation is replayed during an episode, so S_owner only decreases).
"""
from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .serving.state import CardState, LegitHistory

_NS = 1_000_000_000
DAY_NS = 86_400 * _NS
WINDOWS_NS = {"24h": DAY_NS, "7d": 7 * DAY_NS}
HOLIDAYS = ("christmas", "independence_day", "thanksgiving")
MULTIPLIER_KINDS = ("holiday", "weekday", "day_of_month")
MULTIPLIER_SEED = 42  # bootstrap of the multipliers: a constant, independent of the audit seed
_KEYS = ("q", "lookback_days", "floor_24h", "floor_7d", "multipliers", "holiday_weeks", "n_boot", "confidence",
         "confirm_fail_counts_as_decline")


@dataclass(frozen=True)
class Multipliers:
    """Fitted multipliers; only significant ones are stored (absent = 1). Keys are strings: ``"christmas:-1"``,
    weekday ``"0"`` (Monday) … ``"6"``, day of month ``"1"`` … ``"31"``. ``estimates`` lists every estimate
    (kind, group, ratio, lower, upper, significant, n_days)."""
    holiday: dict
    weekday: dict
    day_of_month: dict
    estimates: tuple
    holiday_weeks: tuple

    def factor(self, day_ns: int, window: str, kinds: tuple) -> float:
        m = 1.0
        if "holiday" in kinds and self.holiday:
            lab = holiday_label(day_ns, *self.holiday_weeks)
            if lab is not None:
                m *= self.holiday.get(lab, 1.0)
        if window == "24h":
            d = _day(day_ns)
            if "weekday" in kinds:
                m *= self.weekday.get(str(d.weekday()), 1.0)
            if "day_of_month" in kinds:
                m *= self.day_of_month.get(str(d.day), 1.0)
        return m

    def summary(self) -> dict:
        return {"holiday": dict(self.holiday), "weekday": dict(self.weekday),
                "day_of_month": dict(self.day_of_month), "n_estimates": len(self.estimates),
                "n_significant": sum(1 for e in self.estimates if e["significant"])}


@dataclass(frozen=True)
class CorridorConfig:
    """``corridor:`` section of the threat model (strict loader ``from_mapping``); ``fitted`` is set by ``prepare``
    (never from YAML)."""
    q: float
    lookback_days: int
    floor_24h: float
    floor_7d: float
    multipliers: tuple
    holiday_weeks: tuple
    n_boot: int
    confidence: float
    confirm_fail_counts_as_decline: bool
    fitted: Multipliers | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.q <= 1:
            raise ValueError(f"corridor.q must be in [0, 1], got {self.q!r}")
        if self.lookback_days < 1:
            raise ValueError("corridor.lookback_days must be >= 1")
        for k in ("floor_24h", "floor_7d"):
            v = getattr(self, k)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValueError(f"corridor.{k} must be a finite number >= 0, got {v!r}")
        if any(k not in MULTIPLIER_KINDS for k in self.multipliers) or len(set(self.multipliers)) != len(
                self.multipliers):
            raise ValueError(f"corridor.multipliers must be distinct names of {list(MULTIPLIER_KINDS)}")
        lo, hi = self.holiday_weeks
        if not lo <= hi:
            raise ValueError("corridor.holiday_weeks must be [lo, hi] with lo <= hi")
        if self.n_boot < 1 or not 0 < self.confidence < 1:
            raise ValueError("corridor.n_boot must be >= 1 and confidence in (0, 1)")

    @classmethod
    def from_mapping(cls, raw, where: str) -> "CorridorConfig":
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: corridor must be a mapping or null")
        extra = sorted(set(raw) - set(_KEYS))
        missing = [k for k in _KEYS if k not in raw]
        if extra:
            raise ValueError(f"{where}: unknown keys in corridor {extra}")
        if missing:
            raise ValueError(f"{where}: missing keys in corridor {missing}")

        def num(k):
            v = raw[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{where}: corridor.{k} must be a finite number, got {v!r}")
            return float(v)

        def integer(k):
            v = num(k)
            if not v.is_integer():
                raise ValueError(f"{where}: corridor.{k} must be an integer, got {raw[k]!r}")
            return int(v)

        mult, hw = raw["multipliers"], raw["holiday_weeks"]
        if not isinstance(mult, list) or not all(isinstance(x, str) for x in mult):
            raise ValueError(f"{where}: corridor.multipliers must be a list of names")
        if not (isinstance(hw, list) and len(hw) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                               for x in hw)):
            raise ValueError(f"{where}: corridor.holiday_weeks must be [lo, hi] (integers)")
        if not isinstance(raw["confirm_fail_counts_as_decline"], bool):
            raise ValueError(f"{where}: corridor.confirm_fail_counts_as_decline must be true or false")
        try:
            return cls(q=num("q"), lookback_days=integer("lookback_days"), floor_24h=num("floor_24h"),
                       floor_7d=num("floor_7d"), multipliers=tuple(mult), holiday_weeks=(hw[0], hw[1]),
                       n_boot=integer("n_boot"), confidence=num("confidence"),
                       confirm_fail_counts_as_decline=raw["confirm_fail_counts_as_decline"])
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None

    def check_ready(self) -> None:
        if self.multipliers and self.fitted is None:
            raise ValueError("corridor multipliers are not fitted: call corridor.prepare(tm, df) first")

    def public(self) -> dict:
        """Parameters for reports (without the fitted estimates)."""
        d = {k: getattr(self, k) for k in _KEYS}
        d["multipliers"], d["holiday_weeks"] = list(self.multipliers), list(self.holiday_weeks)
        return d


# --- holiday calendar ------------------------------------------------------------------------------------------

def holiday_dates(year: int) -> dict[str, pd.Timestamp]:
    """Dates of the major US holidays of ``year`` from their rules."""
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    thanks = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7 + 21)  # 4th Thursday
    return {"christmas": pd.Timestamp(year=year, month=12, day=25),
            "independence_day": pd.Timestamp(year=year, month=7, day=4), "thanksgiving": thanks}


def _day(day_ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(day_ns) - int(day_ns) % DAY_NS)


@functools.lru_cache(maxsize=65_536)
def holiday_label(day_ns: int, lo: int, hi: int) -> str | None:
    """``"<holiday>:<k>"`` of the day's week offset k from the NEAREST major holiday (ties: the earlier holiday),
    if ``lo <= k <= hi``; else None."""
    d = _day(day_ns)
    best = None
    for y in (d.year - 1, d.year, d.year + 1):
        for name, h in holiday_dates(y).items():
            diff = (d - h).days
            cand = (abs(diff), h, name, diff)
            if best is None or cand[:2] < best[:2]:
                best = cand
    _, _, name, diff = best
    k = diff // 7
    return f"{name}:{k}" if lo <= k <= hi else None


# --- multipliers -------------------------------------------------------------------------------------------------

def _boot_ratio(g: np.ndarray, base: np.ndarray, n_boot: int, confidence: float, seed: list) -> tuple:
    rng = np.random.default_rng(seed)
    gm = g[rng.integers(0, g.size, size=(n_boot, g.size))].mean(axis=1)
    bm = base[rng.integers(0, base.size, size=(n_boot, base.size))].mean(axis=1)
    r = gm / bm
    a = 1 - confidence
    lo, hi = np.quantile(r, [a / 2, 1 - a / 2])
    return float(lo), float(hi)


def estimate_from_daily(daily: pd.Series, cfg: CorridorConfig) -> Multipliers:
    """Multipliers from a daily population spend series (index: days). See the module docstring."""
    days = pd.DatetimeIndex(daily.index).normalize()
    v = np.asarray(daily, dtype=np.float64)
    lo, hi = cfg.holiday_weeks
    labels = np.array([holiday_label(d.value, lo, hi) or "" for d in days], dtype=object)
    base_m = labels == ""
    base = v[base_m]
    if base.size == 0:
        raise ValueError("no baseline days (every day is in a holiday week)")
    base_mean = base.mean()
    groups: list[tuple[str, str, np.ndarray]] = []
    if "holiday" in cfg.multipliers:
        for lab in sorted(set(labels[~base_m])):
            groups.append(("holiday", lab, v[labels == lab]))
    if "weekday" in cfg.multipliers:
        for w in range(7):
            groups.append(("weekday", str(w), v[base_m & (days.weekday == w)]))
    if "day_of_month" in cfg.multipliers:
        for dm in range(1, 32):
            groups.append(("day_of_month", str(dm), v[base_m & (days.day == dm)]))
    out = {"holiday": {}, "weekday": {}, "day_of_month": {}}
    est = []
    for gi, (kind, group, g) in enumerate(groups):
        if g.size == 0:
            continue
        ratio = float(g.mean() / base_mean)
        lower, upper = _boot_ratio(g, base, cfg.n_boot, cfg.confidence,
                                   [MULTIPLIER_SEED, MULTIPLIER_KINDS.index(kind), _group_id(kind, group)])
        sig = bool(lower > 1 or upper < 1)
        if sig:
            out[kind][group] = ratio
        est.append({"kind": kind, "group": group, "ratio": ratio, "lower": lower, "upper": upper,
                    "significant": sig, "n_days": int(g.size)})
    return Multipliers(holiday=out["holiday"], weekday=out["weekday"], day_of_month=out["day_of_month"],
                       estimates=tuple(est), holiday_weeks=tuple(cfg.holiday_weeks))


def _group_id(kind: str, group: str) -> int:
    if kind != "holiday":
        return int(group)
    name, k = group.split(":")
    return HOLIDAYS.index(name) * 1000 + int(k) + 500


def daily_population_spend(df: pd.DataFrame, period_name: str = "fit") -> pd.Series:
    """Sum of legitimate amounts per calendar day of ``period_name`` (every day of the period, zeros included)."""
    from .data import period
    m = (period(df) == period_name).to_numpy() & (df["is_fraud"].to_numpy() == 0)
    sub = df.loc[m, ["ts", "amt"]]
    if sub.empty:
        raise ValueError(f"no legitimate operations in the {period_name!r} period")
    s = sub.groupby(sub["ts"].dt.normalize())["amt"].sum()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full, fill_value=0.0)


def estimate_multipliers(df: pd.DataFrame, cfg: CorridorConfig) -> Multipliers:
    """``estimate_from_daily`` on the population's daily legitimate spend of the fit period."""
    return estimate_from_daily(daily_population_spend(df, "fit"), cfg)


def prepare(tm, df: pd.DataFrame):
    """``tm`` with its corridor's multipliers fitted on ``df`` (unchanged when there is no corridor, no multiplier
    is enabled or they are already fitted)."""
    c = getattr(tm, "corridor", None)
    if c is None or not c.multipliers or c.fitted is not None:
        return tm
    return dataclasses.replace(tm, corridor=dataclasses.replace(c, fitted=estimate_multipliers(df, c)))


# --- corridor and decision -------------------------------------------------------------------------------------

def limit_cents(cfg: CorridorConfig, legit: LegitHistory, day_ns: int, window: str) -> int:
    """``C_W`` in whole cents (rounded down) for the calendar day of ``day_ns``."""
    day = int(day_ns) - int(day_ns) % DAY_NS
    q = legit.window_quantile(day, cfg.lookback_days * DAY_NS, WINDOWS_NS[window], cfg.q)
    base = max(cfg.floor_24h if window == "24h" else cfg.floor_7d, q)
    m = cfg.fitted.factor(day, window, cfg.multipliers) if cfg.multipliers else 1.0
    return int(math.floor(base * m * 100 + 1e-6))


def limits_cents(cfg: CorridorConfig, state: CardState, ts) -> tuple[int, int]:
    t = pd.Timestamp(ts).value
    return limit_cents(cfg, state.legit, t, "24h"), limit_cents(cfg, state.legit, t, "7d")


def within(cfg: CorridorConfig, state: CardState, ts, amt: float) -> bool:
    """Rule 2: ``S_W(t⁻) + a <= C_W(t)`` for both windows (whole cents)."""
    t = pd.Timestamp(ts).value
    a = int(round(float(amt) * 100))
    for w in ("24h", "7d"):
        if state.auto_sum_cents(t, WINDOWS_NS[w]) + a > limit_cents(cfg, state.legit, t, w):
            return False
    return True


def episode_bound(cfg: CorridorConfig, state: CardState, start, window_days: int) -> dict:
    """The guaranteed bound of an episode starting at ``start`` on a card in ``state`` (state right before start):
    ``bound_cents = max_{t ∈ [start, start + D)} C_7d`` and the tighter ``headroom_cents`` (module docstring).
    ``ValueError`` if D > W (the Proposition needs D <= W)."""
    D = int(window_days) * DAY_NS
    W = WINDOWS_NS["7d"]
    if D > W:
        raise ValueError(f"the guarantee needs D <= W: episode {window_days} days, window 7 days")
    t0 = pd.Timestamp(start).value
    end = t0 + D
    bound, head = 0, 0
    d = t0 - t0 % DAY_NS
    while d < end:
        c = limit_cents(cfg, state.legit, d, "7d")
        seg_end = min(d + DAY_NS, end)
        s_min = state.auto_sum_cents(seg_end - 1, W)  # owner ops still in (t − W, t] as t → seg_end⁻
        bound = max(bound, c)
        head = max(head, c - s_min)
        d += DAY_NS
    return {"bound_cents": int(bound), "headroom_cents": int(max(0, head))}


def batch_state(t_ns: np.ndarray, amt: np.ndarray, auto: np.ndarray, legit: np.ndarray,
                cfg: CorridorConfig) -> dict:
    """Batch path of the corridor state for one card's time-ordered stream with KNOWN decisions: per operation i,
    ``s24``/``s7`` = auto-approved cents of the earlier operations (in stream order) in ``(t_i − W, t_i]`` and
    ``c24``/``c7`` = the corridor from the legitimate operations before i. Must equal the serving path
    (``CardState.auto_sum_cents`` + ``limits_cents`` with ``push(…, auto_approved, legit)``)."""
    t = np.asarray(t_ns, dtype=np.int64)
    cents = np.rint(np.asarray(amt, dtype=np.float64) * 100).astype(np.int64) * np.asarray(auto, dtype=bool)
    cum = np.concatenate([[0], np.cumsum(cents)])
    n = t.size
    idx = np.arange(n)
    out = {}
    for name, w in (("s24", WINDOWS_NS["24h"]), ("s7", WINDOWS_NS["7d"])):
        lo = np.minimum(np.searchsorted(t, t - w, "right"), idx)
        out[name] = (cum[idx] - cum[lo]).tolist()
    lm = np.asarray(legit, dtype=bool)
    lt, la = t[lm], np.asarray(amt, dtype=np.float64)[lm]
    c24, c7 = [], []
    cache = {}
    for i in range(n):
        day = int(t[i] - t[i] % DAY_NS)
        if day not in cache:
            k = int(np.searchsorted(lt, day, "left"))  # legitimate ops strictly before the day
            h = LegitHistory(lt[:k], la[:k])
            cache[day] = (limit_cents(cfg, h, day, "24h"), limit_cents(cfg, h, day, "7d"))
        c24.append(cache[day][0])
        c7.append(cache[day][1])
    out["c24"], out["c7"] = c24, c7
    return out


def bound_stats(per_target: list[dict], rho: float) -> dict:
    """Summary of the per-target bounds for reports (dollars and shares of A)."""
    b = np.array([t["corridor_bound"] for t in per_target], dtype=np.float64)
    h = np.array([t["corridor_headroom"] for t in per_target], dtype=np.float64)
    a = np.array([t["amount"] for t in per_target], dtype=np.float64)

    def s(x):
        return {"mean": float(x.mean()), "median": float(np.median(x)), "max": float(x.max())}
    return {"bound": s(b), "bound_share": s(b / a), "headroom": s(h), "headroom_share": s(h / a),
            "share_targets_bound_below_rho_a": float(np.mean(b < rho * a))}
