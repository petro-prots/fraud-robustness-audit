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

* ``Q_W(c, d)`` – the q-quantile of the card's own rolling-W spend, LAGGED by 7 days (ruling R26): the values
  ``V_j = sum{a : t_j − W < t <= t_j}`` of every operation j of the profile with ``d − lookback <= t_j < d − 7 d``
  (``SpendHistory.window_quantile`` with ``PROFILE_LAG_NS``; the same lag for both windows; recomputed once per
  day). ``V_j`` sums over time, so operations with the same timestamp as j all count. The profile is every
  operation of the card that PASSED: the real history before the test or episode (it happened; no fraud label is
  consulted) and every later operation that was auto-approved or confirmed by the owner, whoever sent it; model
  declines and failed confirmations never enter it. No fraud label is used anywhere in the profile;
* ``m_W(d)`` – multipliers of statistically expected cycles, estimated on the fit period from the POPULATION
  (``estimate_multipliers``). Per-window scaling: a HOLIDAY factor multiplies BOTH windows (24 h and 7 d) – the
  week offset ``k = floor((d − h) / 7 days)`` from the nearest of Christmas (Dec 25), Thanksgiving (4th Thursday
  of November) and Independence Day (Jul 4), ``k ∈ holiday_weeks`` (US holidays: Sparkov is US data); WEEKDAY and
  DAY-OF-MONTH (salary days) factors multiply the 24 h window only (a 7-day window always spans every weekday and
  a week of days of month). A factor is the ratio of the mean daily population spend of its days to the mean daily
  spend of the baseline days (days outside every holiday week; weekday and day-of-month groups use baseline days
  only); it is used only when its percentile bootstrap interval (days resampled with replacement within the group
  and within the baseline, an i.i.d. resampling of days that ignores serial correlation) excludes 1, otherwise
  m = 1. Every group is one test at ``confidence`` without a multiple-testing correction, so with n tests about
  n · (1 − confidence) factors are expected to be «significant» by chance (``Multipliers.summary``:
  ``n_tests``, ``expected_false_positives``). Holiday dates come from their rules; the model never gets a
  calendar date.

Attack independence (thesis ch. 2, §1): for t ∈ [t0, t0 + D) with D <= W = 7 d and d <= t the start of t's day,
``d − 7 d <= t − 7 d < t0 + D − 7 d <= t0``, so ``C_W(c, t)`` depends only on operations before the episode start:
it can be computed before the episode, no action of the attacker (passed, declined or not confirmed) changes it,
and the bank needs no fraud labels for the recent operations.

Guarantee (Proposition): if the episode lasts D <= W = 7 days, for ANY attack the auto-approved attacker amount is
``A_auto <= max_{t ∈ [t0, t0 + D)} C_7d(c, t)`` (``episode_bound``, from the pre-episode state only: C is constant
per day, so the max over the days the window touches; ``c7_by_day`` are the per-day values, which the bank's
decisions use – ``check_used_corridor``). ``headroom_cents`` is a tighter valid bound,
``max(0, max_t (C_7d(c, t) − S_owner(t)))`` with S_owner the owner's auto-approved operations still in the window
(no owner operation is replayed during an episode, so S_owner only decreases).

Floor (ruling R28, ``floor_mode = "population_percentile"``): ``floor_W`` = the ``floor_percentile``-th percentile,
over the cards established at the fit end, of each card's own ``Q_W(q)`` on the fit period (``estimate_floor``,
refitted per q by ``prepare``); applied only to cards NOT established on day d (first passed operation after
``d − lookback``); established cards: ``C_W = Q_W · m_W``. ``floor_mode = "fixed"``: ``floor_24h`` / ``floor_7d`` for
every card (Task 22).

V1 (ruling R27, ``daily_mode = "weekly_split"``): ``C_24h(c, t) = κ · s(weekday(t)) · C_7d(c, t)`` (cents, rounded
down), s from the population fit-period spend (``estimate_weekly_split``); it inherits the attack independence of C_7d.
Proposition 2: the auto-approved attacker amount in the first h 24-hour blocks ``B_i = [t0 + (i−1)·24 h, t0 + i·24 h)``
is ``<= B(h) = min(max_t C_7d, Σ_{i<=h} max_{t ∈ B_i} C_24h)`` (``episode_bound``: ``b_h_cents``; it holds in every
daily mode).

V2 (ruling R27, ``confirmed_extends``): the rule compares with ``C'_W(c, t) = C_W(c, t) + Σ owner-confirmed amounts in
(t − W, t]`` (``CardState.confirmed_sum_cents``; a fraudster's confirmation always fails, so only the owner extends it).
The bounds become ``max_t C'_7d`` and B'(h): attack-independent but known only after the episode (``posthoc_bound``).
"""
from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .serving.state import CardState, SpendHistory

_NS = 1_000_000_000
DAY_NS = 86_400 * _NS
WINDOWS_NS = {"24h": DAY_NS, "7d": 7 * DAY_NS}
PROFILE_LAG_NS = 7 * DAY_NS  # R26: the profile of day d uses operations before d − 7 d only (= the 7-day window W)
HOLIDAYS = ("christmas", "independence_day", "thanksgiving")
MULTIPLIER_KINDS = ("holiday", "weekday", "day_of_month")
MULTIPLIER_SEED = 42  # bootstrap of the multipliers: a constant, independent of the audit seed
_KEYS = ("q", "lookback_days", "floor_24h", "floor_7d", "multipliers", "holiday_weeks", "n_boot", "confidence",
         "confirm_fail_counts_as_decline")
_REQUIRED = ("q", "lookback_days", "multipliers", "holiday_weeks", "n_boot", "confidence",
             "confirm_fail_counts_as_decline")
_OPTIONAL = ("floor_24h", "floor_7d", "floor", "daily", "confirmed_extends")
FLOOR_MODES = ("fixed", "population_percentile")
DAILY_MODES = ("quantile", "weekly_split")


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
    confidence: float = 0.95

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
        """Significant factors, the number of tests (one per estimated group, no multiple-testing correction) and the
        number of «significant» factors expected by chance alone at ``confidence``."""
        n = len(self.estimates)
        return {"holiday": dict(self.holiday), "weekday": dict(self.weekday),
                "day_of_month": dict(self.day_of_month), "n_estimates": n, "n_tests": n,
                "n_significant": sum(1 for e in self.estimates if e["significant"]),
                "confidence": float(self.confidence),
                "expected_false_positives": float(n * (1 - self.confidence))}


@dataclass(frozen=True)
class FloorFit:
    """The statistical floor (ruling R28) for one quantile q: the ``percentile``-th percentile, over the cards
    established at the fit end (first legitimate fit-period operation at least ``lookback_days`` before it), of each
    card's own corridor quantile Q_W(q) on the fit period (``estimate_floor``), in $."""
    q: float
    lookback_days: int
    percentile: float
    floor_24h: float
    floor_7d: float
    n_cards: int
    fit_end: str


@dataclass(frozen=True)
class WeeklySplit:
    """V1 (ruling R27): ``shares[j]`` = s(j), the population share of weekly legitimate spend that falls on weekday j
    (0 = Monday), Σ s = 1, from the mean daily spend per weekday on the fit period (``estimate_weekly_split``);
    ``n_days`` = fit days per weekday."""
    shares: tuple
    n_days: tuple


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
    floor_mode: str = "fixed"  # fixed: floor_24h / floor_7d for every card; population_percentile: R28
    floor_percentile: float | None = None
    daily_mode: str = "quantile"  # quantile: own 24 h quantile; weekly_split: V1, C_24h = κ · s(j) · C_7d
    kappa: float | None = None
    confirmed_extends: bool = False  # V2: C'_W = C_W + owner-confirmed amounts in (t − W, t]
    floor_fit: FloorFit | None = None  # set by ``prepare`` (population floor of this q)
    split: WeeklySplit | None = None  # set by ``prepare`` (V1)

    def __post_init__(self) -> None:
        if self.floor_mode not in FLOOR_MODES:
            raise ValueError(f"corridor floor mode must be one of {list(FLOOR_MODES)}, got {self.floor_mode!r}")
        if self.floor_mode == "population_percentile":
            p = self.floor_percentile
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 100:
                raise ValueError(f"corridor floor percentile must be in [0, 100], got {p!r}")
        elif self.floor_percentile is not None:
            raise ValueError("corridor floor percentile is used only with the population_percentile floor mode")
        if self.daily_mode not in DAILY_MODES:
            raise ValueError(f"corridor daily mode must be one of {list(DAILY_MODES)}, got {self.daily_mode!r}")
        if self.daily_mode == "weekly_split":
            k = self.kappa
            if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k < 1:
                raise ValueError(f"corridor daily kappa must be a finite number >= 1, got {k!r}")
        elif self.kappa is not None:
            raise ValueError("corridor daily kappa is used only with the weekly_split daily mode")
        if not isinstance(self.confirmed_extends, bool):
            raise ValueError("corridor.confirmed_extends must be true or false")
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
        """Strict loader. Required: ``q, lookback_days, multipliers, holiday_weeks, n_boot, confidence,
        confirm_fail_counts_as_decline``. Floor: ``floor: {mode: population_percentile, percentile: P}`` (R28; then
        ``floor_24h`` / ``floor_7d`` must be absent), or fixed floors ``floor_24h`` + ``floor_7d`` (with no ``floor``
        key or ``floor: {mode: fixed}``; the Task 22 form). Optional ``daily: {mode: quantile}`` (default) or
        ``{mode: weekly_split, kappa: κ}`` (V1) and ``confirmed_extends: false`` (default) / ``true`` (V2)."""
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: corridor must be a mapping or null")
        extra = sorted(set(raw) - set(_REQUIRED) - set(_OPTIONAL))
        missing = [k for k in _REQUIRED if k not in raw]
        if extra:
            raise ValueError(f"{where}: unknown keys in corridor {extra}")
        if missing:
            raise ValueError(f"{where}: missing keys in corridor {missing}")

        def num(k, v=None):
            v = raw[k] if v is None else v
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{where}: corridor.{k} must be a finite number, got {v!r}")
            return float(v)

        def integer(k):
            v = num(k)
            if not v.is_integer():
                raise ValueError(f"{where}: corridor.{k} must be an integer, got {raw[k]!r}")
            return int(v)

        def mapping(k, allowed):
            v = raw[k]
            if not isinstance(v, dict) or "mode" not in v:
                raise ValueError(f"{where}: corridor.{k} must be a mapping with a mode")
            bad = sorted(set(v) - set(allowed))
            if bad:
                raise ValueError(f"{where}: unknown keys in corridor.{k} {bad}")
            return v

        mult, hw = raw["multipliers"], raw["holiday_weeks"]
        if not isinstance(mult, list) or not all(isinstance(x, str) for x in mult):
            raise ValueError(f"{where}: corridor.multipliers must be a list of names")
        if not (isinstance(hw, list) and len(hw) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                               for x in hw)):
            raise ValueError(f"{where}: corridor.holiday_weeks must be [lo, hi] (integers)")
        if not isinstance(raw["confirm_fail_counts_as_decline"], bool):
            raise ValueError(f"{where}: corridor.confirm_fail_counts_as_decline must be true or false")
        floor_mode, pct = "fixed", None
        if "floor" in raw:
            fl = mapping("floor", ("mode", "percentile"))
            floor_mode = fl["mode"]
            if floor_mode == "population_percentile":
                if "percentile" not in fl:
                    raise ValueError(f"{where}: corridor.floor needs a percentile")
                pct = num("floor.percentile", fl["percentile"])
            elif "percentile" in fl:
                raise ValueError(f"{where}: corridor.floor.percentile is used only with population_percentile")
        fixed = [k for k in ("floor_24h", "floor_7d") if k in raw]
        if floor_mode == "fixed" and len(fixed) != 2:
            raise ValueError(f"{where}: the fixed floor needs floor_24h and floor_7d (or floor: {{mode: "
                             "population_percentile, percentile: P}})")
        if floor_mode != "fixed" and fixed:
            raise ValueError(f"{where}: {fixed} are used only with the fixed floor mode")
        daily_mode, kappa = "quantile", None
        if "daily" in raw:
            dl = mapping("daily", ("mode", "kappa"))
            daily_mode = dl["mode"]
            if "kappa" in dl:
                kappa = num("daily.kappa", dl["kappa"])
            elif daily_mode == "weekly_split":
                raise ValueError(f"{where}: corridor.daily weekly_split needs kappa")
        ce = raw.get("confirmed_extends", False)
        if not isinstance(ce, bool):
            raise ValueError(f"{where}: corridor.confirmed_extends must be true or false")
        try:
            return cls(q=num("q"), lookback_days=integer("lookback_days"),
                       floor_24h=num("floor_24h") if floor_mode == "fixed" else 0.0,
                       floor_7d=num("floor_7d") if floor_mode == "fixed" else 0.0, multipliers=tuple(mult),
                       holiday_weeks=(hw[0], hw[1]), n_boot=integer("n_boot"), confidence=num("confidence"),
                       confirm_fail_counts_as_decline=raw["confirm_fail_counts_as_decline"], floor_mode=floor_mode,
                       floor_percentile=pct, daily_mode=daily_mode, kappa=kappa, confirmed_extends=ce)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None

    def check_ready(self) -> None:
        if self.multipliers and self.fitted is None:
            raise ValueError("corridor multipliers are not fitted: call corridor.prepare(tm, df) first")
        if self.floor_mode == "population_percentile":
            f = self.floor_fit
            if f is None or f.q != self.q or f.lookback_days != self.lookback_days or \
                    f.percentile != self.floor_percentile:
                raise ValueError("the population floor is not fitted for this q / lookback / percentile: call "
                                 "corridor.prepare(tm, df) first")
        if self.daily_mode == "weekly_split" and self.split is None:
            raise ValueError("the weekly split s(j) of the V1 daily corridor is not fitted: call "
                             "corridor.prepare(tm, df) first")

    def public(self) -> dict:
        """Parameters for reports (without the fitted estimates)."""
        d = {k: getattr(self, k) for k in _KEYS}
        d["multipliers"], d["holiday_weeks"] = list(self.multipliers), list(self.holiday_weeks)
        if self.floor_mode == "fixed":
            d["floor"] = {"mode": "fixed"}
        else:
            d["floor"] = {"mode": self.floor_mode, "percentile": float(self.floor_percentile)}
            d["floor_24h"] = d["floor_7d"] = None
        d["daily"] = ({"mode": "quantile"} if self.daily_mode == "quantile"
                      else {"mode": self.daily_mode, "kappa": float(self.kappa)})
        d["confirmed_extends"] = bool(self.confirmed_extends)
        return d

    def variant(self) -> str:
        """Short name: base, V1(κ=…), V2, V1(κ=…)+V2."""
        parts = [f"V1(κ={self.kappa:g})"] if self.daily_mode == "weekly_split" else []
        parts += ["V2"] if self.confirmed_extends else []
        return "+".join(parts) or "base"


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
                       estimates=tuple(est), holiday_weeks=tuple(cfg.holiday_weeks), confidence=float(cfg.confidence))


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


def estimate_weekly_split(daily: pd.Series) -> WeeklySplit:
    """V1's s(j) from a daily population spend series (index: days): the mean spend of the days of weekday j divided
    by the sum of the seven means (Σ s = 1). Only the weekday of a day is used (no absolute dates); means, not sums,
    so an incomplete week does not tilt the shares."""
    days = pd.DatetimeIndex(daily.index)
    v = np.asarray(daily, dtype=np.float64)
    wd = np.asarray(days.weekday)
    means, n = [], []
    for j in range(7):
        m = wd == j
        if not m.any():
            raise ValueError(f"no day of weekday {j} in the series")
        means.append(float(v[m].mean()))
        n.append(int(m.sum()))
    tot = sum(means)
    if not tot > 0:
        raise ValueError("no spend in the series")
    return WeeklySplit(shares=tuple(x / tot for x in means), n_days=tuple(n))


def estimate_floor(df: pd.DataFrame, q: float, lookback_days: int, percentile: float) -> FloorFit:
    """The statistical floor (ruling R28): for every card ESTABLISHED at the fit end (its first legitimate fit-period
    operation at least ``lookback_days`` before ``VAL_START``), its own corridor quantile
    ``Q_W(q) = SpendHistory.window_quantile(fit_end, L, W, q, lag 0)`` – the q-quantile of the rolling-W sums after
    each legitimate fit operation in ``[fit_end − L, fit_end)`` – for W = 24 h and 7 d; the floor of each window is
    the ``percentile``-th percentile (``np.percentile``) of these values over the established cards. Fit period only;
    the fraud label selects the legitimate operations of this population estimate (as for the multipliers), never
    the per-card profile."""
    from .data import VAL_START, period
    m = (period(df) == "fit").to_numpy() & (df["is_fraud"].to_numpy() == 0)
    sub = df.loc[m, ["cc_num", "ts", "amt"]].sort_values(["cc_num", "ts"], kind="stable")
    fit_end = pd.Timestamp(VAL_START).value
    L = int(lookback_days) * DAY_NS
    q24, q7 = [], []
    for _, g in sub.groupby("cc_num", sort=True):
        t = g["ts"].to_numpy(dtype="datetime64[ns]").view("int64")
        if t[0] > fit_end - L:
            continue  # not established at the fit end
        h = SpendHistory(t, g["amt"].to_numpy(dtype=np.float64))
        q24.append(h.window_quantile(fit_end, L, WINDOWS_NS["24h"], float(q), 0))
        q7.append(h.window_quantile(fit_end, L, WINDOWS_NS["7d"], float(q), 0))
    if not q7:
        raise ValueError(f"no established card (>= {lookback_days} days of legitimate fit history) for the floor")
    return FloorFit(q=float(q), lookback_days=int(lookback_days), percentile=float(percentile),
                    floor_24h=float(np.percentile(q24, percentile)), floor_7d=float(np.percentile(q7, percentile)),
                    n_cards=len(q7), fit_end=pd.Timestamp(fit_end).isoformat())


def prepare(tm, df: pd.DataFrame):
    """``tm`` with its corridor fitted on ``df``: the multipliers (if enabled and not fitted yet), the V1 weekly split
    (if ``weekly_split`` and not fitted yet) and the population floor (if that mode is set and it is not fitted for
    the current q / lookback / percentile – the floor depends on q, so a sweep refits it per q). Unchanged when there
    is no corridor or nothing is missing."""
    c = getattr(tm, "corridor", None)
    if c is None:
        return tm
    upd = {}
    if c.multipliers and c.fitted is None:
        upd["fitted"] = estimate_multipliers(df, c)
    if c.daily_mode == "weekly_split" and c.split is None:
        upd["split"] = estimate_weekly_split(daily_population_spend(df, "fit"))
    if c.floor_mode == "population_percentile":
        f = c.floor_fit
        if f is None or f.q != c.q or f.lookback_days != c.lookback_days or f.percentile != c.floor_percentile:
            upd["floor_fit"] = estimate_floor(df, c.q, c.lookback_days, c.floor_percentile)
    if not upd:
        return tm
    return dataclasses.replace(tm, corridor=dataclasses.replace(c, **upd))


# --- corridor and decision -------------------------------------------------------------------------------------

def weekday_of(day_ns: int) -> int:
    """Weekday (0 = Monday) of the calendar day of ``day_ns`` (1970-01-01 was a Thursday)."""
    return int((int(day_ns) // DAY_NS + 3) % 7)


def established(cfg: CorridorConfig, spend: SpendHistory, day_ns: int) -> bool:
    """R28: the card has at least ``lookback_days`` of history before day d (its first passed operation is at or
    before ``d − L``). Only operations before the episode can make it so (L >= 7 days), so the floor stays
    attack-independent."""
    return len(spend) > 0 and int(spend.t[0]) <= int(day_ns) - cfg.lookback_days * DAY_NS


def _own_and_floor(cfg: CorridorConfig, spend: SpendHistory, day: int, window: str) -> tuple[float, float]:
    q = spend.window_quantile(day, cfg.lookback_days * DAY_NS, WINDOWS_NS[window], cfg.q, PROFILE_LAG_NS)
    if cfg.floor_mode == "fixed":
        fl = cfg.floor_24h if window == "24h" else cfg.floor_7d
    elif established(cfg, spend, day):
        fl = 0.0
    else:
        fl = cfg.floor_fit.floor_24h if window == "24h" else cfg.floor_fit.floor_7d
    return q, fl


def limit_cents(cfg: CorridorConfig, spend: SpendHistory, day_ns: int, window: str) -> int:
    """``C_W`` in whole cents (rounded down) for the calendar day of ``day_ns`` (lagged profile, module docstring).
    V1 (``daily_mode == "weekly_split"``): ``C_24h = floor(κ · s(weekday) · C_7d)`` from the 7-day corridor of the
    same day in cents (no own 24 h quantile, floor or multiplier)."""
    day = int(day_ns) - int(day_ns) % DAY_NS
    if window == "24h" and cfg.daily_mode == "weekly_split":
        c7 = limit_cents(cfg, spend, day, "7d")
        return int(math.floor(cfg.kappa * cfg.split.shares[weekday_of(day)] * c7 + 1e-6))
    q, fl = _own_and_floor(cfg, spend, day, window)
    m = cfg.fitted.factor(day, window, cfg.multipliers) if cfg.multipliers else 1.0
    return int(math.floor(max(fl, q) * m * 100 + 1e-6))


def floor_active(cfg: CorridorConfig, spend: SpendHistory, day_ns: int, window: str) -> bool:
    """The floor is above the card's own quantile on that day (it sets the corridor). Never for V1's 24 h window."""
    if window == "24h" and cfg.daily_mode == "weekly_split":
        return False
    day = int(day_ns) - int(day_ns) % DAY_NS
    q, fl = _own_and_floor(cfg, spend, day, window)
    return fl > q


def limits_cents(cfg: CorridorConfig, state: CardState, ts) -> tuple[int, int]:
    """The base corridors ``(C_24h, C_7d)`` of ts's day (attack-independent; no V2 extension)."""
    t = pd.Timestamp(ts).value
    return limit_cents(cfg, state.spend, t, "24h"), limit_cents(cfg, state.spend, t, "7d")


def effective_limits_cents(cfg: CorridorConfig, state: CardState, ts, base: tuple[int, int] | None = None
                           ) -> tuple[int, int]:
    """The corridors the rule compares with (the single source, used by ``check``): the base ones (``base`` if given,
    else ``limits_cents``), plus with V2 (``confirmed_extends``) the owner-confirmed cents in ``(t − W, t]`` of each
    window (``C'_W``)."""
    t = pd.Timestamp(ts).value
    c24, c7 = limits_cents(cfg, state, t) if base is None else base
    if cfg.confirmed_extends:
        c24 += state.confirmed_sum_cents(t, WINDOWS_NS["24h"])
        c7 += state.confirmed_sum_cents(t, WINDOWS_NS["7d"])
    return c24, c7


def decision_limits(cfg: CorridorConfig, state: CardState, ts) -> tuple[int, int, int, int]:
    """``(C_24h, C_7d, C'_24h, C'_7d)`` at ts: the base corridors (recorded by the bank, attack-independent) and the
    ones ``check`` compares with."""
    t = pd.Timestamp(ts).value
    base = limits_cents(cfg, state, t)
    return (*base, *effective_limits_cents(cfg, state, t, base))


def check(cfg: CorridorConfig, state: CardState, ts, amt: float) -> tuple[bool, int, int]:
    """Rule 2 with the base corridors used: ``(S_W(t⁻) + a <= C'_W(t) for both windows, C_24h, C_7d)`` (whole
    cents; ``C'_W = C_W`` without V2; limits from ``decision_limits``)."""
    t = pd.Timestamp(ts).value
    a = int(round(float(amt) * 100))
    c24, c7, e24, e7 = decision_limits(cfg, state, t)
    ok = (state.auto_sum_cents(t, WINDOWS_NS["24h"]) + a <= e24
          and state.auto_sum_cents(t, WINDOWS_NS["7d"]) + a <= e7)
    return ok, c24, c7


def within(cfg: CorridorConfig, state: CardState, ts, amt: float) -> bool:
    """Rule 2: ``S_W(t⁻) + a <= C'_W(t)`` for both windows (whole cents)."""
    return check(cfg, state, ts, amt)[0]


def _blocks(t0: int, window_days: int) -> list[tuple[int, int]]:
    return [(t0 + i * DAY_NS, t0 + (i + 1) * DAY_NS) for i in range(int(window_days))]


def episode_bound(cfg: CorridorConfig, state: CardState, start, window_days: int) -> dict:
    """The guaranteed bounds of an episode starting at ``start``, from the state right before it (``ValueError`` if
    the state has seen anything at or after ``start``; ``ValueError`` if D > W, the Propositions need D <= W):

    * ``bound_cents = max_{t ∈ [start, start + D)} C_7d`` (Proposition 1), the per-day values ``c7_by_day`` and
      ``c24_by_day`` ({day ns: cents} for every calendar day the window touches; the bank's decisions use exactly
      these – ``check_used_corridor``) and the tighter ``headroom_cents`` (module docstring);
    * Proposition 2: ``block_c24_cents[i]`` = max of C_24h over the 24-hour block ``B_{i+1} = [t0 + i·24 h,
      t0 + (i+1)·24 h)`` (C is constant per calendar day, so the max over the <= 2 days the block touches) and
      ``b_h_cents[h−1] = B(h) = min(bound_cents, Σ_{i<=h} block max)``, h = 1..D;
    * ``floor_touched``: on some day of the window the floor set a corridor (``floor_active``)."""
    D = int(window_days) * DAY_NS
    W = WINDOWS_NS["7d"]
    if D > W:
        raise ValueError(f"the guarantee needs D <= W: episode {window_days} days, window 7 days")
    t0 = pd.Timestamp(start).value
    if state.now is not None and state.now > t0:
        raise ValueError(f"the bound needs the state right before the episode start {pd.Timestamp(t0)}; "
                         f"the state is at {pd.Timestamp(state.now)}")
    if len(state.spend) and int(state.spend.t[-1]) >= t0:
        raise ValueError("the bound needs the state right before the episode start: the profile has operations at "
                         "or after it")
    end = t0 + D
    head = 0
    by_day, by_day24 = {}, {}
    touched = False
    d = t0 - t0 % DAY_NS
    while d < end:
        c = limit_cents(cfg, state.spend, d, "7d")
        seg_end = min(d + DAY_NS, end)
        s_min = state.auto_sum_cents(seg_end - 1, W)  # owner ops still in (t − W, t] as t → seg_end⁻
        by_day[int(d)] = int(c)
        by_day24[int(d)] = int(limit_cents(cfg, state.spend, d, "24h"))
        touched = touched or floor_active(cfg, state.spend, d, "7d") or floor_active(cfg, state.spend, d, "24h")
        head = max(head, c - s_min)
        d += DAY_NS
    bound = int(max(by_day.values()))
    blocks = [max(by_day24[bs - bs % DAY_NS], by_day24[(be - 1) - (be - 1) % DAY_NS])
              for bs, be in _blocks(t0, window_days)]
    return {"bound_cents": bound, "headroom_cents": int(max(0, head)), "c7_by_day": by_day, "c24_by_day": by_day24,
            "block_c24_cents": blocks, "b_h_cents": [min(bound, x) for x in np.cumsum(blocks).tolist()],
            "floor_touched": bool(touched)}


def _ext_max(c_by_day: dict, conf, w: int, lo: int, hi: int) -> int:
    """max over t ∈ [lo, hi) of ``c_by_day[day(t)] + Σ{cents_j : t − w < t_j <= t}``: both terms are step functions
    that only rise at ``lo``, a midnight or a confirmation time, so the max is at one of these points."""
    cands = {lo} | {t for t, _ in conf if lo <= t < hi}
    d = lo - lo % DAY_NS + DAY_NS
    while d < hi:
        cands.add(d)
        d += DAY_NS
    return max(c_by_day[t - t % DAY_NS] + sum(c for tj, c in conf if tj <= t < tj + w) for t in cands)


def posthoc_bound(bound: dict, confirmed_ops, start, window_days: int) -> dict:
    """V2 (``confirmed_extends``): the bounds of ``episode_bound`` with ``C'_W = C_W + owner-confirmed cents in
    (t − W, t]``, computed AFTER the episode from the owner's actual confirmations ``confirmed_ops`` ((t ns, cents),
    before and during the episode). Post-hoc but attack-independent: only the owner's actions enter it.
    ``bound_cents = max_t C'_7d``, ``b_h_cents`` from the block maxima of ``C'_24h``, ``headroom_cents`` = the
    pre-episode headroom + every confirmed cent in ``(t0 − W, t0 + D)``. Without confirmations = ``bound``."""
    t0 = pd.Timestamp(start).value
    end = t0 + int(window_days) * DAY_NS
    conf = tuple((int(t), int(c)) for t, c in confirmed_ops if c)
    b7 = _ext_max(bound["c7_by_day"], conf, WINDOWS_NS["7d"], t0, end)
    blocks = [_ext_max(bound["c24_by_day"], conf, WINDOWS_NS["24h"], bs, be) for bs, be in _blocks(t0, window_days)]
    extra = sum(c for t, c in conf if t0 - WINDOWS_NS["7d"] < t < end)
    return {**bound, "bound_cents": int(b7), "headroom_cents": int(bound["headroom_cents"] + extra),
            "block_c24_cents": blocks, "b_h_cents": [min(b7, x) for x in np.cumsum(blocks).tolist()]}


def auto_by_block(auto_ops, start, window_days: int) -> list[int]:
    """Cumulative auto-approved cents after each 24-hour block of the episode (Proposition 2's A_auto(h), h = 1..D);
    ``ValueError`` for an operation outside ``[t0, t0 + D)``."""
    t0 = pd.Timestamp(start).value
    per = [0] * int(window_days)
    for t, c in auto_ops:
        i = (int(t) - t0) // DAY_NS
        if not 0 <= i < len(per):
            raise ValueError(f"auto-approved operation at {pd.Timestamp(int(t))} outside the episode window")
        per[i] += int(c)
    return np.cumsum(per).astype(int).tolist()


def check_used_corridor(bound: dict, used, *, used24=(), inside_only: bool = False) -> None:
    """``RuntimeError`` unless every corridor the bank used during an episode equals the pre-episode per-day value of
    ``episode_bound``: 7-day (``used``: (day ns, cents) pairs, ``EpisodeOutcome.corridor_c7``) and 24 h (``used24``,
    ``EpisodeOutcome.corridor_c24``; Proposition 2's bound is built from them). A used day outside the episode window
    is an error too, unless ``inside_only`` (the clean leg replays real operations, which may run past the window;
    the guarantee covers the window only)."""
    for name, by_day, pairs in (("", bound["c7_by_day"], used), ("24 h ", bound.get("c24_by_day", {}), used24)):
        for day, c in pairs:
            if day not in by_day:
                if inside_only:
                    continue
                raise RuntimeError(f"{name}corridor used on day {pd.Timestamp(day).date()} outside the episode window")
            if by_day[day] != c:
                raise RuntimeError(f"{name}corridor on day {pd.Timestamp(day).date()} was {c} cents, the pre-episode "
                                   f"value is {by_day[day]}")


def batch_state(t_ns: np.ndarray, amt: np.ndarray, auto: np.ndarray, passed: np.ndarray,
                cfg: CorridorConfig, confirmed: np.ndarray | None = None) -> dict:
    """Batch path of the corridor state for one card's time-ordered stream with KNOWN decisions: per operation i,
    ``s24``/``s7`` = auto-approved cents of the operations before i IN STREAM ORDER with time in ``(t_i − W, t_i]``
    (earlier operations of the same second count, later ones of that second do not) and ``c24``/``c7`` = the
    corridor the rule compares with: the base corridor of t_i's day from the passed operations (lagged profile),
    plus with V2 the owner-confirmed cents (``confirmed``) of the earlier operations in the same window. Must equal
    the serving path (``CardState.auto_sum_cents`` + ``effective_limits_cents`` with ``push(…, auto_approved,
    passed, confirmed)``)."""
    t = np.asarray(t_ns, dtype=np.int64)
    whole = np.rint(np.asarray(amt, dtype=np.float64) * 100).astype(np.int64)
    n = t.size
    idx = np.arange(n)
    conf = np.zeros(n, dtype=bool) if confirmed is None else np.asarray(confirmed, dtype=bool)

    def win(cents):
        cum = np.concatenate([[0], np.cumsum(cents)])
        res = {}
        for name, w in (("24h", WINDOWS_NS["24h"]), ("7d", WINDOWS_NS["7d"])):
            lo = np.minimum(np.searchsorted(t, t - w, "right"), idx)
            res[name] = cum[idx] - cum[lo]
        return res
    s = win(whole * np.asarray(auto, dtype=bool))
    k = win(whole * conf) if cfg.confirmed_extends else {"24h": np.zeros(n, np.int64), "7d": np.zeros(n, np.int64)}
    out = {"s24": s["24h"].tolist(), "s7": s["7d"].tolist()}
    pm = np.asarray(passed, dtype=bool)
    pt, pa = t[pm], np.asarray(amt, dtype=np.float64)[pm]
    c24, c7 = [], []
    cache = {}
    for i in range(n):
        day = int(t[i] - t[i] % DAY_NS)
        if day not in cache:
            # the passed ops before the day: the quantile reads those before d − lag, «established» the first one
            # (an op that makes the card established is at or before d − L < d, so before every op of the day)
            j = int(np.searchsorted(pt, day, "left"))
            h = SpendHistory(pt[:j], pa[:j])
            cache[day] = (limit_cents(cfg, h, day, "24h"), limit_cents(cfg, h, day, "7d"))
        c24.append(cache[day][0] + int(k["24h"][i]))
        c7.append(cache[day][1] + int(k["7d"][i]))
    out["c24"], out["c7"] = c24, c7
    return out


def bound_stats(per_target: list[dict], rho: float) -> dict:
    """Summary of the per-target bounds for reports (dollars and shares of A)."""
    b = np.array([t["corridor_bound"] for t in per_target], dtype=np.float64)
    h = np.array([t["corridor_headroom"] for t in per_target], dtype=np.float64)
    a = np.array([t["amount"] for t in per_target], dtype=np.float64)

    def s(x):
        q25, q75, q90 = np.quantile(x, [0.25, 0.75, 0.9])
        return {"mean": float(x.mean()), "median": float(np.median(x)), "max": float(x.max()),
                "q25": float(q25), "q75": float(q75), "q90": float(q90)}
    out = {"bound": s(b), "bound_share": s(b / a), "headroom": s(h), "headroom_share": s(h / a),
           "share_targets_bound_below_rho_a": float(np.mean(b < rho * a))}
    if per_target and all("b_h" in t for t in per_target):
        bh = np.array([t["b_h"] for t in per_target], dtype=np.float64)  # targets × h, $

        def cols(x):
            return {"mean": x.mean(axis=0).tolist(), "median": np.median(x, axis=0).tolist(),
                    **{k: np.quantile(x, v, axis=0).tolist() for k, v in (("q25", 0.25), ("q75", 0.75),
                                                                           ("q90", 0.9))}}
        out["b_h"], out["b_h_share"] = cols(bh), cols(bh / a[:, None])
    if per_target and all("floor_touched" in t for t in per_target):
        out["n_floor_touched"] = int(sum(bool(t["floor_touched"]) for t in per_target))
    return out
