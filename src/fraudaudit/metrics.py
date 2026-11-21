"""Metrics: Clopper-Pearson upper bound, the threshold for a target FPR, bootstrap intervals of VER
and of the paired uplift VER_adv − VER_clean.

``clopper_pearson_upper`` and ``threshold_at_fpr`` are ported unchanged from fraudgate
(``src/fraudgate/metrics.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_curve


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided upper confidence bound for the proportion k/n."""
    if n == 0:
        return 1.0
    return 1.0 if k == n else float(beta.ppf(confidence, k + 1, n - k))


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold tau with FPR (decision ``p >= tau``) not above target_fpr."""
    fpr, _, thr = roc_curve(y, p)
    return float(thr[np.flatnonzero(fpr <= target_fpr)[-1]])


def ver_ci(vers: np.ndarray, confidence: float = 0.95, n_boot: int = 2000,
           seed: int = 42) -> tuple[float, float, float]:
    """(mean, lower, upper): mean VER over episodes and its two-sided percentile bootstrap
    interval. Episodes are resampled with replacement ``n_boot`` times (``default_rng(seed)``)."""
    v = np.asarray(vers, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("ver_ci needs at least one episode")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return _bootstrap_mean(v, confidence, n_boot, seed)


def _boot_means(v: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Means of ``n_boot`` resamples of ``v`` with replacement (``default_rng(seed)``)."""
    idx = np.random.default_rng(seed).integers(0, v.size, size=(n_boot, v.size))
    return v[idx].mean(axis=1)


def _bootstrap_mean(v: np.ndarray, confidence: float, n_boot: int, seed: int,
                    one_sided: bool = False) -> tuple[float, float, float]:
    means = _boot_means(v, n_boot, seed)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    if one_sided:
        hi = np.quantile(means, confidence)
    return float(v.mean()), float(lo), float(hi)


def _paired(ver_adv, ver_clean) -> np.ndarray:
    a = np.asarray(ver_adv, dtype=np.float64).reshape(-1)
    c = np.asarray(ver_clean, dtype=np.float64).reshape(-1)
    if a.shape != c.shape:
        raise ValueError(f"paired arrays differ in length: {a.size} vs {c.size}")
    if a.size == 0:
        raise ValueError("uplift_ci needs at least one episode")
    return a - c


def uplift_ci(ver_adv: np.ndarray, ver_clean: np.ndarray, confidence: float = 0.95, n_boot: int = 2000,
              seed: int = 42, *, one_sided: bool = False) -> tuple[float, float, float]:
    """(mean, lower, upper) of the per-episode difference ``ver_adv_i − ver_clean_i``: a paired
    percentile bootstrap (the same resampled episodes for both legs, ``default_rng(seed)``).

    Two-sided (default): ``lower``/``upper`` are the (1 − c)/2 and (1 + c)/2 percentiles. ``one_sided=True``
    (the gate, ruling R14: it only tests «uplift ≤ δ»): ``upper`` is the one-sided upper bound at level c,
    the c-th percentile (95th for c = 0.95), so it is never above the two-sided upper bound; ``lower`` stays
    the two-sided lower bound (descriptive only, not part of the one-sided statement, kept finite for JSON).
    The gate rule «upper bound of VER_adv − VER_clean ≤ δ» uses ``upper``."""
    d = _paired(ver_adv, ver_clean)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return _bootstrap_mean(d, confidence, n_boot, seed, one_sided)


def uplift_se(ver_adv: np.ndarray, ver_clean: np.ndarray, n_boot: int = 2000, seed: int = 42) -> float:
    """Bootstrap standard error of the mean uplift: the standard deviation (ddof=1) of the same resampled
    means ``uplift_ci`` uses (same ``seed``)."""
    d = _paired(ver_adv, ver_clean)
    return float(_boot_means(d, n_boot, seed).std(ddof=1))


# --- false blocks of legitimate cards (bank-policy Pareto front) ----------------------------------------------


@dataclass(frozen=True)
class FalseBlockResult:
    """Legitimate behaviour of the test period under one bank policy.

    A (card, window) pair exists when the card has at least one legitimate operation in that calendar window
    (``window_days`` from the test start). ``share_blocked_card_windows`` is the metric of the Pareto front."""
    share_blocked_card_windows: float
    share_blocked_cards: float          # cards blocked in at least one window / cards with legitimate test ops
    mean_declines_per_card_window: float
    decline_rate: float                 # declined / decided operations (operations after a block are not decided)
    n_card_windows: int
    n_blocked_card_windows: int
    n_cards: int
    n_blocked_cards: int
    n_ops: int                          # legitimate test operations replayed
    n_decided_ops: int
    n_declines: int
    window_days: int
    max_declines: int | None
    tau: float
    tau_after_decline: float | None


class _TauServing:
    """``inner`` with the decision ``p >= tau`` for a given ``tau`` (the bank policy's threshold)."""

    def __init__(self, inner, tau: float) -> None:
        self.inner, self.tau = inner, float(tau)

    def decide(self, state, tx):
        from .serving.core import Decision
        d = self.inner.decide(state, tx)
        return Decision(p=d.p, declined=bool(d.p >= self.tau))


_FB: dict = {}


def _fb_init(model_dir, params) -> None:
    from .serving.core import Serving
    from .skew import _load_model
    _FB["model"] = Serving(_load_model(model_dir))
    _FB["params"] = params


def _fb_job(args):
    return _card_windows(args, _FB["model"], _FB["params"])


def _fb_cached_job(args):
    """One card from cached scores: ``(g, test_start, p, params)``."""
    g, test_start, p, params = args
    seq = _SeqServing(p)
    out = _card_windows((g, test_start), seq, params)
    if seq.i != len(p):
        raise RuntimeError(f"card {int(g['cc_num'].iloc[0])}: {seq.i} of {len(p)} cached scores consumed")
    return out


def _card_windows(args, serving, params) -> list[tuple]:
    """Replay of one card: ``[(window, n_ops, n_decided, n_declines, blocked)]``, in window order."""
    from types import SimpleNamespace

    from .episode import _Bank
    from .serving.state import CardState, Tx, profile_of
    g, test_start = args
    tau, max_declines, tau2, window_ns = params
    prof = profile_of(g, int(g["cc_num"].iloc[0]))
    state = CardState.from_history(prof, g[g["ts"] < test_start], at=test_start)
    test = g[g["ts"] >= test_start]
    t_ns = test["ts"].to_numpy(dtype="datetime64[ns]").view("int64")
    win = (t_ns - test_start.value) // window_ns
    srv = _TauServing(serving, tau)
    out = []
    bank = None
    cur = None
    for w, r in zip(win.tolist(), test.itertuples(index=False)):
        tx = Tx(ts=pd.Timestamp(r.ts), amt=float(r.amt), merchant=str(r.merchant), category=str(r.category),
                merch_lat=float(r.merch_lat), merch_long=float(r.merch_long))
        if w != cur:
            if bank is not None:
                state = bank.state
                out.append((cur, n_ops, len(bank.decisions), bank.n_declines, bank.blocked))
            cur, n_ops = w, 0
            bank = _Bank(srv, SimpleNamespace(episode=SimpleNamespace(amount=1.0), state=state), max_declines, 1.0,
                         tau2)
        n_ops += 1
        if bank.blocked:
            bank.state.push(tx)  # the real operation still happens (elsewhere); only the decision is not asked
            skip = getattr(serving, "skip", None)
            if skip is not None:  # cached scores: this operation's score is not used
                skip()
        else:
            bank.submit(tx)
    if bank is not None:
        out.append((cur, n_ops, len(bank.decisions), bank.n_declines, bank.blocked))
    return out


def false_block_rate(df: pd.DataFrame, model_or_serving, tau: float, max_declines: int | None,
                     window_days: int = 30, *, tau_after_decline: float | None = None,
                     n_jobs: int | None = None, model_dir=None, scores: dict | None = None) -> FalseBlockResult:
    """Share of legitimate (card, window) pairs a bank policy would block.

    Replay: the state of every card is built from all its operations before the test start (history);
    its LEGITIMATE test-period operations are then decided one by one through the serving path in time
    order, every one pushed into the state (the bank sees the attempt). Fraud operations of the test period are
    excluded from the replay entirely, so the metric measures purely legitimate behaviour. Calendar windows
    of ``window_days`` start at the test start (the earliest test operation); the bank is reset per window with the
    same semantics as ``run_episode`` (``_Bank``): declines are counted, the threshold switches to
    ``tau_after_decline`` after the first decline in the window, the card is blocked at ``max_declines``
    declines (``None`` = never blocked, the decline rate is still reported). After a block, the rest of that
    window's operations are not decided (they are pushed into the state without a decision). The decision is
    ``p >= tau`` (the policy's own ``tau``).

    ``model_or_serving``: a ``Serving`` or a fitted model. ``n_jobs > 1`` needs ``model_dir`` (every worker
    loads its own model); results are identical for any ``n_jobs``. ``scores`` (the dict of ``legit_test_scores``
    of the same model; ``model_or_serving`` and ``model_dir`` are then unused): the cached p of every legitimate test
    operation – the state sees every operation whatever the decision, so p does not depend on τ or the reaction –,
    replayed in the same order (a blocked operation's score is skipped); identical to the live replay."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    from .data import period
    from .serving.core import Serving
    from .skew import _REPLAY_COLS
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    if max_declines is not None and max_declines < 1:
        raise ValueError("max_declines must be None or >= 1")
    if tau_after_decline is not None and not (0 < tau_after_decline < tau):
        raise ValueError("tau_after_decline must be None or in (0, tau)")
    per = period(df)
    is_test = (per == "test").to_numpy()
    if not is_test.any():
        raise ValueError("no test-period operations")
    test_start = pd.Timestamp(df.loc[is_test, "ts"].min())
    keep = ~(is_test & (df["is_fraud"].to_numpy() == 1))
    sub = df.loc[keep, _REPLAY_COLS].sort_values(["ts", "row"], kind="stable")
    cards = sub.loc[sub["ts"] >= test_start, "cc_num"].unique()
    sub = sub[sub["cc_num"].isin(cards)]
    groups = [(g.reset_index(drop=True), test_start) for _, g in sub.groupby("cc_num", sort=True)]
    params = (float(tau), max_declines, tau_after_decline, window_days * 86_400 * 10**9)
    jobs = (os.cpu_count() or 1) if n_jobs is None else n_jobs
    if scores is not None:
        args = [(g, ts, scores[int(g["cc_num"].iloc[0])], params) for g, ts in groups]
        if jobs > 1 and len(groups) > 1:
            with ProcessPoolExecutor(max_workers=min(jobs, len(groups)), mp_context=mp.get_context("spawn")) as ex:
                parts = list(ex.map(_fb_cached_job, args, chunksize=max(1, len(groups) // (4 * jobs))))
        else:
            parts = [_fb_cached_job(a) for a in args]
    elif jobs > 1 and len(groups) > 1:
        if model_dir is None:
            raise ValueError("n_jobs > 1 needs model_dir: every worker loads its own model")
        with ProcessPoolExecutor(max_workers=min(jobs, len(groups)), mp_context=mp.get_context("spawn"),
                                 initializer=_fb_init, initargs=(str(model_dir), params)) as ex:
            parts = list(ex.map(_fb_job, groups, chunksize=max(1, len(groups) // (4 * jobs))))
    else:
        if model_dir is not None:
            from .skew import _load_model
            serving = Serving(_load_model(model_dir))
        else:
            serving = model_or_serving if hasattr(model_or_serving, "decide") else Serving(model_or_serving)
        parts = [_card_windows(a, serving, params) for a in groups]
    cw = [t for p in parts for t in p]
    n_cw = len(cw)
    n_blocked = sum(1 for t in cw if t[4])
    n_blocked_cards = sum(1 for p in parts if any(t[4] for t in p))
    n_decided = sum(t[2] for t in cw)
    n_decl = sum(t[3] for t in cw)
    return FalseBlockResult(
        share_blocked_card_windows=n_blocked / n_cw, share_blocked_cards=n_blocked_cards / len(parts),
        mean_declines_per_card_window=n_decl / n_cw, decline_rate=n_decl / n_decided if n_decided else 0.0,
        n_card_windows=n_cw, n_blocked_card_windows=n_blocked, n_cards=len(parts),
        n_blocked_cards=n_blocked_cards, n_ops=sum(t[1] for t in cw), n_decided_ops=n_decided, n_declines=n_decl,
        window_days=int(window_days), max_declines=max_declines, tau=float(tau),
        tau_after_decline=tau_after_decline)


# --- confirmation burden of the spending corridor on the legitimate stream (Task 22) --------------------------


@dataclass(frozen=True)
class BurdenResult:
    """Owner-confirmation burden of a spending corridor on the legitimate test-period operations.

    Every legitimate test operation is decided (no blocking, ``max_declines=None``), so ``share_confirm_ops`` =
    confirmations / legitimate operations; a card-day exists when the card has a legitimate test operation that
    calendar day, ``share_confirm_card_days`` = card-days with at least one confirmation / card-days."""
    share_confirm_ops: float
    share_confirm_card_days: float
    n_ops: int
    n_confirms: int
    n_card_days: int
    n_confirm_card_days: int
    n_declines: int
    decline_rate: float
    n_cards: int
    tau: float


def _legit_groups(df: pd.DataFrame) -> tuple[list, pd.Timestamp]:
    """Per card (sorted by card): its operations without test-period fraud, in (ts, row) order, and the test start."""
    from .data import period
    from .skew import _REPLAY_COLS
    per = period(df)
    is_test = (per == "test").to_numpy()
    if not is_test.any():
        raise ValueError("no test-period operations")
    test_start = pd.Timestamp(df.loc[is_test, "ts"].min())
    keep = ~(is_test & (df["is_fraud"].to_numpy() == 1))
    cols = _REPLAY_COLS + ["is_fraud"]
    sub = df.loc[keep, cols].sort_values(["ts", "row"], kind="stable")
    cards = sub.loc[sub["ts"] >= test_start, "cc_num"].unique()
    sub = sub[sub["cc_num"].isin(cards)]
    return [g.reset_index(drop=True) for _, g in sub.groupby("cc_num", sort=True)], test_start


def _test_txs(g: pd.DataFrame, test_start: pd.Timestamp):
    from .serving.state import CardState, Tx, profile_of
    prof = profile_of(g, int(g["cc_num"].iloc[0]))
    state = CardState.from_history(prof, g[g["ts"] < test_start], at=test_start)
    test = g[g["ts"] >= test_start]
    txs = [Tx(ts=pd.Timestamp(r.ts), amt=float(r.amt), merchant=str(r.merchant), category=str(r.category),
              merch_lat=float(r.merch_lat), merch_long=float(r.merch_long)) for r in test.itertuples(index=False)]
    return state, txs


def _card_scores(g, test_start, serving) -> np.ndarray:
    """p of every legitimate test operation of one card, every operation pushed (no decision affects the state)."""
    state, txs = _test_txs(g, test_start)
    out = np.empty(len(txs), dtype=np.float64)
    for i, tx in enumerate(txs):
        out[i] = serving.decide(state, tx).p
        state.push(tx)
    return out


def _sc_job(args):
    return _card_scores(args[0], args[1], _FB["model"])


def legit_test_scores(df: pd.DataFrame, model_or_serving, *, n_jobs: int | None = None,
                      model_dir=None) -> dict[int, np.ndarray]:
    """``{card: p of its legitimate test operations in replay order}``: the scores of ``confirmation_burden``'s
    replay, computed once per model (they do not depend on τ or the corridor, because without blocking every
    operation enters the state). Parallel per card like ``false_block_rate``; identical for any ``n_jobs``."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    from .serving.core import Serving
    groups, test_start = _legit_groups(df)
    jobs = (os.cpu_count() or 1) if n_jobs is None else n_jobs
    if jobs > 1 and len(groups) > 1:
        if model_dir is None:
            raise ValueError("n_jobs > 1 needs model_dir: every worker loads its own model")
        with ProcessPoolExecutor(max_workers=min(jobs, len(groups)), mp_context=mp.get_context("spawn"),
                                 initializer=_fb_init, initargs=(str(model_dir), None)) as ex:
            parts = list(ex.map(_sc_job, [(g, test_start) for g in groups],
                                chunksize=max(1, len(groups) // (4 * jobs))))
    else:
        serving = model_or_serving if hasattr(model_or_serving, "decide") else Serving(model_or_serving)
        parts = [_card_scores(g, test_start, serving) for g in groups]
    return {int(g["cc_num"].iloc[0]): p for g, p in zip(groups, parts)}


class _SeqServing:
    """Replays cached scores in call order; ``skip`` passes over the score of an operation that is not decided."""

    def __init__(self, p: np.ndarray) -> None:
        self.p, self.i = p, 0

    def skip(self) -> None:
        self.i += 1

    def decide(self, state, tx):
        from .serving.core import Decision
        p = float(self.p[self.i])
        self.i += 1
        return Decision(p=p, declined=False)


def _card_burden(args) -> tuple:
    """One card: (n_ops, n_confirms, n_declines, n_card_days, n_confirm_card_days)."""
    from types import SimpleNamespace

    from .episode import _Bank
    g, test_start, scorer, tau, corridor = args
    state, txs = _test_txs(g, test_start)
    srv = _TauServing(scorer if hasattr(scorer, "decide") else _SeqServing(scorer), tau)
    bank = _Bank(srv, SimpleNamespace(episode=SimpleNamespace(amount=1.0), state=state), None, 1.0, None, corridor,
                 owner=True)
    days, cdays = set(), set()
    for tx in txs:
        before = bank.n_confirms
        bank.submit(tx)
        day = tx.ts.normalize()
        days.add(day)
        if bank.n_confirms > before:
            cdays.add(day)
    return len(txs), bank.n_confirms, bank.n_declines, len(days), len(cdays)


def confirmation_burden(df: pd.DataFrame, scores, tau: float, corridor, *, n_jobs: int | None = None) -> BurdenResult:
    """Owner-confirmation burden of ``corridor`` (a fitted ``corridor.CorridorConfig``) at threshold ``tau`` on the
    legitimate test-period operations (``BurdenResult``).

    Replay as ``false_block_rate`` (state from all operations before the test start, test-period fraud excluded,
    shared ``episode._Bank``) with the owner's semantics: a confirmation passes and is not auto-approved; every
    operation that passed (auto-approved or confirmed, not a model decline) feeds the lagged corridor profile, the
    history before the test start enters it whole (no fraud label is used for the profile; the label only selects
    which test operations form the legitimate stream); ``max_declines=None`` (the corridor's own cost, independent
    of blocking). ``scores``: a model or ``Serving`` (live, ``n_jobs`` must be 1) or the dict of
    ``legit_test_scores`` (cached; parallel per card, identical for any ``n_jobs``)."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    from .serving.core import Serving
    corridor.check_ready()
    groups, test_start = _legit_groups(df)
    live = not isinstance(scores, dict)
    if live:
        scorer = scores if hasattr(scores, "decide") else Serving(scores)
        args = [(g, test_start, scorer, float(tau), corridor) for g in groups]
    else:
        args = [(g, test_start, scores[int(g["cc_num"].iloc[0])], float(tau), corridor) for g in groups]
    jobs = (os.cpu_count() or 1) if n_jobs is None else n_jobs
    if jobs > 1 and len(groups) > 1 and not live:
        with ProcessPoolExecutor(max_workers=min(jobs, len(groups)), mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_card_burden, args, chunksize=max(1, len(groups) // (4 * jobs))))
    else:
        parts = [_card_burden(a) for a in args]
    n_ops, n_conf, n_decl, n_days, n_cdays = (sum(p[i] for p in parts) for i in range(5))
    return BurdenResult(share_confirm_ops=n_conf / n_ops if n_ops else 0.0,
                        share_confirm_card_days=n_cdays / n_days if n_days else 0.0, n_ops=n_ops,
                        n_confirms=n_conf, n_card_days=n_days, n_confirm_card_days=n_cdays, n_declines=n_decl,
                        decline_rate=n_decl / n_ops if n_ops else 0.0, n_cards=len(parts), tau=float(tau))
