"""Training-serving skew check: the batch feature path and the serving path must agree.

After the ML Test Score principle "training and serving features compute the same values"
(Breck et al., 2017). Sampled cards are replayed operation by operation through ``CardState.push``
(serving path) and compared with ``features(df, encoder)`` (batch path) on the same rows.

Tolerance: integer-valued columns (``cnt_*``, ``secs_since_prev``) and every base feature must be
exactly equal; only the three float means (``mean_amt_*``) get ``atol`` (+ 1e-6 relative, the same
as ``np.allclose`` in ``tests/test_serving.py``), because float32 rounding of a float64 mean may
differ by one ulp between a summation over a window and a running list.

Decisions are compared on the same two feature matrices with one ``predict_proba`` call each
(a batch call instead of one call per operation: the probabilities agree up to float32 kernel
noise, and a per-operation call would dominate the runtime).
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .features.columns import AGG_COLUMNS, FEATURE_COLUMNS
from .model.pipeline import features
from .serving.core import Serving
from .serving.state import CardState, Tx, profile_of

_MEAN_IDX = np.array([FEATURE_COLUMNS.index(c) for c in AGG_COLUMNS if c.startswith("mean_amt_")])
_RTOL = 1e-6
_REPLAY_COLS = ["cc_num", "ts", "row", "amt", "merchant", "category", "merch_lat", "merch_long",
                "gender", "dob", "lat", "long", "city_pop", "state"]


@dataclass(frozen=True)
class SkewReport:
    max_abs_diff: dict[str, float]
    decision_mismatches: int
    n_tx: int
    passed: bool
    n_cards: int = 0


def replay_card(g: pd.DataFrame, encoder) -> np.ndarray:
    """Serving-path features of one card's rows (``g`` sorted by (ts, row)), in ``g`` order."""
    # features need neither tau nor the network: a worker never loads the model
    serving = Serving(SimpleNamespace(encoder=encoder, tau=0.0, predict_proba=None))
    state = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), g.iloc[:0], at=g["ts"].iloc[0])
    out = np.empty((len(g), len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, r in enumerate(g.itertuples(index=False)):
        tx = Tx(ts=pd.Timestamp(r.ts), amt=float(r.amt), merchant=str(r.merchant), category=str(r.category),
                merch_lat=float(r.merch_lat), merch_long=float(r.merch_long))
        out[i] = serving.features(state, tx)
        state.push(tx)
    return out


def _replay_job(args) -> np.ndarray:
    return replay_card(*args)


def skew_check(df: pd.DataFrame, model, n_cards: int, seed: int, atol: float = 1e-4,
               n_jobs: int | None = None) -> SkewReport:
    """Compare both feature paths on ``n_cards`` cards drawn with ``default_rng(seed)``.

    Per-card replays run in a process pool (``n_jobs``, default ``os.cpu_count()``); a replay is
    deterministic, so the result is identical for any ``n_jobs``.
    """
    cards = np.sort(df["cc_num"].unique())
    k = min(n_cards, len(cards))
    chosen = np.random.default_rng(seed).choice(cards, size=k, replace=False)
    sub = df[df["cc_num"].isin(chosen)].sort_values(["ts", "row"], kind="stable")
    sub = sub[_REPLAY_COLS].reset_index(drop=True)
    batch = features(sub, model.encoder)  # aggregates are per card: a card subset gives the same values

    groups = [(g, model.encoder) for _, g in sub.groupby("cc_num", sort=True)]
    jobs = (os.cpu_count() or 1) if n_jobs is None else n_jobs
    if jobs <= 1 or len(groups) <= 1:
        parts = [_replay_job(a) for a in groups]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(groups)), mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_replay_job, groups))
    served = np.empty_like(batch)
    for (g, _), part in zip(groups, parts):
        served[g.index.to_numpy()] = part

    diff = np.abs(served.astype(np.float64) - batch.astype(np.float64))
    max_diff = {c: float(diff[:, i].max()) if len(diff) else 0.0 for i, c in enumerate(FEATURE_COLUMNS)}
    tol = np.zeros_like(diff)  # exact everywhere except the means
    tol[:, _MEAN_IDX] = atol + _RTOL * np.abs(batch[:, _MEAN_IDX].astype(np.float64))
    features_ok = bool((diff <= tol).all())

    p_batch = np.asarray(model.predict_proba(batch))
    p_served = np.asarray(model.predict_proba(served))
    mismatches = int(((p_batch >= model.tau) != (p_served >= model.tau)).sum())
    return SkewReport(max_abs_diff=max_diff, decision_mismatches=mismatches, n_tx=int(len(sub)),
                      passed=features_ok and mismatches == 0, n_cards=int(k))


def summary_markdown(rep: SkewReport, atol: float) -> str:
    """Short Ukrainian Markdown section for ``$GITHUB_STEP_SUMMARY``."""
    worst = max(rep.max_abs_diff.items(), key=lambda kv: kv[1])
    lines = [
        "## Звірка навчального і продуктивного шляхів ознак",
        "",
        f"- Карток: {rep.n_cards}, операцій: {rep.n_tx}",
        f"- Найбільша розбіжність: `{worst[0]}` = {worst[1]:.3g} (допуск лише для середніх: {atol:g}, решта точно)",
        f"- Розбіжностей рішень: {rep.decision_mismatches}",
        f"- Результат: **{'PASS' if rep.passed else 'FAIL'}**",
        "",
        "| ознака | макс. розбіжність |",
        "|---|---|",
    ]
    lines += [f"| `{c}` | {rep.max_abs_diff[c]:.3g} |" for c in AGG_COLUMNS]
    return "\n".join(lines) + "\n"
