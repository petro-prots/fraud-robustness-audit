"""Training-serving skew check: the batch feature path and the serving path must agree.

After the ML Test Score principle "training and serving features compute the same values"
(Breck et al., 2017). Sampled cards are replayed operation by operation through ``CardState.push``
(serving path) and compared with ``features(df, encoder)`` (batch path) on the same rows.

Tolerance: integer-valued columns (``cnt_*``, ``secs_since_prev``) and every base feature must be
exactly equal; only the three float means (``mean_amt_*``) get ``atol`` (+ 1e-6 relative, the same
as ``np.allclose`` in ``tests/test_serving.py``), because float32 rounding of a float64 mean may
differ by one ulp between a summation over a window and a running list.

Decisions: every replayed operation goes through ``Serving.decide`` (single-row compiled call)
and is compared with the batch decision ``p_batch >= tau`` (a flip is a tie, counted in ``n_ties`` and not a
mismatch, only if both |p_batch - tau| and |p_served - p_batch| are within float32 noise ``1e-4``: tau is a validation
quantile, so some score equals it exactly, and a one-ulp difference of p would flip the decision). The check also fails
if ``max_abs_p_diff`` exceeds the same bound; ``max_abs_p_diff`` is the largest
|p_serving - p_batch|.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features.columns import AGG_COLUMNS, FEATURE_COLUMNS
from .model.pipeline import features
from .serving.core import Serving
from .serving.state import CardState, Tx, profile_of

_MEAN_IDX = np.array([FEATURE_COLUMNS.index(c) for c in AGG_COLUMNS if c.startswith("mean_amt_")])
_RTOL = 1e-6
_P_TIE = 1e-4  # score noise bound (same as the mean_amt atol): real model, 99k ops, batch vs single-row |dp| reaches 5e-6
_REPLAY_COLS = ["cc_num", "ts", "row", "amt", "merchant", "category", "merch_lat", "merch_long",
                "gender", "dob", "lat", "long", "city_pop", "state"]


@dataclass(frozen=True)
class SkewReport:
    max_abs_diff: dict[str, float]
    decision_mismatches: int
    n_tx: int
    passed: bool
    n_cards: int = 0
    max_abs_p_diff: float = 0.0
    n_ties: int = 0  # decision flips ignored as ties: |p_batch - tau| and |p_served - p_batch| both <= _P_TIE


_WORKER_MODEL = None  # per worker process: loaded once by _init_worker


def _load_model(model_dir):
    from .model.artifacts import load_artifacts
    from .model.pipeline import FraudModel
    return FraudModel.from_artifacts(load_artifacts(Path(model_dir)))


def _init_worker(model_dir) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = _load_model(model_dir)


def replay_card(g: pd.DataFrame, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serving path of one card (``g`` sorted by (ts, row)): features, p and declined of every
    operation, in ``g`` order. ``Serving.decide`` (single-row call) runs before each ``push``."""
    serving = Serving(model)
    state = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), g.iloc[:0], at=g["ts"].iloc[0])
    F = np.empty((len(g), len(FEATURE_COLUMNS)), dtype=np.float32)
    p = np.empty(len(g))
    declined = np.empty(len(g), dtype=bool)
    for i, r in enumerate(g.itertuples(index=False)):
        tx = Tx(ts=pd.Timestamp(r.ts), amt=float(r.amt), merchant=str(r.merchant), category=str(r.category),
                merch_lat=float(r.merch_lat), merch_long=float(r.merch_long))
        F[i] = serving.features(state, tx)
        d = serving.decide(state, tx)
        p[i], declined[i] = d.p, d.declined
        state.push(tx)
    return F, p, declined


def _replay_job(g: pd.DataFrame):
    return replay_card(g, _WORKER_MODEL)


def skew_check(df: pd.DataFrame, model, n_cards: int, seed: int, atol: float = 1e-4,
               n_jobs: int | None = None, model_dir: Path | str | None = None) -> SkewReport:
    """Compare both paths on ``n_cards`` cards drawn with ``default_rng(seed)``.

    Per card, every operation goes through ``Serving.decide`` (single-row compiled call) and is
    compared with the batch path (``features`` + one batched ``predict_proba``): feature diffs,
    decision mismatches (declined differs) and ``max_abs_p_diff``.

    ``n_jobs > 1`` needs ``model_dir`` (artifact directory): each worker loads its own model, as
    Keras models are not picklable. Without ``model_dir`` the given ``model`` object is used
    in-process (stub models in tests). Results are identical for any ``n_jobs``.
    """
    cards = np.sort(df["cc_num"].unique())
    k = min(n_cards, len(cards))
    chosen = np.random.default_rng(seed).choice(cards, size=k, replace=False)
    sub = df[df["cc_num"].isin(chosen)].sort_values(["ts", "row"], kind="stable")
    sub = sub[_REPLAY_COLS].reset_index(drop=True)
    batch = features(sub, model.encoder)  # aggregates are per card: a card subset gives the same values
    p_batch = np.asarray(model.predict_proba(batch), dtype=np.float64)

    groups = [g for _, g in sub.groupby("cc_num", sort=True)]
    jobs = (os.cpu_count() or 1) if n_jobs is None else n_jobs
    if jobs > 1 and len(groups) > 1:
        if model_dir is None:
            raise ValueError("n_jobs > 1 needs model_dir: every worker loads its own model")
        with ProcessPoolExecutor(max_workers=min(jobs, len(groups)), mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=(str(model_dir),)) as ex:
            parts = list(ex.map(_replay_job, groups))
    else:
        m = model if model_dir is None else _load_model(model_dir)
        parts = [replay_card(g, m) for g in groups]
    served = np.empty_like(batch)
    p_served = np.empty(len(sub))
    declined = np.empty(len(sub), dtype=bool)
    for g, (F, p, dec) in zip(groups, parts):
        ix = g.index.to_numpy()
        served[ix], p_served[ix], declined[ix] = F, p, dec

    diff = np.abs(served.astype(np.float64) - batch.astype(np.float64))
    max_diff = {c: float(diff[:, i].max()) if len(diff) else 0.0 for i, c in enumerate(FEATURE_COLUMNS)}
    tol = np.zeros_like(diff)  # exact everywhere except the means
    tol[:, _MEAN_IDX] = atol + _RTOL * np.abs(batch[:, _MEAN_IDX].astype(np.float64))
    features_ok = bool((diff <= tol).all())

    flip = declined != (p_batch >= model.tau)
    tie = flip & (np.abs(p_batch - model.tau) <= _P_TIE) & (np.abs(p_served - p_batch) <= _P_TIE)
    mismatches = int((flip & ~tie).sum())
    p_diff = float(np.abs(p_served - p_batch).max()) if len(sub) else 0.0
    return SkewReport(max_abs_diff=max_diff, decision_mismatches=mismatches, n_tx=int(len(sub)),
                      passed=features_ok and mismatches == 0 and p_diff <= _P_TIE, n_cards=int(k),
                      max_abs_p_diff=p_diff, n_ties=int(tie.sum()))


def summary_markdown(rep: SkewReport, atol: float) -> str:
    """Short English Markdown section for ``$GITHUB_STEP_SUMMARY``."""
    worst = max(rep.max_abs_diff.items(), key=lambda kv: kv[1])
    lines = [
        "## Training vs serving feature path comparison",
        "",
        f"- Cards: {rep.n_cards}, operations: {rep.n_tx}",
        f"- Largest discrepancy among all {len(rep.max_abs_diff)} features: `{worst[0]}` = {worst[1]:.3g} "
        f"(tolerance for means only: {atol:g}, the rest exact)",
        f"- Decision mismatches: {rep.decision_mismatches}; largest |Δp|: {rep.max_abs_p_diff:.3g}",
        f"- Ties ignored (decision flips within float32 noise of tau, |dp| <= {_P_TIE:g}): {rep.n_ties}",
        f"- Result: **{'PASS' if rep.passed else 'FAIL'}**",
        "",
        "| feature | max discrepancy |",
        "|---|---|",
    ]
    lines += [f"| `{c}` | {v:.3g} |" for c, v in rep.max_abs_diff.items()]
    return "\n".join(lines) + "\n"


def skew_to_json(rep: SkewReport, **extra) -> dict:
    """Every field of ``rep`` (plus ``extra``, e.g. seed, atol, git) as JSON; read back with ``skew_from_json``."""
    return {"passed": bool(rep.passed), "n_cards": int(rep.n_cards), "n_tx": int(rep.n_tx),
            "decision_mismatches": int(rep.decision_mismatches), "max_abs_p_diff": float(rep.max_abs_p_diff),
            "n_ties": int(rep.n_ties),
            "max_abs_diff": {k: float(v) for k, v in rep.max_abs_diff.items()}, **extra}


def skew_from_json(d: dict) -> SkewReport:
    """``SkewReport`` from ``skew_to_json`` output (``fraudaudit skew --out``); extra keys are ignored."""
    return SkewReport(max_abs_diff={k: float(v) for k, v in d["max_abs_diff"].items()},
                      decision_mismatches=int(d["decision_mismatches"]), n_tx=int(d["n_tx"]),
                      passed=bool(d["passed"]), n_cards=int(d["n_cards"]),
                      max_abs_p_diff=float(d["max_abs_p_diff"]), n_ties=int(d.get("n_ties", 0)))
