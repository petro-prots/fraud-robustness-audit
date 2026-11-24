"""Trace of the adversarial leg of an audit run (Task 27 chain 2, R48 / R50): what the attacker executed and what the
bank decided on every operation, for the mechanism of R19 (27b (ii), where the card is blocked under md2 vs md3) and the
scores of the passed operations (27d).

``policy-sweep --trace`` writes one gzip-compressed JSON per cell, ``<out>/<label>/trace/<safe key>.json.gz``
(``trace_path``; ``|`` → ``__`` and ``=`` → ``-`` in the cell key), with the cell key, its config hash, τ of the cell,
``tau_esc`` of the model and per run (``es/4``, …) the attack's best candidate (``audit.candidate_dict``) and per target,
in target order, ``target_trace``: ``card``, ``start``, ``amount``, ``ver_adv``, ``n_ops``, ``n_declines``,
``blocked``, ``block_op`` (1-based index of the executed operation on which the card was blocked, None if it was not)
and ``ops``, every executed adversarial operation in order: ``offset_s`` (seconds from the episode start), ``amt``,
``category``, ``p`` (the score of the bank's decision, ``serving.decide(...).p`` inside ``episode._Bank.submit`` – not a
re-scoring), ``declined`` (the model's decline at the active threshold: τ, after an escalation τ′), ``passed`` and
``escalated`` (the threshold was already escalated for this operation). Only the adversarial leg of the attack runs is
traced (not the clean leg, not friction). The trace changes no number and no config hash (it is not part of the cell's
configuration).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

TRACE_DIR = "trace"


def safe_key(key: str) -> str:
    """The cell key as a file name: ``fpr=0.01|md2|corridor=off`` → ``fpr-0.01__md2__corridor-off``."""
    return key.replace("|", "__").replace("=", "-")


def trace_path(sweep_dir, key: str) -> Path:
    """The trace file of cell ``key`` of the sweep in ``sweep_dir`` (the directory of ``policy-sweep.json``)."""
    return Path(sweep_dir) / TRACE_DIR / f"{safe_key(key)}.json.gz"


def write_trace(path, data: dict) -> None:
    """Gzip JSON, atomically (temporary file + rename); ``mtime=0`` and no file name in the header, so equal content gives equal bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    raw = (json.dumps(data, sort_keys=True) + "\n").encode("utf-8")
    with open(tmp, "wb") as fh, gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(raw)
    tmp.replace(path)


def load_trace(path) -> dict:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def trace_ok(path, config_hash: str) -> bool:
    """True iff the trace file exists, can be read and belongs to the cell configuration ``config_hash``."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        return load_trace(p).get("config_hash") == config_hash
    except (OSError, ValueError, EOFError):
        return False


def target_trace(adv, executed, start) -> dict:
    """``block_op`` and ``ops`` of one adversarial episode: ``adv`` is its ``EpisodeOutcome`` (with the per-operation
    ``scores``, ``model_declines``, ``escalated_ops``), ``executed`` the operations the bank saw, ``start`` the
    episode start. Plain types only (crosses process boundaries)."""
    n = int(adv.n_ops)
    if not (len(executed) == len(adv.scores) == len(adv.model_declines) == len(adv.escalated_ops)
            == len(adv.decisions) == n):
        raise RuntimeError(f"trace: {len(executed)} executed operations, {len(adv.scores)} scores, {n} decisions")
    t0 = pd.Timestamp(start)
    ops = [{"offset_s": float((pd.Timestamp(tx.ts) - t0).total_seconds()), "amt": float(tx.amt),
            "category": str(tx.category), "p": float(p), "declined": bool(dec), "passed": not bool(notp),
            "escalated": bool(esc)}
           for tx, p, dec, notp, esc in zip(executed, adv.scores, adv.model_declines, adv.decisions,
                                            adv.escalated_ops)]
    return {"block_op": n if adv.blocked else None, "ops": ops}


TARGET_FIELDS = ("card", "start", "amount", "ver_adv", "n_ops", "n_declines", "blocked")


def run_trace(result, merchants) -> dict:
    """The trace of one run (``audit.AuditResult`` audited with ``trace=True``): the best candidate and per target the
    fields of ``TARGET_FIELDS`` with ``block_op`` and ``ops``."""
    from .audit import candidate_dict
    per = []
    for t in result.per_target:
        if "trace" not in t:
            raise RuntimeError("the run was audited without trace=True")
        per.append({**{k: t[k] for k in TARGET_FIELDS}, **t["trace"]})
    return {"best": candidate_dict(result.best, merchants), "per_target": per}
