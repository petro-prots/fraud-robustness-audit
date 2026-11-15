import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import make_sparkov_like
from fraudaudit import audit as audit_mod
from fraudaudit.attacks.random_search import RandomSearch
from fraudaudit.audit import AuditResult, clean_ver, episode_txs, run_audit, write_outputs
from fraudaudit.data import fraud_episodes
from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.metrics import uplift_ci, ver_ci
from fraudaudit.model.artifacts import load_artifacts
from fraudaudit.model.pipeline import FraudModel
from fraudaudit.serving.core import Serving
from fraudaudit.threat import ThreatModel, contexts_for

TM_PATH = Path(__file__).resolve().parents[1] / "threat-models" / "realistic.yaml"
AMT = FEATURE_COLUMNS.index("amt")
ALL = ("fit", "val", "test")  # sparkov_like has too few test episodes for a 50/50 card split


class LimitModel:
    """Duck-typed model: decline iff amt > limit."""

    def __init__(self, encoder, limit=60.0):
        self.encoder, self.tau, self.limit = encoder, 0.5, limit

    def predict_proba(self, F):
        return (np.asarray(F)[:, AMT] > self.limit).astype(np.float64)


@pytest.fixture(scope="module")
def df():
    return make_sparkov_like()


@pytest.fixture(scope="module")
def tm():
    return dataclasses.replace(ThreatModel.from_yaml(TM_PATH), recon_budget_ops=300)


@pytest.fixture(scope="module")
def model_dir(df, tmp_path_factory):
    from fraudaudit.model.pipeline import train_model
    model, _ = train_model(df, seed=0, epochs=1)
    d = tmp_path_factory.mktemp("model")
    model.save(d, meta={"seed": 0})
    return d


@pytest.fixture(scope="module")
def model(model_dir):
    return FraudModel.from_artifacts(load_artifacts(model_dir))


@pytest.fixture(scope="module")
def result(df, model, tm, model_dir):
    return run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir,
                     target_periods=ALL, threat_path=TM_PATH)


def test_ver_ci_bounds():
    v = np.array([0.0, 0.2, 0.5, 0.9, 1.0, 0.3, 0.3, 0.7])
    mean, lo, hi = ver_ci(v)
    assert mean == pytest.approx(v.mean())
    assert v.min() <= lo <= mean <= hi <= v.max()
    assert lo < hi
    assert ver_ci(v) == ver_ci(v)  # seeded
    assert ver_ci(v, seed=1) != ver_ci(v, seed=2)
    wide, narrow = ver_ci(v, confidence=0.99), ver_ci(v, confidence=0.5)
    assert wide[1] <= narrow[1] and narrow[2] <= wide[2]
    assert ver_ci(np.full(10, 0.4)) == pytest.approx((0.4, 0.4, 0.4))
    with pytest.raises(ValueError):
        ver_ci(np.array([]))


def test_clean_ver_replays_real_operations(df, tm):
    serving = Serving(LimitModel(BaseEncoder().fit(df)))
    ctxs = contexts_for(df, fraud_episodes(df, set(ALL)))
    got = clean_ver(serving, ctxs, df, max_declines=1, rho=tm.rho)
    assert got.shape == (len(ctxs),)
    by_row = df.set_index("row")
    for ctx, v in zip(ctxs, got):
        ops = by_row.loc[list(ctx.episode.rows)].sort_values(["ts"], kind="stable")
        passed = 0.0
        for a in ops["amt"]:  # stop at the first decline (max_declines=1)
            if a > 60.0:
                break
            passed += a
        assert v == pytest.approx(passed / ctx.episode.amount)
    # without a bank reaction every op under the limit passes
    free = clean_ver(serving, ctxs, df, max_declines=None, rho=tm.rho)
    assert (free >= got).all()
    txs = episode_txs(df, ctxs[0].episode)
    assert [t.ts for t in txs] == sorted(t.ts for t in txs)
    assert sum(t.amt for t in txs) == pytest.approx(ctxs[0].episode.amount)


def test_audit_smoke(result, tm):
    r = result
    assert isinstance(r, AuditResult)
    for lo_mean_hi in (r.ver_adv, r.ver_clean):
        mean, lo, hi = lo_mean_hi
        assert 0.0 <= lo <= mean <= hi <= 1.0
    assert r.n_targets == len(r.per_target) >= 2
    assert 0.0 <= r.asr <= r.asr_upper <= 1.0
    k = sum(t["success"] for t in r.per_target)
    assert r.asr == pytest.approx(k / r.n_targets)
    assert 0 < r.recon_ops <= tm.recon_budget_ops
    assert r.best is not None and 1 <= r.best.k <= tm.k_max
    assert r.feasible_share == 1.0
    assert np.mean([t["ver_adv"] for t in r.per_target]) == pytest.approx(r.ver_adv[0])
    assert np.mean([t["ver_clean"] for t in r.per_target]) == pytest.approx(r.ver_clean[0])
    adv = np.array([t["ver_adv"] for t in r.per_target])
    cl = np.array([t["ver_clean"] for t in r.per_target])
    assert r.uplift == uplift_ci(adv, cl, seed=7)
    assert r.uplift[0] == pytest.approx(r.ver_adv[0] - r.ver_clean[0])
    assert r.meta["n_ver_adv_gt_1"] == int((adv > 1).sum()) and r.meta["max_ver_adv"] == adv.max()
    assert r.meta["n_ver_clean_gt_1"] == int((cl > 1).sum()) and r.meta["max_ver_clean"] == cl.max()
    assert r.meta["replay"] == {"context": "history ts < episode.start (episode rows excluded)",
                                "in_window_legit_ops": "excluded in both runs"}
    assert sum(t["n_feasible_ops"] for t in r.per_target) == sum(t["n_ops"] for t in r.per_target)
    for p in ("val", "test"):
        assert {"recall", "fpr", "precision"} <= set(r.clean_quality[p])
    m = r.meta
    for key in ("seed", "git_sha", "git_dirty", "platform", "versions", "threat_model", "model",
                "recon_budget_ops", "n_jobs", "wall_seconds", "recon"):
        assert key in m, key
    assert m["seed"] == 7 and m["n_jobs"] == 1
    assert len(m["threat_model"]["sha256"]) == 64 and len(m["model"]["model_json_sha256"]) == 64


def test_clean_ver_equals_audit_clean_leg(df, model, tm, result):
    key = {(t["card"], t["start"]): t for t in result.per_target}
    eps = [e for e in fraud_episodes(df, set(ALL)) if (e.card, e.start.isoformat()) in key]
    assert len(eps) == result.n_targets
    got = clean_ver(Serving(model), contexts_for(df, eps), df, tm.max_declines, tm.rho)
    assert list(got) == [key[(e.card, e.start.isoformat())]["ver_clean"] for e in eps]


def test_targets_and_spares_disjoint_in_audit(df, model, tm, model_dir, monkeypatch):
    seen = {}
    real = audit_mod.ReconOracle

    def spy(serving, spares, *a, **kw):
        seen["spares"] = list(spares)
        return real(serving, spares, *a, **kw)

    monkeypatch.setattr(audit_mod, "ReconOracle", spy)
    r = run_audit(df, model, tm, RandomSearch(), seed=3, n_jobs=1, model_dir=model_dir, target_periods=ALL)
    spare_cards = {c.episode.card for c in seen["spares"]}
    target_cards = {t["card"] for t in r.per_target}
    assert spare_cards and target_cards
    assert not spare_cards & target_cards
    assert spare_cards == set(r.spare_cards)
    assert target_cards | spare_cards == {e.card for e in fraud_episodes(df, set(ALL))}


def test_audit_identical_for_n_jobs(df, model, tm, model_dir, result):
    r2 = run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=2, model_dir=model_dir,
                   target_periods=ALL, threat_path=TM_PATH)
    assert r2.per_target == result.per_target
    assert (r2.ver_adv, r2.ver_clean, r2.asr, r2.asr_upper, r2.recon_ops, r2.feasible_share) == \
           (result.ver_adv, result.ver_clean, result.asr, result.asr_upper, result.recon_ops,
            result.feasible_share)
    np.testing.assert_array_equal(r2.best.to_vector(), result.best.to_vector())
    assert r2.meta["recon"] == result.meta["recon"]


def test_write_outputs(result, tm, tmp_path):
    write_outputs(result, tmp_path)
    data = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert data["ver_adv"]["mean"] == pytest.approx(result.ver_adv[0])
    assert data["feasible_share"] == 1.0
    assert data["uplift"]["upper"] == pytest.approx(result.uplift[2])
    assert data["best"]["k"] == result.best.k and len(data["best"]["shares"]) == tm.k_max
    assert data["n_targets"] == result.n_targets and len(data["per_target"]) == result.n_targets
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    for s in ("VER_clean", "VER_adv", "ASR", "95 %", "Найкраща стратегія", result.meta["git_sha"],
              "Приріст", "легітимн", "до перенесення"):
        assert s in md, s


def test_cli_audit_parser():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["audit", "--data", "data", "--model", "models", "--threat",
                                   "threat-models/realistic.yaml", "--attack", "random", "--out", "out/",
                                   "--n-jobs", "3"])
    assert (a.command, a.attack, a.seed, a.n_jobs, str(a.out)) == ("audit", "random", 42, 3, "out")
    assert build_parser().parse_args(["audit", "--attack", "es"]).attack == "es"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["audit", "--attack", "ga"])


def test_audit_meta_has_es_params(result, tm):
    import json
    params = result.meta["threat_model"]["params"]
    assert params["es"] == {"mu": 5, "lam": 10, "sigma": 0.15, "p_merchant": 0.1, "p_k": 0.2}
    json.dumps(result.meta)  # serialisable
    assert result.meta["recon"]["attack_params"] == {"batch_size": None}  # RandomSearch
