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
    for s in ("VER_clean", "VER_adv", "ASR", "95 %", "Best strategy", result.meta["git_sha"],
              "uplift", "legitimate", "before wrapping"):
        assert s in md, s


def test_cli_audit_parser():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["audit", "--data", "data", "--model", "models", "--threat",
                                   "threat-models/realistic.yaml", "--gate", "gate.yaml", "--out", "out/",
                                   "--n-jobs", "3"])
    assert (a.command, str(a.gate), a.seed, a.n_jobs, str(a.out)) == ("audit", "gate.yaml", 42, 3, "out")
    with pytest.raises(SystemExit):  # the attacks come from gate.yaml (every attack x seed)
        build_parser().parse_args(["audit", "--attack", "es"])


def test_audit_meta_has_es_params(result, tm):
    import json
    params = result.meta["threat_model"]["params"]
    assert params["es"] == {"mu": 5, "lam": 10, "sigma": 0.15, "p_merchant": 0.1, "p_k": 0.2}
    json.dumps(result.meta)  # serialisable
    assert result.meta["recon"]["attack_params"] == {"batch_size": None}  # RandomSearch


# --- single-transaction baseline ("conventional audit") -------------------------------------------------------

import pandas as pd  # noqa: E402

from fraudaudit.audit import (BaselineResult, _wrap_offset, original_stream_state,  # noqa: E402
                              run_single_tx_baseline, single_tx_target)
from fraudaudit.data import Episode  # noqa: E402
from fraudaudit.serving.state import CardState, Tx, profile_of  # noqa: E402
from fraudaudit.threat import EpisodeContext, MerchantTable, SingleTxThreatModel  # noqa: E402

SINGLE_PATH = Path(__file__).resolve().parents[1] / "threat-models" / "single-transaction.yaml"
CNT_1D = FEATURE_COLUMNS.index("cnt_1d")


class Cnt1dModel:
    """Duck-typed model: decline iff the card had >= 3 operations in the last day (``cnt_1d >= 3``)."""

    def __init__(self, encoder):
        self.encoder, self.tau = encoder, 0.5

    def predict_proba(self, F):
        return (np.asarray(F)[:, CNT_1D] >= 3).astype(np.float64)


@pytest.fixture(scope="module")
def tms():
    return SingleTxThreatModel.from_yaml(SINGLE_PATH)


def test_single_transaction_yaml_values_and_strict(tms, tmp_path):
    import yaml
    assert (tms.amount_min, tms.amount_max, tms.max_shift_days, tms.single_tx_budget, tms.knowledge,
            tms.merchant_coord_radius, tms.amount_rule) == (1.0, 1376.04, 7, 50, "decision", 1.0, "first")
    raw = yaml.safe_load(SINGLE_PATH.read_text())
    for change in ({"extra": 1}, {"knowledge": "score"}, {"single_tx_budget": 0}, {"single_tx_budget": 2.5},
                   {"max_shift_days": 0}, {"amount_min": 0}, {"single_tx_budget": True},
                   {"amount_rule": "mean"}, {"amount_rule": None}):
        bad = dict(raw, **change)
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            SingleTxThreatModel.from_yaml(p)
    for key in ("single_tx_budget", "amount_rule"):
        missing = {k: v for k, v in raw.items() if k != key}
        p = tmp_path / "missing.yaml"
        p.write_text(yaml.safe_dump(missing))
        with pytest.raises(ValueError):
            SingleTxThreatModel.from_yaml(p)
    p.write_text(yaml.safe_dump(dict(raw, amount_rule="max")))
    assert SingleTxThreatModel.from_yaml(p).amount_rule == "max"


def _spread_episode(df, n=13, spacing_h=12.5):
    """A hand-made episode on a real card after all its history: n operations ``spacing_h`` hours apart,
    so any 24-hour window holds at most 2 of them (``cnt_1d <= 2`` in the original stream)."""
    card = int(df["cc_num"].iloc[0])
    g = df[df["cc_num"] == card]
    prof = profile_of(g, card)
    start = pd.Timestamp("2021-06-01 00:00:00")  # > 30 days after the last row: empty history
    merchants = MerchantTable.from_df(df)
    real = [Tx(ts=start + pd.Timedelta(hours=spacing_h * j), amt=100.0, merchant=merchants.names[0],
               category=merchants.category[0], merch_lat=prof.lat, merch_long=prof.long) for j in range(n)]
    ep = Episode(card=card, start=start, amount=100.0 * n, rows=())
    ctx = EpisodeContext(episode=ep, profile=prof, state=CardState.from_history(prof, g, at=start))
    return ctx, real, merchants


def test_single_tx_uses_original_state(df, tm, tms):
    serving = Serving(Cnt1dModel(BaseEncoder().fit(df)))
    ctx, real, merchants = _spread_episode(df)
    for seed in range(5):
        t = single_tx_target(serving, ctx, real, tms, tm, merchants, seed, 0)
        # every operation alone, against the original stream (history + the real preceding operations),
        # sees at most 2 operations in the last day: each one evades at its first query
        assert t["n_ops"] == t["n_evaded"] == len(real)
        assert t["queries"] == len(real)
        assert t["ver_clean"] == pytest.approx(1.0)  # the real spread episode passes untouched
        # the same chosen operations as one episode (aggregates recomputed): they bunch up, the bank declines
        assert t["replay_declines"] >= 1 and t["replay_blocked"]
        assert t["ver_replay"] < t["chosen_amount"] / ctx.episode.amount


def test_single_tx_candidates_respect_threat_model(df, tm, tms):
    serving = Serving(LimitModel(BaseEncoder().fit(df), limit=60.0))  # declines amt > 60
    ctx, real, merchants = _spread_episode(df, n=5)
    t = single_tx_target(serving, ctx, real, tms, tm, merchants, 3, 1)
    assert t["n_evaded"] == 5 and t["queries"] >= 5
    win_end = ctx.episode.start + pd.Timedelta(days=tm.window_days)
    for op in t["chosen"]:
        assert op["evaded"] and op["amt"] <= 60.0 and tms.amount_min <= op["amt"]
        ts = pd.Timestamp(op["ts"])
        assert ctx.episode.start <= ts < win_end
        assert abs(op["merch_lat"] - ctx.profile.lat) <= tms.merchant_coord_radius
        assert merchants.index(op["merchant"]) is not None
    strict = Serving(LimitModel(BaseEncoder().fit(df), limit=0.5))  # declines everything
    t0 = single_tx_target(strict, ctx, real, tms, tm, merchants, 3, 1)
    assert t0["n_evaded"] == 0 and t0["queries"] == 5 * tms.single_tx_budget
    assert [pd.Timestamp(o["ts"]) for o in t0["chosen"]] == [r.ts for r in real]  # originals kept
    assert t0["chosen_amount"] == 0.0 and t0["evaded_orig_amount"] == 0.0


@pytest.fixture(scope="module")
def baseline(df, model, tm, tms, model_dir):
    return run_single_tx_baseline(df, model, tms, tm, seed=7, n_jobs=1, model_dir=model_dir,
                                  target_periods=ALL, single_path=SINGLE_PATH)


def test_baseline_result(baseline, result):
    b = baseline
    assert isinstance(b, BaselineResult)
    # the same target episodes as the episode audit, with the same clean leg
    assert [(t["card"], t["start"]) for t in b.per_target] == [(t["card"], t["start"]) for t in result.per_target]
    assert [t["ver_clean"] for t in b.per_target] == [t["ver_clean"] for t in result.per_target]
    assert b.ver_clean == result.ver_clean
    n_ops = sum(t["n_ops"] for t in b.per_target)
    assert b.n_ops == n_ops and b.n_evaded == sum(t["n_evaded"] for t in b.per_target)
    assert b.evaded_tx_share == pytest.approx(b.n_evaded / n_ops)
    total = sum(t["amount"] for t in b.per_target)
    assert b.evaded_amount_share == pytest.approx(sum(t["evaded_orig_amount"] for t in b.per_target) / total)
    assert b.chosen_amount_ratio == pytest.approx(sum(t["chosen_amount"] for t in b.per_target) / total)
    assert 0.0 <= b.evaded_amount_share <= 1.0 + 1e-12 and b.amount_rule == "first"
    assert b.meta["n_wrapped_earlier_than_original"] <= b.meta["n_wrapped_candidates"] <= b.meta["queries"]
    v = np.array([t["ver_replay"] for t in b.per_target])
    c = np.array([t["ver_clean"] for t in b.per_target])
    assert b.replay_ver == ver_ci(v, 0.95, 2000, 7) and b.replay_uplift == uplift_ci(v, c, 0.95, 2000, 7)
    for key in ("seed", "n_jobs", "single_threat_model", "wall_seconds", "definitions"):
        assert key in b.meta, key
    assert len(b.meta["single_threat_model"]["sha256"]) == 64
    json.dumps(audit_mod.baseline_to_json(b))


def test_baseline_identical_for_n_jobs(df, model, tm, tms, model_dir, baseline):
    b2 = run_single_tx_baseline(df, model, tms, tm, seed=7, n_jobs=2, model_dir=model_dir,
                                target_periods=ALL, single_path=SINGLE_PATH)
    assert b2.per_target == baseline.per_target
    assert (b2.evaded_tx_share, b2.evaded_amount_share, b2.replay_ver, b2.replay_uplift) == \
           (baseline.evaded_tx_share, baseline.evaded_amount_share, baseline.replay_ver, baseline.replay_uplift)


def test_wrap_offset_edges():
    day, win = 86_400, 7 * 86_400
    assert _wrap_offset(0, win) == 0 and _wrap_offset(win - 1, win) == win - 1
    assert _wrap_offset(win, win) == win - day  # exactly at the window end: one day back
    for n in range(7):
        for r in (0, 1, 3600, day - 1):
            off = win + n * day + r
            got = _wrap_offset(off, win)
            assert win - day <= got < win and (off - got) % day == 0 and got == win - day + r


def test_original_stream_includes_later_real_ops(df):
    """R15: the state at a shifted candidate time holds every other real op of the episode before that time,
    also the ones that come after the attacked op in the original order."""
    ctx, _, merchants = _spread_episode(df, n=1)
    s = ctx.episode.start
    real = [dataclasses.replace(_spread_episode(df, n=1)[1][0], ts=s + pd.Timedelta(seconds=x))
            for x in (0, 10, 20, 30, 40)]  # op 0, then a burst of 4 later ops
    at = s + pd.Timedelta(hours=5)
    st = original_stream_state(ctx, real, 0, at)
    assert st.aggregates(at)[0] == 4  # cnt_1d: the 4 later ops, not op 0 itself
    st2 = original_stream_state(ctx, real, 2, at)
    assert st2.aggregates(at)[0] == 4  # ops 0, 1, 3, 4
    early = s + pd.Timedelta(seconds=25)
    assert original_stream_state(ctx, real, 0, early).aggregates(early)[0] == 2  # only ops 1, 2 are before


def test_single_tx_shifted_past_burst_sees_it(df, tm, tms):
    serving = Serving(Cnt1dModel(BaseEncoder().fit(df)))
    ctx, one, merchants = _spread_episode(df, n=1)
    s = ctx.episode.start
    real = [dataclasses.replace(one[0], ts=s + pd.Timedelta(seconds=x)) for x in (0, 10, 20, 30, 40)]
    one_day = dataclasses.replace(tms, max_shift_days=1)  # every candidate of op 0 lands in [start, start + 1 d)
    t = single_tx_target(serving, ctx, real, one_day, tm, merchants, 0, 0)
    assert t["chosen"][0]["evaded"] is False  # shifted past the burst, op 0 sees its 4 later ops and is declined
    assert t["queries"] >= one_day.single_tx_budget


def test_amount_rule_max(df, tm, tms):
    serving = Serving(LimitModel(BaseEncoder().fit(df), limit=600.0))
    ctx, real, merchants = _spread_episode(df, n=5)
    first = single_tx_target(serving, ctx, real, tms, tm, merchants, 3, 1, "first")
    mx = single_tx_target(serving, ctx, real, tms, tm, merchants, 3, 1, "max")
    assert mx["queries"] == 5 * tms.single_tx_budget and first["queries"] < mx["queries"]
    assert first["n_evaded"] == mx["n_evaded"] == 5
    for a, b in zip(first["chosen"], mx["chosen"]):
        assert a["amt"] <= b["amt"] <= 600.0
    assert mx["chosen_amount"] >= first["chosen_amount"]
    with pytest.raises(ValueError):
        single_tx_target(serving, ctx, real, tms, tm, merchants, 3, 1, "median")


# ---- tau_after_decline wiring ------------------------------------------------------------------

def _spy_run_episode(monkeypatch, module):
    seen = []
    orig = module.run_episode

    def spy(*a, **kw):
        seen.append(kw.get("tau_after_decline", "missing"))
        return orig(*a, **kw)

    monkeypatch.setattr(module, "run_episode", spy)
    return seen


def test_tau_after_decline_reaches_every_run_episode_call(df, model, tm, model_dir, monkeypatch):
    from fraudaudit import oracle as oracle_mod
    tm2 = dataclasses.replace(tm, tau_after_decline=1e-9)
    seen_a, seen_o = _spy_run_episode(monkeypatch, audit_mod), _spy_run_episode(monkeypatch, oracle_mod)
    r = run_audit(df, model, tm2, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir,
                  target_periods=ALL, threat_path=TM_PATH)
    assert seen_a and seen_o and set(seen_a) == {1e-9} and set(seen_o) == {1e-9}
    assert r.meta["threat_model"]["params"]["tau_after_decline"] == 1e-9
    assert "tau_after_decline" in audit_mod.summary_markdown(audit_mod.to_json(r))


def test_tau_after_decline_reaches_baseline_replay(df, model, tm, tms, model_dir, monkeypatch):
    seen = _spy_run_episode(monkeypatch, audit_mod)
    run_single_tx_baseline(df, model, tms, dataclasses.replace(tm, tau_after_decline=1e-9), seed=7, n_jobs=1,
                           model_dir=model_dir, target_periods=ALL, single_path=SINGLE_PATH)
    assert seen and set(seen) == {1e-9}


def test_audit_with_escalation_identical_for_n_jobs(df, model, tm, model_dir):
    tm2 = dataclasses.replace(tm, tau_after_decline=1e-9)
    r1, r2 = (run_audit(df, model, tm2, RandomSearch(), seed=7, n_jobs=j, model_dir=model_dir,
                        target_periods=ALL, threat_path=TM_PATH) for j in (1, 2))
    assert r1.per_target == r2.per_target
    np.testing.assert_array_equal(r1.best.to_vector(), r2.best.to_vector())


def test_clean_ver_passes_tau_after_decline(df, tm):
    eps = fraud_episodes(df, set(ALL))[:5]
    ctxs = contexts_for(df, eps)
    s = Serving(LimitModel(BaseEncoder().fit(df), limit=60.0))
    base = clean_ver(s, ctxs, df, None, tm.rho)
    assert np.array_equal(base, clean_ver(s, ctxs, df, None, tm.rho, tau_after_decline=None))


# --- R14: targets = every episode of the target periods, spares from the fit period -------------------------


def _spy_spares(monkeypatch):
    seen = {}
    real = audit_mod.ReconOracle

    def spy(serving, spares, *a, **kw):
        seen["spares"] = list(spares)
        return real(serving, spares, *a, **kw)

    monkeypatch.setattr(audit_mod, "ReconOracle", spy)
    return seen


def test_fit_spares_targets_are_all_test_episodes(df, model, tm, model_dir, monkeypatch):
    seen = _spy_spares(monkeypatch)
    r = run_audit(df, model, tm, RandomSearch(), seed=3, n_jobs=1, model_dir=model_dir,
                  target_periods=("test",), spares_period="fit")
    want = fraud_episodes(df, {"test"})
    assert [(t["card"], t["start"]) for t in r.per_target] == [(e.card, e.start.isoformat()) for e in want]
    assert r.n_targets == len(want)
    fit = fraud_episodes(df, {"fit"})
    assert [(c.episode.card, c.episode.start) for c in seen["spares"]] == [(e.card, e.start) for e in fit]
    per = audit_mod.period(df)
    spare_rows = [row for c in seen["spares"] for row in c.episode.rows]
    assert set(per[df["row"].isin(spare_rows)]) == {"fit"}
    assert r.meta["spares_period"] == "fit" and r.meta["target_periods"] == ["test"]
    assert r.spare_cards == sorted({e.card for e in fit})
    assert r.meta["recon"]["n_spare_episodes"] == len(fit)
    # every spare context is the card's own history right before the fit-period spare episode
    import pandas as pd
    for c in seen["spares"]:
        assert c.profile.cc_num == c.episode.card
        start = pd.Timestamp(c.episode.start).value
        assert c.state._now == start and all(t < start for t in c.state._t)
    assert any(c.state._t for c in seen["spares"])  # history exists for fit-period spares


def test_spares_period_test_reproduces_the_old_split(df, model, tm, model_dir, result):
    from fraudaudit.data import split_cards
    r = run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir,
                  target_periods=ALL, threat_path=TM_PATH, spares_period="test")
    assert r.per_target == result.per_target and r.spare_cards == result.spare_cards
    assert r.uplift == result.uplift and r.meta["spares_period"] == "test"
    tgt, spare = split_cards(fraud_episodes(df, set(ALL)), 7)
    assert [t["card"] for t in r.per_target] == [e.card for e in tgt]
    assert r.spare_cards == sorted({e.card for e in spare})


@pytest.mark.parametrize("bad", ["val", "train", "", None])
def test_spares_period_strict(df, model, tm, model_dir, bad):
    with pytest.raises(ValueError):
        run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir,
                  target_periods=("test",), spares_period=bad)


def test_fit_spares_need_targets_outside_fit(df, model, tm, model_dir):
    with pytest.raises(ValueError, match="fit"):
        run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir,
                  target_periods=ALL, spares_period="fit")


@pytest.fixture(scope="module")
def fit_result(df, model, tm, model_dir):
    return run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir,
                     target_periods=("val", "test"), threat_path=TM_PATH, spares_period="fit", one_sided=True)


def test_fit_spares_identical_for_n_jobs(df, model, tm, model_dir, fit_result):
    r2 = run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=2, model_dir=model_dir,
                   target_periods=("val", "test"), threat_path=TM_PATH, spares_period="fit", one_sided=True)
    assert r2.per_target == fit_result.per_target
    assert (r2.ver_adv, r2.ver_clean, r2.uplift, r2.recon_ops) == \
           (fit_result.ver_adv, fit_result.ver_clean, fit_result.uplift, fit_result.recon_ops)
    np.testing.assert_array_equal(r2.best.to_vector(), fit_result.best.to_vector())
    assert r2.meta["recon"] == fit_result.meta["recon"] and r2.meta["uplift_se"] == fit_result.meta["uplift_se"]


def test_one_sided_uplift_in_audit(fit_result):
    from fraudaudit.metrics import uplift_se
    adv = np.array([t["ver_adv"] for t in fit_result.per_target])
    clean = np.array([t["ver_clean"] for t in fit_result.per_target])
    assert fit_result.uplift == uplift_ci(adv, clean, 0.95, audit_mod.N_BOOT, 7, one_sided=True)
    assert fit_result.uplift[2] <= uplift_ci(adv, clean, 0.95, audit_mod.N_BOOT, 7)[2]
    assert fit_result.meta["one_sided"] is True
    assert fit_result.meta["uplift_se"] == uplift_se(adv, clean, audit_mod.N_BOOT, 7)
    assert fit_result.ver_adv == ver_ci(adv, 0.95, audit_mod.N_BOOT, 7)  # VER intervals stay two-sided


def test_baseline_uses_the_audit_targets_with_fit_spares(df, model, tm, model_dir, fit_result):
    from fraudaudit.audit import run_single_tx_baseline
    from fraudaudit.threat import SingleTxThreatModel
    b = run_single_tx_baseline(df, model, SingleTxThreatModel.from_yaml(SINGLE_PATH), tm, seed=7, n_jobs=1,
                               model_dir=model_dir, target_periods=("val", "test"), spares_period="fit")
    assert [(t["card"], t["start"]) for t in b.per_target] == \
           [(t["card"], t["start"]) for t in fit_result.per_target]
    assert b.meta["spares_period"] == "fit"
