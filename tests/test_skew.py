import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_sparkov_like
from fraudaudit import cli
from fraudaudit.features.base import BaseEncoder
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.serving.core import Serving
from fraudaudit.serving.state import CardState, Tx, profile_of
from fraudaudit.skew import skew_check

CNT_1D = FEATURE_COLUMNS.index("cnt_1d")


class CountModel:
    """Duck-typed model: p grows with cnt_1d, so a wrong count can flip a decision."""

    def __init__(self, encoder, tau=0.5):
        self.encoder, self.tau = encoder, tau

    def predict_proba(self, F):
        return 1.0 / (1.0 + np.exp(-(np.asarray(F)[:, CNT_1D].astype(np.float64) - 1.0)))


@pytest.fixture(scope="module")
def df():
    return make_sparkov_like()


@pytest.fixture(scope="module")
def count_model(df):
    return CountModel(BaseEncoder().fit(df))


def None_model(model_dir):
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    return FraudModel.from_artifacts(load_artifacts(model_dir))


def test_skew_passes_on_consistent_paths(df, count_model):
    assert df.duplicated(["cc_num", "ts"]).sum() > 0  # same-second ops are in the sample
    rep = skew_check(df, count_model, n_cards=10, seed=42, n_jobs=1)
    assert rep.passed
    assert rep.decision_mismatches == 0
    assert rep.n_tx > 0
    for col in ("cnt_1d", "cnt_7d", "cnt_30d", "secs_since_prev"):
        assert rep.max_abs_diff[col] == 0.0
    assert set(rep.max_abs_diff) == set(FEATURE_COLUMNS)


def test_skew_identical_for_n_jobs(df, real_model_dir):
    a = skew_check(df, None_model(real_model_dir), n_cards=6, seed=7, n_jobs=1, model_dir=real_model_dir)
    b = skew_check(df, None_model(real_model_dir), n_cards=6, seed=7, n_jobs=2, model_dir=real_model_dir)
    assert a == b
    assert a.passed and a.decision_mismatches == 0 and a.max_abs_p_diff < 1e-5


def test_skew_detects_injected_fault(df, count_model, monkeypatch):
    """FAULT INJECTION, not a found defect: simulates a known defect class (the serving path counting
    operations of the same second) by monkeypatching ``CardState.aggregates``; the check must flag it."""
    real = CardState.aggregates

    def same_second_counted(self, ts):
        out = real(self, ts)
        t = pd.Timestamp(ts).value
        same = sum(1 for x in self._t if x == t)
        out[:3] += same  # cnt_1d, cnt_7d, cnt_30d
        return out

    monkeypatch.setattr(CardState, "aggregates", same_second_counted)
    rep = skew_check(df, count_model, n_cards=len(df["cc_num"].unique()), seed=1, n_jobs=1)
    assert rep.passed is False
    assert rep.max_abs_diff["cnt_1d"] > 0


# ---- decide CLI ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_model_dir(df, tmp_path_factory):
    from fraudaudit.model.pipeline import train_model
    model, _ = train_model(df, seed=0, epochs=1)
    d = tmp_path_factory.mktemp("model")
    model.save(d, meta={"seed": 0})
    return d


def _input_rows(df):
    cards = df["cc_num"].unique()[:3]
    sub = df[df["cc_num"].isin(cards) & (df["ts"] >= pd.Timestamp("2020-03-15"))]
    sub = sub.sort_values(["ts", "row"], kind="stable")
    return [{"cc_num": int(r.cc_num), "ts": str(r.ts), "amt": float(r.amt), "merchant": r.merchant,
             "category": r.category, "merch_lat": float(r.merch_lat), "merch_long": float(r.merch_long)}
            for r in sub.itertuples()]


def test_decide_cli_matches_serving(df, real_model_dir, tmp_path, monkeypatch, capsys):
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel

    rows = _input_rows(df)
    assert len(rows) > 10
    inp = tmp_path / "tx.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr("fraudaudit.data.load_raw", lambda d: df)
    monkeypatch.delenv("EXPOSE_SCORE", raising=False)
    monkeypatch.delenv("ENV", raising=False)

    assert cli.main(["decide", "--data", "unused", "--model", str(real_model_dir), "--input", str(inp)]) == 0
    got = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(set(g) == {"declined"} for g in got)  # p is not exposed by default

    serving = Serving(FraudModel.from_artifacts(load_artifacts(real_model_dir)))
    states, want = {}, []
    for r in rows:  # direct replay, history strictly before the card's first input ts
        c = r["cc_num"]
        if c not in states:
            states[c] = CardState.from_history(profile_of(df, c), df, at=pd.Timestamp(r["ts"]))
        tx = Tx(pd.Timestamp(r["ts"]), r["amt"], r["merchant"], r["category"], r["merch_lat"], r["merch_long"])
        d = serving.decide(states[c], tx)
        states[c].push(tx)
        want.append({"declined": d.declined})
    assert got == want

    monkeypatch.setenv("EXPOSE_SCORE", "1")
    assert cli.main(["decide", "--data", "unused", "--model", str(real_model_dir), "--input", str(inp)]) == 0
    with_p = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(0.0 <= g["p"] <= 1.0 and isinstance(g["declined"], bool) for g in with_p)
    assert [g["declined"] for g in with_p] == [w["declined"] for w in want]


def test_decide_cli_refuses_score_in_production(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "tx.jsonl"
    inp.write_text("")
    monkeypatch.setenv("EXPOSE_SCORE", "1")
    monkeypatch.setenv("ENV", "production")
    rc = cli.main(["decide", "--data", "unused", "--model", "unused", "--input", str(inp)])
    assert rc != 0
    assert "production" in capsys.readouterr().err


@pytest.mark.parametrize("env", ["Production", " prod ", "PRODUCTION", "prod"])
def test_decide_cli_refuses_score_in_production_any_spelling(env, tmp_path, monkeypatch, capsys):
    inp = tmp_path / "tx.jsonl"
    inp.write_text("")
    monkeypatch.setenv("EXPOSE_SCORE", "1")
    monkeypatch.setenv("ENV", env)
    assert cli.main(["decide", "--data", "unused", "--model", "unused", "--input", str(inp)]) != 0
    assert "production" in capsys.readouterr().err


def _decide_err(rows, df, real_model_dir, tmp_path, monkeypatch, capsys):
    inp = tmp_path / "bad.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr("fraudaudit.data.load_raw", lambda d: df)
    monkeypatch.delenv("EXPOSE_SCORE", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    rc = cli.main(["decide", "--data", "unused", "--model", str(real_model_dir), "--input", str(inp)])
    return rc, capsys.readouterr().err


def test_decide_cli_bad_input_is_clear_error(df, real_model_dir, tmp_path, monkeypatch, capsys):
    rows = _input_rows(df)
    missing = [{k: v for k, v in rows[0].items() if k != "amt"}]
    rc, err = _decide_err(missing, df, real_model_dir, tmp_path, monkeypatch, capsys)
    assert rc != 0 and "amt" in err and "Traceback" not in err
    unknown = [dict(rows[0], cc_num=1)]
    rc, err = _decide_err(unknown, df, real_model_dir, tmp_path, monkeypatch, capsys)
    assert rc != 0 and "not in the history" in err
    card = [r for r in rows if r["cc_num"] == rows[0]["cc_num"]]
    assert len(card) > 2
    rc, err = _decide_err([card[1], card[0]] if card[0]["ts"] != card[1]["ts"] else [card[2], card[0]],
                          df, real_model_dir, tmp_path, monkeypatch, capsys)
    assert rc != 0 and "time only moves forward" in err


def test_skew_cli_writes_step_summary(df, real_model_dir, tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr("fraudaudit.data.load_raw", lambda d: df)
    rc = cli.main(["skew", "--data", "unused", "--model", str(real_model_dir), "--cards", "5", "--n-jobs", "1"])
    text = summary.read_text(encoding="utf-8")
    assert "Звірка навчального і продуктивного шляхів ознак" in text
    assert rc == 0 and "PASS" in text and "max_abs_p_diff" not in text


def test_skew_cli_out_binds_model_hash(df, real_model_dir, tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr("fraudaudit.data.load_raw", lambda d: df)
    out = tmp_path / "skew.json"
    rc = cli.main(["skew", "--data", "unused", "--model", str(real_model_dir), "--cards", "5", "--n-jobs", "1",
                   "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert data["model_sha256"] == hashlib.sha256((real_model_dir / "model.json").read_bytes()).hexdigest()
    assert "git_sha" in data


# ---- tie tolerance of decision flips -----------------------------------------------------------

def _patch_decide(monkeypatch, flip_if, dp=0.0):
    """Serving.decide with injected drift: ``flip_if(p)`` flips the decision, ``dp`` shifts the served p."""
    from types import SimpleNamespace
    real = Serving.decide

    def decide(self, state, tx):
        d = real(self, state, tx)
        return SimpleNamespace(p=d.p + dp, declined=(not d.declined) if flip_if(d.p) else d.declined)

    monkeypatch.setattr(Serving, "decide", decide)


def _tie_check(df, count_model):
    return skew_check(df, count_model, n_cards=10, seed=42, n_jobs=1)  # CountModel: p == tau == 0.5 at cnt_1d == 1


def test_tie_flip_with_tiny_dp_is_not_a_mismatch(df, count_model, monkeypatch):
    _patch_decide(monkeypatch, lambda p: abs(p - 0.5) < 1e-9, dp=1e-9)
    rep = _tie_check(df, count_model)
    assert rep.n_ties > 0 and rep.decision_mismatches == 0 and rep.passed


def test_tie_flip_with_large_dp_fails(df, count_model, monkeypatch):
    _patch_decide(monkeypatch, lambda p: abs(p - 0.5) < 1e-9, dp=1e-3)
    rep = _tie_check(df, count_model)
    assert rep.n_ties == 0 and rep.decision_mismatches > 0 and not rep.passed


def test_flip_outside_tie_band_is_a_mismatch(df, count_model, monkeypatch):
    _patch_decide(monkeypatch, lambda p: abs(p - 0.5) > 0.1)
    rep = _tie_check(df, count_model)
    assert rep.decision_mismatches > 0 and not rep.passed


def test_p_drift_alone_fails_without_flips(df, count_model, monkeypatch):
    _patch_decide(monkeypatch, lambda p: False, dp=1e-3)
    rep = _tie_check(df, count_model)
    assert rep.decision_mismatches == 0 and rep.max_abs_p_diff > 1e-6 and not rep.passed


def test_n_ties_in_json_and_summary(df, count_model):
    from fraudaudit.skew import skew_from_json, skew_to_json, summary_markdown
    rep = skew_check(df, count_model, n_cards=3, seed=1, n_jobs=1)
    rep = type(rep)(**{**rep.__dict__, "n_ties": 2})
    assert skew_from_json(skew_to_json(rep)).n_ties == 2
    assert "Ties ignored" in summary_markdown(rep, 1e-4)
