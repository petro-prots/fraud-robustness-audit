"""GRU model kind (Task 24, R35/R37): history interface (batch = serving), artifact format, trainer, selection on clean
val, strict reproducibility, hardening wiring, exact permutation Spearman."""
import os
os.environ.setdefault("FRAUDAUDIT_GRU_ALLOW_NONSTRICT", "1")  # in-process tests: TF may have run before

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_sparkov_like
from fraudaudit import harden as hmod
from fraudaudit.attacks.random_search import RandomSearch
from fraudaudit.audit import run_audit
from fraudaudit.features.columns import FEATURE_COLUMNS
from fraudaudit.gate import GateConfig
from fraudaudit.model import gru as gmod
from fraudaudit.model.artifacts import ArtifactIntegrityError, derive_with_tau, load_artifacts
from fraudaudit.model.gru import GRUFraudModel, pick_best, train_gru_model
from fraudaudit.model.history import ArrayHistory, StreamHistory, previous_index, score_rows, stream_history
from fraudaudit.model.pipeline import features, load_model
from fraudaudit.serving.core import Serving
from fraudaudit.serving.state import HISTORY_ROWS, CardState, Tx, profile_of
from fraudaudit.skew import skew_check
from fraudaudit.spearman import critical_rho, leave_one_out, spearman_exact
from fraudaudit.threat import ThreatModel

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
TINY = {"k": 5, "hidden": 8}
N = len(FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def df():
    return make_sparkov_like(n_cards=160, days=500, seed=3, start=pd.Timestamp("2019-04-01"))


@pytest.fixture(scope="module")
def trained(df):
    return train_gru_model(df, seed=0, params=TINY, epochs=1)


@pytest.fixture(scope="module")
def model_dir(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("gru")
    trained[0].save(d, meta=dict(trained[1], seed=0))
    return d


# ---------------------------------------------------------------- history of a stream


def test_previous_index_hand_made_stream():
    cards = np.array([7, 9, 7, 7, 9, 7, 7])  # stream order; card 7 at rows 0, 2, 3, 5, 6; card 9 at rows 1, 4
    prev = previous_index(cards, length=3)
    assert prev.shape == (7, 3) and prev.dtype == np.int32
    assert prev[0].tolist() == [-1, -1, -1]
    assert prev[1].tolist() == [-1, -1, -1]
    assert prev[2].tolist() == [0, -1, -1]
    assert prev[3].tolist() == [2, 0, -1]
    assert prev[4].tolist() == [1, -1, -1]
    assert prev[5].tolist() == [3, 2, 0]  # most recent first
    assert prev[6].tolist() == [5, 3, 2]  # only the last three of five


def test_previous_index_rejects_unsorted_card_and_accepts_interleaved_cards():
    t = np.array([5, 1, 6, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="time order"):
        previous_index(np.array([1, 2, 1, 1]), t)  # card 1: 5, 6, 4
    previous_index(np.array([1, 2, 1, 2]), np.array([5, 1, 6, 4], dtype=np.int64))  # the cards' own order is fine


def test_stream_history_last_pads_left_and_keeps_order():
    F = np.arange(5 * 2, dtype=np.float32).reshape(5, 2) + 1
    h = StreamHistory(F, previous_index(np.array([1, 1, 1, 2, 1]), length=3))
    rows, valid = h.last(3)  # the fifth operation: previous of card 1 are rows 2, 1, 0 -> oldest first 0, 1, 2
    assert valid[4].tolist() == [True, True, True]
    np.testing.assert_array_equal(rows[4], F[[0, 1, 2]])
    assert valid[1].tolist() == [False, False, True]  # one earlier operation, left padding
    np.testing.assert_array_equal(rows[1], np.vstack([np.zeros((2, 2)), F[[0]]]))
    assert valid[3].tolist() == [False] * 3 and not rows[3].any()
    rows2, valid2 = h.subset(np.array([4, 1])).last(2)
    np.testing.assert_array_equal(rows2[0], F[[1, 2]])
    assert valid2[1].tolist() == [False, True]
    with pytest.raises(ValueError):
        h.last(4)


def test_array_history_last_slices_the_newest_rows():
    rows = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    h = ArrayHistory(rows, np.ones((2, 4), dtype=bool))
    r, v = h.last(2)
    np.testing.assert_array_equal(r, rows[:, 2:])
    with pytest.raises(ValueError):
        h.last(5)


def test_stream_history_of_real_stream_rejects_unsorted(df):
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="time order"):
        stream_history(shuffled, np.zeros((len(df), N), dtype=np.float32))


# ---------------------------------------------------------------- card state: history rows = batch history rows


def _tx(r) -> Tx:
    return Tx(ts=pd.Timestamp(r.ts), amt=float(r.amt), merchant=str(r.merchant), category=str(r.category),
              merch_lat=float(r.merch_lat), merch_long=float(r.merch_long))


def test_serving_history_equals_batch_history_row_by_row(df, trained):
    """The oracle with card state (R32): at every operation the serving state hands the model the same previous
    feature rows as the batch path (float32 noise only on the mean columns, as in the skew check)."""
    model = trained[0]
    serving = Serving(model)
    r = model.history_rows
    cards = np.sort(df["cc_num"].unique())[:6]
    sub = df[df["cc_num"].isin(cards)].sort_values(["ts", "row"], kind="stable").reset_index(drop=True)
    F = features(sub, model.encoder)
    batch_rows, batch_valid = stream_history(sub, F).last(r)
    pos = {}
    for card, g in sub.groupby("cc_num", sort=True):
        state = CardState.from_history(profile_of(g, card), g.iloc[:0], at=g["ts"].iloc[0])
        for i, row in zip(g.index, g.itertuples(index=False)):
            tx = _tx(row)
            h = serving.history(state)
            np.testing.assert_array_equal(h.valid[0], batch_valid[i])
            np.testing.assert_allclose(h.rows[0], batch_rows[i], rtol=1e-5, atol=1e-3)
            state.push(tx)
            pos[i] = True
    assert len(pos) == len(sub)


def test_card_with_fewer_operations_than_k_is_left_padded(df, trained):
    model = trained[0]
    serving = Serving(model)
    g = df[df["cc_num"] == df["cc_num"].iloc[0]].sort_values(["ts", "row"], kind="stable")
    state = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), g.iloc[:0], at=g["ts"].iloc[0])
    rows = list(g.itertuples(index=False))
    h0 = serving.history(state)
    assert h0.valid.shape == (1, model.history_rows) and not h0.valid.any() and not h0.rows.any()
    p0 = serving.decide(state, _tx(rows[0])).p
    # the same card, the same single operation, scored with an explicitly all-padding history
    F0 = serving.features(state, _tx(rows[0]))[None, :]
    pad = ArrayHistory(np.zeros((1, model.history_rows, N), dtype=np.float32),
                       np.zeros((1, model.history_rows), dtype=bool))
    assert p0 == model.predict_proba(F0, history=pad)[0]
    for row in rows[:2]:
        state.push(_tx(row))
    h2 = serving.history(state)
    assert h2.valid[0].tolist() == [False, False, True, True]  # two earlier operations, k - 1 = 4 slots
    assert 0.0 <= serving.decide(state, _tx(rows[2])).p <= 1.0


def test_from_history_and_push_give_the_same_history_entries(df):
    g = df[df["cc_num"] == df["cc_num"].iloc[5]].sort_values(["ts", "row"], kind="stable")
    cut = len(g) // 2
    rows = list(g.itertuples(index=False))
    by_push = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), g.iloc[:0], at=g["ts"].iloc[0])
    for row in rows[:cut]:
        by_push.push(_tx(row))
    at = rows[cut].ts
    by_hist = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), g[g["ts"] < at], at=at)
    a, b = by_push._hx, by_hist._hx
    n = min(len(a), len(b), HISTORY_ROWS)
    assert n > 0 and by_hist._hx_known
    for x, y in zip(a[-n:], b[-n:]):
        assert x.tx == y.tx
        np.testing.assert_allclose(x.agg, y.agg, rtol=1e-9, atol=1e-9)
    assert len(by_push._hx) <= HISTORY_ROWS and len(by_hist._hx) <= HISTORY_ROWS


def test_history_holds_every_operation_of_the_card_declined_or_not_and_copies_are_independent(df):
    g = df[df["cc_num"] == df["cc_num"].iloc[0]].sort_values(["ts", "row"], kind="stable")
    rows = list(g.itertuples(index=False))
    s = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), g.iloc[:0], at=g["ts"].iloc[0])
    s.push(_tx(rows[0]), passed=False, auto_approved=False)  # a declined operation still enters the card state
    c = s.copy()
    c.push(_tx(rows[1]))
    assert len(s._hx) == 1 and len(c._hx) == 2
    assert s._hx[0].tx == _tx(rows[0]) and c._hx[0] is s._hx[0]


def test_state_without_tx_columns_cannot_feed_a_sequence_model(df, trained):
    g = df[df["cc_num"] == df["cc_num"].iloc[0]].sort_values(["ts", "row"], kind="stable")
    thin = g[["cc_num", "ts", "amt", "category"]]
    s = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), thin, at=g["ts"].iloc[-1] + pd.Timedelta(days=1))
    assert not s._hx_known
    with pytest.raises(ValueError, match="sequence model"):
        Serving(trained[0]).history(s)


def test_aggregate_memo_does_not_change_values(df):
    """``aggregates`` answers from a memo when ``decide`` and ``push`` ask for the same operation: same numbers."""
    g = df[df["cc_num"] == df["cc_num"].iloc[3]].sort_values(["ts", "row"], kind="stable")
    s = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), g.iloc[:0], at=g["ts"].iloc[0])
    for row in g.itertuples(index=False):
        tx = _tx(row)
        a = s.aggregates(tx.ts, tx.amt, tx.category)
        b = s.aggregates(tx.ts, tx.amt, tx.category)  # memo hit
        assert a is not b
        np.testing.assert_array_equal(a, b)
        b[:] = -1  # mutating the answer must not poison the memo
        np.testing.assert_array_equal(a, s.aggregates(tx.ts, tx.amt, tx.category))
        s.push(tx)
        fresh = CardState.from_history(profile_of(g, g["cc_num"].iloc[0]), g[g["ts"] <= tx.ts], at=tx.ts)
        del fresh  # (construction must not fail on a frame that ends at the state time)


# ---------------------------------------------------------------- artifact


def test_manifest_and_files(model_dir):
    m = json.loads((model_dir / "manifest.json").read_text())
    assert m["kind"] == "gru" and m["format"] == 1
    assert set(m["sha256"]) == {"gru.keras", "scaler.skops", "encoder.json", "model.json"}
    info = json.loads((model_dir / "model.json").read_text())
    assert info["kind"] == "gru" and info["gru_params"] == TINY and "tensorflow" in info["versions"]
    assert not any(p.suffix in {".pkl", ".pickle", ".joblib"} for p in model_dir.iterdir())


def test_roundtrip_same_predictions(trained, model_dir, df):
    model, report = trained
    a = load_artifacts(model_dir)
    assert a.kind == "gru" and a.mlp is None and a.iforest is None and a.scaler is not None
    loaded = load_model(model_dir)
    assert isinstance(loaded, GRUFraudModel) and loaded.kind == "gru" and loaded.history_rows == TINY["k"] - 1
    F = features(df, model.encoder)
    mask = np.zeros(len(df), dtype=bool)
    mask[::7] = True
    np.testing.assert_array_equal(score_rows(model, df, F, mask), score_rows(loaded, df, F, mask))
    assert loaded.tau == model.tau and loaded.params == TINY
    assert report["gru"]["params"] == TINY and report["gru"]["selection"] is None


def test_predict_proba_needs_a_history_and_checks_shapes(trained):
    model, _ = trained
    F = np.random.default_rng(1).normal(size=(5, N)).astype("float32")
    with pytest.raises(ValueError, match="history"):
        model.predict_proba(F)
    h = ArrayHistory(np.zeros((5, 4, N), dtype=np.float32), np.ones((5, 4), dtype=bool))
    with pytest.raises(ValueError):
        model.predict_proba(F[:, :10], history=h)
    with pytest.raises(ValueError):
        model.predict_proba(F[:3], history=h)
    p = model.predict_proba(F, history=h)
    assert p.dtype == np.float64 and p.shape == (5,) and ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("name", ["gru.keras", "scaler.skops", "encoder.json", "model.json"])
def test_tampered_file_rejected_before_deserializing(trained, tmp_path, monkeypatch, name):
    trained[0].save(tmp_path, meta={"seed": 0})
    p = tmp_path / name
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))

    def boom(*a, **k):
        raise AssertionError("deserializer called before integrity check")

    import keras
    monkeypatch.setattr(keras.models, "load_model", boom)
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


def test_manifest_kind_must_match_model_json(trained, tmp_path):
    trained[0].save(tmp_path, meta={"seed": 0})
    m = json.loads((tmp_path / "manifest.json").read_text())
    for bad in ({"kind": "mlp"}, {"kind": "xgb"}):
        (tmp_path / "manifest.json").write_text(json.dumps({**m, **bad}))
        with pytest.raises(ArtifactIntegrityError):
            load_artifacts(tmp_path)
    del m["kind"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


@pytest.mark.parametrize("bad", [{"k": 1, "hidden": 8}, {"k": "5", "hidden": 8}, {"k": 5}, {"k": 5, "hidden": True},
                                 {"k": 5, "hidden": 0}, None])
def test_params_validated_on_load(trained, tmp_path, bad):
    model, _ = trained
    model.save(tmp_path, meta={})
    info = json.loads((tmp_path / "model.json").read_text())
    info["gru_params"] = bad
    (tmp_path / "model.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["sha256"]["model.json"] = hashlib.sha256((tmp_path / "model.json").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactIntegrityError):
        load_artifacts(tmp_path)


def test_derive_with_tau_keeps_kind(model_dir, tmp_path):
    derive_with_tau(model_dir, tmp_path / "d", 0.123, {"target_fpr": 0.02, "note": "t"})
    a = load_artifacts(tmp_path / "d")
    assert a.kind == "gru" and a.tau == 0.123 and a.meta["target_fpr"] == 0.02
    assert (tmp_path / "d" / "gru.keras").read_bytes() == (model_dir / "gru.keras").read_bytes()
    assert load_model(tmp_path / "d").tau == 0.123


# ---------------------------------------------------------------- trainer


def test_pick_best_rule():
    rows = [{"k": 20, "hidden": 64, "recall": 0.80}, {"k": 5, "hidden": 32, "recall": 0.78},
            {"k": 10, "hidden": 32, "recall": 0.80}, {"k": 5, "hidden": 64, "recall": 0.80},
            {"k": 20, "hidden": 32, "recall": 0.80}]
    best = pick_best(rows)  # highest recall; a tie goes to fewest parameters (hidden), then the shorter window
    assert (best["k"], best["hidden"]) == (10, 32)
    assert pick_best(rows[:2])["k"] == 20
    assert gmod.n_parameters(5, 32) == gmod.n_parameters(20, 32) < gmod.n_parameters(5, 64)
    assert gmod.GRU_GRID == tuple((k, h) for k in (5, 10, 20) for h in (32, 64))


def test_n_parameters_matches_keras():
    for k, h in ((5, 8), (5, 32), (10, 64)):
        assert gmod.build_gru(k, h).count_params() == gmod.n_parameters(k, h)


def test_fixed_training_settings():
    assert gmod.N_THREADS == 1 and gmod.EPOCHS == 20 and gmod.BATCH_SIZE == 32
    net = gmod.build_gru(5, 8)
    assert type(net.optimizer).__name__ == "Adamax" and float(net.optimizer.learning_rate) == pytest.approx(0.01)
    assert [i.name for i in net.inputs] == ["seq", "mask"] and net.inputs[1].dtype == "bool"
    assert net.input_shape == [(None, 5, N), (None, 5)]


def test_selection_runs_grid_on_clean_val_only_and_records_it(df, monkeypatch):
    grid = ((3, 4), (4, 4))
    seen = []
    real = gmod.score_rows
    monkeypatch.setattr(gmod, "score_rows", lambda m, d, F, mask=None: (seen.append(len(mask)), real(m, d, F, mask))[1])
    model, report = train_gru_model(df, seed=0, grid=grid, epochs=1)
    sel = report["gru"]["selection"]
    assert [(s["k"], s["hidden"]) for s in sel] == list(grid)
    assert all({"recall", "fpr", "tau", "n_parameters"} <= set(s) for s in sel)
    assert report["gru"]["selected"] in [{"k": k, "hidden": h} for k, h in grid]
    assert report["gru"]["selection_data"] == "val (clean)"
    n_val = report["rows"]["val"]
    assert seen[:2] == [n_val, n_val] and set(seen[:3]) == {n_val}  # grid entries scored on val rows only
    assert model.history_rows == report["gru"]["selected"]["k"] - 1
    json.dumps(report, allow_nan=False)


def test_training_uses_balanced_fit_rows_with_their_true_history(df, monkeypatch):
    captured = {}

    def spy(Fs, prev, pos_train, y_train, k, hidden, seed, epochs, verbose=0):
        captured.update(Fs=Fs, prev=prev, pos=pos_train, y=y_train, k=k)
        return gmod.build_gru(k, hidden)  # untrained net: only the inputs are checked

    monkeypatch.setattr(gmod, "_fit_net", spy)
    train_gru_model(df, seed=0, params=TINY, epochs=1)
    from fraudaudit.data import period
    per = period(df).to_numpy()
    pos, y = captured["pos"], captured["y"]
    assert (per[pos] == "fit").all() and (df["is_fraud"].to_numpy()[pos] == y).all()
    assert (y == 0).sum() == (y == 1).sum()  # balanced: fraud resampled with replacement to the number of legit
    assert len(set(pos[y == 1].tolist())) < (y == 1).sum()  # fraud rows repeat (resampled)
    full = previous_index(df["cc_num"].to_numpy(), None, TINY["k"] - 1)
    np.testing.assert_array_equal(captured["prev"], full)  # history = the stream's, not a resampled sequence
    dup = pos[y == 1][0]
    ts = df["ts"].to_numpy()
    for j in captured["prev"][dup]:
        assert j < 0 or (df["cc_num"].iloc[j] == df["cc_num"].iloc[dup] and ts[j] <= ts[dup])


def test_report_and_val_quality_at_tau(trained, df):
    model, report = trained
    assert report["model_kind"] == "gru" and report["val"]["fpr"] <= 0.0105
    assert report["gru"]["tf"]["op_determinism"] is True
    assert report["rows"]["fit"] > 0 and report["train_balanced_rows"] > report["rows"]["fit"] // 2
    from fraudaudit.audit import clean_quality
    q = clean_quality(df, model)
    assert q["val"]["recall"] == pytest.approx(report["val"]["recall"]) and q["val"]["fpr"] == report["val"]["fpr"]


# ---------------------------------------------------------------- strict reproducibility (R37)

PROBE = r"""
import hashlib, json, sys
sys.path.insert(0, {tests!r})
import numpy as np, pandas as pd
from conftest import make_sparkov_like
from fraudaudit.model.gru import train_gru_model
from fraudaudit.model.history import score_rows
from fraudaudit.model.pipeline import features
from fraudaudit.data import period
df = make_sparkov_like(n_cards=60, days=500, seed=3, start=pd.Timestamp("2019-04-01"))
m, rep = train_gru_model(df, seed=42, params={{"k": 5, "hidden": 32}}, epochs=2)
F = features(df, m.encoder)
p = score_rows(m, df, F, (period(df) == "val").to_numpy())
w = hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes() for x in m.net.get_weights())).hexdigest()
print(json.dumps({{"p": hashlib.sha256(p.tobytes()).hexdigest(), "w": w, "tau": m.tau, "tf": rep["gru"]["tf"]}}))
"""


def _probe(hashseed: str = "0") -> dict:
    out = subprocess.run([sys.executable, "-c", PROBE.format(tests=str(TESTS))], capture_output=True, text=True,
                         check=True, cwd=ROOT, env={**__import__("os").environ, "PYTHONHASHSEED": hashseed})
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_training_twice_is_bit_for_bit_identical_k5_h32():
    """R37 addendum: the smallest grid variant (k 5, hidden 32) trained twice (two fresh processes) gives the same
    weights and the same validation scores, bit for bit; strict mode = deterministic ops and one thread each."""
    a, b = _probe("0"), _probe("12345")  # different hash seeds: the real chain does not pin PYTHONHASHSEED
    assert a["tf"]["strict"] is True and a["tf"]["intra_op_threads"] == 1 and a["tf"]["inter_op_threads"] == 1
    assert a == b


# ---------------------------------------------------------------- batch = serving, 1 process = several


def test_skew_serving_equals_batch_50_cards_and_n_jobs_identical(model_dir, df):
    m = load_model(model_dir)
    a = skew_check(df, m, n_cards=50, seed=7, n_jobs=1, model_dir=model_dir)
    b = skew_check(df, m, n_cards=6, seed=7, n_jobs=1, model_dir=model_dir)
    c = skew_check(df, m, n_cards=6, seed=7, n_jobs=2, model_dir=model_dir)
    assert a.n_cards == 50 and a.passed and a.decision_mismatches == 0 and a.max_abs_p_diff <= 1e-4
    assert b == c and b.passed  # 1 process vs several: identical report


TM_PATH = ROOT / "threat-models" / "realistic.yaml"


def test_audit_cell_identical_for_n_jobs_and_equal_to_clean_path(df, model_dir):
    tm = dataclasses.replace(ThreatModel.from_yaml(TM_PATH), recon_budget_ops=120)
    model = load_model(model_dir)
    res = [run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=j, model_dir=model_dir,
                     target_periods=("fit", "val", "test"), threat_path=TM_PATH, spares_period="test", one_sided=False)
           for j in (1, 2)]
    assert res[0].per_target == res[1].per_target
    np.testing.assert_array_equal(res[0].best.to_vector(), res[1].best.to_vector())
    assert res[0].meta["recon"] == res[1].meta["recon"]


# ---------------------------------------------------------------- hardening


def test_harden_round_with_gru_adversarial_rows_enter_the_sequences(df, model_dir, tmp_path):
    model = load_model(model_dir)
    tm = dataclasses.replace(ThreatModel.from_yaml(ROOT / "threat-models" / "realistic.yaml"),
                             recon_budget_ops=40, cards_per_eval=2)
    cfg = GateConfig(target_fpr=0.01, recall_clean_min=0.5, delta=None, attacks=["random", "es", "stepwise"],
                     attack_seeds=[1, 2, 3], confidence=0.95, one_sided=True, spares_period="fit")
    final, reports, info = hmod.harden(df, model, tm, cfg, rounds=1, seed=7, model_dir=model_dir, out_dir=tmp_path,
                                       n_jobs=1, budget_ops=40, gen_seeds=(1,), gen_max_declines=(1, 3),
                                       mini_audit_seeds=(1,), baseline_audit=False, epochs=1)
    assert isinstance(final, GRUFraudModel) and [r.round for r in reports] == [1]
    assert final.params == model.params  # k/hidden stay the clean-val choice, no re-selection on attacked data
    a = load_artifacts(tmp_path / "round-1")
    assert a.kind == "gru" and a.meta["harden"]["round"] == 1 and "test" not in a.meta
    assert a.meta["gru"]["selection"] is None
    df_r = info["df"]
    is_adv = (df_r["row"] > df["row"].max()).to_numpy()  # insert_rows gives the new rows the ids after the largest
    assert is_adv.any() and len(df_r) == len(df) + is_adv.sum()
    F = features(df_r, final.encoder)
    prev = stream_history(df_r, F).prev
    followers = np.flatnonzero((prev[:, 0] >= 0) & np.isin(prev[:, 0], np.flatnonzero(is_adv)))
    assert len(followers) > 0  # a later operation of the card has an adversarial row as its previous operation
    j = followers[0]
    rows, valid = stream_history(df_r, F).subset([j]).last(4)
    np.testing.assert_array_equal(rows[0][-1], F[prev[j, 0]])


# ---------------------------------------------------------------- CLI


def test_gru_params_cli_validation():
    from fraudaudit import cli
    ok = cli.build_parser().parse_args(["train", "--model-kind", "gru", "--gru-params", "5,32"])
    assert ok.gru_params == {"k": 5, "hidden": 32}
    for bad in ("5", "5,32,1", "1,32", "5,0", "a,b"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["train", "--model-kind", "gru", "--gru-params", bad])


def test_window_beyond_the_state_is_refused(trained):
    m = trained[0]
    with pytest.raises(ValueError, match="history"):
        GRUFraudModel(net=gmod.build_gru(HISTORY_ROWS + 2, 4), scaler=m.scaler, encoder=m.encoder, tau=0.5,
                      params={"k": HISTORY_ROWS + 2, "hidden": 4})


# ---------------------------------------------------------------- exact permutation Spearman (Task 24 §3)


def test_spearman_monotone_n8_gives_minus_one_and_p_two_over_8_factorial():
    r = spearman_exact(np.arange(8), -np.arange(8) * 3.0)
    assert r.rho == pytest.approx(-1.0) and r.n == 8 and r.n_permutations == 40320
    assert r.p == pytest.approx(2 / 40320) and r.significant
    r2 = spearman_exact([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 8, 16, 32, 64, 128, 256])
    assert r2.rho == pytest.approx(1.0) and r2.p == pytest.approx(2 / 40320)


def _brute_p(x, y):
    """Reference for the exact two-sided p: plain loops over all permutations of the ranks of ``y``."""
    import itertools
    from scipy.stats import rankdata, spearmanr
    rx, ry = rankdata(x), rankdata(y)
    obs = abs(spearmanr(rx, ry)[0])
    hits = total = 0
    for perm in itertools.permutations(ry):
        total += 1
        hits += abs(spearmanr(rx, list(perm))[0]) >= obs - 1e-12
    return hits / total


def test_spearman_hand_computed_rho_and_p():
    x, y = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]  # d = 1 everywhere: rho = 1 - 6 * 6 / (6 * 35) = 0.828571
    r = spearman_exact(x, y)
    assert r.rho == pytest.approx(1 - 6 * 6 / (6 * 35))
    assert r.p == pytest.approx(_brute_p(x, y))
    assert r.p >= 0.05 and not r.significant and abs(r.rho) < r.critical_rho  # critical rho at n = 6 is 0.886


def test_spearman_p_crosses_alpha_at_the_critical_value():
    near = spearman_exact([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5])  # one adjacent swap: rho = 1 - 6 * 2 / 210
    assert near.rho == pytest.approx(1 - 12 / 210) and near.rho > near.critical_rho and near.p < 0.05
    assert near.p == pytest.approx(_brute_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5]))


def test_spearman_ties_use_average_ranks_and_permute_tied_ranks():
    x = [1, 2, 2, 4, 5, 6]
    y = [6, 5, 4, 4, 2, 1]
    r = spearman_exact(x, y)
    from scipy.stats import spearmanr
    assert r.rho == pytest.approx(spearmanr(x, y)[0]) and -1.0 < r.rho < -0.9
    assert r.p == pytest.approx(_brute_p(x, y)) and r.p > 2 / 720  # |rho| < 1 with ties: more than the two extremes
    with pytest.raises(ValueError, match="constant"):
        spearman_exact([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])


@pytest.mark.parametrize("n,expected", [(8, 0.738), (7, 0.786), (6, 0.886)])
def test_critical_rho_fallback_thresholds_of_r37(n, expected):
    assert round(critical_rho(n), 3) == expected
    sig = spearman_exact(np.arange(n), np.arange(n)[::-1] * 1.0)  # n and the threshold are returned with the result
    assert sig.n == n and sig.critical_rho == critical_rho(n) and sig.significant


def test_spearman_n_not_8_and_input_checks():
    r7 = spearman_exact(np.arange(7), np.arange(7)[::-1] * 2.0)
    assert r7.n_permutations == 5040 and r7.p == pytest.approx(2 / 5040)
    r6 = spearman_exact(np.arange(6), np.arange(6)[::-1] * 2.0)
    assert r6.n_permutations == 720 and r6.p == pytest.approx(2 / 720)
    for bad in (([1, 2], [1, 2]), ([1, 2, 3], [1, 2]), ([1, 2, np.nan], [1, 2, 3]), (np.arange(10), np.arange(10))):
        with pytest.raises(ValueError):
            spearman_exact(*bad)


def test_leave_one_out_range():
    from scipy.stats import spearmanr
    x = np.arange(8.0)
    y = np.array([8, 7, 6, 5, 4, 3, 1, 2], dtype=float)
    loo = leave_one_out(x, y)
    assert len(loo) == 8
    for i, v in enumerate(loo):
        keep = np.arange(8) != i
        assert v == pytest.approx(spearmanr(x[keep], y[keep])[0])


def test_k_out_of_range_rejected_before_training():
    import argparse
    from fraudaudit.cli import _gru_params
    from fraudaudit.model.gru import train_gru_model
    with pytest.raises(argparse.ArgumentTypeError):
        _gru_params("30,32")
    with pytest.raises(ValueError):
        train_gru_model(None, 1, params={"k": 30, "hidden": 8})
    with pytest.raises(ValueError):
        train_gru_model(None, 1, grid=[(5, 8), (21, 8)])


def test_non_strict_training_refused_without_flag(monkeypatch):
    from fraudaudit.model import gru
    monkeypatch.delenv(gru.ALLOW_NONSTRICT_ENV, raising=False)
    monkeypatch.setattr(gru, "configure_tf", lambda: {"strict": False})
    with pytest.raises(RuntimeError, match="strict"):
        gru.train_gru_model(None, 1, params={"k": 5, "hidden": 8})


def test_batches_are_reshuffled_every_epoch(monkeypatch):
    """R38: the batch order is default_rng([seed, epoch]); Keras must really call on_epoch_end each epoch."""
    import numpy as np
    from fraudaudit.model import gru as g
    seen = []
    orig = g._Batches.on_epoch_end

    def spy(self):
        orig(self)
        seen.append(self._perm.copy())
    monkeypatch.setattr(g._Batches, "on_epoch_end", spy)
    n = 200
    Fs = np.random.default_rng(0).normal(size=(n, g.N_FEATURES)).astype("float32")
    prev = np.full((n, 4), -1, dtype=np.int64)
    pos = np.arange(n)
    y = (np.arange(n) % 2).astype("float32")
    monkeypatch.setenv(g.ALLOW_NONSTRICT_ENV, "1")
    g._fit_net(Fs, prev, pos, y, 5, 8, 42, 3)
    # Keras calls on_epoch_end after each of the 3 epochs and once more after the last one (a look-ahead that trains
    # nothing), so at least 3 calls; the order for epoch e must be default_rng([seed, e]) and differ between epochs
    assert len(seen) >= 3
    for e in (1, 2, 3):
        assert np.array_equal(seen[e - 1], np.random.default_rng([42, e]).permutation(n))
    assert not np.array_equal(seen[0], seen[1]) and not np.array_equal(seen[1], seen[2])


def test_audit_start_state_from_history_scores_like_the_batch_path(df, trained):
    """The state every audit episode starts from is ``CardState.from_history(profile, earlier rows, at)``; the skew
    check only replays from an empty state. Compare, at several cut points of several cards (more than 19 earlier
    operations and more than 30 days of history, timestamps differing at the cut), the probability of
    ``Serving.decide`` on that state with the batch path (``features`` + ``score_rows`` over the ordered card)."""
    from fraudaudit.skew import _REPLAY_COLS
    from fraudaudit.model.pipeline import features
    m = trained[0]
    serving = Serving(m)
    checked = 0
    for cc in sorted(df["cc_num"].unique())[:12]:
        g = df[df["cc_num"] == cc].sort_values(["ts", "row"], kind="stable")[_REPLAY_COLS].reset_index(drop=True)
        if len(g) < 40:
            continue
        F = features(g, m.encoder)
        p_batch = score_rows(m, g, F)
        profile = profile_of(g, cc)
        for cut in (len(g) // 3, len(g) // 2, len(g) - 2):
            while cut < len(g) - 1 and g["ts"].iloc[cut] == g["ts"].iloc[cut - 1]:
                cut += 1
            at = g["ts"].iloc[cut]
            hist = g.iloc[:cut]
            if cut <= HISTORY_ROWS or at - hist["ts"].iloc[0] <= pd.Timedelta(days=30):
                continue
            state = CardState.from_history(profile, hist, at=at)
            p = serving.decide(state, _tx(next(g.iloc[cut:cut + 1].itertuples(index=False)))).p
            assert abs(p - p_batch[cut]) <= 1e-4, (cc, cut, p, p_batch[cut])
            checked += 1
    assert checked >= 6
