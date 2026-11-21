"""harden: round-based adversarial retraining on the H half of the fit episodes (fixture scale, 1 round)."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_sparkov_like
from fraudaudit import harden as hmod
from fraudaudit.data import FIT_SPLIT_SEED, VAL_START, fit_split, fraud_episodes, period
from fraudaudit.features.aggregates import batch_aggregates
from fraudaudit.gate import GateConfig
from fraudaudit.model.artifacts import load_artifacts
from fraudaudit.model.pipeline import FraudModel
from fraudaudit.threat import MerchantTable, ThreatModel, contexts_for

ROOT = Path(__file__).resolve().parents[1]
TM_PATH = ROOT / "threat-models" / "realistic.yaml"
UNITS = (("es", 1, 1), ("es", 2, 1), ("es", 1, 3), ("random", 1, 1), ("random", 2, 3))  # (attack, seed, md)


@pytest.fixture(scope="module")
def df():
    # a long fit period (2019-04-01 … 2020-03-21) so the host cards have slots away from their real episodes
    return make_sparkov_like(n_cards=160, days=500, seed=3, start=pd.Timestamp("2019-04-01"))


@pytest.fixture(scope="module")
def tm():
    return dataclasses.replace(ThreatModel.from_yaml(TM_PATH), recon_budget_ops=40, cards_per_eval=2)


@pytest.fixture(scope="module")
def gate_cfg():
    return GateConfig(target_fpr=0.01, recall_clean_min=0.5, delta=0.10, attacks=["random", "es", "stepwise"],
                      attack_seeds=[1, 2, 3], confidence=0.95, one_sided=True, spares_period="fit")


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
def hardened(df, model, model_dir, tm, gate_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("harden")
    final, reports, info = hmod.harden(df, model, tm, gate_cfg, rounds=1, seed=7, model_dir=model_dir, out_dir=out,
                                       n_jobs=1, budget_ops=40, epochs=1, gen_seeds=(1, 2), gen_max_declines=(1, 3),
                                       mini_audit_seeds=(1,), baseline_audit=True)
    return final, reports, info, out


def test_fixture_has_enough_episodes(df):
    h, s = fit_split(fraud_episodes(df, {"fit"}), FIT_SPLIT_SEED)
    recon, hosts = hmod.generation_split(h, 7)
    assert len(recon) >= 2 and len(hosts) >= 2 and len(s) >= 2
    assert fraud_episodes(df, {"val"})


def test_host_slots_are_isolated(df, tm):
    h, _ = fit_split(fraud_episodes(df, {"fit"}), FIT_SPLIT_SEED)
    _, hosts = hmod.generation_split(h, 7)
    slots = hmod.host_slots(df, hosts, tm)
    assert slots
    step = pd.Timedelta(days=tm.window_days + 30)
    fraud = df[df["is_fraud"] == 1]
    amounts = {}
    for e in hosts:
        amounts.setdefault(e.card, set()).add(e.amount)
    for e in slots:
        assert e.card in amounts and e.amount in amounts[e.card] and e.rows == ()  # A of a real host episode
        assert e.start + step <= VAL_START  # no val row ever sees an adversarial operation
        assert e.start - pd.Timedelta(days=30) >= df["ts"].min().normalize()
        f = fraud.loc[fraud["cc_num"] == e.card, "ts"]
        assert not ((f >= e.start - pd.Timedelta(days=30)) & (f < e.start + step)).any()
    for a, b in zip(slots, slots[1:]):
        assert a.card != b.card or b.start - a.start >= step


def test_generation_uses_only_h_cards(hardened, df):
    """R20: hardening generation ⊆ H; recon and host cards disjoint; S untouched."""
    _, _, info, _ = hardened
    h, s = fit_split(fraud_episodes(df, {"fit"}), FIT_SPLIT_SEED)
    hc, sc = {e.card for e in h}, {e.card for e in s}
    assert set(info["recon_cards"]) | set(info["host_cards"]) == hc
    assert not set(info["recon_cards"]) & set(info["host_cards"])
    adv = info["df"][info["df"]["trans_num"].str.startswith("adv-")]
    assert len(adv) and set(adv["cc_num"]) <= set(info["host_cards"]) and not set(adv["cc_num"]) & sc
    assert info["fit_split"]["seed"] == FIT_SPLIT_SEED


def test_harden_never_touches_test_period(hardened, df):
    _, reports, info, out = hardened
    aug = info["df"]
    adv = aug[aug["trans_num"].str.startswith("adv-")]
    assert (adv["ts"] < VAL_START).all() and set(period(adv)) == {"fit"} and set(adv["part"]) == {"train"}
    # every original row unchanged; test and val rows identical, in the same order
    orig = aug[~aug["trans_num"].str.startswith("adv-")].reset_index(drop=True)
    pd.testing.assert_frame_equal(orig, df.reset_index(drop=True))
    for p in ("test", "val"):
        a = aug[period(aug) == p].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, df[period(df) == p].reset_index(drop=True))
        # aggregates of val/test rows unchanged by the insertion
        np.testing.assert_array_equal(batch_aggregates(aug)[(period(aug) == p).to_numpy()].to_numpy(),
                                      batch_aggregates(df)[(period(df) == p).to_numpy()].to_numpy())
    # the mini-audit targets are val episodes, its clean quality has no test entry
    for r in reports:
        assert set(r.val_quality) >= {"recall", "fpr"}
    meta = json.loads((out / "round-1" / "model.json").read_text())["meta"]
    assert "test" not in meta and meta["harden"]["round"] == 1
    assert meta["rows"]["test"] == int((period(df) == "test").sum())  # same test rows, never trained on
    assert meta["fraud_rows"]["fit"] == int(((period(df) == "fit") & (df["is_fraud"] == 1)).sum()) + len(adv)


def test_adversarial_share(hardened, df):
    """Design §11 / R22: adversarial rows = share_among_fraud of all fit fraud rows (0.5 -> as many as real)."""
    _, reports, info, _ = hardened
    real = int(((period(df) == "fit") & (df["is_fraud"] == 1)).sum())
    assert hmod.quota_for(real, 0.5) == real and hmod.quota_for(300, 0.25) == 100
    r1 = reports[-1]
    assert r1.round == 1 and r1.quota == real == info["quota"]
    longest = max(len(e["ops"]) for e in info["selected"])
    assert r1.n_adv_rows <= real + longest / 2  # whole episodes, overshoot at most half an episode
    assert r1.shortfall == max(0, real - r1.n_adv_rows)
    assert r1.n_rows_by_generation_round == {"1": r1.n_adv_rows}
    adv = info["df"]["trans_num"].str.startswith("adv-")
    assert int(adv.sum()) == r1.n_adv_rows and (info["df"].loc[adv, "is_fraud"] == 1).all()


def _pool(n_slots, lens):
    return {g: {i: {"slot": i, "round": g, "ops": [{"i": (g, i, j)} for j in range(n)]} for i in range(n_slots)}
            for g, n in lens.items()}


def test_select_rows_equal_row_quotas_per_round():
    """R22: every generation round gets quota // r rows (remainder to the newest), on disjoint slots."""
    assert hmod.round_quotas(41, 2) == {1: 20, 2: 21} and hmod.round_quotas(10, 3) == {1: 3, 2: 3, 3: 4}
    pool = _pool(30, {1: 3, 2: 4})
    sel = hmod.select_rows(pool, 2, 41, seed=1)
    by = {g: sum(len(x["ops"]) for x in sel if x["round"] == g) for g in (1, 2)}
    assert by == {1: 21, 2: 20}  # 7 × 3 (overshoot 1 <= 1.5), 5 × 4 (stops: the 6th would overshoot by 3 > 2)
    assert len({x["slot"] for x in sel}) == len(sel)                  # one episode per slot, disjoint
    assert all(x["ops"] == pool[x["round"]][x["slot"]]["ops"] for x in sel)  # whole episodes
    assert hmod.select_rows(pool, 2, 41, seed=1) == sel               # deterministic
    # overshoot by more than half an episode stops the round
    sel = hmod.select_rows(_pool(30, {1: 4}), 1, 5, seed=1)
    assert [len(x["ops"]) for x in sel] == [4]
    # fewer generated than the quota: everything there is, per round
    few = hmod.select_rows(_pool(4, {1: 3, 2: 4}), 2, 100, seed=1)
    assert sum(len(x["ops"]) for x in few) == 3 * len([x for x in few if x["round"] == 1]) + \
        4 * len([x for x in few if x["round"] == 2]) and len(few) == 4
    assert hmod.select_rows(_pool(30, {1: 3}), 1, 0, seed=1) == []


def test_inserted_rows_change_later_aggregates(df):
    card = int(df.loc[df["is_fraud"] == 0, "cc_num"].iloc[0])
    g = df[df["cc_num"] == card].sort_values("ts")
    later = g.iloc[len(g) // 2]  # a legit row of the card
    t = pd.Timestamp(later["ts"]) - pd.Timedelta(days=2)
    slot = hmod.Episode(card=card, start=t, amount=100.0, rows=())
    ops = [{"ts": (t + pd.Timedelta(minutes=10 * j)).isoformat(), "amt": 50.0 + j, "merchant": df["merchant"].iloc[0],
            "category": df["category"].iloc[0], "merch_lat": float(g["lat"].iloc[0]),
            "merch_long": float(g["long"].iloc[0]), "declined": False} for j in range(3)]
    adv = hmod.adversarial_frame(df, [{"slot": 0, "round": 1, "ops": ops}], [slot])
    assert list(adv["trans_num"]) == ["adv-r1-slot0-op0", "adv-r1-slot0-op1", "adv-r1-slot0-op2"]
    assert (adv["gender"] == g["gender"].iloc[0]).all() and (adv["cc_num"] == card).all()
    aug = hmod.insert_rows(df, adv)
    assert len(aug) == len(df) + 3 and aug["row"].is_unique
    assert aug["ts"].is_monotonic_increasing
    before = batch_aggregates(df).loc[df.index[df["row"] == later["row"]][0]]
    after = batch_aggregates(aug).loc[aug.index[aug["row"] == later["row"]][0]]
    assert after["cnt_7d"] == before["cnt_7d"] + 3 and after["cnt_30d"] == before["cnt_30d"] + 3
    with pytest.raises(ValueError, match="fit period"):
        late = adv.assign(ts=pd.Timestamp(VAL_START))
        hmod.insert_rows(df, late)


def test_generation_identical_for_n_jobs(df, model, model_dir, tm):
    h, _ = fit_split(fraud_episodes(df, {"fit"}), FIT_SPLIT_SEED)
    recon, hosts = hmod.generation_split(h, 7)
    slots = hmod.host_slots(df, hosts, tm)
    rc, sc = contexts_for(df, recon), contexts_for(df, slots)
    m = MerchantTable.from_df(df)
    a = hmod.generate_round(model, rc, sc, tm, m, 7, 1, unit_list=list(UNITS), budget_ops=40, n_jobs=1)
    b = hmod.generate_round(model, rc, sc, tm, m, 7, 1, unit_list=list(UNITS), budget_ops=40, n_jobs=2,
                            model_dir=model_dir)
    assert a == b
    assert sum(u["n_slots"] for u in a) == len(slots)
    assert all(u["n_feasible_ops"] == u["n_ops"] for u in a)  # every generated operation is in the threat model
    assert [(u["attack"], u["seed"], u["max_declines"]) for u in a] == list(UNITS)
    assert {u["attack"] for u in a} == {"es", "random"}
    for u in a:
        for e in u["episodes"]:
            ts = [pd.Timestamp(o["ts"]) for o in e["ops"]]
            assert all(t < VAL_START for t in ts)
            nd = sum(o["declined"] for o in e["ops"])
            assert nd <= u["max_declines"] and (not e["blocked"] or e["ops"][-1]["declined"])


def test_round_reports_and_files(hardened):
    final, reports, info, out = hardened
    assert [r.round for r in reports] == [0, 1]
    r0, r1 = reports
    assert r0.n_adv_rows == 0 and r1.model_dir == str(out / "round-1")
    assert final.tau == r1.tau
    for r in reports:  # the mini-audit is the gate's 3-attack ensemble (stepwise only measures)
        assert set(r.val_uplift) == {"random/1", "es/1", "stepwise/1"}
        assert r.val_uplift_worst_upper == max(v["upper"] for v in r.val_uplift.values())
        assert r.val_uplift[r.val_uplift_worst_run]["upper"] == r.val_uplift_worst_upper
        assert set(r.val_uplift_by_attack) == {"random", "es", "stepwise"}
        assert r.val_uplift_by_attack["stepwise"]["worst_upper"] == r.val_uplift["stepwise/1"]["upper"]
    assert info["mini_audit_attacks"] == ["random", "es", "stepwise"] and info["gen_attacks"] == ["es", "random"]
    # generation used both attacks, never stepwise: every adversarial row comes from an es or random unit
    assert {u["attack"] for u in r1.units} == {"es", "random"} and len(r1.units) == 8
    assert (out / "round-1" / "adversarial-rows.csv").exists()
    meta = {"input_model": "m", "threat_file": "t", "gate_file": "g", "delta": 0.1, "rounds": 1, "seed": 7,
            "git_sha": "x", "git_dirty": False, "platform": "p", "wall_seconds": 1.0, "threat_sha256": "a",
            "gate_sha256": "b", "versions": {"numpy": np.__version__}}
    data = hmod.write_report(out, reports, info, meta)
    assert data["final_round"] == 1 and json.loads((out / "harden-report.json").read_text())["final_round"] == 1
    md = (out / "harden-report.md").read_text()
    assert md.startswith("# Adversarial hardening") and "| 1 |" in md and "held-out" in md
    assert "random search" in md and "stepwise" in md and "worst upper per attack" in md
    assert "Test false positives (diagnostic only)" in md and "| month |" in md and "| category |" in md
    assert "df" not in data["data"]


def test_multi_round_reload_replay_and_early_stop(df, model, model_dir, tm, gate_cfg, tmp_path):
    common = dict(seed=7, model_dir=model_dir, n_jobs=1, budget_ops=40, epochs=1, gen_seeds=(1,),
                  gen_max_declines=(1, 3), mini_audit_seeds=(1,), baseline_audit=False)
    # without δ no early stop: exactly the 5-round cap (the default), never more
    assert hmod.ROUNDS == 5
    final, reports, info = hmod.harden(df, model, tm, dataclasses.replace(gate_cfg, delta=None),
                                       out_dir=tmp_path / "a", **common)
    assert [r.round for r in reports] == [1, 2, 3, 4, 5]
    assert not (tmp_path / "a" / "round-6").exists()
    r5 = reports[-1]
    assert r5.model_dir == str(tmp_path / "a" / "round-5") and final.tau == r5.tau  # final = last round reached
    reloaded = FraudModel.from_artifacts(load_artifacts(tmp_path / "a" / "round-5"))
    assert reloaded.tau == final.tau
    assert set(r5.n_rows_by_generation_round) == {"1", "2", "3", "4", "5"}  # replay of every generation round
    q = hmod.round_quotas(r5.quota, 5)
    for g, n in r5.n_rows_by_generation_round.items():
        assert n <= q[int(g)] + tm.k_max / 2
    assert sum(r5.n_rows_by_generation_round.values()) == r5.n_adv_rows
    for r in reports:
        assert {u["attack"] for u in r.units} == {"es", "random"}
        assert set(r.val_uplift) == {"random/1", "es/1", "stepwise/1"}
    fp = info["fp_diagnostic"] if "fp_diagnostic" in info else info["test_fp_diagnostic"]
    for name in ("baseline", "final"):
        assert sum(v[name]["fp"] for v in fp["by_month"].values()) == fp["total"][name]["fp"]
        assert sum(v[name]["fp"] for v in fp["by_category"].values()) == fp["total"][name]["fp"]
    _, early, _ = hmod.harden(df, model, tm, dataclasses.replace(gate_cfg, delta=10.0),
                              out_dir=tmp_path / "b", **common)
    assert [r.round for r in early] == [1]                           # upper bound <= δ after round 1: stop
    assert not (tmp_path / "b" / "round-2").exists()


def test_cli_harden_parser_defaults():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["harden"])
    assert str(a.out) == "models-harden" and a.rounds == 5 and a.budget_ops == 20000 and a.epochs == 20
    assert str(a.model) == "models" and str(a.gate) == "gate.yaml" and a.seed == 42
    assert a.share_among_fraud == 0.5 and a.gen_attacks == ["es", "random"]


def test_generation_units_es_and_random():
    """64 units: attack {es, random} × seeds 1..16 × max_declines {1, 3}; stepwise never generates."""
    u = hmod.units()
    assert len(u) == 64 and len(set(u)) == 64
    assert {a for a, _, _ in u} == {"es", "random"} and hmod.GEN_ATTACKS == ("es", "random")
    assert {s for _, s, _ in u} == set(range(1, 17)) and {md for _, _, md in u} == {1, 3}
    assert sum(a == "random" for a, _, _ in u) == 32
    assert hmod.slot_assignment(130, 64, 1)[0] == [0, 64, 128]
