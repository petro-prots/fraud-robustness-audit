"""Task 27 chain 2: the trace of the adversarial leg (policy-sweep --trace) and the summaries md-study (27b),
reaction-pairs (27c) and passed-scores (27d)."""
import dataclasses
import gzip
import json

import numpy as np
import pytest

from fraudaudit.episode import run_episode
from fraudaudit.policy_summary import _ver_matrix, rel_overestimation
from fraudaudit.policy_sweep import cell_key
from fraudaudit.trace import load_trace, safe_key, trace_path, write_trace
from test_episode import StubServing, _ctx, _txs
from test_policy_sweep import _setup, env  # noqa: F401  (module fixture: synthetic data and a trained model)


# ---- EpisodeOutcome: the score of every decision -----------------------------------------------------------------

def test_outcome_scores_are_the_decision_scores_with_escalation():
    ps = [0.1, 0.6, 0.3, 0.2, 0.45, 0.1]
    o = run_episode(StubServing(ps), _ctx(), _txs([10, 20, 30, 40, 50, 60]), 3, 0.8, tau_after_decline=0.25)
    # op1 passes, op2 declined (τ 0.5) → escalated; op3 0.3 ≥ 0.25 declined; op4 passes; op5 declined → 3rd decline
    assert o.n_ops == 5 and o.blocked
    assert o.scores == tuple(ps[:5])
    assert o.model_declines == (False, True, True, False, True)
    assert o.escalated_ops == (False, False, True, True, True)
    assert o.decisions == o.model_declines  # no corridor: «did not pass» = declined


def test_outcome_scores_equal_serving_decide_on_a_real_model(env):  # noqa: F811
    from fraudaudit.data import fraud_episodes
    from fraudaudit.serving.core import Serving
    from fraudaudit.threat import MerchantTable, Strategy, ThreatModel, context_for, realize
    df, model, _ = env
    tm, _, _ = _setup(env)
    merchants = MerchantTable.from_df(df)
    ctx = context_for(df, fraud_episodes(df, {"fit", "val", "test"})[0])
    rng = np.random.default_rng(5)
    txs = realize(dataclasses.replace(Strategy.random(tm, len(merchants), rng), k=6), ctx, tm, merchants, rng)

    class Recording(Serving):
        seen: list = []

        def decide(self, state, tx):
            d = super().decide(state, tx)
            self.seen.append(d.p)
            return d

    s = Recording(model)
    s.seen = []
    tau2 = float(model.tau) / 2
    o = run_episode(s, ctx, txs, 3, tm.rho, tau_after_decline=tau2)
    assert list(o.scores) == s.seen and len(s.seen) == o.n_ops
    esc = False
    for p, dec, e in zip(o.scores, o.model_declines, o.escalated_ops):
        assert e == esc and dec == (p >= (tau2 if esc else model.tau))
        esc = esc or dec


# ---- policy-sweep --trace on a synthetic cell -------------------------------------------------------------------

def _sweep(env, out, trace, n_jobs=1, reactions=("md2", "md3+esc")):  # noqa: F811
    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    return policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01,), reactions=reactions, corridors=("off",),
                        esc_fpr=0.2, seed=1, n_jobs=n_jobs, out_dir=out / "m", friction=False, trace=trace)


VOLATILE = ("wall_seconds", "ensemble_seconds", "provenance")


def _stable(cells):
    return [{k: v for k, v in c.items() if k not in VOLATILE} for c in cells]


@pytest.fixture(scope="module")
def traced(env, tmp_path_factory):  # noqa: F811
    plain_dir, traced_dir = tmp_path_factory.mktemp("plain"), tmp_path_factory.mktemp("traced")
    return _sweep(env, plain_dir, False), plain_dir, _sweep(env, traced_dir, True), traced_dir


def test_trace_changes_no_number_and_no_hash(traced):
    plain, plain_dir, tr, tr_dir = traced
    assert _stable(plain["cells"]) == _stable(tr["cells"])
    assert [c["config_hash"] for c in plain["cells"]] == [c["config_hash"] for c in tr["cells"]]
    assert not (plain_dir / "m" / "trace").exists()
    a = json.loads((plain_dir / "m" / "policy-sweep.json").read_text())
    b = json.loads((tr_dir / "m" / "policy-sweep.json").read_text())
    a.pop("meta"), b.pop("meta")
    assert _stable(a.pop("cells")) == _stable(b.pop("cells")) and a == b


def test_trace_content_is_consistent(env, traced):  # noqa: F811
    _, _, data, out = traced
    seen = {"blocked": 0, "escalated": 0, "passed": 0, "declined": 0}
    for c in data["cells"]:
        p = trace_path(out / "m", c["key"])
        assert p.name == safe_key(c["key"]) + ".json.gz" and "|" not in p.name and "=" not in p.name
        tr = load_trace(p)
        assert tr["key"] == c["key"] and tr["config_hash"] == c["config_hash"] and tr["tau"] == c["tau"]
        assert tr["tau_esc"] == data["tau_esc"] and sorted(tr["runs"]) == sorted(c["runs"])
        esc = c["tau_after_decline"]
        for rk, r in tr["runs"].items():
            assert [t["ver_adv"] for t in r["per_target"]] == c["runs"][rk]["ver_adv"]
            assert [[t["card"], t["start"]] for t in r["per_target"]] == c["targets"]
            if rk.startswith("es/"):
                assert r["best"]["ops"] and all("category" in o for o in r["best"]["ops"])
            for t, a in zip(r["per_target"], c["amounts"]):
                ops = t["ops"]
                assert len(ops) == t["n_ops"] and t["amount"] == a
                assert (t["block_op"] == t["n_ops"]) == t["blocked"]
                assert t["block_op"] is None or t["blocked"]
                assert sum(o["amt"] for o in ops if o["passed"]) == pytest.approx(t["ver_adv"] * a)
                assert sum(o["declined"] for o in ops) == t["n_declines"]
                assert all(o["passed"] != o["declined"] for o in ops)  # corridor off
                assert all(o["offset_s"] >= 0 for o in ops)
                seen_decline = False
                for o in ops:
                    assert o["escalated"] == (esc is not None and seen_decline)
                    thr = esc if o["escalated"] else c["tau"]
                    assert o["declined"] == (o["p"] >= thr)
                    seen_decline = seen_decline or o["declined"]
                if c["max_declines"] is not None and t["blocked"]:
                    assert t["n_declines"] == c["max_declines"]
                seen["blocked"] += t["blocked"]
                seen["escalated"] += sum(o["escalated"] for o in ops)
                seen["passed"] += sum(o["passed"] for o in ops)
                seen["declined"] += t["n_declines"]
    assert all(seen.values()), seen  # every branch of the consistency checks is exercised


def test_trace_with_workers_equals_in_process(env, traced, tmp_path):  # noqa: F811
    _, _, data, out = traced
    _sweep(env, tmp_path, True, n_jobs=2, reactions=("md3+esc",))
    k = cell_key(0.01, "md3+esc", "off")
    assert load_trace(trace_path(tmp_path / "m", k)) == load_trace(trace_path(out / "m", k))
    assert trace_path(tmp_path / "m", k).read_bytes() == trace_path(out / "m", k).read_bytes()


def test_trace_resume_reruns_only_cells_without_trace(env, traced, monkeypatch, tmp_path):  # noqa: F811
    import shutil

    from fraudaudit import policy_sweep as mod
    plain, plain_dir, data, out = traced
    calls = []
    real = mod.run_cell
    monkeypatch.setattr(mod, "run_cell", lambda *a, **k: (calls.append(k.get("trace")), real(*a, **k))[1])
    work = tmp_path / "w"
    shutil.copytree(out, work)
    again = _sweep(env, work, True)
    assert calls == []  # every cell has its trace: skipped
    gone = cell_key(0.01, "md2", "off")
    trace_path(work / "m", gone).unlink()
    again = _sweep(env, work, True)
    assert calls == [True]  # only the cell without a trace file, with the trace
    assert trace_path(work / "m", gone).is_file()
    assert _stable(again["cells"]) == _stable(data["cells"])
    _sweep(env, work, False)
    assert calls == [True]  # without --trace a finished cell is skipped whatever the trace
    work2 = tmp_path / "p"
    shutil.copytree(plain_dir, work2)
    _sweep(env, work2, True)  # a sweep made without the trace: every cell rerun once, numbers identical
    assert calls == [True, True, True]
    assert _stable(json.loads((work2 / "m" / "policy-sweep.json").read_text())["cells"]) == _stable(plain["cells"])


def test_trace_file_of_another_config_is_missing(tmp_path):
    from fraudaudit.trace import trace_ok
    p = tmp_path / "t.json.gz"
    assert not trace_ok(p, "h")
    write_trace(p, {"config_hash": "h"})
    assert trace_ok(p, "h") and not trace_ok(p, "other")
    p.write_bytes(b"broken")
    assert not trace_ok(p, "h")


def test_audit_per_target_unchanged_without_trace(env):  # noqa: F811
    from fraudaudit.audit import run_audit, to_json
    from fraudaudit.gate import make_attack
    df, model, d = env
    tm, _, _ = _setup(env)
    tm = dataclasses.replace(tm, max_declines=3)
    kw = dict(seed=1, n_jobs=1, model_dir=d, attack_seed=1, spares_period="fit", one_sided=True)
    a = run_audit(df, model, tm, make_attack("es", tm), **kw)
    b = run_audit(df, model, tm, make_attack("es", tm), trace=True, **kw)
    assert all("trace" not in t for t in a.per_target) and all("trace" in t for t in b.per_target)
    ja, jb = to_json(a), to_json(b)
    jb["per_target"] = [{k: v for k, v in t.items() if k != "trace"} for t in jb["per_target"]]
    for j in (ja, jb):
        j["meta"].pop("wall_seconds")
    assert ja == jb


# ---- md-study (27b) on hand-made inputs ---------------------------------------------------------------------------

MD2, INF = cell_key(0.01, "md2", "off"), cell_key(0.01, "mdinf", "off")
TARGETS = [[1, "2020-01-01T00:00:00"], [2, "2020-01-02T00:00:00"], [3, "2020-01-03T00:00:00"],
           [4, "2020-01-04T00:00:00"]]
AMTS = [100.0, 200.0, 50.0, 80.0]


def _ver(seed, md, i):
    """Deterministic per-target VERs; mdinf ≥ md2 on most targets."""
    base = [0.6, 0.3, 0.9, 0.45][i] + 0.01 * (seed % 5)
    return round(base if md is None else base - (0.2 if (i + seed) % 2 else 0.0), 6)


def _tcell(key, md, seeds, tau=0.4, sha_hash="h"):
    runs = {}
    for s in seeds:
        v = [_ver(s, md, i) for i in range(len(TARGETS))]
        runs[f"es/{s}"] = {"ver_adv": v, "ver_clean": [0.05] * len(v), "uplift_upper": 0.1 * s % 0.7,
                           "uplift_mean": 0.01, "ver_adv_mean": float(np.mean(v)), "ver_clean_mean": 0.05, "asr": 0.2}
    return {"key": key, "config_hash": f"{sha_hash}-{key}", "tau": tau, "max_declines": md,
            "tau_after_decline": None, "corridor": "off", "target_fpr": 0.01, "recon_budget_ops": 2000,
            "reaction": "mdinf" if md is None else f"md{md}", "targets": TARGETS, "amounts": AMTS, "runs": runs}


def _sweep_file(path, cells, sha="sha", role="confirmation", seeds=(), tau_esc=0.05):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"label": "x", "role": role, "model_json_sha256": sha, "tau_esc": tau_esc,
                                "esc_fpr": 0.05, "gate": {"attack_seeds": list(seeds)}, "cells": cells,
                                "meta": {"git_sha": "abc", "git_dirty": False}}))
    return path


def _ops_for(ver, amount, md, p_pass=0.1, p_decl=0.9):
    """Executed ops of one target: one passed op carrying VER·A, then declines until blocked (md) or two declines."""
    ops = [{"offset_s": 0.0, "amt": ver * amount, "category": "misc_net", "p": p_pass, "declined": False,
            "passed": True, "escalated": False}]
    n_dec = md if md is not None else 2
    for j in range(n_dec):
        ops.append({"offset_s": 3600.0 * (j + 1), "amt": 10.0, "category": "misc_net", "p": p_decl,
                    "declined": True, "passed": False, "escalated": False})
    return ops


def _write_trace_for(sweep_dir, cell, tau_esc=0.05, best=None):
    md = cell["max_declines"]
    runs = {}
    for rk, r in cell["runs"].items():
        per = []
        for (card, start), a, v in zip(cell["targets"], cell["amounts"], r["ver_adv"]):
            ops = _ops_for(v, a, md)
            blocked = md is not None
            per.append({"card": card, "start": start, "amount": a, "ver_adv": v, "n_ops": len(ops),
                        "n_declines": len(ops) - 1, "blocked": blocked, "block_op": len(ops) if blocked else None,
                        "ops": ops})
        runs[rk] = {"best": best or {"k": 3, "u": 0.9, "ops": [
            {"share_of_A": 0.3, "offset_hours": 5.0, "category": "shopping_net"},
            {"share_of_A": 0.2, "offset_hours": 1.0, "category": "misc_net"},
            {"share_of_A": 0.4, "offset_hours": 30.0, "category": "misc_net"}]}, "per_target": per}
    write_trace(trace_path(sweep_dir, cell["key"]), {"key": cell["key"], "config_hash": cell["config_hash"],
                                                     "tau": cell["tau"], "tau_esc": tau_esc, "runs": runs})


@pytest.fixture
def study(tmp_path):
    seeds = range(4, 14)
    md, inf = _tcell(MD2, 2, seeds), _tcell(INF, None, seeds)
    sp = _sweep_file(tmp_path / "new" / "xgb-base" / "policy-sweep.json", [md, inf], seeds=list(seeds))
    for c in (md, inf):
        _write_trace_for(sp.parent, c)
    g_md, g_inf = _tcell(MD2, 2, (1, 2, 3)), _tcell(INF, None, (1, 2, 3))
    gp = _sweep_file(tmp_path / "grid.json", [g_md, g_inf], role="grid", seeds=[1, 2, 3])
    ref = {"r19": {"xgb-base": {"adv_es": {"md2": rel_overestimation(_ver_matrix(g_inf, "es"),
                                                                      _ver_matrix(g_md, "es"), seed=42)}}}}
    rp = tmp_path / "summary8.json"
    rp.write_text(json.dumps(ref))
    return sp, gp, rp, md, inf


def test_md_study_known_answers(study, tmp_path):
    from fraudaudit.md_study import markdown, md_study
    sp, gp, rp, md, inf = study
    res = md_study([("xgb-base", sp, MD2, INF, str(gp))], rp)
    m = res["models"][0]
    o = m["overestimation"]
    for s in range(4, 14):
        want = rel_overestimation(np.array(inf["runs"][f"es/{s}"]["ver_adv"]),
                                  np.array(md["runs"][f"es/{s}"]["ver_adv"]), seed=42)
        assert o["per_seed"][str(s)] == want
    assert o["pooled"] == rel_overestimation(_ver_matrix(inf, "es"), _ver_matrix(md, "es"), seed=42)
    assert sum(o["counts"].values()) == 10
    assert m["grid"]["summary8_check"] == "equal" and m["grid"]["seeds"] == [1, 2, 3]
    t = m["trace"]
    assert t["md"]["pooled"]["share_blocked"] == 1.0 and t["inf"]["pooled"]["share_blocked"] == 0.0
    assert t["md"]["per_seed"]["4"]["block_op_counts"] == {"3": 4}  # one passed op + 2 declines
    assert t["inf"]["per_seed"]["4"]["block_op_counts"] == {} and t["inf"]["pooled"]["n_declines_counts"] == {"2": 40}
    assert t["md"]["per_seed"]["4"]["median_n_ops"] == 3 and t["md"]["pooled"]["mean_n_declines"] == 2
    assert t["md"]["per_seed"]["5"]["mean_ver_adv"] == pytest.approx(np.mean(md["runs"]["es/5"]["ver_adv"]))
    st = t["md"]["strategy"]["es/4"]
    assert st["amount_profile"] == [0.2, 0.3, 0.4] and st["offsets_h"] == [1.0, 5.0, 30.0]
    assert st["median_gap_h"] == 14.5 and st["min_gap_h"] == 4.0 and st["span_h"] == 29.0
    assert st["categories"] == {"misc_net": 2, "shopping_net": 1} and st["k"] == 3 and st["u"] == 0.9
    text = markdown(res)
    assert "summary8: equal" in text and "verdict" not in text.replace("no verdict", "")
    # another summary8 value → «differs»
    ref = json.loads(rp.read_text())
    ref["r19"]["xgb-base"]["adv_es"]["md2"]["rel"] += 1e-12
    rp.write_text(json.dumps(ref))
    assert md_study([("xgb-base", sp, MD2, INF, str(gp))], rp)["models"][0]["grid"]["summary8_check"] == "differs"


def test_cli_md_study_and_reaction_pairs_write_outputs(study, tmp_path):
    from fraudaudit.cli import main
    sp, gp, rp, _, _ = study
    out = tmp_path / "md"
    assert main(["md-study", "--cell", f"xgb-base,{sp},{MD2},{INF},{gp}", "--r19-reference", str(rp),
                 "--out", str(out)]) == 0
    res = json.loads((out / "md-study.json").read_text())
    assert res["models"][0]["grid"]["summary8_check"] == "equal" and "git_sha" in res["meta"]
    assert (out / "md-study.md").is_file()
    s2, g2, _, _ = _pairs_files(tmp_path / "rp")
    out = tmp_path / "rpo"
    assert main(["reaction-pairs", "--sweep", str(s2), "--a", MD3, "--b", ESC, "--reference", str(g2),
                 "--out", str(out)]) == 0
    assert json.loads((out / "reaction-pairs.json").read_text())["pairs"]["n_zero"] == 4
    assert main(["reaction-pairs", "--sweep", str(s2), "--a", MD3, "--b", "nope", "--reference", str(g2),
                 "--out", str(out)]) == 2


def test_md_study_grid_from_two_files(study, tmp_path):
    from fraudaudit.md_study import md_study
    sp, gp, rp, _, _ = study
    g = json.loads(gp.read_text())
    a = _sweep_file(tmp_path / "g1.json", [g["cells"][0]], role="grid")
    b = _sweep_file(tmp_path / "g2.json", [g["cells"][1]], role="grid")
    res = md_study([("xgb-base", sp, MD2, INF, f"{a}+{b}")], rp)
    assert res["models"][0]["grid"]["summary8_check"] == "equal" and len(res["models"][0]["inputs"]["grid"]) == 2
    with pytest.raises(ValueError, match="more than one file"):
        md_study([("xgb-base", sp, MD2, INF, f"{a}+{gp}")], rp)
    c = _sweep_file(tmp_path / "g3.json", [g["cells"][1]], role="grid", sha="other")
    with pytest.raises(ValueError, match="another model"):
        md_study([("xgb-base", sp, MD2, INF, f"{a}+{c}")], rp)


def test_md_study_refusals(study, tmp_path):
    from fraudaudit.md_study import md_study
    sp, gp, rp, md, inf = study
    other = _sweep_file(tmp_path / "other.json", json.loads(gp.read_text())["cells"], sha="other", role="grid")
    with pytest.raises(ValueError, match="model_json_sha256"):
        md_study([("xgb-base", sp, MD2, INF, str(other))], rp)
    d = json.loads(sp.read_text())
    del d["cells"][0]["runs"]["es/13"]
    miss = tmp_path / "miss" / "policy-sweep.json"
    miss.parent.mkdir()
    miss.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="expected exactly"):
        md_study([("xgb-base", miss, MD2, INF, str(gp))], rp)
    notrace = _sweep_file(tmp_path / "nt" / "policy-sweep.json", [md, inf])
    with pytest.raises(ValueError, match="no trace file"):
        md_study([("xgb-base", notrace, MD2, INF, str(gp))], rp)
    bad = json.loads(sp.read_text())
    bad["cells"][0]["config_hash"] = "changed"
    bp = tmp_path / "bad" / "policy-sweep.json"
    bp.parent.mkdir()
    bp.write_text(json.dumps(bad))
    for c in (md, inf):
        _write_trace_for(bp.parent, c)
    with pytest.raises(ValueError, match="trace belongs"):
        md_study([("xgb-base", bp, MD2, INF, str(gp))], rp)
    with pytest.raises(ValueError, match="mdinf"):
        md_study([("xgb-base", sp, MD2, MD2, str(gp))], rp)


# ---- reaction-pairs (27c) -------------------------------------------------------------------------------------------

MD3, ESC = "fpr=0.01|md3|corridor=on", "fpr=0.01|md3+esc|corridor=on"


def _ucell(key, ups: dict, seeds, tau=0.4, esc=None, cp=None):
    runs = {f"{a}/{s}": {"uplift_upper": ups.get((a, s), 0.05)} for a in ("es", "random", "stepwise") for s in seeds}
    return {"key": key, "U": max(r["uplift_upper"] for r in runs.values()), "runs": runs, "tau": tau,
            "max_declines": 3, "tau_after_decline": esc, "corridor_params": cp or {"q": 0.99},
            "target_fpr": 0.01, "recon_budget_ops": 2000}


def _pairs_files(tmp_path, seeds=range(4, 16), sha="sha", tau=0.4, b_ups=None):
    a_ups = {("stepwise", s): 0.10 + 0.01 * (s - 4) for s in range(4, 16)}
    b_ups = b_ups or {("es", s): 0.10 + 0.01 * (s - 4) + (0.02 if s % 3 == 0 else -0.01 if s % 3 == 1 else 0.0)
                      for s in range(4, 16)}
    sp = _sweep_file(tmp_path / "s.json", [_ucell(MD3, a_ups, seeds, tau), _ucell(ESC, b_ups, seeds, tau, esc=0.1)],
                     sha=sha)
    gp = _sweep_file(tmp_path / "g.json", [_ucell(MD3, {("random", 2): 0.2}, (1, 2, 3)),
                                           _ucell(ESC, {("es", 1): 0.15}, (1, 2, 3), esc=0.1)], role="grid")
    return sp, gp, a_ups, b_ups


def test_reaction_pairs_known_answers(tmp_path):
    from fraudaudit.reaction_pairs import markdown, reaction_pairs
    sp, gp, a_ups, b_ups = _pairs_files(tmp_path)
    res = reaction_pairs(sp, MD3, ESC, gp)
    d = res["pairs"]["d_s"]
    want = {str(s): b_ups[("es", s)] - a_ups[("stepwise", s)] for s in range(4, 16)}
    assert d.keys() == want.keys() and all(d[k] == pytest.approx(want[k]) for k in d)
    assert res["cells"]["a"]["U_s"]["4"] == {"U_s": 0.10, "attack": "stepwise"}
    assert res["cells"]["b"]["U_s"]["6"]["attack"] == "es"
    vals = np.array(list(want.values()))
    dist = res["pairs"]["distribution"]
    assert dist["median"] == pytest.approx(np.median(vals)) and dist["p25"] == pytest.approx(np.percentile(vals, 25))
    assert dist["min"] == pytest.approx(vals.min()) and dist["max"] == pytest.approx(vals.max())
    assert (res["pairs"]["n_negative"], res["pairs"]["n_zero"], res["pairs"]["n_positive"]) == (4, 4, 4)
    assert res["reference"]["seeds"] == [1, 2, 3] and res["reference"]["a"]["2"] == {"U_s": 0.2, "attack": "random"}
    assert res["reference"]["pairs"]["d_s"]["2"] == pytest.approx(0.05 - 0.2)
    assert res["cells"]["a"]["distribution"]["max"] == pytest.approx(0.21)
    text = markdown(res)
    assert "| 4 | 10.0 % (stepwise)" in text and "grid 2" in text


def test_reaction_pairs_reference_without_per_cell_corridor_or_budget(tmp_path):
    """Task 19 records keep the corridor only at the sweep level and no budget: the sweep's corridor is used, the
    budget is compared only when both records carry it."""
    from fraudaudit.reaction_pairs import reaction_pairs
    sp, gp, _, _ = _pairs_files(tmp_path)
    g = json.loads(gp.read_text())
    g["corridor_params"] = {"q": 0.99}
    for c in g["cells"]:
        del c["corridor_params"], c["recon_budget_ops"]
    gp.write_text(json.dumps(g))
    s = json.loads(sp.read_text())
    for c in s["cells"]:
        c["corridor"] = "on"
    sp.write_text(json.dumps(s))
    for c in g["cells"]:
        c["corridor"] = "on"
    gp.write_text(json.dumps(g))
    assert reaction_pairs(sp, MD3, ESC, gp)["reference"]["seeds"] == [1, 2, 3]
    g["corridor_params"] = {"q": 0.98}
    gp.write_text(json.dumps(g))
    with pytest.raises(ValueError, match="corridor_params"):
        reaction_pairs(sp, MD3, ESC, gp)
    g["corridor_params"] = {"q": 0.99}
    g["meta"]["recon_budget_ops"] = 20000
    gp.write_text(json.dumps(g))
    with pytest.raises(ValueError, match="recon_budget_ops"):
        reaction_pairs(sp, MD3, ESC, gp)


def test_reaction_pairs_refusals(tmp_path):
    from fraudaudit.reaction_pairs import reaction_pairs
    sp, gp, _, _ = _pairs_files(tmp_path / "a", sha="other")
    with pytest.raises(ValueError, match="model_json_sha256"):
        reaction_pairs(sp, MD3, ESC, gp)
    sp, gp, _, _ = _pairs_files(tmp_path / "b", seeds=range(4, 15))
    with pytest.raises(ValueError, match="seeds"):
        reaction_pairs(sp, MD3, ESC, gp)
    sp, gp, _, _ = _pairs_files(tmp_path / "c", tau=0.41)
    with pytest.raises(ValueError, match="tau"):
        reaction_pairs(sp, MD3, ESC, gp)
    sp, gp, _, _ = _pairs_files(tmp_path / "d")
    d = json.loads(sp.read_text())
    del d["cells"][1]["runs"]["random/9"]
    sp.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="attack"):
        reaction_pairs(sp, MD3, ESC, gp)


# ---- passed-scores (27d) --------------------------------------------------------------------------------------------

MD3_OFF = cell_key(0.01, "md3", "off")


def _passed_inputs(tmp_path, sha="sha", grid_change=None):
    seeds = (1, 2, 3)
    c = _tcell(MD3_OFF, 3, seeds, tau=0.4)
    sp = _sweep_file(tmp_path / "new" / "xgb-base" / "policy-sweep.json", [c], seeds=list(seeds), tau_esc=0.05)
    # per target one passed op: p = 0.01 · (target index + 1) · seed  → 0.01 … 0.12; τ′ = 0.05
    runs = {}
    for s in seeds:
        per = []
        for i, ((card, start), a, v) in enumerate(zip(c["targets"], c["amounts"], c["runs"][f"es/{s}"]["ver_adv"])):
            ops = _ops_for(v, a, 3, p_pass=0.01 * (i + 1) * s)
            per.append({"card": card, "start": start, "amount": a, "ver_adv": v, "n_ops": len(ops), "n_declines": 3,
                        "blocked": True, "block_op": len(ops), "ops": ops})
        runs[f"es/{s}"] = {"best": {"k": 1, "u": 1.0, "ops": []}, "per_target": per}
    write_trace(trace_path(sp.parent, c["key"]), {"key": c["key"], "config_hash": c["config_hash"], "tau": 0.4,
                                                  "tau_esc": 0.05, "runs": runs})
    g = json.loads(json.dumps(c))
    if grid_change:
        grid_change(g)
    gp = _sweep_file(tmp_path / "grid.json", [g], role="grid", sha=sha, seeds=list(seeds))
    return sp, gp


def test_passed_scores_known_answers(tmp_path):
    from fraudaudit.passed_scores import markdown, passed_scores
    sp, gp = _passed_inputs(tmp_path)
    res = passed_scores([("xgb-base", sp, MD3_OFF, gp)])
    m = res["models"][0]
    assert res["all_equal"] and m["cross_check"]["all_equal"]
    p_all = [0.01 * (i + 1) * s for s in (1, 2, 3) for i in range(4)]
    po = m["pooled"]
    assert po["n_passed"] == 12 and po["n_declined"] == 36 and po["n_targets"] == 12
    assert po["p"]["median"] == pytest.approx(np.median(p_all)) and po["p"]["p10"] == pytest.approx(
        np.percentile(p_all, 10))
    assert po["p_over_tau"]["max"] == pytest.approx(0.12 / 0.4)
    ge = [x >= 0.05 for x in p_all]
    assert po["n_passed_ge_tau_esc"] == sum(ge) and po["share_passed_ge_tau_esc"] == pytest.approx(np.mean(ge))
    assert po["share_targets_with_passed_ge_tau_esc"] == pytest.approx(np.mean(ge))  # one passed op per target
    s1 = m["per_seed"]["1"]
    assert s1["n_passed_ge_tau_esc"] == 0 and s1["p"]["max"] == pytest.approx(0.04)
    assert m["per_seed"]["3"]["share_passed_ge_tau_esc"] == pytest.approx(0.75)  # 0.03, 0.06, 0.09, 0.12 vs τ′ 0.05
    assert len(m["log10_p_over_tau"]) == 12
    assert "All runs equal to the grid: yes" in markdown(res)


def test_passed_scores_cross_check_difference_exits_non_zero(tmp_path, capsys):
    from fraudaudit.cli import main

    def change(g):
        g["runs"]["es/2"]["ver_adv"][1] += 1e-9
        g["runs"]["es/3"]["uplift_upper"] += 1e-12
    sp, gp = _passed_inputs(tmp_path, grid_change=change)
    out = tmp_path / "out"
    rc = main(["passed-scores", "--cell", f"xgb-base,{sp},{MD3_OFF},{gp}", "--out", str(out)])
    assert rc == 1 and "differs from the grid" in capsys.readouterr().err
    res = json.loads((out / "passed-scores.json").read_text())
    bad = [(r["run"], r["field"]) for r in res["models"][0]["cross_check"]["rows"] if not r["equal"]]
    assert bad == [("es/2", "ver_adv"), ("es/3", "uplift_upper")] and not res["all_equal"]
    assert (out / "passed-scores-uk.png").is_file() and (out / "passed-scores-en.png").is_file()
    assert "DIFFERS" in (out / "passed-scores.md").read_text()
    sp2, gp2 = _passed_inputs(tmp_path / "ok")
    assert main(["passed-scores", "--cell", f"xgb-base,{sp2},{MD3_OFF},{gp2}", "--out", str(tmp_path / "o2")]) == 0


def test_passed_scores_refusals(tmp_path):
    from fraudaudit.passed_scores import passed_scores
    sp, gp = _passed_inputs(tmp_path / "a", sha="other")
    with pytest.raises(ValueError, match="model_json_sha256"):
        passed_scores([("xgb-base", sp, MD3_OFF, gp)])
    sp, gp = _passed_inputs(tmp_path / "b")
    d = json.loads(sp.read_text())
    del d["cells"][0]["runs"]["es/3"]
    sp.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="expected exactly"):
        passed_scores([("xgb-base", sp, MD3_OFF, gp)])
    sp, gp = _passed_inputs(tmp_path / "c")
    tr = load_trace(trace_path(sp.parent, MD3_OFF))
    tr["runs"]["es/1"]["per_target"][0]["ops"][0]["p"] = 0.5  # a passed op above τ = 0.4
    write_trace(trace_path(sp.parent, MD3_OFF), tr)
    with pytest.raises(ValueError, match="inconsistent"):
        passed_scores([("xgb-base", sp, MD3_OFF, gp)])


def test_cli_parsers():
    from fraudaudit.cli import build_parser
    p = build_parser()
    a = p.parse_args(["md-study", "--cell", "a,s.json,k1,k2,g.json", "--r19-reference", "s8.json", "--out", "o"])
    assert a.command == "md-study" and a.cell == ["a,s.json,k1,k2,g.json"]
    a = p.parse_args(["reaction-pairs", "--sweep", "s", "--a", MD3, "--b", ESC, "--reference", "g", "--out", "o"])
    assert a.a == MD3 and a.b == ESC
    a = p.parse_args(["passed-scores", "--cell", "a,s,k,g", "--out", "o"])
    assert a.cell == ["a,s,k,g"]
    assert p.parse_args(["policy-sweep", "--model", "m", "--label", "l"]).trace is False
    assert p.parse_args(["policy-sweep", "--model", "m", "--label", "l", "--trace"]).trace is True


def test_cli_cell_spec_refused(capsys):
    from fraudaudit.cli import main
    assert main(["md-study", "--cell", "a,s.json,k1", "--out", "o"]) == 2
    assert "LABEL,SWEEP,MD_KEY,INF_KEY,GRID_SWEEP" in capsys.readouterr().err


def test_gate_configs_of_task27_chain2():
    from pathlib import Path

    from fraudaudit.gate import GateConfig
    root = Path(__file__).resolve().parents[1]
    base = GateConfig.from_yaml(root / "gate.yaml")
    for name, attacks, seeds in (("gate-27b-es-4-13.yaml", ["es"], list(range(4, 14))),
                                 ("gate-seeds-4-15.yaml", base.attacks, list(range(4, 16))),
                                 ("gate-27d-es-1-3.yaml", ["es"], [1, 2, 3])):
        g = GateConfig.from_yaml(root / name)
        assert g.attacks == attacks and g.attack_seeds == seeds, name
        assert dataclasses.replace(g, attacks=base.attacks, attack_seeds=base.attack_seeds) == base, name


def test_gzip_is_deterministic(tmp_path):
    write_trace(tmp_path / "a.json.gz", {"x": 1})
    write_trace(tmp_path / "b.json.gz", {"x": 1})
    assert (tmp_path / "a.json.gz").read_bytes() == (tmp_path / "b.json.gz").read_bytes()
    with gzip.open(tmp_path / "a.json.gz", "rt") as fh:
        assert json.load(fh) == {"x": 1}
