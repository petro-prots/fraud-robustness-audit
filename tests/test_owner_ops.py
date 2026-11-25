"""Task 28 (R54): the owner's operations inside the attack window (episode interleaving, audit legs, policy-sweep
--owner-ops, the owner-ops summary) and the reproduction check audit-diff."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from fraudaudit.episode import run_adaptive_episode, run_episode
from fraudaudit.owner_ops import load_trace, owner_ok, owner_path, owner_summary, write_owner
from fraudaudit.policy_sweep import cell_key
from fraudaudit.serving.core import Decision
from fraudaudit.serving.state import Tx
from test_episode import T0, _ctx
from test_policy_sweep import _setup, env  # noqa: F401  (module fixture: synthetic data and a trained model)


class AmtServing:
    """p by the operation's amount (declined iff p >= 0.5); records the amounts in decision order and the state size."""

    tau = 0.5

    def __init__(self, p_by_amt):
        self.p, self.seen, self.sizes = dict(p_by_amt), [], []

    def decide(self, state, tx):
        self.seen.append(tx.amt)
        self.sizes.append(len(state))
        p = self.p.get(tx.amt, 0.0)
        return Decision(p=p, declined=p >= 0.5)


def _tx(minutes, amt):
    return Tx(T0 + pd.Timedelta(minutes=minutes), float(amt), "fraud_A", "misc_pos", 40.0, -80.0)


ATT = [_tx(0, 10), _tx(10, 20), _tx(20, 30)]


# ---- interleaving in run_episode ------------------------------------------------------------------------------------

def test_owner_ops_interleaved_by_time_and_ties_attacker_first():
    s = AmtServing({})
    own = [_tx(5, 101), _tx(10, 102), _tx(25, 103)]
    o = run_episode(s, _ctx(), ATT, None, 0.8, owner_txs=own)
    # 101 (5 min) before 20 (10 min); 102 at 10 min ties with 20 -> attacker first; 103 after the last attacker op: not run
    assert s.seen == [10, 101, 20, 102, 30]
    assert s.sizes == [1, 2, 3, 4, 5]  # every operation enters the state
    assert (o.n_ops, o.passed_amount, o.decisions, o.owner_ops, o.owner_lost) == (3, 60.0, (False,) * 3, 2, 0)
    assert o.ver == pytest.approx(0.6)  # the owner's amounts never count


def test_without_owner_ops_nothing_changes():
    for md in (None, 1, 2):
        for ps in ({}, {20: 0.9}, {10: 0.7, 30: 0.6}):
            a = run_episode(AmtServing(ps), _ctx(), ATT, md, 0.8)
            for own in (None, []):
                assert run_episode(AmtServing(ps), _ctx(), ATT, md, 0.8, owner_txs=own) == a
            assert run_episode(AmtServing(ps), _ctx(), ATT, md, 0.8, owner_txs=[], owner_declines_count=False) == a


def test_counted_owner_decline_blocks_the_card():
    s = AmtServing({101: 0.9})
    own = [_tx(5, 101), _tx(15, 102), _tx(16, 104)]
    o = run_episode(s, _ctx(), ATT, 1, 0.8, owner_txs=own, owner_declines_count=True)
    assert s.seen == [10, 101]  # blocked by the owner's decline before the attacker's 2nd op
    assert (o.n_ops, o.passed_amount, o.blocked, o.blocked_by_owner, o.n_declines) == (1, 10.0, True, True, 1)
    assert (o.owner_ops, o.owner_declined, o.owner_lost) == (1, 1, 2) and o.decisions == (False,)


def test_uncounted_owner_decline_neither_blocks_nor_escalates_but_enters_the_state():
    s = AmtServing({101: 0.9, 20: 0.4})
    own = [_tx(5, 101)]
    o = run_episode(s, _ctx(), ATT, 1, 0.8, owner_txs=own, owner_declines_count=False, tau_after_decline=0.3)
    assert s.seen == [10, 101, 20, 30] and s.sizes == [1, 2, 3, 4]
    assert (o.blocked, o.n_declines, o.escalated, o.owner_declined, o.passed_amount) == (False, 0, False, 1, 60.0)
    # counted: the owner's decline escalates the threshold, 20 (p 0.4 >= 0.3) is declined -> 2nd decline
    s = AmtServing({101: 0.9, 20: 0.4})
    o = run_episode(s, _ctx(), ATT, 2, 0.8, owner_txs=own, owner_declines_count=True, tau_after_decline=0.3)
    assert s.seen == [10, 101, 20]
    assert (o.blocked, o.n_declines, o.escalated, o.blocked_by_owner) == (True, 2, True, False)
    assert o.decisions == (False, True) and o.owner_lost == 0


def test_attacker_block_loses_the_remaining_owner_ops():
    o = run_episode(AmtServing({20: 0.9}), _ctx(), ATT, 1, 0.8, owner_txs=[_tx(5, 101), _tx(15, 102), _tx(30, 103)])
    assert (o.blocked, o.blocked_by_owner, o.owner_ops, o.owner_lost) == (True, False, 1, 2)


def test_owner_ops_must_be_in_time_order():
    with pytest.raises(ValueError, match="time order"):
        run_episode(AmtServing({}), _ctx(), ATT, 1, 0.8, owner_txs=[_tx(15, 1), _tx(5, 2)])


# ---- closed loop: the policy is told only about its own operations -------------------------------------------------

class FixedPolicy:
    def __init__(self, txs):
        self.txs, self.got = list(txs), []

    def act(self, ctx, tm, merchants, rng):
        for tx in self.txs:
            flag = yield tx
            self.got.append(flag)


def test_adaptive_with_owner_ops_equals_run_episode_and_sends_only_own_flags():
    tm = dataclasses.replace(_threat(), max_declines=None, tau_after_decline=None, corridor=None, k_max=20)
    own = [_tx(5, 101), _tx(12, 102)]
    for counts in (True, False):
        pol = FixedPolicy(ATT)
        a, executed = run_adaptive_episode(AmtServing({101: 0.9, 20: 0.8}), _ctx(), pol, tm, None,
                                           np.random.default_rng(0), tau_after_decline=None, owner_txs=own,
                                           owner_declines_count=counts)
        b = run_episode(AmtServing({101: 0.9, 20: 0.8}), _ctx(), ATT, None, tm.rho, owner_txs=own,
                        owner_declines_count=counts)
        assert a == b and executed == ATT
        assert pol.got == [False, True, False]  # the decisions on 10, 20, 30 only; nothing about 101 / 102
    tm1 = dataclasses.replace(tm, max_declines=1)
    pol = FixedPolicy(ATT)
    o, executed = run_adaptive_episode(AmtServing({101: 0.9}), _ctx(), pol, tm1, None, np.random.default_rng(0),
                                       tau_after_decline=None, owner_txs=own, owner_declines_count=True)
    assert executed == ATT[:1] and o.blocked_by_owner and o.owner_lost == 1 and pol.got == [False]


def _threat():
    from pathlib import Path

    from fraudaudit.threat import ThreatModel
    return ThreatModel.from_yaml(Path(__file__).resolve().parents[1] / "threat-models" / "realistic.yaml")


# ---- the owner's operations of a window, the audit legs --------------------------------------------------------------

def test_owner_window_txs_on_synthetic_data(env):  # noqa: F811
    from fraudaudit.audit import owner_window_txs
    from fraudaudit.data import fraud_episodes
    df = env[0]
    eps = fraud_episodes(df, {"test"})
    n_any = 0
    for e in eps:
        txs, n_other = owner_window_txs(df, e, 7)
        g = df[(df["cc_num"] == e.card) & (df["ts"] >= e.start) & (df["ts"] < e.start + pd.Timedelta(days=7))]
        legit = g[(g["is_fraud"] == 0) & ~g["row"].isin(e.rows)].sort_values(["ts", "row"])
        assert [(t.ts, t.amt) for t in txs] == list(zip(legit["ts"], legit["amt"]))
        assert n_other == int(((g["is_fraud"] == 1) & ~g["row"].isin(e.rows)).sum())
        n_any += bool(txs)
    assert n_any > 0  # the synthetic windows hold owner operations


@pytest.fixture(scope="module")
def audits(env):  # noqa: F811
    from fraudaudit.audit import run_audit
    from fraudaudit.gate import make_attack
    df, model, d = env
    tm, _, _ = _setup(env)
    tm = dataclasses.replace(tm, max_declines=2)
    kw = dict(seed=1, n_jobs=1, model_dir=d, attack_seed=1, spares_period="fit", one_sided=True)
    out = {}
    for att in ("es", "stepwise"):
        out[att] = (run_audit(df, model, tm, make_attack(att, tm), **kw),
                    run_audit(df, model, tm, make_attack(att, tm), owner_ops=True, **kw))
    return out


def test_audit_plain_leg_unchanged_with_owner_ops(audits):
    from fraudaudit.audit import to_json
    for a, b in audits.values():
        assert all("owner" not in t for t in a.per_target) and all("owner" in t for t in b.per_target)
        ja, jb = to_json(a), to_json(b)
        jb["per_target"] = [{k: v for k, v in t.items() if k != "owner"} for t in jb["per_target"]]
        for j in (ja, jb):
            j["meta"].pop("wall_seconds")
        assert ja == jb


def test_owner_legs_equal_plain_without_owner_ops_and_are_consistent(audits):
    n_with, n_decl, n_diff = 0, 0, 0
    for a, b in audits.values():
        for t in b.per_target:
            o = t["owner"]
            for v in ("count", "nocount"):
                x = o[v]
                assert x["owner_ops"] + x["owner_lost"] <= o["n_window_ops"]
                assert x["owner_declined"] <= x["owner_ops"]
                assert not x["blocked_by_owner"] or (x["blocked"] and v == "count")
                if o["n_window_ops"] == 0:
                    assert (x["ver_adv"], x["ver_clean"], x["blocked"], x["n_ops"]) == \
                        (t["ver_adv"], t["ver_clean"], t["blocked"], t["n_ops"])
                n_decl += x["owner_declined"]
                n_diff += x["ver_adv"] != t["ver_adv"] or x["ver_clean"] != t["ver_clean"]
            if o["count"]["owner_declined"] == 0:  # without an owner decline both variants are the same run
                assert {k: v for k, v in o["count"].items()} == {k: v for k, v in o["nocount"].items()}
            n_with += o["n_window_ops"] > 0
    assert n_with > 0 and n_decl > 0 and n_diff > 0  # the branches are exercised


# ---- policy-sweep --owner-ops and the summary ---------------------------------------------------------------------------

def _sweep(env, out, owner, n_jobs=1, reactions=("md1", "md3")):  # noqa: F811
    from fraudaudit.policy_sweep import policy_sweep
    df, model, d = env
    tm, ctm, g = _setup(env)
    return policy_sweep(df, model, d, "m", tm, ctm, g, fprs=(0.01,), reactions=reactions, corridors=("off", "on"),
                        esc_fpr=0.2, seed=1, n_jobs=n_jobs, out_dir=out / "m", friction=False, owner_ops=owner)


VOLATILE = ("wall_seconds", "ensemble_seconds", "provenance")


def _stable(cells):
    return [{k: v for k, v in c.items() if k not in VOLATILE} for c in cells]


@pytest.fixture(scope="module")
def swept(env, tmp_path_factory):  # noqa: F811
    plain_dir, own_dir = tmp_path_factory.mktemp("plain"), tmp_path_factory.mktemp("owner")
    return _sweep(env, plain_dir, False), plain_dir, _sweep(env, own_dir, True), own_dir


def test_owner_ops_change_no_number_and_no_hash(swept):
    plain, plain_dir, own, own_dir = swept
    assert _stable(plain["cells"]) == _stable(own["cells"])
    assert not (plain_dir / "m" / "owner").exists()
    for c in own["cells"]:
        p = owner_path(own_dir / "m", c["key"])
        assert owner_ok(p, c["config_hash"])
        f = load_trace(p)
        assert f["targets"] == c["targets"] and sorted(f["runs"]) == sorted(c["runs"])
        assert all(len(v) == len(c["targets"]) for v in f["runs"].values())


def _cells(sweep_dir, ref_dir, data):
    return [("m", sweep_dir / "m" / "policy-sweep.json", c["key"], ref_dir / "m" / "policy-sweep.json")
            for c in data["cells"]]


def test_summary_statistic_and_cross_check(swept):
    from fraudaudit.metrics import uplift_ci
    plain, plain_dir, own, own_dir = swept
    res = owner_summary(_cells(own_dir, plain_dir, own))
    assert res["all_equal"]
    g, seen = own["gate"], 0
    for c, rec in zip(res["cells"], own["cells"]):
        assert c["plain"]["U"] == rec["U"] and c["plain"]["worst_run"] == rec["worst_run"]
        f = load_trace(owner_path(own_dir / "m", rec["key"]))
        for v in ("count", "nocount"):
            for k, lst in f["runs"].items():
                _, _, hi = uplift_ci(np.array([o[v]["ver_adv"] for o in lst]), np.array([o[v]["ver_clean"] for o in lst]),
                                     g["confidence"], g["n_boot"], own["seed"], one_sided=g["one_sided"])
                assert c[v]["runs"][k]["uplift_upper"] == hi
            assert c[v]["U"] == max(r["uplift_upper"] for r in c[v]["runs"].values())
            assert c[v]["dU"] == c[v]["U"] - c["plain"]["U"]
            seen += c[v]["dU"] != 0
    assert seen > 0
    from fraudaudit.owner_ops import markdown
    assert "verdict" not in markdown(res).lower().replace("no verdict", "")


def test_summary_cross_check_difference_exits_non_zero(swept, tmp_path, capsys):
    import shutil

    from fraudaudit.cli import main
    _, plain_dir, own, own_dir = swept
    ref = tmp_path / "ref" / "m"
    shutil.copytree(plain_dir / "m", ref)
    sp = ref / "policy-sweep.json"
    d = json.loads(sp.read_text())
    d["cells"][0]["runs"]["es/1"]["ver_adv"][0] += 1e-9
    sp.write_text(json.dumps(d))
    k = d["cells"][0]["key"]
    rc = main(["owner-ops", "--cell", f"m,{own_dir / 'm' / 'policy-sweep.json'},{k},{sp}", "--out", str(tmp_path / "o")])
    assert rc == 1 and "differs" in (tmp_path / "o" / "owner-ops.md").read_text()
    rc = main(["owner-ops", "--cell", f"m,{own_dir / 'm' / 'policy-sweep.json'},{k},"
                                      f"{plain_dir / 'm' / 'policy-sweep.json'}", "--out", str(tmp_path / "o2")])
    assert rc == 0
    rc = main(["owner-ops", "--cell", f"m,{plain_dir / 'm' / 'policy-sweep.json'},{k},"
                                      f"{plain_dir / 'm' / 'policy-sweep.json'}", "--out", str(tmp_path / "o3")])
    assert rc == 2  # a sweep without owner files is refused


def test_owner_resume_reruns_only_cells_without_owner_file(env, swept, monkeypatch, tmp_path):  # noqa: F811
    import shutil

    from fraudaudit import policy_sweep as mod
    plain, plain_dir, own, own_dir = swept
    calls = []
    real = mod.run_cell
    monkeypatch.setattr(mod, "run_cell", lambda *a, **k: (calls.append(k.get("owner_ops")), real(*a, **k))[1])
    work = tmp_path / "w"
    shutil.copytree(own_dir, work)
    _sweep(env, work, True)
    assert calls == []
    gone = cell_key(0.01, "md3", "off")
    owner_path(work / "m", gone).unlink()
    again = _sweep(env, work, True)
    assert calls == [True] and owner_path(work / "m", gone).is_file()
    assert load_trace(owner_path(work / "m", gone)) == load_trace(owner_path(own_dir / "m", gone))
    assert _stable(again["cells"]) == _stable(own["cells"])
    work2 = tmp_path / "p"
    shutil.copytree(plain_dir, work2)
    _sweep(env, work2, True, reactions=("md1",))  # a plain sweep gains the owner files, numbers identical
    assert calls == [True, True, True]


def test_owner_file_with_workers_equals_in_process(env, swept, tmp_path):  # noqa: F811
    _, _, _, own_dir = swept
    _sweep(env, tmp_path, True, n_jobs=2, reactions=("md3",))
    for k in (cell_key(0.01, "md3", "off"), cell_key(0.01, "md3", "on")):
        assert load_trace(owner_path(tmp_path / "m", k))["runs"] == load_trace(owner_path(own_dir / "m", k))["runs"]


def test_owner_ok(tmp_path):
    p = tmp_path / "o.json.gz"
    assert not owner_ok(p, "h")
    write_owner(p, {"config_hash": "h"})
    assert owner_ok(p, "h") and not owner_ok(p, "x")
    p.write_bytes(b"broken")
    assert not owner_ok(p, "h")


# ---- audit-diff -----------------------------------------------------------------------------------------------------

def test_audit_diff(tmp_path):
    from fraudaudit.audit_diff import add_comparison, compare
    ref = {"meta": {"git_sha": "a", "wall_seconds": 1.0}, "runs": {"es/1": {"uplift": [0.1, 0.0, 0.2],
                                                                             "per_target": [{"ver": 0.5}]}},
           "baseline": {"rules": {"first": {"meta": {"wall_seconds": 3}, "n": 4}}}, "old_key": 1}
    new = json.loads(json.dumps(ref))
    new["meta"].update(git_sha="b", wall_seconds=9.0, versions={"numpy": "2"})
    ref["meta"]["versions"] = {"numpy": "1"}
    new["baseline"]["rules"]["first"]["meta"]["wall_seconds"] = 7
    new["new_key"] = 2
    new.pop("old_key")
    r = compare(ref, new)
    assert r["equal"] and r["only_reference"] == ["old_key"] and r["only_new"] == ["new_key"]
    assert r["volatile_differences"] == ['versions: {"numpy": "1"} -> {"numpy": "2"}']
    new["runs"]["es/1"]["per_target"][0]["ver"] = 0.5 + 1e-12
    new["runs"]["es/1"]["uplift"] = [0.1, 0.0]
    r = compare(ref, new)
    assert not r["equal"] and r["n_value_differences"] == 2
    assert {d["path"] for d in r["differences"]} == {"runs.es/1.per_target[0].ver", "runs.es/1.uplift"}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(ref)), b.write_text(json.dumps(new))
    add_comparison(a, a, "same", tmp_path / "out")
    res = add_comparison(a, b, "other", tmp_path / "out")
    add_comparison(a, b, "other", tmp_path / "out")  # the same label is replaced, not appended
    allr = json.loads((tmp_path / "out" / "repro.json").read_text())
    assert [c["label"] for c in allr["comparisons"]] == ["same", "other"] and not res["equal"]
    md = (tmp_path / "out" / "repro.md").read_text()
    assert "| same |" in md and "**differs**" in md and "runs.es/1.uplift" in md


def test_cli_parsers():
    from fraudaudit.cli import build_parser
    p = build_parser()
    a = p.parse_args(["policy-sweep", "--model", "m", "--label", "x", "--owner-ops"])
    assert a.owner_ops
    a = p.parse_args(["owner-ops", "--cell", "a,b,c,d", "--out", "o"])
    assert a.cell == ["a,b,c,d"]
    a = p.parse_args(["audit-diff", "--reference", "a", "--new", "b", "--label", "l", "--out", "o"])
    assert a.label == "l"


# ---- review fixes: the corridor path of an owner operation; a rerun that does not reproduce; blocked statistics ------

def test_owner_operation_over_the_corridor_is_confirmed_and_extends_v2():
    from fraudaudit import corridor as cor
    from fraudaudit.episode import _Bank
    from fraudaudit.serving.state import CardState
    from test_corridor import PROF, StubServing, T0 as CT0, cfg, ctx_with, tx
    h = pd.Timedelta(hours=1)
    for extends in (False, True):
        c = cfg(floor_24h=1e6, floor_7d=500.0, confirm_fail_counts_as_decline=False, confirmed_extends=extends)
        bank = _Bank(StubServing(), ctx_with(CardState(PROF)), 1, 0.8, None, c)
        assert bank.submit(tx(CT0, 60.0)) is False  # attacker, auto-approved
        n_spend = len(bank.state.spend)
        assert bank.submit(tx(CT0 + h, 600.0), background=True) is False  # 60 + 600 > 500: confirmed, passes
        assert (bank.owner_ops, bank.owner_confirms, bank.passed, bank.n_confirms) == (1, 1, 60.0, 0)
        assert len(bank.state.spend) == n_spend + 1 and bank.confirmed_ops == [] and len(bank.auto_ops) == 1
        t = (CT0 + 2 * h).value
        assert bank.state.confirmed_sum_cents(t, cor.WINDOWS_NS["7d"]) == 60_000  # recorded; used only with V2
        # the attacker's next 480: 60 + 480 > 500 without V2; with V2 C' = 500 + 600
        assert cor.check(c, bank.state, CT0 + 2 * h, 480.0)[0] is extends
        o = bank.outcome()
        assert o.owner_confirms == 1 and o.passed_amount == 60.0 and o.decisions == (False,)


def test_strategy_legs_realise_identical_attacker_operations(env):  # noqa: F811
    from fraudaudit.data import fraud_episodes
    from fraudaudit.episode import execute_candidate
    from fraudaudit.audit import owner_window_txs
    from fraudaudit.serving.core import Serving
    from fraudaudit.threat import MerchantTable, Strategy, context_for
    df, model, _ = env
    tm = dataclasses.replace(_setup(env)[0], max_declines=None)
    merchants = MerchantTable.from_df(df)
    n = 0
    for e in fraud_episodes(df, {"test"})[:10]:
        ctx = context_for(df, e)
        own, _ = owner_window_txs(df, e, 7)
        s = dataclasses.replace(Strategy.random(tm, len(merchants), np.random.default_rng(1)), k=8)
        _, plain = execute_candidate(Serving(model), ctx, s, tm, merchants, np.random.default_rng([1, 3, 0]))
        for counts in (True, False):
            o, ops = execute_candidate(Serving(model), ctx, s, tm, merchants, np.random.default_rng([1, 3, 0]),
                                       owner_txs=own, owner_declines_count=counts)
            assert ops == plain and o.n_ops == len(plain)  # never blocked (mdinf): every planned operation, same ops
        n += bool(own)
    assert n > 0


def test_owner_rerun_that_differs_from_the_record_is_refused(env, swept, tmp_path):  # noqa: F811
    import shutil
    _, _, _, own_dir = swept
    work = tmp_path / "w"
    shutil.copytree(own_dir, work)
    k = cell_key(0.01, "md1", "off")
    owner_path(work / "m", k).unlink()
    sp = work / "m" / "policy-sweep.json"
    d = json.loads(sp.read_text())
    next(c for c in d["cells"] if c["key"] == k)["runs"]["es/1"]["ver_adv"][0] += 1e-9
    sp.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    before = sp.read_bytes()
    with pytest.raises(ValueError, match="differs from the stored record"):
        _sweep(env, work, True)
    assert sp.read_bytes() == before and not owner_path(work / "m", k).exists()


def test_blocked_statistics():
    from fraudaudit.owner_ops import owner_counts
    base = {k: 0 for k in ("owner_ops", "owner_declined", "owner_confirms", "owner_lost", "clean_owner_ops",
                           "clean_owner_declined", "clean_owner_confirms", "clean_owner_lost")}
    mk = lambda **kw: {"count": {**base, "blocked": False, "blocked_by_owner": False, "clean_blocked": False,  # noqa: E731
                                 "clean_blocked_by_owner": False, **kw}}
    runs = {"es/1": [mk(blocked=True, owner_declined=1, owner_ops=2),  # md3: owner declined once, attacker blocked it
                     mk(blocked=True, blocked_by_owner=True, owner_declined=1, owner_ops=1),
                     mk(blocked=True), mk(owner_declined=1, owner_ops=1)]}
    o = owner_counts(runs, "count")
    assert (o["n"], o["blocked"], o["blocked_with_owner_decline"], o["blocked_by_owner"]) == (4, 3, 2, 1)
