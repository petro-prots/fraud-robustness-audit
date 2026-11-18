import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import make_sparkov_like
from fraudaudit import audit as audit_mod
from fraudaudit.attacks.random_search import RandomSearch
from fraudaudit.audit import run_audit
from fraudaudit.gate import Check, GateConfig, decide, passed, run_key
from fraudaudit.model.artifacts import load_artifacts
from fraudaudit.model.pipeline import FraudModel
from fraudaudit.report import summary_markdown, write_gate_outputs
from fraudaudit.skew import SkewReport
from fraudaudit.threat import ThreatModel

ROOT = Path(__file__).resolve().parents[1]
TM_PATH = ROOT / "threat-models" / "realistic.yaml"
GATE_PATH = ROOT / "gate.yaml"
ALL = ("fit", "val", "test")  # sparkov_like has too few test episodes for a 50/50 card split


@pytest.fixture(scope="module")
def df():
    return make_sparkov_like(n_cards=80)  # R20: the S half of the fit episodes needs >= cards_per_eval spares


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
def base(df, model, tm, model_dir):
    return run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir,
                     target_periods=ALL, threat_path=TM_PATH, attack_seed=1, one_sided=True,
                     spares_period="test")


def cfg(**kw):
    d = dict(target_fpr=0.01, recall_clean_min=0.5, delta=0.05, attacks=["random", "es"],
             attack_seeds=[1, 2, 3], confidence=0.95, one_sided=True, spares_period="test")
    d.update(kw)
    return GateConfig(**d)


def fake(base, uplift_hi, feasible=1.0, recall=0.9, attack_seed=1):
    """``base`` with a chosen uplift upper bound, feasible share and clean test recall."""
    q = {p: dict(v) for p, v in base.clean_quality.items()}
    q["test"]["recall"] = recall
    return dataclasses.replace(base, uplift=(uplift_hi - 0.02, uplift_hi - 0.04, uplift_hi),
                               feasible_share=feasible, clean_quality=q,
                               meta=dict(base.meta, attack_seed=attack_seed))


def by_name(checks):
    return {c.name: c for c in checks}


def test_gate_yaml_committed_values():
    c = GateConfig.from_yaml(GATE_PATH)
    assert c == GateConfig(target_fpr=0.01, recall_clean_min=0.5, delta=0.10, attacks=["random", "es", "stepwise"],
                           attack_seeds=[1, 2, 3], confidence=0.95, one_sided=True, spares_period="fit")


@pytest.mark.parametrize("change", [
    {"extra": 1},                        # unknown key
    {"delta": "drop"},                   # missing key
    {"attacks": ["random", "ga"]},       # unknown attack
    {"attacks": []},
    {"attack_seeds": []},
    {"attack_seeds": [1, 1]},
    {"attack_seeds": [1.5]},
    {"attack_seeds": [1, -2]},
    {"delta": -0.1},
    {"recall_clean_min": 1.5},
    {"confidence": 1.0},
    {"target_fpr": 0.0},
    {"one_sided": "drop"},
    {"one_sided": "yes"},
    {"one_sided": 1},
    {"one_sided": None},
    {"spares_period": "drop"},
    {"spares_period": "val"},
    {"spares_period": "Fit"},
    {"spares_period": None},
])
def test_gate_yaml_strict(tmp_path, change):
    raw = yaml.safe_load(GATE_PATH.read_text())
    for k, v in change.items():
        if v == "drop":
            raw.pop(k)
        else:
            raw[k] = v
    p = tmp_path / "gate.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        GateConfig.from_yaml(p)


def test_gate_stepwise_456_yaml_differs_from_gate_yaml_only_in_attacks_and_seeds():
    """The tracked fresh-seed file (stepwise seeds 4-6) is gate.yaml with another ensemble, nothing else."""
    extra = GateConfig.from_yaml(ROOT / "gate-stepwise-456.yaml")
    base = GateConfig.from_yaml(GATE_PATH)
    assert extra.attacks == ["stepwise"] and extra.attack_seeds == [4, 5, 6]
    assert dataclasses.replace(extra, attacks=base.attacks, attack_seeds=base.attack_seeds) == base


def test_gate_yaml_delta_null_allowed(tmp_path):
    raw = yaml.safe_load(GATE_PATH.read_text())
    raw["delta"] = None
    p = tmp_path / "gate.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert GateConfig.from_yaml(p).delta is None


def test_worst_attack_decides(base):
    res = {run_key("random", 1): fake(base, 0.02), run_key("es", 1): fake(base, 0.09)}
    c = by_name(decide(cfg(attack_seeds=[1]), res, None))["ver_uplift"]
    assert not c.passed and c.value == pytest.approx(0.09) and c.threshold == 0.05
    assert c.worst == "es/1"
    assert "harden" in c.recommendation
    res[run_key("es", 1)] = fake(base, 0.04)
    c = by_name(decide(cfg(attack_seeds=[1]), res, None))["ver_uplift"]
    assert c.passed and c.value == pytest.approx(0.04) and c.worst == "es/1"
    assert c.passed and c.recommendation == ""


def test_worst_seed_decides(base):
    res = {run_key("random", s): fake(base, u, attack_seed=s) for s, u in ((1, 0.01), (2, 0.07), (3, 0.02))}
    c = by_name(decide(cfg(attacks=["random"]), res, None))["ver_uplift"]
    assert not c.passed and c.value == pytest.approx(0.07) and c.worst == "random/2"


def test_delta_none_fails_closed(base):
    res = {run_key("random", 1): fake(base, -0.5)}  # even a negative uplift does not pass without δ
    c = by_name(decide(cfg(delta=None, attacks=["random"], attack_seeds=[1]), res, None))["ver_uplift"]
    assert not c.passed and c.threshold is None
    assert "δ is not set" in c.recommendation


def test_feasibility_below_one_fails(base):
    one = cfg(attacks=["random"], attack_seeds=[1, 2])
    res = {run_key("random", 1): fake(base, 0.0), run_key("random", 2): fake(base, 0.0, feasible=0.999)}
    c = by_name(decide(one, res, None))["feasibility"]
    assert not c.passed and c.value == pytest.approx(0.999) and c.threshold == 1.0
    assert "tool error" in c.recommendation.lower()
    res[run_key("random", 2)] = fake(base, 0.0, feasible=float("nan"))  # no executed ops: not proven feasible
    assert not by_name(decide(one, res, None))["feasibility"].passed
    res[run_key("random", 2)] = fake(base, 0.0)
    assert by_name(decide(one, res, None))["feasibility"].passed


def test_recall_clean(base):
    one = cfg(attacks=["random"], attack_seeds=[1])
    c = by_name(decide(one, {run_key("random", 1): fake(base, 0.0, recall=0.49)}, None))["recall_clean"]
    assert not c.passed and c.value == pytest.approx(0.49) and c.threshold == 0.5
    assert "model owner" in c.recommendation.lower()
    assert by_name(decide(one, {run_key("random", 1): fake(base, 0.0, recall=0.5)}, None))["recall_clean"].passed


def test_skew_check_only_with_report(base):
    one = cfg(attacks=["random"], attack_seeds=[1])
    res = {run_key("random", 1): fake(base, 0.0)}
    assert [c.name for c in decide(one, res, None)] == ["ver_uplift", "recall_clean", "feasibility"]
    ok = SkewReport(max_abs_diff={"amt": 0.0}, decision_mismatches=0, n_tx=10, passed=True, n_cards=2)
    bad = dataclasses.replace(ok, decision_mismatches=3, passed=False)
    assert by_name(decide(one, res, ok))["skew"].passed
    c = by_name(decide(one, res, bad))["skew"]
    assert not c.passed and c.value == 3 and "DevSecOps" in c.recommendation


def test_decide_requires_every_run(base):
    with pytest.raises(ValueError):  # es/1 missing: the ensemble is incomplete
        decide(cfg(attack_seeds=[1]), {run_key("random", 1): fake(base, 0.0)}, None)
    with pytest.raises(ValueError):  # confidence of the runs differs from the gate
        decide(cfg(attacks=["random"], attack_seeds=[1], confidence=0.9), {run_key("random", 1): fake(base, 0.0)},
               None)


def test_summary_has_dollar_interpretation(base):
    c = cfg(attacks=["random", "es"], attack_seeds=[1])
    res = {run_key("random", 1): fake(base, 0.02), run_key("es", 1): fake(base, 0.30)}
    checks = decide(c, res, None)
    meta = {"gate": dataclasses.asdict(c), "seed": 7, "wall_seconds": 12.3}
    md = summary_markdown(checks, res, meta)
    assert md.splitlines()[0].startswith("# ") and "❌ FAILED" in md.split("\n## ")[0]
    mean_a = base.meta["mean_amount"]  # R32: δ$ = δ · mean A of the targets, not the median
    assert mean_a == pytest.approx(float(np.mean([t["amount"] for t in base.per_target])))
    x = f"{0.30 * mean_a:.0f}"
    y = f"{0.05 * mean_a:.0f}"
    assert f"up to ~{x} $" in md and f"an allowed ~{y} $" in md
    assert "δ$ = δ · mean A" in md and "median A" in md
    assert ("per compromised card" in md) if base.meta["n_target_cards"] == base.n_targets else ("per episode" in md)
    assert "| es | 1 |" in md and "| random | 1 |" in md and "worst" in md
    assert "Recommendations" in md and "fraudaudit harden" in md and "do not weaken the attack or δ" in md
    assert "without the card's legitimate operations inside the episode window (same for the attack)" in md  # replay assumption
    assert "recall" in md and "VER_clean" in md and base.meta["git_sha"] in md
    for name in ("ver_uplift", "recall_clean", "feasibility"):
        assert name in md
    ok = {run_key("random", 1): fake(base, 0.01), run_key("es", 1): fake(base, 0.02)}
    md_ok = summary_markdown(decide(c, ok, None), ok, meta)
    assert "✅ PASSED" in md_ok.split("\n## ")[0] and "FAILED" not in md_ok
    assert "Recommendations" not in md_ok


def test_write_gate_outputs(base, tmp_path):
    c = cfg(attacks=["random"], attack_seeds=[1])
    res = {run_key("random", 1): fake(base, 0.01)}
    checks = decide(c, res, None)
    data = write_gate_outputs(tmp_path, c, checks, res, {"seed": 7})
    saved = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(data))
    assert saved["passed"] is True and saved["verdict"] == "PASSED" and saved["worst_run"] == "random/1"
    assert [ch["name"] for ch in saved["checks"]] == ["ver_uplift", "recall_clean", "feasibility"]
    assert saved["runs"]["random/1"]["uplift"]["upper"] == pytest.approx(0.01)
    assert saved["gate"]["delta"] == 0.05
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("# ")


def test_attack_seed_changes_attack_only(df, model, tm, model_dir, base):
    r2 = run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir,
                   target_periods=ALL, threat_path=TM_PATH, attack_seed=2, spares_period="test",
                   one_sided=True)
    assert [t["card"] for t in r2.per_target] == [t["card"] for t in base.per_target]  # same split
    assert r2.spare_cards == base.spare_cards
    assert [t["ver_clean"] for t in r2.per_target] == [t["ver_clean"] for t in base.per_target]
    assert not np.array_equal(r2.best.to_vector(), base.best.to_vector())  # another attack stream
    assert r2.meta["attack_seed"] == 2 and base.meta["attack_seed"] == 1
    legacy = run_audit(df, model, tm, RandomSearch(), seed=7, n_jobs=1, model_dir=model_dir, target_periods=ALL,
                       spares_period="test", one_sided=True)
    assert legacy.meta["attack_seed"] is None


# --- CLI: exit codes on the synthetic fixture with a tiny model --------------------------------------------


def _fixture_periods(kw):
    """sparkov_like has 2 test episodes: the legacy 50/50 split uses every period; fit-period spares need
    targets outside fit (val + test)."""
    return ALL if kw.get("spares_period", "test") == "test" else ("val", "test")


@pytest.fixture
def cli_env(df, model_dir, tmp_path, monkeypatch):
    import fraudaudit.data as data_mod
    monkeypatch.setattr(data_mod, "load_raw", lambda _d: df)
    real = audit_mod.run_audit

    def all_periods(*a, **kw):
        kw["target_periods"] = _fixture_periods(kw)
        return real(*a, **kw)

    monkeypatch.setattr(audit_mod, "run_audit", all_periods)
    real_baseline = audit_mod.run_single_tx_baseline

    def baseline_all_periods(*a, **kw):
        kw["target_periods"] = _fixture_periods(kw)
        return real_baseline(*a, **kw)

    monkeypatch.setattr(audit_mod, "run_single_tx_baseline", baseline_all_periods)
    threat = tmp_path / "threat.yaml"
    raw = yaml.safe_load(TM_PATH.read_text())
    raw["recon_budget_ops"] = 300
    threat.write_text(yaml.safe_dump(raw))
    summary = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def run(gate, *extra, model=None, command="audit"):
        from fraudaudit.cli import main
        g = tmp_path / "gate.yaml"
        g.write_text(gate if isinstance(gate, str) else yaml.safe_dump(gate))
        out = tmp_path / "out"
        code = main([command, "--data", "unused", "--model", str(model or model_dir), "--threat", str(threat),
                     "--gate", str(g), "--out", str(out), "--seed", "7", "--n-jobs", "1", *extra])
        return code, out, summary
    return run


GATE_LOOSE = dict(target_fpr=0.01, recall_clean_min=0.0, delta=10.0, attacks=["random"], attack_seeds=[1],
                  confidence=0.95, one_sided=True, spares_period="fit")


def test_cli_audit_exit_0_when_all_checks_pass(cli_env):
    code, out, summary = cli_env(GATE_LOOSE)
    assert code == 0
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert data["passed"] is True and list(data["runs"]) == ["random/1"]
    assert "✅ PASSED" in summary.read_text(encoding="utf-8")


def test_cli_audit_exit_1_when_a_check_fails(cli_env):
    code, out, summary = cli_env(dict(GATE_LOOSE, delta=None))
    assert code == 1
    assert json.loads((out / "audit.json").read_text(encoding="utf-8"))["passed"] is False
    assert "❌ FAILED" in (out / "summary.md").read_text(encoding="utf-8")
    assert "❌ FAILED" in summary.read_text(encoding="utf-8")


def test_cli_audit_skew_report_fails_gate(cli_env, tmp_path, model_dir):
    from fraudaudit.skew import skew_to_json
    bad = SkewReport(max_abs_diff={"amt": 0.5}, decision_mismatches=2, n_tx=10, passed=False, n_cards=2)
    p = tmp_path / "skew.json"
    from fraudaudit.cli import _model_sha256
    p.write_text(json.dumps(skew_to_json(bad, model_sha256=_model_sha256(model_dir))))
    code, out, _ = cli_env(GATE_LOOSE, "--skew-report", str(p))
    assert code == 1
    checks = {c["name"]: c for c in json.loads((out / "audit.json").read_text(encoding="utf-8"))["checks"]}
    assert checks["skew"]["passed"] is False and checks["ver_uplift"]["passed"] is True


def test_cli_parsers():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["audit", "--gate", "gate.yaml", "--skew-report", "out/skew.json"])
    assert str(a.gate) == "gate.yaml" and str(a.skew_report) == "out/skew.json" and a.seed == 42
    assert str(build_parser().parse_args(["audit"]).gate) == "gate.yaml"
    assert str(build_parser().parse_args(["skew", "--out", "out/skew.json"]).out) == "out/skew.json"


def test_skew_json_roundtrip():
    from fraudaudit.skew import skew_from_json, skew_to_json
    rep = SkewReport(max_abs_diff={"amt": 0.0, "cnt_1d": 1.0}, decision_mismatches=1, n_tx=5, passed=False,
                     n_cards=3, max_abs_p_diff=0.2)
    assert skew_from_json(json.loads(json.dumps(skew_to_json(rep)))) == rep


# --- fail-closed edge cases ------------------------------------------------------------------------------


def nan_uplift(r):
    return dataclasses.replace(r, uplift=(float("nan"), float("nan"), float("nan")))


def test_nonfinite_uplift_in_a_later_run_fails(base):
    c3 = cfg(attacks=["random"], attack_seeds=[1, 2, 3])
    res = {run_key("random", 1): fake(base, 0.01), run_key("random", 2): nan_uplift(fake(base, 0.01)),
           run_key("random", 3): fake(base, 0.02)}
    c = by_name(decide(c3, res, None))["ver_uplift"]
    assert not c.passed and c.worst == "random/2" and c.recommendation
    res[run_key("random", 2)] = dataclasses.replace(fake(base, 0.01), uplift=(0.0, 0.0, float("inf")))
    assert not by_name(decide(c3, res, None))["ver_uplift"].passed
    md = summary_markdown(decide(c3, res, None), res, {})
    assert "❌ FAILED" in md.split("\n## ")[0]


def test_audit_summary_dollars_use_the_mean_amount(base):
    """R32 p. 7: the dollar line of the audit summary uses the mean A of the targets (the median only as extra)."""
    from fraudaudit.audit import summary_markdown as audit_md, to_json
    md = audit_md(to_json(base))
    mean_a, med = base.meta["mean_amount"], base.meta["median_amount"]
    assert f"share × mean episode amount A of the targets = {mean_a:.2f} $" in md
    assert f"median A {med:.2f} $" in md
    assert f"≈ {base.uplift[0] * mean_a:.0f} $" in md


def test_nonfinite_recall_in_a_later_run_fails(base):
    c2 = cfg(attacks=["random"], attack_seeds=[1, 2])
    res = {run_key("random", 1): fake(base, 0.01), run_key("random", 2): fake(base, 0.01, recall=float("nan"))}
    c = by_name(decide(c2, res, None))["recall_clean"]
    assert not c.passed and c.worst == "random/2"


def test_summary_dollar_wording_per_card_or_episode(base):
    c = cfg(attacks=["random"], attack_seeds=[1])
    res = {run_key("random", 1): fake(base, 0.2)}
    md = summary_markdown(decide(c, res, None), res, {})
    assert "approximately: share × mean episode amount A of the targets" in md
    one_per_card = base.meta["n_target_cards"] == base.n_targets
    assert ("per compromised card (one target episode per card)" in md) == one_per_card
    many = dataclasses.replace(res[run_key("random", 1)],
                               meta=dict(res[run_key("random", 1)].meta, n_target_cards=base.n_targets - 1))
    md2 = summary_markdown(decide(c, {run_key("random", 1): many}, None), {run_key("random", 1): many}, {})
    assert "per episode" in md2 and "per compromised card" not in md2


def test_summary_feasibility_failure_not_rounded_to_100(base):
    c = cfg(attacks=["random"], attack_seeds=[1])
    res = {run_key("random", 1): fake(base, 0.01, feasible=0.9999)}
    md = summary_markdown(decide(c, res, None), res, {})
    row = next(line for line in md.splitlines() if line.startswith("| `feasibility`"))
    assert "99.99 %" in row and "| 100.0 % | = 100.0 %" not in row


def test_summary_model_target_fpr_missing_noted(base):
    c = cfg(attacks=["random"], attack_seeds=[1])
    res = {run_key("random", 1): fake(base, 0.01)}
    md = summary_markdown(decide(c, res, None), res, {"model_target_fpr": None})
    assert "the model's target_fpr is not given – the check was skipped" in md


def test_cli_audit_tampered_model_exit_2_with_summary(cli_env, model_dir, tmp_path):
    import shutil
    bad = tmp_path / "bad_model"
    shutil.copytree(model_dir, bad)
    with open(bad / "mlp.keras", "ab") as f:
        f.write(b"\0")
    code, out, summary = cli_env(GATE_LOOSE, model=bad)
    assert code == 2
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert md.startswith("# ❌ FAILED – configuration/artifact error") and "SHA" in md.upper()
    assert "❌ FAILED – configuration/artifact error" in summary.read_text(encoding="utf-8")
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert data["passed"] is False and data["error"]


@pytest.mark.parametrize("extra", [{"model_sha256": "0" * 64}, {}])
def test_cli_audit_skew_report_of_another_model_exit_2(cli_env, tmp_path, extra):
    from fraudaudit.skew import skew_to_json
    ok = SkewReport(max_abs_diff={"amt": 0.0}, decision_mismatches=0, n_tx=10, passed=True, n_cards=2)
    p = tmp_path / "skew.json"
    p.write_text(json.dumps(skew_to_json(ok, **extra)))
    code, out, summary = cli_env(GATE_LOOSE, "--skew-report", str(p))
    assert code == 2
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "another model" in md and "configuration/artifact error" in md
    assert "another model" in summary.read_text(encoding="utf-8")


def test_cli_audit_bad_yaml_exit_2_with_summary(cli_env):
    code, out, summary = cli_env("target_fpr: [0.01\n")
    assert code == 2
    assert "configuration/artifact error" in (out / "summary.md").read_text(encoding="utf-8")
    assert "configuration/artifact error" in summary.read_text(encoding="utf-8")


def test_cli_audit_records_model_target_fpr(cli_env):
    code, out, _ = cli_env(GATE_LOOSE)
    assert code == 0
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert data["meta"]["model_target_fpr"] is None  # the tiny test model has no meta.target_fpr
    assert "the model's target_fpr is not given – the check was skipped" in (out / "summary.md").read_text(encoding="utf-8")


# --- single-transaction baseline in the gate report (informational) ----------------------------------------

SINGLE_PATH = ROOT / "threat-models" / "single-transaction.yaml"


@pytest.fixture(scope="module")
def baseline(df, model, tm, model_dir):
    """``{amount_rule: BaselineResult}``, the primary rule (``first``) first."""
    from fraudaudit.audit import run_single_tx_baseline
    from fraudaudit.threat import SingleTxThreatModel
    tms = SingleTxThreatModel.from_yaml(SINGLE_PATH)
    return {r: run_single_tx_baseline(df, model, tms, tm, seed=7, n_jobs=1, model_dir=model_dir,
                                      target_periods=ALL, single_path=SINGLE_PATH, amount_rule=r,
                                      spares_period="test")
            for r in ("first", "max")}


SECTION = "## Conventional audit (single-transaction attack) vs episode audit"


def test_baseline_section_in_summary(base, baseline):
    from fraudaudit.audit import _pct
    from fraudaudit.report import baseline_comparison
    c = cfg(attacks=["random"], attack_seeds=[1, 2])
    res = {run_key("random", 1): fake(base, 0.02), run_key("random", 2): fake(base, 0.30, attack_seed=2)}
    checks = decide(c, res, None)
    md = summary_markdown(checks, res, {}, baseline=baseline)
    assert SECTION in md
    sec = md.split(SECTION)[1].split("\n## ")[0]
    for rule, b in baseline.items():
        row = next(line for line in sec.splitlines() if line.startswith(f"| `{rule}`"))
        assert _pct(b.evaded_tx_share) in row and _pct(b.evaded_amount_share) in row
        assert f"{b.n_evaded} of {b.n_ops}" in row and _pct(b.replay_ver[0]) in row
        assert _pct(b.replay_uplift[2]) in row and str(b.meta["queries"]) in row
    assert "(primary)" in next(line for line in sec.splitlines() if line.startswith("| `first`"))
    w = res[run_key("random", 2)]
    assert _pct(w.uplift[0]) in sec and _pct(w.ver_adv[0]) in sec and "`random/2`" in sec
    cmp = baseline_comparison(baseline["first"], res, "random/2")
    assert cmp["n_runs"] == 2 and _pct(cmp["uplift_mean_runs"]) in sec
    assert _pct(cmp["uplift_min_runs"]) in sec and _pct(cmp["uplift_max_runs"]) in sec
    assert _pct(cmp["adv_minus_replay"][0]) in sec
    assert "does not affect the gate verdict" in sec and "state of the target card itself" in sec
    assert "earlier than the original operation" in sec
    # informational only: the verdict and the checks do not change
    md0 = summary_markdown(checks, res, {})
    assert SECTION not in md0
    assert md.split("\n## ")[0] == md0.split("\n## ")[0]


def test_baseline_comparison_paired(base, baseline):
    from fraudaudit.metrics import uplift_ci
    from fraudaudit.report import baseline_comparison
    b = baseline["first"]
    res = {run_key("random", 1): base}
    cmp = baseline_comparison(b, res, "random/1")
    adv = np.array([t["ver_adv"] for t in base.per_target])
    rep = np.array([t["ver_replay"] for t in b.per_target])
    assert cmp["adv_minus_replay"] == uplift_ci(adv, rep, 0.95, 2000, 7)
    assert cmp["worst_uplift"] == tuple(base.uplift)
    other = dataclasses.replace(b, per_target=b.per_target[::-1])
    with pytest.raises(ValueError):
        baseline_comparison(other, res, "random/1")


def test_baseline_interpretation_direction(baseline):
    from fraudaudit.report import baseline_interpretation
    b = dataclasses.replace(baseline["first"], replay_ver=(0.3, 0.25, 0.35), replay_uplift=(0.28, 0.2, 0.33))
    cmp = {"worst_uplift": (0.6, 0.5, 0.7)}
    high, inside = (dataclasses.replace(b, evaded_amount_share=x) for x in (1.0, 0.3))
    weaker = dict(cmp, adv_minus_replay=(0.4, 0.3, 0.5))
    stronger = dict(cmp, adv_minus_replay=(-0.4, -0.5, -0.3))
    same = dict(cmp, adv_minus_replay=(0.01, -0.05, 0.07))
    t = baseline_interpretation(high, weaker)
    assert t.startswith("The per-operation evasion rate is not a loss estimate") and "yield only 30.0 %" in t
    assert "the attack it finds is weaker than the episode attack: uplift 28.0 % vs 60.0 %" in t
    assert "is consistent with" in baseline_interpretation(inside, weaker)
    assert "stronger than the episode attack" in baseline_interpretation(high, stronger)
    assert "statistically indistinguishable" in baseline_interpretation(high, same)
    for x in (high, inside):
        for c in (weaker, stronger, same):
            text = baseline_interpretation(x, c)
            assert text.endswith(".") and text.count(". ") == 0  # one sentence


def test_write_gate_outputs_with_baseline(base, baseline, tmp_path):
    c = cfg(attacks=["random"], attack_seeds=[1])
    res = {run_key("random", 1): fake(base, 0.01)}
    checks = decide(c, res, None)
    data = write_gate_outputs(tmp_path, c, checks, res, {"seed": 7}, baseline=baseline)
    saved = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    bl = saved["baseline"]
    assert bl["primary"] == "first" and list(bl["rules"]) == ["first", "max"]
    assert bl["rules"]["max"]["chosen_amount_ratio"] == pytest.approx(baseline["max"].chosen_amount_ratio)
    assert bl["rules"]["first"]["replay_ver"]["mean"] == pytest.approx(baseline["first"].replay_ver[0])
    assert len(bl["comparison"]["adv_minus_replay"]) == 3
    assert saved["passed"] is True and data["passed"] is True
    assert SECTION in (tmp_path / "summary.md").read_text(encoding="utf-8")
    write_gate_outputs(tmp_path, c, checks, res, {"seed": 7})
    assert json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))["baseline"] is None
    write_gate_outputs(tmp_path, c, checks, res, {"seed": 7}, baseline_error="RuntimeError: boom")
    assert json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))["baseline"] == {
        "error": "RuntimeError: boom"}
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert SECTION in md and "RuntimeError: boom" in md and "✅ PASSED" in md.split("\n## ")[0]


def test_baseline_comparison_failure_is_a_note_not_a_verdict(base, baseline, tmp_path, monkeypatch):
    from fraudaudit import report
    c = cfg(attacks=["random"], attack_seeds=[1])
    res = {run_key("random", 1): fake(base, 0.01)}
    checks = decide(c, res, None)

    def boom(*a, **kw):
        raise RuntimeError("comparison boom")

    monkeypatch.setattr(report, "baseline_comparison", boom)
    data = write_gate_outputs(tmp_path, c, checks, res, {"seed": 7}, baseline=baseline)
    saved = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert saved["baseline"] == {"error": "RuntimeError: comparison boom"}
    assert saved["passed"] is True and data["passed"] is True
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "comparison boom" in md and "✅ PASSED" in md.split("\n## ")[0]


def test_cli_audit_runs_baseline_informational(cli_env):
    code, out, _ = cli_env(GATE_LOOSE)
    assert code == 0
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert list(data["baseline"]["rules"]) == ["first", "max"]
    assert 0 <= data["baseline"]["rules"]["first"]["evaded_tx_share"] <= 1
    assert SECTION in (out / "summary.md").read_text(encoding="utf-8")
    code, out, _ = cli_env(dict(GATE_LOOSE, delta=None))  # the baseline never changes the verdict
    assert code == 1
    code, out, _ = cli_env(GATE_LOOSE, "--no-baseline")
    assert code == 0
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert data["baseline"] is None and SECTION not in (out / "summary.md").read_text(encoding="utf-8")


def test_cli_audit_baseline_error_keeps_verdict(cli_env, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("baseline exploded")
    monkeypatch.setattr(audit_mod, "run_single_tx_baseline", boom)
    code, out, _ = cli_env(GATE_LOOSE)
    assert code == 0
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert data["passed"] is True and data["baseline"] == {"error": "RuntimeError: baseline exploded"}
    assert "baseline exploded" in (out / "summary.md").read_text(encoding="utf-8")


def test_cli_audit_bad_single_threat_exit_2(cli_env, tmp_path):
    bad = tmp_path / "single.yaml"
    bad.write_text("knowledge: score\n")
    code, out, _ = cli_env(GATE_LOOSE, "--single-threat", str(bad))
    assert code == 2
    assert "configuration/artifact error" in (out / "summary.md").read_text(encoding="utf-8")


# --- R14: one-sided uplift bound and the spares period in the gate --------------------------------------------


def test_decide_rejects_runs_of_another_method(base):
    res = {run_key("random", 1): base}
    decide(cfg(attacks=["random"], attack_seeds=[1]), res, None)  # base: one-sided, legacy split
    with pytest.raises(ValueError, match="one_sided"):
        decide(cfg(attacks=["random"], attack_seeds=[1], one_sided=False), res, None)
    with pytest.raises(ValueError, match="spares_period"):
        decide(cfg(attacks=["random"], attack_seeds=[1], spares_period="fit"), res, None)


def test_run_ensemble_passes_method(df, model, tm, model_dir, monkeypatch):
    from fraudaudit.gate import run_ensemble
    seen = []

    def spy(*a, **kw):
        seen.append((kw["spares_period"], kw["one_sided"], kw["confidence"]))
        return base_like

    base_like = object()
    monkeypatch.setattr(audit_mod, "run_audit", spy)
    out = run_ensemble(df, model, tm, cfg(attacks=["random"], attack_seeds=[1, 2], spares_period="fit"), 7,
                       n_jobs=1, log=None)
    assert list(out) == ["random/1", "random/2"]
    assert seen == [("fit", True, 0.95)] * 2


def test_summary_names_the_one_sided_bound(base):
    c = cfg(attacks=["random"], attack_seeds=[1])
    res = {run_key("random", 1): base}
    md = summary_markdown(decide(c, res, None), res, {"gate": dataclasses.asdict(c)})
    assert "one-sided" in md and "two-sided" in md  # uplift one-sided, VER intervals two-sided


def test_cli_audit_fit_spares(cli_env):
    code, out, _ = cli_env(GATE_LOOSE)
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    run = data["runs"]["random/1"]
    assert run["meta"]["spares_period"] == "fit" and run["meta"]["one_sided"] is True
    assert data["gate"]["spares_period"] == "fit" and data["gate"]["one_sided"] is True
    assert "fit period" in (out / "summary.md").read_text(encoding="utf-8")


# --- calibrate: resolution of the gate on the baseline model --------------------------------------------------


def _with_uplift(base, mean, upper, se, attack_seed=1):
    return dataclasses.replace(base, uplift=(mean, mean - 0.1, upper),
                               meta=dict(base.meta, attack_seed=attack_seed, uplift_se=se))


def test_calibration_numbers(base):
    from fraudaudit.calibrate import calibration
    c = cfg(attacks=["random", "es"], attack_seeds=[1, 2])
    res = {"random/1": _with_uplift(base, 0.30, 0.36, 0.035),
           "random/2": _with_uplift(base, 0.40, 0.47, 0.04, 2),
           "es/1": _with_uplift(base, 0.10, 0.18, 0.05),
           "es/2": _with_uplift(base, 0.25, 0.30, 0.03, 2)}
    data = calibration(res, c, [0.02, 0.2, 0.5], {"seed": 7})
    r = data["resolution"]
    assert r["max_upper_minus_mean"] == pytest.approx(0.08) and r["max_upper_minus_mean_run"] == "es/1"
    assert r["seed_spread"] == {"random": pytest.approx(0.10), "es": pytest.approx(0.15)}
    assert r["seed_spread_max"] == pytest.approx(0.15)
    assert r["mean_range_all_runs"] == pytest.approx(0.30)
    assert r["min_certifiable_delta"] == pytest.approx(0.08)
    assert data["worst_run"] == "random/2" and data["worst_upper"] == pytest.approx(0.47)
    assert data["runs"]["es/1"] == {"attack": "es", "attack_seed": 1, "mean": 0.10, "upper": 0.18,
                                    "upper_minus_mean": pytest.approx(0.08), "se": 0.05,
                                    "lower": pytest.approx(0.0), "ver_adv_mean": base.ver_adv[0],
                                    "ver_clean_mean": base.ver_clean[0], "n_targets": base.n_targets}
    rows = {row["delta"]: row for row in data["verdict_vs_delta"]}
    assert rows[0.02]["baseline_verdict"] == "FAILED" and rows[0.5]["baseline_verdict"] == "PASSED"
    assert rows[0.2]["baseline_verdict"] == "FAILED"
    assert rows[0.02]["zero_uplift_model_passes"] is False and rows[0.2]["zero_uplift_model_passes"] is True
    assert data["one_sided"] is True and data["gate"]["delta"] == 0.05


def test_calibration_markdown(base):
    from fraudaudit.calibrate import calibration, calibration_markdown
    c = cfg(attacks=["random"], attack_seeds=[1, 2])
    res = {"random/1": _with_uplift(base, 0.30, 0.36, 0.035), "random/2": _with_uplift(base, 0.40, 0.47, 0.04, 2)}
    md = calibration_markdown(calibration(res, c, [0.02, 0.05, 0.10, 0.15, 0.20], {"seed": 7}))
    assert md.startswith("# Gate resolution calibration")
    assert "| δ |" in md and "| 2.0 % |" in md and "| 20.0 % |" in md
    assert "Resolution" in md and "spread" in md and "minimal δ" in md.lower()
    assert "| random | 2 |" in md
    assert "gate.yaml" in md and "not changed" in md


def test_cli_calibrate_writes_table(cli_env):
    gate_text = yaml.safe_dump(dict(GATE_LOOSE, attack_seeds=[1, 2]))
    code, out, _ = cli_env(gate_text, "--deltas", "0.02,0.05,0.1", command="calibrate")
    assert code == 0
    data = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert list(data["runs"]) == ["random/1", "random/2"]
    assert [row["delta"] for row in data["verdict_vs_delta"]] == [0.02, 0.05, 0.1]
    for k, r in data["runs"].items():
        assert r["upper"] >= r["mean"] and r["se"] >= 0
    assert data["resolution"]["max_upper_minus_mean"] == pytest.approx(
        max(r["upper"] - r["mean"] for r in data["runs"].values()))
    md = (out / "calibration.md").read_text(encoding="utf-8")
    assert "| 5.0 % |" in md and "| 10.0 % |" in md
    assert data["meta"]["spares_period"] == "fit" and data["meta"]["git_sha"]
    assert (out.parent / "gate.yaml").read_text() == gate_text  # never changes gate.yaml
    assert not (out / "audit.json").exists()


def test_cli_calibrate_parser_defaults():
    from fraudaudit.cli import build_parser
    a = build_parser().parse_args(["calibrate"])
    assert str(a.out) == "out/calibration" and a.deltas == [0.02, 0.05, 0.10, 0.15, 0.20] and a.seed == 42
    with pytest.raises(SystemExit):
        build_parser().parse_args(["calibrate", "--deltas", "0.1,x"])


# ---- positive control: the replay attack (Task 19, R30) -------------------------------------------------------

def test_replay_attack_gives_zero_uplift_and_passes(df, model, tm, model_dir):
    from fraudaudit.gate import run_ensemble
    g = cfg(attacks=["replay"], attack_seeds=[1, 2], delta=0.10, recall_clean_min=0.0, spares_period="fit",
            positive_control=True)
    res = run_ensemble(df, model, tm, g, 7, n_jobs=1, model_dir=model_dir)
    for r in res.values():
        assert r.uplift == (0.0, 0.0, 0.0) and r.recon_ops == 0 and r.feasible_share == 1.0
        assert all(t["ver_adv"] == t["ver_clean"] and t["replay"] for t in r.per_target)
        assert any(t["ver_clean"] > 0 for t in r.per_target)  # the control is not trivially zero VER
    checks = by_name(decide(g, res))
    assert checks["ver_uplift"].passed and checks["ver_uplift"].value == 0.0
    assert all(c.passed for c in checks.values())


def test_replay_parallel_identical(df, model, tm, model_dir):
    from fraudaudit.gate import run_ensemble
    g = cfg(attacks=["replay"], attack_seeds=[1], spares_period="fit", positive_control=True)
    a = run_ensemble(df, model, tm, g, 7, n_jobs=1, model_dir=model_dir)["replay/1"]
    b = run_ensemble(df, model, tm, g, 7, n_jobs=2, model_dir=model_dir)["replay/1"]
    assert a.per_target == b.per_target and a.uplift == b.uplift


def test_cli_positive_control_green(cli_env):
    code, out, summary = cli_env(dict(GATE_LOOSE, attacks=["replay"], delta=0.10, positive_control=True),
                                 "--no-baseline")
    assert code == 0
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert data["passed"] is True and list(data["runs"]) == ["replay/1"]
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "Positive control" in md and "not a gate verdict" in md


def test_committed_positive_control_and_seed_gates():
    base = GateConfig.from_yaml(GATE_PATH)
    pc = GateConfig.from_yaml(ROOT / "gate-positive-control.yaml")
    s456 = GateConfig.from_yaml(ROOT / "gate-seeds-456.yaml")
    assert pc == dataclasses.replace(base, attacks=["replay"], positive_control=True)
    assert "replay" not in base.attacks and not base.positive_control
    assert "replay" not in GATE_PATH.read_text() and "positive_control" not in GATE_PATH.read_text()
    assert s456 == dataclasses.replace(base, attack_seeds=[4, 5, 6])
    s49 = GateConfig.from_yaml(ROOT / "gate-seeds-4-9.yaml")  # Task 19b front confirmation (R32)
    assert s49 == dataclasses.replace(base, attack_seeds=[4, 5, 6, 7, 8, 9])


@pytest.mark.parametrize("attacks,pc", [(["replay"], False), (["random", "replay"], True), (["random", "replay"], False),
                                        (["random"], True)])
def test_replay_only_in_a_marked_positive_control_gate(attacks, pc):
    raw = yaml.safe_load(GATE_PATH.read_text())
    raw["attacks"] = attacks
    if pc:
        raw["positive_control"] = True
    with pytest.raises(ValueError, match="positive"):
        GateConfig.from_mapping(raw)
    with pytest.raises(ValueError, match="positive"):
        cfg(attacks=attacks, positive_control=pc)


def test_positive_control_discriminates_ver_vs_uplift_mutant(df, model, tm, model_dir):
    """M3 (review): on a toy case with VER_clean above δ, a gate that compared the VER_adv upper bound instead of the
    uplift upper bound would turn the replay control RED; the real gate passes. On the real data VER_clean is ≈ 0.9 %,
    so there the control cannot catch this mutant (README: scope of the positive control)."""
    from fraudaudit.gate import run_ensemble
    g = cfg(attacks=["replay"], attack_seeds=[1], delta=0.001, recall_clean_min=0.0, spares_period="fit",
            positive_control=True)  # a small δ: the toy case needs VER_clean > δ
    res = run_ensemble(df, model, tm, g, 7, n_jobs=1, model_dir=model_dir)
    r = res["replay/1"]
    assert r.ver_clean[0] > g.delta  # the toy case: normal fraud already above δ
    assert passed(decide(g, res))
    mutant = {k: dataclasses.replace(v, uplift=v.ver_adv) for k, v in res.items()}  # «VER_adv upper ≤ δ»
    assert not by_name(decide(g, mutant))["ver_uplift"].passed
