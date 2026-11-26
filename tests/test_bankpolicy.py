"""Bank policy and bar as parameters of the gate (R33): ``--policy`` / ``--delta`` of ``fraudaudit audit``."""
import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from fraudaudit.bankpolicy import BankPolicy
from fraudaudit.threat import ThreatModel

ROOT = Path(__file__).resolve().parents[1]
TM_V1V2 = ROOT / "threat-models" / "realistic-corridor-v1v2.yaml"
GOOD = {"name": "p", "max_declines": 3, "corridor": {"kappa": 1.25, "q": 0.98}}


def test_committed_policy_values():
    p = BankPolicy.from_yaml(ROOT / "policies" / "strict-to-fraud.yaml")
    assert (p.name, p.max_declines, p.corridor_on, p.kappa, p.q) == ("strict-to-fraud", 3, True, 1.25, 0.98)


@pytest.mark.parametrize("change", [
    {"extra": 1}, {"name": ""}, {"name": 3}, {"max_declines": 0}, {"max_declines": True}, {"max_declines": 1.5},
    {"corridor": {"kappa": 0.9, "q": 0.98}}, {"corridor": {"kappa": 2, "q": 1.0}}, {"corridor": {"kappa": 2}},
    {"corridor": {"kappa": 2, "q": 0.99, "floor": 1}}, {"corridor": {"kappa": True, "q": 0.99}},
    {"corridor": {"kappa": float("nan"), "q": 0.99}}, {"corridor": "on"}, {"corridor": [1, 2]},
])
def test_policy_strict(change):
    with pytest.raises(ValueError):
        BankPolicy.from_mapping({**GOOD, **change})


@pytest.mark.parametrize("missing", ["name", "max_declines", "corridor"])
def test_policy_requires_every_key(missing):
    raw = {k: v for k, v in GOOD.items() if k != missing}
    with pytest.raises(ValueError, match="missing"):
        BankPolicy.from_mapping(raw)


def test_policy_not_a_mapping():
    with pytest.raises(ValueError):
        BankPolicy.from_mapping([1])


@pytest.mark.parametrize("off", ["off", None, False])
def test_corridor_off_spellings(off):
    p = BankPolicy.from_mapping({**GOOD, "corridor": off})
    assert p.corridor_on is False and p.public()["corridor"] == "off"


def test_apply_replaces_reaction_and_corridor_kappa_q_only():
    tm = ThreatModel.from_yaml(TM_V1V2)
    out = BankPolicy.from_mapping(GOOD).apply(tm)
    assert out.max_declines == 3 and tm.max_declines == 1  # the input is not changed
    assert (out.corridor.kappa, out.corridor.q) == (1.25, 0.98)
    # everything else of the corridor and of the threat model is the file's
    assert dataclasses.replace(out.corridor, kappa=tm.corridor.kappa, q=tm.corridor.q) == tm.corridor
    assert dataclasses.replace(out, max_declines=tm.max_declines, corridor=tm.corridor) == tm
    assert out.corridor.confirmed_extends is True  # V2 stays on


def test_apply_corridor_off():
    out = BankPolicy.from_mapping({**GOOD, "corridor": "off"}).apply(ThreatModel.from_yaml(TM_V1V2))
    assert out.corridor is None and out.max_declines == 3


def test_apply_refuses_escalation_in_the_threat_file():
    tm = dataclasses.replace(ThreatModel.from_yaml(TM_V1V2), tau_after_decline=0.5)
    with pytest.raises(ValueError, match="tau_after_decline"):
        BankPolicy.from_mapping(GOOD).apply(tm)


def test_apply_needs_a_corridor_section_when_the_policy_switches_it_on():
    tm = ThreatModel.from_yaml(ROOT / "threat-models" / "realistic.yaml")
    with pytest.raises(ValueError, match="no `corridor:` section"):
        BankPolicy.from_mapping(GOOD).apply(tm)
    assert BankPolicy.from_mapping({**GOOD, "corridor": "off"}).apply(tm).max_declines == 3


def test_apply_needs_the_weekly_split_daily_mode():
    tm = ThreatModel.from_yaml(TM_V1V2)
    tm = dataclasses.replace(tm, corridor=dataclasses.replace(tm.corridor, daily_mode="quantile", kappa=None))
    with pytest.raises(ValueError, match="weekly_split"):
        BankPolicy.from_mapping(GOOD).apply(tm)


def test_ci_policy_job_pins_policy_and_bar():
    """The demonstration job: strict-to-fraud on the hardened M3 with δ′ = U + X rounded up (21.9 + 5.98 = 27.9 → 28 %)."""
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "audit.yml").read_text(encoding="utf-8"))
    env = wf["jobs"]["policy-gate"]["env"]
    assert (env["MODEL"], env["POLICY"], env["THREAT"], env["DELTA"]) == (
        "models-hardened-m3", "policies/strict-to-fraud.yaml", "threat-models/realistic-corridor-v1v2.yaml", "0.28")
    assert 0.219 + 0.0598 <= float(env["DELTA"]) < 0.30
    for path in (env["POLICY"], env["THREAT"], env["MODEL"] + "/model.json"):
        assert (ROOT / path).exists()
    run = " ".join(s.get("run", "") for s in wf["jobs"]["policy-gate"]["steps"])
    assert '--policy "$POLICY"' in run and '--delta "$DELTA"' in run and "--skew-report" in run
    assert "policies/**" in wf[True]["pull_request"]["paths"] and "models-hardened-m3/**" in wf[True]["pull_request"]["paths"]


def test_hardened_model_matches_its_manifest():
    from fraudaudit.model.artifacts import load_artifacts
    a = load_artifacts(ROOT / "models-hardened-m3")  # SHA-256 verified before deserialisation
    assert a is not None
    meta = json.loads((ROOT / "models-hardened-m3" / "model.json").read_text(encoding="utf-8"))["meta"]
    assert meta["target_fpr"] == 0.01 and meta["harden"]["round"] == 5
