"""Quality-gate rules: the verdict that turns a PR check red or green.

``gate.yaml`` (``GateConfig.from_yaml``, strict: every key required, unknown keys rejected)::

    target_fpr: 0.01        # FPR per operation the model's τ was chosen for (checked against model.json)
    recall_clean_min: 0.5   # recall on clean test operations at τ
    delta: 0.05             # allowed uplift VER_adv − VER_clean (share of the episode amount A)
    attacks: [random, es, stepwise]  # the ensemble: every attack ... (stepwise: held-out, no hardening data)
    attack_seeds: [1, 2, 3] # ... with every attack seed, on the same target/spare split
    confidence: 0.95        # level of every interval
    one_sided: true         # the uplift upper bound is one-sided (the gate tests only uplift ≤ δ); VER stays two-sided
    spares_period: fit      # reconnaissance spares: fit-period episodes, targets = every test episode (R14);
                            # test = legacy 50/50 card split of the test episodes

Checks (``decide``):

* ``ver_uplift`` – the worst case over attacks × seeds: max of ``AuditResult.uplift[2]`` (upper bound of
  the paired bootstrap of per-episode VER_adv − VER_clean, one-sided when ``one_sided``) must be ≤ δ.
  ``delta: null`` fails closed. Runs made with another ``confidence``, ``one_sided`` or ``spares_period``
  are rejected.
* ``recall_clean`` – recall on clean ``test`` operations at τ ≥ ``recall_clean_min``.
* ``feasibility`` – every run's ``feasible_share`` == 1 (ruling R3): every executed adversarial operation
  passes the threat model; anything else is a bug of the tool and invalidates the audit.
* ``skew`` – only when a ``SkewReport`` is given: the batch and serving feature paths agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .data import SPARES_PERIODS

ATTACKS = ("random", "es", "stepwise")
HELD_OUT_ATTACKS = ("stepwise",)  # never generates hardening data (``harden.check_generation_attack`` rejects them)
GATE_KEYS = ("target_fpr", "recall_clean_min", "delta", "attacks", "attack_seeds", "confidence", "one_sided",
             "spares_period")

REC_VER_UPLIFT = ("Model owner: the model is vulnerable to the episode attack – retrain it with adversarial examples "
                  "(`fraudaudit harden`) and rerun the audit; do not weaken the attack or δ.")
REC_DELTA_NONE = ("Gate owner: δ is not set – the gate author must approve `delta` in `gate.yaml` after "
                  "VER_clean has been measured; until then the gate stays closed.")
REC_RECALL = ("Model owner: on clean data the model misses too much fraud – check the data, "
              "the training and τ, and retrain the model; do not lower `recall_clean_min`.")
REC_FEASIBILITY = ("Audit tool developer: tool error – the attack executed operations outside the threat model, "
                   "so the audit result is invalid – report the bug (issue) with `out/audit.json`; "
                   "the model is not at fault.")
REC_NONFINITE = ("Audit tool developer: a metric was not computed (NaN or infinity) in run "
                 "`{run}` – the audit result is invalid; report the bug (issue) with `out/audit.json`.")
REC_SKEW = ("DevSecOps: the serving feature path diverges from the training one – fix the serving path "
            "before merging; an audit of such a model does not describe its behaviour in production.")


@dataclass(frozen=True)
class GateConfig:
    target_fpr: float
    recall_clean_min: float
    delta: float | None
    attacks: list[str]
    attack_seeds: list[int]
    confidence: float
    one_sided: bool
    spares_period: str

    @classmethod
    def from_yaml(cls, path: Path | str) -> "GateConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(raw, str(path))

    @classmethod
    def from_mapping(cls, raw, where: str = "gate") -> "GateConfig":
        """Strict: every key of ``GATE_KEYS`` required, unknown keys rejected, every value checked."""
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: expected a mapping")
        extra = sorted(set(raw) - set(GATE_KEYS))
        missing = [k for k in GATE_KEYS if k not in raw]
        if extra:
            raise ValueError(f"{where}: unknown keys {extra}")
        if missing:
            raise ValueError(f"{where}: missing keys {missing}")

        def num(k):
            v = raw[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{where}: {k} must be a finite number, got {v!r}")
            return float(v)

        attacks, seeds = raw["attacks"], raw["attack_seeds"]
        if not isinstance(attacks, list) or not attacks or any(a not in ATTACKS for a in attacks) \
                or len(set(attacks)) != len(attacks):
            raise ValueError(f"{where}: attacks must be a non-empty list of distinct {list(ATTACKS)}, got {attacks!r}")
        if not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0
                                                           for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"{where}: attack_seeds must be a non-empty list of distinct integers >= 0, "
                             f"got {seeds!r}")
        if not isinstance(raw["one_sided"], bool):
            raise ValueError(f"{where}: one_sided must be true or false, got {raw['one_sided']!r}")
        if raw["spares_period"] not in SPARES_PERIODS:
            raise ValueError(f"{where}: spares_period must be one of {list(SPARES_PERIODS)}, "
                             f"got {raw['spares_period']!r}")
        c = cls(target_fpr=num("target_fpr"), recall_clean_min=num("recall_clean_min"),
                delta=None if raw["delta"] is None else num("delta"), attacks=list(attacks),
                attack_seeds=[int(s) for s in seeds], confidence=num("confidence"),
                one_sided=raw["one_sided"], spares_period=str(raw["spares_period"]))
        if not 0 < c.target_fpr < 1:
            raise ValueError(f"{where}: target_fpr must be in (0, 1)")
        if not 0 <= c.recall_clean_min <= 1:
            raise ValueError(f"{where}: recall_clean_min must be in [0, 1]")
        if c.delta is not None and c.delta < 0:
            raise ValueError(f"{where}: delta must be >= 0 or null")
        if not 0 < c.confidence < 1:
            raise ValueError(f"{where}: confidence must be in (0, 1)")
        return c

    def run_keys(self) -> list[str]:
        """Every run of the ensemble, attacks outer, seeds inner."""
        return [run_key(a, s) for a in self.attacks for s in self.attack_seeds]


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None
    threshold: float | None
    passed: bool
    recommendation: str = ""  # empty when passed; else what to do and whose parameter it is
    worst: str | None = None  # run key that decided the check (attacks × seeds)
    detail: dict = field(default_factory=dict)


def run_key(attack: str, attack_seed: int) -> str:
    return f"{attack}/{int(attack_seed)}"


def parse_run_key(key: str) -> tuple[str, int]:
    attack, seed = key.rsplit("/", 1)
    return attack, int(seed)


def make_attack(name: str, tm):
    from .attacks.evolution import EvolutionStrategy
    from .attacks.random_search import RandomSearch
    from .attacks.stepwise import StepwiseAttack
    if name == "random":
        return RandomSearch()
    if name == "es":
        return EvolutionStrategy.from_params(tm.es)
    if name == "stepwise":
        return StepwiseAttack()
    raise ValueError(f"unknown attack {name!r}")


def run_ensemble(df, model, tm, cfg: GateConfig, seed: int, *, n_jobs: int | None = None, model_dir=None,
                 threat_path=None, log=None) -> dict:
    """``run_audit`` for every attack × attack seed of ``cfg``, one after another (each run uses its own
    process pool internally), on the same split ``seed``. Returns ``{run_key: AuditResult}``."""
    from . import audit
    results = {}
    for name in cfg.attacks:
        for s in cfg.attack_seeds:
            r = audit.run_audit(df, model, tm, make_attack(name, tm), seed=seed, n_jobs=n_jobs,
                                model_dir=model_dir, threat_path=threat_path, attack_seed=s,
                                confidence=cfg.confidence, one_sided=cfg.one_sided,
                                spares_period=cfg.spares_period)
            results[run_key(name, s)] = r
            if log is not None:
                log(f"{run_key(name, s)}: uplift mean {r.uplift[0]:.4f}, upper {r.uplift[2]:.4f}, "
                    f"VER_adv {r.ver_adv[0]:.4f}, {r.meta['wall_seconds']} s")
    return results


def decide(cfg: GateConfig, results: dict, skew=None) -> list:
    """The gate checks over ``results`` (``{run_key: AuditResult}``, exactly ``cfg.run_keys()``)."""
    if not results:
        raise ValueError("no audit runs")
    missing = [k for k in cfg.run_keys() if k not in results]
    if missing:
        raise ValueError(f"runs missing for the gate ensemble: {missing}")
    for k, r in results.items():
        conf = r.meta.get("confidence")
        if conf is not None and not math.isclose(float(conf), cfg.confidence):
            raise ValueError(f"run {k}: confidence {conf} differs from the gate's {cfg.confidence}")
        for key in ("one_sided", "spares_period"):
            got = r.meta.get(key)
            if got is not None and got != getattr(cfg, key):
                raise ValueError(f"run {k}: {key} {got!r} differs from the gate's {getattr(cfg, key)!r}")

    per_run = {k: float(r.uplift[2]) for k, r in results.items()}
    bad_up = [k for k, v in per_run.items() if not math.isfinite(v)]  # max() over NaN is order-dependent
    worst = bad_up[0] if bad_up else max(per_run, key=per_run.get)
    w_up = per_run[worst]
    if bad_up:
        up = Check("ver_uplift", w_up, cfg.delta, False, REC_NONFINITE.format(run=worst), worst,
                   {"uplift_upper": per_run, "non_finite": bad_up})
    elif cfg.delta is None:
        up = Check("ver_uplift", w_up, None, False, REC_DELTA_NONE, worst, {"uplift_upper": per_run})
    else:
        ok = w_up <= cfg.delta
        up = Check("ver_uplift", w_up, cfg.delta, ok, "" if ok else REC_VER_UPLIFT, worst, {"uplift_upper": per_run})

    recalls = {k: float(r.clean_quality["test"]["recall"]) for k, r in results.items()}
    bad_rec = [k for k, v in recalls.items() if not math.isfinite(v)]
    if bad_rec:
        rec = Check("recall_clean", recalls[bad_rec[0]], cfg.recall_clean_min, False,
                    REC_NONFINITE.format(run=bad_rec[0]), bad_rec[0])
    else:
        r_key = min(recalls, key=recalls.get)
        r_ok = recalls[r_key] >= cfg.recall_clean_min
        rec = Check("recall_clean", recalls[r_key], cfg.recall_clean_min, r_ok, "" if r_ok else REC_RECALL, r_key)

    shares = {k: float(r.feasible_share) for k, r in results.items()}
    f_key = min(shares, key=lambda k: -1.0 if math.isnan(shares[k]) else shares[k])
    f_ok = all(v == 1.0 for v in shares.values())  # nan (no executed ops) is not proven feasible
    feas = Check("feasibility", shares[f_key], 1.0, f_ok, "" if f_ok else REC_FEASIBILITY, f_key,
                 {"feasible_share": shares})

    checks = [up, rec, feas]
    if skew is not None:
        checks.append(Check("skew", int(skew.decision_mismatches), 0, bool(skew.passed),
                            "" if skew.passed else REC_SKEW, None,
                            {"n_tx": int(skew.n_tx), "n_cards": int(skew.n_cards),
                             "max_abs_p_diff": float(skew.max_abs_p_diff)}))
    return checks


def passed(checks) -> bool:
    return all(c.passed for c in checks)
