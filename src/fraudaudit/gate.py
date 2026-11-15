"""Quality-gate rules: the verdict that turns a PR check red or green.

``gate.yaml`` (``GateConfig.from_yaml``, strict: every key required, unknown keys rejected)::

    target_fpr: 0.01        # FPR per operation the model's τ was chosen for (checked against model.json)
    recall_clean_min: 0.5   # recall on clean test operations at τ
    delta: 0.05             # allowed uplift VER_adv − VER_clean (share of the episode amount A)
    attacks: [random, es]   # the ensemble: every attack ...
    attack_seeds: [1, 2, 3] # ... with every attack seed, on the same target/spare split
    confidence: 0.95        # level of every interval

Checks (``decide``):

* ``ver_uplift`` – the worst case over attacks × seeds: max of ``AuditResult.uplift[2]`` (upper bound of
  the paired bootstrap of per-episode VER_adv − VER_clean) must be ≤ δ. ``delta: null`` fails closed.
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

ATTACKS = ("random", "es")
GATE_KEYS = ("target_fpr", "recall_clean_min", "delta", "attacks", "attack_seeds", "confidence")

REC_VER_UPLIFT = ("Власник моделі: модель вразлива до епізодної атаки – донавчіть її змагальними прикладами "
                  "(`fraudaudit harden`) і повторіть аудит; не послаблюйте атаку чи δ.")
REC_DELTA_NONE = ("Власник шлюзу: δ не задано – автор шлюзу має затвердити `delta` у `gate.yaml` після "
                  "вимірювання VER_clean; до того шлюз закритий.")
REC_RECALL = ("Власник моделі: на чистих даних модель пропускає забагато шахрайства – перевірте дані, "
              "навчання і τ, перенавчіть модель; не знижуйте `recall_clean_min`.")
REC_FEASIBILITY = ("Розробник інструмента аудиту: помилка інструмента – атака виконала операції поза моделлю "
                   "загроз, тож результат аудиту недійсний – повідомте про помилку (issue) з `out/audit.json`; "
                   "модель тут ні до чого.")
REC_SKEW = ("DevSecOps: продуктивний шлях ознак (serving) розходиться з навчальним – виправте serving path "
            "до злиття; аудит такої моделі не описує її поведінку в продакшені.")


@dataclass(frozen=True)
class GateConfig:
    target_fpr: float
    recall_clean_min: float
    delta: float | None
    attacks: list[str]
    attack_seeds: list[int]
    confidence: float

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
        if not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) or not isinstance(s, int)
                                                           for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"{where}: attack_seeds must be a non-empty list of distinct integers, got {seeds!r}")
        c = cls(target_fpr=num("target_fpr"), recall_clean_min=num("recall_clean_min"),
                delta=None if raw["delta"] is None else num("delta"), attacks=list(attacks),
                attack_seeds=[int(s) for s in seeds], confidence=num("confidence"))
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
    if name == "random":
        return RandomSearch()
    if name == "es":
        return EvolutionStrategy.from_params(tm.es)
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
                                confidence=cfg.confidence)
            results[run_key(name, s)] = r
            if log is not None:
                log(f"{run_key(name, s)}: uplift upper {r.uplift[2]:.4f}, VER_adv {r.ver_adv[0]:.4f}, "
                    f"{r.meta['wall_seconds']} s")
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

    worst = max(results, key=lambda k: results[k].uplift[2])
    w_up = float(results[worst].uplift[2])
    per_run = {k: float(r.uplift[2]) for k, r in results.items()}
    if cfg.delta is None:
        up = Check("ver_uplift", w_up, None, False, REC_DELTA_NONE, worst, {"uplift_upper": per_run})
    else:
        ok = w_up <= cfg.delta
        up = Check("ver_uplift", w_up, cfg.delta, ok, "" if ok else REC_VER_UPLIFT, worst, {"uplift_upper": per_run})

    recalls = {k: float(r.clean_quality["test"]["recall"]) for k, r in results.items()}
    r_key = min(recalls, key=recalls.get)
    r_ok = recalls[r_key] >= cfg.recall_clean_min  # nan → False
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
