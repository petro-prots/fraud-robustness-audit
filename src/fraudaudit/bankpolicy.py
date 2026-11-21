"""Bank policy as a parameter of the gate (R33): the bank chooses the policy and the bar, the gate checks them.

``policies/<name>.yaml`` (strict: both keys required, unknown keys rejected)::

    name: strict-to-fraud           # free text, shown in the report
    max_declines: 3                 # the card is blocked after this many declines (md3)
    corridor: {kappa: 1.25, q: 0.98}   # V1 daily split κ and corridor quantile q; or ``corridor: off``

The policy overlays the threat model: ``max_declines`` and the corridor's κ and q are replaced, every other
corridor setting (lookback, floor, multipliers, V2 …) comes from the threat file, which must therefore carry a
``corridor:`` section when the policy switches the corridor on. The threshold τ is the model's own (the gate checks
``target_fpr`` against ``model.json``). Card escalation (``tau_after_decline``) is not a policy key here: the
threat file must not set it, so the policy file describes the whole reaction of the bank.

The bar is δ′ of the gate (``gate.yaml`` ``delta`` or ``audit --delta``): the share of the episode amount the bank
accepts the attacker to gain on top of ordinary fraud. By the margin rule (R33) it is chosen as
``U + X`` (the worst measured uplift of the policy plus the margin X), not as the bare ``U``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

POLICY_KEYS = ("name", "max_declines", "corridor")


@dataclass(frozen=True)
class BankPolicy:
    name: str
    max_declines: int
    corridor_on: bool
    kappa: float | None = None
    q: float | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BankPolicy":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(raw, str(path))

    @classmethod
    def from_mapping(cls, raw, where: str = "policy") -> "BankPolicy":
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: expected a mapping")
        extra, missing = sorted(set(raw) - set(POLICY_KEYS)), [k for k in POLICY_KEYS if k not in raw]
        if extra:
            raise ValueError(f"{where}: unknown keys {extra}")
        if missing:
            raise ValueError(f"{where}: missing keys {missing}")
        name = raw["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}: name must be a non-empty string")
        md = raw["max_declines"]
        if isinstance(md, bool) or not isinstance(md, int) or md < 1:
            raise ValueError(f"{where}: max_declines must be an integer >= 1, got {md!r}")
        c = raw["corridor"]
        if c is None or c == "off" or c is False:
            return cls(name=name, max_declines=md, corridor_on=False)
        if not isinstance(c, dict) or set(c) != {"kappa", "q"}:
            raise ValueError(f"{where}: corridor must be `off` or a mapping with exactly kappa and q, got {c!r}")

        def num(k):
            v = c[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{where}: corridor.{k} must be a finite number, got {v!r}")
            return float(v)

        kappa, q = num("kappa"), num("q")
        if kappa < 1:
            raise ValueError(f"{where}: corridor.kappa must be >= 1, got {kappa}")
        if not 0 < q < 1:
            raise ValueError(f"{where}: corridor.q must be in (0, 1), got {q}")
        return cls(name=name, max_declines=md, corridor_on=True, kappa=kappa, q=q)

    def apply(self, tm):
        """The threat model under this policy. ValueError if the threat file already sets the escalation, or has no
        corridor section while the policy needs one (the file would otherwise silently decide)."""
        if tm.tau_after_decline is not None:
            raise ValueError("the threat file sets tau_after_decline: the policy file must describe the whole "
                             "reaction of the bank, so the threat file may not (set it to null)")
        if not self.corridor_on:
            return dataclasses.replace(tm, max_declines=self.max_declines, corridor=None)
        if tm.corridor is None:
            raise ValueError("the policy switches the corridor on, but the threat file has no `corridor:` section "
                             "(use threat-models/realistic-corridor-v1v2.yaml)")
        if tm.corridor.daily_mode != "weekly_split":
            raise ValueError("the policy sets the V1 daily split κ: the threat file's corridor must use "
                             "`daily: {mode: weekly_split, ...}`")
        corridor = dataclasses.replace(tm.corridor, q=self.q, kappa=self.kappa)
        return dataclasses.replace(tm, max_declines=self.max_declines, corridor=corridor)

    def public(self) -> dict:
        return {"name": self.name, "max_declines": self.max_declines,
                "corridor": "off" if not self.corridor_on else {"kappa": self.kappa, "q": self.q}}


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
