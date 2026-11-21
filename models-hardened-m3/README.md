# models-hardened-m3

M3 (38 features, MLP + IsolationForest) after 5 rounds of adversarial hardening (RS + ES generation, stepwise held out),
copied unchanged from the local run `models-harden3/round-5` (hardening code at f330b91; `model.json` records the
parameters, `adversarial-rows.csv` of the run is not kept here). τ = 0.872019 at target FPR 0.01. Used by the
`policy-gate` job of `.github/workflows/audit.yml` (demonstration of the bank policy as a parameter of the gate).
