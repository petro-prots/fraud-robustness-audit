# models-bachelor-hour – baseline with calendar features (kept for comparison)

The baseline model trained with the 36-feature **bachelor+hour** feature set (29 base features: the 27 of the bachelor model incl. `trans_month`/`trans_year`, plus `hour_sin`/`hour_cos`; 7 windowed aggregates; MLP input 37), seed 42, from the clean tree at commit `1a51b7b` (`meta.git_sha` in `model.json`).

**Why superseded (ruling R23):** `trans_year` × `trans_month` put the test period (Jun–Dec 2020) outside the training support. The baseline barely used them, but adversarially retrained models extrapolated on them (hardened test recall fell from 0.91 in June to 0.45 in December; freezing the month restored 0.943). `models/` now holds the `bachelor+hour-calendar` model (27 base features, 34 features, MLP input 35), introduced in `2f3dce5`.

**Not loadable by the current code**: `model.json` `feature_columns` (36) differ from `FEATURE_COLUMNS` (34), so `load_artifacts` refuses it. To use it, check out a commit before `2f3dce5` (e.g. `69c02e1`).

Clean quality at τ = 0.005090115591883659 (seed 42): test recall 0.9692, FPR 0.0083, precision 0.3103; val recall 0.9769, FPR 0.0089, precision 0.3802.
Audit gate (R20 spares, `realistic.yaml`, seed 42, commit `47db3cd`): FAILED `ver_uplift`, worst run `es/1` – one-sided upper bound 57.0 %; calibration resolution (commit `c4f1d9f`, before R20) max(upper − mean) 4.1 pp.
