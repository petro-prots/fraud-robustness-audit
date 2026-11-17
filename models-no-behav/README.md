# models-no-behav – baseline without behavioural features (kept for comparison)

The baseline model trained with the 34-feature **bachelor+hour-calendar** feature set (27 base features: the bachelor set without `trans_month`/`trans_year`, plus `hour_sin`/`hour_cos`; 7 windowed aggregates; MLP input 35), seed 42, from the clean tree at commit `2f3dce5` (`meta.git_sha` in `model.json`), committed in `25484c9`.

**Why superseded (Task 23):** its shortest window is one day. The held-out stepwise attack beat the models hardened from it with fast, small, same-category bursts (~75 s apart, ~100 $): final gate of Task 21 RED, worst `stepwise/2` one-sided upper bound 62.2 %. `models/` now holds the `bachelor+hour-calendar+behav` model (38 features: + `cnt_1h`, `sum_1h`, `cnt_cat_24h`, `amt_ratio_30d`; MLP input 39), introduced in `41dea63`.

**Loadable again (Task 19):** `model.json` `feature_columns` (34) are the registered feature set `bachelor+hour-calendar` (`features.columns.FEATURE_SETS`), an ordered subset of `FEATURE_COLUMNS` (38). Features are computed as for the current model and the model selects its 34 columns by index (`FraudModel.feature_columns`). Checked on the real data: val and test quality below reproduced exactly (test TP 2090, FP 5002), τ at 1 % val FPR recomputed equal, skew check (50 cards, 99 460 operations) 0 decision mismatches.

Clean quality at τ = 0.013072659261524677 (seed 42): test recall 0.9744, FPR 0.0090, precision 0.2947; val recall 0.9817, FPR 0.0098, precision 0.3601.
Audit gate (3 attacks random/es/stepwise × seeds 1–3, S spares, `realistic.yaml`, seed 42, Task 20): FAILED `ver_uplift`; worst one-sided upper bound per attack: random 62.9 %, es 67.3 %, stepwise 92.4 %. Calibration (Task 17b, 2 attacks): resolution max(upper − mean) 4.1 pp.
