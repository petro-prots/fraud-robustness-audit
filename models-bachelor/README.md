# models-bachelor – baseline without time of day (kept for comparison)

The baseline model trained with the 34-feature **bachelor** feature set (27 base features of the bachelor model + 7 windowed aggregates; MLP input 35), seed 42. Pipeline and artifacts from commit `f3adaea` (`feat(model): bachelor MLP pipeline, baseline artifacts`), provenance fields added in `6a6151f`; see `meta` in `model.json` (`git_sha`, `provenance_note`).

It was replaced in `models/` by the `bachelor+hour` model (29 base features incl. `hour_sin`/`hour_cos`, 36 features, MLP input 37). It is kept only as the comparison row "model without time of day".

**Not loadable by the current code**: `model.json` `feature_columns` (34) differ from `FEATURE_COLUMNS` (36), so `load_artifacts` refuses it. To use it, check out a commit before `1a51b7b` (e.g. `6727d6a`).

Clean quality at τ = 0.047462500631809235 (seed 42): test recall 0.9524, FPR 0.0086, precision 0.3014; val recall 0.9777, FPR 0.0098, precision 0.3579.
Audit gate (`out/` of `e93dc6e`, `realistic.yaml`, seed 42): FAILED `ver_uplift`, worst run `es/1` – uplift 67.7 % [61.8 %; 73.1 %]; single-transaction baseline replay uplift (`first`) 30.6 % [23.7 %; 39.5 %].
