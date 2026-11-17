# fraud-robustness-audit

`fraudaudit`: episode-level adversarial robustness audit of a fraud-detection MLP, run as a GitHub Actions gate (master's thesis tool).

## Data

Sparkov dataset (Kaggle `kartik2112/fraud-detection`), ~200 MB, git-ignored:

    scripts/download_data.sh      # downloads into data/, verifies SHA-256 (macOS and Linux)

## Build and test

    docker build -t fraudaudit:dev .
    docker run --rm --entrypoint pytest fraudaudit:dev -q
    docker run --rm fraudaudit:dev --help

## Jupyter (dev stage)

    docker build --target dev -t fraudaudit:jupyter .

The `dev` stage adds `jupyterlab==4.2.5`; the image entrypoint is `fraudaudit`, so start Jupyter with:

    docker run --rm -p 8888:8888 -v "$PWD":/app --entrypoint jupyter fraudaudit:jupyter lab --ip=0.0.0.0 --allow-root

The package is installed in editable mode (`pip install -e .`), so mounting the repo at `/app` picks up source changes.

## Pinned versions

All versions in `requirements.txt` are exact (`==`) and were resolved as specified (no deviations from the starting versions). `keras==3.15.1` (the version TensorFlow 2.18.0 resolved to, recorded in `models/model.json`) and `scipy==1.18.1` were added as explicit pins (`fraudaudit.metrics` imports scipy directly). Python 3.12; `jupyterlab==4.2.5` in the dev stage only.

## Data periods and fraud episodes

`fraudaudit.data`: `fit` = train before 2020-03-21, `val` = train from 2020-03-21, `test` = Sparkov test file. Fraud episodes are per-card fraud operations split at a gap > 7 days. Measured on the real data (1 852 394 rows, 9 651 fraud): `fit` 631 episodes, `val` 135, `test` 218 (one episode per card in every period; test median 10 ops, max 19).

## Windowed aggregates

`fraudaudit.features.aggregates.batch_aggregates`: 11 per-card features: 7 windowed aggregates (`cnt_1d/7d/30d`, `mean_amt_1d/7d/30d`, `secs_since_prev`) and 4 behavioural features (`cnt_1h`, `sum_1h`, `cnt_cat_24h`, `amt_ratio_30d`). Operation j counts for i iff same card, `t_j < t_i` (same-second operations do not see each other) and `t_j >= t_i − w` (an operation exactly w earlier is in the window). Empty window → 0 / 0.0; no earlier operation or a gap strictly greater than 30 days → `secs_since_prev = 2 592 000`.

Behavioural features (Task 23, pre-registered as one fixed set before any result): `cnt_1h` / `sum_1h` = number / sum of the amounts of the card's operations in `[t − 1 h, t)`; `cnt_cat_24h` = number of the card's operations of the same category as the current one in `[t − 24 h, t)`; `amt_ratio_30d = log1p(amt / max(mean_amt_30d, 1.0))`. Reason: the held-out stepwise attack beat the hardened models with fast, small, same-category bursts (~75 s apart, ~100 $), while the shortest window was one day (Carminati et al., 2020, use 1 h windows). The serving state (`CardState`) keeps every recent operation's category for `cnt_cat_24h`; both paths are checked against an O(n²) reference in the tests.

Measured on the real data (1 852 394 rows, 999 cards, Docker `fraudaudit:dev` on an Apple-silicon Mac): `batch_aggregates` 1.3 s for all 11 features (`load_raw` 11.3 s; 0.5 s for the 7 aggregates before Task 23).

## Baseline model (bachelor MLP, bachelor+hour-calendar+behav features)

Features (`fraudaudit.features.columns`, `meta.feature_set: "bachelor+hour-calendar+behav"`): the 27 base features of the bachelor model without the calendar features `trans_month` and `trans_year` (25; ruling R23: the test period Jun–Dec 2020 lies outside the training support of `trans_year` × `trans_month`, and adversarially retrained models extrapolated on it), the time of day `hour_sin = sin(2π·h/24)` and `hour_cos = cos(2π·h/24)` with h = hour + minute/60 + second/3600 of the operation time (not in the bachelor set: 84.6 % of Sparkov fraud happens at night, so the audit is run on a model that sees the hour), the 7 windowed aggregates and the 4 behavioural features – 38 features (27 base + 11). Batch (`BaseEncoder.transform`) and serving (`Serving`) share one implementation (`fraudaudit.features.base.hour_features`, float64, cast to float32 at the end).

`fraudaudit train --data data --out models --seed 42` (`fraudaudit.model.pipeline`). Architecture and hyperparameters as in the bachelor listings: `StandardScaler` → `IsolationForest()` flag (`predict(X) > -1`, the 39th input) → MLP Dense 55 relu, BN, Dense 26 sigmoid, BN, Dense 12 sigmoid, BN, Dense 1 sigmoid; `Adamax(0.01)`, `BinaryCrossentropy`, 20 epochs, batch 32. Corrections: features on the whole time-ordered stream; encoder, scaler and IsolationForest (`random_state=seed`) fit on `fit` rows only; fraud rows of `fit` resampled with replacement up to the number of legit `fit` rows (`val`/`test` unbalanced); τ = `threshold_at_fpr(y_val, p_val, 0.01)` on clean `val`, decision `p >= τ`.

Run of 2026-11-16, seed 42, Docker `fraudaudit:dev` on an Apple-silicon Mac (Linux aarch64, 6 CPUs for Docker), TensorFlow 2.18.0 on CPU (not guaranteed bit-identical on other platforms):

| | rows | fraud rows |
|---|---:|---:|
| `fit` | 1 070 966 | 6 250 |
| `fit` after balancing | 2 129 432 | 1 064 716 |
| `val` | 225 709 | 1 256 |
| `test` | 555 719 | 2 145 |

τ = 0.012350881472229958

| at τ | recall | FPR | precision | TP | FP |
|---|---:|---:|---:|---:|---:|
| `val` | 0.9817 | 0.0100 | 0.3547 | 1 233 | 2 243 |
| `test` | 0.9767 | 0.0091 | 0.2945 | 2 095 | 5 019 |

Test recall by month (Jun–Dec 2020) stays between 0.965 and 0.993 and test FPR between 0.77 % and 1.03 %, so the score is stable over the test period.

The previous `bachelor+hour-calendar` baseline without the behavioural features (34 features, τ = 0.013072659261524677: test recall 0.9744, FPR 0.0090, precision 0.2947) is kept in `models-no-behav/` (see its README). The `bachelor+hour` baseline with the calendar features (36 features, τ = 0.005090115591883659: test recall 0.9692, FPR 0.0083, precision 0.3103) is kept in `models-bachelor-hour/` (see its README). The older baseline without time of day (34 features incl. the calendar features, τ = 0.047462500631809235: test recall 0.9524, FPR 0.0086, precision 0.3014; val recall 0.9777, FPR 0.0098) is kept in `models-bachelor/` for comparison; the current code does not load it (see `models-bachelor/README.md`).

Every run records `git_sha` and `git_dirty` in `meta`: from the env vars `GIT_SHA` / `GIT_DIRTY` if set, else from `git`, else `"unknown"`. The image has no git, so pass them in Docker:

    docker run --rm -v "$PWD":/app -w /app -e GIT_SHA="$(git rev-parse HEAD)" \
      -e GIT_DIRTY="$(test -z "$(git status --porcelain)" && echo 0 || echo 1)" \
      --entrypoint fraudaudit fraudaudit:dev train --data data --out models --seed 42

The committed baseline was trained from the clean tree at `41dea63` (`meta.git_sha`, `git_dirty: false`).

Wall time: 26 min 13 s in total (feature preparation 13 s, `mlp.fit` 25 min 54 s, ~78 s per epoch of 66 545 steps; `load_raw` not included). The full numbers are in `models/model.json` (`meta`). `FraudModel.predict_proba` on one row: ~1.1 ms.

## Audit gate

`fraudaudit audit --data data --model models --threat threat-models/realistic.yaml --gate gate.yaml --out out/` runs every attack of `gate.yaml` (`attacks: [random, es, stepwise]`) with every attack seed (`attack_seeds: [1, 2, 3]`) on the same targets and spares (`--seed`, default 42), one run after another, each with its own process pool. Targets and spares (`spares_period`, ruling R14): with `fit` the targets are all 218 fraud episodes of the `test` period and the attacker reconnoitres on the 315 fraud episodes of the S half of the `fit` period (a card can be in both, at another time and with another state). The fit-period episodes are split by card with the fixed seed `FIT_SPLIT_SEED = 42` (`fraudaudit.data.fit_split`, ruling R20; independent of `--seed`) into H (316 cards, reserved for the adversarial generation of `fraudaudit harden`) and S (315 cards, the gate's spares), so the gate never reconnoitres on cards whose adversarial episodes trained a hardened model; seed and sizes are in every run's `meta.fit_split`; `test` keeps the old 50/50 card split of the test episodes for comparison. Checks (`fraudaudit.gate.decide`): `ver_uplift` (worst run: upper bound of the paired bootstrap of VER_adv − VER_clean ≤ `delta`; with `one_sided: true` the bound is one-sided at `confidence`, i.e. the 95th percentile, because the gate only tests uplift ≤ δ; VER intervals in the report stay two-sided; `delta: null` fails closed), `recall_clean` (clean test recall at τ ≥ `recall_clean_min`), `feasibility` (every run's `feasible_share` = 1) and, with `--skew-report out/skew.json` (written by `fraudaudit skew --out out/skew.json`), `skew`. Writes `out/audit.json` (every run, checks, verdict) and the English `out/summary.md` (appended to `$GITHUB_STEP_SUMMARY` when set); exit code 1 if a check fails, 2 on a configuration error.

`gate-stepwise-456.yaml` is `gate.yaml` with another ensemble only (`attacks: [stepwise]`, `attack_seeds: [4, 5, 6]`): fresh seeds of the held-out stepwise attack for a final check (`--gate gate-stepwise-456.yaml --no-baseline`, same targets, spares and `--seed`). `gate.yaml` stays the gate of record; a test pins that the two files differ only in these two keys.

### Held-out stepwise attack

`stepwise` (`fraudaudit.attacks.stepwise`, `fraudaudit.policy`) is a closed-loop attacker: a parametric policy that chooses every next operation after seeing the bank's decision on the previous one (declined yes/no only, `knowledge: decision`). Its parameters are relative, so a policy found on spare cards transfers to target cards: the first amount `s0` ∈ [0.02, 1] as a share of `amount_max`, multipliers `g` ∈ [1, 3] after a passed and `b` ∈ [0.1, 0.9] after a declined operation, a daily band of active hours `hour` ± `hour_jitter` (`hour_jitter` ∈ [0, 12]), the delay `gap_h` ∈ [`min_gap_secs`/3600, 48] hours, a preferred merchant category `cat_pref` with an optional category switch after a decline, and the stop share `u` ∈ [ρ, 1] of A. It also stops after `k_max` operations, at the end of the window or when the card is blocked. The bank side (`max_declines`, `tau_after_decline`, cent-exact success) is shared with the fixed-plan runner (`episode.run_adaptive_episode` / `run_episode`). The policy vector is tuned by random search on the same reconnaissance oracle and budget; an evaluation reserves the worst case `cards_per_eval × k_max` operations and is charged the executed ones. The attack never generates hardening data (`fraudaudit harden` refuses it; it only measures in the val mini-audit), so on the test period it checks whether robustness gained against ES and random search transfers to an attacker with a different search principle; the summary marks it as held-out.

Under the realistic threat model (`max_declines: 1`) the first decline blocks the card, so the policy never gets to react to a decline: it is effectively open-loop. Its strength comes from its low-dimensional structured search space (geometric amounts, a fixed cadence, one merchant category: 9 parameters against the 82 of a fixed `Strategy` with `k_max` 20), not from adaptation to declines; `b` and `switch_on_decline` only matter with `max_declines` > 1.

### Calibration of the gate's resolution

`fraudaudit calibrate --data data --model models --threat threat-models/realistic.yaml --gate gate.yaml --out out/calibration/ [--deltas 0.02,0.05,0.10,0.15,0.20]` runs the gate's ensemble on the baseline model and writes `calibration.json` and `calibration.md`: per attack × seed the uplift mean, its upper bound and bootstrap SE; the sampling resolution (max over runs of upper − mean, the smallest δ the gate can certify for a model with zero true uplift) and the spread of the means over attack seeds; the verdict of the baseline model for each δ. It only reads `gate.yaml`; the author fixes δ from these numbers before hardening.

### Conventional-audit baseline (informational)

After the ensemble, `audit` also runs the single-transaction baseline (`fraudaudit.audit.run_single_tx_baseline`, threat model `--single-threat threat-models/single-transaction.yaml`; `--no-baseline` skips it). It never changes the verdict or the exit code; if it fails, `audit.json` gets `baseline: {"error": …}` and the summary gets a note. It uses the same target episodes as the episode audit.

Every real fraud operation is attacked on its own with a random search of at most `single_tx_budget: 50` decision-only queries over three things:
- the amount (1.00–1376.04 $);
- a time shift in [0, 7 days) from the original time, wrapped into the episode window by whole days, so a candidate can land before the original time (counted in `meta`);
- the merchant.

The card state at the candidate time is the recorded stream (ruling R15): the history before the episode plus every other real operation of the episode before that time. Aggregates are not recomputed with the attacker's other changes, and there is no bank reaction. The conventional audit queries the target card's own state; the episode attacker only probes spare cards.

`amount_rule` (`first` in the YAML) chooses which candidate is kept:
- `first`: the first non-declined candidate;
- `max`: all 50 queries are made and the non-declined candidate with the largest amount is kept.

The gate runs both rules, the YAML's rule first (primary). If no candidate evades, the original operation is kept and counts as not evaded.

Results:
- `evaded_tx_share`;
- `evaded_amount_share` (ORIGINAL amounts of the evaded operations / sum of A);
- `chosen_amount_ratio` (amounts of the chosen candidates / sum of A, no cap);
- the replay: the chosen operations of each episode run as one episode through `run_episode` (recomputed aggregates, `min_gap_secs`, `max_declines`, ρ of the episode threat model), with VER and its paired uplift over VER_clean.

The summary section "Conventional audit (single-transaction attack) vs episode audit" compares these with the worst episode run, the mean and range of the uplift over all attack × seed runs, and the paired bootstrap of VER_adv − VER_replay. In `audit.json` they are under `baseline` (`primary`, `rules`, `comparison`).

## Spending corridor with a guaranteed loss bound (`fraudaudit.corridor`)

A bank policy on top of the model (soft corridor, Task 22). The `corridor:` section of the threat model
(`threat-models/realistic-corridor.yaml`; `corridor: null` in `realistic.yaml` = off, the previous behaviour) turns it
on for every path that decides operations through the shared bank logic (`episode._Bank`): the audit's adversarial and
clean legs, reconnaissance, the stepwise policy, the single-transaction baseline replay and the legitimate replays.
Decision order per operation: the model declines → decline (`max_declines`, `tau_after_decline` as before); the model
approves and the card's AUTO-APPROVED amount in the rolling windows `(t − 24 h, t]` and `(t − 7 d, t]` plus this
operation stays within the corridor of both windows → auto-approve; otherwise → owner confirmation in the bank app.
The owner confirms (friction; the operation passes but is not auto-approved); the fraudster has the card details but
not the owner's second factor (threat-model assumption), so the operation does not pass and, with
`confirm_fail_counts_as_decline: true`, counts as a decline.

Corridor of card c on day d: `C_W = max(floor_W, Q_W) · m_W(d)`, where `Q_W` is the `q`-quantile of the card's own
rolling-W spend taken after each of its legitimate operations in `[d − lookback_days, d)` (attacker operations never
enter the profile) and `m_W` are population multipliers estimated on the fit period – weeks `holiday_weeks` around the
nearest of Christmas, Thanksgiving and Independence Day (both windows), weekday and day of month (24 h window only);
each factor is used only when its bootstrap interval excludes 1. Amounts are compared in whole cents.

Guarantee (Proposition): an episode of at most 7 days cannot get more auto-approved than
`max_{t ∈ [t0, t0 + D)} C_7d(c, t)`, whatever the attack. Every corridor run reports this bound per target
(`per_target[].corridor_bound`, share of A, and the tighter `corridor_headroom` net of the owner's own spend still in
the window), its summary in `meta.corridor` and a bound check (`n_bound_violations`, always 0 by construction).

`fraudaudit corridor-sweep --model models --threat threat-models/realistic-corridor.yaml --gate gate.yaml --out
out/corridor-sweep` evaluates the pre-registered grid q ∈ {0.95, 0.98, 0.99, 0.995, 0.999} × τ at target FPR
∈ {0.003, 0.01} (τ from clean val, same weights; plus a no-corridor reference per τ): the guaranteed bound, the gate
ensemble's worst uplift under the corridor and the confirmation burden on the legitimate test stream (share of
operations and of card-days with a confirmation; `metrics.confirmation_burden`, scores cached once per model). It
writes `corridor-sweep.json`, `.md` (table, Pareto flags) and `.png` (front).

## Hardening (`fraudaudit harden`, local)

`fraudaudit harden --data data --model models --threat threat-models/realistic.yaml --gate gate.yaml --out models-harden [--rounds 5] [--n-jobs N]` retrains the model with adversarial episodes in up to 5 rounds (`fraudaudit.harden`, module docstring has every rule). Data isolation: generation uses only the H half of the fit-period episodes (R20), split by card again into reconnaissance cards and host cards; the gate's S spares, the val and the test periods are never generated on or inserted into. Each round: 64 generation units (attack ES and random search × seeds 1–16 × `max_declines` 1 and 3, budget `--budget-ops 20000` each, in a process pool) attack the current model; each unit's best strategy is executed on its host slots (the host card's real episode start ± multiples of 37 days, away from real fraud, ending ≥ 30 days before val), and every executed operation up to blocking becomes a fraud row (`trans_num` `adv-r{round}-…`). adversarial rows = `--share-among-fraud 0.5` of all fit fraud rows before balancing (design §11: as many as the 6250 real ones), split into equal row quotas per generation round so far (ruling R22; disjoint slots, whole episodes) are inserted into the original stream, features are recomputed by the batch path and the whole pipeline is retrained with the `train` procedure and seed (τ at 1 % FPR on clean val). A mini-audit (the gate's ensemble random, es and stepwise × attack seeds 1–3, gate budget; targets = the 135 val episodes, spares = H) measures the val uplift; hardening stops when the worst one-sided upper bound over the 9 runs ≤ δ, and the final model is the last round reached (no pick among rounds). Output: `round-{r}/` model directories (with `adversarial-rows.csv`) and `harden-report.json`/`harden-report.md`. ES and random search generate hardening data (`--gen-attacks es,random`, the default; `stepwise` is refused). The stepwise attack only measures: it takes part in the val mini-audit (and so in the early-stop decision) but never produces a training row; in the final gate on the test period it is the held-out check.
