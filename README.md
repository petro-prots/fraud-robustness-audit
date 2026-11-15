# fraud-robustness-audit

`fraudaudit`: episode-level adversarial robustness audit of a fraud-detection MLP, run as a GitHub Actions gate (master's thesis tool).
Інструмент аудиту стійкості MLP-моделі виявлення шахрайства до епізодних атак ухилення; працює як шлюз у GitHub Actions.

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

`fraudaudit.features.aggregates.batch_aggregates`: 7 per-card features (`cnt_1d/7d/30d`, `mean_amt_1d/7d/30d`, `secs_since_prev`). Operation j counts for i iff same card, `t_j < t_i` (same-second operations do not see each other) and `t_j >= t_i − w` (an operation exactly w earlier is in the window). Empty window → 0 / 0.0; no earlier operation or a gap strictly greater than 30 days → `secs_since_prev = 2 592 000`. Measured on the real data (1 852 394 rows, 999 cards, Docker `fraudaudit:dev` on an Apple-silicon Mac): `batch_aggregates` 0.5 s (`load_raw` 8.4 s).

## Baseline model (bachelor MLP)

`fraudaudit train --data data --out models --seed 42` (`fraudaudit.model.pipeline`). Architecture and hyperparameters as in the bachelor listings: `StandardScaler` → `IsolationForest()` flag (`predict(X) > -1`, the 35th input) → MLP Dense 55 relu, BN, Dense 26 sigmoid, BN, Dense 12 sigmoid, BN, Dense 1 sigmoid; `Adamax(0.01)`, `BinaryCrossentropy`, 20 epochs, batch 32. Corrections: features on the whole time-ordered stream; encoder, scaler and IsolationForest (`random_state=seed`) fit on `fit` rows only; fraud rows of `fit` resampled with replacement up to the number of legit `fit` rows (`val`/`test` unbalanced); τ = `threshold_at_fpr(y_val, p_val, 0.01)` on clean `val`, decision `p >= τ`.

Run of 2026-11-15, seed 42, Docker `fraudaudit:dev` on an Apple-silicon Mac (Linux aarch64, 6 CPUs for Docker), TensorFlow 2.18.0 on CPU (not guaranteed bit-identical on other platforms):

| | rows | fraud rows |
|---|---:|---:|
| `fit` | 1 070 966 | 6 250 |
| `fit` after balancing | 2 129 432 | 1 064 716 |
| `val` | 225 709 | 1 256 |
| `test` | 555 719 | 2 145 |

τ = 0.047462500631809235

| at τ | recall | FPR | precision | TP | FP |
|---|---:|---:|---:|---:|---:|
| `val` | 0.9777 | 0.0098 | 0.3579 | 1 228 | 2 203 |
| `test` | 0.9524 | 0.0086 | 0.3014 | 2 043 | 4 735 |

Every run records `git_sha` and `git_dirty` in `meta`: from the env vars `GIT_SHA` / `GIT_DIRTY` if set, else from `git`, else `"unknown"`. The image has no git, so pass them in Docker:

    docker run --rm -v "$PWD":/app -w /app -e GIT_SHA="$(git rev-parse HEAD)" \
      -e GIT_DIRTY="$(test -z "$(git status --porcelain)" && echo 0 || echo 1)" \
      --entrypoint fraudaudit fraudaudit:dev train --data data --out models --seed 42

The committed baseline was trained from uncommitted code on top of `c1625d0`; that code was committed with the artifacts in `f3adaea` (`meta.provenance_note`; the provenance fields were added afterwards by re-saving the same artifacts, predictions unchanged).

Wall time: 19 min 31 s in total (feature preparation 9 s, `mlp.fit` 19 min 19 s, ~57 s per epoch of 66 545 steps; `load_raw` not included). The full numbers are in `models/model.json` (`meta`). `FraudModel.predict_proba` on one row: ~1.1 ms.

## Audit gate

`fraudaudit audit --data data --model models --threat threat-models/realistic.yaml --gate gate.yaml --out out/` runs every attack of `gate.yaml` (`attacks: [random, es]`) with every attack seed (`attack_seeds: [1, 2, 3]`) on the same target/spare card split (`--seed`, default 42), one run after another, each with its own process pool. Checks (`fraudaudit.gate.decide`): `ver_uplift` (worst run: upper bound of the paired bootstrap interval of VER_adv − VER_clean ≤ `delta`; `delta: null` fails closed), `recall_clean` (clean test recall at τ ≥ `recall_clean_min`), `feasibility` (every run's `feasible_share` = 1) and, with `--skew-report out/skew.json` (written by `fraudaudit skew --out out/skew.json`), `skew`. Writes `out/audit.json` (every run, checks, verdict) and the Ukrainian `out/summary.md` (appended to `$GITHUB_STEP_SUMMARY` when set); exit code 1 if a check fails, 2 on a configuration error.

### Conventional-audit baseline (informational)

After the ensemble, `audit` also runs the single-transaction baseline (`fraudaudit.audit.run_single_tx_baseline`, threat model `--single-threat threat-models/single-transaction.yaml`; `--no-baseline` skips it). It never changes the verdict. It uses the same target episodes as the episode audit. Every real fraud operation is attacked on its own: a random search of `single_tx_budget: 50` decision-only queries over the amount (1.00–1376.04 $), a time shift in [0, 7 days) from the original time (whole-day wrapping into the episode window) and the merchant. The card state is the original stream: the history before the episode plus the real preceding fraud operations. Aggregates are not recomputed and there is no bank reaction. The first non-declined candidate is kept; if there is none, the original operation is kept and counts as not evaded. Results: `evaded_tx_share`, `evaded_amount_share` (amounts of the chosen operations / sum of A) and the replay. The replay runs the chosen operations of each episode as one episode through `run_episode` (recomputed aggregates, `min_gap_secs`, `max_declines`, ρ of the episode threat model) and reports VER with its paired uplift over VER_clean. The summary section «Звичайний аудит (атака на одну транзакцію) проти епізодного» compares these numbers with the worst episode run, in `audit.json` under `baseline`.
