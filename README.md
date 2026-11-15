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

All versions in `requirements.txt` are exact (`==`) and were resolved as specified (no deviations from the starting versions). `scipy==1.18.1` was added as an explicit pin (the version pip resolved with scikit-learn), because `fraudaudit.metrics` imports it directly. Python 3.12; `jupyterlab==4.2.5` in the dev stage only.

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

Wall time: 19 min 31 s in total (feature preparation 9 s, `mlp.fit` 19 min 19 s, ~57 s per epoch of 66 545 steps; `load_raw` not included). The full numbers are in `models/model.json` (`meta`). `FraudModel.predict_proba` on one row: ~1.1 ms.
