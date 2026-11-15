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

All versions in `requirements.txt` are exact (`==`) and were resolved as specified (no deviations from the starting versions). Python 3.12; `jupyterlab==4.2.5` in the dev stage only.

## Data periods and fraud episodes

`fraudaudit.data`: `fit` = train before 2020-03-21, `val` = train from 2020-03-21, `test` = Sparkov test file. Fraud episodes are per-card fraud operations split at a gap > 7 days. Measured on the real data (1 852 394 rows, 9 651 fraud): `fit` 631 episodes, `val` 135, `test` 218 (one episode per card in every period; test median 10 ops, max 19).
