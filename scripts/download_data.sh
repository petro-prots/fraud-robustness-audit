#!/usr/bin/env bash
# Downloads the Sparkov fraud dataset (Kaggle, anonymous endpoint) into data/ and verifies SHA-256.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

TRAIN_SHA256="fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d"
TEST_SHA256="12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0"

if command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  sha256() { sha256sum "$1" | awk '{print $1}'; }
fi

if [ ! -f data/fraudTrain.csv ] || [ ! -f data/fraudTest.csv ]; then
  curl -L --fail https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection -o data/sparkov.zip
  unzip -o data/sparkov.zip fraudTrain.csv fraudTest.csv -d data
fi

if [ "$TRAIN_SHA256" = "TBD" ]; then
  echo "fraudTrain.csv $(sha256 data/fraudTrain.csv)"
  echo "fraudTest.csv  $(sha256 data/fraudTest.csv)"
  exit 0
fi

status=0
for pair in "fraudTrain.csv:$TRAIN_SHA256" "fraudTest.csv:$TEST_SHA256"; do
  f="${pair%%:*}"; want="${pair##*:}"; got="$(sha256 "data/$f")"
  if [ "$got" = "$want" ]; then echo "$f: OK"; else echo "$f: FAILED (got $got)"; status=1; fi
done
exit $status
