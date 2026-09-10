#!/usr/bin/env bash
# End-to-end run: both variants, both evaluations, error analysis, final report.
#
#   bash run_all.sh                                   # benchmark only
#   bash run_all.sh --native-file data/native/gold.csv # full pipeline
#
# Add --smoke to run a few hundred rows for one epoch, to prove the pipeline
# works before committing the GPU to the real thing.

set -euo pipefail

NATIVE_FILE=""
SMOKE=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --native-file) NATIVE_FILE="$2"; shift 2 ;;
    --smoke) SMOKE="--max-train-samples 512 --max-eval-samples 256 --epochs 1"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

VARIANTS=(proportional match_minority)

echo "== token length profile =="
python -m src.token_stats ${NATIVE_FILE:+--native-file "$NATIVE_FILE"} || true

for V in "${VARIANTS[@]}"; do
  echo
  echo "== training: $V =="
  # shellcheck disable=SC2086
  python -m src.train --variant "$V" --overwrite $SMOKE ${EXTRA[@]+"${EXTRA[@]}"}

  echo
  echo "== benchmark held-out test: $V =="
  python -m src.evaluate --model-dir "models/$V" --benchmark-test

  if [[ -n "$NATIVE_FILE" ]]; then
    echo
    echo "== native Odia gold set: $V =="
    python -m src.evaluate --model-dir "models/$V" --native-file "$NATIVE_FILE"
    python -m src.error_analysis --predictions "results/$V/predictions_native.parquet"
  fi

  python -m src.error_analysis --predictions "results/$V/predictions_benchmark_test.parquet"

  echo
  echo "== baselines: $V =="
  python -m src.baselines --variant "$V" ${NATIVE_FILE:+--native-file "$NATIVE_FILE"}
done

echo
echo "== final report =="
python -m src.report
echo "results/final_report.md is ready"
