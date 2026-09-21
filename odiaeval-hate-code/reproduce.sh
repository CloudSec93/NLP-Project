#!/usr/bin/env bash
# =============================================================================
# OdiaEval hate-speech dataset (Task 1) — one-command reproduction
#
#   bash reproduce.sh
#
# From an empty clone to two frozen releases under data/processed/ plus the
# data card. Target: fresh Google Colab T4, Python 3.10+. Also runs on CPU.
#
# Approximate runtime (cold; a warm rerun skips acquisition and re-uses the
# score shards):
#
#   0  manifest + pinned-SHA check        ~10 s   network
#   1  acquire IndicAlign, column-pruned  ~13 min network, ~71 MB to data/raw/
#   2  extract turn-0 prompt pairs        ~1 min
#   3  teacher scoring (canary first)     ~60 min CPU / much faster on a T4
#   4  threshold + confounding analysis   ~30 s
#   5  exact + near-duplicate dedup       ~8 min
#   6  balance + split, both variants     ~1 min
#   7a surface-feature diagnostic         ~10 s
#   7  freeze releases + data card        ~1 min
#
# Every stage reads from data/ and writes to data/, appends its counts to
# artifacts/stage_stats.json, and never mutates its input. Stages 1 and 3
# checkpoint to disk and resume, so a dropped Colab session costs minutes
# rather than the whole run.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

echo "==> [0/7] environment + pinned-SHA manifest"
python -m pip install -q -r requirements.txt
python -m src.resolve_manifest

echo "==> [1/7] acquire IndicAlign (column-pruned, cached to data/raw/)"
python -m src.acquire

echo "==> [2/7] extract turn-0 Odia/English prompt pairs"
python -m src.extract

echo "==> [3/7] teacher scoring — canary and HateCheck first, then the corpus"
python -m src.label --hatecheck

echo "==> [4/7] threshold (0.90 / 0.05) + source-confounding analysis"
python -m src.filter

echo "==> [5/7] exact + near-duplicate dedup (with threshold sensitivity sweep)"
python -m src.dedup --sweep 20000

echo "==> [6/7] balance + deterministic split — both variants"
python -m src.balance_split --all-modes

echo "==> [7a] surface-feature diagnostic (hypothesis-only baseline)"
python -m src.diagnostics

echo "==> [7/7] freeze releases + write the data card"
python -m src.report

echo
echo "==> tests"
python -m pytest -q

cat <<'DONE'

============================================================================
Done. Frozen releases:

  data/processed/odia_hate_v1_proportional/{train,validation,test}.{parquet,jsonl}
  data/processed/odia_hate_v1_match_minority/{train,validation,test}.{parquet,jsonl}

  artifacts/data_card.md            read this first
  artifacts/manifest.json           pinned SHAs + per-file hashes
  artifacts/stage_stats.json        every row kept or dropped with a reason
  artifacts/label_distribution.json histograms, contingencies, baselines

WARNING: text_eng ships in the release for error analysis only.
         It must never be given to the student model.

Optional, needs a human:
  python -m src.agreement sample    then fill artifacts/agreement_sample.csv
  python -m src.agreement score     -> Cohen's kappa
============================================================================
DONE
