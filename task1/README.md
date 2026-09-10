# OdiaEval Group 7 — hate-speech detection (Task 2)

Fine-tuning, evaluation and error analysis for the hate-speech track. Task 1
(the machine-translated Odia training data) was built separately and is consumed
here as a frozen input.

## What this does

1. Fine-tunes `ai4bharat/IndicBERTv2-MLM-only` on the Task 1 release, once per
   balancing variant, selecting the checkpoint on validation macro-F1.
2. Evaluates each saved model on the held-out benchmark test split.
3. Evaluates the same saved model, unchanged, on the professor's native Odia
   gold set.
4. Computes the translationese gap as benchmark macro-F1 minus native macro-F1.
5. Runs the error analysis, centred on the question-versus-statement cut the
   Task 1 data card asks for.
6. Writes `results/final_report.md` in the assignment's result template.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Where this sits on the server

```
~/DTSC422/
  odiaeval-hate-dataset-v1/   the frozen Task 1 release, read-only for us
  odiaeval-hate-code/         the Task 1 build pipeline, not modified here
  odiaeval-hate-task2/        this repo
```

The dataset is found automatically from that layout. If you move it, either set
`data.root` in `config.yaml`, pass `--data-root`, or export `ODIAEVAL_DATA_ROOT`.
The loader tries each in turn and names every path it tried before failing.

## Running it

Prove the pipeline works before spending GPU time:

```bash
bash run_all.sh --smoke
```

The real run, once the native gold set is in hand:

```bash
bash run_all.sh --native-file data/native/<gold file>
```

Individual stages:

```bash
python -m src.token_stats
python -m src.train --variant proportional
python -m src.evaluate --model-dir models/proportional --benchmark-test
python -m src.evaluate --model-dir models/proportional --native-file data/native/gold.csv
python -m src.error_analysis --predictions results/proportional/predictions_native.parquet
python -m src.baselines --variant proportional
python -m src.report
```

The native loader auto-detects the text and label columns and accepts csv, tsv,
parquet, jsonl or xlsx. Override with `--text-col`, `--label-col` and
`--label-map '{"HOF": 1, "NOT": 0}'` when the auto-detection guesses wrong.

## Rules the code enforces

- **`text_eng` never reaches the model.** It is the English source the weak
  labels came from, so it is a direct label leak. It is dropped before
  tokenisation and asserted against inside `tokenize`.
- **The native gold set is evaluation-only.** Nothing in the training path can
  load it, and no threshold is fitted on it or anywhere else. Decisions are
  argmax throughout.
- **Split leakage is re-verified, not assumed.** `assert_no_split_leakage` runs
  at the start of every training run and refuses to proceed on any overlap.
- **Every run records what produced it.** `models/<variant>/run_config.json`
  carries the resolved Hub revision, seed, hyperparameters, row counts and
  environment.

## Both variants, and why

`proportional` preserves the natural source mix and is the reproduction target.
`match_minority` forces NON_HATE to the same source profile as HATE, which drives
source-label mutual information to zero. The two corpora are identical except for
which NON_HATE rows were kept, so the difference between their scores isolates how
much of the number comes from a source shortcut rather than Odia ability. That
difference is a reportable result, not a robustness check.

## Floors

`src/baselines.py` reports what a model that cannot read achieves on the same
data. On the benchmark test split, surface features alone reach about 63 macro-F1
and the bare rule *no question mark means HATE* reaches about the same. Any
fine-tuned score has to be quoted against those numbers, not against zero.

## Layout

```
config.yaml              experiment configuration, copied into every run record
src/common.py            seeding, device, metrics, run metadata
src/data.py              benchmark and native loaders, preprocessing, leak guards
src/train.py             fine-tuning, checkpoint selection on validation macro-F1
src/evaluate.py          one evaluation path for both test sets
src/error_analysis.py    question-versus-statement analysis and the advance prediction
src/baselines.py         surface-feature, question-mark and trivial floors
src/report.py            assembles results/final_report.md
src/token_stats.py       tokenised length profile behind the max_length choice
run_all.sh               end-to-end driver
```
