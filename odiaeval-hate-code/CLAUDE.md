# CLAUDE.md — OdiaEval / Group 6 & 7 / Hate-Speech Detection

Standing constraints for this repository. Read before every task. Do not violate
anything here without the user explicitly telling you to.

---

## 1. What this project is

**OdiaEval** is a course project measuring the *translationese gap* in Odia NLP.

The claim being tested: every existing Odia benchmark was machine-translated from
English, so published scores may measure a model's ability to handle its own
translation artefacts rather than actual Odia ability. Translated Odia is flat and
literal — English word order, dictionary equivalents, no idiom. Native Odia is not.
A model trained on the translated version has seen the flat form thousands of times
and the real form never.

The protocol every group follows:

| Step | What |
|---|---|
| 1 | Reproduce a published benchmark result on machine-translated data, within 1–2 points |
| 2 | Freeze that exact model — no retraining, no re-tuning |
| 3 | Evaluate it unchanged on a **native Odia gold test set supplied by the professor** |
| 4 | `translationese gap = benchmark score − native Odia score` |

Never average, merge, or substitute one score for the other. They are reported
separately, always.

## 2. This group's assignment (Groups 6 and 7, hate-speech detection)

Domain listed for this group: **News / YouTube Comments** — that describes the register
of the native Odia gold test set, not the training data built here.

Paraphrased closely from the assignment document (`docs/OdiaEval_updated.docx`, the
Group 6 / Group 7 section) — go to the original for the exact wording:

- Use the Odia-script user prompt (`ory_Orya`) from the **Toxic_Matrix**, **HHRLHF_T**,
  and **Dolly_T** configurations of IndicAlign; **do not train on assistant responses**.
- Generate weak labels from each aligned English prompt (`eng_Latn`) with the pinned
  RoBERTa teacher: **HATE at p ≥ 0.90**, **NON_HATE at p ≤ 0.05**; **discard uncertain**
  examples in between.
- Deduplicate the Odia prompts, balance HATE and NON_HATE, and create **deterministic
  stratified train / validation / test splits with seed 42**.
- Fine-tune IndicBERTv2 on the training split only, select the checkpoint on validation
  macro-F1, keep test labels out of training and model selection.
- Evaluate the same fine-tuned model on the native Odia test set. Do not use that set
  for training, threshold selection, or model selection.

Reported metrics: accuracy, macro-F1, and HATE precision / recall / F1 — on the
held-out machine-translated split **and** on the native Odia set. Gap is computed on
macro-F1.

Reproduction target: **90.94 macro-F1** on the machine-translated held-out split.

## 3. SCOPE BOUNDARY — read this twice

**This repository covers Task 1 only: dataset construction and the labelling pipeline.**

In scope:
: acquiring IndicAlign · extracting aligned Odia/English prompt pairs · running the
  teacher · thresholding · quality filtering · deduplication · balancing · splitting ·
  freezing the artefacts · the data card · the handoff schema.

Out of scope — **do not build these and do not stub them**:
: fine-tuning IndicBERTv2 · any training loop · `Trainer` / `TrainingArguments` /
  optimiser / LR-schedule code · checkpoint selection · any evaluation harness ·
  benchmark evaluation · native Odia evaluation · computing the translationese gap.

`torch` and `transformers` *are* needed here — the teacher runs inference in Stage 3.
The line is inference vs. training, not which library gets imported.

The deliverable is a **frozen, documented dataset** plus the reproducible code that
produced it. Another group member picks it up from the schema in §7 of `PROMPT.md`.

If a request would cross this boundary, say so and stop rather than building it.

## 4. Pinned artefacts — never substitute

| Role | Identifier | Notes |
|---|---|---|
| Dataset | `ai4bharat/indic-align` | configs `Toxic_Matrix`, `HHRLHF_T`, `Dolly_T` only |
| Teacher (labeller) | `facebook/roberta-hate-speech-dynabench-r4-target` | `id2label = {0: "nothate", 1: "hate"}`, `max_position_embeddings = 514` |
| Student (later, not here) | `ai4bharat/IndicBERTv2-MLM-only` | 278M params, BERT-style, MLM on IndicCorp v2 |

Record the resolved **commit SHA** of every dataset config and every model into
`manifest.json`. "Latest" is not a version. If a pinned SHA cannot be resolved, stop
and report — do not fall back to `main`.

## 5. Hard rules

1. **Seed 42 everywhere.** `random`, `numpy`, `torch`, and any hashing salt. Two runs
   of the same commit must produce byte-identical output files.
2. **Determinism over convenience.** Split assignment must be a pure function of a
   stable per-example id — not of row order, not of shuffle state, not of dict
   iteration order. Adding a row must not move existing rows between splits.
3. **No test-set contamination.** Nothing derived from the test split may influence
   filtering thresholds, balancing, or any later modelling decision.
4. **Deduplicate before splitting.** Near-duplicates that straddle a split boundary
   are leakage. Order is: extract → label → filter → dedup → balance → split.
5. **Prompts only.** Each language column is a list of turns, each turn a `[prompt,
   response]` pair — so `row["ory_Orya"][0][0]` is the turn-0 user prompt and
   `row["ory_Orya"][0][1]` is the assistant response. Index `[1]` never enters the
   dataset, in any column, in any language.
6. **Every threshold and proportion lives in `config.yaml`.** No magic numbers in code.
   The two teacher thresholds (0.90 / 0.05) are fixed by the assignment — they may be
   surfaced in config but must default to those values and any change must be loud.
7. **Every stage writes a manifest** — input row count, output row count, rows dropped,
   and *why*. A row that disappears without a recorded reason is a bug.
8. **Failures are loud.** No silent `except: pass`, no silent coercion of empty strings,
   no quietly skipping malformed rows. Count them, log them, report them.
9. **Report what you find, not what is convenient.** A low HATE yield or an ugly class
   imbalance is a result. Do not tune thresholds to make a number look better; if you
   believe a threshold is wrong, write it up in the data card and leave the default.

## 6. Environment

Google Colab, free tier, single **T4** GPU (~15 GB VRAM, ~12.7 GB system RAM, ~107 GB disk).

- Sessions die. Every expensive stage checkpoints to disk and resumes from it.
- RAM is the binding constraint, not VRAM. IndicAlign has 28 language columns; read
  only the ones needed (`doc_id`, `num_turns`, `eng_Latn`, `ory_Orya`).
- Never load the full `ai4bharat/indic-align` dataset. The `IndoWordNet` config alone is
  ~96.8M rows and will kill the runtime. Load named configs only.
- Teacher inference runs fp16 on the T4, batched, with a progress bar and periodic flush.
- Code must run identically as `python -m src.stage_name` from a terminal. Notebooks are
  thin callers, never where logic lives.

## 7. Repository conventions

```
odiaeval-hate-task1/
├── CLAUDE.md
├── README.md                  # one-command reproduction
├── reproduce.sh               # the one command
├── config.yaml                # every tunable
├── requirements.txt           # pinned ==versions
├── docs/                      # the professor's .docx and .pptx — source of truth
├── notebooks/
│   └── run_pipeline.ipynb     # Colab driver, thin
├── src/
│   ├── acquire.py             # 01 download + column-pruned load
│   ├── extract.py             # 02 aligned prompt pairs
│   ├── label.py               # 03 teacher inference
│   ├── filter.py              # 04 thresholds + quality gates
│   ├── dedup.py               # 05 exact + near-duplicate
│   ├── balance_split.py       # 06 balance + deterministic split
│   └── report.py              # 07 stats + data card
├── data/
│   ├── raw/  interim/  processed/
├── artifacts/
│   ├── manifest.json
│   ├── stage_stats.json
│   └── data_card.md
└── tests/
```

- Python 3.10+, type hints on public functions, `pathlib` not string paths.
- Parquet for tabular intermediates, JSONL only for the final release copy.
- Every `src/*.py` has a `main()` and a `if __name__ == "__main__":` guard.
- Tests are real assertions on real (small, fixture) data — not smoke tests that pass
  because nothing threw.

## 8. Academic honesty

This is graded coursework. Every design decision the group did not specify is yours to
make, but it must be **written down in the data card with its rationale**. The assignment
states explicitly that a careful result which finds nothing dramatic scores the same as
an exciting one. Optimise for defensibility, not for a good-looking number.

## 9. Reference papers and links

| What | Link |
|---|---|
| IndicAlign (the dataset) | https://aclanthology.org/2024.acl-long.843/ |
| IndicBERTv2 (the student) | https://aclanthology.org/2023.acl-long.693/ |
| Dynabench R4 (the weak-label teacher) | https://aclanthology.org/2021.acl-long.132/ |
| Dataset | https://huggingface.co/datasets/ai4bharat/indic-align |
| Teacher | https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target |
| Student | https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only |

Read the Dynabench paper before trusting the teacher — its definition of "hate" is
narrower than "toxic", and that mismatch is the central risk in this pipeline
(see `PROMPT.md` §4②).

## 10. Grading context

The project is 30% of the course grade: code + report 20 marks, presentation 10.
Project-internal weighting: **data quality 20%** · Half 1 (diagnose) 20% · Half 2 (cure)
20% · error analysis 20% · code with one-command reproduction 10% · write-up 10%.

Two of those — data quality and one-command reproduction, 30% between them — are decided
almost entirely by what this repository does. The data card and `reproduce.sh` are not
paperwork; they are a third of the marks.

Checkpoints: Wk 2 kick-off · Wk 4 design review · Wk 7 test set frozen · Wk 9 results
locked · Wk 13 camera-ready. Presentations 7 & 9 Dec, final submission 10 Dec.
