# OdiaEval hate-speech — Task 1 code

The pipeline that builds the machine-translated Odia hate-speech training set.
Written by Rishi. `README.md` has the run instructions; this page is the map.

**Scope:** dataset construction and labelling only. No fine-tuning, no evaluation,
no translationese-gap calculation — those are Half 1 and Half 2, deliberately kept
out of this repository (see `CLAUDE.md` §3).

---

## The pipeline — 6 stages, in order

Each stage reads the previous stage's output and never modifies its input.

| # | file | what it does |
|---|---|---|
| 1 | `src/acquire.py` | downloads the 3 IndicAlign configs, keeping 4 of 31 columns |
| 2 | `src/extract.py` | flattens to Odia/English prompt pairs, applies quality gates |
| 3 | `src/label.py` | runs the teacher model over all 138k English prompts |
| 4 | `src/filter.py` | applies the 0.90 / 0.05 thresholds, discards the uncertain middle |
| 5 | `src/dedup.py` | exact + near-duplicate removal (MinHash/LSH) |
| 6 | `src/balance_split.py` | balances the classes, splits train/validation/test |

## Supporting the teacher model

- `src/teacher.py` — loads `facebook/roberta-hate-speech-dynabench-r4-target`
- `src/canary.py` — proves the model is oriented correctly before 138k rows are scored
- `tests/fixtures/canary.py` — the canary sentences themselves, in three tiers

## Producing the outputs

- `src/card.py` — renders `data_card.md` from recorded artefacts, so the write-up
  cannot drift from what the pipeline actually did
- `src/report.py` — stage statistics
- `src/diagnostics.py` — the surface-feature baseline (the 0.64 result)
- `src/agreement.py` — builds the blind annotation sample, later scores Cohen's kappa

## Plumbing

`src/_common.py` (shared helpers) · `src/hashing.py` (stable IDs and split assignment) ·
`src/textnorm.py` (Unicode NFC, script purity) · `src/resolve_manifest.py` (pins exact
dataset and model commit SHAs).

## Running it

```bash
pip install -r requirements.txt
bash reproduce.sh
```

Roughly one hour, almost all of it Stage 3 scoring 138k prompts on CPU. It resumes from
checkpoint shards if interrupted. `reproduce.sh` ends by running the test suite, so a
broken test fails the build.

Every threshold, proportion and seed lives in `config.yaml`. No magic numbers in code.

## Tests

```bash
python -m pytest -q
```

176 tests. They are real assertions on real fixtures, not smoke tests. The ones that
matter most: the teacher's label mapping is not inverted · no `dedup_group` spans two
splits (the leakage guard), with a companion test proving the naive approach *would*
leak · threshold banding at exactly 0.05 and 0.90 · split assignment stays stable when
rows are added · the assistant response at index `[1]` can never reach the output.

## Reproducibility

Seed 42 everywhere. Split assignment is a pure function of a stable hash, not of row
order — adding a row cannot move existing rows between splits. All 12 release files come
out byte-identical across two independent runs, verified by comparing SHA-256 rather
than asserted.

## Not in this zip

The dataset itself (~17 MB of parquet) ships separately in
`odiaeval-hate-dataset-v1.zip`, along with the data card. `data/` is git-ignored here
because everything in it is reproducible from this code, and `artifacts/manifest.json`
pins a SHA-256 for every release file so integrity is checkable without shipping them.

## Build specs

`CLAUDE.md` and `PROMPT.md` are the specifications this was built against — the
constraints, the pinned versions, and the reasoning behind each design decision.
