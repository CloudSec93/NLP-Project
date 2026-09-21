# OdiaEval — hate-speech dataset & weak-labelling pipeline (Task 1)

Builds a **frozen, reproducible, machine-translated Odia hate-speech dataset** from
`ai4bharat/indic-align`, weakly labelled through a pinned RoBERTa teacher. The dataset is
consumed by the modelling half of the project (fine-tuning `ai4bharat/IndicBERTv2-MLM-only`)
— **this repository does not train or evaluate any model.**

> ## ⚠️ `text_eng` must never be given to the student model.
> The aligned English prompt ships in the release because error analysis without it is
> painful. It is the text the labels were derived from, so a model given it can score
> near-perfectly while learning nothing about Odia. **It is a trivially available leak.**
> Use it for error analysis and nothing else.

**Read [`artifacts/data_card.md`](artifacts/data_card.md) before using this data.** It
documents a labelling artefact that changes how any score on this dataset should be
interpreted (§7 of the card).

---

## The releases

Two variants, identical in every respect **except which NON_HATE rows the class balancing
kept**. HATE rows and their split assignments are byte-for-byte identical between them.

| variant | NON_HATE source mix | NMI(source; label) | when to use |
|---|---|---:|---|
| **`odia_hate_v1_proportional`** | natural (70.5 / 12.4 / 17.1) | 0.0925 | **primary** — the likeliest provenance of the published 90.94 target |
| **`odia_hate_v1_match_minority`** | matched to HATE (91.7 / 7.9 / 0.4) | 0.0000 | ablation — source carries zero label information |

```
data/processed/odia_hate_v1_<variant>/{train,validation,test}.{parquet,jsonl}
```

37,272 rows each, exactly balanced (18,636 HATE / 18,636 NON_HATE), split 80/10/10.

**Training on both and reporting the difference is recommended.** It costs one extra
fine-tuning run and measures how much of a reported score is source shortcut rather than
language ability — see §8 of the data card.

### Schema — this is the interface; do not rename these columns

| column | type | meaning |
|---|---|---|
| `example_id` | string | stable 16-hex id, unique across the corpus |
| `doc_id` | string | source `doc_id` from IndicAlign |
| `source_config` | string | `Toxic_Matrix` \| `HHRLHF_T` \| `Dolly_T` |
| `text_ory` | string | **the model input** — Odia-script user prompt, turn 0 |
| `text_eng` | string | aligned English prompt — **error analysis only, never a feature** |
| `label` | int8 | `0` = NON_HATE, `1` = HATE |
| `label_str` | string | `NON_HATE` \| `HATE` |
| `teacher_prob_hate` | float32 | `p(hate)`, rounded to 6 dp — the value thresholded on |
| `teacher_prob_hate_raw` | float32 | `p(hate)` exactly as computed — audit trail |
| `dedup_group` | string | near-duplicate cluster id; never spans two splits |
| `split` | string | `train` \| `validation` \| `test` |
| `num_turns` | int16 | from source, for auditing |
| `ory_script_purity` | float32 | fraction of Odia-block letter codepoints |

## For downstream users

**Do:**
- train on `text_ory` and `label` only
- select checkpoints on the **validation** split
- evaluate once on `test`, and report accuracy, macro-F1 and HATE P/R/F1
- report the surface-feature baseline (0.6389 / 0.6220 macro-F1) alongside any score
- break the native-set error analysis down by **question vs statement** — the card makes a
  falsifiable prediction about this in §7

**Do not:**
- feed `text_eng` to the model in any form
- use the `test` split for anything before the final evaluation
- use the native Odia gold set for training, thresholding or model selection
- deploy a model trained on this data to moderate real Odia content — the labels carry a
  systematic, documented bias (§2, §6 of the card)

## Reproduction

```bash
bash reproduce.sh
```

One command, from an empty clone to both frozen releases plus the data card. Cold runtime
is roughly 90 minutes on CPU, dominated by the teacher scoring pass; much faster on a T4.
Stages 1 and 3 checkpoint and resume, so a dropped Colab session costs minutes rather than
the whole run.

| stage | module | in → out | ~time |
|---|---|---|---|
| 0 | `resolve_manifest.py` | — → `artifacts/manifest.json` | 10 s |
| 1 | `acquire.py` | HF parquet → `data/raw/*.parquet` | 13 min |
| 2 | `extract.py` | `data/raw/` → `pairs.parquet` | 1 min |
| 3 | `label.py` | `pairs` → `scored.parquet` | 60 min CPU |
| 4 | `filter.py` | `scored` → `labelled` + `discarded` | 30 s |
| 5 | `dedup.py` | `labelled` → `deduped.parquet` | 8 min |
| 6 | `balance_split.py` | `deduped` → `split_<variant>.parquet` | 1 min |
| 7a | `diagnostics.py` | `split_*` → surface baseline | 10 s |
| 7 | `report.py` | `split_*` → releases + data card | 1 min |

Individual stages run as `python -m src.<module>`; notebooks are thin callers only.

### Data volume

Stage 1 range-reads the hosted parquet over HTTP and prunes to four columns before
materialising anything, so the **1.71 GB source is never stored locally** — about 71 MB
lands in `data/raw/`. Peak RAM is one pruned row group, a few MB.

### Determinism

Seed 42 throughout. Split membership is `sha256(f"{seed}:{dedup_group}")` bucketed by
proportion — a pure function of a stable id, so **adding a row cannot move an existing row
between splits**. Balancing uses a seeded RNG over id-sorted pools; near-duplicate clusters
are built with union-find over *sorted* pairs so the clustering library's insertion order
cannot leak in.

**Verified:** on CPU with an fp32 forward pass, all 12 release files are byte-identical
SHA-256 across two independent runs. On a T4 in fp16 the stored floats will differ in their
low-order bits; the *labels* are protected because the softmax is float32 regardless and
thresholding uses the 6-dp rounded value. See §14 of the data card for the exact conditions.

## Tests

```bash
pytest
```

176 tests. They cover the `List[List[string]]` extraction including multi-turn and ragged
rows, script purity, teacher `id2label` orientation (an inverted map is fatal), the canary
tiers, threshold banding at exactly 0.05 and 0.90, checkpoint resume, clustering
determinism, the majority-cell tie-break, release schema conformance, manifest
completeness, end-to-end row accounting, and **the leakage check** — that no `dedup_group`
spans two splits, plus a companion test proving that hashing `example_id` instead would
leak.

## Layout

```
config.yaml            every tunable; teacher thresholds fixed at 0.90 / 0.05
requirements.txt       == pins
reproduce.sh           the one command
src/                   one module per stage, plus _common / textnorm / hashing /
                       teacher / canary / diagnostics / agreement / card
tests/                 real assertions on fixture and frozen data
docs/                  the professor's assignment .docx / .pptx (source of truth)
data/{raw,interim}/    git-ignored working data, rebuilt by the pipeline
data/processed/        the two frozen releases
artifacts/             data_card.md, manifest.json, stage_stats.json,
                       label_distribution.json, teacher_canary.json, agreement CSVs
```

`CLAUDE.md` holds the standing constraints and the scope boundary; `PROMPT.md` is the
build spec.

## Outstanding

**Human agreement (§12 of the card) is the one incomplete section.** The sample is prepared
— `artifacts/agreement_sample.csv`, 100 rows, balanced by label, with the teacher's labels
deliberately withheld so annotation measures agreement rather than anchoring. It needs a
human to fill the `human_label` column, then:

```bash
python -m src.agreement score
```

No κ is reported until that happens, and none is fabricated.
