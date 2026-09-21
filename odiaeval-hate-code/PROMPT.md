# Claude Code — build spec: OdiaEval hate-speech dataset & labelling pipeline (Task 1)

> This is the build specification referenced throughout `CLAUDE.md` (as "PROMPT.md").
> `CLAUDE.md` holds the standing constraints and wins if the two ever disagree.
> The handoff schema in §7 is the interface the modelling half consumes.

---

## Your role

You are building the **data half** of a course NLP project. You are not building the
model half. Your output is a **frozen, defensible, reproducible dataset** plus the code
that made it. Another team member will fine-tune `ai4bharat/IndicBERTv2-MLM-only` on it.
They should never need to read your code — only the schema in §7 and the data card.

Work in stages. **Stop after each stage, show the numbers, and wait for go-ahead
before starting the next one.**

---

## 1. The problem, in one paragraph

Odia has 40M+ speakers and essentially zero natively-authored evaluation sets — the
existing benchmarks were machine-translated from English. Translated text is flat and
literal; it keeps English word order and dictionary equivalents and loses idiom. A model
trained on it learns to handle translation artefacts and can score well without
understanding Odia. Our group measures that gap for **hate-speech detection**: train on
machine-translated Odia (this repo's output), then evaluate the frozen model on a native
Odia gold set the professor supplies. The difference is the result.

For hate speech there is a second wrinkle: **no Odia hate-speech dataset exists at all.**
There is no gold-labelled training corpus to fine-tune on. So the labels have to be
manufactured — which is what this pipeline does, and why every decision needs to be
written down.

## 2. What the assignment mandates (non-negotiable)

- Source: `ai4bharat/indic-align`, configs **`Toxic_Matrix`**, **`HHRLHF_T`**, **`Dolly_T`**.
- Input text: the **Odia-script user prompt** (`ory_Orya`). **Never assistant responses.**
- Labels: from the **aligned English prompt** (`eng_Latn`) scored by
  `facebook/roberta-hate-speech-dynabench-r4-target`.
  - `p(hate) >= 0.90` -> **HATE (1)**
  - `p(hate) <= 0.05` -> **NON_HATE (0)**
  - anything in `(0.05, 0.90)` -> **discarded**
- Deduplicate the Odia prompts.
- Balance HATE and NON_HATE.
- Deterministic stratified **train / validation / test** splits, **seed 42**.

Choices made for you are marked **[choice]** and must be recorded in the data card.
Genuinely open questions are marked **[decide + justify]** — reason it out before implementing.

## 3. Verified facts about the data — assert them, don't re-derive

**Configs available in `ai4bharat/indic-align`** (all have exactly one split, `train`):
`Anudesh`, `Dolly_T`, `HHRLHF_T`, `Indic_ShareLlama`, `IndoWordNet`, `OpenAssistant_T`,
`Toxic_Matrix`, `WikiHow`, `Wiki_Chat`, `Wiki_Conv`.

**Approximate sizes:** `Toxic_Matrix` ~= 90.4k rows, `HHRLHF_T` ~= 32.7k, `Dolly_T` ~= 15k.
Total ~= 138k rows before any filtering.

**Row schema.** Each row has `doc_id` (string), `num_turns` (float64), then **28 parallel
language columns**, plus `__index_level_0__` (int64) on `Toxic_Matrix` and `HHRLHF_T`
but *not* on `Dolly_T`. Every language column has type `List[List[string]]` with shape
`[[prompt, response], ...]` — an outer list of turns, each turn a two-element list.

Columns needed: **`eng_Latn`** and **`ory_Orya`**. `ory_Latn` (romanised Odia) also
exists — that is Group 4's register, not ours; ignore it but note it exists.

**Teacher model config**, verified: `id2label = {"0": "nothate", "1": "hate"}`,
`max_position_embeddings = 514` (tokenizer `max_length = 512`). RoBERTa-base sequence
classifier from Dynabench R4. `p(hate) = softmax(logits)[1]`.
**Assert the `id2label` mapping at load time.**

**Two ways to get the data**, in preference order:
1. **Parquet, column-pruned.** `GET https://huggingface.co/api/datasets/ai4bharat/indic-align/parquet/{config}/train`
   returns parquet URLs. Download, then read with
   `pyarrow.parquet.read_table(path, columns=[...])`. Strongly preferred on Colab.
2. `datasets.load_dataset("ai4bharat/indic-align", "<config>")` as a fallback. Named configs only.

**Never** call `load_dataset` without a config name. `IndoWordNet` is ~96.8M rows.

## 4. Traps — each needs a measurement in the output, not just a comment

**(1) The config-as-label shortcut.** `Toxic_Matrix` is synthetic toxic prompts
(Mistral-7B). `HHRLHF_T` is toxic prompts from Anthropic HH-RLHF. `Dolly_T` is a
translation of Dolly-15k — essentially all benign. So the teacher will label almost
everything from the first two HATE and almost everything from the third NON_HATE. If that
stands, `source_config` **is** the label, and the fine-tuned model learns "which corpus
is this" — register, length, phrasing — rather than hate. It then collapses on the native
Odia test set for the wrong reason, and the headline gap measures an artefact of our own
construction.
> **Required:** the label x source_config contingency table + normalised mutual
> information between `source_config` and `label`, in the data card, prominently. If
> NON_HATE is >= 95% single-source, say so bluntly and flag it as the biggest threat.
> **[decide + justify]** whether to mitigate (e.g. pull low-`p(hate)` NON_HATE examples
> out of `Toxic_Matrix` / `HHRLHF_T` too). Propose with numbers before implementing.

**(2) Toxicity is not hate speech.** The teacher detects *targeted hate*. `Toxic_Matrix`
contains *harmful requests* which are toxic but frequently not hate under Dynabench's
definition. Expect the `p >= 0.90` gate to pass far fewer rows than the ~123k toxic
prompts available. **Report yield per config and the full `p(hate)` histogram.** A low
yield is a finding. Do not lower the threshold.

**(3) The labels are themselves translationese.** We label the *English* text and project
the label onto the *Odia* translation, assuming translation preserves hatefulness. It
often does not. State this plainly in the data card.

**(4) Near-duplicates, especially in `Toxic_Matrix`.** Model-generated, heavily templated.
Exact-string dedup will not catch template families; a family split across train/test is
leakage that inflates the benchmark and the gap.
> Exact dedup on a normalised form (NFC, whitespace collapsed, zero-width stripped)
> **and** near-duplicate clustering. **[choice]** MinHash + LSH over character 5-grams,
> Jaccard 0.85, via `datasketch`. Keep one representative per cluster; keep the cluster
> id on every surviving row. **[decide + justify]** the tie-break when a cluster carries
> more than one `(label, source_config)` combination. Dedup on **Odia** text; also report
> how many clusters would be found on the English side.

**(5) Malformed and untranslated cells.** Some `ory_Orya` entries are empty, whitespace,
or Latin script. Some rows have `num_turns > 1` — take **turn index 0 only**, and assert
`len(row["eng_Latn"]) == len(row["ory_Orya"])` before indexing; a mismatch means the row
is dropped and counted, not patched.
> Quality gates, all counted and reported separately: non-empty after strip · **Odia
> script purity** (fraction of letter codepoints in `U+0B00–U+0B7F`) **[choice]** >= 0.60 ·
> length bounds **[choice]** 3–2000 characters · both members of the pair present.

**(6) Split determinism that survives edits.** Assign by hashing the stable example id:
`sha256(f"{salt}:{example_id}")`, low 64 bits, map to `[0,1)`, bucket by proportions.
Stratify within each `(label, source_config)` cell. Cluster-aware: hash the `dedup_group`,
not the row. Same commit, same `config.yaml` -> byte-identical parquet.

## 5. Build it in these seven stages

Stop and report after each. Every stage reads from `data/` and writes to `data/`, appends
counts to `artifacts/stage_stats.json`, and never mutates its input.

- **Stage 0 — scaffold.** Repo layout, `config.yaml`, `requirements.txt` with `==` pins,
  `README.md`, resolved dataset/model commit SHAs into `artifacts/manifest.json`.
- **Stage 1 — acquire.** Parquet route, column-pruned. Cache to `data/raw/`; rerun hits
  cache. Assert config list, row counts (warn on drift), `List[List[string]]` shape.
  Report rows/config, schema/config, 5 example rows side by side.
- **Stage 2 — extract.** Flatten to one row per document: turn-0 user prompt both
  languages. `example_id = sha256(source_config + "|" + doc_id)[:16]`, assert uniqueness.
  Apply §4(5) gates. Report rows in->out per config, drop table by gate reason, 10 rejects
  per reason.
- **Stage 3 — label.** Teacher inference over `text_eng`. fp16, `max_length=512`, sort by
  length, **[choice]** batch 64 with auto-halve on OOM. Assert `id2label`. Checkpoint +
  resume. Store raw `p(hate)` on every row. Report wall-clock, histogram overall + per
  config, counts per band.
- **Stage 4 — filter.** Apply 0.90 / 0.05. Report yield table, 15 examples per band
  including discarded. Then the §4(1) contingency table + MI.
- **Stage 5 — dedup.** Per §4(4). Report exact dups removed, clusters found, size
  distribution, 10 largest clusters, English-side count.
- **Stage 6 — balance and split.** Balance **[choice]** by downsampling majority class,
  stratified across `source_config`. Split **[choice]** 80/10/10, cluster-aware,
  stratified by `(label, source_config)`. Report final three-way x label x source table;
  confirm no `dedup_group` in more than one split.
- **Stage 7 — freeze and document.** Release artefacts (§7), data card (§6), reproduction
  test (§8). Print the funnel from 138k raw to final counts.

## 6. The data card — `artifacts/data_card.md`

Prose with tables. Required: what the dataset is / for · provenance with commit SHAs ·
full funnel table · labelling procedure + teacher limitations · `p(hate)` histogram ·
label x source contingency table + MI number **with interpretation** · every **[choice]**
with rationale · near-duplicate statistics · **Limitations** leading with the three real
ones (labels projected across translation · toxicity vs hate mismatch · source-config
confounding) · **Intended use** stating this is machine-translated training data built to
be compared against a native gold set, not for deployment.

The native Odia test set is supplied by the professor, **not** built here — the pipeline
only needs its schema, and it must never be touched by any thresholding/selection step.

## 7. Handoff contract — freeze before modelling

`data/processed/odia_hate_v1/{train,validation,test}.parquet`, plus `.jsonl` mirrors.
**These column names are the interface. Do not rename them.**

| column | type | meaning |
|---|---|---|
| `example_id` | string | stable 16-hex id, unique across the corpus |
| `doc_id` | string | source `doc_id` from IndicAlign |
| `source_config` | string | `Toxic_Matrix` \| `HHRLHF_T` \| `Dolly_T` |
| `text_ory` | string | **the model input** — Odia-script user prompt, turn 0 |
| `text_eng` | string | aligned English prompt — labelling/error-analysis **only**, never a feature |
| `label` | int8 | `0` = NON_HATE, `1` = HATE |
| `label_str` | string | `NON_HATE` \| `HATE` |
| `teacher_prob_hate` | float32 | raw `p(hate)` from the teacher |
| `dedup_group` | string | near-duplicate cluster id — all rows sharing one land in the same split |
| `split` | string | `train` \| `validation` \| `test` |
| `num_turns` | int16 | from source, for auditing |
| `ory_script_purity` | float32 | fraction of Odia-block letter codepoints |

Alongside: `artifacts/manifest.json` (SHAs, config hash, git commit, per-file SHA-256,
row counts, UTC timestamp) · `artifacts/stage_stats.json` · `artifacts/label_distribution.json`
· `artifacts/data_card.md`.

README must state: **`text_eng` must not be given to the student model.**

## 8. Definition of done

- `bash reproduce.sh` on a fresh Colab T4 runs end to end and produces the frozen
  artefacts. One command. Documented runtime.
- Running it twice yields **identical SHA-256** for every output parquet. A test asserts
  this on a small fixture subset.
- `tests/` covers, with real assertions: `List[List[string]]` extraction incl. a
  multi-turn row · script-purity scoring on Odia/Latin/mixed · teacher `id2label`
  orientation · threshold banding at exactly 0.05 and 0.90 · split determinism under row
  insertion · **no `dedup_group` spanning two splits**.
- No stage crosses the §3 scope boundary in `CLAUDE.md`.
- The data card is readable end to end by someone who has not seen the code.
