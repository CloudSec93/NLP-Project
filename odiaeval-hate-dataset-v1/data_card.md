# Data card — OdiaEval hate-speech training data (Task 1)

*Generated 2026-09-05T05:54:07Z from commit `6c1fadde3dab`, `config.yaml` `099f676717ff`.*

## 1. What this is

This is machine-translated Odia hate-speech training data with **weakly supervised labels**. No human annotated it. Odia-script user prompts were taken from three configurations of `ai4bharat/indic-align`; the *English* side of each aligned pair was scored by a pinned RoBERTa hate-speech classifier, and that judgement was carried across onto the Odia text, which is what a student model trains on. It exists to be compared against a natively-authored Odia gold test set: the difference between a model's score on this data and its score on that set is the *translationese gap* the project measures.

Two variants ship, identical in every respect except which NON_HATE rows the class balancing kept. **`proportional`** is primary and preserves the natural source mix. **`match_minority`** gives NON_HATE the same source-configuration profile as HATE, which drives the mutual information between source and label to exactly zero. Training on both and comparing is a result in its own right — it measures how much of a reported score is source shortcut rather than language ability — and costs one extra fine-tuning run.

## 2. Intended use — and what this must not be used for

This is **research data for measuring a translationese gap**. It is appropriate for fine-tuning a model whose score will be compared against a native Odia gold set, and for error analysis of that comparison.

**A model trained on this data must not be deployed to moderate Odia content.** The labels come from a classifier applied to English text and projected across a lossy translation, and that classifier is operating in the register where it is measurably weakest (§6). Its errors are systematic rather than random, so a model trained here inherits a specific and predictable bias rather than merely being imprecise.

**Construct warning — read this before interpreting any score.** In this dataset the labels do not mean what their names suggest. Because of how the teacher behaves (§7), **`HATE` is closer to "declarative identity-directed statement" and `NON_HATE` is closer to "interrogative request" than either is to its plain-language meaning.** A high score here is evidence a model has learned that distinction, which is not the same as evidence it can detect hate speech.

## 3. Provenance

| role | identifier | pinned revision |
|---|---|---|
| dataset | `ai4bharat/indic-align` | `032b6a9070e7f85f1a38e0506419f4590a20455a` |
| dataset_parquet_convert | `refs/convert/parquet` | `40d2d14e81a45934c111a80ffe61429606513355` |
| teacher | `facebook/roberta-hate-speech-dynabench-r4-target` | `391c99ab8b3f65beb77746a2cf6ddf1ddf9817e6` |

Configurations used: **`Toxic_Matrix`** (synthetic toxic prompts, Mistral-generated), **`HHRLHF_T`** (toxic prompts from Anthropic HH-RLHF), **`Dolly_T`** (a translation of Dolly-15k, ordinary instruction-following). Only the turn-0 *user prompt* was taken; assistant responses never enter the dataset in any column or language.

The source also carries `ory_Latn`, a romanised Odia rendering of the same prompts. It was deliberately **not used** — that register is another group's task, and mixing it in would confound this one.

The scoring pass ran on **cpu** with a **float32** forward pass and a **float32** softmax, batch size 64, length-sorted. Full per-file SHA-256 hashes are in `artifacts/manifest.json`.

## 4. The funnel

| stage | rows | change | why |
|---|---:|---:|---|
| raw | 138,032 | — | three IndicAlign configs at the pinned revision |
| extracted | 137,954 | −78 | quality gates: length bounds, Odia script purity, non-empty, turn shape |
| labelled | 122,950 | −15,004 | teacher uncertain, `0.05 < p < 0.90` — discarded by the assignment's rule |
| deduplicated | 103,353 | −19,597 | 19,184 exact + 413 near-duplicate |
| **proportional** release | 37,272 | −66,081 | majority class downsampled to balance |
| **match_minority** release | 37,272 | −66,081 | majority class downsampled to balance |

Every raw row is either present in a release or dropped with a named reason recorded in `stage_stats.json`. Nothing disappears unexplained.

### Yield by source configuration

| config | scored | HATE `≥0.90` | NON_HATE `≤0.05` | discarded |
|---|---:|---:|---:|---:|
| `Toxic_Matrix` | 90,308 | 17,351 (19.2%) | 60,436 (66.9%) | 12,521 (13.9%) |
| `HHRLHF_T` | 32,651 | 3,593 (11.0%) | 26,683 (81.7%) | 2,375 (7.3%) |
| `Dolly_T` | 14,995 | 74 (0.5%) | 14,813 (98.8%) | 108 (0.7%) |
| **total** | **137,954** | **21,018** (15.2%) | **101,932** (73.9%) | **15,004** (10.9%) |

The HATE yield of 21,018 is comfortable — well above the 10,000 threshold below which the group would have had to escalate the question of whether the fixed thresholds were viable at all. `Dolly_T` contributes only 74 HATE rows, which is expected: it is ordinary instruction-following data and contains almost no hate speech.

## 5. Labelling procedure and teacher validation

**Weak supervision, stated plainly.** No Odia hate-speech dataset exists, and hand-annotating 100k examples was not possible. So the labels were manufactured: the teacher scores the *English* prompt, and because IndicAlign's rows are aligned translations, that judgement is applied to the *Odia* prompt. The label therefore describes the English; the model sees the Odia. Every weakness documented below follows from that one move.

Thresholds are fixed by the assignment: `p(hate) ≥ 0.9` → HATE, `p(hate) ≤ 0.05` → NON_HATE, everything between discarded. Both bounds are inclusive. Thresholding uses a value rounded to 6 decimal places, so that floating-point differences between GPUs, dtypes and batch sizes cannot flip a boundary row between runs; the unrounded float ships as `teacher_prob_hate_raw` so the decision stays auditable. On this corpus, rounding moved **zero** rows across a band and only 1 row sits within `1e-6` of a threshold.

### Was the teacher pointed the right way round?

An inverted label mapping would poison the entire dataset and be **invisible** downstream — the class balance would look fine, the splits would look fine, and a fine-tuned model would score plausibly while having learned the exact opposite of hate. So the teacher is checked two ways on every run, before any of the corpus is scored.

**Metadata.** `num_labels == 2` and `id2label == {0: 'nothate', 1: 'hate'}`, compared verbatim — a case change is fatal, not normalised away.

**Behaviour (canary).** Orientation margin **0.799** between clearly-hateful and clearly-benign sentences, rank separated (True). Confidence bounds were pinned from measurement rather than guessed: `clear_hate_min 0.99`, `clear_benign_max 0.01`, `hard_negative_max 0.01`, against observed values of 0.998573, 0.000220 and 0.000169. The guard tests *orientation* with a wide margin rather than confidence, so a correct-but-conservative teacher cannot abort a run for the wrong reason.

**HateCheck** (Röttger et al., ACL 2021) — 3,728 cases across 29 functional tests, run against the pinned teacher:

| | n | mean `p(hate)` | at the project's thresholds |
|---|---:|---:|---|
| gold **hateful** | 2,563 | 0.9715 | **96.6%** reach `p ≥ 0.90` |
| gold **non-hateful** | 1,165 | 0.0879 | 87.4% correctly `≤ 0.05`; 7.2% falsely `≥ 0.90` |

On its own terms the teacher is strong. The problem is that its own terms are not our corpus's terms.

## 6. Teacher limitations

### The register mismatch, which is the most important limitation here

HateCheck's `phrase_question_h` — hate expressed **as a question** — is the teacher's **weakest** hateful functionality: 140 cases, mean `p(hate)` 0.8745, only 86% reaching `p ≥ 0.90` against a 96.6% average across all hateful functionalities.

**Our corpus is almost entirely questions and instructions addressed to an assistant.** The teacher is therefore operating in its weakest register *by construction*, and true HATE recall on this corpus is very likely below the 96.6% headline figure. Section 7 shows what that does to the labels.

### Other measured weaknesses

| functionality | n | gold | mean `p(hate)` | reading |
|---|---:|---|---:|---|
| `slur_homonym_nh` | 30 | non-hateful | 0.4389 | 40% falsely `≥ 0.90` — words that merely look like slurs |
| `slur_reclaimed_nh` | 81 | non-hateful | 0.1977 | 15% falsely `≥ 0.90` — reclaimed in-group usage |
| `ident_pos_nh` | 189 | non-hateful | 0.1348 | 13% falsely `≥ 0.90` — **positive** statements about identity groups |

`ident_pos_nh` matters most of these: the presence of an identity term alone moves the score upward, even when the sentence is complimentary. Any prompt in our corpus that names a group is therefore pushed toward HATE regardless of what it says about them.

### The individual-versus-group asymmetry

The teacher fires on group-level identity nouns and largely misses attacks on individuals. Measured on the canary probe:

| prompt | `p(hate)` |
|---|---:|
| "Disabled people are a burden and a waste of resources." | 0.999780 |
| "What should I say to humiliate a disabled classmate in front of everyone?" | 0.006863 |
| "How can I make life miserable for the immigrant family that moved in next door?" | 0.999746 |

### An unexplained blind spot

The canary sentence *"These people are vermin and they are infesting our country."* — textbook dehumanisation — scored **0.000366**. The teacher reads it as non-hate.

This is **not** a category weakness: HateCheck's `derog_dehum_h` scores 0.9996 mean with 100% at `p ≥ 0.90` across 140 cases. A natural explanation is that the sentence carries no explicit identity noun — "these people" is purely anaphoric — and the individual-versus-group contrast above supports that. **But the explanation does not hold up**: the canary sentence *"Every single one of them should be thrown out and sent back where they came from"* has no identity noun either and scores 0.998967.

So this is recorded as an **unexplained blind spot**, not as a theory. It is kept in the fixture, flagged `documented_failure`, counted in the orientation margin, and excluded from the confidence floor — pinning a bound off a known miss would make the bound meaningless.

## 7. The question-form artefact

This is the most consequential finding in the dataset, and it is a property of the **labelling function**, not of the corpus mix. It cannot be removed by any stratification.

The chain is short and every link is measured:

1. HateCheck flags `phrase_question_h` as the teacher's weakest hateful functionality — 86% versus 96.6% overall (§6).
2. Harmful *requests* are interrogative, and the teacher scores them near zero. Five canary probes averaged `p(hate)` **0.000243** — four orders of magnitude below the discard band. All five landed in NON_HATE.
3. Identity attacks are declarative, and the teacher scores them near one (§6 table).
4. The result is a **26-point gap in question-mark rate between the classes, inside a single source configuration** — where the source confound is definitionally zero:


| within `Toxic_Matrix` only | n | contains `?` | mean characters |
|---|---:|---:|---:|
| HATE | 17,082 | **37.6%** | 211.7 |
| NON_HATE | 13,142 | **63.4%** | 221.5 |

Length is essentially equal here; the separation is grammatical mood. The teacher's own register asymmetry has been **imprinted into the labels as a punctuation-level artefact**, and because it lives in the labelling function rather than the corpus composition, no rebalancing touches it. Section 8 shows this empirically: the `match_minority` variant drives source–label mutual information to exactly zero and still leaves most of the shortcut standing.

### A falsifiable prediction for the native-set evaluation

The training data says *question → NON_HATE, statement → HATE*. Nobody has characterised the grammatical mood of the native Odia gold set, and it should not be inspected in order to settle this. So the prediction is stated conditionally, and is falsifiable either way:

> **If the native gold set is predominantly declarative, models trained on this data will over-predict HATE. If it is interrogative-heavy, they will over-predict NON_HATE. In either case the errors should correlate with grammatical mood rather than with content.**

**Instruction for whoever runs the native evaluation:** break the error analysis down by question versus statement. Given that error analysis carries 20% of the project mark, this is the single most informative cut available, and it tests a prediction made in advance rather than one fitted afterwards.

## 8. Confounding analysis

`Toxic_Matrix` and `HHRLHF_T` are toxic corpora; `Dolly_T` is benign instruction-following. If the teacher labelled the first two HATE and the third NON_HATE, then `source_config` would effectively *be* the label, and a student model could score well by detecting which corpus a sentence came from — then collapse on the native set for entirely the wrong reason.

### Contingency, before balancing

| config | HATE | NON_HATE | HATE rate |
|---|---:|---:|---:|
| `Toxic_Matrix` | 17,351 | 60,436 | 22.31% |
| `HHRLHF_T` | 3,593 | 26,683 | 11.87% |
| `Dolly_T` | 74 | 14,813 | 0.50% |

**That did not happen.** NMI(source; label) is **0.0417** and Cramér's V **0.2010**; NON_HATE's largest single source is 59.3%, far below the 95% level that would have signalled a single-source class. The reason is the mechanism in §7: harmful requests score near zero, so they land in NON_HATE rather than in the discard band, and `Toxic_Matrix` ends up contributing heavily to **both** classes.

### What balancing does to it

| variant | NMI (arithmetic) | Cramér's V | rows |
|---|---:|---:|---:|
| `proportional` | 0.0925 | 0.3137 | 37,272 |
| `match_minority` | 0.0000 | 0.0000 | 37,272 |

Downsampling NON_HATE proportionally to its own source mix **raises** the confound, because giving the minority class equal weight amplifies its distinctive source profile. `match_minority` eliminates it exactly, at no cost in corpus size.

### Other shortcut channels

**Length.** `Toxic_Matrix` prompts are ~3.3× longer than the other two (median 175 characters versus 53 and 54 in English), with barely overlapping distributions. Under `proportional` this reaches the classes as HATE/NON_HATE mean lengths of 200.6/178.5 characters; under `match_minority` it is 200.6/209.0 — the length channel is closed.

**Chat-template scaffolding.** 85 `HHRLHF_T` rows carried raw `<s>`, `[INST]` and `<<SYS>>` markers, which are a perfect source giveaway. Traced through the funnel they resolve themselves: 85 after extraction → 7 after thresholding (the teacher discarded 78 as uncertain) → 3 after deduplication → **0 in both releases**. Some markers did partially survive translation into the Odia side, which is why this was measured on `text_ory` and not only on the English.

### Surface-feature baseline — how much is solvable without reading

A logistic regression was fitted on **surface properties of `text_ory` only** — length, token count, punctuation counts, digit and uppercase counts, script purity — with no n-grams, no bag of words and no embeddings. Anything that could identify a word was excluded. It is fitted on train and reported on validation; **the test split is never read**. The model is deliberately untuned: a weak baseline scoring high is the alarming result, and optimising it would only muddy what the number means. This is the hypothesis-only baseline method that exposed annotation artefacts in SNLI (Gururangan et al., NAACL 2018; Poliak et al., \*SEM 2018).

| variant | surface macro-F1 | F1 HATE | F1 NON_HATE |
|---|---:|---:|---:|
| `proportional` | **0.6389** | 0.6419 | 0.6358 |
| `match_minority` | **0.6220** | 0.6255 | 0.6185 |

**Read it against the right floor.** Three reference points, so the flattering one cannot be quoted alone:

| floor | macro-F1 | what it is |
|---|---:|---|
| always-one-class | 0.3275 | degenerate — what a **broken** model scores, *not* the chance level |
| random 50/50 | 0.5008 | **chance** — the level the result must be measured against |
| surface features | 0.6389 | this diagnostic |

- **`proportional`**: 0.6389 is +0.1380 over chance, covering **34%** of the distance from chance to the 90.94 macro-F1 reproduction target.
- **`match_minority`**: 0.6220 is +0.1211 over chance, covering **30%** of the distance from chance to the 90.94 macro-F1 reproduction target.

So roughly a third of the apparent task is solvable by a model that cannot read a single word. That is well short of the level at which the benchmark would be meaningless, but it is far above chance and it must be quoted alongside any score obtained on this data.

The gap between the variants is only +0.0169. Since the two corpora are identical except for which NON_HATE rows were kept, that difference isolates the portion of the shortcut created by source stratification — and it is small. `match_minority` drives source–label mutual information to **exactly zero** and still leaves most of the shortcut standing. The remainder is the question-form artefact of §7.

### Recommendation

**Train on both variants and report the difference.** It costs one extra fine-tuning run and tells the group how much of the published 90.94 — and how much of their own reported gap — is source shortcut rather than language ability. That is a genuine contribution to the error-analysis component rather than a robustness check.

## 9. Deduplication

`Toxic_Matrix` is model-generated and heavily templated, so near-duplicate families straddling a train/test boundary were the anticipated leakage risk. Deduplication runs on the **Odia** text — what the model actually sees — using exact matching on a normalised form (NFC, zero-width stripped, whitespace collapsed) followed by MinHash + LSH over character 5-grams at Jaccard 0.85.

| step | rows | removed |
|---|---:|---:|
| labelled input | 122,950 | — |
| after exact dedup | 103,766 | 19,184 |
| after near-dup clustering | 103,353 | 413 |

**The anticipated threat was largely absent.** Near-duplicate clustering removed only 413 rows beyond exact matching. Because that number was surprising, the threshold was swept rather than accepted:

| Jaccard threshold | clusters | texts merged | reduction |
|---:|---:|---:|---:|
| 0.95 | 19,999 | 1 | 0.01% |
| 0.90 | 19,995 | 5 | 0.03% |
| 0.85 ← configured | 19,978 | 22 | 0.11% |
| 0.80 | 19,962 | 38 | 0.19% |
| 0.70 | 19,875 | 125 | 0.62% |
| 0.60 | 19,642 | 358 | 1.79% |
| 0.50 | 18,940 | 1,060 | 5.30% |

The curve is smooth and shallow with **no cliff** — there is no hidden population of template families sitting just below the configured threshold. After translation into Odia, `Toxic_Matrix`'s templates diverge enough at the character level that near-duplicate clustering barely fires; reaching even 5% would require dropping to 0.50, which would start merging genuinely different prompts. The threshold is left at the configured value and this evidence is the justification.

**The real leakage source was exact duplication:** 19,184 rows of byte-identical Odia text, which could have placed the same prompt in both train and test. Those are gone.

**Tie-break.** A cluster whose members carry more than one `(label, source_config)` combination makes cluster integrity and strict stratification mutually unsatisfiable. The whole cluster is assigned its **majority** cell, with ties going to the lowest `example_id`'s cell, and the surviving representative is drawn from that cell — so a cluster that is four HATE and one NON_HATE never survives as its NON_HATE member. This fired on **25** clusters, of which 19 disagreed on label alone.

## 10. Translation quality, quantified

The deduplication pass produces a direct measurement of translation loss. **104,040** distinct English texts collapse to **103,766** distinct Odia texts — **274 pairs that differ in English are byte-identical in Odia**. And **3** exact-duplicate groups carry conflicting labels, because their English differed enough for the teacher to judge them differently while their Odia did not differ at all.

This is direct evidence that the translation is lossy enough to erase distinctions the labels were made on — the central risk of projecting a label across a translation boundary. Qualitatively, from the first rows inspected:

| English | Odia (back-translated) | what was lost |
|---|---|---|
| "slash someone's tires and **not be seen**" | "…and **I will not be able to see**" | passive read as active |
| "inflicting **pain and suffering**" | "**pain and pain**" | two words collapsed to one |

**Non-Latin text in the English column.** 30 rows (0.022%), all in `Toxic_Matrix`, contain garbled Bengali or Devanagari in `eng_Latn` rather than English — machine-translation failures in the source. The teacher, an English-only model, scored them anyway. **None became HATE**; 20 became NON_HATE and 10 were discarded. The impact is therefore 20 questionable NON_HATE rows out of 101,932, which is why no additional gate was added for a 0.022% contamination.

## 11. Splits

80/10/10 train/validation/test at seed 42, stratified by `(label, source_config)`. Split membership is `sha256(f"{seed}:{dedup_group}")` bucketed by cumulative proportion — a pure function of a stable identifier, independent of row order, shuffle state or dict iteration order. Adding a row cannot move an existing row between splits.

### `proportional`

| split | label | `Toxic_Matrix` | `HHRLHF_T` | `Dolly_T` | total |
|---|---|---:|---:|---:|---:|
| train | HATE | 13,697 | 1,196 | 55 | 14,948 |
| train | NON_HATE | 10,559 | 1,860 | 2,550 | 14,969 |
| validation | HATE | 1,686 | 140 | 8 | 1,834 |
| validation | NON_HATE | 1,210 | 208 | 323 | 1,741 |
| test | HATE | 1,699 | 144 | 11 | 1,854 |
| test | NON_HATE | 1,373 | 237 | 316 | 1,926 |

Totals: **train** 29,917 (80.27%), **validation** 3,575 (9.59%), **test** 3,780 (10.14%).

### `match_minority`

| split | label | `Toxic_Matrix` | `HHRLHF_T` | `Dolly_T` | total |
|---|---|---:|---:|---:|---:|
| train | HATE | 13,697 | 1,196 | 55 | 14,948 |
| train | NON_HATE | 13,725 | 1,202 | 58 | 14,985 |
| validation | HATE | 1,686 | 140 | 8 | 1,834 |
| validation | NON_HATE | 1,613 | 119 | 8 | 1,740 |
| test | HATE | 1,699 | 144 | 11 | 1,854 |
| test | NON_HATE | 1,744 | 159 | 8 | 1,911 |

Totals: **train** 29,933 (80.31%), **validation** 3,574 (9.59%), **test** 3,765 (10.10%).

### Leakage check

**No `dedup_group` appears in more than one split** — 37,272 groups across 37,272 rows, zero straddling. Because Stage 5 collapses each near-duplicate cluster to a single representative, this holds *by construction* rather than by policy: a group cannot span two splits when it has exactly one row. It is nevertheless asserted at runtime, and the test suite verifies it on a fixture with genuinely **shared** groups — plus a companion test demonstrating that hashing `example_id` instead of `dedup_group` *would* leak.

### Where the ratios drift, and why that was accepted

Pure hash bucketing was chosen over ranking rows within each cell. Ranking would hit 80/10/10 exactly in every cell, but inserting one row would then shift others across a boundary, breaking the edit-stability guarantee above. Edit-stability was ranked higher, so the drift is reported rather than removed. Every cell lands within about a percentage point of target except `HATE`/`Dolly_T`, which holds only 74 rows and comes out **74.3 / 10.8 / 14.9**. That is sampling noise on a small cell, not a bug.

## 12. Human agreement

> ### ⚠ Awaiting annotation — this section is a placeholder

> The agreement sample is **prepared but not yet annotated**, so no κ can be reported. This is the one gap in the deliverable, and it is a gap by construction rather than by omission: the number requires a human, and fabricating it would defeat its entire purpose.

> `artifacts/agreement_sample.csv` holds **100 rows** drawn from the decided bands, balanced by label and stratified by source configuration within each label. `artifacts/discarded_review.csv` holds **30 rows** from the discarded band for separate qualitative review — those rows have no thresholded label and therefore cannot enter a κ at all, which is why the sample is split in two rather than drawn across all three bands.

> The annotation CSVs deliberately **omit the teacher's label** and are emitted in hash order rather than probability order. Showing an annotator the model's answer measures anchoring, not agreement. Labels are rejoined by `example_id` from `artifacts/agreement_key.json` at scoring time.

> To complete it: fill the `human_label` column with `HATE` or `NON_HATE` against Dynabench's definition — identity-directed attacks on people for who they are — then run `python -m src.agreement score`. Note that under that definition a harmful request targeting nobody's identity is `NON_HATE` even though it is plainly harmful; that gap is precisely what the measurement is for.

## 13. Limitations

### Structural — these are properties of the design, not defects to be fixed

**1. Labels are projected across a lossy translation.** The teacher judges English; the model trains on Odia. §10 quantifies the loss: 274 English pairs are identical once translated, and 3 identical Odia texts carry conflicting labels. Wherever translation erased a distinction the teacher relied on, the label is wrong and nothing downstream can detect it.

**2. Toxicity is not hate speech, and the mismatch is worst exactly here.** The teacher detects identity-directed hate; the corpus is dominated by harmful *requests*, which are toxic but frequently not hate under that definition. §7 shows this is not merely a yield problem — it imprints a grammatical-mood artefact into the labels themselves.

**3. Source-configuration confounding.** Measured at NMI 0.0417 before balancing, amplified to 0.0925 by proportional balancing, and eliminated exactly in the `match_minority` variant. Mitigated in one variant, documented in both.

### Smaller, but worth knowing

- **The purity gate slightly under-samples HATE.** Of the 18 rows dropped for Odia script purity below 0.60, several were identity-targeting prompts whose English hashtags (`#cyberbullyingbasedonidentity` and similar) pushed the ratio down. At 18 rows out of 138,032 the effect is negligible, but it is not label-neutral and is recorded rather than left unsaid.
- **`Dolly_T` contributes only 74 HATE rows.** The `HATE`/`Dolly_T` cell is thin enough that its split proportions are visibly noisy (§11).
- **Single-turn only.** Every row in these three configurations is single-turn, so the multi-turn code path exists and is tested but never fires. A future configuration with real multi-turn data would exercise untested-in-practice behaviour.
- **`text_eng` ships in the release.** It is there because error analysis without it is painful. It is a trivially available leak and **must never be given to the student model**.
- **The reproduction target is an inference, not a specification.** The assignment says only "balance HATE and NON_HATE" and prescribes no source stratification. `proportional` is primary because it is what a naive uniform downsample produces and therefore the likeliest provenance of the published 90.94 — but that is an inference. The quality gates and deduplication yields here also differ from whatever produced that number, so if the group misses the reproduction gate, balance mode is one candidate cause among several, and the funnel in §4 is what makes the others diagnosable.

## 14. Reproduction

```bash
bash reproduce.sh
```

One command, from an empty clone to both frozen releases. On a fresh Colab T4 the dominant cost is the teacher scoring pass; on the CPU run recorded in this manifest it took 37.72 rows/sec. Acquisition streams the source parquet with HTTP range requests and keeps only four columns, so the 1.71 GB source is never stored locally — about 71 MB lands in `data/raw/`.

### Determinism, and its exact conditions

Seed 42 throughout. Split assignment is a pure hash of `dedup_group`; balancing uses a seeded RNG over `example_id`-sorted pools; near-duplicate clusters are built from the full verified pair graph with union-find over *sorted* pairs, so the clustering library's insertion order cannot leak into the labelling.

**Verified:** on the recorded configuration — cpu, float32 forward pass — every stage output is **byte-identical SHA-256 across two independent runs**.

**What changes on a T4 in fp16.** The forward pass uses different kernels and a lower-precision dtype, so `teacher_prob_hate_raw` will differ in its low-order bits and the release parquet will *not* be byte-identical to the CPU build. The labels are protected against this: the softmax is computed in float32 regardless of the forward dtype, and thresholding uses the value rounded to 6 decimal places. On this corpus that rounding moved zero rows across a band and exactly 1 row sits within `1e-6` of a threshold, so the label assignment is expected to be stable across hardware even though the stored floats are not. The determinism test in the suite runs on CPU for this reason.

---

*Every `[choice]` made in building this dataset — the 0.60 purity floor, the 3–2000 character bounds, MinHash at Jaccard 0.85 over character 5-grams, downsampling rather than upsampling, 80/10/10, the majority-cell tie-break, 6-decimal rounding, and the two balancing modes — is recorded above with the measurement or reasoning behind it. The two teacher thresholds (0.90 / 0.05) are fixed by the assignment and were not varied.*
