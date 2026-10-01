# Native Odia evaluation — Group 7

File: `05_hate_speech_groups_6_7.jsonl` (shared native test set, 1,034 YouTube comments,
labelled neither / offensive / hateful by one annotator). The same file Group 6 evaluated
against, so our numbers are directly comparable to theirs.

**Protocol, fixed before any native prediction was read:**
- Frozen checkpoints from the benchmark reproduction. No retraining, no threshold tuning,
  no model selection on this data. Decisions are argmax, same as the benchmark evaluation.
- Two label mappings, both reported — primary is primary regardless of which scores higher:
  - **Primary**: `hateful` → HATE, `offensive`/`neither` → NON_HATE
  - **Sensitivity**: `hateful` + `offensive` → HATE, `neither` → NON_HATE
- Text/label columns named explicitly (`text`/`label`) — auto-detection would pick `id`
  as the text column and `source` as the label column on this file.
- Uncertainty: 2,000-sample bootstrap, 95% percentile CI, seed 42.

## Headline numbers

| Variant | Mapping | n (HATE) | Accuracy | Macro-F1 [95% CI] | HATE P/R/F1 |
|---|---|---:|---:|---|---|
| proportional | Primary | 1034 (10) | 88.01 | 49.83 [47.17, 52.95] | 3.28/40.00/6.06 |
| proportional | Sensitivity | 1034 (43) | 87.33 | 56.86 [52.58, 61.47] | 13.93/39.53/20.61 |
| match_minority | Primary | 1034 (10) | 70.41 | 44.00 [41.99, 46.20] | 2.87/90.00/5.56 |
| match_minority | Sensitivity | 1034 (43) | 71.47 | 50.06 [47.15, 53.19] | 9.87/72.09/17.37 |

## Translationese gap, decomposed

Macro-F1 depends heavily on class prevalence. The benchmark test split is ~50% HATE;
the native set is ~1–4% HATE. Reporting the raw gap alone conflates the metric reacting
to a 50× rarer positive class with the model actually behaving differently on native text.
We separate the two by asking: what would the benchmark model's own true/false-positive
rates produce if applied at native prevalence? Whatever gap remains is real behaviour change.

| Variant | Mapping | Benchmark F1 | Raw gap | — from prevalence | — from behaviour | Balanced-acc. gap |
|---|---|---:|---:|---:|---:|---:|
| proportional | Primary | 89.68 | 39.85 | 35.64 | 4.21 | 25.45 |
| proportional | Sensitivity | 89.68 | 32.82 | 22.16 | 10.66 | 25.22 |
| match_minority | Primary | 89.06 | 45.05 | 36.08 | 8.97 | 8.97 |
| match_minority | Sensitivity | 89.06 | 38.99 | 23.29 | 15.70 | 17.31 |

**Reading this table.** Both variants land a raw macro-F1 gap of roughly 33–45 points,
comparable to Group 6's +38.80 (primary). Under the primary mapping, `proportional`'s
majority-class-baseline macro-F1 (49.76) falls *inside* its 95% CI — with only 10 positive
examples, this model's native score cannot be statistically distinguished from a classifier
that never predicts HATE at all. `match_minority` has a far smaller **balanced-accuracy gap**
(+8.97 vs. +25.45 for proportional, primary mapping) — the variant explicitly built to remove
the source-based shortcut from training is also the one whose *behaviour* (not raw score)
transfers best to real native data. That is purchased at a real cost in precision: it fires
on 30% of all non-hateful native comments, against 11–12% for proportional.

## By script

| Variant | Mapping | Script | n | predicted HATE | gold HATE | macro-F1 |
|---|---|---|---:|---:|---:|---:|
| proportional | Primary | mixed | 18 | 5.56% | 5.56% | 47.06 |
| proportional | Primary | odia_script | 125 | 15.20% | 3.20% | 54.51 |
| proportional | Primary | romanized | 891 | 11.45% | 0.56% | 48.79 |
| proportional | Sensitivity | mixed | 18 | 5.56% | 5.56% | 47.06 |
| proportional | Sensitivity | odia_script | 125 | 15.20% | 6.40% | 68.86 |
| proportional | Sensitivity | romanized | 891 | 11.45% | 3.82% | 54.63 |
| match_minority | Primary | mixed | 18 | 22.22% | 5.56% | 65.16 |
| match_minority | Primary | odia_script | 125 | 32.80% | 3.20% | 49.86 |
| match_minority | Primary | romanized | 891 | 30.19% | 0.56% | 42.64 |
| match_minority | Sensitivity | mixed | 18 | 22.22% | 5.56% | 65.16 |
| match_minority | Sensitivity | odia_script | 125 | 32.80% | 6.40% | 58.12 |
| match_minority | Sensitivity | romanized | 891 | 30.19% | 3.82% | 48.50 |

86% of the native set (891/1034) is romanized, not Odia script — our model was trained only
on Odia-script text. Both variants predict HATE more often on `odia_script` rows than on
`romanized` rows, consistent with Group 6's finding on their own model: a model trained only
on Odia script is not reliably reading the 86% of this set that isn't in that script.

## Two variants, compared on real native data

This is the one thing Group 6 could not test — they trained a single model. We trained both
Task 1 balancing variants, and the gap between them on native data is far larger than the
gap on the benchmark test (0.62 points there vs. a 17–18-point accuracy swing here):

| | proportional | match_minority |
|---|---:|---:|
| Benchmark macro-F1 | 89.68 | 89.06 |
| Native accuracy (primary) | 88.01 | 70.41 |
| Native HATE recall (primary) | 40.00 | 90.00 |
| Native false-positive rate | 11.52 | 29.79 |
| Balanced-accuracy gap (primary) | 25.45 | 8.97 |

The source-shortcut removal that cost `match_minority` 0.62 points on the benchmark test
bought it a much smaller behavioural gap on real native data — the benchmark comparison alone
would have called `proportional` the better model; the native comparison complicates that.

## Limitations (same as the benchmark-side ones, plus these)

- **Single annotator, no agreement measure.** All 1,034 rows were labelled by one person
  (G7-HARSH). No κ is available.
- **10 positive examples under the primary mapping.** Any single prediction moves HATE recall
  by 10 points. The bootstrap CIs reflect this; point estimates alone should not be over-read.
- **Label definitions differ from the teacher's.** The native labels target a general notion
  of hate/offence; Task 1's teacher model was trained on identity-directed hate specifically.
  Some of the gap is a definition shift we cannot separate from the script/domain shift.
- **86% of the set is not Odia script.** Script, domain (song comments vs. the benchmark's
  toxic-prompt style), and label definition are confounded in this file; this data cannot
  separate them.