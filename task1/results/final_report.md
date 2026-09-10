# OdiaEval Group 7 — hate-speech detection results

Model: IndicBERTv2-MLM-only, fine-tuned on the Task 1 weak-labelled Odia release. Published target: 90.94 macro-F1. Decisions are argmax; no threshold was fitted on any set.

## Variant: proportional

### Result template

| Evaluation Dataset | Metric | Published Target | Our Score | Difference from Published Target |
|---|---|---:|---:|---:|
| Benchmark Test | macro-F1 | 90.94 | 89.68 | -1.26 |
| Native Odia Test | macro-F1 | N/A | not run | N/A |

| Comparison | Score |
|---|---:|
| Benchmark Score | 89.68 |
| Native Odia Score | — |
| Translationese Gap | — |

### Full metric set

| Set | n | Accuracy | Macro-F1 | HATE precision | HATE recall | HATE F1 |
|---|---:|---:|---:|---:|---:|---:|
| Benchmark held-out test | 3780 | 89.68 | 89.68 | 88.89 | 90.24 | 89.56 |
| Native Odia gold | — | — | — | — | — | — |

Reproduction gate (2 points of 90.94): **met**.

### Floors this score must be read against

| Baseline | Macro-F1 |
|---|---:|
| Random 50/50 | 49.17 |
| Surface features only, benchmark test | 63.15 |
| Question-mark rule, benchmark test | 63.45 |

## Variant: match_minority

### Result template

| Evaluation Dataset | Metric | Published Target | Our Score | Difference from Published Target |
|---|---|---:|---:|---:|
| Benchmark Test | macro-F1 | 90.94 | 89.06 | -1.88 |
| Native Odia Test | macro-F1 | N/A | not run | N/A |

| Comparison | Score |
|---|---:|
| Benchmark Score | 89.06 |
| Native Odia Score | — |
| Translationese Gap | — |

### Full metric set

| Set | n | Accuracy | Macro-F1 | HATE precision | HATE recall | HATE F1 |
|---|---:|---:|---:|---:|---:|---:|
| Benchmark held-out test | 3765 | 89.06 | 89.06 | 87.75 | 90.40 | 89.05 |
| Native Odia gold | — | — | — | — | — | — |

Reproduction gate (2 points of 90.94): **met**.

### Floors this score must be read against

| Baseline | Macro-F1 |
|---|---:|
| Random 50/50 | 50.17 |
| Surface features only, benchmark test | 62.38 |
| Question-mark rule, benchmark test | 63.08 |

## Variant comparison

The two corpora differ only in which NON_HATE rows were kept, so the difference between them isolates how much of the score comes from a source-based shortcut rather than Odia ability.

| | proportional | match_minority | difference |
|---|---:|---:|---:|
| Benchmark macro-F1 | 89.68 | 89.06 | +0.62 |

## Reproduction

```bash
bash run_all.sh --native-file data/native/<gold set>
```

Each model directory holds `run_config.json` with the resolved model revision, seed, hyperparameters and environment that produced it.
