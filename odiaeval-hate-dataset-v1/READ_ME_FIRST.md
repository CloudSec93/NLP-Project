# OdiaEval hate-speech training data — v1

Built by Rishi (Task 1). This is the machine-translated Odia training set for the
hate-speech track. Read this page, then `data_card.md` for the full detail.

---

## What's in the box

Two variants of the same dataset, 6 parquet files. Both are exactly class-balanced
and split with seed 42.

| variant | train | validation | test |
|---|---:|---:|---:|
| `proportional/` | 29,917 | 3,575 | 3,780 |
| `match_minority/` | 29,933 | 3,574 | 3,765 |

## Loading it

```python
import pandas as pd
train = pd.read_parquet("proportional/train.parquet")
```

## The two columns that matter

- **`text_ory`** — the Odia prompt. **This is the model input.**
- **`label`** — `0` = NON_HATE, `1` = HATE. (`label_str` is the readable version.)

## Do not use `text_eng`

`text_eng` is the English source text the labels were derived from. It ships for error
analysis only. **Never feed it to the model** — it is a direct leak of the label.

## Please train BOTH variants and report both

They differ only in how NON_HATE was sampled.

- `proportional` — natural source mix. Most likely how the published 90.94 macro-F1
  was produced, so this is the one for the reproduction target.
- `match_minority` — NON_HATE forced to the same source mix as HATE, which removes a
  source-based shortcut entirely.

The **difference between the two scores is a result in itself**: it measures how much of
the benchmark number comes from a dataset shortcut rather than actual Odia ability.
Section 8 of the data card has the numbers behind this.

## Three things to know before you trust a score

1. **The labels are machine-made.** An English hate-speech classifier scored the English
   text; that label was applied to the Odia translation. No human labelled the training set.

2. **A model that cannot read scores 0.64.** Using only punctuation and length — no words
   at all — a logistic regression reaches 0.64 macro-F1 where random guessing gets 0.50.
   That is roughly a third of the distance from chance to 90.94. Anything you score should
   be read against that floor, not against zero.

3. **The main shortcut is question marks, not topic.** Harmful requests are phrased as
   questions and were labelled NON_HATE; identity attacks are statements and were labelled
   HATE. So the labels partly encode grammatical mood. This cannot be fixed by rebalancing —
   it comes from the labelling model itself.

## One request for the error analysis

When you evaluate on the native Odia gold set, **break the errors down by question vs
statement** (a `?` in the text is a good enough proxy). Given point 3, that is the single
most informative cut available, and it tests a prediction the data card makes explicitly.

## Provenance

`manifest.json` pins the exact dataset and model commit SHAs, plus a SHA-256 for every
release file. Everything here is reproducible from the repo with one command.
