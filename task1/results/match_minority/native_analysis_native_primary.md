# Native evaluation — match_minority / native_primary

n = 1034, n_hate = 10 (0.97% prevalence)

Accuracy 70.41, macro-F1 44.00 [95% CI 42.0, 46.2]
HATE precision/recall/F1: 2.9 / 90.0 / 5.6
Confusion [[TN,FP],[FN,TP]] = [[719, 305], [1, 9]]

Majority-class baseline (always NON_HATE): accuracy 99.03, macro-F1 49.76 — inside our 95% CI, not distinguishable from trivial

## Gap decomposition (macro-F1, benchmark minus native)

Raw gap: +45.05 points
  - if the benchmark model's own TPR/FPR applied at native prevalence: 52.97 macro-F1
  - of which from prevalence alone: +36.08
  - of which from changed behaviour: +8.97
Balanced accuracy (prevalence-free): benchmark 89.08, native 80.11, gap +8.97

Native false-positive rate: 29.8%. Native recall: 90.0%.
Question rate on native set: 1.6%.

## By script

| script | n | n_hate | accuracy | macro-F1 | predicted HATE | gold HATE |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 18 | 1 | 83.33 | 65.16 | 22.2% | 5.6% |
| odia_script | 125 | 4 | 70.40 | 49.86 | 32.8% | 3.2% |
| romanized | 891 | 5 | 70.15 | 42.64 | 30.2% | 0.6% |
