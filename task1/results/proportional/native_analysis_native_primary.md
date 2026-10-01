# Native evaluation — proportional / native_primary

n = 1034, n_hate = 10 (0.97% prevalence)

Accuracy 88.01, macro-F1 49.83 [95% CI 47.2, 52.9]
HATE precision/recall/F1: 3.3 / 40.0 / 6.1
Confusion [[TN,FP],[FN,TP]] = [[906, 118], [6, 4]]

Majority-class baseline (always NON_HATE): accuracy 99.03, macro-F1 49.76 — inside our 95% CI, not distinguishable from trivial

## Gap decomposition (macro-F1, benchmark minus native)

Raw gap: +39.85 points
  - if the benchmark model's own TPR/FPR applied at native prevalence: 54.04 macro-F1
  - of which from prevalence alone: +35.64
  - of which from changed behaviour: +4.21
Balanced accuracy (prevalence-free): benchmark 89.69, native 64.24, gap +25.45

Native false-positive rate: 11.5%. Native recall: 40.0%.
Question rate on native set: 1.6%.

## By script

| script | n | n_hate | accuracy | macro-F1 | predicted HATE | gold HATE |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 18 | 1 | 88.89 | 47.06 | 5.6% | 5.6% |
| odia_script | 125 | 4 | 84.80 | 54.51 | 15.2% | 3.2% |
| romanized | 891 | 5 | 88.44 | 48.79 | 11.5% | 0.6% |
