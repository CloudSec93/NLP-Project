# Native evaluation — proportional / native_sensitivity

n = 1034, n_hate = 43 (4.16% prevalence)

Accuracy 87.33, macro-F1 56.86 [95% CI 52.6, 61.5]
HATE precision/recall/F1: 13.9 / 39.5 / 20.6
Confusion [[TN,FP],[FN,TP]] = [[886, 105], [26, 17]]

Majority-class baseline (always NON_HATE): accuracy 95.84, macro-F1 48.94 — clearly beaten

## Gap decomposition (macro-F1, benchmark minus native)

Raw gap: +32.82 points
  - if the benchmark model's own TPR/FPR applied at native prevalence: 67.52 macro-F1
  - of which from prevalence alone: +22.16
  - of which from changed behaviour: +10.66
Balanced accuracy (prevalence-free): benchmark 89.69, native 64.47, gap +25.22

Native false-positive rate: 10.6%. Native recall: 39.5%.
Question rate on native set: 1.6%.

## By script

| script | n | n_hate | accuracy | macro-F1 | predicted HATE | gold HATE |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 18 | 1 | 88.89 | 47.06 | 5.6% | 5.6% |
| odia_script | 125 | 8 | 88.00 | 68.86 | 15.2% | 6.4% |
| romanized | 891 | 34 | 87.21 | 54.63 | 11.5% | 3.8% |
