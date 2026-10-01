# Native evaluation — match_minority / native_sensitivity

n = 1034, n_hate = 43 (4.16% prevalence)

Accuracy 71.47, macro-F1 50.06 [95% CI 47.1, 53.2]
HATE precision/recall/F1: 9.9 / 72.1 / 17.4
Confusion [[TN,FP],[FN,TP]] = [[708, 283], [12, 31]]

Majority-class baseline (always NON_HATE): accuracy 95.84, macro-F1 48.94 — inside our 95% CI, not distinguishable from trivial

## Gap decomposition (macro-F1, benchmark minus native)

Raw gap: +38.99 points
  - if the benchmark model's own TPR/FPR applied at native prevalence: 65.76 macro-F1
  - of which from prevalence alone: +23.29
  - of which from changed behaviour: +15.70
Balanced accuracy (prevalence-free): benchmark 89.08, native 71.77, gap +17.31

Native false-positive rate: 28.6%. Native recall: 72.1%.
Question rate on native set: 1.6%.

## By script

| script | n | n_hate | accuracy | macro-F1 | predicted HATE | gold HATE |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 18 | 1 | 83.33 | 65.16 | 22.2% | 5.6% |
| odia_script | 125 | 8 | 73.60 | 58.12 | 32.8% | 6.4% |
| romanized | 891 | 34 | 70.93 | 48.50 | 30.2% | 3.8% |
