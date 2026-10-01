"""Deeper analysis of a native-set evaluation: uncertainty, script mix, and the
prevalence-vs-behaviour decomposition of the translationese gap.

Why this exists: the native set is ~1% HATE while the benchmark test split is
~50% HATE. Macro-F1 falls sharply under a prevalence shift even if the model's
underlying error rates (true-positive rate, false-positive rate) don't change
at all. Reporting the raw macro-F1 gap alone conflates "the metric reacting to
a 50x rarer positive class" with "the model actually behaving differently on
native text." This script separates the two, using only the two confusion
matrices already produced by src.evaluate -- no model, no re-prediction.

    python -m src.native_analysis --model-dir models/proportional --tag native_primary

Bootstrap CIs use 2,000 resamples at seed 42, matching the convention used by
the other group evaluating this same file, so the two sets of numbers are
directly comparable.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REPO_ROOT, compute_metrics, read_json, write_json

SEED = 42
N_BOOT = 2000
ODIA_LO, ODIA_HI = 0x0B00, 0x0B7F


def script_of(text: str) -> str:
    """Odia / romanized / mixed, by the fraction of *letters* in the Odia block.

    Matches Group 6's method exactly (>=60% of letters Odia -> odia_script),
    so the two groups' script breakdowns are comparable.
    """
    letters = [c for c in str(text) if unicodedata.category(c).startswith("L")]
    if not letters:
        return "no_letters"
    frac = sum(ODIA_LO <= ord(c) <= ODIA_HI for c in letters) / len(letters)
    if frac >= 0.6:
        return "odia_script"
    if frac > 0:
        return "mixed"
    return "romanized"


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, key: str,
                  n: int = N_BOOT, seed: int = SEED):
    rng = np.random.RandomState(seed)
    n_rows = len(y_true)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, n_rows, n_rows)
        vals[i] = compute_metrics(y_true[idx], y_pred[idx])[key]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), vals


def rates_from_confusion(cm) -> dict:
    (tn, fp), (fn, tp) = cm
    total = tn + fp + fn + tp
    return {
        "tpr": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "prevalence": (tp + fn) / total if total else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def expected_macro_f1(prevalence: float, tpr: float, fpr: float) -> float:
    """Macro-F1 the benchmark model's own TPR/FPR would produce at a different prevalence."""
    tp, fn = prevalence * tpr, prevalence * (1 - tpr)
    fp, tn = (1 - prevalence) * fpr, (1 - prevalence) * (1 - fpr)
    f1_hate = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_non = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return (f1_hate + f1_non) / 2


def balanced_accuracy(rates: dict) -> float:
    return (rates["tpr"] + (1 - rates["fpr"])) / 2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tag", required=True, help="native tag used in src.evaluate, e.g. native_primary")
    p.add_argument("--benchmark-tag", default="benchmark_test",
                   help="tag of the paired held-out benchmark evaluation, for the gap decomposition")
    p.add_argument("--results-dir", default=None)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    variant = Path(args.model_dir).name
    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results" / variant

    pred_path = results_dir / f"predictions_{args.tag}.parquet"
    bench_path = results_dir / f"metrics_{args.benchmark_tag}.json"
    df = pd.read_parquet(pred_path)
    bench = read_json(bench_path)

    y_true = df["gold"].to_numpy()
    y_pred = df["pred"].to_numpy()
    n_hate = int(y_true.sum())

    overall = compute_metrics(y_true, y_pred)
    lo, hi, _ = bootstrap_ci(y_true, y_pred, "macro_f1")

    # majority baseline: predict NON_HATE for everything
    majority = compute_metrics(y_true, np.zeros_like(y_true))

    # prevalence-vs-behaviour decomposition against the benchmark model's own rates
    bench_rates = rates_from_confusion(bench["confusion_matrix"])
    native_rates = rates_from_confusion(overall["confusion_matrix"])
    mt_at_native_prevalence = expected_macro_f1(
        native_rates["prevalence"], bench_rates["tpr"], bench_rates["fpr"]
    )
    raw_gap = bench["macro_f1"] - overall["macro_f1"]
    gap_from_prevalence = bench["macro_f1"] - mt_at_native_prevalence
    gap_from_behaviour = mt_at_native_prevalence - overall["macro_f1"]
    ba_bench = balanced_accuracy(bench_rates)
    ba_native = balanced_accuracy(native_rates)

    # script and question-mark breakdown
    df["script"] = df["text"].map(script_of)
    df["is_question"] = df["text"].str.contains(r"[?？]", regex=True)
    by_script = {}
    for name, sub in df.groupby("script", observed=True):
        m = compute_metrics(sub["gold"], sub["pred"])
        by_script[name] = {
            "n": int(len(sub)), "n_hate": int(sub["gold"].sum()),
            "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
            "predicted_hate_rate": round(float(sub["pred"].mean()), 4),
            "gold_hate_rate": round(float(sub["gold"].mean()), 4),
        }

    result = {
        "variant": variant,
        "tag": args.tag,
        "n": overall["n"],
        "n_hate": n_hate,
        "prevalence_pct": round(100 * native_rates["prevalence"], 2),
        "accuracy": round(overall["accuracy"], 4),
        "macro_f1": round(overall["macro_f1"], 4),
        "macro_f1_ci95": [round(lo, 4), round(hi, 4)],
        "hate_precision": round(overall["hate_precision"], 4),
        "hate_recall": round(overall["hate_recall"], 4),
        "hate_f1": round(overall["hate_f1"], 4),
        "confusion_matrix": overall["confusion_matrix"],
        "majority_baseline_non_hate": {
            "accuracy": round(majority["accuracy"], 4),
            "macro_f1": round(majority["macro_f1"], 4),
        },
        "benchmark_reference": {
            "tag": args.benchmark_tag,
            "macro_f1": bench["macro_f1"],
            "tpr": round(bench_rates["tpr"], 4),
            "fpr": round(bench_rates["fpr"], 4),
        },
        "gap_decomposition": {
            "raw_macro_f1_gap": round(raw_gap, 4),
            "mt_rates_at_native_prevalence_macro_f1": round(mt_at_native_prevalence, 4),
            "gap_from_prevalence": round(gap_from_prevalence, 4),
            "gap_from_behaviour": round(gap_from_behaviour, 4),
            "balanced_accuracy_benchmark": round(ba_bench, 4),
            "balanced_accuracy_native": round(ba_native, 4),
            "balanced_accuracy_gap": round(ba_bench - ba_native, 4),
        },
        "native_fpr": round(native_rates["fpr"], 4),
        "native_tpr": round(native_rates["tpr"], 4),
        "question_rate": round(float(df["is_question"].mean()), 4),
        "script_mix": {k: int((df["script"] == k).sum()) for k in df["script"].unique()},
        "by_script": by_script,
    }

    out_json = write_json(results_dir / f"native_analysis_{args.tag}.json", result)

    md = _render_markdown(result)
    out_md = results_dir / f"native_analysis_{args.tag}.md"
    out_md.write_text(md, encoding="utf-8")

    print(md)
    print(f"\nwrote {out_json}\nwrote {out_md}")
    return result


def _render_markdown(r: dict) -> str:
    gd = r["gap_decomposition"]
    lines = [
        f"# Native evaluation — {r['variant']} / {r['tag']}",
        "",
        f"n = {r['n']}, n_hate = {r['n_hate']} ({r['prevalence_pct']}% prevalence)",
        "",
        f"Accuracy {100*r['accuracy']:.2f}, macro-F1 {100*r['macro_f1']:.2f} "
        f"[95% CI {100*r['macro_f1_ci95'][0]:.1f}, {100*r['macro_f1_ci95'][1]:.1f}]",
        f"HATE precision/recall/F1: {100*r['hate_precision']:.1f} / {100*r['hate_recall']:.1f} / {100*r['hate_f1']:.1f}",
        f"Confusion [[TN,FP],[FN,TP]] = {r['confusion_matrix']}",
        "",
        f"Majority-class baseline (always NON_HATE): accuracy {100*r['majority_baseline_non_hate']['accuracy']:.2f}, "
        f"macro-F1 {100*r['majority_baseline_non_hate']['macro_f1']:.2f} "
        f"{'— inside our 95% CI, not distinguishable from trivial' if r['majority_baseline_non_hate']['macro_f1'] >= r['macro_f1_ci95'][0] else '— clearly beaten'}",
        "",
        "## Gap decomposition (macro-F1, benchmark minus native)",
        "",
        f"Raw gap: {100*gd['raw_macro_f1_gap']:+.2f} points",
        f"  - if the benchmark model's own TPR/FPR applied at native prevalence: {100*gd['mt_rates_at_native_prevalence_macro_f1']:.2f} macro-F1",
        f"  - of which from prevalence alone: {100*gd['gap_from_prevalence']:+.2f}",
        f"  - of which from changed behaviour: {100*gd['gap_from_behaviour']:+.2f}",
        f"Balanced accuracy (prevalence-free): benchmark {100*gd['balanced_accuracy_benchmark']:.2f}, "
        f"native {100*gd['balanced_accuracy_native']:.2f}, gap {100*gd['balanced_accuracy_gap']:+.2f}",
        "",
        f"Native false-positive rate: {100*r['native_fpr']:.1f}%. Native recall: {100*r['native_tpr']:.1f}%.",
        f"Question rate on native set: {100*r['question_rate']:.1f}%.",
        "",
        "## By script",
        "",
        "| script | n | n_hate | accuracy | macro-F1 | predicted HATE | gold HATE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in r["by_script"].items():
        lines.append(
            f"| {name} | {s['n']} | {s['n_hate']} | {100*s['accuracy']:.2f} | {100*s['macro_f1']:.2f} | "
            f"{100*s['predicted_hate_rate']:.1f}% | {100*s['gold_hate_rate']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
