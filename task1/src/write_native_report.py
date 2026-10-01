"""Assemble the native-evaluation writeup from native_analysis_* outputs, both variants,
both label mappings. Separate from report.py since the benchmark report assumes a single
native tag; this one is specific to the two-mapping protocol used here.

    python -m src.write_native_report
"""
from __future__ import annotations
from pathlib import Path
from .common import REPO_ROOT, read_json, write_json

VARIANTS = ["proportional", "match_minority"]
MAPPINGS = [("native_primary", "Primary — hateful only → HATE"),
            ("native_sensitivity", "Sensitivity — hateful + offensive → HATE")]


def pct(x):
    return f"{100*x:.2f}"


def main():
    results_dir = REPO_ROOT / "results"
    data = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for tag, _ in MAPPINGS:
            data[v][tag] = read_json(results_dir / v / f"native_analysis_{tag}.json")
        data[v]["benchmark"] = read_json(results_dir / v / "metrics_benchmark_test.json")

    lines = [
        "# Native Odia evaluation — Group 7",
        "",
        "File: `05_hate_speech_groups_6_7.jsonl` (shared native test set, 1,034 YouTube comments,",
        "labelled neither / offensive / hateful by one annotator). The same file Group 6 evaluated",
        "against, so our numbers are directly comparable to theirs.",
        "",
        "**Protocol, fixed before any native prediction was read:**",
        "- Frozen checkpoints from the benchmark reproduction. No retraining, no threshold tuning,",
        "  no model selection on this data. Decisions are argmax, same as the benchmark evaluation.",
        "- Two label mappings, both reported — primary is primary regardless of which scores higher:",
        "  - **Primary**: `hateful` → HATE, `offensive`/`neither` → NON_HATE",
        "  - **Sensitivity**: `hateful` + `offensive` → HATE, `neither` → NON_HATE",
        "- Text/label columns named explicitly (`text`/`label`) — auto-detection would pick `id`",
        "  as the text column and `source` as the label column on this file.",
        "- Uncertainty: 2,000-sample bootstrap, 95% percentile CI, seed 42.",
        "",
        "## Headline numbers",
        "",
        "| Variant | Mapping | n (HATE) | Accuracy | Macro-F1 [95% CI] | HATE P/R/F1 |",
        "|---|---|---:|---:|---|---|",
    ]
    for v in VARIANTS:
        for tag, label in MAPPINGS:
            r = data[v][tag]
            lines.append(
                f"| {v} | {label.split(chr(8212))[0].strip()} | {r['n']} ({r['n_hate']}) | "
                f"{pct(r['accuracy'])} | {pct(r['macro_f1'])} "
                f"[{pct(r['macro_f1_ci95'][0])}, {pct(r['macro_f1_ci95'][1])}] | "
                f"{pct(r['hate_precision'])}/{pct(r['hate_recall'])}/{pct(r['hate_f1'])} |"
            )

    lines += [
        "",
        "## Translationese gap, decomposed",
        "",
        "Macro-F1 depends heavily on class prevalence. The benchmark test split is ~50% HATE;",
        "the native set is ~1–4% HATE. Reporting the raw gap alone conflates the metric reacting",
        "to a 50× rarer positive class with the model actually behaving differently on native text.",
        "We separate the two by asking: what would the benchmark model's own true/false-positive",
        "rates produce if applied at native prevalence? Whatever gap remains is real behaviour change.",
        "",
        "| Variant | Mapping | Benchmark F1 | Raw gap | — from prevalence | — from behaviour | Balanced-acc. gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        for tag, label in MAPPINGS:
            r = data[v][tag]
            gd = r["gap_decomposition"]
            lines.append(
                f"| {v} | {label.split(chr(8212))[0].strip()} | {pct(data[v]['benchmark']['macro_f1'])} | "
                f"{pct(gd['raw_macro_f1_gap'])} | {pct(gd['gap_from_prevalence'])} | "
                f"{pct(gd['gap_from_behaviour'])} | {pct(gd['balanced_accuracy_gap'])} |"
            )

    lines += [
        "",
        "**Reading this table.** Both variants land a raw macro-F1 gap of roughly 33–45 points,",
        "comparable to Group 6's +38.80 (primary). Under the primary mapping, `proportional`'s",
        "majority-class-baseline macro-F1 (49.76) falls *inside* its 95% CI — with only 10 positive",
        "examples, this model's native score cannot be statistically distinguished from a classifier",
        "that never predicts HATE at all. `match_minority` has a far smaller **balanced-accuracy gap**",
        "(+8.97 vs. +25.45 for proportional, primary mapping) — the variant explicitly built to remove",
        "the source-based shortcut from training is also the one whose *behaviour* (not raw score)",
        "transfers best to real native data. That is purchased at a real cost in precision: it fires",
        "on 30% of all non-hateful native comments, against 11–12% for proportional.",
        "",
        "## By script",
        "",
        "| Variant | Mapping | Script | n | predicted HATE | gold HATE | macro-F1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        for tag, label in MAPPINGS:
            r = data[v][tag]
            for script, s in r["by_script"].items():
                lines.append(
                    f"| {v} | {label.split(chr(8212))[0].strip()} | {script} | {s['n']} | "
                    f"{pct(s['predicted_hate_rate'])}% | {pct(s['gold_hate_rate'])}% | {pct(s['macro_f1'])} |"
                )

    lines += [
        "",
        "86% of the native set (891/1034) is romanized, not Odia script — our model was trained only",
        "on Odia-script text. Both variants predict HATE more often on `odia_script` rows than on",
        "`romanized` rows, consistent with Group 6's finding on their own model: a model trained only",
        "on Odia script is not reliably reading the 86% of this set that isn't in that script.",
        "",
        "## Two variants, compared on real native data",
        "",
        "This is the one thing Group 6 could not test — they trained a single model. We trained both",
        "Task 1 balancing variants, and the gap between them on native data is far larger than the",
        "gap on the benchmark test (0.62 points there vs. a 17–18-point accuracy swing here):",
        "",
        "| | proportional | match_minority |",
        "|---|---:|---:|",
        f"| Benchmark macro-F1 | {pct(data['proportional']['benchmark']['macro_f1'])} | {pct(data['match_minority']['benchmark']['macro_f1'])} |",
        f"| Native accuracy (primary) | {pct(data['proportional']['native_primary']['accuracy'])} | {pct(data['match_minority']['native_primary']['accuracy'])} |",
        f"| Native HATE recall (primary) | {pct(data['proportional']['native_primary']['hate_recall'])} | {pct(data['match_minority']['native_primary']['hate_recall'])} |",
        f"| Native false-positive rate | {pct(data['proportional']['native_primary']['native_fpr'])} | {pct(data['match_minority']['native_primary']['native_fpr'])} |",
        f"| Balanced-accuracy gap (primary) | {pct(data['proportional']['native_primary']['gap_decomposition']['balanced_accuracy_gap'])} | {pct(data['match_minority']['native_primary']['gap_decomposition']['balanced_accuracy_gap'])} |",
        "",
        "The source-shortcut removal that cost `match_minority` 0.62 points on the benchmark test",
        "bought it a much smaller behavioural gap on real native data — the benchmark comparison alone",
        "would have called `proportional` the better model; the native comparison complicates that.",
        "",
        "## Limitations (same as the benchmark-side ones, plus these)",
        "",
        "- **Single annotator, no agreement measure.** All 1,034 rows were labelled by one person",
        "  (G7-HARSH). No κ is available.",
        "- **10 positive examples under the primary mapping.** Any single prediction moves HATE recall",
        "  by 10 points. The bootstrap CIs reflect this; point estimates alone should not be over-read.",
        "- **Label definitions differ from the teacher's.** The native labels target a general notion",
        "  of hate/offence; Task 1's teacher model was trained on identity-directed hate specifically.",
        "  Some of the gap is a definition shift we cannot separate from the script/domain shift.",
        "- **86% of the set is not Odia script.** Script, domain (song comments vs. the benchmark's",
        "  toxic-prompt style), and label definition are confounded in this file; this data cannot",
        "  separate them.",
    ]

    out_md = results_dir / "native_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    write_json(results_dir / "native_report_data.json", data)
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
