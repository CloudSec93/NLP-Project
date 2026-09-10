"""Assemble the final report in the exact template the assignment asks for.

Reads whatever metrics files exist under results/ and writes results/final_report.md
plus results/final_report.json. Missing pieces are reported as missing rather
than quietly skipped, so an incomplete run is visible at a glance.

    python -m src.report
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import REPO_ROOT, load_config, read_json, write_json

VARIANTS = ("proportional", "match_minority")
METRIC_KEYS = ("accuracy", "macro_f1", "hate_precision", "hate_recall", "hate_f1")


def pct(value) -> str:
    return "—" if value is None else f"{100 * value:.2f}"


def collect(results_dir: Path) -> dict:
    out = {}
    for variant in VARIANTS:
        vdir = results_dir / variant
        entry = {"model_dir": str(REPO_ROOT / "models" / variant)}
        for tag, key in (("benchmark_test", "benchmark"), ("native", "native")):
            path = vdir / f"metrics_{tag}.json"
            entry[key] = read_json(path) if path.exists() else None
        ea = vdir / "error_analysis_native.json"
        entry["error_analysis_native"] = read_json(ea) if ea.exists() else None
        bl = results_dir / f"baselines_{variant}.json"
        entry["baselines"] = read_json(bl) if bl.exists() else None
        rc = REPO_ROOT / "models" / variant / "run_config.json"
        entry["run_config"] = read_json(rc) if rc.exists() else None
        out[variant] = entry
    return out


def gap(entry: dict) -> float | None:
    if not entry.get("benchmark") or not entry.get("native"):
        return None
    return entry["benchmark"]["macro_f1"] - entry["native"]["macro_f1"]


def render(collected: dict, target: float, tolerance: float) -> str:
    lines = [
        "# OdiaEval Group 7 — hate-speech detection results",
        "",
        "Model: IndicBERTv2-MLM-only, fine-tuned on the Task 1 weak-labelled Odia "
        "release. Published target: "
        f"{target:.2f} macro-F1. Decisions are argmax; no threshold was fitted on any set.",
        "",
    ]

    for variant in VARIANTS:
        e = collected[variant]
        b, n = e.get("benchmark"), e.get("native")
        lines += [f"## Variant: {variant}", ""]

        if b is None and n is None:
            lines += ["_Not yet run._", ""]
            continue

        lines += [
            "### Result template",
            "",
            "| Evaluation Dataset | Metric | Published Target | Our Score | Difference from Published Target |",
            "|---|---|---:|---:|---:|",
        ]
        if b:
            diff = 100 * b["macro_f1"] - target
            lines.append(
                f"| Benchmark Test | macro-F1 | {target:.2f} | {pct(b['macro_f1'])} | {diff:+.2f} |"
            )
        else:
            lines.append(f"| Benchmark Test | macro-F1 | {target:.2f} | not run | — |")
        lines.append(
            f"| Native Odia Test | macro-F1 | N/A | {pct(n['macro_f1']) if n else 'not run'} | N/A |"
        )

        g = gap(e)
        lines += [
            "",
            "| Comparison | Score |",
            "|---|---:|",
            f"| Benchmark Score | {pct(b['macro_f1']) if b else '—'} |",
            f"| Native Odia Score | {pct(n['macro_f1']) if n else '—'} |",
            f"| Translationese Gap | {pct(g) if g is not None else '—'} |",
            "",
            "### Full metric set",
            "",
            "| Set | n | Accuracy | Macro-F1 | HATE precision | HATE recall | HATE F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in (("Benchmark held-out test", b), ("Native Odia gold", n)):
            if not m:
                lines.append(f"| {name} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {name} | {m['n']} | " + " | ".join(pct(m[k]) for k in METRIC_KEYS) + " |"
            )

        if b:
            within = abs(100 * b["macro_f1"] - target) <= tolerance
            lines += [
                "",
                f"Reproduction gate ({tolerance:.0f} points of {target:.2f}): "
                f"**{'met' if within else 'not met'}**.",
            ]

        bl = e.get("baselines")
        if bl:
            bench_bl = bl.get("benchmark", {})
            lines += [
                "",
                "### Floors this score must be read against",
                "",
                "| Baseline | Macro-F1 |",
                "|---|---:|",
                f"| Random 50/50 | {pct(bench_bl.get('trivial_floors', {}).get('random_50_50', {}).get('macro_f1'))} |",
                f"| Surface features only, benchmark test | {pct(bench_bl.get('surface_features', {}).get('macro_f1'))} |",
                f"| Question-mark rule, benchmark test | {pct(bench_bl.get('question_mark_rule', {}).get('macro_f1'))} |",
            ]
            nat_bl = bl.get("native")
            if nat_bl:
                lines += [
                    f"| Surface features transferred, native set | {pct(nat_bl.get('surface_features_transferred', {}).get('macro_f1'))} |",
                    f"| Question-mark rule, native set | {pct(nat_bl.get('question_mark_rule', {}).get('macro_f1'))} |",
                ]

        ea = e.get("error_analysis_native")
        if ea:
            v = ea["data_card_prediction"]
            q = ea["question_vs_statement"]
            verdict = {True: "holds", False: "does not hold", None: "cannot be tested"}[v["prediction_holds"]]
            if v["prediction_holds"] is None:
                expectation = (
                    "The set is not mood-skewed enough for the data card to make a "
                    "directional prediction."
                )
            else:
                expectation = f"The data card predicted the model would {v['predicted_by_data_card']}."
            lines += [
                "",
                "### Native-set error analysis, question versus statement",
                "",
                f"The native set is {v['set_composition']} at a "
                f"{100 * v['question_share']:.1f}% question rate. {expectation} "
                f"Observed HATE bias {100 * v['hate_over_prediction']:+.1f} points, so the "
                f"advance prediction **{verdict}**.",
                "",
                "| cut | n | gold HATE | predicted HATE | accuracy | macro-F1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for name in ("question", "statement"):
                c = q.get(name, {})
                if not c.get("n"):
                    lines.append(f"| {name} | 0 | — | — | — | — |")
                    continue
                lines.append(
                    f"| {name} | {c['n']} | {100 * c['gold_hate_rate']:.1f}% | "
                    f"{100 * c['predicted_hate_rate']:.1f}% | {100 * c['accuracy']:.2f} | "
                    f"{100 * c['macro_f1']:.2f} |"
                )
        lines.append("")

    # ------------------------------------------------------------- variants
    p, m = collected["proportional"], collected["match_minority"]
    if p.get("benchmark") and m.get("benchmark"):
        lines += [
            "## Variant comparison",
            "",
            "The two corpora differ only in which NON_HATE rows were kept, so the "
            "difference between them isolates how much of the score comes from a "
            "source-based shortcut rather than Odia ability.",
            "",
            "| | proportional | match_minority | difference |",
            "|---|---:|---:|---:|",
            f"| Benchmark macro-F1 | {pct(p['benchmark']['macro_f1'])} | "
            f"{pct(m['benchmark']['macro_f1'])} | "
            f"{100 * (p['benchmark']['macro_f1'] - m['benchmark']['macro_f1']):+.2f} |",
        ]
        if p.get("native") and m.get("native"):
            lines += [
                f"| Native macro-F1 | {pct(p['native']['macro_f1'])} | "
                f"{pct(m['native']['macro_f1'])} | "
                f"{100 * (p['native']['macro_f1'] - m['native']['macro_f1']):+.2f} |",
                f"| Translationese gap | {pct(gap(p))} | {pct(gap(m))} | "
                f"{100 * (gap(p) - gap(m)):+.2f} |",
            ]
        lines.append("")

    lines += [
        "## Reproduction",
        "",
        "```bash",
        "bash run_all.sh --native-file data/native/<gold set>",
        "```",
        "",
        "Each model directory holds `run_config.json` with the resolved model "
        "revision, seed, hyperparameters and environment that produced it.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--results-dir", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    collected = collect(results_dir)
    target = cfg["targets"]["published_macro_f1"]
    tolerance = cfg["targets"]["tolerance_points"]

    markdown = render(collected, target, tolerance)
    (results_dir / "final_report.md").write_text(markdown, encoding="utf-8")
    write_json(results_dir / "final_report.json", {
        "target_macro_f1": target,
        "tolerance_points": tolerance,
        "variants": {
            v: {
                "benchmark_macro_f1": collected[v]["benchmark"]["macro_f1"] if collected[v].get("benchmark") else None,
                "native_macro_f1": collected[v]["native"]["macro_f1"] if collected[v].get("native") else None,
                "translationese_gap": gap(collected[v]),
            }
            for v in VARIANTS
        },
    })
    print(markdown)


if __name__ == "__main__":
    main()
