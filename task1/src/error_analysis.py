"""Error analysis, centred on the question-versus-statement cut the data card asks for.

Task 1 recorded a falsifiable prediction before the native set was seen: the
training labels encode *question -> NON_HATE, statement -> HATE*, so a model
trained on them should over-predict HATE on a declarative native set and
over-predict NON_HATE on an interrogative one, with errors tracking grammatical
mood rather than content.

This module tests that prediction rather than describing it. It reports the
mood composition of each evaluation set, the model's behaviour inside each mood
cell, and whether mood and error are statistically associated.

    python -m src.error_analysis --predictions results/proportional/predictions_native.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REPO_ROOT, compute_metrics, write_json

MOOD_NAMES = {True: "question", False: "statement"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="a predictions_*.parquet from src.evaluate")
    p.add_argument("--label", default=None, help="name for this evaluation set in the report")
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# cuts
# --------------------------------------------------------------------------- #

def by_mood(df: pd.DataFrame) -> dict:
    """The headline cut: how the model behaves on questions versus statements."""
    out = {}
    for flag, name in MOOD_NAMES.items():
        sub = df[df["is_question"] == flag]
        if sub.empty:
            out[name] = {"n": 0}
            continue
        m = compute_metrics(sub["gold"], sub["pred"])
        out[name] = {
            "n": int(len(sub)),
            "share_of_set": round(len(sub) / len(df), 4),
            "gold_hate_rate": round(float(sub["gold"].mean()), 4),
            "predicted_hate_rate": round(float(sub["pred"].mean()), 4),
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "error_rate": round(1 - m["accuracy"], 4),
            "false_positives": int(((sub["gold"] == 0) & (sub["pred"] == 1)).sum()),
            "false_negatives": int(((sub["gold"] == 1) & (sub["pred"] == 0)).sum()),
            "confusion_matrix": m["confusion_matrix"],
        }
    return out


def mood_error_association(df: pd.DataFrame) -> dict:
    """Is being wrong associated with grammatical mood, or independent of it?

    Chi-square on the 2x2 of mood against error, plus the phi coefficient so the
    effect size is reported alongside the p-value.
    """
    from scipy import stats

    err = (~df["correct"].astype(bool)).astype(int)
    mood = df["is_question"].astype(int)
    table = pd.crosstab(mood, err).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    if table.values.min() == 0 and table.values.sum() == 0:
        return {"applicable": False}
    chi2, p, dof, _ = stats.chi2_contingency(table.values)
    n = table.values.sum()
    phi = float(np.sqrt(chi2 / n)) if n else 0.0
    return {
        "applicable": True,
        "contingency_rows_mood_statement_question": table.values.tolist(),
        "contingency_cols": ["correct", "error"],
        "chi2": round(float(chi2), 4),
        "p_value": float(p),
        "dof": int(dof),
        "phi": round(phi, 4),
        "reading": (
            "errors are associated with grammatical mood"
            if p < 0.05
            else "no significant association between mood and error"
        ),
    }


def prediction_verdict(df: pd.DataFrame) -> dict:
    """Score the data card's advance prediction against what actually happened."""
    question_share = float(df["is_question"].mean())
    gold_hate = float(df["gold"].mean())
    pred_hate = float(df["pred"].mean())
    bias = pred_hate - gold_hate

    if question_share < 0.35:
        composition = "predominantly declarative"
        expected = "over-predict HATE"
        holds = bias > 0.02
    elif question_share > 0.65:
        composition = "interrogative-heavy"
        expected = "over-predict NON_HATE"
        holds = bias < -0.02
    else:
        composition = "mixed mood"
        expected = "no directional prediction (the set is not mood-skewed)"
        holds = None

    return {
        "question_share": round(question_share, 4),
        "set_composition": composition,
        "predicted_by_data_card": expected,
        "gold_hate_rate": round(gold_hate, 4),
        "predicted_hate_rate": round(pred_hate, 4),
        "hate_over_prediction": round(bias, 4),
        "prediction_holds": holds,
    }


def by_length(df: pd.DataFrame, bins=(0, 50, 100, 200, 400, 10_000)) -> dict:
    """Length was the other shortcut channel the data card measured."""
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    buckets = pd.cut(df["char_len"], bins=list(bins), labels=labels, include_lowest=True)
    out = {}
    for name, sub in df.groupby(buckets, observed=True):
        if sub.empty:
            continue
        out[str(name)] = {
            "n": int(len(sub)),
            "accuracy": round(float(sub["correct"].mean()), 4),
            "gold_hate_rate": round(float(sub["gold"].mean()), 4),
            "predicted_hate_rate": round(float(sub["pred"].mean()), 4),
        }
    return out


def by_column(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {}
    out = {}
    for name, sub in df.groupby(column, observed=True):
        m = compute_metrics(sub["gold"], sub["pred"])
        out[str(name)] = {
            "n": int(len(sub)),
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "gold_hate_rate": round(float(sub["gold"].mean()), 4),
            "predicted_hate_rate": round(float(sub["pred"].mean()), 4),
        }
    return out


def confidence_profile(df: pd.DataFrame) -> dict:
    conf = df[["prob_non_hate", "prob_hate"]].max(axis=1)
    correct = df["correct"].astype(bool)
    return {
        "mean_confidence_correct": round(float(conf[correct].mean()), 4) if correct.any() else None,
        "mean_confidence_wrong": round(float(conf[~correct].mean()), 4) if (~correct).any() else None,
        "confident_errors_over_0.9": int(((~correct) & (conf > 0.9)).sum()),
    }


def sample_errors(df: pd.DataFrame, n: int = 25) -> list[dict]:
    """Most confident mistakes, which are the ones worth reading by hand."""
    wrong = df[~df["correct"].astype(bool)].copy()
    if wrong.empty:
        return []
    wrong["confidence"] = wrong[["prob_non_hate", "prob_hate"]].max(axis=1)
    wrong = wrong.sort_values("confidence", ascending=False).head(n)
    return [
        {
            "text": r["text"][:400],
            "gold": int(r["gold"]),
            "pred": int(r["pred"]),
            "confidence": round(float(r["confidence"]), 4),
            "is_question": bool(r["is_question"]),
            "error_type": "false_positive" if r["gold"] == 0 else "false_negative",
        }
        for _, r in wrong.iterrows()
    ]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def analyse(df: pd.DataFrame, label: str) -> dict:
    overall = compute_metrics(df["gold"], df["pred"])
    return {
        "evaluation_set": label,
        "overall": {
            "n": overall["n"],
            "accuracy": round(overall["accuracy"], 4),
            "macro_f1": round(overall["macro_f1"], 4),
            "hate_precision": round(overall["hate_precision"], 4),
            "hate_recall": round(overall["hate_recall"], 4),
            "hate_f1": round(overall["hate_f1"], 4),
            "confusion_matrix": overall["confusion_matrix"],
        },
        "question_vs_statement": by_mood(df),
        "mood_error_association": mood_error_association(df),
        "data_card_prediction": prediction_verdict(df),
        "by_length": by_length(df),
        "by_source_config": by_column(df, "source_config"),
        "confidence": confidence_profile(df),
        "top_confident_errors": sample_errors(df),
    }


def to_markdown(result: dict) -> str:
    q = result["question_vs_statement"]
    v = result["data_card_prediction"]
    a = result["mood_error_association"]
    o = result["overall"]

    lines = [
        f"# Error analysis — {result['evaluation_set']}",
        "",
        f"Rows: {o['n']}. Accuracy {100 * o['accuracy']:.2f}. "
        f"Macro-F1 {100 * o['macro_f1']:.2f}. "
        f"HATE precision {100 * o['hate_precision']:.2f}, "
        f"recall {100 * o['hate_recall']:.2f}, F1 {100 * o['hate_f1']:.2f}.",
        "",
        "## Question versus statement",
        "",
        "| cut | n | share | gold HATE | predicted HATE | accuracy | macro-F1 | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("question", "statement"):
        c = q.get(name, {})
        if not c.get("n"):
            lines.append(f"| {name} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {c['n']} | {100 * c['share_of_set']:.1f}% | "
            f"{100 * c['gold_hate_rate']:.1f}% | {100 * c['predicted_hate_rate']:.1f}% | "
            f"{100 * c['accuracy']:.2f} | {100 * c['macro_f1']:.2f} | "
            f"{c['false_positives']} | {c['false_negatives']} |"
        )

    holds = v["prediction_holds"]
    verdict = {True: "HOLDS", False: "DOES NOT HOLD", None: "NOT TESTABLE"}[holds]
    lines += [
        "",
        "## The advance prediction",
        "",
        f"The set is **{v['set_composition']}** at a {100 * v['question_share']:.1f}% question rate, "
        f"so the data card predicts the model will **{v['predicted_by_data_card']}**.",
        "",
        f"Gold HATE rate {100 * v['gold_hate_rate']:.1f}%, predicted HATE rate "
        f"{100 * v['predicted_hate_rate']:.1f}%, a bias of "
        f"{100 * v['hate_over_prediction']:+.1f} points.",
        "",
        f"**Verdict: {verdict}.**",
    ]
    if a.get("applicable"):
        lines += [
            "",
            f"Chi-square of mood against error: {a['chi2']:.2f}, p = {a['p_value']:.4g}, "
            f"phi = {a['phi']:.3f}. {a['reading'].capitalize()}.",
        ]

    if result["by_source_config"]:
        lines += ["", "## By source configuration", "",
                  "| source | n | accuracy | macro-F1 |", "|---|---:|---:|---:|"]
        for name, c in result["by_source_config"].items():
            lines.append(f"| {name} | {c['n']} | {100 * c['accuracy']:.2f} | {100 * c['macro_f1']:.2f} |")

    lines += ["", "## By length", "", "| characters | n | accuracy | predicted HATE |",
              "|---|---:|---:|---:|"]
    for name, c in result["by_length"].items():
        lines.append(f"| {name} | {c['n']} | {100 * c['accuracy']:.2f} | {100 * c['predicted_hate_rate']:.1f}% |")

    conf = result["confidence"]
    lines += [
        "",
        "## Confidence",
        "",
        f"Mean confidence when correct: {conf['mean_confidence_correct']}. "
        f"When wrong: {conf['mean_confidence_wrong']}. "
        f"Errors made above 0.9 confidence: {conf['confident_errors_over_0.9']}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    args = parse_args(argv)
    path = Path(args.predictions)
    df = pd.read_parquet(path)

    label = args.label or path.stem.replace("predictions_", "")
    result = analyse(df, label)

    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"error_analysis_{label}.json", result)
    (out_dir / f"error_analysis_{label}.md").write_text(to_markdown(result), encoding="utf-8")

    print(to_markdown(result))
    return result


if __name__ == "__main__":
    main()
