"""Floors that any reported score has to be read against.

Task 1 showed that a model which cannot read a single word reaches about 0.64
macro-F1 on this data using punctuation and length alone. A fine-tuned score is
meaningless without that number beside it, and the same floors on the native
gold set tell us how much of the native score is real.

    python -m src.baselines --variant proportional --native-file data/native/gold.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REPO_ROOT, compute_metrics, load_config, set_seed, write_json
from .data import load_benchmark, load_native, to_model_frame

SEED = 42


def surface_features(text: pd.Series) -> pd.DataFrame:
    """Deliberately word-blind: nothing here can identify a token.

    Punctuation, length and script purity only, matching the diagnostic in
    section 8 of the data card.
    """
    length = text.str.len().clip(lower=1)
    return pd.DataFrame(
        {
            "char_len": text.str.len(),
            "token_count": text.str.split().str.len(),
            "question_marks": text.str.count(r"\?"),
            "exclamations": text.str.count("!"),
            "commas": text.str.count(","),
            "periods": text.str.count(r"\."),
            "digits": text.str.count(r"\d"),
            "latin_upper": text.str.count(r"[A-Z]"),
            "odia_ratio": text.str.count(r"[଀-୿]") / length,
            "mean_word_len": text.str.len() / text.str.split().str.len().clip(lower=1),
        }
    ).fillna(0.0)


def fit_surface_baseline(train_text, train_y, eval_text, eval_y) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    clf.fit(surface_features(train_text), train_y)
    pred = clf.predict(surface_features(eval_text))
    m = compute_metrics(eval_y, pred)
    return {"macro_f1": round(m["macro_f1"], 4), "accuracy": round(m["accuracy"], 4),
            "hate_f1": round(m["hate_f1"], 4)}


def question_mark_rule(text: pd.Series, y_true) -> dict:
    """The artefact stated as a classifier: no question mark means HATE.

    If this scores far above chance, the label really is partly grammatical mood.
    """
    pred = (~text.str.contains(r"[?？]", regex=True)).astype(int)
    m = compute_metrics(y_true, pred)
    return {"macro_f1": round(m["macro_f1"], 4), "accuracy": round(m["accuracy"], 4),
            "hate_f1": round(m["hate_f1"], 4)}


def trivial_floors(y_true) -> dict:
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    always_hate = compute_metrics(y_true, np.ones_like(y_true))
    always_non = compute_metrics(y_true, np.zeros_like(y_true))
    random_pred = rng.integers(0, 2, size=len(y_true))
    random_m = compute_metrics(y_true, random_pred)
    return {
        "always_hate": {"macro_f1": round(always_hate["macro_f1"], 4)},
        "always_non_hate": {"macro_f1": round(always_non["macro_f1"], 4)},
        "random_50_50": {"macro_f1": round(random_m["macro_f1"], 4)},
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--variant", default="proportional", choices=["proportional", "match_minority"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--eval-split", default="test", choices=["validation", "test"])
    p.add_argument("--native-file", default=None)
    p.add_argument("--text-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config)
    set_seed(SEED)
    data_root = args.data_root or cfg["data"]["root"]

    train = to_model_frame(load_benchmark(data_root, args.variant, "train"))
    ev = to_model_frame(load_benchmark(data_root, args.variant, args.eval_split))

    result = {
        "variant": args.variant,
        "benchmark": {
            "split": args.eval_split,
            "n": int(len(ev)),
            "trivial_floors": trivial_floors(ev["label"]),
            "surface_features": fit_surface_baseline(
                train["text"], train["label"], ev["text"], ev["label"]
            ),
            "question_mark_rule": question_mark_rule(ev["text"], ev["label"]),
        },
    }

    if args.native_file:
        native = to_model_frame(load_native(args.native_file, args.text_col, args.label_col))
        result["native"] = {
            "path": str(Path(args.native_file).resolve()),
            "n": int(len(native)),
            "trivial_floors": trivial_floors(native["label"]),
            # Fitted on benchmark train, applied to the native set: this measures
            # how far the surface shortcut transfers across the translationese gap.
            "surface_features_transferred": fit_surface_baseline(
                train["text"], train["label"], native["text"], native["label"]
            ),
            "question_mark_rule": question_mark_rule(native["text"], native["label"]),
        }

    out = Path(args.out) if args.out else REPO_ROOT / cfg["paths"]["results_dir"] / f"baselines_{args.variant}.json"
    write_json(out, result)
    print(f"wrote {out}")
    for scope, block in result.items():
        if isinstance(block, dict) and "surface_features" in block:
            print(f"  {scope} surface macro-F1: {block['surface_features']['macro_f1']}")
        if isinstance(block, dict) and "surface_features_transferred" in block:
            print(f"  {scope} surface macro-F1 (transferred): {block['surface_features_transferred']['macro_f1']}")
    return result


if __name__ == "__main__":
    main()
