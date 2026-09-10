"""Evaluate a saved fine-tuned model on the benchmark test split or the native Odia gold set.

Both evaluations go through this one code path, which is what step 6 of the
assignment requires: same model, same preprocessing, same labels, same metric.

    # benchmark held-out test
    python -m src.evaluate --model-dir models/proportional --benchmark-test

    # native Odia gold set
    python -m src.evaluate --model-dir models/proportional \
        --native-file data/native/odia_gold_test.csv

Decisions are argmax over the two logits. No threshold is fitted anywhere, on
either set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    REPO_ROOT,
    compute_metrics,
    environment_fingerprint,
    load_config,
    read_json,
    resolve_path,
    set_seed,
    write_json,
)
from .data import is_question, load_benchmark, load_native, resolve_data_root, to_model_frame


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--benchmark-test", action="store_true", help="evaluate on the held-out benchmark test split")
    p.add_argument("--benchmark-split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--variant", default=None, help="defaults to the variant recorded in run_config.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--native-file", default=None, help="path to the professor's native Odia gold set")
    p.add_argument("--text-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--label-map", default=None, help='JSON, e.g. \'{"HOF": 1, "NOT": 0}\'')
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default=None, help="name for this evaluation, used in output filenames")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if bool(args.benchmark_test) == bool(args.native_file):
        p.error("pass exactly one of --benchmark-test or --native-file")
    return args


def load_eval_frame(args, cfg) -> tuple[pd.DataFrame, dict]:
    """Return the raw frame plus a provenance record for the metrics file."""
    if args.benchmark_test:
        run_cfg_path = Path(args.model_dir) / "run_config.json"
        variant = args.variant
        if variant is None and run_cfg_path.exists():
            variant = read_json(run_cfg_path).get("variant")
        if variant is None:
            raise SystemExit("--variant is required when run_config.json is absent")
        data_root = args.data_root or cfg["data"]["root"]
        df = load_benchmark(data_root, variant, args.benchmark_split)
        provenance = {
            "kind": "benchmark",
            "variant": variant,
            "split": args.benchmark_split,
            "path": str(resolve_data_root(data_root) / variant / f"{args.benchmark_split}.parquet"),
        }
        return df, provenance

    label_map = json.loads(args.label_map) if args.label_map else None
    df = load_native(args.native_file, args.text_col, args.label_col, label_map)
    provenance = {
        "kind": "native",
        "path": str(Path(args.native_file).resolve()),
        "text_column": df["_source_text_col"].iloc[0],
        "label_column": df["_source_label_col"].iloc[0],
        "label_map_override": label_map,
        "note": "evaluation only; never used for training, threshold or model selection",
    }
    return df, provenance


def predict(model_dir: str | Path, texts: list[str], batch_size: int, max_length: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    probs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config)
    set_seed(args.seed)
    max_length = args.max_length or cfg["tokenizer"]["max_length"]

    df, provenance = load_eval_frame(args, cfg)
    frame = to_model_frame(df)
    texts = frame["text"].tolist()
    y_true = frame["label"].to_numpy()

    probs = predict(args.model_dir, texts, args.batch_size, max_length)
    y_pred = probs.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred)
    metrics["decision_rule"] = "argmax"
    metrics["provenance"] = provenance
    metrics["model_dir"] = str(Path(args.model_dir).resolve())
    metrics["max_length"] = max_length
    metrics["environment"] = environment_fingerprint()

    run_cfg_path = Path(args.model_dir) / "run_config.json"
    if run_cfg_path.exists():
        rc = read_json(run_cfg_path)
        metrics["trained_on_variant"] = rc.get("variant")
        metrics["training_seed"] = rc.get("training", {}).get("seed")

    tag = args.tag or (f"benchmark_{args.benchmark_split}" if args.benchmark_test else "native")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["paths"]["results_dir"] / Path(args.model_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {
            "text": texts,
            "gold": y_true,
            "pred": y_pred,
            "prob_non_hate": probs[:, 0],
            "prob_hate": probs[:, 1],
            "correct": (y_true == y_pred),
            "is_question": is_question(pd.Series(texts)).values,
            "char_len": [len(t) for t in texts],
        }
    )
    # Carry through any metadata that helps the error analysis.
    for col in ("source_config", "example_id", "teacher_prob_hate"):
        if col in df.columns:
            predictions[col] = df[col].values

    pred_path = out_dir / f"predictions_{tag}.parquet"
    predictions.to_parquet(pred_path, index=False)
    predictions.to_csv(out_dir / f"predictions_{tag}.csv", index=False, encoding="utf-8")
    metrics_path = write_json(out_dir / f"metrics_{tag}.json", metrics)

    print(json.dumps(
        {
            "tag": tag,
            "n": metrics["n"],
            "accuracy": round(100 * metrics["accuracy"], 2),
            "macro_f1": round(100 * metrics["macro_f1"], 2),
            "hate_precision": round(100 * metrics["hate_precision"], 2),
            "hate_recall": round(100 * metrics["hate_recall"], 2),
            "hate_f1": round(100 * metrics["hate_f1"], 2),
            "predictions": str(pred_path),
            "metrics": str(metrics_path),
        },
        indent=2,
    ))
    return metrics


if __name__ == "__main__":
    main()
