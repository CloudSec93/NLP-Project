"""Measure the tokenised length distribution so max_length is a recorded choice, not a guess.

    python -m src.token_stats

Writes results/token_stats.json with the percentiles and the truncation cost at
several candidate lengths, for both variants and, if given, the native set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import REPO_ROOT, load_config, write_json
from .data import load_benchmark, load_native, to_model_frame

CANDIDATES = (64, 128, 192, 256, 320, 384, 512)


def profile(tokenizer, texts: list[str]) -> dict:
    lengths = np.array([len(tokenizer(t, truncation=False)["input_ids"]) for t in texts])
    return {
        "n": int(len(lengths)),
        "mean": round(float(lengths.mean()), 1),
        "percentiles": {
            str(p): int(np.percentile(lengths, p)) for p in (50, 75, 90, 95, 99, 99.9)
        },
        "max": int(lengths.max()),
        "truncated_fraction_at": {
            str(c): round(float((lengths > c).mean()), 5) for c in CANDIDATES
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--native-file", default=None)
    p.add_argument("--sample", type=int, default=5000, help="rows sampled per split")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"].get("revision"))
    data_root = args.data_root or cfg["data"]["root"]

    out = {"model": cfg["model"]["name"], "candidates": list(CANDIDATES), "sets": {}}
    for variant in cfg["data"]["variants"]:
        for split in ("train", "test"):
            df = to_model_frame(load_benchmark(data_root, variant, split))
            if len(df) > args.sample:
                df = df.sample(n=args.sample, random_state=42)
            out["sets"][f"{variant}/{split}"] = profile(tokenizer, df["text"].tolist())

    if args.native_file:
        native = to_model_frame(load_native(args.native_file))
        out["sets"]["native"] = profile(tokenizer, native["text"].tolist())

    path = write_json(REPO_ROOT / cfg["paths"]["results_dir"] / "token_stats.json", out)
    print(f"wrote {path}")
    for name, block in out["sets"].items():
        p95 = block["percentiles"]["95"]
        trunc = block["truncated_fraction_at"][str(cfg["tokenizer"]["max_length"])]
        print(f"  {name}: p95={p95} tokens, truncated at configured max_length={trunc:.3%}")


if __name__ == "__main__":
    main()
