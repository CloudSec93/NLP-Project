"""Stage 3 — score every English prompt with the weak-label teacher.

The expensive stage, and the one that must not be silently wrong. Order of
operations is deliberate:

1. Load the teacher at its pinned SHA and assert ``num_labels`` / ``id2label``.
2. Run the canary tiers (probe -> hard -> pinned). The hard tier aborts the run
   on failure, *before* 137,954 forward passes are spent on a model that may be
   inverted.
3. Score, length-sorted, checkpointing to shards so a dropped Colab session
   resumes instead of restarting.
4. Verify the shards cover the Stage 2 row set exactly — no gaps, no duplicates.

Two probability columns are stored:

``p_hate_raw``  float32, exactly as computed.
``p_hate``      rounded to ``teacher.prob_decimals`` (6) decimal places.

**Thresholding uses the rounded value.** Floating-point results shift slightly
across GPUs, dtypes and batch sizes; a row at 0.90000001 could come back
0.89999999 elsewhere and flip from HATE to discarded. Rounding stabilises the
decision. It reduces rather than eliminates the problem — a value can still round
to exactly 0.900000 from either side — so keeping the raw float is what makes the
residual auditable. The count of rows within 1e-6 of either threshold is reported.

Run:  python -m src.label [--device cpu] [--batch-size 64] [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src._common import Paths, append_stage_stats, file_sha256, load_config
from src.canary import run_all as run_canary
from src.teacher import iter_scored, load_teacher, token_lengths

STAGE = "03_label"

# Fine bins in the tails, because that is where the thresholds live.
HIST_BINS = [
    0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
]

SHARD_SCHEMA = pa.schema(
    [pa.field("example_id", pa.string()), pa.field("p_hate_raw", pa.float32())]
)


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def load_shards(shard_dir: Path) -> dict[str, float]:
    """Every example_id already scored, from all shards on disk."""
    done: dict[str, float] = {}
    for p in sorted(shard_dir.glob("shard_*.parquet")):
        t = pq.read_table(p, schema=SHARD_SCHEMA)
        for eid, prob in zip(t.column("example_id").to_pylist(), t.column("p_hate_raw").to_pylist()):
            done[eid] = prob
    return done


def write_shard(shard_dir: Path, n: int, ids: list[str], probs: list[float]) -> None:
    tbl = pa.Table.from_pydict(
        {"example_id": ids, "p_hate_raw": np.asarray(probs, dtype=np.float32)},
        schema=SHARD_SCHEMA,
    )
    tmp = shard_dir / f"shard_{n:05d}.parquet.part"
    pq.write_table(tbl, tmp, compression="zstd", compression_level=3)
    tmp.replace(shard_dir / f"shard_{n:05d}.parquet")


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def histogram(probs: np.ndarray) -> list[dict]:
    counts, _ = np.histogram(probs, bins=HIST_BINS)
    # np.histogram's last bin is closed on the right, which is what we want for 1.0
    out = []
    for i, c in enumerate(counts):
        out.append(
            {
                "lo": HIST_BINS[i],
                "hi": HIST_BINS[i + 1],
                "count": int(c),
                "pct": float(c) / len(probs) * 100 if len(probs) else 0.0,
            }
        )
    return out


def print_histogram(title: str, probs: np.ndarray) -> list[dict]:
    h = histogram(probs)
    print(f"\n  {title}  (n={len(probs):,})")
    peak = max((b["count"] for b in h), default=1) or 1
    for b in h:
        bar = "#" * int(b["count"] / peak * 46)
        print(f"    [{b['lo']:.2f}, {b['hi']:.2f})  {b['count']:>8,}  {b['pct']:>6.2f}%  {bar}")
    return h


def band_counts(probs: np.ndarray, hi: float, lo: float) -> dict:
    n = len(probs)
    n_hate = int((probs >= hi).sum())
    n_non = int((probs <= lo).sum())
    return {
        "total": n,
        "hate_band": n_hate,
        "non_hate_band": n_non,
        "discarded_band": n - n_hate - n_non,
        "hate_pct": n_hate / n * 100 if n else 0.0,
        "non_hate_pct": n_non / n * 100 if n else 0.0,
        "discarded_pct": (n - n_hate - n_non) / n * 100 if n else 0.0,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hatecheck", action="store_true")
    parser.add_argument("--skip-canary", action="store_true",
                        help="DEBUG ONLY - never use for a release run")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    tcfg = cfg["teacher"]
    batch_size = args.batch_size or tcfg["batch_size"]
    hi, lo = cfg["thresholds"]["hate_min"], cfg["thresholds"]["non_hate_max"]

    pairs_path = paths.interim / "pairs.parquet"
    if not pairs_path.exists():
        raise SystemExit("FATAL: data/interim/pairs.parquet missing - run Stage 2 first.")
    pairs = pq.read_table(pairs_path)
    ids = pairs.column("example_id").to_pylist()
    texts = pairs.column("text_eng").to_pylist()
    configs = pairs.column("source_config").to_pylist()
    print("=" * 100)
    print(f"STAGE 3 - SCORE   ({len(ids):,} English prompts from Stage 2)")
    print("=" * 100)

    # --- 1+2. teacher + canary, before spending the corpus -------------------- #
    if args.skip_canary:
        print("\n!! --skip-canary: the orientation guard is DISABLED. Debug only.\n")
        handle = load_teacher(cfg, device=args.device)
        canary = {"skipped": True}
    else:
        canary = run_canary(
            cfg, device=args.device, hatecheck=args.hatecheck, batch_size=batch_size
        )
        (paths.artifacts / "teacher_canary.json").write_text(
            json.dumps(canary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        handle = load_teacher(cfg, device=args.device)

    # --- 3. score ------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("SCORING")
    print("=" * 100)
    lengths = token_lengths(handle, texts)
    n_trunc = int((lengths > handle.max_length).sum())
    print(f"  token length: median={int(np.median(lengths))} p95={int(np.percentile(lengths, 95))} "
          f"max={int(lengths.max())}")
    print(f"  truncated at {handle.max_length}: {n_trunc:,} rows "
          f"({n_trunc / len(ids) * 100:.4f}%)")

    shard_dir = paths.interim / "_scores"
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = load_shards(shard_dir)
    if done:
        print(f"  resuming: {len(done):,}/{len(ids):,} already scored in "
              f"{len(list(shard_dir.glob('shard_*.parquet')))} shards")

    todo = [i for i in range(len(ids)) if ids[i] not in done]
    # deterministic length-sorted order; example_id breaks ties so the batch
    # composition is a pure function of the data, not of row order
    todo.sort(key=lambda i: (int(lengths[i]), ids[i]))

    shard_n = len(list(shard_dir.glob("shard_*.parquet")))
    buf_ids: list[str] = []
    buf_p: list[float] = []
    batches_since_flush = 0
    effective_bs = batch_size
    t0 = time.time()
    n_done = 0

    for idx, probs, bs in iter_scored(
        handle, texts, todo, batch_size=batch_size, min_batch_size=tcfg["min_batch_size"]
    ):
        effective_bs = min(effective_bs, bs)
        buf_ids.extend(ids[i] for i in idx)
        buf_p.extend(float(x) for x in probs)
        n_done += len(idx)
        batches_since_flush += 1
        if batches_since_flush >= tcfg["checkpoint_every_batches"]:
            write_shard(shard_dir, shard_n, buf_ids, buf_p)
            shard_n += 1
            buf_ids, buf_p, batches_since_flush = [], [], 0
        if n_done % (batch_size * 20) < batch_size:
            el = time.time() - t0
            rate = n_done / el if el else 0
            eta = (len(todo) - n_done) / rate / 60 if rate else 0
            print(f"\r  {n_done:,}/{len(todo):,}  {rate:6.1f} rows/s  ETA {eta:5.1f} min",
                  end="", flush=True)
    if buf_ids:
        write_shard(shard_dir, shard_n, buf_ids, buf_p)
    print()
    elapsed = time.time() - t0

    # --- 4. verify shard coverage exactly matches the Stage 2 row set ---------- #
    scored = load_shards(shard_dir)
    want, got = set(ids), set(scored)
    if want != got:
        raise SystemExit(
            f"FATAL: shard coverage does not match Stage 2. missing={len(want - got):,} "
            f"unexpected={len(got - want):,}. Refusing to write a partial scored table."
        )
    total_shard_rows = sum(
        pq.ParquetFile(p).metadata.num_rows for p in shard_dir.glob("shard_*.parquet")
    )
    if total_shard_rows != len(ids):
        raise SystemExit(
            f"FATAL: shards hold {total_shard_rows:,} rows for {len(ids):,} example_ids - "
            f"a duplicate was written."
        )
    print(f"  shard coverage verified: {len(got):,} ids, {total_shard_rows:,} rows, no gaps or dupes")

    # --- assemble --------------------------------------------------------------#
    raw = np.array([scored[e] for e in ids], dtype=np.float32)
    rounded = np.round(raw.astype(np.float64), tcfg["prob_decimals"])
    out = pairs.append_column("p_hate_raw", pa.array(raw, pa.float32()))
    out = out.append_column("p_hate", pa.array(rounded, pa.float64()))
    out_path = paths.interim / "scored.parquet"
    pq.write_table(out, out_path, compression="zstd", compression_level=3)

    # --- report ---------------------------------------------------------------#
    rate = len(todo) / elapsed if elapsed else 0
    print("\n" + "=" * 100)
    print("STAGE 3 REPORT")
    print("=" * 100)
    print(f"  wall clock            {elapsed / 60:.1f} min for {len(todo):,} rows "
          f"({rate:.1f} rows/sec)")
    print(f"  device                {handle.meta['device_name']}  "
          f"fp16_forward={handle.meta['fp16_forward']}  softmax=float32")
    print(f"  final batch size      {effective_bs}  (requested {batch_size})")
    print(f"  deterministic kernels {handle.meta['deterministic_algorithms_enforced']}")
    print(f"  truncation            {n_trunc:,} / {len(ids):,} "
          f"({n_trunc / len(ids) * 100:.4f}%)")

    near = int(((np.abs(rounded - hi) <= 1e-6) | (np.abs(rounded - lo) <= 1e-6)).sum())
    print(f"  rows within 1e-6 of a threshold: {near:,}"
          + ("  <-- inspect these; rounding cannot fully stabilise them" if near else "  (none)"))

    overall_hist = print_histogram("p(hate) - ALL", rounded)
    per_config_hist = {}
    per_config_bands = {}
    for c in cfg["dataset"]["configs"]:
        mask = np.array([x == c for x in configs])
        per_config_hist[c] = print_histogram(f"p(hate) - {c}", rounded[mask])
        per_config_bands[c] = band_counts(rounded[mask], hi, lo)

    overall_bands = band_counts(rounded, hi, lo)
    def _band_cells(b):
        return (
            "{:,} ({:.1f}%)".format(b["hate_band"], b["hate_pct"]),
            "{:,} ({:.1f}%)".format(b["non_hate_band"], b["non_hate_pct"]),
            "{:,} ({:.1f}%)".format(b["discarded_band"], b["discarded_pct"]),
        )

    print()
    print("-" * 100)
    print("  {:<16}{:>10}{:>16}{:>18}{:>16}".format(
        "config", "total", "HATE >=0.90", "NON_HATE <=0.05", "discarded"))
    print("-" * 100)
    for c in cfg["dataset"]["configs"]:
        b = per_config_bands[c]
        h, nh, dsc = _band_cells(b)
        print("  {:<16}{:>10,}{:>16}{:>18}{:>16}".format(c, b["total"], h, nh, dsc))
    b = overall_bands
    h, nh, dsc = _band_cells(b)
    print("-" * 100)
    print("  {:<16}{:>10,}{:>16}{:>18}{:>16}".format("TOTAL", b["total"], h, nh, dsc))
    print("=" * 100)

    dist = {
        "histogram_bins": HIST_BINS,
        "histogram_overall": overall_hist,
        "histogram_per_config": per_config_hist,
        "bands_overall": overall_bands,
        "bands_per_config": per_config_bands,
        "rows_within_1e-6_of_threshold": near,
    }
    (paths.artifacts / "label_distribution.json").write_text(
        json.dumps(dist, indent=2), encoding="utf-8"
    )

    append_stage_stats(
        paths.artifacts, STAGE,
        rows_in=len(ids), rows_out=len(ids), dropped_by_reason={},
        extra={
            "teacher_meta": handle.meta,
            "seconds": round(elapsed, 1),
            "rows_per_sec": round(rate, 2),
            "requested_batch_size": batch_size,
            "effective_batch_size": effective_bs,
            "sort_by_length": True,
            "prob_decimals": tcfg["prob_decimals"],
            "n_truncated": n_trunc,
            "truncation_pct": n_trunc / len(ids) * 100,
            "rows_within_1e-6_of_threshold": near,
            "bands_overall": overall_bands,
            "bands_per_config": per_config_bands,
            "output_sha256": file_sha256(out_path),
            "canary_passed": not canary.get("skipped", False),
        },
    )
    print(f"\nwrote data/interim/scored.parquet  ({len(ids):,} rows)")
    print("wrote artifacts/label_distribution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
