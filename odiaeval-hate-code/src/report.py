"""Stage 7 — freeze the releases, write the manifest, write the data card.

7B  Two frozen releases under ``data/processed/``, identical in every respect
    except which NON_HATE rows the balancing kept.
7C  ``artifacts/data_card.md`` — the main deliverable, generated from the
    recorded artefacts so its numbers cannot drift from the pipeline.

The release schema is the handoff contract from PROMPT.md §7 and must not drift.
``teacher_prob_hate_raw`` is carried alongside the rounded value so the audit
trail for the rounding decision lives in the release, not only in an intermediate.

Run:  python -m src.report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src._common import (
    Paths,
    config_sha256,
    file_sha256,
    load_config,
    utc_now_iso,
)

STAGE = "07_report"
VARIANTS = ["proportional", "match_minority"]
SPLITS = ["train", "validation", "test"]

# PROMPT.md §7 handoff contract, plus the raw probability. These names ARE the
# interface for the modelling half - do not rename them.
RELEASE_SCHEMA = pa.schema([
    pa.field("example_id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("source_config", pa.string()),
    pa.field("text_ory", pa.string()),
    pa.field("text_eng", pa.string()),
    pa.field("label", pa.int8()),
    pa.field("label_str", pa.string()),
    pa.field("teacher_prob_hate", pa.float32()),
    pa.field("teacher_prob_hate_raw", pa.float32()),
    pa.field("dedup_group", pa.string()),
    pa.field("split", pa.string()),
    pa.field("num_turns", pa.int16()),
    pa.field("ory_script_purity", pa.float32()),
])


def release_dir(paths: Paths, variant: str) -> Path:
    return paths.processed / f"odia_hate_v1_{variant}"


# --------------------------------------------------------------------------- #
# 7B - freeze
# --------------------------------------------------------------------------- #
def to_release_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            "example_id": r["example_id"],
            "doc_id": r["doc_id"],
            "source_config": r["source_config"],
            "text_ory": r["text_ory"],
            "text_eng": r["text_eng"],
            "label": int(r["label"]),
            "label_str": r["label_str"],
            "teacher_prob_hate": float(r["p_hate"]),
            "teacher_prob_hate_raw": float(r["p_hate_raw"]),
            "dedup_group": r["dedup_group"],
            "split": r["split"],
            "num_turns": int(r["num_turns"]),
            "ory_script_purity": float(r["ory_script_purity"]),
        })
    return out


def freeze_variant(paths: Paths, variant: str) -> dict:
    src = paths.interim / f"split_{variant}.parquet"
    if not src.exists():
        raise SystemExit(
            f"FATAL: {src.name} missing - run 'python -m src.balance_split --all-modes'."
        )
    rows = to_release_rows(pq.read_table(src).to_pylist())
    rows.sort(key=lambda r: r["example_id"])

    out_dir = release_dir(paths, variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}

    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        tbl = pa.Table.from_pylist(sub, schema=RELEASE_SCHEMA)

        pq_path = out_dir / f"{split}.parquet"
        pq.write_table(tbl, pq_path, compression="zstd", compression_level=3)

        jl_path = out_dir / f"{split}.jsonl"
        with open(jl_path, "w", encoding="utf-8", newline="\n") as fh:
            for r in sub:
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

        for p in (pq_path, jl_path):
            files[f"{variant}/{p.name}"] = {
                "rows": len(sub),
                "bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
        print("  {:<16}{:<12}{:>8,} rows   parquet {:>7.2f} MB   jsonl {:>7.2f} MB".format(
            variant, split, len(sub), pq_path.stat().st_size / 1e6,
            jl_path.stat().st_size / 1e6))

    return {"rows": len(rows), "files": files}


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=True, cwd=Path(__file__).resolve().parent.parent)
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_manifest(cfg: dict, paths: Paths, frozen: dict, stats: dict) -> dict:
    label_extra = next(
        (s["extra"] for s in stats["stages"] if s["stage"] == "03_label"), {}
    )
    tm = label_extra.get("teacher_meta", {})
    old = json.loads((paths.artifacts / "manifest.json").read_text(encoding="utf-8"))

    return {
        "build_timestamp_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "config_sha256": config_sha256(),
        "seed": cfg["seed"],
        "pinned_artefacts": old.get("pinned_artefacts", {}),
        "dataset_configs": cfg["dataset"]["configs"],
        "expected_rows": cfg["dataset"]["expected_rows"],
        "scoring_pass": {
            "device": tm.get("device_name"),
            "dtype_forward": "float16" if tm.get("fp16_forward") else "float32",
            "softmax_dtype": tm.get("softmax_dtype"),
            "requested_batch_size": label_extra.get("requested_batch_size"),
            "effective_batch_size": label_extra.get("effective_batch_size"),
            "sort_by_length": label_extra.get("sort_by_length"),
            "prob_decimals": label_extra.get("prob_decimals"),
            "torch_version": tm.get("torch_version"),
            "deterministic_algorithms_enforced": tm.get("deterministic_algorithms_enforced"),
            "rows_per_sec": label_extra.get("rows_per_sec"),
        },
        "release_schema": [f.name for f in RELEASE_SCHEMA],
        "releases": {
            v: {
                "path": f"data/processed/odia_hate_v1_{v}",
                "rows": frozen[v]["rows"],
                "files": frozen[v]["files"],
            }
            for v in frozen
        },
        "stage_completed": "7-freeze",
    }


# --------------------------------------------------------------------------- #
# 7C - data card
# --------------------------------------------------------------------------- #
def _load_artifacts(paths: Paths) -> dict:
    def rd(name):
        p = paths.artifacts / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    return {
        "stats": rd("stage_stats.json"),
        "dist": rd("label_distribution.json"),
        "canary": rd("teacher_canary.json"),
        "manifest": rd("manifest.json"),
        "agreement": rd("agreement.json"),
    }


def _stage(stats: dict, name: str) -> dict:
    return next((s for s in stats.get("stages", []) if s["stage"] == name), {})


def write_data_card(cfg: dict, paths: Paths) -> Path:
    from src.card import render_card

    art = _load_artifacts(paths)
    md = render_card(cfg, art)
    out = paths.artifacts / "data_card.md"
    out.write_text(md, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-card", action="store_true")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    paths.processed.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("STAGE 7 - FREEZE + DATA CARD")
    print("=" * 100)
    print("\n7B  freezing releases")
    frozen = {v: freeze_variant(paths, v) for v in VARIANTS}

    stats = json.loads((paths.artifacts / "stage_stats.json").read_text(encoding="utf-8"))
    manifest = build_manifest(cfg, paths, frozen, stats)
    (paths.artifacts / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n  wrote artifacts/manifest.json  ({} files hashed across {} variants)".format(
        sum(len(f["files"]) for f in frozen.values()), len(frozen)))

    # --- row accounting, end to end ------------------------------------------- #
    raw = sum(cfg["dataset"]["expected_rows"].values())
    e2 = _stage(stats, "02_extract")
    e4 = _stage(stats, "04_filter")
    e5 = _stage(stats, "05_dedup")
    print("\n" + "=" * 100)
    print("FUNNEL - every raw row kept or dropped with a named reason")
    print("-" * 100)
    print("  {:>9,}  raw rows acquired".format(raw))
    for label, st in (("extract", e2), ("label/filter", e4), ("dedup", e5)):
        for reason, n in sorted(st.get("dropped_by_reason", {}).items()):
            print("  {:>9}  -{:<8,} {} ({})".format("", n, reason, label))
    print("  {:>9,}  after deduplication".format(e5.get("rows_out", 0)))
    for v in VARIANTS:
        st = _stage(stats, f"06_balance_split_{v}")
        print("  {:>9,}  {} release (-{:,} majority-class downsample)".format(
            st.get("rows_out", 0), v, st.get("rows_in", 0) - st.get("rows_out", 0)))
    print("=" * 100)

    if not args.skip_card:
        out = write_data_card(cfg, paths)
        print("\nwrote {} ({:,} bytes)".format(out.name, out.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
