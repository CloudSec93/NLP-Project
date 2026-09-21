"""Stage 2 — flatten ``data/raw/`` into one row per document: the turn-0 pair.

Takes ``row["eng_Latn"][0][0]`` and ``row["ory_Orya"][0][0]`` only. Index ``[1]``
is the assistant response and never enters the dataset, in any column, in any
language (CLAUDE.md section 5 rule 5).

Quality gates run in a fixed order and the **first** gate a row fails is its
recorded primary reason, so ``stage_stats.json`` accounts for every input row
exactly once. A co-failure matrix is recorded alongside for diagnostics.

Gate order (``extract.gate_order``):

1. ``turn_shape_invalid``     - turn lists missing, ragged, or turn 0 absent
2. ``empty_after_strip``      - either side empty once stripped
3. ``length_out_of_bounds``   - either side outside [min_chars, max_chars]
4. ``low_ory_script_purity``  - Odia script purity below the configured floor

Purity on an empty string is meaningless, which is why shape and emptiness are
checked first. NFC normalisation is applied before every gate, so the length
bound measures post-NFC codepoints.

Outputs
    ``data/interim/pairs.parquet``           the surviving pairs
    ``data/interim/extract_rejects.parquet`` rejects kept as evidence

Run:  python -m src.extract [--allow-high-drop]
"""

from __future__ import annotations

import argparse
import random
import sys
import unicodedata
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from src._common import Paths, append_stage_stats, example_id, file_sha256, load_config
from src.textnorm import odia_script_purity

STAGE = "02_extract"

PAIRS_SCHEMA = pa.schema(
    [
        pa.field("example_id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("source_config", pa.string()),
        pa.field("text_eng", pa.string()),
        pa.field("text_ory", pa.string()),
        pa.field("num_turns", pa.int16()),
        pa.field("ory_script_purity", pa.float32()),
    ]
)

REJECTS_SCHEMA = pa.schema(
    [
        pa.field("example_id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("source_config", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("detail", pa.string()),
        pa.field("all_failed_gates", pa.string()),
        pa.field("text_eng", pa.string()),
        pa.field("text_ory", pa.string()),
        pa.field("ory_script_purity", pa.float32()),
    ]
)


# --------------------------------------------------------------------------- #
# turn access - index [1] is never read
# --------------------------------------------------------------------------- #
def turn0_prompt(cell: Any, turn_index: int) -> str | None:
    """The turn-``turn_index`` *user prompt*, or None if the shape is unusable.

    Returns element ``[0]`` of the turn (the prompt). Element ``[1]``, the
    assistant response, is never returned by this function and there is a test
    asserting so.
    """
    if cell is None or len(cell) <= turn_index:
        return None
    turn = cell[turn_index]
    if turn is None or len(turn) < 1:
        return None
    return turn[0]


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def evaluate_gates(
    eng_cell: Any,
    ory_cell: Any,
    *,
    turn_index: int,
    min_chars: int,
    max_chars: int,
    min_purity: float,
) -> tuple[str | None, str, list[str], str, str, float]:
    """Run every gate on one row.

    Returns ``(primary_reason, detail, all_failed, text_eng, text_ory, purity)``.
    ``primary_reason`` is None when the row passes. ``all_failed`` lists every
    gate the row failed, for the co-failure matrix.
    """
    failed: list[str] = []
    detail = ""

    # 1. shape --------------------------------------------------------------- #
    n_eng = len(eng_cell) if eng_cell is not None else 0
    n_ory = len(ory_cell) if ory_cell is not None else 0
    eng_raw = turn0_prompt(eng_cell, turn_index)
    ory_raw = turn0_prompt(ory_cell, turn_index)
    if n_eng != n_ory:
        failed.append("turn_shape_invalid")
        detail = f"turn count mismatch eng={n_eng} ory={n_ory}"
    elif eng_raw is None or ory_raw is None:
        failed.append("turn_shape_invalid")
        missing = "eng" if eng_raw is None else "ory"
        detail = f"turn {turn_index} absent or empty on {missing} (eng={n_eng}, ory={n_ory})"

    text_eng = unicodedata.normalize("NFC", eng_raw) if eng_raw is not None else ""
    text_ory = unicodedata.normalize("NFC", ory_raw) if ory_raw is not None else ""

    # 2. non-empty ----------------------------------------------------------- #
    eng_empty = not text_eng.strip()
    ory_empty = not text_ory.strip()
    if eng_empty or ory_empty:
        failed.append("empty_after_strip")
        which = "both" if eng_empty and ory_empty else ("eng" if eng_empty else "ory")
        if not detail:
            detail = f"empty after strip on {which}"

    # 3. length -------------------------------------------------------------- #
    len_eng, len_ory = len(text_eng), len(text_ory)
    bad: list[str] = []
    if len_eng < min_chars:
        bad.append(f"eng<{min_chars} ({len_eng})")
    if len_eng > max_chars:
        bad.append(f"eng>{max_chars} ({len_eng})")
    if len_ory < min_chars:
        bad.append(f"ory<{min_chars} ({len_ory})")
    if len_ory > max_chars:
        bad.append(f"ory>{max_chars} ({len_ory})")
    if bad:
        failed.append("length_out_of_bounds")
        if not detail:
            detail = ", ".join(bad)

    # 4. Odia script purity --------------------------------------------------- #
    purity = odia_script_purity(text_ory)
    if purity < min_purity:
        failed.append("low_ory_script_purity")
        if not detail:
            detail = f"purity {purity:.3f} < {min_purity}"

    primary = failed[0] if failed else None
    return primary, detail, failed, text_eng, text_ory, purity


# --------------------------------------------------------------------------- #
# length-drop side accounting (the Odia/English asymmetry)
# --------------------------------------------------------------------------- #
def _length_side_key(detail: str) -> str:
    sides = []
    if "eng<" in detail:
        sides.append("eng_too_short")
    if "eng>" in detail:
        sides.append("eng_too_long")
    if "ory<" in detail:
        sides.append("ory_too_short")
    if "ory>" in detail:
        sides.append("ory_too_long")
    return "+".join(sides) if sides else "unknown"


# --------------------------------------------------------------------------- #
# per-config extraction
# --------------------------------------------------------------------------- #
def extract_config(cfg: dict, paths: Paths, config: str) -> dict:
    ex = cfg["extract"]
    src = paths.raw / f"{config}.parquet"
    pf = pq.ParquetFile(src)

    kept: list[dict] = []
    rejects: list[dict] = []
    drop_counts: dict[str, int] = {}
    cofail: dict[str, int] = {}
    length_sides: dict[str, int] = {}
    rows_in = 0

    for rg in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(
            rg, columns=["doc_id", "num_turns", "eng_Latn", "ory_Orya"]
        ).to_pylist()
        for row in batch:
            rows_in += 1
            eid = example_id(config, row["doc_id"])
            primary, detail, failed, text_eng, text_ory, purity = evaluate_gates(
                row["eng_Latn"],
                row["ory_Orya"],
                turn_index=ex["turn_index"],
                min_chars=ex["min_chars"],
                max_chars=ex["max_chars"],
                min_purity=ex["min_ory_script_purity"],
            )

            nt = row["num_turns"]
            if nt is None or float(nt) != int(nt):
                raise SystemExit(
                    f"FATAL [{config}]: num_turns={nt!r} for doc_id={row['doc_id']} "
                    f"is not integral - refusing to coerce silently."
                )

            if primary is None:
                kept.append(
                    {
                        "example_id": eid,
                        "doc_id": row["doc_id"],
                        "source_config": config,
                        "text_eng": text_eng,
                        "text_ory": text_ory,
                        "num_turns": int(nt),
                        "ory_script_purity": purity,
                    }
                )
            else:
                drop_counts[primary] = drop_counts.get(primary, 0) + 1
                combo = "+".join(failed)
                cofail[combo] = cofail.get(combo, 0) + 1
                if primary == "length_out_of_bounds":
                    k = _length_side_key(detail)
                    length_sides[k] = length_sides.get(k, 0) + 1
                rejects.append(
                    {
                        "example_id": eid,
                        "doc_id": row["doc_id"],
                        "source_config": config,
                        "reason": primary,
                        "detail": detail,
                        "all_failed_gates": combo,
                        "text_eng": text_eng,
                        "text_ory": text_ory,
                        "ory_script_purity": purity,
                    }
                )

    return {
        "config": config,
        "rows_in": rows_in,
        "rows_out": len(kept),
        "drop_counts": drop_counts,
        "cofailure_matrix": cofail,
        "length_drop_sides": length_sides,
        "_kept": kept,
        "_rejects": rejects,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_drop_table(results: list[dict], gate_order: list[str]) -> None:
    print()
    print("=" * 92)
    print("DROP TABLE - primary reason (first gate failed); every row counted exactly once")
    print("-" * 92)
    header = f"{'config':<16}{'rows in':>10}{'rows out':>10}"
    for g in gate_order:
        header += f"{g[:18]:>20}"
    print(header)
    for r in results:
        line = f"{r['config']:<16}{r['rows_in']:>10,}{r['rows_out']:>10,}"
        for g in gate_order:
            n = r["drop_counts"].get(g, 0)
            pct = n / r["rows_in"] * 100 if r["rows_in"] else 0.0
            line += f"{f'{n:,} ({pct:.2f}%)':>20}"
        print(line)
    print("-" * 92)
    tin = sum(r["rows_in"] for r in results)
    tout = sum(r["rows_out"] for r in results)
    line = f"{'TOTAL':<16}{tin:>10,}{tout:>10,}"
    for g in gate_order:
        n = sum(r["drop_counts"].get(g, 0) for r in results)
        pct = n / tin * 100 if tin else 0.0
        line += f"{f'{n:,} ({pct:.2f}%)':>20}"
    print(line)
    print("=" * 92)
    print(f"row accounting: {tin:,} in = {tout:,} kept + {tin - tout:,} dropped")


def print_reject_samples(
    results: list[dict], gate_order: list[str], seed: int, n_per_reason: int
) -> None:
    """Deterministic samples: pool sorted by example_id, then a seeded sample."""
    pool: dict[str, list[dict]] = {g: [] for g in gate_order}
    for r in results:
        for rej in r["_rejects"]:
            pool.setdefault(rej["reason"], []).append(rej)

    for reason in gate_order:
        rows = sorted(pool.get(reason, []), key=lambda x: x["example_id"])
        print()
        print("=" * 92)
        print(f"SAMPLED REJECTS - {reason}  (pool = {len(rows):,})")
        print("=" * 92)
        if not rows:
            print("  (none)")
            continue
        rng = random.Random(seed)
        sample = rows if len(rows) <= n_per_reason else rng.sample(rows, n_per_reason)
        for rej in sorted(sample, key=lambda x: x["example_id"]):
            print(f"\n  [{rej['source_config']}] {rej['example_id']}  purity="
                  f"{rej['ory_script_purity']:.3f}")
            print(f"  why       : {rej['detail']}")
            print(f"  all gates : {rej['all_failed_gates']}")
            print(f"  text_eng  : {rej['text_eng'][:400]!r}")
            print(f"  text_ory  : {rej['text_ory'][:400]!r}")


def check_drop_thresholds(results: list[dict], max_pct: float) -> list[str]:
    breaches = []
    for r in results:
        for gate, n in sorted(r["drop_counts"].items()):
            pct = n / r["rows_in"] * 100 if r["rows_in"] else 0.0
            if pct > max_pct:
                breaches.append(f"{r['config']}/{gate}: {n:,} rows = {pct:.2f}% > {max_pct}%")
    return breaches


def _write(rows: Iterable[dict], schema: pa.Schema, path) -> None:
    tbl = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(tbl, path, compression="zstd", compression_level=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--allow-high-drop",
        action="store_true",
        help="write output even if a gate exceeds extract.max_gate_drop_pct",
    )
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    ex = cfg["extract"]
    gate_order = list(ex["gate_order"])
    configs = list(cfg["dataset"]["configs"])

    print("=" * 92)
    print("STAGE 2 - EXTRACT")
    print(f"  turn index      : {ex['turn_index']} (user prompt only; index [1] never read)")
    print(f"  length bounds   : {ex['min_chars']}-{ex['max_chars']} chars, post-NFC, both sides")
    print(f"  min ory purity  : {ex['min_ory_script_purity']}")
    print(f"  gate order      : {gate_order}")
    print("=" * 92)

    results = [extract_config(cfg, paths, c) for c in configs]

    # --- global example_id uniqueness (hard failure, not a fallback) ---------- #
    all_ids: dict[str, str] = {}
    for r in results:
        for row in r["_kept"]:
            prior = all_ids.get(row["example_id"])
            if prior is not None:
                raise SystemExit(
                    f"FATAL: example_id collision {row['example_id']} between {prior} "
                    f"and {row['source_config']}|{row['doc_id']}. Stage 1's audit says "
                    f"this cannot happen - this is a bug, not a fallback condition."
                )
            all_ids[row["example_id"]] = f"{row['source_config']}|{row['doc_id']}"

    print_drop_table(results, gate_order)

    sides: dict[str, int] = {}
    for r in results:
        for k, v in r["length_drop_sides"].items():
            sides[k] = sides.get(k, 0) + v
    if sides:
        print("\nlength drops by side (Odia/English codepoint asymmetry):")
        for k, v in sorted(sides.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<28}{v:>8,}")

    print_reject_samples(results, gate_order, cfg["seed"], ex["reject_samples_per_reason"])

    breaches = check_drop_thresholds(results, ex["max_gate_drop_pct"])

    rows_in = sum(r["rows_in"] for r in results)
    rows_out = sum(r["rows_out"] for r in results)
    drop_by_reason: dict[str, int] = {}
    for r in results:
        for k, v in r["drop_counts"].items():
            drop_by_reason[k] = drop_by_reason.get(k, 0) + v

    if breaches and not args.allow_high_drop:
        print()
        print("!" * 92)
        print("STOP - a quality gate dropped more than "
              f"{ex['max_gate_drop_pct']}% of a config:")
        for b in breaches:
            print(f"  * {b}")
        print()
        print("Nothing was written. That drop rate probably means the gate, or the")
        print("assumption behind it, is wrong - look at the sampled rejects above.")
        print("To proceed deliberately once you have: python -m src.extract --allow-high-drop")
        print("!" * 92)
        return 2

    out_pairs = paths.interim / "pairs.parquet"
    out_rejects = paths.interim / "extract_rejects.parquet"
    _write(
        sorted((row for r in results for row in r["_kept"]), key=lambda x: x["example_id"]),
        PAIRS_SCHEMA,
        out_pairs,
    )
    _write(
        sorted((row for r in results for row in r["_rejects"]), key=lambda x: x["example_id"]),
        REJECTS_SCHEMA,
        out_rejects,
    )

    append_stage_stats(
        paths.artifacts,
        STAGE,
        rows_in=rows_in,
        rows_out=rows_out,
        dropped_by_reason=drop_by_reason,
        extra={
            "per_config": [
                {k: v for k, v in r.items() if not k.startswith("_")} for r in results
            ],
            "length_drop_sides": sides,
            "gate_thresholds_breached": breaches,
            "allow_high_drop": args.allow_high_drop,
            "output_pairs_sha256": file_sha256(out_pairs),
            "output_rejects_sha256": file_sha256(out_rejects),
            "distinct_example_ids": len(all_ids),
        },
    )

    print(f"\nwrote data/interim/pairs.parquet            {rows_out:,} rows "
          f"({out_pairs.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote data/interim/extract_rejects.parquet  {rows_in - rows_out:,} rows")
    if breaches:
        print("\nWARNING: proceeding past gate-threshold breaches because "
              "--allow-high-drop was passed:")
        for b in breaches:
            print(f"  * {b}")
    print(f"\nStage 2 complete. {rows_out:,} pairs carried forward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
