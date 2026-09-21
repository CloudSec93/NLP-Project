"""Stage 1 — acquire IndicAlign configs, column-pruned, cached to ``data/raw/``.

Reads the Hugging Face hosted parquet at the **pinned** ``refs/convert/parquet``
revision and keeps only the four columns this project needs
(``doc_id, num_turns, eng_Latn, ory_Orya``).

Two modes (``acquire.mode`` in config.yaml):

``stream`` (default)
    HTTP range-read the remote parquet row group by row group, pruning columns
    before anything is materialised. Transfers roughly 7% of the source bytes
    and never stores the 1.71 GB original on disk. Row groups are written to
    per-config shards so a dropped Colab session resumes where it stopped.

``download``
    Fetch the whole parquet to ``data/raw/hf/`` first, then prune locally.
    Fallback for flaky networks or fully offline reruns.

Either way the stage output is ``data/raw/<config>.parquet`` — that file *is*
the cache, so a rerun does no network I/O at all.

Run:  python -m src.acquire [--only Toxic_Matrix] [--mode download] [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src._common import Paths, append_stage_stats, example_id, file_sha256, load_config

STAGE = "01_acquire"
_HF = "https://huggingface.co"
_DATASETS_SERVER = "https://datasets-server.huggingface.co"

# The shape every language column must have: an outer list of turns, each turn a
# list of strings [prompt, response] (CLAUDE.md section 5 rule 5).
_EXPECTED_TURN_TYPE = pa.list_(pa.list_(pa.string()))


# --------------------------------------------------------------------------- #
# hub interrogation
# --------------------------------------------------------------------------- #
def _get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "odiaeval-task1/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hub_config_names(repo_id: str, timeout: float) -> list[str]:
    """The full list of configs the dataset currently exposes."""
    slug = repo_id.replace("/", "%2F")
    data = _get_json(f"{_DATASETS_SERVER}/splits?dataset={slug}", timeout)
    return sorted({s["config"] for s in data["splits"]})


def source_url(repo_id: str, parquet_revision: str, config: str, template: str) -> str:
    rel = template.format(config=config)
    return f"{_HF}/datasets/{repo_id}/resolve/{parquet_revision}/{rel}"


# --------------------------------------------------------------------------- #
# assertions - loud, per CLAUDE.md section 5 rule 8
# --------------------------------------------------------------------------- #
def assert_configs_available(
    available: list[str], wanted: list[str], expected: list[str]
) -> list[str]:
    """Hard-fail if a wanted config vanished; warn if the wider set drifted."""
    missing = [c for c in wanted if c not in available]
    if missing:
        raise SystemExit(
            f"FATAL: requested config(s) {missing} are not present on the Hub. "
            f"Available: {available}. Not substituting anything (CLAUDE.md section 4)."
        )
    drift_added = sorted(set(available) - set(expected))
    drift_gone = sorted(set(expected) - set(available))
    if drift_added or drift_gone:
        print(
            f"  WARNING: config list drifted from the pinned expectation "
            f"(added={drift_added}, removed={drift_gone}). Not fatal - the three "
            f"configs we need are present - but record this.",
            file=sys.stderr,
        )
    return drift_added + drift_gone


def assert_schema(schema: pa.Schema, config: str, keep_columns: list[str]) -> None:
    """Fail loudly if the column set or the List[List[string]] shape is not pinned-correct."""
    missing = [c for c in keep_columns if c not in schema.names]
    if missing:
        raise SystemExit(f"FATAL [{config}]: required columns missing from source: {missing}")

    if not pa.types.is_string(schema.field("doc_id").type):
        raise SystemExit(
            f"FATAL [{config}]: doc_id is {schema.field('doc_id').type}, expected string"
        )
    num_turns_type = schema.field("num_turns").type
    if not (pa.types.is_floating(num_turns_type) or pa.types.is_integer(num_turns_type)):
        raise SystemExit(
            f"FATAL [{config}]: num_turns is {num_turns_type}, expected numeric"
        )

    for col in ("eng_Latn", "ory_Orya"):
        actual = schema.field(col).type
        if not actual.equals(_EXPECTED_TURN_TYPE):
            raise SystemExit(
                f"FATAL [{config}]: {col} has type {actual}, expected "
                f"{_EXPECTED_TURN_TYPE} (outer list of turns, each [prompt, response]). "
                f"The alignment assumption in CLAUDE.md section 5 rule 5 no longer holds."
            )


def assert_row_count(config: str, actual: int, expected: int | None) -> str | None:
    """Row-count drift is a WARNING, not a crash (spec Stage 1) - upstream may change."""
    if expected is None or actual == expected:
        return None
    msg = (
        f"row count for {config} is {actual}, pinned expectation was {expected} "
        f"(drift {actual - expected:+d})"
    )
    print(f"  WARNING: {msg}", file=sys.stderr)
    return msg


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _open_remote(url: str, timeout: float, retries: int, backoff: float) -> pq.ParquetFile:
    import fsspec

    last: Exception | None = None
    for attempt in range(retries):
        try:
            fs = fsspec.filesystem("http", client_kwargs={"trust_env": True}, timeout=timeout)
            return pq.ParquetFile(fs.open(url, "rb"))
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            wait = backoff * (2**attempt)
            print(
                f"  open attempt {attempt + 1}/{retries} failed ({exc}); retrying in {wait:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"could not open {url} after {retries} attempts") from last


def _download_full(url: str, dest: Path, timeout: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  cached source parquet: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading full parquet -> {dest} (this is the large path)")
    req = urllib.request.Request(url, headers={"User-Agent": "odiaeval-task1/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 22)
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------- #
# per-config acquisition
# --------------------------------------------------------------------------- #
def acquire_config(cfg: dict, paths: Paths, config: str, *, mode: str, force: bool) -> dict:
    ds = cfg["dataset"]
    acq = cfg["acquire"]
    keep = list(ds["keep_columns"])
    out = paths.raw / f"{config}.parquet"
    shard_dir = paths.raw / "_partial" / config

    if out.exists() and not force:
        n_rows = pq.ParquetFile(out).metadata.num_rows
        print(
            f"[{config}] cache hit: {out.name} ({n_rows:,} rows, "
            f"{out.stat().st_size / 1e6:.1f} MB) - no network I/O"
        )
        return {
            "config": config,
            "rows": n_rows,
            "cached": True,
            "output": out.name,
            "output_sha256": file_sha256(out),
            "output_bytes": out.stat().st_size,
        }

    url = source_url(ds["repo_id"], ds["parquet_revision"], config, acq["parquet_path_template"])
    print(f"[{config}] mode={mode}")

    if mode == "download":
        local = _download_full(url, paths.raw / "hf" / f"{config}.parquet", acq["http_timeout_s"])
        pf = pq.ParquetFile(local)
        source_bytes = local.stat().st_size
    else:
        pf = _open_remote(url, acq["http_timeout_s"], acq["max_retries"], acq["retry_backoff_s"])
        source_bytes = None

    md = pf.metadata
    full_schema = pf.schema_arrow
    assert_schema(full_schema, config, keep)
    drift_msg = assert_row_count(config, md.num_rows, ds["expected_rows"].get(config))

    print(
        f"  source: {md.num_rows:,} rows, {md.num_row_groups} row groups, "
        f"{len(full_schema)} columns"
    )

    # --- resume-safe shard pass ---------------------------------------------- #
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem[2:]) for p in shard_dir.glob("rg*.parquet")}
    if done:
        print(f"  resuming: {len(done)}/{md.num_row_groups} row groups already staged")

    t0 = time.time()
    for i in range(md.num_row_groups):
        if i in done:
            continue
        tbl = pf.read_row_group(i, columns=keep)
        tmp = shard_dir / f"rg{i:05d}.parquet.part"
        pq.write_table(
            tbl,
            tmp,
            compression=acq["output_compression"],
            compression_level=acq["output_compression_level"],
        )
        tmp.replace(shard_dir / f"rg{i:05d}.parquet")
        pct = (i + 1) / md.num_row_groups * 100
        print(
            f"\r  row groups: {i + 1}/{md.num_row_groups} ({pct:5.1f}%) "
            f"{time.time() - t0:6.0f}s",
            end="",
            flush=True,
        )
    print()

    # --- concatenate shards into the single cached output --------------------- #
    shards = sorted(shard_dir.glob("rg*.parquet"))
    if len(shards) != md.num_row_groups:
        raise SystemExit(
            f"FATAL [{config}]: staged {len(shards)} shards but source has "
            f"{md.num_row_groups} row groups - refusing to write a partial cache."
        )

    target_schema = pa.schema([full_schema.field(c) for c in keep])
    tmp_out = out.with_suffix(".parquet.part")
    rows_written = 0
    writer = pq.ParquetWriter(
        tmp_out,
        target_schema,
        compression=acq["output_compression"],
        compression_level=acq["output_compression_level"],
    )
    try:
        for shard in shards:
            t = pq.read_table(shard, schema=target_schema)
            for batch in t.to_batches(max_chunksize=acq["output_row_group_size"]):
                writer.write_table(pa.Table.from_batches([batch], schema=target_schema))
                rows_written += batch.num_rows
    finally:
        writer.close()
    tmp_out.replace(out)
    shutil.rmtree(shard_dir, ignore_errors=True)

    if rows_written != md.num_rows:
        raise SystemExit(
            f"FATAL [{config}]: wrote {rows_written} rows but source had {md.num_rows}. "
            f"A row disappeared without a recorded reason (CLAUDE.md section 5 rule 7)."
        )

    print(
        f"  wrote data/raw/{out.name} - {rows_written:,} rows, "
        f"{out.stat().st_size / 1e6:.1f} MB (pruned to {keep})"
    )

    return {
        "config": config,
        "rows": md.num_rows,
        "cached": False,
        "mode": mode,
        "source_row_groups": md.num_row_groups,
        "source_columns": list(full_schema.names),
        "source_n_columns": len(full_schema),
        "has_index_level_0": "__index_level_0__" in full_schema.names,
        "source_bytes": source_bytes,
        "row_count_drift": drift_msg,
        "output": out.name,
        "output_sha256": file_sha256(out),
        "output_bytes": out.stat().st_size,
        "seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# alignment audit - validates the assumptions Stage 2 is about to rely on
# --------------------------------------------------------------------------- #
def audit_alignment(paths: Paths, configs: list[str]) -> dict:
    """Check the structural assumptions before Stage 2 depends on them.

    Uses pyarrow compute over turn-list *lengths* so the prompt text is never
    materialised - peak memory stays at one row group.

    Checks:
      * ``doc_id`` unique within each config, and ``example_id`` unique globally
        (Stage 2 asserts this; if it ever fails the id scheme needs a fallback);
      * ``len(eng_Latn) == len(ory_Orya)`` per row - a mismatch breaks the
        alignment assumption and the row must be dropped and counted, not patched;
      * the observed ``num_turns`` distribution.
    """
    seen_eids: dict[str, str] = {}
    per_config: dict[str, dict] = {}
    total_mismatch = 0
    total_collisions = 0

    for config in configs:
        pf = pq.ParquetFile(paths.raw / f"{config}.parquet")
        doc_ids: list[str] = []
        turn_hist: dict[int, int] = {}
        mismatches = 0

        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i, columns=["doc_id", "num_turns", "eng_Latn", "ory_Orya"])
            doc_ids.extend(t.column("doc_id").to_pylist())
            n_eng = pc.list_value_length(t.column("eng_Latn").combine_chunks())
            n_ory = pc.list_value_length(t.column("ory_Orya").combine_chunks())
            mismatches += pc.sum(pc.not_equal(n_eng, n_ory)).as_py() or 0
            for v in t.column("num_turns").to_pylist():
                k = int(v) if v is not None else -1
                turn_hist[k] = turn_hist.get(k, 0) + 1

        dup_doc_ids = len(doc_ids) - len(set(doc_ids))
        collisions = 0
        for d in doc_ids:
            eid = example_id(config, d)
            prior = seen_eids.get(eid)
            if prior is not None:
                collisions += 1
                print(
                    f"  COLLISION: example_id {eid} produced by both "
                    f"{prior} and {config}|{d}",
                    file=sys.stderr,
                )
            seen_eids[eid] = f"{config}|{d}"

        total_mismatch += mismatches
        total_collisions += collisions
        per_config[config] = {
            "rows": len(doc_ids),
            "unique_doc_ids": len(set(doc_ids)),
            "duplicate_doc_ids": dup_doc_ids,
            "turn_count_mismatches": mismatches,
            "num_turns_histogram": {str(k): v for k, v in sorted(turn_hist.items())},
            "example_id_collisions": collisions,
        }

    print()
    print("=" * 78)
    print("ALIGNMENT AUDIT (assumptions Stage 2 relies on)")
    print("-" * 78)
    print(f"{'config':<16}{'rows':>9}{'dup doc_id':>12}{'turn mismatch':>15}{'num_turns':>18}")
    for config, a in per_config.items():
        hist = " ".join(f"{k}:{v:,}" for k, v in a["num_turns_histogram"].items())
        print(
            f"{config:<16}{a['rows']:>9,}{a['duplicate_doc_ids']:>12,}"
            f"{a['turn_count_mismatches']:>15,}{hist:>18}"
        )
    print("-" * 78)
    print(
        f"example_id unique across corpus: {len(seen_eids):,} ids for "
        f"{sum(a['rows'] for a in per_config.values()):,} rows "
        f"({total_collisions} collisions)"
    )
    print(f"turn-count mismatches (eng vs ory), all configs: {total_mismatch}")
    print("=" * 78)

    return {
        "per_config": per_config,
        "example_id_collisions_total": total_collisions,
        "turn_count_mismatches_total": total_mismatch,
        "distinct_example_ids": len(seen_eids),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def turn_prompt(cell: Any, idx: int) -> str:
    """turn-idx user prompt, or a marker. Index [1] (the response) is never read."""
    if not cell or len(cell) <= idx:
        return "<MISSING TURN>"
    turn = cell[idx]
    if not turn:
        return "<EMPTY TURN>"
    return turn[0]


def print_examples(paths: Paths, config: str, n: int = 5) -> None:
    rows = pq.ParquetFile(paths.raw / f"{config}.parquet").read_row_group(
        0, columns=["doc_id", "num_turns", "eng_Latn", "ory_Orya"]
    ).slice(0, n).to_pylist()
    print(f"\n--- {config}: first {n} rows, turn-0 prompts side by side ---")
    for r in rows:
        n_eng = len(r["eng_Latn"] or [])
        n_ory = len(r["ory_Orya"] or [])
        flag = "" if n_eng == n_ory else "   <<< TURN-COUNT MISMATCH"
        print(f"\n  doc_id    : {r['doc_id']}")
        print(f"  num_turns : {r['num_turns']}  (turns: eng={n_eng}, ory={n_ory}){flag}")
        print(f"  eng_Latn  : {turn_prompt(r['eng_Latn'], 0)}")
        print(f"  ory_Orya  : {turn_prompt(r['ory_Orya'], 0)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--only", default=None, help="acquire a single dataset config")
    parser.add_argument("--mode", choices=["stream", "download"], default=None)
    parser.add_argument("--force", action="store_true", help="ignore the cache and refetch")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Odia text in the report
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    ds = cfg["dataset"]
    mode = args.mode or cfg["acquire"]["mode"]
    wanted = [args.only] if args.only else list(ds["configs"])

    print("=" * 78)
    print("STAGE 1 - ACQUIRE")
    print(f"  repo     : {ds['repo_id']}")
    print(f"  revision : {ds['parquet_revision']}  (refs/convert/parquet, pinned)")
    print(f"  columns  : {ds['keep_columns']}")
    print("=" * 78)

    available = hub_config_names(ds["repo_id"], cfg["acquire"]["http_timeout_s"])
    assert_configs_available(available, wanted, ds["expected_configs"])
    print(f"  hub configs ({len(available)}): {available}\n")

    per_config = [acquire_config(cfg, paths, c, mode=mode, force=args.force) for c in wanted]

    total = sum(r["rows"] for r in per_config)
    print("\n" + "=" * 78)
    print(f"{'config':<16}{'rows':>10}{'src cols':>10}{'idx_lvl_0':>11}{'cached MB':>11}")
    print("-" * 78)
    for r in per_config:
        print(
            f"{r['config']:<16}{r['rows']:>10,}"
            f"{str(r.get('source_n_columns', '-')):>10}"
            f"{str(r.get('has_index_level_0', '-')):>11}"
            f"{r['output_bytes'] / 1e6:>11.1f}"
        )
    print("-" * 78)
    cached_mb = sum(r["output_bytes"] for r in per_config) / 1e6
    print(f"{'TOTAL':<16}{total:>10,}{'':>10}{'':>11}{cached_mb:>11.1f}")
    print("=" * 78)

    audit = audit_alignment(paths, wanted)

    append_stage_stats(
        paths.artifacts,
        STAGE,
        rows_in=total,
        rows_out=total,
        dropped_by_reason={},
        extra={
            "mode": mode,
            "hub_configs": available,
            "per_config": per_config,
            "alignment_audit": audit,
        },
    )

    for c in wanted:
        print_examples(paths, c, args.examples)

    print(f"\nStage 1 complete. {total:,} rows cached under data/raw/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
