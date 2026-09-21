"""Stage 7 tests: manifest completeness, schema conformance, variant equivalence.

These run against the frozen releases if they exist, and skip cleanly if the
pipeline has not been run — so the suite stays usable on a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.report import RELEASE_SCHEMA, SPLITS, VARIANTS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# PROMPT.md §7 contract, plus the raw probability. Frozen here so a rename in
# src/report.py cannot silently redefine the interface the modelling half uses.
CONTRACT_COLUMNS = [
    "example_id", "doc_id", "source_config", "text_ory", "text_eng",
    "label", "label_str", "teacher_prob_hate", "teacher_prob_hate_raw",
    "dedup_group", "split", "num_turns", "ory_script_purity",
]

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "odia_hate_v1_proportional" / "train.parquet").exists(),
    reason="releases not built; run reproduce.sh",
)


def _release(variant: str, split: str) -> pa.Table:
    return pq.read_table(PROCESSED / f"odia_hate_v1_{variant}" / f"{split}.parquet")


class TestSchemaConformance:
    def test_declared_schema_matches_the_contract_exactly(self) -> None:
        assert [f.name for f in RELEASE_SCHEMA] == CONTRACT_COLUMNS

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("split", SPLITS)
    def test_every_release_file_conforms(self, variant: str, split: str) -> None:
        tbl = _release(variant, split)
        assert tbl.schema.equals(RELEASE_SCHEMA), f"{variant}/{split} schema drifted"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_jsonl_mirrors_the_parquet(self, variant: str) -> None:
        for split in SPLITS:
            tbl = _release(variant, split)
            jl = PROCESSED / f"odia_hate_v1_{variant}" / f"{split}.jsonl"
            lines = jl.read_text(encoding="utf-8").splitlines()
            assert len(lines) == tbl.num_rows
            first = json.loads(lines[0])
            assert set(first) == set(CONTRACT_COLUMNS)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_no_assistant_response_column_leaked_in(self, variant: str) -> None:
        names = [f.name for f in _release(variant, "train").schema]
        assert not any(
            k in n.lower() for n in names for k in ("response", "answer", "completion")
        )


class TestReleaseContent:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_split_column_matches_the_file(self, variant: str) -> None:
        for split in SPLITS:
            col = set(_release(variant, split).column("split").to_pylist())
            assert col == {split}

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_labels_are_consistent_and_balanced(self, variant: str) -> None:
        labels, strs = [], []
        for split in SPLITS:
            t = _release(variant, split)
            labels += t.column("label").to_pylist()
            strs += t.column("label_str").to_pylist()
        assert set(labels) == {0, 1}
        for lab, s in zip(labels, strs):
            assert s == ("HATE" if lab == 1 else "NON_HATE")
        assert labels.count(1) == labels.count(0)  # exact 50/50

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_thresholds_hold_for_every_row(self, variant: str) -> None:
        """Every shipped row must sit in the band its label claims."""
        for split in SPLITS:
            t = _release(variant, split)
            for p, lab in zip(t.column("teacher_prob_hate").to_pylist(),
                              t.column("label").to_pylist()):
                if lab == 1:
                    assert p >= 0.90 - 1e-6, f"HATE row with p={p}"
                else:
                    assert p <= 0.05 + 1e-6, f"NON_HATE row with p={p}"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_example_ids_unique_across_all_splits(self, variant: str) -> None:
        ids = [e for s in SPLITS for e in _release(variant, s).column("example_id").to_pylist()]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_no_dedup_group_spans_two_splits(self, variant: str) -> None:
        """The leakage guarantee, re-checked on the frozen artefact itself."""
        seen: dict[str, str] = {}
        for split in SPLITS:
            for g in _release(variant, split).column("dedup_group").to_pylist():
                assert seen.setdefault(g, split) == split, f"{g} spans splits"


class TestVariantEquivalence:
    """The two releases must differ ONLY in which NON_HATE rows were kept."""

    def _rows(self, variant: str) -> dict[str, dict]:
        out = {}
        for split in SPLITS:
            for r in _release(variant, split).to_pylist():
                out[r["example_id"]] = r

        return out

    def test_hate_rows_are_identical_including_split(self) -> None:
        a, b = self._rows("proportional"), self._rows("match_minority")
        ha = {k: v for k, v in a.items() if v["label"] == 1}
        hb = {k: v for k, v in b.items() if v["label"] == 1}
        assert set(ha) == set(hb)
        for k in ha:
            assert ha[k] == hb[k], f"HATE row {k} differs between variants"

    def test_non_hate_selection_differs(self) -> None:
        a, b = self._rows("proportional"), self._rows("match_minority")
        na = {k for k, v in a.items() if v["label"] == 0}
        nb = {k for k, v in b.items() if v["label"] == 0}
        assert na != nb

    def test_shared_non_hate_rows_are_byte_identical(self) -> None:
        a, b = self._rows("proportional"), self._rows("match_minority")
        na = {k for k, v in a.items() if v["label"] == 0}
        nb = {k for k, v in b.items() if v["label"] == 0}
        shared = na & nb
        assert shared, "variants share no NON_HATE rows at all"
        for k in shared:
            assert a[k] == b[k], f"shared NON_HATE row {k} differs"

    def test_both_variants_have_the_same_row_count(self) -> None:
        a, b = self._rows("proportional"), self._rows("match_minority")
        assert len(a) == len(b)


class TestManifestCompleteness:
    @pytest.fixture(scope="class")
    @classmethod
    def manifest(cls) -> dict:
        return json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))

    def test_top_level_fields_present(self, manifest: dict) -> None:
        for key in ("build_timestamp_utc", "git_commit", "config_sha256", "seed",
                    "pinned_artefacts", "scoring_pass", "release_schema", "releases"):
            assert manifest.get(key) is not None, f"manifest missing {key}"

    def test_every_pinned_artefact_has_a_resolved_revision(self, manifest: dict) -> None:
        pinned = manifest["pinned_artefacts"]
        for role in ("dataset", "dataset_parquet_convert", "teacher"):
            info = pinned.get(role, {})
            assert len(info.get("revision", "")) == 40, f"{role} revision not a full SHA"
            assert info.get("repo_id")

    def test_scoring_pass_records_the_reproduction_recipe(self, manifest: dict) -> None:
        sp = manifest["scoring_pass"]
        for key in ("device", "dtype_forward", "softmax_dtype",
                    "effective_batch_size", "prob_decimals", "torch_version"):
            assert sp.get(key) is not None, f"scoring_pass missing {key}"
        assert sp["softmax_dtype"] == "float32"

    def test_every_release_file_is_hashed_with_a_row_count(self, manifest: dict) -> None:
        for variant in VARIANTS:
            rel = manifest["releases"][variant]
            assert rel["rows"] > 0
            for split in SPLITS:
                for ext in ("parquet", "jsonl"):
                    key = f"{variant}/{split}.{ext}"
                    entry = rel["files"][key]
                    assert len(entry["sha256"]) == 64
                    assert entry["rows"] > 0 and entry["bytes"] > 0

    def test_recorded_hashes_match_the_files_on_disk(self, manifest: dict) -> None:
        from src._common import file_sha256

        for variant in VARIANTS:
            for key, entry in manifest["releases"][variant]["files"].items():
                path = PROCESSED / f"odia_hate_v1_{key.split('/')[0]}" / key.split("/")[1]
                assert file_sha256(path) == entry["sha256"], f"{key} changed since freeze"

    def test_manifest_schema_matches_the_contract(self, manifest: dict) -> None:
        assert manifest["release_schema"] == CONTRACT_COLUMNS


class TestRowAccounting:
    def test_every_raw_row_is_kept_or_dropped_with_a_named_reason(self) -> None:
        stats = json.loads((ARTIFACTS / "stage_stats.json").read_text(encoding="utf-8"))
        by = {s["stage"]: s for s in stats["stages"]}
        raw = by["01_acquire"]["rows_out"]
        assert raw == 138_032

        chain = ["02_extract", "04_filter", "05_dedup"]
        current = raw
        for stage in chain:
            st = by[stage]
            assert st["rows_in"] == current, f"{stage} input does not match previous output"
            dropped = sum(st["dropped_by_reason"].values())
            assert st["rows_in"] - st["rows_out"] == dropped, (
                f"{stage}: {st['rows_in'] - st['rows_out']} rows lost but {dropped} "
                "accounted for by reason"
            )
            current = st["rows_out"]

        for variant in VARIANTS:
            st = by[f"06_balance_split_{variant}"]
            assert st["rows_in"] == current
            assert sum(st["dropped_by_reason"].values()) == st["rows_in"] - st["rows_out"]
