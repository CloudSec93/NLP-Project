"""Stage 0 tests: config loading + the fixed-threshold guard (CLAUDE.md §5.6)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src._common import example_id, load_config


def test_default_config_loads_and_has_fixed_thresholds() -> None:
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["thresholds"]["hate_min"] == 0.90
    assert cfg["thresholds"]["non_hate_max"] == 0.05
    assert cfg["dataset"]["configs"] == ["Toxic_Matrix", "HHRLHF_T", "Dolly_T"]
    assert cfg["dataset"]["keep_columns"] == [
        "doc_id",
        "num_turns",
        "eng_Latn",
        "ory_Orya",
    ]


def test_altered_thresholds_fail_loudly(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            seed: 42
            thresholds:
              hate_min: 0.80
              non_hate_max: 0.05
            paths: {raw: data/raw, interim: data/interim, processed: data/processed,
                    artifacts: artifacts, release_name: x}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixed by the assignment"):
        load_config(bad)


def test_example_id_is_16_hex_and_stable() -> None:
    eid = example_id("Toxic_Matrix", "doc-123")
    assert len(eid) == 16
    assert all(c in "0123456789abcdef" for c in eid)
    assert eid == example_id("Toxic_Matrix", "doc-123")
    assert eid != example_id("HHRLHF_T", "doc-123")
