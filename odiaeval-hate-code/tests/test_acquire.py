"""Stage 1 tests: source-schema assertions and turn extraction (spec §8).

Real assertions on a real (tiny) pyarrow fixture — including a multi-turn row
and a row whose English/Odia turn counts disagree.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from src.acquire import (
    assert_configs_available,
    assert_row_count,
    assert_schema,
    source_url,
    turn_prompt,
)

KEEP = ["doc_id", "num_turns", "eng_Latn", "ory_Orya"]

# One single-turn row, one multi-turn row, one row with a turn-count mismatch.
FIXTURE = pa.table(
    {
        "doc_id": pa.array(["d1", "d2", "d3"], pa.string()),
        "num_turns": pa.array([1.0, 3.0, 2.0], pa.float64()),
        "eng_Latn": pa.array(
            [
                [["Why do camels survive without water?", "Because of fat stores."]],
                [
                    ["First question?", "First answer."],
                    ["Second question?", "Second answer."],
                    ["Third question?", "Third answer."],
                ],
                [["Only English turn?", "Yes."], ["Second English turn?", "Yes."]],
            ],
            pa.list_(pa.list_(pa.string())),
        ),
        "ory_Orya": pa.array(
            [
                [["ପାଣି ବିନା କାହିଁକି ଊଁଟ ବଞ୍ଚିପାରିବ?", "ଚର୍ବି ଯୋଗୁଁ।"]],
                [
                    ["ପ୍ରଥମ ପ୍ରଶ୍ନ?", "ପ୍ରଥମ ଉତ୍ତର।"],
                    ["ଦ୍ୱିତୀଯ଼ ପ୍ରଶ୍ନ?", "ଦ୍ୱିତୀଯ଼ ଉତ୍ତର।"],
                    ["ତୃତୀଯ଼ ପ୍ରଶ୍ନ?", "ତୃତୀଯ଼ ଉତ୍ତର।"],
                ],
                [["ଏକମାତ୍ର ଓଡ଼ିଆ ଟର୍ନ?", "ହଁ।"]],  # only 1 turn vs 2 English -> mismatch
            ],
            pa.list_(pa.list_(pa.string())),
        ),
    }
)


class TestSchemaAssertion:
    def test_valid_schema_passes(self) -> None:
        assert_schema(FIXTURE.schema, "Fixture", KEEP)  # must not raise

    def test_missing_column_is_fatal(self) -> None:
        dropped = FIXTURE.drop_columns(["ory_Orya"])
        with pytest.raises(SystemExit, match="required columns missing"):
            assert_schema(dropped.schema, "Fixture", KEEP)

    def test_flat_list_of_strings_is_rejected(self) -> None:
        """A List[string] column means the [prompt, response] pairing is gone."""
        bad = pa.table(
            {
                "doc_id": FIXTURE["doc_id"],
                "num_turns": FIXTURE["num_turns"],
                "eng_Latn": pa.array([["a"], ["b"], ["c"]], pa.list_(pa.string())),
                "ory_Orya": FIXTURE["ory_Orya"],
            }
        )
        with pytest.raises(SystemExit, match="expected list<item: list<item: string>>"):
            assert_schema(bad.schema, "Fixture", KEEP)

    def test_wrong_doc_id_type_is_fatal(self) -> None:
        bad = FIXTURE.set_column(
            0, "doc_id", pa.array([1, 2, 3], pa.int64())
        )
        with pytest.raises(SystemExit, match="doc_id is"):
            assert_schema(bad.schema, "Fixture", KEEP)

    def test_integer_num_turns_is_accepted(self) -> None:
        ok = FIXTURE.set_column(1, "num_turns", pa.array([1, 3, 2], pa.int16()))
        assert_schema(ok.schema, "Fixture", KEEP)  # numeric is numeric


class TestTurnExtraction:
    def test_single_turn_row(self) -> None:
        rows = FIXTURE.to_pylist()
        assert turn_prompt(rows[0]["eng_Latn"], 0).startswith("Why do camels")
        assert "ଊଁଟ" in turn_prompt(rows[0]["ory_Orya"], 0)

    def test_multi_turn_row_takes_turn_zero_only(self) -> None:
        rows = FIXTURE.to_pylist()
        assert turn_prompt(rows[1]["eng_Latn"], 0) == "First question?"
        assert turn_prompt(rows[1]["ory_Orya"], 0) == "ପ୍ରଥମ ପ୍ରଶ୍ନ?"

    def test_never_returns_the_assistant_response(self) -> None:
        """Index [1] must never surface (CLAUDE.md §5.5)."""
        rows = FIXTURE.to_pylist()
        for r in rows:
            for col in ("eng_Latn", "ory_Orya"):
                for i in range(len(r[col] or [])):
                    got = turn_prompt(r[col], i)
                    assert got == r[col][i][0]
                    assert got != r[col][i][1]

    def test_missing_and_empty_turns_are_marked_not_silently_coerced(self) -> None:
        assert turn_prompt([], 0) == "<MISSING TURN>"
        assert turn_prompt(None, 0) == "<MISSING TURN>"
        assert turn_prompt([["a", "b"]], 5) == "<MISSING TURN>"
        assert turn_prompt([[]], 0) == "<EMPTY TURN>"

    def test_turn_count_mismatch_is_detectable(self) -> None:
        """Row d3 has 2 English turns and 1 Odia turn — Stage 2 must drop it."""
        rows = FIXTURE.to_pylist()
        r = rows[2]
        assert len(r["eng_Latn"]) != len(r["ory_Orya"])


class TestHubAssertions:
    AVAILABLE = ["Anudesh", "Dolly_T", "HHRLHF_T", "Toxic_Matrix"]

    def test_missing_wanted_config_is_fatal(self) -> None:
        with pytest.raises(SystemExit, match="not present on the Hub"):
            assert_configs_available(self.AVAILABLE, ["Nonexistent_T"], self.AVAILABLE)

    def test_config_drift_warns_but_does_not_raise(self) -> None:
        drift = assert_configs_available(
            self.AVAILABLE, ["Dolly_T"], ["Anudesh", "Dolly_T", "HHRLHF_T"]
        )
        assert "Toxic_Matrix" in drift  # reported, not fatal

    def test_row_count_match_is_silent(self) -> None:
        assert assert_row_count("Dolly_T", 15011, 15011) is None

    def test_row_count_drift_returns_a_message(self) -> None:
        msg = assert_row_count("Dolly_T", 15000, 15011)
        assert msg is not None and "-11" in msg

    def test_unknown_expectation_is_not_drift(self) -> None:
        assert assert_row_count("New_Config", 123, None) is None


def test_source_url_pins_the_convert_revision() -> None:
    url = source_url(
        "ai4bharat/indic-align", "40d2d14e", "Toxic_Matrix", "{config}/train/0000.parquet"
    )
    assert url == (
        "https://huggingface.co/datasets/ai4bharat/indic-align/resolve/"
        "40d2d14e/Toxic_Matrix/train/0000.parquet"
    )
