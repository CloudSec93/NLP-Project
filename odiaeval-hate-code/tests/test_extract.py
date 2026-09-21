"""Stage 2 tests: turn-0 extraction and the quality gates.

Real assertions on real fixtures — including a multi-turn row and a row whose
English/Odia turn counts disagree (which must be dropped and counted, not patched).
"""

from __future__ import annotations

import pytest

from src.extract import PAIRS_SCHEMA, evaluate_gates, turn0_prompt

ODIA = "ଆପଣ କେମିତି ଅଛନ୍ତି ଏହା ଏକ ପ୍ରଶ୍ନ ଅଟେ"
GATES = {"turn_index": 0, "min_chars": 3, "max_chars": 2000, "min_purity": 0.60}

# Response text that must never appear anywhere in the output.
RESPONSE_MARKER = "ASSISTANT_RESPONSE_MUST_NEVER_LEAK"


class TestTurnAccess:
    def test_single_turn(self) -> None:
        assert turn0_prompt([["prompt", RESPONSE_MARKER]], 0) == "prompt"

    def test_multi_turn_takes_the_requested_turn_only(self) -> None:
        cell = [
            ["first prompt", RESPONSE_MARKER],
            ["second prompt", RESPONSE_MARKER],
            ["third prompt", RESPONSE_MARKER],
        ]
        assert turn0_prompt(cell, 0) == "first prompt"
        assert turn0_prompt(cell, 1) == "second prompt"
        assert turn0_prompt(cell, 2) == "third prompt"

    def test_never_returns_index_one(self) -> None:
        """CLAUDE.md §5.5 — the assistant response never enters the dataset."""
        cell = [["p0", RESPONSE_MARKER], ["p1", RESPONSE_MARKER]]
        for i in range(3):
            assert turn0_prompt(cell, i) != RESPONSE_MARKER

    def test_absent_or_ragged_turns_return_none_not_a_guess(self) -> None:
        assert turn0_prompt(None, 0) is None
        assert turn0_prompt([], 0) is None
        assert turn0_prompt([["p", "r"]], 5) is None
        assert turn0_prompt([[]], 0) is None


class TestGates:
    def _run(self, eng, ory, **over):
        kw = {**GATES, **over}
        return evaluate_gates(eng, ory, **kw)

    def test_clean_row_passes(self) -> None:
        primary, _, failed, te, to, purity = self._run(
            [["Is this a well formed question?", RESPONSE_MARKER]], [[ODIA, RESPONSE_MARKER]]
        )
        assert primary is None and failed == []
        assert te == "Is this a well formed question?"
        assert to == ODIA
        assert purity == pytest.approx(1.0)

    def test_turn_count_mismatch_is_dropped_and_named(self) -> None:
        """A ragged row breaks the alignment assumption — drop it, count it."""
        primary, detail, failed, *_ = self._run(
            [["a prompt", "r"], ["another prompt", "r"]], [[ODIA, "r"]]
        )
        assert primary == "turn_shape_invalid"
        assert "turn count mismatch" in detail
        assert "eng=2" in detail and "ory=1" in detail

    def test_missing_turn_zero_is_shape_invalid(self) -> None:
        primary, _, _, *_ = self._run([], [])
        assert primary == "turn_shape_invalid"

    def test_empty_after_strip(self) -> None:
        primary, detail, _, *_ = self._run([["   \t \n ", "r"]], [[ODIA, "r"]])
        assert primary == "empty_after_strip"
        assert "eng" in detail

    def test_too_short_and_too_long(self) -> None:
        short, _, _, *_ = self._run([["ab", "r"]], [[ODIA, "r"]])
        assert short == "length_out_of_bounds"
        long_primary, detail, _, *_ = self._run([["x" * 2001, "r"]], [[ODIA, "r"]])
        assert long_primary == "length_out_of_bounds"
        assert "eng>2000" in detail

    def test_length_bound_is_inclusive_at_both_ends(self) -> None:
        assert self._run([["abc", "r"]], [[ODIA, "r"]])[0] is None          # exactly 3
        assert self._run([["x" * 2000, "r"]], [[ODIA, "r"]])[0] is None      # exactly 2000

    def test_untranslated_latin_odia_side_is_caught_by_purity(self) -> None:
        primary, detail, _, _, _, purity = self._run(
            [["a valid english prompt", "r"]], [["this row was never translated", "r"]]
        )
        assert primary == "low_ory_script_purity"
        assert purity == 0.0
        assert "0.000" in detail

    def test_purity_boundary_is_inclusive(self) -> None:
        """6 Odia + 4 Latin letters = exactly 0.60, the configured floor -> passes."""
        boundary = "କଖଗଘଚଛ" + "abcd"
        primary, _, _, _, _, purity = self._run(
            [["a valid english prompt", "r"]], [[boundary, "r"]]
        )
        assert purity == pytest.approx(0.60)
        assert primary is None

    def test_nfc_is_applied_before_gating(self) -> None:
        """Decomposed input is composed before length is measured."""
        decomposed = "é" * 3          # 6 codepoints, 3 after NFC
        _, _, _, te, _, _ = self._run([[decomposed, "r"]], [[ODIA, "r"]])
        assert te == "é" * 3
        assert len(te) == 3


class TestGateOrdering:
    """The FIRST gate failed is the recorded reason, so rows are counted once."""

    def test_shape_beats_everything(self) -> None:
        primary, _, failed, *_ = evaluate_gates(
            [["", "r"], ["x", "r"]], [[""]], **GATES
        )
        assert primary == "turn_shape_invalid"
        assert failed[0] == "turn_shape_invalid"
        assert len(failed) > 1  # co-failures still recorded for diagnostics

    def test_empty_beats_length_and_purity(self) -> None:
        primary, _, failed, *_ = evaluate_gates([["", "r"]], [["", "r"]], **GATES)
        assert primary == "empty_after_strip"
        assert "length_out_of_bounds" in failed
        assert "low_ory_script_purity" in failed

    def test_length_beats_purity(self) -> None:
        primary, _, failed, *_ = evaluate_gates(
            [["ab", "r"]], [["ab", "r"]], **GATES
        )
        assert primary == "length_out_of_bounds"
        assert "low_ory_script_purity" in failed


def test_output_schema_carries_no_response_column() -> None:
    """The release columns are prompts only — nothing derived from index [1]."""
    names = set(PAIRS_SCHEMA.names)
    assert names == {
        "example_id",
        "doc_id",
        "source_config",
        "text_eng",
        "text_ory",
        "num_turns",
        "ory_script_purity",
    }
    assert not any("response" in n.lower() or "answer" in n.lower() for n in names)
