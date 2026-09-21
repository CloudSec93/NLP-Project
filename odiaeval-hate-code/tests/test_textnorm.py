"""Stage 0 tests: text normalisation + Odia script-purity scoring (spec §8)."""

from __future__ import annotations

import pytest

from src.textnorm import (
    collapse_whitespace,
    normalise_for_dedup,
    odia_script_purity,
    strip_zero_width,
)

# "how are you" in Odia — every letter is in U+0B00..U+0B7F
ODIA = "ଆପଣ କେମିତି ଅଛନ୍ତି"
LATIN = "how are you doing today"


class TestScriptPurity:
    def test_pure_odia_scores_one(self) -> None:
        assert odia_script_purity(ODIA) == pytest.approx(1.0)

    def test_pure_latin_scores_zero(self) -> None:
        assert odia_script_purity(LATIN) == 0.0

    def test_empty_and_nonletter_score_zero(self) -> None:
        assert odia_script_purity("") == 0.0
        assert odia_script_purity("   \n\t  ") == 0.0
        assert odia_script_purity("123 !!! ??? …") == 0.0

    def test_mixed_is_ratio_of_letters_only(self) -> None:
        # 3 Odia base letters (ଆ ପ ଣ) + 5 Latin letters (h e l l o) -> 3/8.
        # Combining vowel signs (category Mn/Mc) are not letters and are excluded.
        assert odia_script_purity("ଆପଣ hello") == pytest.approx(3 / 8)

    def test_digits_and_punctuation_do_not_move_score(self) -> None:
        base = odia_script_purity(ODIA)
        assert odia_script_purity(f"{ODIA} 2024!! (...) 50% —— ??") == pytest.approx(base)

    def test_threshold_boundary_case(self) -> None:
        # 6 Odia consonants + 4 Latin letters -> 0.60 exactly, the configured cutoff
        s = "କଖଗଘଚଛ" + "abcd"
        assert odia_script_purity(s) == pytest.approx(0.60)


class TestNormalisation:
    def test_strip_zero_width(self) -> None:
        assert strip_zero_width("a​b‍c﻿") == "abc"

    def test_collapse_whitespace(self) -> None:
        assert collapse_whitespace("  a\t b\n\n  c  ") == "a b c"

    def test_zero_width_stripped_before_collapse_does_not_fuse_words(self) -> None:
        assert normalise_for_dedup("foo​ bar") == "foo bar"

    def test_nfc_applied(self) -> None:
        # decomposed 'é' (e + combining acute) -> composed
        assert normalise_for_dedup("é") == "é"

    def test_idempotent(self) -> None:
        once = normalise_for_dedup(f"  {ODIA}​  extra   ")
        assert normalise_for_dedup(once) == once
