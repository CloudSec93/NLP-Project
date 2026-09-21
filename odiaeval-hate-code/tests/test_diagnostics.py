"""Stage 7A tests: the surface-feature diagnostic really is surface-only.

The whole interpretation of this baseline rests on the model having no access to
meaning. If a lexical signal leaked into the features, the number would stop
being a shortcut floor and become meaningless. These tests defend that property.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.diagnostics import FEATURE_NAMES, featurise

ODIA_A = "ଆପଣ କେମିତି ଅଛନ୍ତି"
ODIA_B = "ତୁମେ କଣ କରୁଛ ଏବେ"


def _row(text, purity=1.0):
    return {"text_ory": text, "ory_script_purity": purity}


class TestFeatureShape:
    def test_one_column_per_named_feature(self) -> None:
        X = featurise([_row(ODIA_A)])
        assert X.shape == (1, len(FEATURE_NAMES))

    def test_no_nans_on_degenerate_input(self) -> None:
        X = featurise([_row(""), _row("   "), _row("???"), _row("୧୨୩")])
        assert np.isfinite(X).all()

    def test_empty_text_has_zero_mean_word_length(self) -> None:
        X = featurise([_row("")])
        assert X[0, FEATURE_NAMES.index("mean_word_length")] == 0.0


class TestSurfaceOnly:
    def test_features_cannot_distinguish_different_words_of_the_same_shape(self) -> None:
        """The core guarantee: no lexical content reaches the model.

        Two completely different Odia strings with identical surface statistics
        must produce identical feature vectors.
        """
        a = "ଅଆଇଈ ଉଊଋଌ ଏଐଓଔ"
        b = "କଖଗଘ ଙଚଛଜ ଝଞଟଠ"
        assert a != b
        Xa, Xb = featurise([_row(a)]), featurise([_row(b)])
        np.testing.assert_array_equal(Xa, Xb)

    def test_no_feature_name_implies_lexical_content(self) -> None:
        """Naming hygiene. The real guarantee is the invariance test above.

        Note "mean_word_length" is deliberately allowed: it is the average token
        length in characters, an aggregate that carries no lexical identity.
        What is banned is anything that could encode *which* words are present.
        """
        banned = (
            "ngram", "unigram", "bigram", "token_id", "vocab", "tfidf",
            "embed", "bow", "lemma", "term_", "word_is", "contains_word",
        )
        for name in FEATURE_NAMES:
            assert not any(b in name for b in banned), name

    def test_feature_set_is_the_pinned_eleven(self) -> None:
        assert FEATURE_NAMES == [
            "char_length", "token_count", "mean_word_length",
            "count_question_mark", "count_exclamation", "count_period",
            "count_comma", "count_digit", "count_uppercase",
            "ory_script_purity", "has_chat_scaffolding",
        ]


class TestFeatureValues:
    def test_counts_are_correct(self) -> None:
        X = featurise([_row("ଆ? ଆ! ଆ. ଆ, 12 AB")])[0]
        idx = FEATURE_NAMES.index
        assert X[idx("count_question_mark")] == 1
        assert X[idx("count_exclamation")] == 1
        assert X[idx("count_period")] == 1
        assert X[idx("count_comma")] == 1
        assert X[idx("count_digit")] == 2
        assert X[idx("count_uppercase")] == 2

    def test_length_and_token_count(self) -> None:
        X = featurise([_row(ODIA_A)])[0]
        assert X[FEATURE_NAMES.index("char_length")] == len(ODIA_A)
        assert X[FEATURE_NAMES.index("token_count")] == len(ODIA_A.split())

    def test_purity_column_is_passed_through(self) -> None:
        X = featurise([_row(ODIA_B, purity=0.73)])[0]
        assert X[FEATURE_NAMES.index("ory_script_purity")] == pytest.approx(0.73)

    def test_scaffolding_flag_fires_on_the_markers(self) -> None:
        idx = FEATURE_NAMES.index("has_chat_scaffolding")
        assert featurise([_row("<s>[INST] ଆପଣ")])[0][idx] == 1.0
        assert featurise([_row("<<SYS>> ଆପଣ")])[0][idx] == 1.0
        assert featurise([_row(ODIA_A)])[0][idx] == 0.0


class TestTestSplitIsSealed:
    def test_module_source_never_selects_the_test_split(self) -> None:
        """A diagnostic number in the data card must not have touched test."""
        import inspect

        import src.diagnostics as d

        src = inspect.getsource(d)
        assert 'r["split"] == "train"' in src
        assert 'r["split"] == "validation"' in src
        assert 'r["split"] == "test"' not in src
