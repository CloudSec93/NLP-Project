"""Stage 3 tests: teacher orientation assertions, banding, rounding, checkpoint resume.

None of these load the real model — the machinery that must not be silently wrong
is separated from the model precisely so it can be tested without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.label import band_counts, histogram, load_shards, write_shard
from src.teacher import (
    EXPECTED_ID2LABEL,
    assert_teacher_orientation,
    canonical_id2label,
    check_hard_tier,
    check_pinned_tier,
)

HI, LO = 0.90, 0.05


# --------------------------------------------------------------------------- #
# teacher orientation
# --------------------------------------------------------------------------- #
class TestTeacherOrientation:
    def test_string_keys_are_accepted_values_are_not_normalised(self) -> None:
        assert canonical_id2label({"0": "nothate", "1": "hate"}) == EXPECTED_ID2LABEL

    def test_correct_config_passes(self) -> None:
        cfg = SimpleNamespace(num_labels=2, id2label={0: "nothate", 1: "hate"})
        assert assert_teacher_orientation(cfg) == EXPECTED_ID2LABEL

    def test_inverted_mapping_is_fatal(self) -> None:
        """The whole point: an inverted map is invisible downstream."""
        cfg = SimpleNamespace(num_labels=2, id2label={0: "hate", 1: "nothate"})
        with pytest.raises(SystemExit, match="id2label"):
            assert_teacher_orientation(cfg)

    def test_case_change_is_fatal_not_normalised(self) -> None:
        cfg = SimpleNamespace(num_labels=2, id2label={0: "NotHate", 1: "Hate"})
        with pytest.raises(SystemExit, match="id2label"):
            assert_teacher_orientation(cfg)

    def test_wrong_num_labels_is_fatal(self) -> None:
        cfg = SimpleNamespace(num_labels=3, id2label={0: "a", 1: "b", 2: "c"})
        with pytest.raises(SystemExit, match="num_labels"):
            assert_teacher_orientation(cfg)

    def test_non_integer_key_is_fatal(self) -> None:
        with pytest.raises(SystemExit, match="non-integer key"):
            canonical_id2label({"hate": "hate"})


class TestCanaryTiers:
    def _items(self, ps, prefix):
        return [{"id": f"{prefix}{i}", "text": "x", "form": "f", "p_hate": p}
                for i, p in enumerate(ps)]

    def test_correct_orientation_passes(self) -> None:
        r = check_hard_tier(
            self._items([0.999, 0.998, 0.997], "h"), self._items([0.001, 0.002], "b")
        )
        assert r["passed"] and r["rank_separated"] and r["margin"] > 0.5

    def test_inverted_model_shows_a_large_negative_margin(self) -> None:
        r = check_hard_tier(
            self._items([0.001, 0.002], "h"), self._items([0.999, 0.998], "b")
        )
        assert not r["passed"]
        assert r["margin"] < -0.5

    def test_overlapping_scores_fail_rank_separation(self) -> None:
        r = check_hard_tier(
            self._items([0.99, 0.10], "h"), self._items([0.20, 0.01], "b")
        )
        assert not r["rank_separated"]
        assert not r["passed"]

    def test_documented_failure_excluded_from_confidence_floor_only(self) -> None:
        """A known miss must not drag the pinned floor down to meaninglessness."""
        hate = self._items([0.999, 0.998], "h")
        hate.append({"id": "hate_x", "text": "x", "form": "f",
                     "p_hate": 0.0004, "documented_failure": True})
        scored = {
            "HARD_CLEAR_HATE": hate,
            "HARD_CLEAR_BENIGN": self._items([0.001], "b"),
            "PINNED_HARD_NEG": self._items([0.002], "n"),
        }
        r = check_pinned_tier(scored, {"clear_hate_min": 0.99, "clear_benign_max": 0.01,
                                       "hard_negative_max": 0.01})
        assert r["observed"]["clear_hate_min"] == pytest.approx(0.998)
        assert r["passed"]
        assert r["documented_failures"][0]["id"] == "hate_x"

    def test_unpinned_bounds_report_but_never_fail(self) -> None:
        scored = {
            "HARD_CLEAR_HATE": self._items([0.5], "h"),
            "HARD_CLEAR_BENIGN": self._items([0.4], "b"),
            "PINNED_HARD_NEG": self._items([0.9], "n"),
        }
        r = check_pinned_tier(scored, {"clear_hate_min": None, "clear_benign_max": None,
                                       "hard_negative_max": None})
        assert r["passed"] and not r["asserted"]
        assert set(r["unpinned"]) == {"clear_hate_min", "clear_benign_max", "hard_negative_max"}


# --------------------------------------------------------------------------- #
# banding and rounding
# --------------------------------------------------------------------------- #
class TestBanding:
    def test_exact_boundaries_are_inclusive_on_the_decided_side(self) -> None:
        """0.90 -> HATE, 0.05 -> NON_HATE, per the assignment's at-least/at-most."""
        b = band_counts(np.array([0.90, 0.05]), HI, LO)
        assert b["hate_band"] == 1
        assert b["non_hate_band"] == 1
        assert b["discarded_band"] == 0

    def test_just_inside_the_middle_is_discarded(self) -> None:
        b = band_counts(np.array([0.899999, 0.050001]), HI, LO)
        assert b["hate_band"] == 0 and b["non_hate_band"] == 0
        assert b["discarded_band"] == 2

    def test_bands_partition_the_input(self) -> None:
        rng = np.random.default_rng(42)
        p = rng.random(5000)
        b = band_counts(p, HI, LO)
        assert b["hate_band"] + b["non_hate_band"] + b["discarded_band"] == b["total"] == 5000

    def test_rounding_pulls_boundary_values_onto_the_threshold(self) -> None:
        """The reason we threshold on the rounded value (decision A)."""
        raw = np.array([0.8999999, 0.9000001, 0.0499999, 0.0500001], dtype=np.float64)
        rounded = np.round(raw, 6)
        assert rounded[0] == pytest.approx(0.9)
        assert rounded[1] == pytest.approx(0.9)
        b = band_counts(rounded, HI, LO)
        # both 0.899.../0.900... round to 0.9 -> HATE; both 0.049.../0.050... -> 0.05 -> NON_HATE
        assert b["hate_band"] == 2
        assert b["non_hate_band"] == 2

    def test_rounding_does_not_move_values_away_from_the_tails(self) -> None:
        raw = np.array([0.123456789, 0.987654321], dtype=np.float64)
        assert np.round(raw, 6).tolist() == [0.123457, 0.987654]


def test_histogram_bins_partition_and_count_correctly() -> None:
    p = np.array([0.0, 0.005, 0.03, 0.5, 0.93, 0.97, 0.995, 1.0])
    h = histogram(p)
    assert sum(b["count"] for b in h) == len(p)
    by = {(b["lo"], b["hi"]): b["count"] for b in h}
    assert by[(0.0, 0.01)] == 2       # 0.0, 0.005
    assert by[(0.01, 0.05)] == 1      # 0.03
    assert by[(0.9, 0.95)] == 1       # 0.93
    assert by[(0.99, 1.0)] == 2       # 0.995 and 1.0 (last bin closed on the right)


# --------------------------------------------------------------------------- #
# checkpoint / resume
# --------------------------------------------------------------------------- #
class TestCheckpointResume:
    def test_shard_round_trip(self, tmp_path: Path) -> None:
        write_shard(tmp_path, 0, ["a", "b"], [0.1, 0.9])
        got = load_shards(tmp_path)
        assert set(got) == {"a", "b"}
        assert got["b"] == pytest.approx(0.9, abs=1e-6)

    def test_interrupted_run_resumes_and_covers_input_exactly_once(
        self, tmp_path: Path
    ) -> None:
        """Kill mid-run, restart, assert the union covers the input exactly once."""
        ids = [f"id{i:04d}" for i in range(250)]
        probs = {e: i / 250 for i, e in enumerate(ids)}

        # --- first run dies after two shards (100 of 250 rows) ---
        write_shard(tmp_path, 0, ids[:50], [probs[e] for e in ids[:50]])
        write_shard(tmp_path, 1, ids[50:100], [probs[e] for e in ids[50:100]])

        # --- restart: recompute the todo list exactly as label.main does ---
        done = load_shards(tmp_path)
        assert len(done) == 100
        todo = [i for i in range(len(ids)) if ids[i] not in done]
        assert len(todo) == 150
        assert not set(ids[i] for i in todo) & set(done)  # no rework

        n_existing = len(list(tmp_path.glob("shard_*.parquet")))
        write_shard(tmp_path, n_existing, [ids[i] for i in todo],
                    [probs[ids[i]] for i in todo])

        # --- coverage must be exact: no gaps, no duplicates ---
        final = load_shards(tmp_path)
        assert set(final) == set(ids)
        total_rows = sum(
            __import__("pyarrow.parquet", fromlist=["x"]).ParquetFile(p).metadata.num_rows
            for p in tmp_path.glob("shard_*.parquet")
        )
        assert total_rows == len(ids), "a duplicate row was written across shards"
        for e in ids:
            assert final[e] == pytest.approx(probs[e], abs=1e-6)

    def test_shard_numbering_does_not_collide_on_resume(self, tmp_path: Path) -> None:
        write_shard(tmp_path, 0, ["a"], [0.1])
        write_shard(tmp_path, 1, ["b"], [0.2])
        n = len(list(tmp_path.glob("shard_*.parquet")))
        assert n == 2
        write_shard(tmp_path, n, ["c"], [0.3])
        assert len(list(tmp_path.glob("shard_*.parquet"))) == 3
        assert set(load_shards(tmp_path)) == {"a", "b", "c"}
