"""Stage 6 tests: balancing, quota allocation, and the leakage check.

The last class here is the one that actually protects the group's result: if a
``dedup_group`` can straddle a split boundary, near-duplicates leak from train
into test, the benchmark score inflates, and the reported translationese gap
inflates with it.
"""

from __future__ import annotations

import pytest

from src.balance_split import downsample, largest_remainder, target_counts
from src.hashing import assign_split

PROPORTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
SALT = "42"
CFG = ["Toxic_Matrix", "HHRLHF_T", "Dolly_T"]


class TestLargestRemainder:
    def test_allocation_sums_to_total_exactly(self) -> None:
        for total in (1, 7, 100, 18_636):
            alloc = largest_remainder({"a": 59742.0, "b": 10477.0, "c": 14498.0}, total)
            assert sum(alloc.values()) == total

    def test_allocation_is_proportional(self) -> None:
        alloc = largest_remainder({"a": 90.0, "b": 10.0}, 100)
        assert alloc == {"a": 90, "b": 10}

    def test_deterministic_and_order_independent(self) -> None:
        w1 = {"a": 3.0, "b": 3.0, "c": 3.0, "d": 1.0}
        w2 = {"d": 1.0, "c": 3.0, "b": 3.0, "a": 3.0}
        assert largest_remainder(w1, 7) == largest_remainder(w2, 7)

    def test_degenerate_inputs(self) -> None:
        assert largest_remainder({}, 10) == {}
        assert largest_remainder({"a": 1.0}, 0) == {"a": 0}
        assert largest_remainder({"a": 0.0, "b": 0.0}, 5) == {"a": 0, "b": 0}


class TestTargetCounts:
    MAJ = {"Toxic_Matrix": 59742, "HHRLHF_T": 10477, "Dolly_T": 14498}
    MIN = {"Toxic_Matrix": 17082, "HHRLHF_T": 1480, "Dolly_T": 74}

    def test_proportional_follows_the_majority_mix(self) -> None:
        q = target_counts(self.MAJ, self.MIN, 18_636, "proportional")
        assert sum(q.values()) == 18_636
        # majority mix is ~70.5 / 12.4 / 17.1
        assert q["Toxic_Matrix"] == pytest.approx(13_142, abs=2)
        assert q["Dolly_T"] == pytest.approx(3_189, abs=2)

    def test_match_minority_reproduces_the_minority_mix(self) -> None:
        q = target_counts(self.MAJ, self.MIN, 18_636, "match_minority")
        assert sum(q.values()) == 18_636
        assert q == self.MIN  # exactly the same source profile -> NMI 0

    def test_quota_never_exceeds_availability(self) -> None:
        """A source short of its quota spills onto sources with headroom."""
        maj = {"Toxic_Matrix": 100, "HHRLHF_T": 10, "Dolly_T": 5}
        minority = {"Toxic_Matrix": 10, "HHRLHF_T": 50, "Dolly_T": 50}
        q = target_counts(maj, minority, 100, "match_minority")
        assert sum(q.values()) == 100
        for k, v in q.items():
            assert v <= maj[k], f"{k} quota {v} exceeds available {maj[k]}"

    def test_unknown_mode_is_fatal(self) -> None:
        with pytest.raises(SystemExit, match="stratify_mode"):
            target_counts(self.MAJ, self.MIN, 100, "upsample_minority")


class TestDownsample:
    def _rows(self, n=900):
        return [
            {"example_id": f"{i:016x}", "source_config": CFG[i % 3], "label_str": "NON_HATE"}
            for i in range(n)
        ]

    def test_hits_the_quota_exactly(self) -> None:
        q = {"Toxic_Matrix": 10, "HHRLHF_T": 20, "Dolly_T": 5}
        got = downsample(self._rows(), q, 42)
        assert len(got) == 35
        from collections import Counter

        assert Counter(r["source_config"] for r in got) == q

    def test_deterministic(self) -> None:
        q = {"Toxic_Matrix": 10, "HHRLHF_T": 10, "Dolly_T": 10}
        a = downsample(self._rows(), q, 42)
        b = downsample(self._rows(), q, 42)
        assert [r["example_id"] for r in a] == [r["example_id"] for r in b]

    def test_independent_of_input_row_order(self) -> None:
        q = {"Toxic_Matrix": 10, "HHRLHF_T": 10, "Dolly_T": 10}
        rows = self._rows()
        a = downsample(rows, q, 42)
        b = downsample(list(reversed(rows)), q, 42)
        assert sorted(r["example_id"] for r in a) == sorted(r["example_id"] for r in b)

    def test_never_upsamples_when_quota_exceeds_pool(self) -> None:
        """Duplicated minority rows produce optimistic validation scores."""
        q = {"Toxic_Matrix": 10_000, "HHRLHF_T": 0, "Dolly_T": 0}
        got = downsample(self._rows(30), q, 42)
        ids = [r["example_id"] for r in got]
        assert len(ids) == len(set(ids))  # no duplicates introduced
        assert len(ids) == 10             # only what the pool actually held


class TestLeakageCheck:
    """No dedup_group may appear in more than one split."""

    def test_shared_dedup_group_always_lands_in_one_split(self) -> None:
        # 200 clusters, 5 rows each, all sharing their cluster's dedup_group
        rows = [
            {"example_id": f"e{c:03d}_{m}", "dedup_group": f"g{c:03d}"}
            for c in range(200)
            for m in range(5)
        ]
        for r in rows:
            r["split"] = assign_split(r["dedup_group"], PROPORTIONS, SALT)
        by_group: dict[str, set[str]] = {}
        for r in rows:
            by_group.setdefault(r["dedup_group"], set()).add(r["split"])
        straddling = [g for g, s in by_group.items() if len(s) > 1]
        assert straddling == [], f"{len(straddling)} groups straddle a split"

    def test_hashing_the_row_instead_of_the_group_would_leak(self) -> None:
        """Demonstrates why the split unit is dedup_group, not example_id."""
        rows = [
            {"example_id": f"e{c:03d}_{m}", "dedup_group": f"g{c:03d}"}
            for c in range(200)
            for m in range(5)
        ]
        by_group: dict[str, set[str]] = {}
        for r in rows:
            sp = assign_split(r["example_id"], PROPORTIONS, SALT)  # the WRONG unit
            by_group.setdefault(r["dedup_group"], set()).add(sp)
        straddling = [g for g, s in by_group.items() if len(s) > 1]
        assert len(straddling) > 0, "the wrong unit should visibly leak"

    def test_split_assignment_survives_row_insertion(self) -> None:
        """CLAUDE.md §5.2 — adding rows must not move existing ones."""
        before = {f"g{i:04d}": assign_split(f"g{i:04d}", PROPORTIONS, SALT)
                  for i in range(500)}
        after = {f"g{i:04d}": assign_split(f"g{i:04d}", PROPORTIONS, SALT)
                 for i in range(2000)}
        for g, sp in before.items():
            assert after[g] == sp

    def test_proportions_are_approximately_honoured_per_cell(self) -> None:
        """Pure hash bucketing drifts; the drift must stay small at cell scale."""
        n = 17_000
        counts = {"train": 0, "validation": 0, "test": 0}
        for i in range(n):
            counts[assign_split(f"cell_g{i:06d}", PROPORTIONS, SALT)] += 1
        assert counts["train"] / n == pytest.approx(0.80, abs=0.01)
        assert counts["validation"] / n == pytest.approx(0.10, abs=0.01)
        assert counts["test"] / n == pytest.approx(0.10, abs=0.01)
