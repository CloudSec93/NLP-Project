"""Stage 0 tests: deterministic, edit-stable split assignment (spec §8)."""

from __future__ import annotations

import pytest

from src.hashing import assign_split, unit_fraction

PROPORTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
SALT = "42"


def test_fraction_is_in_unit_interval_and_stable() -> None:
    for i in range(1000):
        x = unit_fraction(f"group-{i}", SALT)
        assert 0.0 <= x < 1.0
    assert unit_fraction("group-7", SALT) == unit_fraction("group-7", SALT)


def test_salt_changes_assignment_space() -> None:
    a = [assign_split(f"g{i}", PROPORTIONS, "42") for i in range(500)]
    b = [assign_split(f"g{i}", PROPORTIONS, "99") for i in range(500)]
    assert a != b  # different salt -> different partition


def test_assignment_is_deterministic() -> None:
    first = {f"g{i}": assign_split(f"g{i}", PROPORTIONS, SALT) for i in range(200)}
    second = {f"g{i}": assign_split(f"g{i}", PROPORTIONS, SALT) for i in range(200)}
    assert first == second


def test_inserting_rows_does_not_move_existing_rows() -> None:
    """The core guarantee (CLAUDE.md §5.2): adding data cannot reshuffle splits."""
    before = {f"g{i}": assign_split(f"g{i}", PROPORTIONS, SALT) for i in range(500)}
    # now pretend the corpus grew — assign a superset
    after = {f"g{i}": assign_split(f"g{i}", PROPORTIONS, SALT) for i in range(2000)}
    for k, v in before.items():
        assert after[k] == v


def test_proportions_are_approximately_honoured() -> None:
    n = 20000
    counts = {"train": 0, "validation": 0, "test": 0}
    for i in range(n):
        counts[assign_split(f"unit-{i}", PROPORTIONS, SALT)] += 1
    assert counts["train"] / n == pytest.approx(0.80, abs=0.02)
    assert counts["validation"] / n == pytest.approx(0.10, abs=0.02)
    assert counts["test"] / n == pytest.approx(0.10, abs=0.02)


def test_bad_proportions_rejected() -> None:
    with pytest.raises(ValueError):
        assign_split("g", {"train": 0.7, "test": 0.1}, SALT)
    with pytest.raises(ValueError):
        assign_split("g", {}, SALT)
