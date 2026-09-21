"""Deterministic, edit-stable split assignment (spec §4⑥, CLAUDE.md §5.2).

Split membership is a pure function of a stable id and the configured
proportions. It does not depend on row order, shuffle state, or dict iteration
order. Adding a row cannot move an existing row between splits.

The hashed unit is the **dedup_group** (near-duplicate cluster id), not the row,
so a template family cannot straddle a split boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

_UINT64 = 1 << 64


def unit_fraction(unit_id: str, salt: str) -> float:
    """Map an id to a stable float in ``[0, 1)`` via the low 64 bits of SHA-256."""
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode("utf-8")).digest()
    low64 = int.from_bytes(digest[:8], "big")
    return low64 / _UINT64


def assign_split(
    unit_id: str,
    proportions: Mapping[str, float],
    salt: str,
) -> str:
    """Bucket ``unit_id`` into a split by cumulative proportion.

    ``proportions`` keys are iterated in insertion order; pass a dict literal or
    an ordered mapping so the cut points are stable (e.g. ``{"train": 0.8,
    "validation": 0.1, "test": 0.1}``). Values should sum to ~1.0; the final
    bucket absorbs any rounding slack.
    """
    keys = list(proportions)
    if not keys:
        raise ValueError("proportions is empty")
    total = sum(proportions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"proportions must sum to 1.0, got {total}")

    x = unit_fraction(unit_id, salt)
    cum = 0.0
    for k in keys[:-1]:
        cum += proportions[k]
        if x < cum:
            return k
    return keys[-1]
