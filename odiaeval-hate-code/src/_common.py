"""Shared utilities: config loading, seeding, paths, hashing, stage bookkeeping.

Every ``src/stage_*.py`` module imports from here. Nothing in this file touches
the network or a model — it is pure plumbing and is fully unit-testable.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` as a plain dict.

    The two teacher thresholds are fixed by the assignment. If the file has been
    edited away from 0.90 / 0.05, fail loudly (CLAUDE.md §5.6) unless the caller
    has explicitly opted in via ``ODIAEVAL_ALLOW_THRESHOLD_OVERRIDE=1``.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    hate_min = cfg["thresholds"]["hate_min"]
    non_hate_max = cfg["thresholds"]["non_hate_max"]
    override = os.environ.get("ODIAEVAL_ALLOW_THRESHOLD_OVERRIDE") == "1"
    if (hate_min, non_hate_max) != (0.90, 0.05) and not override:
        raise ValueError(
            f"Teacher thresholds are fixed by the assignment at 0.90 / 0.05 but "
            f"config.yaml has {hate_min} / {non_hate_max}. This change must be "
            f"loud and justified in the data card. Set "
            f"ODIAEVAL_ALLOW_THRESHOLD_OVERRIDE=1 to proceed deliberately."
        )
    return cfg


def config_sha256(path: str | Path | None = None) -> str:
    """SHA-256 of the raw config file bytes — recorded in the manifest."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    """Seed ``random`` and (if importable) ``numpy`` / ``torch`` (CLAUDE.md §5.1)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # best effort; GPU kernels are not fully bitwise-deterministic — Stage 3
        # additionally rounds p(hate) before thresholding (see data card).
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001 - older torch
            pass
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# hashing helpers
# --------------------------------------------------------------------------- #
def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def example_id(source_config: str, doc_id: str) -> str:
    """Stable 16-hex example id (spec §5 Stage 2)."""
    return sha256_hex(f"{source_config}|{doc_id}")[:16]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Paths":
        p = cfg["paths"]
        return cls(
            root=REPO_ROOT,
            raw=REPO_ROOT / p["raw"],
            interim=REPO_ROOT / p["interim"],
            processed=REPO_ROOT / p["processed"],
            artifacts=REPO_ROOT / p["artifacts"],
        )

    def ensure(self) -> "Paths":
        for d in (self.raw, self.interim, self.processed, self.artifacts):
            d.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- #
# stage bookkeeping — "a row that disappears without a recorded reason is a bug"
# --------------------------------------------------------------------------- #
def append_stage_stats(
    artifacts_dir: str | Path,
    stage: str,
    *,
    rows_in: int,
    rows_out: int,
    dropped_by_reason: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one stage's in / out / dropped-by-reason record to stage_stats.json."""
    path = Path(artifacts_dir) / "stage_stats.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"stages": []}

    dropped = dict(dropped_by_reason or {})
    accounted = rows_out + sum(dropped.values())
    record = {
        "stage": stage,
        "timestamp": utc_now_iso(),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dropped_total": rows_in - rows_out,
        "dropped_by_reason": dropped,
        "balanced": accounted == rows_in,
        **({"extra": extra} if extra else {}),
    }
    data["stages"] = [s for s in data["stages"] if s.get("stage") != stage]
    data["stages"].append(record)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
