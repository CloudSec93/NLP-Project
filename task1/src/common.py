"""Shared helpers: config loading, seeding, device selection, metrics, run metadata.

Everything that more than one stage needs lives here so the stages themselves
stay readable.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

LABEL_TO_NAME = {0: "NON_HATE", 1: "HATE"}
NAME_TO_LABEL = {"NON_HATE": 0, "HATE": 1}

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return ((base or REPO_ROOT) / p).resolve()


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def pick_precision(setting: str = "auto") -> dict:
    """Translate the mixed_precision config value into Trainer kwargs."""
    if setting in ("off", "none", "fp32", False, None):
        return {"fp16": False, "bf16": False}
    try:
        import torch
    except ImportError:
        return {"fp16": False, "bf16": False}
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    if setting == "bf16":
        return {"fp16": False, "bf16": True}
    if setting == "fp16":
        return {"fp16": True, "bf16": False}
    # auto: bf16 is numerically safer and needs compute capability 8.0+.
    if torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}
    return {"fp16": True, "bf16": False}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    """The full metric set the assignment asks for, plus diagnostics.

    Reported: accuracy, macro-F1, and precision / recall / F1 for HATE. The
    per-class block and confusion matrix support the error analysis.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    per_class = {
        LABEL_TO_NAME[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in (0, 1)
    }

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "hate_precision": per_class["HATE"]["precision"],
        "hate_recall": per_class["HATE"]["recall"],
        "hate_f1": per_class["HATE"]["f1"],
        "per_class": per_class,
        # rows are gold NON_HATE / HATE, columns are predicted NON_HATE / HATE
        "confusion_matrix": cm.tolist(),
        "predicted_hate_rate": float((y_pred == 1).mean()),
        "gold_hate_rate": float((y_true == 1).mean()),
    }


def as_percent(metrics: dict, keys: Iterable[str]) -> dict:
    return {k: round(100.0 * metrics[k], 2) for k in keys}


# --------------------------------------------------------------------------- #
# run metadata
# --------------------------------------------------------------------------- #

def git_commit(repo: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo or REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def resolve_model_revision(name: str, revision: str | None) -> str | None:
    """Record the exact Hub commit a run used, so the run stays reproducible."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(name, revision=revision)
        return info.sha
    except Exception:
        return revision


def environment_fingerprint() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    return info


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #

def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
