"""Dataset loading for both the Task 1 benchmark release and the native Odia gold set.

Two rules are enforced here rather than left to convention:

1. `text_eng` never reaches the model. It is the English source the weak labels
   were derived from, so feeding it would leak the label outright. It is dropped
   in `to_model_frame` and asserted against in `tokenize`.
2. The native gold set is only ever loaded for evaluation. Nothing in this module
   exposes it to training, and `load_native` refuses to return unlabelled rows
   silently.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .common import LABEL_TO_NAME, NAME_TO_LABEL, resolve_path

SPLITS = ("train", "validation", "test")
VARIANTS = ("proportional", "match_minority")

# Odia block plus the punctuation the script shares with Latin.
ODIA_RANGE = re.compile("[଀-୿]")
QUESTION_MARK = re.compile("[?？]")
# zero-width space / non-joiner / joiner, BOM, and the bidi marks
ZERO_WIDTH = re.compile("[" + "".join(chr(c) for c in (0x200b, 0x200c, 0x200d, 0x200e, 0x200f, 0xfeff)) + "]")

# Label spellings seen in the wild for this task. Extend rather than guess.
LABEL_ALIASES = {
    "hate": 1, "hateful": 1, "hate_speech": 1, "hs": 1, "yes": 1, "1": 1, "true": 1,
    "offensive": 1, "toxic": 1, "abusive": 1,
    "non_hate": 0, "nonhate": 0, "not_hate": 0, "nothate": 0, "none": 0, "no": 0,
    "0": 0, "false": 0, "normal": 0, "neutral": 0, "clean": 0, "not_offensive": 0,
    "non-hate": 0, "not-hate": 0,
}


# --------------------------------------------------------------------------- #
# benchmark release (Task 1 output)
# --------------------------------------------------------------------------- #

def resolve_data_root(root: str | Path, fallbacks: list[str] | None = None) -> Path:
    """Find the Task 1 release across machines.

    The repo is developed on one box and run on another, so the configured path
    is tried first, then $ODIAEVAL_DATA_ROOT, then the configured fallbacks. A
    candidate counts only if it actually holds the variant folders.
    """
    if fallbacks is None:
        from .common import load_config
        fallbacks = load_config().get("data", {}).get("root_fallbacks", [])

    candidates: list[Path] = [resolve_path(root)]
    env = os.environ.get("ODIAEVAL_DATA_ROOT")
    if env:
        candidates.insert(0, resolve_path(env))
    for fb in fallbacks or []:
        candidates.append(resolve_path(fb))

    for candidate in candidates:
        if all((candidate / v).is_dir() for v in VARIANTS):
            return candidate
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "could not find the Task 1 release (a folder holding proportional/ and "
        f"match_minority/). Tried:\n  {tried}\n"
        "Set data.root in config.yaml, pass --data-root, or export ODIAEVAL_DATA_ROOT."
    )


def load_benchmark(
    data_root: str | Path,
    variant: str,
    split: str,
    fallbacks: list[str] | None = None,
) -> pd.DataFrame:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")

    path = resolve_data_root(data_root, fallbacks) / variant / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"benchmark file not found: {path}")
    df = pd.read_parquet(path)

    missing = {"text_ory", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if df["text_ory"].isna().any() or (df["text_ory"].str.len() == 0).any():
        raise ValueError(f"{path} contains empty text_ory rows")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError(f"{path} has labels outside {{0, 1}}")

    df = df.copy()
    df["split"] = split
    df["variant"] = variant
    return df


def load_all_benchmark(data_root: str | Path, variant: str) -> dict[str, pd.DataFrame]:
    return {s: load_benchmark(data_root, variant, s) for s in SPLITS}


def assert_no_split_leakage(frames: dict[str, pd.DataFrame]) -> dict:
    """Re-verify Task 1's leakage claim rather than taking it on trust."""
    report = {}
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_text = set(frames[a]["text_ory"]) & set(frames[b]["text_ory"])
            entry = {"shared_text": len(shared_text)}
            if "dedup_group" in frames[a].columns and "dedup_group" in frames[b].columns:
                entry["shared_dedup_group"] = len(
                    set(frames[a]["dedup_group"]) & set(frames[b]["dedup_group"])
                )
            report[f"{a}|{b}"] = entry
            if shared_text:
                raise AssertionError(
                    f"{len(shared_text)} texts appear in both {a} and {b} — refusing to train"
                )
    return report


# --------------------------------------------------------------------------- #
# native Odia gold set
# --------------------------------------------------------------------------- #

def load_native(
    path: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    label_map: dict | None = None,
) -> pd.DataFrame:
    """Load the professor's native gold set from csv / tsv / parquet / jsonl / xlsx.

    Column names are not known in advance, so they are auto-detected and the
    detection is reported. Pass ``text_col`` / ``label_col`` to override.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"unsupported native test file type: {suffix}")

    text_col = text_col or _detect_text_column(df)
    label_col = label_col or _detect_label_column(df)

    out = pd.DataFrame(
        {
            "text_ory": df[text_col].astype(str).map(normalise_text),
            "label": df[label_col].map(lambda v: _coerce_label(v, label_map)),
        }
    )
    out["_source_text_col"] = text_col
    out["_source_label_col"] = label_col

    unmapped = out["label"].isna()
    if unmapped.any():
        bad = sorted({str(v) for v in df.loc[unmapped.values, label_col].unique()})[:10]
        raise ValueError(
            f"could not map {int(unmapped.sum())} label values from column {label_col!r}: {bad}\n"
            "Pass --label-map '{\"your_value\": 1, ...}' to resolve them explicitly."
        )
    out["label"] = out["label"].astype(int)
    out["label_str"] = out["label"].map(LABEL_TO_NAME)

    # Carry any extra metadata columns through for the error analysis.
    for col in df.columns:
        if col not in (text_col, label_col) and col not in out.columns:
            out[col] = df[col].values

    empty = out["text_ory"].str.strip().eq("")
    if empty.any():
        raise ValueError(f"{int(empty.sum())} native rows have empty text in {text_col!r}")
    return out


def _detect_text_column(df: pd.DataFrame) -> str:
    """The text column is the object column with the most Odia-script content."""
    best, best_score = None, -1.0
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna().astype(str).head(500)
        if sample.empty:
            continue
        odia = sample.map(lambda s: len(ODIA_RANGE.findall(s)) / max(len(s), 1)).mean()
        length = sample.str.len().mean()
        score = odia * 100 + min(length, 200) / 200
        if score > best_score:
            best, best_score = col, score
    if best is None:
        raise ValueError(f"no text-like column found among {list(df.columns)}")
    return best


def _detect_label_column(df: pd.DataFrame) -> str:
    preferred = [
        "label", "labels", "gold", "gold_label", "class", "category", "target",
        "y", "is_hate", "hate", "hate_label", "annotation", "final_label", "label_str",
    ]
    lowered = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    # Fall back to any low-cardinality column that maps cleanly onto the aliases.
    for col in df.columns:
        values = df[col].dropna().unique()
        if 1 < len(values) <= 4 and all(_coerce_label(v, None) is not None for v in values):
            return col
    raise ValueError(
        f"no label column found among {list(df.columns)}; pass --label-col explicitly"
    )


def _coerce_label(value, label_map: dict | None):
    if label_map:
        for key, mapped in label_map.items():
            if str(value).strip() == str(key).strip():
                return int(mapped)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        if float(value) in (0.0, 1.0):
            return int(value)
        return None
    key = str(value).strip().lower().replace(" ", "_")
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    if key in NAME_TO_LABEL:
        return NAME_TO_LABEL[key]
    return None


# --------------------------------------------------------------------------- #
# preprocessing shared by both sets
# --------------------------------------------------------------------------- #

def normalise_text(text: str) -> str:
    """NFC, strip zero-width marks, collapse whitespace.

    This mirrors the normalisation Task 1 used for deduplication, so the native
    set and the benchmark set reach the tokeniser in the same shape. Step 6 of
    the assignment requires identical preprocessing across both evaluations.
    """
    text = unicodedata.normalize("NFC", str(text))
    text = ZERO_WIDTH.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def to_model_frame(df: pd.DataFrame, text_column: str = "text_ory") -> pd.DataFrame:
    """Reduce to exactly what the model may see: normalised text and a label."""
    out = pd.DataFrame(
        {
            "text": df[text_column].astype(str).map(normalise_text),
            "label": df["label"].astype(int).values,
        }
    )
    return out


def is_question(series: pd.Series) -> pd.Series:
    """Question-vs-statement proxy used throughout the error analysis.

    The data card predicts that errors on the native set will track grammatical
    mood rather than content, and a question mark is the agreed proxy for mood.
    """
    return series.astype(str).str.contains(QUESTION_MARK, regex=True)


def tokenize(frame: pd.DataFrame, tokenizer, max_length: int, padding: str = "max_length"):
    """Tokenise a two-column model frame into a datasets.Dataset."""
    from datasets import Dataset

    forbidden = {"text_eng", "text_english", "english"} & set(frame.columns)
    if forbidden:
        raise AssertionError(
            f"English source columns reached the tokeniser: {sorted(forbidden)}. "
            "text_eng is a direct label leak and must never be a model input."
        )

    ds = Dataset.from_pandas(frame[["text", "label"]].reset_index(drop=True))

    def _encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=padding,
            max_length=max_length,
        )

    ds = ds.map(_encode, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    return ds
