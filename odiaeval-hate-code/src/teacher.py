"""The weak-label teacher: loading, orientation proof, and batched scoring.

Separated from ``src/label.py`` so the correctness machinery — the ``id2label``
assertion and the canary tiers — is unit-testable without running a stage.

An inverted label mapping would poison the whole dataset and be **invisible**
downstream: class balance would look fine, splits would look fine, and the
fine-tuned student would score plausibly while having learned the exact opposite
of hate. Hence: metadata assertions *and* a behavioural canary, every run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

EXPECTED_ID2LABEL = {0: "nothate", 1: "hate"}


# --------------------------------------------------------------------------- #
# metadata assertions
# --------------------------------------------------------------------------- #
def canonical_id2label(raw: dict) -> dict[int, str]:
    """Coerce only the *key type* (HF may hand back str or int keys).

    Label **values** are compared verbatim — no case folding, no stripping. If
    the mapping changed upstream that is fatal, not something to normalise away.
    """
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            ik = int(k)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"FATAL: teacher id2label has a non-integer key {k!r} - refusing to guess."
            ) from exc
        if not isinstance(v, str):
            raise SystemExit(f"FATAL: teacher id2label value for {k!r} is {v!r}, expected str.")
        out[ik] = v
    return out


def assert_teacher_orientation(config: Any) -> dict[int, str]:
    """Fail loudly unless the teacher is exactly the 2-class model we pinned."""
    if getattr(config, "num_labels", None) != 2:
        raise SystemExit(
            f"FATAL: teacher num_labels={getattr(config, 'num_labels', None)}, expected 2."
        )
    got = canonical_id2label(config.id2label)
    if got != EXPECTED_ID2LABEL:
        raise SystemExit(
            f"FATAL: teacher id2label is {got}, expected {EXPECTED_ID2LABEL}. "
            f"An inverted or relabelled mapping would poison every downstream number "
            f"invisibly. Not proceeding."
        )
    return got


# --------------------------------------------------------------------------- #
# handle
# --------------------------------------------------------------------------- #
@dataclass
class TeacherHandle:
    model: Any
    tokenizer: Any
    device: Any
    use_fp16: bool
    max_length: int
    meta: dict = field(default_factory=dict)


def load_teacher(cfg: dict, *, device: str | None = None) -> TeacherHandle:
    """Load the teacher at its pinned commit SHA and prove its orientation."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tcfg = cfg["teacher"]
    repo, rev = tcfg["repo_id"], tcfg["revision"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev)

    id2label = assert_teacher_orientation(model.config)

    # fp16 is a CUDA optimisation; on CPU it is slower and unsupported for many
    # kernels, so honour the config only where it makes sense and say which.
    use_fp16 = bool(tcfg.get("fp16", False)) and dev.type == "cuda"
    if use_fp16:
        model = model.half()
    model.to(dev).eval()

    deterministic = _try_enable_determinism()

    meta = {
        "repo_id": repo,
        "revision": rev,
        "device": str(dev),
        "device_name": (
            torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
        ),
        "fp16_forward": use_fp16,
        "fp16_requested": bool(tcfg.get("fp16", False)),
        "softmax_dtype": "float32",
        "id2label": {str(k): v for k, v in id2label.items()},
        "num_labels": int(model.config.num_labels),
        "max_position_embeddings": int(model.config.max_position_embeddings),
        "max_length": int(tcfg["max_length"]),
        "torch_version": torch.__version__,
        "deterministic_algorithms_enforced": deterministic,
    }
    return TeacherHandle(model, tok, dev, use_fp16, int(tcfg["max_length"]), meta)


def _try_enable_determinism() -> bool:
    """Enable deterministic kernels if the platform allows; report, don't assume."""
    import torch

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        return True
    except Exception:  # noqa: BLE001 - platform-dependent, reported not swallowed
        return False


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def token_lengths(handle: TeacherHandle, texts: Sequence[str], chunk: int = 4096) -> np.ndarray:
    """True (untruncated) token length per text, for sorting and truncation stats."""
    out = np.empty(len(texts), dtype=np.int32)
    for i in range(0, len(texts), chunk):
        enc = handle.tokenizer(list(texts[i : i + chunk]), truncation=False)["input_ids"]
        out[i : i + chunk] = [len(x) for x in enc]
    return out


def iter_scored(
    handle: TeacherHandle,
    texts: Sequence[str],
    order: Sequence[int],
    *,
    batch_size: int,
    min_batch_size: int,
) -> Iterator[tuple[list[int], np.ndarray, int]]:
    """Score ``texts`` in ``order``, yielding ``(indices, p_hate, effective_batch)``.

    ``order`` should be a length-sorted permutation: uniform-length batches cut
    padding waste substantially on a corpus this skewed. Callers restore the
    original order from the yielded indices.

    The forward pass may run in fp16, but logits are **upcast to float32 before
    the softmax** — fp16 carries roughly three decimal digits and both thresholds
    sit in the tails where that precision is thinnest.

    On CUDA OOM the batch size halves (down to ``min_batch_size``) and retries.
    """
    import torch

    i = 0
    bs = batch_size
    order = list(order)
    while i < len(order):
        idx = order[i : i + bs]
        batch = [texts[j] for j in idx]
        try:
            with torch.no_grad():
                enc = handle.tokenizer(
                    batch,
                    truncation=True,
                    max_length=handle.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(handle.device)
                logits = handle.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            yield idx, probs.detach().cpu().numpy().astype(np.float32), bs
            i += len(idx)
        except torch.cuda.OutOfMemoryError:
            if bs <= min_batch_size:
                raise
            torch.cuda.empty_cache()
            bs = max(min_batch_size, bs // 2)
            print(f"    CUDA OOM - halving batch size to {bs}", flush=True)


def score_texts(
    handle: TeacherHandle,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    min_batch_size: int = 4,
    sort_by_length: bool = True,
) -> tuple[np.ndarray, dict]:
    """Convenience wrapper for small sets (canary, HateCheck). Returns p_hate in input order."""
    lengths = token_lengths(handle, texts)
    order = (
        sorted(range(len(texts)), key=lambda k: (lengths[k], k))
        if sort_by_length
        else list(range(len(texts)))
    )
    probs = np.empty(len(texts), dtype=np.float32)
    eff = batch_size
    t0 = time.time()
    for idx, p, bs in iter_scored(
        handle, texts, order, batch_size=batch_size, min_batch_size=min_batch_size
    ):
        probs[idx] = p
        eff = min(eff, bs)
    stats = {
        "n": len(texts),
        "seconds": round(time.time() - t0, 2),
        "effective_batch_size": eff,
        "n_truncated": int((lengths > handle.max_length).sum()),
        "max_token_length": int(lengths.max()) if len(lengths) else 0,
    }
    return probs, stats


# --------------------------------------------------------------------------- #
# canary tiers
# --------------------------------------------------------------------------- #
def run_tier(handle: TeacherHandle, items: list[dict], **kw) -> list[dict]:
    """Score one canary tier, returning each item with its p_hate attached."""
    probs, _ = score_texts(handle, [it["text"] for it in items], **kw)
    return [{**it, "p_hate": float(p)} for it, p in zip(items, probs)]


def check_hard_tier(hate_scored: list[dict], benign_scored: list[dict]) -> dict:
    """Orientation guard. An inverted model shows up as a large NEGATIVE margin.

    Two assertions, both about *ordering* rather than confidence, so the guard
    cannot fail merely because the teacher is more conservative than we guessed:

    * mean margin between the clear-hate and clear-benign groups exceeds 0.5
    * rank separation: every clear-hate item scores above every clear-benign item
    """
    h = np.array([x["p_hate"] for x in hate_scored], dtype=np.float64)
    b = np.array([x["p_hate"] for x in benign_scored], dtype=np.float64)
    margin = float(h.mean() - b.mean())
    separated = bool(h.min() > b.max())
    return {
        "mean_p_hate_clear_hate": float(h.mean()),
        "mean_p_hate_clear_benign": float(b.mean()),
        "margin": margin,
        "min_clear_hate": float(h.min()),
        "max_clear_benign": float(b.max()),
        "rank_separated": separated,
        "passed": bool(margin > 0.5 and separated),
    }


def check_pinned_tier(scored: dict[str, list[dict]], bounds: dict) -> dict:
    """Confidence bounds — asserted only once they have been measured and pinned.

    While any bound is ``None`` this reports observed values and does not fail.
    That is deliberate: an over-tight guessed bound would abort a 138k-row run
    for the wrong reason.
    """
    # Items flagged documented_failure are known, characterised teacher misses.
    # They stay in the orientation margin but are excluded from the confidence
    # floor - pinning clear_hate_min off a known miss would make it meaningless.
    live_hate = [x for x in scored["HARD_CLEAR_HATE"] if not x.get("documented_failure")]
    if not live_hate:
        raise SystemExit("FATAL: every HARD_CLEAR_HATE item is flagged documented_failure.")
    obs = {
        "clear_hate_min": min(x["p_hate"] for x in live_hate),
        "clear_benign_max": max(x["p_hate"] for x in scored["HARD_CLEAR_BENIGN"]),
        "hard_negative_max": max(x["p_hate"] for x in scored["PINNED_HARD_NEG"]),
    }
    unpinned = [k for k, v in bounds.items() if v is None]
    failures: list[str] = []
    if not unpinned:
        if obs["clear_hate_min"] < bounds["clear_hate_min"]:
            failures.append(
                f"clear_hate_min {obs['clear_hate_min']:.6f} < pinned {bounds['clear_hate_min']}"
            )
        if obs["clear_benign_max"] > bounds["clear_benign_max"]:
            failures.append(
                f"clear_benign_max {obs['clear_benign_max']:.6f} > pinned "
                f"{bounds['clear_benign_max']}"
            )
        if obs["hard_negative_max"] > bounds["hard_negative_max"]:
            failures.append(
                f"hard_negative_max {obs['hard_negative_max']:.6f} > pinned "
                f"{bounds['hard_negative_max']}"
            )
    return {
        "documented_failures": [
            {"id": x["id"], "p_hate": x["p_hate"]}
            for x in scored["HARD_CLEAR_HATE"] if x.get("documented_failure")
        ],
        "observed": {k: float(v) for k, v in obs.items()},
        "pinned": dict(bounds),
        "unpinned": unpinned,
        "failures": failures,
        "passed": not failures,
        "asserted": not unpinned,
    }
