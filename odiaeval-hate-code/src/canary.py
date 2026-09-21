"""Prove the teacher before spending 137,954 forward passes on it.

Tier order is deliberate and is *not* the order the tiers appear in the fixture:

1. **PROBE** — harmful requests, with and without an identity target. Not a
   guard. These are the register that dominates ``Toxic_Matrix``, so where they
   land is the cheapest possible advance signal on which Stage 4 outcome we are
   heading for (see the anti-correlation note in the report). Two seconds of
   inference that tells us what 137,954 rows would otherwise take an hour to say.
2. **HARD** — orientation. Must pass or we abort. This is the inversion guard.
3. **PINNED** — confidence bounds. Measured and reported; asserted only once a
   human has looked at the observed values and written them into the fixture.

Optionally also runs **HateCheck** (Röttger et al., ACL 2021) — 3,728 cases
across 29 functional tests — to give the data card a citable characterisation of
the teacher instead of a hand-wave. Read from the pinned parquet, so it costs no
new dependency.

Run:  python -m src.canary [--hatecheck] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from src._common import Paths, load_config
from src.teacher import (
    check_hard_tier,
    check_pinned_tier,
    load_teacher,
    run_tier,
    score_texts,
)
from tests.fixtures.canary import PINNED_BOUNDS, TIERS

HATECHECK_REPO = "Paul/hatecheck"
HATECHECK_PARQUET_API = (
    "https://huggingface.co/api/datasets/Paul/hatecheck/parquet/default/test"
)


# --------------------------------------------------------------------------- #
# tier printing
# --------------------------------------------------------------------------- #
def _print_items(title: str, items: list[dict], note: str = "") -> None:
    print()
    print("-" * 100)
    print(title + (f"   [{note}]" if note else ""))
    print("-" * 100)
    for it in sorted(items, key=lambda x: -x["p_hate"]):
        print(f"  p_hate={it['p_hate']:.6f}  {it['id']:<12} ({it['form']})")
        print(f"      {it['text'][:110]}")


def run_probe_tier(handle, **kw) -> dict:
    """The definitional-gap probe. Reported, recorded, never a pass/fail gate."""
    harmful = run_tier(handle, TIERS["PROBE_HARMFUL"][0], **kw)
    targeted = run_tier(handle, TIERS["PROBE_TARGETED"][0], **kw)

    print()
    print("=" * 100)
    print("PROBE TIER - the definitional gap (trap 2). NOT a pass/fail guard.")
    print("These are harmful requests in the register that dominates Toxic_Matrix.")
    print("=" * 100)
    _print_items(
        "PROBE_HARMFUL_REQUEST - toxic, no identity target", harmful, "expect LOW if trap 2 holds"
    )
    _print_items(
        "PROBE_HARMFUL_TARGETED - harmful request WITH identity target", targeted, "the mixed case"
    )

    hr = np.array([x["p_hate"] for x in harmful])
    ht = np.array([x["p_hate"] for x in targeted])
    summary = {
        "harmful_request_mean": float(hr.mean()),
        "harmful_request_max": float(hr.max()),
        "harmful_request_n_above_0.90": int((hr >= 0.90).sum()),
        "harmful_request_n_below_0.05": int((hr <= 0.05).sum()),
        "harmful_targeted_mean": float(ht.mean()),
        "harmful_targeted_max": float(ht.max()),
        "harmful_targeted_n_above_0.90": int((ht >= 0.90).sum()),
        "items": harmful + targeted,
    }

    print()
    print(f"  harmful-request  mean p_hate = {summary['harmful_request_mean']:.6f}   "
          f"max = {summary['harmful_request_max']:.6f}   "
          f">=0.90: {summary['harmful_request_n_above_0.90']}/5   "
          f"<=0.05: {summary['harmful_request_n_below_0.05']}/5")
    print(f"  harmful+targeted mean p_hate = {summary['harmful_targeted_mean']:.6f}   "
          f"max = {summary['harmful_targeted_max']:.6f}   "
          f">=0.90: {summary['harmful_targeted_n_above_0.90']}/2")

    print()
    print("  READ:", _interpret_probe(summary))
    return summary


def _interpret_probe(s: dict) -> str:
    """Translate the probe into which Stage 4 outcome we are heading for."""
    m = s["harmful_request_mean"]
    if m >= 0.90:
        return (
            "harmful requests score HIGH -> expect a large HATE yield and a NEAR-TOTAL "
            "source confound (Toxic_Matrix ~ HATE, Dolly_T ~ NON_HATE)."
        )
    if m <= 0.05:
        return (
            "harmful requests score LOW -> they will land in NON_HATE, which draws from "
            "all three sources. Expect a SMALL HATE yield but a WEAKENED confound."
        )
    return (
        "harmful requests land in the MIDDLE -> they will be DISCARDED, not relabelled. "
        "This is the worst of the three outcomes: small HATE yield AND NON_HATE still "
        "dominated by Dolly_T. Watch the per-config discarded mass at Stage 4."
    )


# --------------------------------------------------------------------------- #
# HateCheck
# --------------------------------------------------------------------------- #
def run_hatecheck(handle, cfg: dict, **kw) -> dict:
    """Characterise the teacher on HateCheck's 29 functional tests."""
    import fsspec
    import pyarrow.parquet as pq

    print()
    print("=" * 100)
    print("HATECHECK - Rottger et al., ACL 2021: 29 functional tests for hate-speech classifiers")
    print("=" * 100)

    urls = json.loads(
        fsspec.filesystem("http", client_kwargs={"trust_env": True})
        .open(HATECHECK_PARQUET_API, "rb")
        .read()
    )
    fs = fsspec.filesystem("http", client_kwargs={"trust_env": True})
    tbl = pq.ParquetFile(fs.open(urls[0], "rb")).read(
        columns=["functionality", "case_id", "test_case", "label_gold", "target_ident"]
    )
    rows = tbl.to_pylist()
    labels = sorted({r["label_gold"] for r in rows})
    print(f"  loaded {len(rows):,} cases, label_gold values = {labels}")

    probs, stats = score_texts(handle, [r["test_case"] for r in rows], **kw)
    for r, p in zip(rows, probs):
        r["p_hate"] = float(p)

    hate_label = "hateful"
    if hate_label not in labels:
        raise SystemExit(f"FATAL: HateCheck label_gold values changed: {labels}")

    hi = cfg["thresholds"]["hate_min"]
    lo = cfg["thresholds"]["non_hate_max"]

    by_func: dict[str, dict] = {}
    for r in rows:
        f = r["functionality"]
        d = by_func.setdefault(
            f, {"n": 0, "gold_hateful": 0, "p": [], "n_hate_band": 0, "n_nonhate_band": 0}
        )
        d["n"] += 1
        d["gold_hateful"] += int(r["label_gold"] == hate_label)
        d["p"].append(r["p_hate"])
        d["n_hate_band"] += int(r["p_hate"] >= hi)
        d["n_nonhate_band"] += int(r["p_hate"] <= lo)

    for d in by_func.values():
        arr = np.array(d["p"])
        d["mean_p_hate"] = float(arr.mean())
        d["gold_is_hateful"] = d["gold_hateful"] == d["n"]
        d["recall_at_band"] = (
            d["n_hate_band"] / d["n"] if d["gold_is_hateful"] else None
        )
        d["correct_reject_at_band"] = (
            None if d["gold_is_hateful"] else d["n_nonhate_band"] / d["n"]
        )
        del d["p"]

    print()
    print(f"  {'functionality':<28}{'n':>6}{'gold':>10}{'mean p':>10}"
          f"{'>=0.90':>9}{'<=0.05':>9}")
    print("  " + "-" * 74)
    for f, d in sorted(by_func.items(), key=lambda kv: -kv[1]["mean_p_hate"]):
        gold = "hateful" if d["gold_is_hateful"] else "non-hate"
        print(f"  {f:<28}{d['n']:>6}{gold:>10}{d['mean_p_hate']:>10.4f}"
              f"{d['n_hate_band'] / d['n'] * 100:>8.0f}%{d['n_nonhate_band'] / d['n'] * 100:>8.0f}%")

    gold_h = np.array([r["p_hate"] for r in rows if r["label_gold"] == hate_label])
    gold_n = np.array([r["p_hate"] for r in rows if r["label_gold"] != hate_label])
    overall = {
        "n_cases": len(rows),
        "n_functionalities": len(by_func),
        "gold_hateful_n": int(len(gold_h)),
        "gold_hateful_recall_at_0.90": float((gold_h >= hi).mean()),
        "gold_hateful_mean_p": float(gold_h.mean()),
        "gold_nonhateful_n": int(len(gold_n)),
        "gold_nonhateful_correct_reject_at_0.05": float((gold_n <= lo).mean()),
        "gold_nonhateful_mean_p": float(gold_n.mean()),
        "gold_nonhateful_false_hate_rate_at_0.90": float((gold_n >= hi).mean()),
        "seconds": stats["seconds"],
    }
    print()
    print(f"  gold HATEFUL     n={overall['gold_hateful_n']:<6} "
          f"mean p={overall['gold_hateful_mean_p']:.4f}   "
          f"recall at p>=0.90 = {overall['gold_hateful_recall_at_0.90'] * 100:.1f}%")
    print(f"  gold NON-HATEFUL n={overall['gold_nonhateful_n']:<6} "
          f"mean p={overall['gold_nonhateful_mean_p']:.4f}   "
          f"correctly <=0.05 = {overall['gold_nonhateful_correct_reject_at_0.05'] * 100:.1f}%   "
          f"false HATE at >=0.90 = {overall['gold_nonhateful_false_hate_rate_at_0.90'] * 100:.1f}%")

    return {"overall": overall, "by_functionality": by_func}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_all(cfg: dict, *, device: str | None, hatecheck: bool, batch_size: int) -> dict:
    handle = load_teacher(cfg, device=device)
    kw: dict[str, Any] = {"batch_size": batch_size, "min_batch_size": cfg["teacher"]["min_batch_size"]}

    print("=" * 100)
    print("TEACHER LOADED AND METADATA-VERIFIED")
    for k in (
        "repo_id", "revision", "device", "device_name", "fp16_forward",
        "softmax_dtype", "id2label", "num_labels", "max_position_embeddings",
        "torch_version", "deterministic_algorithms_enforced",
    ):
        print(f"  {k:<34}{handle.meta[k]}")
    print("=" * 100)

    # 1. PROBE - first, because it is the cheapest advance signal we have.
    probe = run_probe_tier(handle, **kw)

    # 2. HARD - orientation. Abort on failure.
    scored = {name: run_tier(handle, items, **kw) for name, (items, _, _) in TIERS.items()
              if name.startswith("HARD") or name.startswith("PINNED")}
    hard = check_hard_tier(scored["HARD_CLEAR_HATE"], scored["HARD_CLEAR_BENIGN"])

    print()
    print("=" * 100)
    print("HARD TIER - orientation guard (must pass)")
    print("=" * 100)
    _print_items("HARD_CLEAR_HATE", scored["HARD_CLEAR_HATE"])
    _print_items("HARD_CLEAR_BENIGN", scored["HARD_CLEAR_BENIGN"])
    print()
    print(f"  mean p_hate  clear-hate  = {hard['mean_p_hate_clear_hate']:.6f}")
    print(f"  mean p_hate  clear-benign= {hard['mean_p_hate_clear_benign']:.6f}")
    print(f"  margin                   = {hard['margin']:+.6f}   (must be > +0.5)")
    print(f"  rank separated           = {hard['rank_separated']}   "
          f"(min hate {hard['min_clear_hate']:.6f} > max benign {hard['max_clear_benign']:.6f})")
    print(f"  HARD TIER: {'PASS' if hard['passed'] else 'FAIL'}")

    if not hard["passed"]:
        raise SystemExit(
            "\nFATAL: the teacher failed the orientation canary. A negative or small "
            "margin means the model is inverted or is not the model we pinned. "
            "Aborting before scoring the corpus - an inverted mapping would be "
            "invisible in every downstream number."
        )

    # 3. PINNED - measure, report, assert only once pinned by a human.
    pinned = check_pinned_tier(scored, PINNED_BOUNDS)
    print()
    print("=" * 100)
    print("PINNED TIER - confidence bounds")
    print("=" * 100)
    _print_items("PINNED_HARD_NEG - should read non-hate; the cases shallow models fail",
                 scored["PINNED_HARD_NEG"])
    print()
    for k, v in pinned["observed"].items():
        p = pinned["pinned"].get(k)
        state = "UNPINNED - measure, then write into the fixture" if p is None else f"pinned={p}"
        print(f"  observed {k:<20} = {v:.6f}   [{state}]")
    if pinned["asserted"]:
        print(f"  PINNED TIER: {'PASS' if pinned['passed'] else 'FAIL'}")
        for f in pinned["failures"]:
            print(f"    * {f}")
        if not pinned["passed"]:
            raise SystemExit("FATAL: teacher drifted outside its pinned confidence bounds.")
    else:
        print("  PINNED TIER: NOT ASSERTED (bounds are None - report only, by design)")

    result = {
        "teacher_meta": handle.meta,
        "probe": probe,
        "hard": hard,
        "pinned": pinned,
        "scored_items": {k: v for k, v in scored.items()},
    }
    if hatecheck:
        result["hatecheck"] = run_hatecheck(handle, cfg, **kw)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    parser.add_argument("--hatecheck", action="store_true", help="also run the HateCheck suite")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    result = run_all(
        cfg, device=args.device, hatecheck=args.hatecheck, batch_size=args.batch_size
    )

    out = paths.artifacts / "teacher_canary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote artifacts/teacher_canary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
