"""Stage 4 — apply the teacher thresholds and measure what we actually have.

Banding (both bounds inclusive, per the assignment's "at least" / "at most"):

    p_hate >= 0.90  ->  HATE      (label = 1)
    p_hate <= 0.05  ->  NON_HATE  (label = 0)
    otherwise       ->  discarded

Thresholding uses the **rounded** ``p_hate`` (see Stage 3). Discarded rows are
written out rather than dropped — they are evidence about how the teacher behaves
on out-of-distribution input, not waste.

Then the three numbers that decide whether the project works:

1. the yield table (how big the final dataset can be),
2. the label x source_config contingency, with normalised mutual information —
   if ``source_config`` *is* the label, the student learns to detect which corpus
   a sentence came from rather than whether it is hateful,
3. band samples, including the discarded band.

Run:  python -m src.filter
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src._common import Paths, append_stage_stats, file_sha256, load_config

STAGE = "04_filter"

LABEL_HATE, LABEL_NON_HATE = 1, 0
LABEL_STR = {1: "HATE", 0: "NON_HATE"}

# Escalation rule on the HATE yield (spec). A low yield is a result, not a failure.
YIELD_FINE = 10_000
YIELD_WORKABLE = 2_000


# --------------------------------------------------------------------------- #
# banding
# --------------------------------------------------------------------------- #
def assign_label(p_hate: float, hate_min: float, non_hate_max: float) -> int | None:
    """HATE / NON_HATE / None (discard). Both bounds inclusive."""
    if p_hate >= hate_min:
        return LABEL_HATE
    if p_hate <= non_hate_max:
        return LABEL_NON_HATE
    return None


def assign_labels(probs: np.ndarray, hate_min: float, non_hate_max: float) -> np.ndarray:
    """Vectorised banding. Returns 1 / 0 / -1, where -1 means discard."""
    out = np.full(len(probs), -1, dtype=np.int8)
    out[probs >= hate_min] = LABEL_HATE
    out[probs <= non_hate_max] = LABEL_NON_HATE
    return out


# --------------------------------------------------------------------------- #
# association statistics (hand-rolled: one dependency fewer, and the
# normalisation is then explicit rather than a library default)
# --------------------------------------------------------------------------- #
def _entropy(counts: np.ndarray) -> float:
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum())


def mutual_information(table: np.ndarray) -> dict:
    """MI and NMI for a contingency table, with the normaliser stated.

    ``nmi_arithmetic`` is ``2*I / (H(X) + H(Y))`` — the same normalisation
    scikit-learn uses by default. The other two normalisers are reported as well
    because they disagree materially when the classes are lopsided, and ours are.
    """
    n = table.sum()
    if n == 0:
        return {"mi": 0.0, "nmi_arithmetic": 0.0, "nmi_min": 0.0, "nmi_max": 0.0}
    pij = table / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = pij * np.log(pij / (pi * pj))
    mi = float(np.nansum(np.where(pij > 0, term, 0.0)))
    hx = _entropy(table.sum(axis=1))
    hy = _entropy(table.sum(axis=0))
    return {
        "mi_nats": mi,
        "entropy_source": hx,
        "entropy_label": hy,
        "nmi_arithmetic": (2 * mi / (hx + hy)) if (hx + hy) > 0 else 0.0,
        "nmi_min": (mi / min(hx, hy)) if min(hx, hy) > 0 else 0.0,
        "nmi_max": (mi / max(hx, hy)) if max(hx, hy) > 0 else 0.0,
    }


def cramers_v(table: np.ndarray) -> float:
    n = table.sum()
    if n == 0:
        return 0.0
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)))
    k = min(table.shape) - 1
    return math.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0.0


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_yield_table(per_config: dict, configs: list[str]) -> None:
    print()
    print("=" * 100)
    print("YIELD TABLE")
    print("-" * 100)
    print("  {:<16}{:>10}{:>18}{:>20}{:>18}".format(
        "config", "total", "HATE >=0.90", "NON_HATE <=0.05", "discarded"))
    print("-" * 100)
    tot = {"total": 0, "hate": 0, "non_hate": 0, "discarded": 0}
    for c in configs:
        d = per_config[c]
        for k in tot:
            tot[k] += d[k]
        print("  {:<16}{:>10,}{:>18}{:>20}{:>18}".format(
            c, d["total"],
            "{:,} ({:.1f}%)".format(d["hate"], d["hate"] / d["total"] * 100),
            "{:,} ({:.1f}%)".format(d["non_hate"], d["non_hate"] / d["total"] * 100),
            "{:,} ({:.1f}%)".format(d["discarded"], d["discarded"] / d["total"] * 100)))
    print("-" * 100)
    print("  {:<16}{:>10,}{:>18}{:>20}{:>18}".format(
        "TOTAL", tot["total"],
        "{:,} ({:.1f}%)".format(tot["hate"], tot["hate"] / tot["total"] * 100),
        "{:,} ({:.1f}%)".format(tot["non_hate"], tot["non_hate"] / tot["total"] * 100),
        "{:,} ({:.1f}%)".format(tot["discarded"], tot["discarded"] / tot["total"] * 100)))
    print("=" * 100)


def print_contingency(table: np.ndarray, configs: list[str], stats: dict) -> None:
    print()
    print("=" * 100)
    print("LABEL x SOURCE_CONFIG  (labelled rows only; discarded excluded)")
    print("-" * 100)
    print("  {:<16}{:>12}{:>14}{:>10}{:>16}".format(
        "config", "HATE", "NON_HATE", "total", "HATE rate"))
    print("-" * 100)
    for i, c in enumerate(configs):
        h, nh = int(table[i, 1]), int(table[i, 0])
        t = h + nh
        print("  {:<16}{:>12,}{:>14,}{:>10,}{:>15.2f}%".format(c, h, nh, t, h / t * 100 if t else 0))
    h, nh = int(table[:, 1].sum()), int(table[:, 0].sum())
    print("-" * 100)
    print("  {:<16}{:>12,}{:>14,}{:>10,}{:>15.2f}%".format(
        "TOTAL", h, nh, h + nh, h / (h + nh) * 100))
    print()
    print("  share of each class coming from one source:")
    for j, name in ((1, "HATE"), (0, "NON_HATE")):
        col = table[:, j]
        tot = col.sum()
        shares = sorted(
            ((configs[i], col[i] / tot * 100) for i in range(len(configs))),
            key=lambda kv: -kv[1],
        )
        line = "   ".join("{} {:.1f}%".format(n, s) for n, s in shares)
        print("    {:<10} {}".format(name, line))
        print("    {:<10} -> largest single source = {:.1f}%".format("", shares[0][1]))
    print()
    print("  mutual information (source_config ; label)")
    print("    MI                     {:.6f} nats".format(stats["mi_nats"]))
    print("    NMI (arithmetic)       {:.6f}   <- sklearn default normalisation".format(
        stats["nmi_arithmetic"]))
    print("    NMI (min-entropy)      {:.6f}".format(stats["nmi_min"]))
    print("    NMI (max-entropy)      {:.6f}".format(stats["nmi_max"]))
    print("    Cramer's V             {:.6f}".format(stats["cramers_v"]))
    print("=" * 100)


def print_band_samples(rows: list[dict], seed: int, n: int) -> None:
    """Deterministic: pool sorted by example_id, then a seeded sample."""
    bands = {"HATE": [], "NON_HATE": [], "DISCARDED": []}
    for r in rows:
        bands[r["band"]].append(r)
    for band, pool in bands.items():
        pool.sort(key=lambda x: x["example_id"])
        print()
        print("=" * 100)
        print("BAND SAMPLE - {}   (pool = {:,})".format(band, len(pool)))
        print("=" * 100)
        if not pool:
            print("  (none)")
            continue
        rng = random.Random(seed)
        sample = pool if len(pool) <= n else rng.sample(pool, n)
        for r in sorted(sample, key=lambda x: -x["p_hate"]):
            print("\n  p_hate={:.6f}  [{}]  {}".format(
                r["p_hate"], r["source_config"], r["example_id"]))
            print("    eng: {}".format(r["text_eng"][:220]))
            print("    ory: {}".format(r["text_ory"][:220]))


def escalation_verdict(n_hate: int) -> tuple[str, str]:
    if n_hate >= YIELD_FINE:
        return "FINE", (
            "HATE yield >= {:,}. Proceed; note the yield and move on.".format(YIELD_FINE)
        )
    if n_hate >= YIELD_WORKABLE:
        return "WORKABLE", (
            "HATE yield in [{:,}, {:,}). A balanced corpus of {:,}-{:,} is small but "
            "adequate for fine-tuning a 278M model. Proceed, and record that the small "
            "corpus raises variance in the final numbers.".format(
                YIELD_WORKABLE, YIELD_FINE, YIELD_WORKABLE * 2, YIELD_FINE * 2)
        )
    return "STOP", (
        "HATE yield < {:,}. STOP - do not adjust anything. This is a question for the "
        "group and then the professor: report the low yield as a genuine finding and "
        "proceed with what we have, or ask whether the thresholds may be relaxed with "
        "the change documented. Quietly loosening the gate is not an option.".format(
            YIELD_WORKABLE)
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    hi, lo = cfg["thresholds"]["hate_min"], cfg["thresholds"]["non_hate_max"]
    configs = list(cfg["dataset"]["configs"])

    scored_path = paths.interim / "scored.parquet"
    if not scored_path.exists():
        raise SystemExit("FATAL: data/interim/scored.parquet missing - run Stage 3 first.")
    scored = pq.read_table(scored_path)
    probs = np.array(scored.column("p_hate").to_pylist(), dtype=np.float64)
    src = scored.column("source_config").to_pylist()

    print("=" * 100)
    print("STAGE 4 - THRESHOLD   ({:,} scored rows)".format(len(probs)))
    print("  HATE     p_hate >= {}".format(hi))
    print("  NON_HATE p_hate <= {}   (both bounds inclusive)".format(lo))
    print("=" * 100)

    labels = assign_labels(probs, hi, lo)
    keep = labels >= 0

    # --- outputs ------------------------------------------------------------- #
    label_arr = pa.array(np.where(keep, labels, 0).astype(np.int8), pa.int8())
    label_str = pa.array([LABEL_STR.get(int(x), "") for x in labels], pa.string())
    tagged = scored.append_column("label", label_arr).append_column("label_str", label_str)

    labelled = tagged.filter(pa.array(keep))
    discarded = tagged.filter(pa.array(~keep)).drop_columns(["label", "label_str"])
    out_lab = paths.interim / "labelled.parquet"
    out_dis = paths.interim / "discarded.parquet"
    pq.write_table(labelled, out_lab, compression="zstd", compression_level=3)
    pq.write_table(discarded, out_dis, compression="zstd", compression_level=3)

    # --- yield table --------------------------------------------------------- #
    per_config = {}
    for c in configs:
        m = np.array([s == c for s in src])
        per_config[c] = {
            "total": int(m.sum()),
            "hate": int((labels[m] == LABEL_HATE).sum()),
            "non_hate": int((labels[m] == LABEL_NON_HATE).sum()),
            "discarded": int((labels[m] == -1).sum()),
        }
    print_yield_table(per_config, configs)

    # --- contingency + association ------------------------------------------- #
    table = np.zeros((len(configs), 2), dtype=np.int64)
    for i, c in enumerate(configs):
        table[i, 1] = per_config[c]["hate"]
        table[i, 0] = per_config[c]["non_hate"]
    stats = mutual_information(table.astype(np.float64))
    stats["cramers_v"] = cramers_v(table.astype(np.float64))
    print_contingency(table, configs, stats)

    single_source = {}
    for j, name in ((1, "HATE"), (0, "NON_HATE")):
        col = table[:, j].astype(np.float64)
        tot = col.sum()
        single_source[name] = {
            configs[i]: float(col[i] / tot * 100) if tot else 0.0 for i in range(len(configs))
        }
    max_non_hate_share = max(single_source["NON_HATE"].values())

    print()
    if max_non_hate_share >= 95.0:
        print("!" * 100)
        print("NON_HATE is {:.1f}% single-source. This is the largest single threat to the "
              "group's conclusion:".format(max_non_hate_share))
        print("source_config is effectively the label, and the student can score well by "
              "detecting which")
        print("corpus a sentence came from rather than whether it is hateful.")
        print("!" * 100)
    else:
        print("  NON_HATE is {:.1f}% single-source (largest share), below the 95% alarm "
              "threshold.".format(max_non_hate_share))

    # --- band samples --------------------------------------------------------- #
    rows = []
    tl = tagged.to_pylist()
    for r, lab in zip(tl, labels):
        rows.append({
            "example_id": r["example_id"],
            "source_config": r["source_config"],
            "text_eng": r["text_eng"],
            "text_ory": r["text_ory"],
            "p_hate": r["p_hate"],
            "band": "DISCARDED" if lab == -1 else LABEL_STR[int(lab)],
        })
    print_band_samples(rows, cfg["seed"], args.samples)

    # --- escalation ----------------------------------------------------------- #
    n_hate = int((labels == LABEL_HATE).sum())
    n_non = int((labels == LABEL_NON_HATE).sum())
    n_dis = int((labels == -1).sum())
    verdict, note = escalation_verdict(n_hate)
    print()
    print("=" * 100)
    print("HATE YIELD VERDICT: {}".format(verdict))
    print("  {}".format(note))
    print("  balanced corpus ceiling (before dedup) = 2 x {:,} = {:,} rows".format(
        min(n_hate, n_non), 2 * min(n_hate, n_non)))
    print("=" * 100)

    # --- artefacts ------------------------------------------------------------ #
    dist_path = paths.artifacts / "label_distribution.json"
    dist = json.loads(dist_path.read_text(encoding="utf-8")) if dist_path.exists() else {}
    dist.update({
        "thresholds": {"hate_min": hi, "non_hate_max": lo, "bounds": "both inclusive"},
        "yield_per_config": per_config,
        "yield_total": {
            "total": len(probs), "hate": n_hate, "non_hate": n_non, "discarded": n_dis
        },
        "contingency_label_x_source": {
            "configs": configs,
            "columns": ["NON_HATE", "HATE"],
            "table": table.tolist(),
        },
        "association": stats,
        "class_source_shares_pct": single_source,
        "max_non_hate_single_source_pct": max_non_hate_share,
        "hate_yield_verdict": verdict,
    })
    dist_path.write_text(json.dumps(dist, indent=2), encoding="utf-8")

    append_stage_stats(
        paths.artifacts, STAGE,
        rows_in=len(probs), rows_out=int(keep.sum()),
        dropped_by_reason={"teacher_uncertain_0.05_to_0.90": n_dis},
        extra={
            "yield_per_config": per_config,
            "association": stats,
            "max_non_hate_single_source_pct": max_non_hate_share,
            "hate_yield_verdict": verdict,
            "output_labelled_sha256": file_sha256(out_lab),
            "output_discarded_sha256": file_sha256(out_dis),
        },
    )

    print()
    print("wrote data/interim/labelled.parquet   {:,} rows".format(int(keep.sum())))
    print("wrote data/interim/discarded.parquet  {:,} rows (evidence, not waste)".format(n_dis))
    print("wrote artifacts/label_distribution.json")
    print("\nrow accounting: {:,} in = {:,} labelled + {:,} discarded".format(
        len(probs), int(keep.sum()), n_dis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
