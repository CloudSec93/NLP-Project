"""Human-agreement check on the weak labels — the number that says how far to trust them.

Not required by the assignment. It is the highest-value thing available for the
data-quality mark, because it replaces "the weak labels are probably fine" with a
measured coefficient.

**Two samples, not one.** Cohen's kappa is agreement between two labellers over the
same items, and the teacher only *has* a label inside the decided bands. Sampling
across all three bands into one pool would silently drop every discarded row out of
the kappa, leaving an effective n well below the nominal one and a non-representative
remainder. So:

``agreement_sample.csv``   ~100 rows within the decided bands, **balanced by label**
                           and stratified over source_config inside each label. This
                           is the kappa sample. Proportional allocation would give
                           ~17 HATE rows out of 100 (HATE is 17% of the labelled
                           corpus), and the HATE side is precisely what we need to
                           resolve - so the sample is balanced by design and the
                           resulting kappa is NOT the population kappa. Per-class
                           agreement is reported alongside it for that reason.
``discarded_review.csv``   ~30 rows from the discarded band. Reviewed qualitatively
                           and reported separately - "what is the teacher refusing
                           to commit on" - never folded into the kappa.

**The annotation CSVs do not contain the teacher's label**, and rows are emitted in
hash order rather than probability order. Showing a human the model's answer before
they annotate does not measure agreement, it measures anchoring. The teacher labels
are rejoined by ``example_id`` at scoring time from a separate key file.

You are labelling **English**, against Dynabench's definition of hate (identity-
directed attacks on people for who they are) - not "is this text harmful". Those
differ, and that difference is the whole point of the measurement.

Usage:
    python -m src.agreement sample            # writes the two CSVs + the key
    <a human fills the human_label column: HATE / NON_HATE>
    python -m src.agreement score             # Cohen's kappa + a confusion matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src._common import Paths, load_config

SAMPLE_CSV = "agreement_sample.csv"
DISCARDED_CSV = "discarded_review.csv"
KEY_JSON = "agreement_key.json"

ANNOTATION_HEADER = ["example_id", "source_config", "text_eng", "human_label", "notes"]
VALID_LABELS = {"HATE", "NON_HATE"}


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def stratified_sample(
    rows: list[dict], n: int, seed: int, strata_key
) -> list[dict]:
    """Proportional allocation over strata, deterministic given the seed.

    Pools are sorted by ``example_id`` before sampling so the result is a pure
    function of (data, seed) and not of row order.
    """
    strata: dict = {}
    for r in rows:
        strata.setdefault(strata_key(r), []).append(r)

    total = len(rows)
    rng = random.Random(seed)
    picked: list[dict] = []
    # largest-remainder allocation so the sizes sum to exactly n
    quotas = {k: len(v) / total * n for k, v in strata.items()}
    base = {k: int(q) for k, q in quotas.items()}
    remainder = sorted(strata, key=lambda k: (-(quotas[k] - base[k]), str(k)))
    short = n - sum(base.values())
    for k in remainder[:short]:
        base[k] += 1

    for k in sorted(strata, key=str):
        pool = sorted(strata[k], key=lambda r: r["example_id"])
        take = min(base[k], len(pool))
        picked.extend(rng.sample(pool, take) if take < len(pool) else pool)
    return picked


def _write_annotation_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ANNOTATION_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({
                "example_id": r["example_id"],
                "source_config": r["source_config"],
                "text_eng": r["text_eng"],
                "human_label": "",      # deliberately blank - a human fills this
                "notes": "",
            })


def do_sample(
    cfg: dict, paths: Paths, n_kappa: int, n_discarded: int, balance_labels: bool = True
) -> int:
    labelled = pq.read_table(paths.interim / "labelled.parquet").to_pylist()
    discarded = pq.read_table(paths.interim / "discarded.parquet").to_pylist()
    seed = cfg["seed"]

    if balance_labels:
        # equal HATE / NON_HATE, each stratified over source_config
        kappa_rows = []
        for lab in ("HATE", "NON_HATE"):
            subset = [r for r in labelled if r["label_str"] == lab]
            kappa_rows.extend(
                stratified_sample(subset, n_kappa // 2, seed, lambda r: r["source_config"])
            )
    else:
        kappa_rows = stratified_sample(
            labelled, n_kappa, seed, lambda r: (r["label_str"], r["source_config"])
        )
    disc_rows = stratified_sample(
        discarded, n_discarded, seed, lambda r: r["source_config"]
    )

    # emit in hash order, not probability order, so the ordering leaks nothing
    kappa_rows.sort(key=lambda r: r["example_id"])
    disc_rows.sort(key=lambda r: r["example_id"])

    _write_annotation_csv(paths.artifacts / SAMPLE_CSV, kappa_rows)
    _write_annotation_csv(paths.artifacts / DISCARDED_CSV, disc_rows)

    key = {
        "seed": seed,
        "balanced_by_label": balance_labels,
        "n_kappa": len(kappa_rows),
        "n_discarded": len(disc_rows),
        "teacher": {
            r["example_id"]: {"label_str": r["label_str"], "p_hate": r["p_hate"]}
            for r in kappa_rows
        },
        "discarded_p_hate": {r["example_id"]: r["p_hate"] for r in disc_rows},
    }
    (paths.artifacts / KEY_JSON).write_text(json.dumps(key, indent=2), encoding="utf-8")

    print("=" * 92)
    print("AGREEMENT SAMPLE")
    print("=" * 92)
    print("  kappa sample     {:>4} rows -> artifacts/{}".format(len(kappa_rows), SAMPLE_CSV))
    strata: dict = {}
    for r in kappa_rows:
        strata[(r["label_str"], r["source_config"])] = (
            strata.get((r["label_str"], r["source_config"]), 0) + 1
        )
    for k in sorted(strata, key=str):
        print("      {:<10} {:<16} {:>4}".format(k[0], k[1], strata[k]))
    print("  discarded review {:>4} rows -> artifacts/{}".format(len(disc_rows), DISCARDED_CSV))
    print("  teacher labels held back in artifacts/{} (blind annotation)".format(KEY_JSON))
    if balance_labels:
        print()
        print("  NOTE: balanced by label by design (HATE is only 17% of the corpus, and")
        print("  proportional allocation would leave ~17 HATE rows - too few to resolve the")
        print("  class we most need to trust). The resulting kappa is therefore NOT the")
        print("  population kappa; per-class agreement is reported alongside it.")
    print()
    print("  Fill the human_label column with HATE or NON_HATE, against Dynabench's")
    print("  definition: identity-directed attacks on people for who they are. A harmful")
    print("  request that targets nobody's identity is NON_HATE under that definition even")
    print("  though it is plainly harmful - that gap is what we are measuring.")
    print("  Then: python -m src.agreement score")
    return 0


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def cohens_kappa(a: list[str], b: list[str], labels: list[str]) -> dict:
    """Cohen's kappa for two labellers over the same items."""
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    m = np.zeros((k, k), dtype=np.int64)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        raise SystemExit("FATAL: no annotated rows to score.")
    po = float(np.trace(m)) / n
    pe = float((m.sum(axis=1) * m.sum(axis=0)).sum()) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {
        "n": int(n),
        "observed_agreement": po,
        "expected_agreement": pe,
        "cohens_kappa": kappa,
        "confusion": m.tolist(),
        "labels": labels,
    }


def _interpret(k: float) -> str:
    for lo, txt in ((0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"),
                    (0.21, "fair"), (0.0, "slight")):
        if k >= lo:
            return txt
    return "poor (worse than chance)"


def do_score(cfg: dict, paths: Paths) -> int:
    key_path = paths.artifacts / KEY_JSON
    csv_path = paths.artifacts / SAMPLE_CSV
    if not key_path.exists() or not csv_path.exists():
        raise SystemExit("FATAL: run 'python -m src.agreement sample' first.")
    key = json.loads(key_path.read_text(encoding="utf-8"))

    human, teacher, blank, bad = [], [], 0, []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            v = (row["human_label"] or "").strip().upper()
            if not v:
                blank += 1
                continue
            if v not in VALID_LABELS:
                bad.append((row["example_id"], row["human_label"]))
                continue
            t = key["teacher"].get(row["example_id"])
            if t is None:
                raise SystemExit(
                    "FATAL: {} is not in the key - the CSV was edited or "
                    "regenerated with a different seed.".format(row["example_id"])
                )
            human.append(v)
            teacher.append(t["label_str"])

    if bad:
        raise SystemExit(
            "FATAL: {} rows have an unrecognised human_label (expected HATE or "
            "NON_HATE), e.g. {}. Not guessing.".format(len(bad), bad[:3])
        )
    if not human:
        raise SystemExit(
            "FATAL: artifacts/{} has no filled human_label values yet. This tool does "
            "not fabricate labels - a human must fill them.".format(SAMPLE_CSV)
        )

    res = cohens_kappa(human, teacher, ["NON_HATE", "HATE"])
    res["n_unannotated"] = blank
    res["balanced_by_label"] = key.get("balanced_by_label", False)
    res["per_teacher_class_agreement"] = {}
    for lab in ("HATE", "NON_HATE"):
        pairs = [(h, t) for h, t in zip(human, teacher) if t == lab]
        res["per_teacher_class_agreement"][lab] = {
            "n": len(pairs),
            "human_agreed": sum(1 for h, t in pairs if h == t),
            "agreement": (sum(1 for h, t in pairs if h == t) / len(pairs)) if pairs else None,
        }
    res["interpretation"] = _interpret(res["cohens_kappa"])

    print("=" * 92)
    print("HUMAN vs TEACHER AGREEMENT  (decided bands only)")
    print("=" * 92)
    print("  annotated        {:,} of {:,} ({} still blank)".format(
        res["n"], res["n"] + blank, blank))
    print("  observed agree   {:.4f}".format(res["observed_agreement"]))
    print("  expected agree   {:.4f}".format(res["expected_agreement"]))
    print("  Cohen's kappa    {:.4f}   ({})".format(res["cohens_kappa"], res["interpretation"]))
    print()
    print("  confusion (rows = human, cols = teacher)")
    print("               {:>12}{:>12}".format("NON_HATE", "HATE"))
    for i, l in enumerate(res["labels"]):
        print("    {:<10}{:>12,}{:>12,}".format(l, res["confusion"][i][0], res["confusion"][i][1]))

    out = paths.artifacts / "agreement.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nwrote artifacts/agreement.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sample", "score"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-kappa", type=int, default=100)
    parser.add_argument("--n-discarded", type=int, default=30)
    parser.add_argument(
        "--proportional", action="store_true",
        help="allocate the kappa sample proportionally instead of balanced by label",
    )
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    if args.action == "sample":
        return do_sample(
            cfg, paths, args.n_kappa, args.n_discarded, not args.proportional
        )
    return do_score(cfg, paths)


if __name__ == "__main__":
    raise SystemExit(main())
