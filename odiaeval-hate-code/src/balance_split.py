"""Stage 6 — balance HATE/NON_HATE, then assign deterministic stratified splits.

**Balance** by downsampling the majority class, never by upsampling the minority.
Duplicated minority rows in a small corpus produce optimistic validation scores
and there is no honest way to report around it.

The downsample is stratified across ``source_config``. Which stratification is a
``[decide + justify]`` point, because it decides whether balancing amplifies or
removes the source confound (both measured, both in the data card):

``proportional``    keep NON_HATE's own source mix. Non-restructuring, and the
                    default. Balancing *raises* NMI(source; label) because giving
                    the minority class equal weight amplifies its distinctive
                    source profile.
``match_minority``  give NON_HATE the same source mix as HATE, driving
                    NMI to 0 exactly - source then carries no label information.
                    Costs nothing in corpus size, but makes the corpus
                    register-monotone.

**Split** 80/10/10 by hashing the ``dedup_group``, per PROMPT.md §4⑥. Two
properties matter more than hitting the ratios exactly, and the priority order is
pinned in ``config.yaml``:

1. *cluster integrity* — no ``dedup_group`` in two splits. Hard.
2. *class balance* — 50/50. Hard.
3. *per-cell ratios* — best effort.

Pure hash bucketing is used rather than ranking within each cell. Ranking would
hit the per-cell ratios exactly, but adding one row would then shift others
across a boundary, violating CLAUDE.md §5.2 ("adding a row must not move existing
rows between splits"). Edit-stability wins; the achieved per-cell proportions are
reported so the drift is visible.

Run:  python -m src.balance_split [--stratify-mode match_minority]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src._common import Paths, append_stage_stats, file_sha256, load_config
from src.filter import cramers_v, mutual_information
from src.hashing import assign_split

STAGE = "06_balance_split"


# --------------------------------------------------------------------------- #
# balancing
# --------------------------------------------------------------------------- #
def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Allocate ``total`` across keys by weight, summing to exactly ``total``."""
    if not weights or total <= 0:
        return {k: 0 for k in weights}
    s = sum(weights.values())
    if s <= 0:
        return {k: 0 for k in weights}
    exact = {k: v / s * total for k, v in weights.items()}
    base = {k: int(v) for k, v in exact.items()}
    short = total - sum(base.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:short]:
        base[k] += 1
    return base


def target_counts(
    majority_by_source: dict[str, int],
    minority_by_source: dict[str, int],
    target_total: int,
    mode: str,
) -> dict[str, int]:
    """Per-source quota for the downsampled majority class."""
    if mode == "match_minority":
        weights = {k: float(v) for k, v in minority_by_source.items()}
    elif mode == "proportional":
        weights = {k: float(v) for k, v in majority_by_source.items()}
    else:
        raise SystemExit(
            "FATAL: balance.stratify_mode must be 'proportional' or "
            "'match_minority', got {!r}.".format(mode)
        )
    quota = largest_remainder(weights, target_total)

    # a quota can exceed what a source actually has; spill the excess onto the
    # sources that still have headroom, largest first, deterministically
    short = 0
    for k in list(quota):
        avail = majority_by_source.get(k, 0)
        if quota[k] > avail:
            short += quota[k] - avail
            quota[k] = avail
    while short > 0:
        headroom = {
            k: majority_by_source.get(k, 0) - quota[k]
            for k in quota
            if majority_by_source.get(k, 0) > quota[k]
        }
        if not headroom:
            break
        k = max(sorted(headroom), key=lambda x: headroom[x])
        take = min(short, headroom[k])
        quota[k] += take
        short -= take
    return quota


def downsample(rows: list[dict], quota: dict[str, int], seed: int) -> list[dict]:
    """Seeded, order-independent downsample to the per-source quota."""
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source_config"], []).append(r)
    picked: list[dict] = []
    for src in sorted(by_source):
        pool = sorted(by_source[src], key=lambda r: r["example_id"])
        n = min(quota.get(src, 0), len(pool))
        rng = random.Random(f"{seed}:{src}")
        picked.extend(pool if n >= len(pool) else rng.sample(pool, n))
    return picked


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def contingency(rows: list[dict], configs: list[str]) -> np.ndarray:
    c = Counter((r["label_str"], r["source_config"]) for r in rows)
    return np.array(
        [[c[("NON_HATE", s)], c[("HATE", s)]] for s in configs], dtype=np.float64
    )


def association(rows: list[dict], configs: list[str]) -> dict:
    tab = contingency(rows, configs)
    s = mutual_information(tab)
    s["cramers_v"] = cramers_v(tab)
    return s


def print_three_way(rows: list[dict], configs: list[str], splits: list[str]) -> dict:
    c = Counter((r["split"], r["label_str"], r["source_config"]) for r in rows)
    print()
    print("=" * 100)
    print("FINAL  split x label x source_config")
    print("-" * 100)
    header = "  {:<12}{:<10}".format("split", "label")
    for s in configs:
        header += "{:>16}".format(s)
    header += "{:>10}".format("total")
    print(header)
    print("-" * 100)
    out: dict = {}
    for sp in splits:
        for lab in ("HATE", "NON_HATE"):
            line = "  {:<12}{:<10}".format(sp, lab)
            tot = 0
            for s in configs:
                n = c[(sp, lab, s)]
                tot += n
                out[f"{sp}|{lab}|{s}"] = n
                line += "{:>16,}".format(n)
            line += "{:>10,}".format(tot)
            print(line)
        print("-" * 100)
    for sp in splits:
        n = sum(1 for r in rows if r["split"] == sp)
        print("  {:<22}{:>10,}  ({:.2f}%)".format(
            sp + " total", n, n / len(rows) * 100))
    print("=" * 100)
    return out


def print_cell_proportions(rows: list[dict], configs: list[str], splits: list[str]) -> dict:
    """Achieved per-cell split proportions — the 'best effort' third priority."""
    cells: dict[tuple, Counter] = {}
    for r in rows:
        cells.setdefault((r["label_str"], r["source_config"]), Counter())[r["split"]] += 1
    print()
    print("  achieved per-cell split proportions (pure hash bucketing, so these drift)")
    print("  {:<10}{:<16}{:>8}{:>10}{:>12}{:>10}".format(
        "label", "source", "n", "train%", "val%", "test%"))
    print("  " + "-" * 66)
    out = {}
    for cell in sorted(cells, key=str):
        cc = cells[cell]
        n = sum(cc.values())
        pcts = {sp: cc[sp] / n * 100 for sp in splits}
        out["|".join(cell)] = {"n": n, **{sp: cc[sp] for sp in splits}}
        print("  {:<10}{:<16}{:>8,}{:>9.1f}%{:>11.1f}%{:>9.1f}%".format(
            cell[0], cell[1], n, pcts["train"], pcts["validation"], pcts["test"]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stratify-mode", choices=["proportional", "match_minority"],
                        default=None)
    parser.add_argument("--all-modes", action="store_true",
                        help="build every stratification variant in one pass")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    if args.all_modes:
        for m in ("proportional", "match_minority"):
            rc = run_one(cfg, m)
            if rc:
                return rc
        return 0
    return run_one(cfg, args.stratify_mode or cfg["balance"]["stratify_mode"])


def run_one(cfg: dict, mode: str) -> int:
    paths = Paths.from_config(cfg).ensure()
    seed = cfg["seed"]
    configs = list(cfg["dataset"]["configs"])
    proportions = cfg["split"]["proportions"]
    splits = list(proportions)

    src_path = paths.interim / "deduped.parquet"
    if not src_path.exists():
        raise SystemExit("FATAL: data/interim/deduped.parquet missing - run Stage 5 first.")
    tbl = pq.read_table(src_path)
    rows = tbl.to_pylist()

    print("=" * 100)
    print("STAGE 6 - BALANCE + SPLIT   ({:,} deduplicated rows)".format(len(rows)))
    print("  balance        : {} , stratified across source_config".format(
        cfg["balance"]["method"]))
    print("  stratify mode  : {}".format(mode))
    print("  split          : {}  by hashing {}".format(
        dict(proportions), cfg["split"]["unit"]))
    print("=" * 100)

    by_label: dict[str, list[dict]] = {"HATE": [], "NON_HATE": []}
    for r in rows:
        by_label[r["label_str"]].append(r)
    minority_lab = min(by_label, key=lambda k: len(by_label[k]))
    majority_lab = "HATE" if minority_lab == "NON_HATE" else "NON_HATE"
    n_target = len(by_label[minority_lab])

    maj_by_src = Counter(r["source_config"] for r in by_label[majority_lab])
    min_by_src = Counter(r["source_config"] for r in by_label[minority_lab])
    quota = target_counts(dict(maj_by_src), dict(min_by_src), n_target, mode)

    print("\n  minority class : {} ({:,} rows) - kept in full".format(minority_lab, n_target))
    print("  majority class : {} ({:,} rows) - downsampled to {:,}".format(
        majority_lab, len(by_label[majority_lab]), n_target))
    print("\n  {:<16}{:>12}{:>12}{:>12}".format(
        "source", "available", "quota", "minority n"))
    for s in configs:
        print("  {:<16}{:>12,}{:>12,}{:>12,}".format(
            s, maj_by_src.get(s, 0), quota.get(s, 0), min_by_src.get(s, 0)))

    kept_majority = downsample(by_label[majority_lab], quota, seed)
    balanced = by_label[minority_lab] + kept_majority
    balanced.sort(key=lambda r: r["example_id"])

    if len(kept_majority) != n_target:
        raise SystemExit(
            "FATAL: downsample produced {:,} rows, expected {:,}.".format(
                len(kept_majority), n_target))

    # --- confound before / after balancing ------------------------------------ #
    before = association(rows, configs)
    after = association(balanced, configs)
    print()
    print("  confound (source_config ; label)")
    print("    {:<28}{:>12}{:>12}".format("", "NMI(arith)", "Cramer's V"))
    print("    {:<28}{:>12.4f}{:>12.4f}".format(
        "post-dedup, unbalanced", before["nmi_arithmetic"], before["cramers_v"]))
    print("    {:<28}{:>12.4f}{:>12.4f}".format(
        "after balancing ({})".format(mode), after["nmi_arithmetic"], after["cramers_v"]))
    if after["nmi_arithmetic"] > before["nmi_arithmetic"] + 1e-9:
        print("    NOTE: balancing RAISED the confound. See config.yaml balance."
              "stratify_mode;")
        print("          'match_minority' drives it to 0.0000 at no cost in corpus size.")

    # --- split ---------------------------------------------------------------- #
    salt = str(seed)
    for r in balanced:
        r["split"] = assign_split(r["dedup_group"], proportions, salt)

    # --- leakage check: THE assertion that protects the result ----------------- #
    groups: dict[str, str] = {}
    straddling = []
    for r in balanced:
        g, sp = r["dedup_group"], r["split"]
        if g in groups and groups[g] != sp:
            straddling.append(g)
        groups[g] = sp
    if straddling:
        raise SystemExit(
            "FATAL: {:,} dedup_group(s) appear in more than one split, e.g. {}. "
            "That is train/test leakage - refusing to write.".format(
                len(straddling), straddling[:3]))
    print("\n  LEAKAGE CHECK: {:,} dedup_groups across {:,} rows, none spanning two "
          "splits. PASS".format(len(groups), len(balanced)))

    three_way = print_three_way(balanced, configs, splits)
    cell_props = print_cell_proportions(balanced, configs, splits)

    label_counts = Counter(r["label_str"] for r in balanced)
    print("\n  class balance: HATE {:,}  NON_HATE {:,}  (ratio {:.4f})".format(
        label_counts["HATE"], label_counts["NON_HATE"],
        label_counts["HATE"] / label_counts["NON_HATE"]))

    # --- output ---------------------------------------------------------------- #
    schema = tbl.schema.append(pa.field("split", pa.string()))
    out = pa.Table.from_pylist(balanced, schema=schema)
    out_path = paths.interim / "split_{}.parquet".format(mode)
    pq.write_table(out, out_path, compression="zstd", compression_level=3)

    stats = {
        "stratify_mode": mode,
        "minority_label": minority_lab,
        "majority_label": majority_lab,
        "target_per_class": n_target,
        "majority_quota_by_source": dict(quota),
        "minority_by_source": dict(min_by_src),
        "association_before_balance": before,
        "association_after_balance": after,
        "three_way_counts": three_way,
        "per_cell_split_proportions": cell_props,
        "n_dedup_groups": len(groups),
        "dedup_groups_spanning_splits": 0,
        "class_balance": dict(label_counts),
        "split_totals": {sp: sum(1 for r in balanced if r["split"] == sp) for sp in splits},
    }
    append_stage_stats(
        paths.artifacts, "{}_{}".format(STAGE, mode),
        rows_in=len(rows), rows_out=len(balanced),
        dropped_by_reason={"majority_class_downsampled": len(rows) - len(balanced)},
        extra={**stats, "output_sha256": file_sha256(out_path)},
    )
    dist_path = paths.artifacts / "label_distribution.json"
    d = json.loads(dist_path.read_text(encoding="utf-8")) if dist_path.exists() else {}
    d.setdefault("balance_split", {})[mode] = stats
    dist_path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    print()
    print("wrote data/interim/{}  {:,} rows".format(out_path.name, len(balanced)))
    print("row accounting: {:,} in = {:,} out + {:,} downsampled away".format(
        len(rows), len(balanced), len(rows) - len(balanced)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
