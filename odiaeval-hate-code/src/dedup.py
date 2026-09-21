"""Stage 5 — exact and near-duplicate deduplication on the Odia text.

``Toxic_Matrix`` is model-generated and heavily templated: many prompts differ
only in the target noun. Exact-string dedup will not catch that, and a template
family split across train and test is straightforward leakage that inflates both
the benchmark score and therefore the reported translationese gap.

Pipeline:

1. **Exact** dedup on a normalised form (NFC, zero-width stripped, whitespace
   collapsed). Rows are grouped by normalised text, so this also collapses the
   MinHash work to one signature per *distinct* text.
2. **Near-duplicate** clustering with MinHash + LSH over character 5-grams,
   Jaccard 0.85. Candidate pairs from LSH are re-checked against the MinHash
   Jaccard estimate before being accepted, so band collisions do not merge
   unrelated documents.
3. Connected components over the verified pair graph, giving clusters.
4. One representative kept per cluster; ``dedup_group`` recorded on it.

**Determinism.** ``datasketch``'s streaming insertion order can affect an online
clustering, so clusters are built from the *full* verified pair graph with
union-find over pairs iterated in sorted order, and MinHash permutations are
seeded. Same input, same clusters, every run.

**Cluster tie-break.** A cluster whose members carry more than one
``(label, source_config)`` combination makes cluster-integrity and strict
stratification mutually unsatisfiable. Resolved by ``dedup.cluster_cell_assignment``
= ``majority``: the whole cluster is assigned its majority cell, and the surviving
representative is the lowest ``example_id`` *within that cell* — so a cluster that
is 4 HATE and 1 NON_HATE never survives as its NON_HATE member. The number of
affected clusters is reported.

Dedup runs on the **Odia** text, which is what the model sees. The same clustering
is also run on the English side and reported — a large gap between the two counts
is itself evidence about translation quality.

Run:  python -m src.dedup [--skip-english]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from src._common import Paths, append_stage_stats, file_sha256, load_config
from src.textnorm import normalise_for_dedup

STAGE = "05_dedup"


# --------------------------------------------------------------------------- #
# union-find
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # deterministic: the smaller key always becomes the root
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


# --------------------------------------------------------------------------- #
# shingling + clustering
# --------------------------------------------------------------------------- #
def shingles(text: str, k: int) -> list[bytes]:
    """Character k-grams of the normalised text."""
    s = normalise_for_dedup(text)
    if len(s) < k:
        return [s.encode("utf-8")] if s else []
    return list({s[i : i + k].encode("utf-8") for i in range(len(s) - k + 1)})


def cluster_texts(
    texts: dict[str, str], *, num_perm: int, char_ngram: int, threshold: float, seed: int
) -> tuple[dict[str, str], dict]:
    """Cluster ``{key: text}`` by near-duplicate similarity.

    Returns ``({key: cluster_root_key}, stats)``. Cluster roots are the smallest
    key in each component, so the labelling is a pure function of the input set.
    """
    from datasketch import MinHash, MinHashLSH

    keys = sorted(texts)
    t0 = time.time()
    sigs: dict[str, "MinHash"] = {}
    for i, k in enumerate(keys):
        m = MinHash(num_perm=num_perm, seed=seed)
        sh = shingles(texts[k], char_ngram)
        if sh:
            m.update_batch(sh)
        sigs[k] = m
        if i % 20000 == 0 and i:
            print(f"    minhash {i:,}/{len(keys):,}", flush=True)
    t_minhash = time.time() - t0

    t0 = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    with lsh.insertion_session() as sess:
        for k in keys:
            sess.insert(k, sigs[k])

    # collect the full verified pair graph, then union-find in sorted order
    pairs: set[tuple[str, str]] = set()
    for k in keys:
        for cand in lsh.query(sigs[k]):
            if cand == k:
                continue
            a, b = (k, cand) if k < cand else (cand, k)
            if (a, b) in pairs:
                continue
            # re-check: LSH gives candidates, not guarantees
            if sigs[a].jaccard(sigs[b]) >= threshold:
                pairs.add((a, b))
    uf = UnionFind(keys)
    for a, b in sorted(pairs):
        uf.union(a, b)
    t_lsh = time.time() - t0

    assignment = {k: uf.find(k) for k in keys}
    sizes = Counter(assignment.values())
    return assignment, {
        "n_texts": len(keys),
        "n_clusters": len(sizes),
        "n_multi_text_clusters": sum(1 for v in sizes.values() if v > 1),
        "n_verified_pairs": len(pairs),
        "minhash_seconds": round(t_minhash, 1),
        "lsh_seconds": round(t_lsh, 1),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def select_representative(members: list[dict]) -> tuple[dict, tuple, bool]:
    """Pick the surviving row for one cluster, by the majority-cell rule.

    Majority ``(label, source_config)`` cell wins; on a tie the cell of the
    lowest ``example_id`` wins; failing that, sorted order — so the choice is a
    pure function of the cluster's contents and never of row order.

    Returns ``(representative, majority_cell, spans_multiple_cells)``.
    """
    cells = Counter((m["label"], m["source_config"]) for m in members)
    max_count = max(cells.values())
    tied = [c for c, n in cells.items() if n == max_count]
    lowest = min(members, key=lambda m: m["example_id"])
    lowest_cell = (lowest["label"], lowest["source_config"])
    maj_cell = lowest_cell if lowest_cell in tied else min(tied)
    eligible = [m for m in members if (m["label"], m["source_config"]) == maj_cell]
    rep = min(eligible, key=lambda m: m["example_id"])
    return rep, maj_cell, len(cells) > 1


def threshold_sweep(
    texts: dict[str, str], thresholds: list[float], *, num_perm: int, char_ngram: int, seed: int
) -> list[dict]:
    """How much near-duplicate merging each Jaccard threshold would buy.

    The configured 0.85 is fixed by the spec. This does not change it - it
    records what the alternatives would have given, so the choice is defensible
    in the data card rather than merely asserted. Signatures are computed once
    and reused; only the LSH banding changes.
    """
    from datasketch import MinHash, MinHashLSH

    keys = sorted(texts)
    sigs = {}
    for k in keys:
        m = MinHash(num_perm=num_perm, seed=seed)
        sh = shingles(texts[k], char_ngram)
        if sh:
            m.update_batch(sh)
        sigs[k] = m

    out = []
    for th in thresholds:
        lsh = MinHashLSH(threshold=th, num_perm=num_perm)
        with lsh.insertion_session() as sess:
            for k in keys:
                sess.insert(k, sigs[k])
        pairs = set()
        for k in keys:
            for cand in lsh.query(sigs[k]):
                if cand == k:
                    continue
                a, b = (k, cand) if k < cand else (cand, k)
                if (a, b) not in pairs and sigs[a].jaccard(sigs[b]) >= th:
                    pairs.add((a, b))
        uf = UnionFind(keys)
        for a, b in sorted(pairs):
            uf.union(a, b)
        n_clusters = len({uf.find(k) for k in keys})
        merged = len(keys) - n_clusters
        out.append({
            "threshold": th,
            "n_texts": len(keys),
            "n_clusters": n_clusters,
            "texts_merged": merged,
            "pct_reduction": merged / len(keys) * 100 if keys else 0.0,
        })
    return out


def size_distribution(sizes: Counter) -> dict[str, int]:
    buckets = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, "11-50": 0, "51-100": 0, "100+": 0}
    for n in sizes.values():
        if n == 1:
            buckets["1"] += 1
        elif n == 2:
            buckets["2"] += 1
        elif n <= 5:
            buckets["3-5"] += 1
        elif n <= 10:
            buckets["6-10"] += 1
        elif n <= 50:
            buckets["11-50"] += 1
        elif n <= 100:
            buckets["51-100"] += 1
        else:
            buckets["100+"] += 1
    return buckets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-english", action="store_true",
                        help="skip the English-side comparison clustering")
    parser.add_argument("--sweep", type=int, default=0, metavar="N",
                        help="also report a Jaccard threshold sensitivity sweep over an "
                             "N-text sample (0 = off)")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()
    dcfg = cfg["dedup"]
    mh = dcfg["minhash"]
    seed = cfg["seed"]

    src_path = paths.interim / "labelled.parquet"
    if not src_path.exists():
        raise SystemExit("FATAL: data/interim/labelled.parquet missing - run Stage 4 first.")
    tbl = pq.read_table(src_path)
    rows = tbl.to_pylist()

    print("=" * 100)
    print("STAGE 5 - DEDUPLICATION   ({:,} labelled rows)".format(len(rows)))
    print("  exact form   : NFC + zero-width stripped + whitespace collapsed")
    print("  near-dup     : MinHash {} perms, char {}-grams, Jaccard >= {}".format(
        mh["num_perm"], mh["char_ngram"], mh["jaccard_threshold"]))
    print("  dedup on     : {}  (the text the model sees)".format(dcfg["dedup_language"]))
    print("  cluster cell : {}".format(dcfg["cluster_cell_assignment"]))
    print("=" * 100)

    # --- 1. exact dedup ------------------------------------------------------ #
    by_norm: dict[str, list[dict]] = {}
    for r in rows:
        by_norm.setdefault(normalise_for_dedup(r["text_ory"]), []).append(r)
    n_exact_dupe_rows = len(rows) - len(by_norm)
    print("\n  exact duplicates (Odia, normalised): {:,} rows collapse into {:,} distinct "
          "texts ({:,} removed)".format(len(rows), len(by_norm), n_exact_dupe_rows))

    by_norm_eng: dict[str, int] = Counter(normalise_for_dedup(r["text_eng"]) for r in rows)
    print("  exact duplicates (English, for comparison): {:,} distinct texts "
          "({:,} removed)".format(len(by_norm_eng), len(rows) - len(by_norm_eng)))

    # exact-duplicate groups that disagree on label are a real signal about
    # translation collapse: different English, same Odia, different teacher label
    label_conflicts = sum(
        1 for g in by_norm.values() if len({r["label"] for r in g}) > 1
    )
    print("  exact-duplicate groups whose members disagree on label: {:,}".format(
        label_conflicts))

    # --- 2/3. near-duplicate clustering on Odia ------------------------------- #
    print("\n  clustering Odia side ({:,} distinct texts)...".format(len(by_norm)))
    key_for_text = {norm: min(r["example_id"] for r in grp) for norm, grp in by_norm.items()}
    texts_by_key = {key_for_text[norm]: norm for norm in by_norm}
    assignment, ory_stats = cluster_texts(
        texts_by_key,
        num_perm=mh["num_perm"],
        char_ngram=mh["char_ngram"],
        threshold=mh["jaccard_threshold"],
        seed=seed,
    )
    print("    {:,} distinct texts -> {:,} clusters ({:,} multi-text) in {:.0f}s".format(
        ory_stats["n_texts"], ory_stats["n_clusters"],
        ory_stats["n_multi_text_clusters"],
        ory_stats["minhash_seconds"] + ory_stats["lsh_seconds"]))

    # --- map every row to a cluster ------------------------------------------ #
    cluster_rows: dict[str, list[dict]] = {}
    for norm, grp in by_norm.items():
        root = assignment[key_for_text[norm]]
        cluster_rows.setdefault(root, []).extend(grp)

    sizes = Counter({k: len(v) for k, v in cluster_rows.items()})
    dist = size_distribution(sizes)
    print("\n  cluster size distribution (rows per cluster):")
    for k, v in dist.items():
        print("    {:<8}{:>10,} clusters".format(k, v))

    # --- 4. representative selection with the majority-cell tie-break --------- #
    survivors: list[dict] = []
    n_cell_conflict = 0
    n_label_conflict = 0
    for root in sorted(cluster_rows):
        members = cluster_rows[root]
        rep, _maj_cell, spans = select_representative(members)
        if spans:
            n_cell_conflict += 1
        if len({m["label"] for m in members}) > 1:
            n_label_conflict += 1
        rep = dict(rep)
        rep["dedup_group"] = root
        rep["cluster_size"] = len(members)
        survivors.append(rep)

    survivors.sort(key=lambda r: r["example_id"])

    print("\n  clusters spanning more than one (label, source_config) cell: {:,}".format(
        n_cell_conflict))
    print("  clusters whose members disagree on LABEL alone:              {:,}".format(
        n_label_conflict))

    # --- 10 largest clusters, 3 members each --------------------------------- #
    largest = sorted(cluster_rows.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
    print("\n" + "=" * 100)
    print("10 LARGEST CLUSTERS")
    print("=" * 100)
    for root, members in largest:
        cells = Counter((m["label_str"], m["source_config"]) for m in members)
        print("\n  cluster {}  size={:,}  cells={}".format(
            root, len(members), dict(cells)))
        for m in sorted(members, key=lambda x: x["example_id"])[:3]:
            print("    [{}] p={:.4f} {}".format(
                m["label_str"], m["p_hate"], m["text_ory"][:150]))

    # --- English-side comparison --------------------------------------------- #
    eng_stats = None
    if not args.skip_english and dcfg.get("also_report_english_side", True):
        print("\n  clustering English side for comparison ({:,} distinct texts)...".format(
            len(by_norm_eng)))
        eng_groups: dict[str, list[dict]] = {}
        for r in rows:
            eng_groups.setdefault(normalise_for_dedup(r["text_eng"]), []).append(r)
        eng_keys = {n: min(r["example_id"] for r in g) for n, g in eng_groups.items()}
        _, eng_stats = cluster_texts(
            {eng_keys[n]: n for n in eng_groups},
            num_perm=mh["num_perm"], char_ngram=mh["char_ngram"],
            threshold=mh["jaccard_threshold"], seed=seed,
        )
        print("    {:,} distinct texts -> {:,} clusters ({:,} multi-text)".format(
            eng_stats["n_texts"], eng_stats["n_clusters"],
            eng_stats["n_multi_text_clusters"]))

    # --- threshold sensitivity (reported, never applied) ---------------------- #
    sweep_rows = None
    if args.sweep:
        import random as _random

        pool = sorted(texts_by_key)
        take = min(args.sweep, len(pool))
        picked = _random.Random(seed).sample(pool, take)
        print()
        print("=" * 100)
        print("JACCARD THRESHOLD SENSITIVITY  ({:,}-text sample; configured = {})".format(
            take, mh["jaccard_threshold"]))
        print("  Reported only - the configured threshold is fixed by the spec and is not "
              "changed here.")
        print("-" * 100)
        sweep_rows = threshold_sweep(
            {k: texts_by_key[k] for k in picked},
            [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50],
            num_perm=mh["num_perm"], char_ngram=mh["char_ngram"], seed=seed,
        )
        print("  {:>10}{:>12}{:>15}{:>14}".format(
            "threshold", "clusters", "texts merged", "% reduction"))
        for r in sweep_rows:
            mark = "  <- configured" if r["threshold"] == mh["jaccard_threshold"] else ""
            print("  {:>10.2f}{:>12,}{:>15,}{:>13.2f}%{}".format(
                r["threshold"], r["n_clusters"], r["texts_merged"],
                r["pct_reduction"], mark))
        print("=" * 100)

    # --- output --------------------------------------------------------------- #
    schema = tbl.schema.append(pa.field("dedup_group", pa.string())).append(
        pa.field("cluster_size", pa.int32())
    )
    out = pa.Table.from_pylist(survivors, schema=schema)
    out_path = paths.interim / "deduped.parquet"
    pq.write_table(out, out_path, compression="zstd", compression_level=3)

    # --- summary -------------------------------------------------------------- #
    label_after = Counter(r["label_str"] for r in survivors)
    src_after = Counter(r["source_config"] for r in survivors)
    print("\n" + "=" * 100)
    print("STAGE 5 SUMMARY")
    print("-" * 100)
    print("  rows in                       {:>10,}".format(len(rows)))
    print("  distinct after exact dedup    {:>10,}  (-{:,})".format(
        len(by_norm), n_exact_dupe_rows))
    print("  clusters after near-dedup     {:>10,}  (-{:,})".format(
        len(survivors), len(by_norm) - len(survivors)))
    print("  rows out                      {:>10,}  ({:.1f}% of input)".format(
        len(survivors), len(survivors) / len(rows) * 100))
    print()
    print("  surviving label balance:  HATE {:,}   NON_HATE {:,}".format(
        label_after["HATE"], label_after["NON_HATE"]))
    print("  surviving by source    :  {}".format(dict(src_after)))
    print("  balanced ceiling after dedup = 2 x {:,} = {:,}".format(
        min(label_after.values()), 2 * min(label_after.values())))
    if eng_stats:
        print()
        print("  translation-collapse comparison:")
        print("    Odia    {:,} distinct texts -> {:,} clusters".format(
            ory_stats["n_texts"], ory_stats["n_clusters"]))
        print("    English {:,} distinct texts -> {:,} clusters".format(
            eng_stats["n_texts"], eng_stats["n_clusters"]))
    print("=" * 100)

    stats = {
        "rows_in": len(rows),
        "distinct_after_exact_ory": len(by_norm),
        "exact_dupe_rows_removed_ory": n_exact_dupe_rows,
        "distinct_after_exact_eng": len(by_norm_eng),
        "exact_dupe_rows_removed_eng": len(rows) - len(by_norm_eng),
        "exact_groups_with_label_conflict": label_conflicts,
        "ory_clustering": ory_stats,
        "eng_clustering": eng_stats,
        "cluster_size_distribution": dist,
        "clusters_spanning_multiple_cells": n_cell_conflict,
        "clusters_with_label_conflict": n_label_conflict,
        "rows_out": len(survivors),
        "label_balance_after": dict(label_after),
        "source_balance_after": dict(src_after),
        "largest_clusters": [
            {"dedup_group": root, "size": len(m)} for root, m in largest
        ],
        "threshold_sensitivity": sweep_rows,
    }
    append_stage_stats(
        paths.artifacts, STAGE,
        rows_in=len(rows), rows_out=len(survivors),
        dropped_by_reason={
            "exact_duplicate_ory": n_exact_dupe_rows,
            "near_duplicate_cluster_member": len(by_norm) - len(survivors),
        },
        extra={**stats, "output_sha256": file_sha256(out_path)},
    )
    dist_path = paths.artifacts / "label_distribution.json"
    d = json.loads(dist_path.read_text(encoding="utf-8")) if dist_path.exists() else {}
    d["dedup"] = stats
    dist_path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    print("\nwrote data/interim/deduped.parquet  {:,} rows".format(len(survivors)))
    print("row accounting: {:,} in = {:,} out + {:,} exact + {:,} near-dup".format(
        len(rows), len(survivors), n_exact_dupe_rows, len(by_norm) - len(survivors)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
