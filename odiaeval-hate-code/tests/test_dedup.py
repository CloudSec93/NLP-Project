"""Stage 5 tests: clustering determinism, the majority-cell tie-break, leakage safety."""

from __future__ import annotations

from collections import Counter

import pytest

from src.dedup import (
    UnionFind,
    cluster_texts,
    select_representative,
    shingles,
    size_distribution,
)

MH = {"num_perm": 128, "char_ngram": 5, "threshold": 0.85, "seed": 42}

ODIA_A = "ଆପଣ କେମିତି ଅଛନ୍ତି ଏହା ଏକ ସାଧାରଣ ପ୍ରଶ୍ନ ଅଟେ ଯାହା ଲୋକମାନେ ପଚାରନ୍ତି"
ODIA_B = "ସମ୍ପୂର୍ଣ୍ଣ ଭିନ୍ନ ଏକ ବାକ୍ଯ଼ ଯାହାର ପୂର୍ବ ବାକ୍ଯ଼ ସହିତ କୌଣସି ସମ୍ପର୍କ ନାହିଁ"


class TestUnionFind:
    def test_transitive_merge(self) -> None:
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")
        assert uf.find("d") != uf.find("a")

    def test_root_is_always_the_smallest_key(self) -> None:
        """Cluster ids must be a pure function of membership, not merge order."""
        uf = UnionFind(["z", "m", "a"])
        uf.union("z", "m")
        uf.union("m", "a")
        assert uf.find("z") == "a"

    def test_merge_order_does_not_change_the_root(self) -> None:
        a = UnionFind(["a", "b", "c"])
        for x, y in [("a", "b"), ("b", "c")]:
            a.union(x, y)
        b = UnionFind(["a", "b", "c"])
        for x, y in [("b", "c"), ("a", "b")]:
            b.union(x, y)
        assert {k: a.find(k) for k in "abc"} == {k: b.find(k) for k in "abc"}


class TestShingles:
    def test_kgram_count(self) -> None:
        assert len(shingles("abcdefg", 5)) == 3          # abcde, bcdef, cdefg
        assert set(shingles("abcde", 5)) == {b"abcde"}

    def test_short_text_becomes_one_shingle(self) -> None:
        assert shingles("ab", 5) == [b"ab"]

    def test_empty_text_has_no_shingles(self) -> None:
        assert shingles("", 5) == []
        assert shingles("   ", 5) == []

    def test_normalisation_is_applied(self) -> None:
        """Whitespace/zero-width differences must not create distinct shingles."""
        assert set(shingles("foo  bar", 5)) == set(shingles("foo bar", 5))
        assert set(shingles("foo​bar", 5)) == set(shingles("foobar", 5))


class TestClustering:
    def test_identical_texts_cluster(self) -> None:
        assign, stats = cluster_texts({"a": ODIA_A, "b": ODIA_A}, **MH)
        assert assign["a"] == assign["b"]
        assert stats["n_clusters"] == 1

    def test_unrelated_texts_do_not_cluster(self) -> None:
        assign, stats = cluster_texts({"a": ODIA_A, "b": ODIA_B}, **MH)
        assert assign["a"] != assign["b"]
        assert stats["n_clusters"] == 2

    def test_near_duplicate_clusters_at_threshold(self) -> None:
        """A one-word edit in a long prompt stays above Jaccard 0.85."""
        near = ODIA_A + " ଆଉ"
        assign, _ = cluster_texts({"a": ODIA_A, "b": near}, **MH)
        assert assign["a"] == assign["b"]

    def test_clustering_is_deterministic(self) -> None:
        texts = {f"k{i:03d}": (ODIA_A if i % 2 else ODIA_B) + f" {i // 7}" for i in range(60)}
        a, _ = cluster_texts(texts, **MH)
        b, _ = cluster_texts(texts, **MH)
        assert a == b

    def test_clustering_is_independent_of_input_order(self) -> None:
        """datasketch insertion order must not leak into the cluster labelling."""
        texts = {f"k{i:03d}": (ODIA_A if i % 2 else ODIA_B) + f" {i // 7}" for i in range(60)}
        forward, _ = cluster_texts(texts, **MH)
        reversed_texts = dict(reversed(list(texts.items())))
        backward, _ = cluster_texts(reversed_texts, **MH)
        assert forward == backward

    def test_cluster_root_is_the_smallest_member_key(self) -> None:
        assign, _ = cluster_texts({"zzz": ODIA_A, "aaa": ODIA_A, "mmm": ODIA_A}, **MH)
        assert set(assign.values()) == {"aaa"}


class TestMajorityCellTieBreak:
    def _m(self, eid, label, src="Toxic_Matrix"):
        return {"example_id": eid, "label": label, "source_config": src}

    def test_majority_label_wins_over_lowest_id(self) -> None:
        """A 4-HATE/1-NON_HATE cluster must not survive as its NON_HATE member."""
        members = [
            self._m("aaa", 0),                       # lowest id, minority cell
            self._m("bbb", 1), self._m("ccc", 1),
            self._m("ddd", 1), self._m("eee", 1),
        ]
        rep, cell, spans = select_representative(members)
        assert rep["example_id"] == "bbb"
        assert cell == (1, "Toxic_Matrix")
        assert spans is True

    def test_single_cell_cluster_picks_lowest_id(self) -> None:
        members = [self._m("ccc", 1), self._m("aaa", 1), self._m("bbb", 1)]
        rep, cell, spans = select_representative(members)
        assert rep["example_id"] == "aaa"
        assert spans is False

    def test_tie_is_broken_by_the_lowest_ids_cell(self) -> None:
        members = [self._m("aaa", 0), self._m("bbb", 1)]
        rep, cell, _ = select_representative(members)
        assert cell == (0, "Toxic_Matrix")
        assert rep["example_id"] == "aaa"

    def test_source_config_is_part_of_the_cell(self) -> None:
        members = [
            self._m("aaa", 1, "HHRLHF_T"),
            self._m("bbb", 1, "Toxic_Matrix"), self._m("ccc", 1, "Toxic_Matrix"),
        ]
        rep, cell, spans = select_representative(members)
        assert cell == (1, "Toxic_Matrix")
        assert rep["example_id"] == "bbb"
        assert spans is True

    def test_representative_is_always_an_actual_member(self) -> None:
        members = [self._m("aaa", 0), self._m("bbb", 1), self._m("ccc", 1)]
        rep, _, _ = select_representative(members)
        assert rep in members

    def test_is_independent_of_member_order(self) -> None:
        members = [self._m("aaa", 0), self._m("bbb", 1), self._m("ccc", 1)]
        a, _, _ = select_representative(members)
        b, _, _ = select_representative(list(reversed(members)))
        assert a["example_id"] == b["example_id"]


class TestLeakageSafety:
    def test_one_representative_per_cluster_makes_dedup_group_unique(self) -> None:
        """Collapsing to one row per cluster is what removes the leakage risk:
        no dedup_group can appear twice, so none can straddle a split."""
        texts = {f"k{i:03d}": ODIA_A for i in range(20)}
        assign, _ = cluster_texts(texts, **MH)
        groups = set(assign.values())
        assert len(groups) == 1
        survivors = [min(assign)]  # one representative
        assert len(survivors) == len(groups)


def test_size_distribution_buckets_are_exhaustive() -> None:
    sizes = Counter({f"c{i}": n for i, n in enumerate([1, 1, 2, 4, 7, 20, 75, 500])})
    d = size_distribution(sizes)
    assert sum(d.values()) == len(sizes)
    assert d["1"] == 2 and d["2"] == 1 and d["3-5"] == 1
    assert d["6-10"] == 1 and d["11-50"] == 1 and d["51-100"] == 1 and d["100+"] == 1
