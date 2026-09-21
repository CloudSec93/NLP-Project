"""Stage 4 tests: banding, association statistics, and the agreement sampler."""

from __future__ import annotations

import numpy as np
import pytest

from src.agreement import ANNOTATION_HEADER, cohens_kappa, stratified_sample
from src.filter import (
    LABEL_HATE,
    LABEL_NON_HATE,
    assign_label,
    assign_labels,
    cramers_v,
    escalation_verdict,
    mutual_information,
)

HI, LO = 0.90, 0.05


class TestBanding:
    def test_exact_boundaries_are_inclusive(self) -> None:
        assert assign_label(0.90, HI, LO) == LABEL_HATE
        assert assign_label(0.05, HI, LO) == LABEL_NON_HATE

    def test_just_inside_the_middle_is_discarded(self) -> None:
        assert assign_label(0.899999, HI, LO) is None
        assert assign_label(0.050001, HI, LO) is None

    def test_extremes(self) -> None:
        assert assign_label(1.0, HI, LO) == LABEL_HATE
        assert assign_label(0.0, HI, LO) == LABEL_NON_HATE

    def test_vectorised_matches_scalar(self) -> None:
        p = np.array([0.0, 0.05, 0.050001, 0.5, 0.899999, 0.90, 1.0])
        vec = assign_labels(p, HI, LO)
        scalar = [assign_label(float(x), HI, LO) for x in p]
        for v, s in zip(vec, scalar):
            assert (int(v) if v >= 0 else None) == s

    def test_bands_partition_exactly(self) -> None:
        rng = np.random.default_rng(42)
        p = rng.random(10_000)
        lab = assign_labels(p, HI, LO)
        assert (lab == LABEL_HATE).sum() + (lab == LABEL_NON_HATE).sum() + (lab == -1).sum() == 10_000


class TestAssociation:
    def test_perfect_association_gives_nmi_one(self) -> None:
        """Each source maps to exactly one label — source IS the label."""
        table = np.array([[0, 100], [0, 100], [100, 0]], dtype=np.float64)
        s = mutual_information(table)
        assert s["nmi_min"] == pytest.approx(1.0, abs=1e-9)

    def test_independence_gives_nmi_zero(self) -> None:
        table = np.array([[50, 50], [100, 100], [25, 25]], dtype=np.float64)
        s = mutual_information(table)
        assert s["nmi_arithmetic"] == pytest.approx(0.0, abs=1e-12)
        assert cramers_v(table) == pytest.approx(0.0, abs=1e-9)

    def test_perfect_association_gives_cramers_v_one(self) -> None:
        table = np.array([[0, 100], [100, 0]], dtype=np.float64)
        assert cramers_v(table) == pytest.approx(1.0, abs=1e-9)

    def test_empty_table_does_not_divide_by_zero(self) -> None:
        table = np.zeros((3, 2), dtype=np.float64)
        assert mutual_information(table)["nmi_arithmetic"] == 0.0
        assert cramers_v(table) == 0.0

    def test_normalisers_disagree_when_marginal_entropies_differ(self) -> None:
        """Why the normaliser is stated explicitly rather than left to a default.

        A symmetric 2x2 table has H(X) == H(Y) by construction, so the three
        normalisers coincide. The real corpus does not look like that: 3 sources
        of very unequal size against 2 lopsided classes.
        """
        table = np.array([[60436, 17351], [26683, 3593], [14813, 74]], dtype=np.float64)
        s = mutual_information(table)
        assert s["entropy_source"] != pytest.approx(s["entropy_label"], abs=1e-6)
        assert s["nmi_min"] > s["nmi_arithmetic"] > s["nmi_max"]


class TestEscalationRule:
    def test_thresholds(self) -> None:
        assert escalation_verdict(21_018)[0] == "FINE"
        assert escalation_verdict(10_000)[0] == "FINE"
        assert escalation_verdict(9_999)[0] == "WORKABLE"
        assert escalation_verdict(2_000)[0] == "WORKABLE"
        assert escalation_verdict(1_999)[0] == "STOP"
        assert escalation_verdict(0)[0] == "STOP"

    def test_stop_message_forbids_quiet_loosening(self) -> None:
        _, note = escalation_verdict(500)
        assert "Quietly loosening" in note


class TestAgreementSampler:
    def _rows(self, n: int = 300) -> list[dict]:
        out = []
        for i in range(n):
            out.append({
                "example_id": f"{i:016x}",
                "label_str": "HATE" if i % 5 == 0 else "NON_HATE",
                "source_config": ["Toxic_Matrix", "HHRLHF_T", "Dolly_T"][i % 3],
                "text_eng": f"text {i}",
                "p_hate": i / n,
            })
        return out

    def test_sample_size_is_exact(self) -> None:
        s = stratified_sample(self._rows(), 50, 42, lambda r: r["source_config"])
        assert len(s) == 50

    def test_sampling_is_deterministic_given_the_seed(self) -> None:
        a = stratified_sample(self._rows(), 50, 42, lambda r: r["source_config"])
        b = stratified_sample(self._rows(), 50, 42, lambda r: r["source_config"])
        assert [r["example_id"] for r in a] == [r["example_id"] for r in b]

    def test_sampling_is_independent_of_input_row_order(self) -> None:
        rows = self._rows()
        shuffled = list(reversed(rows))
        a = stratified_sample(rows, 50, 42, lambda r: r["source_config"])
        b = stratified_sample(shuffled, 50, 42, lambda r: r["source_config"])
        assert sorted(r["example_id"] for r in a) == sorted(r["example_id"] for r in b)

    def test_strata_are_represented(self) -> None:
        s = stratified_sample(self._rows(), 60, 42, lambda r: r["source_config"])
        seen = {r["source_config"] for r in s}
        assert seen == {"Toxic_Matrix", "HHRLHF_T", "Dolly_T"}

    def test_requesting_more_than_available_returns_all(self) -> None:
        s = stratified_sample(self._rows(30), 500, 42, lambda r: r["source_config"])
        assert len(s) == 30

    def test_annotation_csv_never_carries_the_teacher_label(self) -> None:
        """Showing the model's answer measures anchoring, not agreement."""
        assert "human_label" in ANNOTATION_HEADER
        for leaky in ("label", "label_str", "p_hate", "teacher", "p_hate_raw"):
            assert leaky not in ANNOTATION_HEADER


class TestCohensKappa:
    LABELS = ["NON_HATE", "HATE"]

    def test_perfect_agreement_is_one(self) -> None:
        a = ["HATE", "NON_HATE", "HATE", "NON_HATE"]
        assert cohens_kappa(a, list(a), self.LABELS)["cohens_kappa"] == pytest.approx(1.0)

    def test_total_disagreement_is_negative(self) -> None:
        a = ["HATE", "HATE", "NON_HATE", "NON_HATE"]
        b = ["NON_HATE", "NON_HATE", "HATE", "HATE"]
        assert cohens_kappa(a, b, self.LABELS)["cohens_kappa"] < 0

    def test_chance_level_agreement_is_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        a = list(rng.choice(self.LABELS, 4000))
        b = list(rng.choice(self.LABELS, 4000))
        assert abs(cohens_kappa(a, b, self.LABELS)["cohens_kappa"]) < 0.05

    def test_observed_agreement_is_reported_alongside(self) -> None:
        a = ["HATE"] * 9 + ["NON_HATE"]
        b = ["HATE"] * 10
        r = cohens_kappa(a, b, self.LABELS)
        assert r["observed_agreement"] == pytest.approx(0.9)
        assert r["n"] == 10

    def test_confusion_matrix_orientation(self) -> None:
        """Rows are the human, columns the teacher."""
        human = ["HATE", "HATE", "NON_HATE"]
        teacher = ["NON_HATE", "HATE", "NON_HATE"]
        m = cohens_kappa(human, teacher, self.LABELS)["confusion"]
        # human=HATE (index 1), teacher=NON_HATE (index 0) -> one row
        assert m[1][0] == 1
        assert m[1][1] == 1
        assert m[0][0] == 1
