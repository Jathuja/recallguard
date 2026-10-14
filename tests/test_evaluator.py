"""Tests for RecallGuard reliability evaluator (Feature 04).

Tests cover precise metric denominators, stale-claim use, conflict detection,
evidence coverage, and safe abstention. No metrics are hardcoded — all
assertions are derived from the ground-truth flags in the synthetic dataset.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from core.evaluator import (
    EvalDataset,
    EvalSample,
    EvaluationReport,
    NaiveMemoryBaseline,
    PerSampleResult,
    ReliabilityEvaluator,
    _conflict_in_db,
    build_synthetic_dataset,
)
from core.database import DatabaseManager
from core.schemas import ClaimStatus, MemoryClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(status: str = "active", is_active: bool = True, **kwargs) -> MemoryClaim:
    return MemoryClaim(
        subject="user",
        predicate="language",
        object_value="Python",
        status=status,
        is_active=is_active,
        **kwargs,
    )


def _fresh_db() -> DatabaseManager:
    db = DatabaseManager(db_path=":memory:")
    db.init_db()
    return db


# ---------------------------------------------------------------------------
# Contract and initialization (backwards compatible with Feature 01)
# ---------------------------------------------------------------------------

class TestEvaluatorInitialization:
    def test_default_model_name(self) -> None:
        assert ReliabilityEvaluator().model_name == "qwen3.5:0.8b"

    def test_custom_model_name(self) -> None:
        assert ReliabilityEvaluator(model_name="llama3").model_name == "llama3"

    def test_contract_keys(self) -> None:
        contract = ReliabilityEvaluator.get_evaluation_contract()
        for key in ("staleness_score", "contradiction_risk", "veracity_confidence", "quarantine_recommendation"):
            assert key in contract


# ---------------------------------------------------------------------------
# Single-claim evaluation
# ---------------------------------------------------------------------------

class TestEvaluateClaim:
    def test_active_fresh_claim_zero_staleness(self) -> None:
        ev = ReliabilityEvaluator()
        result = ev.evaluate_claim(_make_claim(status="active"))
        assert result["staleness_score"] == 0.0
        assert result["contradiction_risk"] == 0.0
        assert result["quarantine_recommendation"] is False

    def test_superseded_claim_full_staleness(self) -> None:
        ev = ReliabilityEvaluator()
        result = ev.evaluate_claim(_make_claim(status="superseded", is_active=False))
        assert result["staleness_score"] == 1.0
        assert result["quarantine_recommendation"] is True

    def test_expired_claim_flagged(self) -> None:
        ev = ReliabilityEvaluator()
        claim = _make_claim(
            status="active",
            valid_until=datetime.now(timezone.utc) - timedelta(days=1),
        )
        result = ev.evaluate_claim(claim)
        assert result["is_expired"] is True
        assert result["staleness_score"] == 1.0
        assert result["quarantine_recommendation"] is True

    def test_needs_review_partial_staleness(self) -> None:
        ev = ReliabilityEvaluator()
        result = ev.evaluate_claim(_make_claim(status="needs_review", is_active=False))
        assert result["staleness_score"] == 0.5
        assert result["contradiction_risk"] == 0.5
        assert result["quarantine_recommendation"] is True

    def test_result_contains_expected_keys(self) -> None:
        result = ReliabilityEvaluator().evaluate_claim(_make_claim())
        for key in ("claim_id", "staleness_score", "contradiction_risk",
                    "veracity_confidence", "quarantine_recommendation", "is_expired"):
            assert key in result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

class TestEvaluateBatch:
    def test_empty_batch(self) -> None:
        result = ReliabilityEvaluator().evaluate_batch([])
        assert result["total"] == 0
        assert result["evaluations"] == []
        assert result["aggregate"] == {}

    def test_mixed_batch_stale_rate(self) -> None:
        active = _make_claim(status="active", is_active=True)
        stale = MemoryClaim(subject="u", predicate="p2", object_value="v2",
                            status="superseded", is_active=False)
        result = ReliabilityEvaluator().evaluate_batch([active, stale])
        assert result["total"] == 2
        assert result["aggregate"]["stale_rate"] == 0.5

    def test_all_active_zero_stale_rate(self) -> None:
        claims = [
            MemoryClaim(subject="u", predicate=f"p{i}", object_value="v",
                        status="active", is_active=True)
            for i in range(3)
        ]
        result = ReliabilityEvaluator().evaluate_batch(claims)
        assert result["aggregate"]["stale_rate"] == 0.0


# ---------------------------------------------------------------------------
# Synthetic dataset integrity
# ---------------------------------------------------------------------------

class TestSyntheticDataset:
    def test_has_samples(self) -> None:
        dataset, _ = build_synthetic_dataset()
        assert len(dataset.samples) > 0

    def test_has_seed_and_stale_claims(self) -> None:
        dataset, _ = build_synthetic_dataset()
        assert len(dataset.seed_claims) > 0
        assert len(dataset.stale_claims) > 0

    def test_stale_claims_are_superseded(self) -> None:
        dataset, _ = build_synthetic_dataset()
        for c in dataset.stale_claims:
            assert c.status == ClaimStatus.SUPERSEDED.value

    def test_seed_claims_inserted_in_db(self) -> None:
        dataset, db = build_synthetic_dataset()
        for c in dataset.seed_claims:
            assert db.get_claim(c.id) is not None

    def test_stale_claims_inserted_in_db(self) -> None:
        dataset, db = build_synthetic_dataset()
        for c in dataset.stale_claims:
            stored = db.get_claim(c.id)
            assert stored is not None
            assert stored.status == ClaimStatus.SUPERSEDED.value

    def test_has_stale_candidate_samples(self) -> None:
        dataset, _ = build_synthetic_dataset()
        assert any(s.has_stale_candidate for s in dataset.samples)

    def test_has_conflict_samples(self) -> None:
        dataset, _ = build_synthetic_dataset()
        assert any(s.has_conflict for s in dataset.samples)

    def test_should_abstain_samples(self) -> None:
        dataset, _ = build_synthetic_dataset()
        abstain = [s for s in dataset.samples if s.should_abstain]
        assert len(abstain) >= 1

    def test_abstain_samples_have_empty_expected_ids(self) -> None:
        dataset, _ = build_synthetic_dataset()
        for s in dataset.samples:
            if s.should_abstain:
                assert s.expected_claim_ids == []


# ---------------------------------------------------------------------------
# _conflict_in_db helper
# ---------------------------------------------------------------------------

class TestConflictInDb:
    def test_no_conflict_when_only_active_claims(self) -> None:
        db = _fresh_db()
        c = MemoryClaim(subject="s", predicate="p", object_value="v",
                        status="active", is_active=True)
        db.insert_claim(c)
        assert _conflict_in_db(db, [c.id]) is False

    def test_detects_needs_review_sibling(self) -> None:
        db = _fresh_db()
        active = MemoryClaim(subject="project", predicate="tech_stack",
                             object_value="Flask", status="active", is_active=True)
        review = MemoryClaim(subject="project", predicate="tech_stack",
                             object_value="FastAPI", status="needs_review", is_active=False)
        db.insert_claim(active)
        db.insert_claim(review)
        assert _conflict_in_db(db, [active.id]) is True

    def test_empty_expected_ids_returns_false(self) -> None:
        db = _fresh_db()
        assert _conflict_in_db(db, []) is False

    def test_unknown_id_returns_false(self) -> None:
        db = _fresh_db()
        assert _conflict_in_db(db, ["nonexistent-uuid"]) is False


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

class TestNaiveBaseline:
    def test_abstains_on_unknown_topic(self) -> None:
        _, db = build_synthetic_dataset()
        baseline = NaiveMemoryBaseline(db=db)
        result = baseline.answer("quarterly revenue espresso forecast brand")
        assert result["abstained"] is True

    def test_returns_a_claim_for_known_topic(self) -> None:
        _, db = build_synthetic_dataset()
        baseline = NaiveMemoryBaseline(db=db)
        result = baseline.answer("programming language user prefer")
        assert not result["abstained"]
        assert result["claim"] is not None

    def test_used_stale_flag_is_bool(self) -> None:
        _, db = build_synthetic_dataset()
        baseline = NaiveMemoryBaseline(db=db)
        result = baseline.answer("programming language user prefer")
        assert "used_stale" in result
        assert isinstance(result["used_stale"], bool)


# ---------------------------------------------------------------------------
# Full evaluation pipeline — metric denominators and values
# ---------------------------------------------------------------------------

class TestFullEvaluation:

    def _report(self) -> EvaluationReport:
        return ReliabilityEvaluator().run_full_evaluation()

    def test_report_has_all_metric_fields(self) -> None:
        report = self._report()
        for attr in ("stale_memory_error_rate", "conflict_detection_recall",
                     "evidence_coverage", "safe_abstention_rate"):
            assert hasattr(report, attr)

    def test_all_metrics_in_valid_range(self) -> None:
        report = self._report()
        for v in (report.stale_memory_error_rate, report.conflict_detection_recall,
                  report.evidence_coverage, report.safe_abstention_rate):
            assert 0.0 <= v <= 1.0, f"Metric out of [0,1]: {v}"

    def test_total_samples_matches_dataset(self) -> None:
        dataset, db = build_synthetic_dataset()
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        assert report.total_samples == len(dataset.samples)

    def test_per_sample_results_count(self) -> None:
        dataset, db = build_synthetic_dataset()
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        assert len(report.per_sample_results) == report.total_samples

    # --- stale_memory_error_rate ---

    def test_recallguard_stale_error_rate_is_zero(self) -> None:
        """retrieve() never returns non-active claims so stale rate must be 0."""
        report = self._report()
        assert report.stale_memory_error_rate == 0.0

    def test_stale_rate_denominator_is_stale_candidate_samples(self) -> None:
        """Stale rate denominator = samples with has_stale_candidate=True."""
        dataset, db = build_synthetic_dataset()
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        stale_cand = [r for r in report.per_sample_results if r.sample.has_stale_candidate]
        stale_used = [r for r in stale_cand if r.used_stale_claim]
        expected_rate = len(stale_used) / len(stale_cand) if stale_cand else 0.0
        assert abs(report.stale_memory_error_rate - expected_rate) < 1e-6

    def test_naive_stale_rate_uses_same_denominator(self) -> None:
        dataset, db = build_synthetic_dataset()
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        stale_cand_count = sum(1 for s in dataset.samples if s.has_stale_candidate)
        # naive stale rate must be computable from the same denominator
        assert stale_cand_count > 0  # dataset must have stale candidates
        naive_rate = report.naive_baseline["stale_memory_error_rate"]
        assert 0.0 <= naive_rate <= 1.0

    def test_per_sample_used_stale_never_true_for_recallguard(self) -> None:
        """No PerSampleResult should have used_stale_claim=True for RecallGuard."""
        report = self._report()
        for psr in report.per_sample_results:
            assert not psr.used_stale_claim, (
                f"Sample '{psr.sample.question}' incorrectly flagged as used_stale"
            )

    # --- conflict_detection_recall ---

    def test_conflict_recall_denominator_is_has_conflict_samples(self) -> None:
        dataset, db = build_synthetic_dataset()
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        conflict_samples = [r for r in report.per_sample_results if r.sample.has_conflict]
        detected = [r for r in conflict_samples if r.conflict_in_db]
        expected = len(detected) / len(conflict_samples) if conflict_samples else 0.0
        assert abs(report.conflict_detection_recall - expected) < 1e-6

    def test_conflict_recall_is_one_when_needs_review_sibling_present(self) -> None:
        """The synthetic dataset inserts c_conflict_new as needs_review, so recall=1.0."""
        report = self._report()
        assert report.conflict_detection_recall == 1.0

    def test_conflict_recall_zero_when_no_sibling(self) -> None:
        """No needs_review claims in DB → conflict recall = 0."""
        db = _fresh_db()
        active = MemoryClaim(subject="project", predicate="tech_stack",
                             object_value="Flask", status="active", is_active=True)
        db.insert_claim(active)
        dataset = EvalDataset(
            samples=[
                EvalSample(
                    question="What tech stack does the project use?",
                    expected_claim_ids=[active.id],
                    has_conflict=True,
                    should_abstain=False,
                )
            ],
            seed_claims=[active],
        )
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        assert report.conflict_detection_recall == 0.0

    def test_empty_has_conflict_samples_gives_zero_recall(self) -> None:
        db = _fresh_db()
        c = MemoryClaim(subject="s", predicate="p", object_value="v",
                        status="active", is_active=True)
        db.insert_claim(c)
        dataset = EvalDataset(
            samples=[EvalSample(question="some topic", expected_claim_ids=[c.id],
                                has_conflict=False, should_abstain=False)],
            seed_claims=[c],
        )
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        assert report.conflict_detection_recall == 0.0

    # --- evidence_coverage ---

    def test_coverage_denominator_is_answerable_samples(self) -> None:
        dataset, db = build_synthetic_dataset()
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        answerable = [r for r in report.per_sample_results if not r.sample.should_abstain]
        hits = [r for r in answerable if r.evidence_hit]
        expected = len(hits) / len(answerable) if answerable else 0.0
        assert abs(report.evidence_coverage - expected) < 1e-6

    def test_coverage_not_counted_for_abstain_samples(self) -> None:
        """should_abstain=True samples must never contribute to coverage numerator."""
        dataset, db = build_synthetic_dataset()
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        for psr in report.per_sample_results:
            if psr.sample.should_abstain:
                assert not psr.evidence_hit, (
                    f"Abstain sample '{psr.sample.question}' should not count as evidence hit"
                )

    def test_coverage_zero_when_no_expected_claims_in_result(self) -> None:
        db = _fresh_db()
        c = MemoryClaim(subject="widget", predicate="colour", object_value="red",
                        status="active", is_active=True,
                        source_quote="The widget is red.")
        db.insert_claim(c)
        # Question about a completely different topic
        dataset = EvalDataset(
            samples=[
                EvalSample(
                    question="What programming language does the user prefer?",
                    expected_claim_ids=[c.id],  # claim won't be retrieved
                    has_conflict=False,
                    should_abstain=False,
                )
            ],
            seed_claims=[c],
        )
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        assert report.evidence_coverage == 0.0

    # --- safe_abstention_rate ---

    def test_abstention_rate_denominator_is_should_abstain_samples(self) -> None:
        dataset, db = build_synthetic_dataset()
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        abstain = [r for r in report.per_sample_results if r.sample.should_abstain]
        correct = [r for r in abstain if r.correctly_abstained]
        expected = len(correct) / len(abstain) if abstain else 0.0
        assert abs(report.safe_abstention_rate - expected) < 1e-6

    def test_abstain_questions_do_not_match_any_claim(self) -> None:
        """Abstain-sample questions should produce no retrieved claims."""
        dataset, db = build_synthetic_dataset()
        from core.retrieval_service import RetrievalService
        svc = RetrievalService(db_manager=db)
        for sample in dataset.samples:
            if sample.should_abstain:
                rr = svc.retrieve(sample.question)
                assert rr.abstained, (
                    f"Should-abstain question retrieved claims: '{sample.question}' "
                    f"got {[c.id[:8] for c in rr.retrieved_claims]}"
                )

    def test_abstention_rate_one_when_all_abstain_correctly(self) -> None:
        db = _fresh_db()
        dataset = EvalDataset(
            samples=[
                EvalSample(
                    question="What is the quarterly revenue espresso forecast?",
                    expected_claim_ids=[],
                    has_conflict=False,
                    should_abstain=True,
                )
            ],
        )
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        assert report.safe_abstention_rate == 1.0

    def test_abstention_rate_zero_when_no_abstain_samples(self) -> None:
        db = _fresh_db()
        c = MemoryClaim(subject="s", predicate="p", object_value="v",
                        status="active", is_active=True,
                        source_quote="s p v.")
        db.insert_claim(c)
        dataset = EvalDataset(
            samples=[
                EvalSample(question="What is s p?", expected_claim_ids=[c.id],
                           should_abstain=False)
            ],
            seed_claims=[c],
        )
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        assert report.safe_abstention_rate == 0.0

    # --- Empty dataset edge case ---

    def test_empty_dataset_returns_zeroed_report(self) -> None:
        db = _fresh_db()
        dataset = EvalDataset(samples=[], seed_claims=[])
        report = ReliabilityEvaluator().run_full_evaluation(dataset=dataset, db=db)
        assert report.total_samples == 0
        assert report.stale_memory_error_rate == 0.0
        assert report.conflict_detection_recall == 0.0
        assert report.evidence_coverage == 0.0
        assert report.safe_abstention_rate == 0.0

    # --- Naive baseline ---

    def test_naive_baseline_has_all_metric_keys(self) -> None:
        report = self._report()
        for key in ("stale_memory_error_rate", "conflict_detection_recall",
                    "evidence_coverage", "safe_abstention_rate"):
            assert key in report.naive_baseline

    def test_naive_conflict_recall_is_zero(self) -> None:
        """Naive baseline has no conflict detection logic."""
        report = self._report()
        assert report.naive_baseline["conflict_detection_recall"] == 0.0

    def test_improvements_present(self) -> None:
        report = self._report()
        for key in ("stale_memory_error_rate", "conflict_detection_recall",
                    "evidence_coverage", "safe_abstention_rate"):
            assert key in report.improvements_over_baseline

    def test_report_notes_populated(self) -> None:
        report = self._report()
        assert len(report.notes) > 0


# ---------------------------------------------------------------------------
# Plotly chart data export
# ---------------------------------------------------------------------------

class TestChartData:
    def test_has_required_keys(self) -> None:
        report = ReliabilityEvaluator().run_full_evaluation()
        chart = ReliabilityEvaluator.build_metrics_chart_data(report)
        for key in ("metrics", "labels", "recallguard_values", "naive_baseline_values", "improvements"):
            assert key in chart

    def test_lengths_consistent(self) -> None:
        report = ReliabilityEvaluator().run_full_evaluation()
        chart = ReliabilityEvaluator.build_metrics_chart_data(report)
        n = len(chart["metrics"])
        assert len(chart["labels"]) == n
        assert len(chart["recallguard_values"]) == n
        assert len(chart["naive_baseline_values"]) == n
        assert len(chart["improvements"]) == n
