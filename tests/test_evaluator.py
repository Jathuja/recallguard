"""Tests for RecallGuard reliability evaluator interface."""

import pytest
from core.evaluator import ReliabilityEvaluator
from core.schemas import MemoryClaim


def test_evaluator_initialization():
    """Verify evaluator default initialization with qwen3.5:0.8b."""
    evaluator = ReliabilityEvaluator()
    assert evaluator.model_name == "qwen3.5:0.8b"

    custom_evaluator = ReliabilityEvaluator(model_name="custom-model")
    assert custom_evaluator.model_name == "custom-model"


def test_evaluator_contract():
    """Verify evaluator provides planned metric definitions."""
    contract = ReliabilityEvaluator.get_evaluation_contract()
    assert "staleness_score" in contract
    assert "contradiction_risk" in contract
    assert "veracity_confidence" in contract
    assert "quarantine_recommendation" in contract


def test_evaluator_methods_raise_not_implemented():
    """Verify that reliability evaluation algorithms are stubs in Feature 01."""
    evaluator = ReliabilityEvaluator()
    claim = MemoryClaim(subject="test", predicate="prop", object_value="val")

    with pytest.raises(NotImplementedError) as exc_info1:
        evaluator.evaluate_claim(claim)
    assert "Feature 05" in str(exc_info1.value)

    with pytest.raises(NotImplementedError) as exc_info2:
        evaluator.evaluate_batch([claim])
    assert "Feature 05" in str(exc_info2.value)
