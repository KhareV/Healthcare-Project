import pytest

from evidence.audit import (
    EvidenceAuditError,
    validate_calibration_evidence,
    validate_selected_routing,
)


def valid():
    return {
        "method": "isotonic",
        "fit_partition": "validation",
        "threshold_criterion": "validation_f1",
        "selected_classifier_only": True,
        "refit_during_evidence": False,
    }


def test_evidence_generation_cannot_refit_or_test_optimize_calibration():
    validate_calibration_evidence(valid())
    for field, value in (("fit_partition", "test"), ("refit_during_evidence", True)):
        changed = valid()
        changed[field] = value
        with pytest.raises(EvidenceAuditError, match="calibration"):
            validate_calibration_evidence(changed)


def test_selected_family_routing_matches_frozen_family():
    validate_selected_routing({
        "recovery": {"family": "xgboost", "explanation_method": "tree_shap"},
        "organ_support": {"family": "gru", "explanation_method": "integrated_gradients"},
    })
    with pytest.raises(EvidenceAuditError, match="routing"):
        validate_selected_routing({
            "recovery": {"family": "xgboost", "explanation_method": "integrated_gradients"},
        })
