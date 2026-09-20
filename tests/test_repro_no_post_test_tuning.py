from pathlib import Path

import pytest

from reproducibility.audit import (
    FORBIDDEN_OPERATIONS,
    ReproducibilityAuditError,
    audit_path_portability,
    audit_restricted_data,
    reject_post_test_operation,
    require_cross_member_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reproduction_exposes_no_post_test_tuning_operation():
    for operation in FORBIDDEN_OPERATIONS:
        with pytest.raises(ReproducibilityAuditError, match="forbidden"):
            reject_post_test_operation(operation)


def test_owner_cannot_self_certify_cross_member_reproduction():
    with pytest.raises(ReproducibilityAuditError, match="non-owner"):
        require_cross_member_evidence({"reproducer_role": "vedant"})


def test_operational_paths_are_portable_and_artifacts_have_no_restricted_class():
    assert audit_path_portability(ROOT)["status"] == "PASS"
    assert audit_restricted_data(ROOT)["status"] == "PASS"
