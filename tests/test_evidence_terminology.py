import pytest

from evidence.audit import EvidenceAuditError, audit_terminology


@pytest.mark.parametrize(
    "phrase",
    (
        "hospital discharge ETA",
        "clinically validated",
        "patient-specific confidence interval",
    ),
)
def test_project_specific_misleading_claims_are_rejected(tmp_path, phrase):
    report = tmp_path / "report.md"
    report.write_text(phrase, encoding="utf-8")
    with pytest.raises(EvidenceAuditError, match="prohibited terminology"):
        audit_terminology((report,))
