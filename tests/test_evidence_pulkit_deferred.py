from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pulkit_items_are_deferred_without_fake_outputs():
    checklist = (
        ROOT / "docs/evidence/models/deferred/pulkit_evidence_checklist.md"
    ).read_text(encoding="utf-8")
    assert checklist.count("DEFERRED — PULKIT TRACK") >= 8
    evidence_root = ROOT / "docs/evidence/models"
    prohibited = tuple(evidence_root.rglob("*.png")) + tuple(evidence_root.rglob("*.jpg"))
    assert prohibited == ()


def test_report_values_contain_no_untraceable_scientific_number():
    import json

    values = json.loads(
        (ROOT / "docs/evidence/models/report_values_v1.json").read_text(encoding="utf-8")
    )
    assert values["scientific_values"] == []
    assert values["final_test_results"]["status"] == "BLOCKED"
