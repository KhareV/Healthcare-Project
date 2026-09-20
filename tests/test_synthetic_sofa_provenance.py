import json
from pathlib import Path

from data.synthetic.provenance import sha256_file
from data.synthetic.sofa import sofa_at
from data.synthetic.sofa_validation import summarize_sofa_results
from synthetic_sofa_helpers import BINDINGS, CUTOFF, ROOT, SPEC, STAY, event, normal_history, provider, vaso


LINEAGE = ROOT / "artifacts/data/synthetic/sofa/phase6_sofa_lineage_v1.json"


def test_lineage_hashes_are_computed_file_identities():
    payload = json.loads(LINEAGE.read_text())
    for name in (
        "project_scope", "phase5_processed_manifest", "canonical_timeline",
        "processed_schema", "concept_map", "feature_provenance", "sofa_spec",
        "implementation", "current_sofa_provider", "support_contract", "fixture_qa",
    ):
        entry = payload[name]
        assert sha256_file(ROOT / entry["path"]) == entry["sha256"], name
    assert payload["final_production_sofa"]["status"].startswith("BLOCKED_")


def test_fixture_qa_is_recomputed_from_three_declared_golden_cases():
    normal = sofa_at(normal_history(), STAY, CUTOFF, spec=SPEC, bindings=BINDINGS, support_provider=provider())
    maximum_rows = [
        event("pao2", 75, event_id="p"), event("fio2", 1.0, event_id="f"),
        event("platelet_count", 15), event("bilirubin_total", 13),
        event("mean_arterial_pressure", 60), event("glasgow_coma_scale", 5),
        event("creatinine", 5.1),
    ]
    maximum = sofa_at(
        maximum_rows, STAY, CUTOFF, spec=SPEC, bindings=BINDINGS,
        support_provider=provider(ventilated=True, vaso=(vaso("norepinephrine", .2),)),
    )
    missing = sofa_at([], STAY, CUTOFF, spec=SPEC, bindings=BINDINGS, support_provider=provider())
    computed = summarize_sofa_results(
        (normal, maximum, missing), status="ENGINEERING_SYNTHETIC_FIXTURE_NON_SCIENTIFIC",
    )
    stored = json.loads((ROOT / "artifacts/data/synthetic/sofa/fixture/phase6_fixture_v1/sofa_fixture_qa.json").read_text())
    for key, value in computed.items():
        assert stored[key] == value


def test_lineage_declares_no_later_phase_artifacts():
    payload = json.loads(LINEAGE.read_text())
    assert set(payload["not_created"]) == {
        "recovery_deltas", "remaining_stay_labels", "feature_grid", "split",
        "preprocessor", "model",
    }
