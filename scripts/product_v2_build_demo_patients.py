"""Product V2 — build a demo-safe patient alias manifest.

Reads the already-frozen, structurally-selected V2 demo manifest
(configs/performance_v2/v2_demo_manifest_v1.json) and layers a stable,
human-friendly alias onto each subject (e.g. "DEMO-CARDIAC-001") so the
product UI never has to show a raw synthetic subject/stay ID by default.

This is a PRODUCT artifact, not a scientific one: it does not select,
re-select, or reorder subjects by any performance-dependent criterion, and it
does not touch any frozen Performance-V2 governance/model/metric artifact.
Alias assignment is a pure deterministic function of stay_id (lexicographic
sort), so it is reproducible and independent of prediction outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_MANIFEST = ROOT / "configs" / "performance_v2" / "v2_demo_manifest_v1.json"
OUTPUT_PATH = ROOT / "artifacts" / "performance_v2" / "product" / "demo_patients_v1.json"


def build() -> dict:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    subjects = sorted(manifest["demo_subjects"], key=lambda s: s["stay_id"])

    aliased = []
    for index, subject in enumerate(subjects, start=1):
        alias = f"DEMO-CARDIAC-{index:03d}"
        aliased.append(
            {
                "patient_alias": alias,
                "stay_id": subject["stay_id"],
                "subject_id": subject["subject_id"],
                "cardiac_subtype": subject["cardiac_condition_group"].replace("SYNTHETIC_", ""),
                "age_years": subject["age_years"],
                "sex_category": subject["sex_category"],
                "n_legal_cutoffs": subject["n_legal_cutoffs"],
                "source_population": manifest["selection_criteria"]["population"],
            }
        )

    return {
        "artifact_version": "product_v2_demo_patients_v1",
        "status": "V2_PRODUCT_DEMO_PATIENTS_READY",
        "alias_rule": "lexicographic sort of stay_id, 1-indexed DEMO-CARDIAC-NNN; independent of prediction outcome or accuracy",
        "source_manifest_ref": {
            "ref": "configs/performance_v2/v2_demo_manifest_v1.json",
            "artifact_version": manifest["artifact_version"],
        },
        "selection_independent_of_model_performance": manifest["selection_criteria"][
            "selection_independent_of_model_performance"
        ],
        "n_demo_patients": len(aliased),
        "demo_patients": aliased,
    }


def main() -> None:
    payload = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({payload['n_demo_patients']} demo patients)")


if __name__ == "__main__":
    main()
