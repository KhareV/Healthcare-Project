"""Build the deterministic V2 dashboard demo manifest.

Selects a small number of TRAIN-split DEV subjects (never fresh V2 test
subjects) for interactive dashboard replay, using ONLY structural criteria
(cardiac subtype diversity, >=5 legal replay cutoffs, availability of every
input field the frozen V2 pipeline needs) -- never model performance.

Legal cutoffs come from the real canonical timestamp generator
(data.timestamps.generate_prediction_rows_for_stay), not invented.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from data.synthetic.validation import load_jsonl  # noqa: E402
from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

DEV_MANIFEST_PATH = ROOT / "artifacts/performance_v2/phase3/dev_subject_manifest_v1.json"
STATICS_PATH = ROOT / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl"
MIN_CUTOFFS = 5
SUBTYPES = ("SYNTHETIC_ARRHYTHMIA", "SYNTHETIC_HF", "SYNTHETIC_ISCHEMIC")

OUT_PATH = ROOT / "configs/performance_v2/v2_demo_manifest_v1.json"


def _dt(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    dev_manifest = json.loads(DEV_MANIFEST_PATH.read_text())
    train_subjects = set(dev_manifest["train_subjects"])
    subject_to_stay = dev_manifest["subject_to_stay"]

    statics_by_stay = {row["stay_id"]: row for row in load_jsonl(STATICS_PATH)}

    # Deterministic candidate pool: TRAIN subjects only, sorted lexicographically
    # (never by any model-performance signal).
    candidates = sorted(train_subjects)

    selected = []
    used_subtypes = set()
    for subtype in SUBTYPES:
        for subject_id in candidates:
            if subject_id in [s["subject_id"] for s in selected]:
                continue
            stay_id = subject_to_stay[subject_id]
            static = statics_by_stay.get(stay_id)
            if static is None or static.get("cardiac_condition_group") != subtype:
                continue
            stay = RetainedICUStay(
                subject_id=subject_id, stay_id=stay_id,
                intime=_dt(static["intime"]), outtime=_dt(static["outtime"]),
            )
            rows = generate_prediction_rows_for_stay(stay)
            if len(rows) < MIN_CUTOFFS:
                continue
            selected.append({
                "subject_id": subject_id,
                "stay_id": stay_id,
                "cardiac_condition_group": subtype,
                "age_years": static["age_years"],
                "sex_category": static["sex_category"],
                "intime": static["intime"],
                "outtime": static["outtime"],
                "legal_cutoffs": [row.prediction_time.isoformat().replace("+00:00", "Z") for row in rows],
                "n_legal_cutoffs": len(rows),
            })
            used_subtypes.add(subtype)
            break

    if len(selected) < 3:
        raise SystemExit(f"could not select >=3 demo subjects with structural criteria; got {len(selected)}")

    manifest = {
        "status": "V2_DEMO_MANIFEST_FROZEN",
        "artifact_version": "performance_v2_v2_demo_manifest_v1",
        "selection_criteria": {
            "population": "DEV TRAIN split only (never fresh V2 test)",
            "min_legal_cutoffs": MIN_CUTOFFS,
            "cutoff_source": "data.timestamps.generate_prediction_rows_for_stay (canonical generator, not invented)",
            "diversity_rule": "one subject per distinct cardiac_condition_group, first lexicographically-sorted subject_id satisfying the structural criteria",
            "selection_independent_of_model_performance": True,
        },
        "dev_subject_manifest_ref": {"ref": "artifacts/performance_v2/phase3/dev_subject_manifest_v1.json", "sha256": sha256_file(DEV_MANIFEST_PATH)},
        "demo_subjects": selected,
        "n_demo_subjects": len(selected),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(canonical_json_bytes(manifest))
    print("wrote", OUT_PATH, sha256_file(OUT_PATH))
    for item in selected:
        print(" ", item["subject_id"], item["stay_id"], item["cardiac_condition_group"], item["n_legal_cutoffs"], "cutoffs")


if __name__ == "__main__":
    main()
