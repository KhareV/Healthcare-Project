"""Performance-v2 Phase 3, Parts 1-2: DEV subject manifest + deterministic
5-fold DEV crossfit folds (support OOF calibration only, not model
selection). DEVELOPMENT_VALIDATION_ONLY-successor; still never touches v1
TEST or the future fresh v2 test.
"""

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase3"
CROSSFIT_VERSION = "performance_v2_phase3_dev_crossfit_fold_v1"


def crossfit_fold_for_subject(subject_id: str, n_folds: int = 5, version: str = CROSSFIT_VERSION) -> int:
    digest = hashlib.sha256((version + "||" + subject_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % n_folds


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    val_rows = load_dev_rows(ROOT, splits=("validation",))
    dev_rows = train_rows + val_rows

    subject_to_stay = {}
    subject_to_split = {}
    for row in dev_rows:
        subject_to_stay[row["subject_id"]] = row["stay_id"]
        subject_to_split[row["subject_id"]] = row["split"]

    dev_subjects = sorted(subject_to_stay.keys())
    train_subjects = sorted({r["subject_id"] for r in train_rows})
    val_subjects = sorted({r["subject_id"] for r in val_rows})
    assert not (set(train_subjects) & set(val_subjects)), "train/validation subject overlap"
    assert len(dev_subjects) == len(train_subjects) + len(val_subjects)

    split_path = ROOT / "artifacts/splits/synthetic_split_v2.csv"
    manifest = {
        "status": "DEV_SUBJECT_MANIFEST_FROZEN",
        "artifact_version": "performance_v2_phase3_dev_subject_manifest_v1",
        "parent_split_ref": "artifacts/splits/synthetic_split_v2.csv",
        "parent_split_sha256": sha256_file(split_path),
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(val_subjects),
        "dev_subject_count": len(dev_subjects),
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "dev_subjects": dev_subjects,
        "subject_to_stay": {s: subject_to_stay[s] for s in dev_subjects},
        "subject_to_original_split": {s: subject_to_split[s] for s in dev_subjects},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "dev_subject_manifest_v1.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print("wrote", manifest_path, sha256_file(manifest_path))
    print("DEV subject count:", len(dev_subjects), "(train", len(train_subjects), "+ validation", len(val_subjects), ")")

    fold_path = OUT_DIR / "dev_crossfit_fold_assignment_v1.csv"
    with fold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "stay_id", "fold_version", "crossfit_fold_5"])
        for subject_id in dev_subjects:
            writer.writerow([subject_id, subject_to_stay[subject_id], CROSSFIT_VERSION, crossfit_fold_for_subject(subject_id)])
    print("wrote", fold_path, sha256_file(fold_path))

    counts = {}
    for subject_id in dev_subjects:
        fold = crossfit_fold_for_subject(subject_id)
        counts[fold] = counts.get(fold, 0) + 1
    print("crossfit fold sizes (subjects):", counts)


if __name__ == "__main__":
    main()
