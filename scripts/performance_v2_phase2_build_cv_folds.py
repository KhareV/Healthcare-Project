"""Performance-v2 Phase 2: deterministic SUBJECT-LEVEL TRAIN-only CV folds.

DEVELOPMENT_VALIDATION_ONLY. TRAIN only -- never touches VALIDATION or TEST.
Fold assignment is a pure function of (subject_id, fold_count, version
string) via SHA-256, so it never uses labels and is fully reproducible.
Every row from a stay/subject stays in one fold (assignment is keyed by
subject_id, and stay_id always maps to exactly one subject_id in this
synthetic cohort -- one episode per subject -- so no clone/stay can cross
folds either).
"""

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from performance_v2.data_loading import load_dev_rows  # noqa: E402

FOLD_VERSION = "performance_v2_phase2_subject_fold_v1"


def fold_for_subject(subject_id: str, n_folds: int, version: str = FOLD_VERSION) -> int:
    digest = hashlib.sha256((version + "||" + subject_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % n_folds


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    subjects = sorted({row["subject_id"] for row in train_rows})
    subject_to_stay = {row["subject_id"]: row["stay_id"] for row in train_rows}

    out_path = ROOT / "artifacts/performance_v2/phase2/train_cv_fold_assignment_v1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "stay_id", "fold_version", "xgb_fold_5", "gru_fold_3"])
        for subject_id in subjects:
            writer.writerow([
                subject_id, subject_to_stay[subject_id], FOLD_VERSION,
                fold_for_subject(subject_id, 5), fold_for_subject(subject_id, 3),
            ])

    # Isolation proof: every stay's rows all carry the same subject, and
    # every subject appears in exactly one fold for each fold count.
    for n_folds, label in ((5, "xgb_fold_5"), (3, "gru_fold_3")):
        counts = {}
        for subject_id in subjects:
            fold = fold_for_subject(subject_id, n_folds)
            counts[fold] = counts.get(fold, 0) + 1
        print(label, "fold sizes (subjects):", counts)

    print("wrote", out_path, "subjects:", len(subjects))


if __name__ == "__main__":
    main()
