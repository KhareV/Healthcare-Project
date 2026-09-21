#!/usr/bin/env python3
"""Fail-closed Phase-10 audit that never computes sealed-test target metrics."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.provenance import sha256_file


def main() -> None:
    manifest_path = ROOT / "artifacts/manifests/synthetic_phase10_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PHASE10_COMPLETE" or manifest["test_opened"] or manifest["g3_freeze_active"]:
        raise RuntimeError("Phase-10 governance status is invalid")
    if any(manifest[name] for name in ("model_training_performed", "hyperparameter_search_performed", "calibration_performed", "threshold_selection_performed")):
        raise RuntimeError("Phase 11+ work leaked into Phase 10")
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError("artifact hash mismatch: " + artifact["logical_name"])
    split_path = ROOT / "artifacts/splits/synthetic_split_v2.csv"
    with split_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_split = {name: {row["subject_id"] for row in rows if row["split"] == name} for name in ("train", "validation", "test")}
    if tuple(len(by_split[name]) for name in ("train", "validation", "test")) != (1400, 300, 300):
        raise RuntimeError("unexpected Phase-10 subject counts")
    if (by_split["train"] & by_split["validation"]) or (by_split["train"] & by_split["test"]) or (by_split["validation"] & by_split["test"]):
        raise RuntimeError("subject overlap across partitions")
    clone_splits = {}
    for row in rows:
        clone_splits.setdefault(row["clone_fingerprint_sha256"], set()).add(row["split"])
    if any(len(values) != 1 for values in clone_splits.values()):
        raise RuntimeError("clone group crossed partitions")
    fit_ids = set((ROOT / "artifacts/preprocessors/synthetic_train_fit_subjects_v1.txt").read_text().splitlines())
    if fit_ids != by_split["train"]:
        raise RuntimeError("train-fit subject inventory differs from split")
    preprocessor = json.loads((ROOT / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json").read_text())
    if preprocessor["fit_partition"] != "train" or preprocessor["validation_or_test_statistics_used"] is not False:
        raise RuntimeError("preprocessor fit provenance is unsafe")
    if preprocessor["split_sha256"] != manifest["split_sha256"] or preprocessor["fit_subjects_sha256"] != sha256_file(ROOT / "artifacts/preprocessors/synthetic_train_fit_subjects_v1.txt"):
        raise RuntimeError("preprocessor lineage mismatch")
    if any(item["logical_name"] == "model_ready_test" for item in manifest["artifacts"]):
        raise RuntimeError("sealed final-test rows were materialized before G3")
    if manifest["test_model_ready_artifact"] != "NOT_MATERIALIZED_UNTIL_G3_EQUIVALENT_FREEZE" or manifest["test_label_statistics_computed"]:
        raise RuntimeError("final-test seal is invalid")
    print(json.dumps({"status": "PASS", "subjects": len(rows), "subject_counts": manifest["subject_counts"], "row_counts": manifest["row_counts"], "feature_dimension": manifest["feature_dimension"], "encoded_static_dimension": manifest["encoded_static_dimension"], "test_opened": False, "model_training_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
