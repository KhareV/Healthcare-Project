"""Performance-v2 Phase 3, Part 4: final V2 context-normalization
preprocessing, fit on ALL DEV subjects (old TRAIN + old VALIDATION) only.
Never touches the future fresh v2 test cohort. Never mutates the frozen
Phase-2 contract (configs/performance_v2/feature_contract_v2.json) --
these are successor artifacts under artifacts/performance_v2/phase3/preprocessing/.
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from performance_v2.context_normalization import fit_normalization  # noqa: E402
from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.gru_training import fit_age_normalization  # noqa: E402
from performance_v2.v2_features import group_a_feature_names  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase3/preprocessing"


def main() -> None:
    train_rows = load_dev_rows(ROOT, splits=("train",))
    val_rows = load_dev_rows(ROOT, splits=("validation",))
    dev_rows = train_rows + val_rows

    dev_subject_hash = hashlib.sha256("||".join(sorted({str(r["subject_id"]) for r in dev_rows})).encode("utf-8")).hexdigest()
    contract_path = ROOT / "configs/performance_v2/feature_contract_v2.json"
    contract_hash = sha256_file(contract_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_a_names = group_a_feature_names(dev_rows)

    for variant in ("B_MIN", "B_FULL", "B_PLUS_F"):
        normalization = fit_normalization(dev_rows, variant)
        age_mean, age_std = fit_age_normalization(dev_rows)
        payload = {
            "status": "DEV_PREPROCESSING_FROZEN",
            "artifact_version": "performance_v2_phase3_preprocessing_v1",
            "variant": variant,
            "feature_names": normalization["feature_names"],
            "mean": normalization["mean"],
            "std": normalization["std"],
            "age_mean": age_mean, "age_std": age_std,
            "group_a_feature_names": list(group_a_names),
            "group_a_feature_count": len(group_a_names),
            "dev_subject_hash": dev_subject_hash,
            "dev_subject_count": len({str(r["subject_id"]) for r in dev_rows}),
            "feature_contract_ref": "configs/performance_v2/feature_contract_v2.json",
            "feature_contract_sha256": contract_hash,
            "information_equivalent_to_phase2": True,
            "note": "Same context_v2.py variant definitions and same Group-A raw-canonical representation as Phase 2 (src/performance_v2/v2_features.py, context_v2.py), refit on the union of old TRAIN+VALIDATION subjects instead of Phase-2's TRAIN-only fit -- an information-equivalent successor, not a redefinition.",
        }
        out_path = OUT_DIR / f"context_normalization_v2_{variant}.json"
        out_path.write_bytes(canonical_json_bytes(payload))
        print("wrote", out_path, sha256_file(out_path))

    print("dev_subject_hash", dev_subject_hash)
    print("group_a_feature_count", len(group_a_names))


if __name__ == "__main__":
    main()
