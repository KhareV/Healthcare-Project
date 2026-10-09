"""V3 protocol identity: protocol hash, deterministic cohort identities, cohort configs.

Seeds are derived from the SHA-256 of the frozen protocol file (development and
learning-curve-evaluation cohorts) or of the frozen selected-models manifest
(sealed test cohort), never chosen after seeing any outcome.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional

PROTOCOL_REL = "configs/performance_v3/scientific_protocol_v1.yaml"
BASE_COHORT_CONFIG_REL = "configs/performance_v2/fresh_test_cohort_v1.json"
SELECTED_MODELS_V3_REL = "artifacts/performance_v3/phase3/selected_models_v3.json"
COHORT_CONFIG_DIR_REL = "configs/performance_v3/cohorts"
COHORT_DATA_DIR_REL = "artifacts/performance_v3/cohorts"
GENERATION_TIMESTAMP = "2026-10-11T00:00:00Z"

DEV_CHUNK_SUBJECTS = 2100
LCE_SUBJECTS = 1500
TEST_SUBJECTS = 1500

COHORT_SPECS: Mapping[str, Mapping[str, object]] = {
    "D1": {"namespace": "SYN-V3-D1", "n_subjects": DEV_CHUNK_SUBJECTS, "role": "dev", "material": "performance_v3|dev_chunk_1|{protocol}|2100"},
    "D2": {"namespace": "SYN-V3-D2", "n_subjects": DEV_CHUNK_SUBJECTS, "role": "dev", "material": "performance_v3|dev_chunk_2|{protocol}|2100"},
    "D3": {"namespace": "SYN-V3-D3", "n_subjects": DEV_CHUNK_SUBJECTS, "role": "dev", "material": "performance_v3|dev_chunk_3|{protocol}|2100"},
    "LCE": {"namespace": "SYN-V3-LCE", "n_subjects": LCE_SUBJECTS, "role": "lce", "material": "performance_v3|learning_curve_eval|{protocol}|1500"},
    "T": {"namespace": "SYN-V3-T", "n_subjects": TEST_SUBJECTS, "role": "sealed_test", "material": "performance_v3_final_test_v1|{selected_models}|1500"},
}

SCIENTIFIC_CONFIG_KEYS = (
    "population", "calendar", "duration", "latent", "observation", "variables",
    "latent_process_path", "support_process_path", "event_dictionary_path",
    "minimum_accepted_episode_hours",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(material: str) -> int:
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def protocol_sha256(root: Path) -> str:
    return sha256_file(root / PROTOCOL_REL)


def cohort_identity(root: Path, cohort_id: str) -> Dict[str, object]:
    spec = COHORT_SPECS[cohort_id]
    if cohort_id == "T":
        path = root / SELECTED_MODELS_V3_REL
        if not path.exists():
            raise FileNotFoundError("sealed test identity requires the frozen selected_models_v3.json")
        material = str(spec["material"]).format(selected_models=sha256_file(path))
    else:
        material = str(spec["material"]).format(protocol=protocol_sha256(root))
    return {
        "cohort_id": cohort_id,
        "namespace": spec["namespace"],
        "n_subjects": spec["n_subjects"],
        "role": spec["role"],
        "seed": derive_seed(material),
        "seed_material": material,
        "seed_material_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
    }


def cohort_config(root: Path, cohort_id: str) -> Dict[str, object]:
    """Runtime generator config: V2 fresh-cohort config with only seed / n / namespace / provenance changed."""

    identity = cohort_identity(root, cohort_id)
    base = json.loads((root / BASE_COHORT_CONFIG_REL).read_text(encoding="utf-8"))
    config = copy.deepcopy(base)
    n = int(identity["n_subjects"])
    config.update(
        {
            "runtime_config_version": "performance_v3_cohort_" + cohort_id.lower() + "_v1",
            "primary_seed": int(identity["seed"]),
            "n_subjects": n,
            "subject_id_namespace": identity["namespace"],
            "accepted_subject_policy": f"DETERMINISTIC_REPLACEMENT_UNTIL_{n}_WITH_AT_LEAST_ONE_LEGAL_CUTOFF",
            "purpose": "PERFORMANCE_V3_" + str(identity["role"]).upper() + "_COHORT",
            "not_a_new_development_split": identity["role"] == "sealed_test",
            "seed_derivation": {
                "method": "performance_v3_cohort_seed_v1",
                "seed_material": identity["seed_material"],
                "seed_material_sha256": identity["seed_material_sha256"],
                "derivation_rule": "seed = int(SHA256(seed_material).hexdigest()[:8], 16)",
                "seed": identity["seed"],
            },
            "scientific_equivalence_note": (
                "All scientific generator parameters and process/event-dictionary paths are byte-identical to "
                + BASE_COHORT_CONFIG_REL + " and therefore to configs/synthetic/final_benchmark_v1.json; only "
                "primary_seed, n_subjects, subject_id_namespace, runtime_config_version and provenance differ."
            ),
        }
    )
    return config


def cohort_dir(root: Path, cohort_id: str) -> Path:
    return root / COHORT_DATA_DIR_REL / cohort_id
