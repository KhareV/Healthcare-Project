"""V3.1 cohort identities (SHA-derived from the frozen diagnostic protocol) and runtime configs."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
from performance_v3.protocol import BASE_COHORT_CONFIG_REL, GENERATION_TIMESTAMP, derive_seed, sha256_file

PROTOCOL_REL = "configs/performance_v3_1/diagnostic_protocol_v1.yaml"
COHORT_DIR_REL = "artifacts/performance_v3_1/cohorts"
CONFIG_DIR_REL = "configs/performance_v3_1/cohorts"
SCI_PROTOCOL_REL = "configs/performance_v3_1/scientific_protocol_v1.yaml"
SPECS = {
    "DIAG_FIT": {"namespace": "SYN-V31-DF", "n": 1600, "role": "dev", "tag": "diag_fit", "protocol": PROTOCOL_REL},
    "DIAG_EVAL": {"namespace": "SYN-V31-DE", "n": 900, "role": "dev", "tag": "diag_eval", "protocol": PROTOCOL_REL},
    "V31_DEV1": {"namespace": "SYN-V31-D1", "n": 2000, "role": "dev", "tag": "v31_dev1", "protocol": SCI_PROTOCOL_REL},
    "V31_DEV2": {"namespace": "SYN-V31-D2", "n": 2000, "role": "dev", "tag": "v31_dev2", "protocol": SCI_PROTOCOL_REL},
    "V31_HOLD": {"namespace": "SYN-V31-H", "n": 1500, "role": "dev", "tag": "v31_hold", "protocol": SCI_PROTOCOL_REL},
}

def identity(root: Path, cid: str) -> dict:
    s = SPECS[cid]
    material = f"performance_v3_1|{s['tag']}|{sha256_file(root / s['protocol'])}|{s['n']}"
    return {"cohort_id": cid, "namespace": s["namespace"], "n_subjects": s["n"], "seed": derive_seed(material),
            "seed_material": material, "seed_material_sha256": hashlib.sha256(material.encode()).hexdigest()}

def runtime_config(root: Path, cid: str) -> dict:
    ident = identity(root, cid)
    cfg = copy.deepcopy(json.loads((root / BASE_COHORT_CONFIG_REL).read_text()))
    n = ident["n_subjects"]
    cfg.update({"runtime_config_version": f"performance_v3_1_{cid.lower()}_v1", "primary_seed": ident["seed"], "n_subjects": n,
                "subject_id_namespace": ident["namespace"],
                "accepted_subject_policy": f"DETERMINISTIC_REPLACEMENT_UNTIL_{n}_WITH_AT_LEAST_ONE_LEGAL_CUTOFF",
                "purpose": "PERFORMANCE_V3_1_DIAGNOSTIC_" + cid, "not_a_new_development_split": False,
                "seed_derivation": {"method": "performance_v3_1_cohort_seed_v1", **{k: ident[k] for k in ("seed_material", "seed_material_sha256", "seed")}}})
    return cfg

def cohort_dir(root: Path, cid: str) -> Path:
    return root / COHORT_DIR_REL / cid
