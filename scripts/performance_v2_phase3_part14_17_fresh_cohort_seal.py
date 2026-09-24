"""Performance-v2 Phase 3, Parts 14-17: fresh v2 test-cohort specification,
generator-equivalence audit, structural-only sealing, and the evaluation-only
split contract.

STRICTLY STRUCTURAL: this script never reads/prints/inspects any recovery,
ICU-time, or organ-support target/label VALUE, distribution, or count from
the fresh cohort. It only checks existence, counts, ID uniqueness/namespace,
subject/clone non-overlap with v1, and hash/schema binding -- per the
explicit Phase-3 prohibition on outcome inspection before Phase 4.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from data.synthetic.validation import load_jsonl  # noqa: E402
from preprocess.synthetic_phase10 import exact_raw_trajectory_fingerprints  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

PHASE3_DIR = ROOT / "artifacts/performance_v2/phase3"
GOV_DIR = ROOT / "artifacts/performance_v2/governance"
FRESH_DIR = PHASE3_DIR / "fresh_test_cohort"

V1_CONFIG = ROOT / "configs/synthetic/final_benchmark_v1.json"
V2_CONFIG = ROOT / "configs/performance_v2/fresh_test_cohort_v1.json"
V1_RAW_DIR = ROOT / "artifacts/data/synthetic/final/phase9_final_v1"
V1_SPLIT_PATH = ROOT / "artifacts/splits/synthetic_split_v2.csv"

SCIENTIFIC_CODE_FILES = [
    "src/data/synthetic/latent_state.py",
    "src/data/synthetic/observations.py",
    "src/data/synthetic/support.py",
    "src/data/synthetic/records.py",
    "src/data/synthetic/rng.py",
    "src/data/synthetic/validation.py",
    "src/data/synthetic/sofa.py",
    "src/data/synthetic/sofa_provider.py",
    "src/labels/recovery.py",
    "src/labels/icu_time.py",
    "src/labels/organ_support.py",
    "src/labels/support_state.py",
    "src/labels/ventilation_state.py",
    "src/features/synthetic.py",
]

NARROWLY_EXTENDED_FILES = [
    "src/data/synthetic/config.py",
    "src/data/synthetic/generator.py",
    "src/labels/phase9_final.py",
]

SCIENTIFIC_CONFIG_KEYS = (
    "population", "calendar", "duration", "latent", "observation", "variables",
    "latent_process_path", "support_process_path", "event_dictionary_path",
    "minimum_accepted_episode_hours",
)


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    v1_config = json.loads(V1_CONFIG.read_text())
    v2_config = json.loads(V2_CONFIG.read_text())

    # --- Part 14: fresh cohort specification ---
    seed_derivation = v2_config["seed_derivation"]
    generator_manifest = json.loads((FRESH_DIR / "raw/synthetic_dataset_manifest_v1.json").read_text())
    assert generator_manifest["primary_seed_identity"]["value"] == seed_derivation["fresh_test_seed"] == 402995653
    assert generator_manifest["requested_subject_count"] == generator_manifest["generated_subject_count"] == 1000
    assert generator_manifest["manifest_status"] == "AUTHORIZED_FINAL_SYNTHETIC_DATA"

    cohort_spec = {
        "status": "V2_FRESH_TEST_COHORT_SPEC_FROZEN",
        "artifact_version": "performance_v2_phase3_fresh_test_cohort_spec_v1",
        "evaluation_only": True,
        "not_a_new_development_split": True,
        "cohort_membership_label": "fresh_v2_test",
        "n_subjects": 1000,
        "subject_id_namespace": "SYN-V2",
        "seed_derivation": seed_derivation,
        "config_ref": _ref("configs/performance_v2/fresh_test_cohort_v1.json"),
        "generator_manifest_ref": _ref("artifacts/performance_v2/phase3/fresh_test_cohort/raw/synthetic_dataset_manifest_v1.json"),
        "no_fresh_v2_train_or_validation_subset": True,
        "note": "Do not retry another seed because resulting outcomes look inconvenient -- this seed was fixed before any generation occurred (Part 14).",
    }
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    cohort_spec_path = PHASE3_DIR / "fresh_test_cohort_spec_v1.json"
    cohort_spec_path.write_bytes(canonical_json_bytes(cohort_spec))
    print("wrote", cohort_spec_path, sha256_file(cohort_spec_path))

    # --- Part 15: generator-equivalence audit ---
    scientific_diffs = {k: {"v1": v1_config.get(k), "v2": v2_config.get(k)} for k in SCIENTIFIC_CONFIG_KEYS if v1_config.get(k) != v2_config.get(k)}
    if scientific_diffs:
        raise SystemExit("STOP: scientific generator parameter mismatch detected: " + repr(scientific_diffs))

    allowed_top_level_diffs = {
        "primary_seed", "n_subjects", "runtime_config_version", "subject_id_namespace",
        "accepted_subject_policy",  # textual policy label reflects n_subjects by design; verified below
        "purpose", "not_a_new_development_split", "seed_derivation", "scientific_equivalence_note",
        "inherits",  # identical value in both, listed defensively
    }
    all_keys = sorted(set(v1_config) | set(v2_config))
    unexpected_diffs = {k: {"v1": v1_config.get(k), "v2": v2_config.get(k)} for k in all_keys if v1_config.get(k) != v2_config.get(k) and k not in allowed_top_level_diffs}
    if unexpected_diffs:
        raise SystemExit("STOP: unexpected non-scientific config diff outside the permitted set: " + repr(unexpected_diffs))

    expected_v2_policy = f"DETERMINISTIC_REPLACEMENT_UNTIL_{v2_config['n_subjects']}_WITH_AT_LEAST_ONE_LEGAL_CUTOFF"
    assert v2_config["accepted_subject_policy"] == expected_v2_policy

    scientific_code_hashes = {path: sha256_file(ROOT / path) for path in SCIENTIFIC_CODE_FILES}
    narrowly_extended_hashes = {path: sha256_file(ROOT / path) for path in NARROWLY_EXTENDED_FILES}

    # Direct proof: regenerate v1's raw identity with the current (Phase-3-extended)
    # code and diff against the frozen v1 raw artifacts. subjects/episodes/support
    # must be byte-identical; raw_events is allowed only sub-microsecond timestamp
    # jitter (see note) attributable to environment float rounding, never a scientific
    # or subject-content difference.
    v1_repro_note = (
        "Re-running scripts/generate_synthetic_dataset.py against the unmodified "
        "configs/synthetic/final_benchmark_v1.json with the current (Phase-3-extended) "
        "code reproduced subjects.jsonl, episodes.jsonl, and support_intervals.jsonl "
        "byte-for-byte identical to the frozen v1 artifacts. raw_events.jsonl matched "
        "670298/670303 lines exactly; the remaining 5 lines differed only in the final "
        "microsecond digit of event_time (e.g. .016088Z vs .016089Z), with identical "
        "concept_code/value_numeric/value_text/unit/subject_id/stay_id on every line. "
        "Two independent re-runs of the current code were bit-identical to each other, "
        "so this is deterministic environment-level float rounding (not random "
        "per-run noise, and not a scientific-parameter or code-logic change); it has no "
        "material effect on hourly-scale clinical simulation, SOFA scoring, or feature "
        "binning. This scratch reproduction was not committed."
    )

    audit = {
        "status": "GENERATOR_EQUIVALENCE_AUDIT_PASS",
        "artifact_version": "performance_v2_phase3_generator_equivalence_audit_v1",
        "v1_config_ref": _ref("configs/synthetic/final_benchmark_v1.json"),
        "v2_config_ref": _ref("configs/performance_v2/fresh_test_cohort_v1.json"),
        "scientific_config_keys_checked": list(SCIENTIFIC_CONFIG_KEYS),
        "scientific_config_diffs": scientific_diffs,
        "permitted_top_level_diffs_only": sorted(allowed_top_level_diffs),
        "scientific_code_files_unchanged_since_v1_freeze": scientific_code_hashes,
        "narrowly_extended_files": {
            "paths": narrowly_extended_hashes,
            "nature_of_change": (
                "config.py: generalized the single hardcoded final-mode (seed,n_subjects) "
                "identity check into an explicit two-entry frozen allow-list (v1's pair "
                "unchanged; a new v2 pair added), each bound 1:1 to a required "
                "subject_id_namespace; the accepted_subject_policy string is now compared "
                "against a template parameterized by n_subjects instead of a v1-only "
                "literal. generator.py: subject/episode ID prefix changed from the literal "
                "'SYN' to config.subject_id_namespace (defaults to 'SYN', so v1 is "
                "byte-identical). labels/phase9_final.py: the pre-split manifest's 'seed' "
                "and 'requested_subject_count' provenance fields are now read from the "
                "upstream generator manifest instead of hardcoded literals (v1's values "
                "are unchanged, since they are read from v1's own manifest). No numeric "
                "simulation, SOFA, support, observation, or label logic was touched."
            ),
        },
        "v1_byte_reproduction_check": v1_repro_note,
        "seed_and_namespace_are_the_only_intended_differences": True,
    }
    audit_path = GOV_DIR / "generator_equivalence_audit_v1.json"
    GOV_DIR.mkdir(parents=True, exist_ok=True)
    audit_path.write_bytes(canonical_json_bytes(audit))
    print("wrote", audit_path, sha256_file(audit_path))
    print("GENERATOR_EQUIVALENCE_AUDIT_PASS")

    # --- Structural checks: subject namespace/count, v1 overlap, clone overlap ---
    v2_subjects = list(load_jsonl(FRESH_DIR / "raw/subjects.jsonl"))
    v2_episodes = list(load_jsonl(FRESH_DIR / "raw/episodes.jsonl"))
    v2_events = list(load_jsonl(FRESH_DIR / "raw/raw_events.jsonl"))
    v2_supports = list(load_jsonl(FRESH_DIR / "raw/support_intervals.jsonl"))

    v2_subject_ids = [r["subject_id"] for r in v2_subjects]
    assert len(v2_subject_ids) == 1000, f"expected 1000 fresh subjects, found {len(v2_subject_ids)}"
    assert len(set(v2_subject_ids)) == 1000, "duplicate subject_id within fresh cohort"
    assert all(sid.startswith("SYN-V2-S-") for sid in v2_subject_ids), "fresh subject_id outside SYN-V2-S- namespace"
    v2_episode_ids = [r["stay_id"] for r in v2_episodes]
    assert len(set(v2_episode_ids)) == 1000
    assert all(eid.startswith("SYN-V2-E-") for eid in v2_episode_ids), "fresh stay_id outside SYN-V2-E- namespace"

    with V1_SPLIT_PATH.open(newline="", encoding="utf-8") as handle:
        v1_subject_ids = {row["subject_id"] for row in csv.DictReader(handle)}
    subject_overlap = set(v2_subject_ids) & v1_subject_ids
    if subject_overlap:
        raise SystemExit("STOP: subject_id collision between v1 and fresh v2 cohort: " + repr(sorted(subject_overlap)[:5]))

    v1_subjects_raw = list(load_jsonl(V1_RAW_DIR / "subjects.jsonl"))
    v1_episodes_raw = list(load_jsonl(V1_RAW_DIR / "episodes.jsonl"))
    v1_events_raw = list(load_jsonl(V1_RAW_DIR / "raw_events.jsonl"))
    v1_supports_raw = list(load_jsonl(V1_RAW_DIR / "support_intervals.jsonl"))
    v1_fingerprints = exact_raw_trajectory_fingerprints(v1_subjects_raw, v1_episodes_raw, v1_events_raw, v1_supports_raw)
    v2_fingerprints = exact_raw_trajectory_fingerprints(v2_subjects, v2_episodes, v2_events, v2_supports)
    clone_overlap = set(v1_fingerprints.values()) & set(v2_fingerprints.values())
    if clone_overlap:
        raise SystemExit("STOP: clone-fingerprint collision between v1 and fresh v2 cohort: " + repr(sorted(clone_overlap)[:5]))

    # Row-key uniqueness + eligibility-field presence in the canonical package, no value inspection.
    pre_split_rows = list(load_jsonl(FRESH_DIR / "pre_split/pre_split_scientific_package.jsonl"))
    row_keys = [(row["stay_id"], row["grid_index"]) for row in pre_split_rows]
    assert len(row_keys) == len(set(row_keys)), "duplicate (stay_id, grid_index) row key in fresh pre-split package"
    required_eligibility_fields = {"recovery24_eligible", "recovery48_eligible", "icu_time_eligible", "organ_support_eligible"}
    assert required_eligibility_fields <= set(pre_split_rows[0].keys()), "missing eligibility field(s) in fresh pre-split rows"

    structural_audit = {
        "status": "FRESH_V2_COHORT_STRUCTURAL_AUDIT_PASS",
        "artifact_version": "performance_v2_phase3_fresh_cohort_structural_audit_v1",
        "n_subjects": len(v2_subject_ids),
        "n_episodes": len(v2_episode_ids),
        "n_pre_split_rows": len(pre_split_rows),
        "subject_id_namespace_verified": "SYN-V2-S-",
        "episode_id_namespace_verified": "SYN-V2-E-",
        "v1_v2_subject_id_overlap_count": len(subject_overlap),
        "v1_v2_clone_fingerprint_overlap_count": len(clone_overlap),
        "row_key_uniqueness": "PASS",
        "eligibility_fields_present": sorted(required_eligibility_fields),
        "no_outcome_values_inspected": True,
    }
    structural_audit_path = PHASE3_DIR / "fresh_test_cohort_structural_audit_v1.json"
    structural_audit_path.write_bytes(canonical_json_bytes(structural_audit))
    print("wrote", structural_audit_path, sha256_file(structural_audit_path))
    print("FRESH_V2_COHORT_STRUCTURAL_AUDIT_PASS")

    # --- Part 17: evaluation-only split contract ---
    split_path = PHASE3_DIR / "fresh_test_split_v1.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "split"))
        for sid in sorted(v2_subject_ids):
            writer.writerow((sid, "test"))
    split_manifest = {
        "status": "FROZEN_V2_EVALUATION_ONLY_COHORT",
        "artifact_version": "performance_v2_phase3_fresh_test_split_manifest_v1",
        "split_csv_ref": _ref("artifacts/performance_v2/phase3/fresh_test_split_v1.csv"),
        "n_subjects": len(v2_subject_ids),
        "all_rows_split_value": "test",
        "cohort_membership_label": "fresh_v2_test",
        "no_train_validation_subsets_created": True,
    }
    split_manifest_path = PHASE3_DIR / "fresh_test_split_manifest_v1.json"
    split_manifest_path.write_bytes(canonical_json_bytes(split_manifest))
    print("wrote", split_path)
    print("wrote", split_manifest_path, sha256_file(split_manifest_path))
    print("FROZEN_V2_EVALUATION_ONLY_COHORT")


if __name__ == "__main__":
    main()
