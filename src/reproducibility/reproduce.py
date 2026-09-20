"""Config-driven Phase-21 audit and synthetic reproduction command."""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Mapping, Optional

from evaluation.bootstrap import bootstrap_recovery
from evaluation.metrics import PredictionRecord, evaluate_recovery_horizon
from experiments.lineage import (
    ArtifactRecord,
    read_artifact_index,
    read_run_registry,
    register_artifact,
)
from reproducibility.audit import (
    audit_environment_lock,
    audit_path_portability,
    audit_registered_artifacts,
    audit_restricted_data,
    load_config,
)
from reproducibility.compare import compare_numeric_sequences
from vedant_infra.hashing import sha256_file


def _atomic_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _synthetic_records():
    return (
        PredictionRecord("stay-a", "t1", -2.0, -1.0, True),
        PredictionRecord("stay-a", "t2", 2.0, 1.0, True),
        PredictionRecord("stay-b", "t3", 3.0, 1.0, True),
    )


def synthetic_reproduction(config: Mapping[str, object]):
    records = _synthetic_records()
    expected_metrics = (1.5, 1.5811388300841898)
    result = evaluate_recovery_horizon(records, horizon="24h")
    reproduced_metrics = (result.metrics["mae"], result.metrics["rmse"])
    compare_numeric_sequences(
        expected_metrics,
        reproduced_metrics,
        tolerance=config.get("metric_regeneration_tolerance"),
    )
    synthetic = config["synthetic_ci"]
    first = bootstrap_recovery(
        records,
        records,
        n_bootstrap=synthetic["bootstrap_replicates"],
        seed=synthetic["bootstrap_seed"],
    )
    second = bootstrap_recovery(
        records,
        records,
        n_bootstrap=synthetic["bootstrap_replicates"],
        seed=synthetic["bootstrap_seed"],
    )
    first_ci = first["24h"]["mae"]
    second_ci = second["24h"]["mae"]
    compare_numeric_sequences(
        first_ci.bootstrap_distribution,
        second_ci.bootstrap_distribution,
        tolerance=0.0,
    )
    return {
        "status": "PASS",
        "scope": "synthetic_ci_only",
        "metric_regeneration": {
            "expected": expected_metrics,
            "reproduced": reproduced_metrics,
            "status": "PASS",
        },
        "bootstrap_regeneration": {
            "requested_replicates": first_ci.n_requested_replicates,
            "valid_replicates": first_ci.n_valid_replicates,
            "invalid_replicates": first_ci.n_invalid_replicates,
            "ci_lower": first_ci.ci_lower,
            "ci_upper": first_ci.ci_upper,
            "status": "PASS",
        },
    }


def build_manifest(root: Path, config: Mapping[str, object]):
    records = tuple(
        record
        for record in read_artifact_index(root / "experiments/artifacts.csv")
        if not record.artifact_id.startswith("phase21:")
    )
    final_refs = {
        "selected_models": "artifacts/models/selected_models_v1.json",
        "g3": "artifacts/governance/g3_freeze.json",
        "g4": "artifacts/governance/g4_evaluation_freeze.json",
    }
    required = {
        key: {
            "ref": reference,
            "sha256": sha256_file(root / reference) if (root / reference).is_file() else None,
            "status": "PRESENT" if (root / reference).is_file() else "MISSING",
        }
        for key, reference in final_refs.items()
    }
    return {
        "manifest_version": "reproduction_manifest_v1",
        "status": "BLOCKED_REAL_REPRODUCTION_PREREQUISITES",
        "repository_commit": config.get("repository_commit"),
        "repository_commit_status": "BLOCKED_NOT_A_GIT_REPOSITORY",
        "environment_lock_ref": config.get("environment_lock_ref"),
        "environment_lock_sha256": config.get("environment_lock_sha256"),
        "registered_artifacts": tuple(
            {
                "artifact_id": record.artifact_id,
                "artifact_ref": record.artifact_path,
                "artifact_sha256": record.artifact_sha256,
                "artifact_version": record.artifact_version,
            }
            for record in records
        ),
        "required_final_artifacts": required,
        "scientific_bindings": {
            "status": "UNAVAILABLE_REAL_PHASE19_20_NOT_EXECUTED",
            "dataset_version": None,
            "extraction_sha256": None,
            "feature_version": None,
            "feature_schema_sha256": None,
            "label_versions": None,
            "split_hash": None,
            "preprocessor_hashes": None,
            "selected_model_hashes": None,
            "support_calibrator_sha256": None,
            "support_threshold_sha256": None,
            "final_prediction_hashes": None,
            "final_metric_hashes": None,
        },
        "reproduction_commands": (
            "PYTHONPATH=src python3 -m reproducibility.reproduce audit --root .",
            "PYTHONPATH=src python3 -m reproducibility.reproduce synthetic --root .",
            "PYTHONPATH=src python3 -m reproducibility.reproduce frozen-artifacts --root .",
        ),
        "training_reproduction_tolerance": config.get("training_reproduction_tolerance"),
        "frozen_inference_tolerance": config.get("frozen_inference_tolerance"),
        "metric_regeneration_tolerance": config.get("metric_regeneration_tolerance"),
        "cross_member": config.get("cross_member"),
        "credentials_included": False,
    }


def build_report(root: Path, config: Mapping[str, object]):
    environment = audit_environment_lock(root, config)
    artifacts = audit_registered_artifacts(root)
    portability = audit_path_portability(root)
    restricted = audit_restricted_data(root)
    synthetic = synthetic_reproduction(config)
    real_artifacts = all(
        (root / path).is_file()
        for path in (
            "artifacts/models/selected_models_v1.json",
            "artifacts/governance/g4_evaluation_freeze.json",
        )
    )
    return {
        "report_version": "reproduction_report_v1",
        "overall_status": "BLOCKED",
        "mode": "repository_audit_plus_synthetic_ci",
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "layers": {
            "environment": environment,
            "artifact_integrity": artifacts,
            "path_portability": portability,
            "restricted_data": restricted,
            "synthetic_ci": synthetic,
            "frozen_inference": {
                "status": "BLOCKED",
                "detail": "real selected model/prediction fixture unavailable",
            },
            "metric_regeneration": {
                "status": "BLOCKED" if not real_artifacts else "NOT_EXECUTED",
                "detail": "real frozen final predictions/metrics unavailable",
            },
            "bootstrap_regeneration": {
                "status": "BLOCKED",
                "detail": "real bootstrap B/seed and final predictions unavailable",
            },
            "registry_trace": {
                "status": artifacts["status"],
                "detail": "registered development/synthetic artifacts trace; no final result row exists",
            },
            "cross_member_train_eval": {
                "status": "BLOCKED",
                "detail": "no designated real frozen run and no non-owner evidence",
            },
            "upstream_data_regeneration": {
                "status": "PENDING",
                "detail": "pending Sanskruti/team cross-reproduction",
            },
            "full_demo": {
                "status": "DEFERRED_TO_PULKIT",
                "detail": "product packaging and demo layer are outside Phase 21",
            },
        },
        "scientific_outputs_modified": False,
        "final_test_reopened": False,
        "failures": (
            "BLOCKED — EXACT ENVIRONMENT LOCK REQUIRED FOR FINAL REPRODUCIBILITY CLAIM",
            "BLOCKED — REAL PHASE-19/20/G4 ARTIFACTS REQUIRED",
            "UNLOCKED ENGINEERING PARAMETER — CROSS-MEMBER REPRODUCTION RUN",
            "BLOCKED — RETRAINING REPRODUCIBILITY TOLERANCE REQUIRED",
        ),
    }


def cross_member_placeholder(config: Mapping[str, object]):
    return {
        "evidence_version": "cross_member_reproduction_v1",
        "status": "BLOCKED",
        "reproducer_name": None,
        "reproducer_role": None,
        "designated_run_id": config["cross_member"]["designated_run_id"],
        "command": None,
        "resulting_checkpoint_sha256": None,
        "expected_metrics": None,
        "reproduced_metrics": None,
        "comparison_outcome": None,
        "detail": "non-owner execution has not occurred; this is not signoff evidence",
    }


def write_outputs(root: Path, config: Mapping[str, object]):
    outputs = config["outputs"]
    manifest = _atomic_json(root / outputs["manifest"], build_manifest(root, config))
    report = _atomic_json(root / outputs["report"], build_report(root, config))
    cross_member = _atomic_json(
        root / outputs["cross_member"], cross_member_placeholder(config)
    )
    return manifest, report, cross_member


def register_outputs(root: Path, paths) -> None:
    definitions = (
        ("manifest", paths[0], "reproduction_manifest"),
        ("report", paths[1], "reproduction_report"),
        ("cross_member", paths[2], "cross_member_reproduction_status"),
    )
    index_path = root / "experiments/artifacts.csv"
    runs = read_run_registry(root / "experiments/registry.csv")
    config_hash = sha256_file(root / "configs/reproducibility_v1.json")
    for name, path, artifact_type in definitions:
        relative = str(path.relative_to(root))
        digest = sha256_file(path)
        identifier = "phase21:" + name + ":" + digest[:16]
        existing = read_artifact_index(index_path)
        matching = [record for record in existing if record.artifact_id == identifier]
        if matching:
            if matching[0].artifact_path != relative or matching[0].artifact_sha256 != digest:
                raise RuntimeError("registered reproducibility evidence conflicts")
            continue
        register_artifact(
            index_path,
            ArtifactRecord(
                artifact_id=identifier,
                artifact_path=relative,
                artifact_type=artifact_type,
                artifact_version="reproducibility_v1",
                artifact_sha256=digest,
                producing_run_id="",
                config_hash=config_hash,
                generating_script="src/reproducibility/reproduce.py",
                run_type="development",
                status="registered",
            ),
            runs,
            root,
        )


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("audit", "synthetic", "frozen-artifacts", "train-eval")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = load_config(root / "configs/reproducibility_v1.json")
    if args.mode == "train-eval":
        designated = config["cross_member"].get("designated_run_id")
        if not designated or args.run_id != designated:
            parser.exit(2, "BLOCKED — DESIGNATED CROSS-MEMBER RUN REQUIRED\n")
        parser.exit(2, "BLOCKED — NON-OWNER EXECUTION EVIDENCE REQUIRED\n")
    if args.mode == "frozen-artifacts":
        if not (root / "artifacts/governance/g4_evaluation_freeze.json").is_file():
            parser.exit(2, "BLOCKED — FROZEN G4 ARTIFACTS REQUIRED\n")
    manifest, report, cross_member = write_outputs(root, config)
    register_outputs(root, (manifest, report, cross_member))
    payload = synthetic_reproduction(config) if args.mode == "synthetic" else build_report(root, config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("manifest=" + str(manifest))
    print("report=" + str(report))
    print("cross_member=" + str(cross_member))
    if args.mode == "audit":
        parser.exit(2, "real reproducibility audit is BLOCKED\n")


if __name__ == "__main__":
    main(sys.argv[1:])
