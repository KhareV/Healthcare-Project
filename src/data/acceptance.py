"""Machine-readable structural acceptance for real canonical data."""

import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional

from data.real_adapter import (
    RealDataBundle,
    RealDataContractError,
    assert_information_parity,
    load_real_bundle,
    loader_projection,
)
from experiments.search_governance import canonical_sha256
from experiments.lineage import (
    ArtifactRecord,
    read_artifact_index,
    read_run_registry,
    register_artifact,
)
from vedant_infra.hashing import sha256_file


REPORT_VERSION = "real_data_acceptance_v1"
ACCEPTED = "ACCEPTED"
BLOCKED = "BLOCKED"


class DataAcceptanceError(RuntimeError):
    """Raised when an acceptance artifact cannot authorize Phase 19."""


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataAcceptanceError("unreadable acceptance JSON: " + str(path)) from error
    if not isinstance(payload, Mapping):
        raise DataAcceptanceError("acceptance JSON must contain an object")
    return payload


def _counts(bundle: RealDataBundle):
    counts = {}
    for split in ("train", "validation", "test"):
        structural = [row for row in bundle.structural_rows if row.split == split]
        examples = [row for row in bundle.examples if row.split == split]
        counts[split] = {
            "canonical_examples": len(structural),
            "icu_stays": len({(type(row.stay_id).__name__, str(row.stay_id)) for row in structural}),
            "subjects": len({(type(row.subject_id).__name__, str(row.subject_id)) for row in structural}),
            "recovery24_eligible": sum(row.eligibility.recovery24_eligible for row in examples) if split != "test" else "NOT_INSPECTED",
            "recovery48_eligible": sum(row.eligibility.recovery48_eligible for row in examples) if split != "test" else "NOT_INSPECTED",
            "icu_time_eligible": sum(row.eligibility.icu_time_eligible for row in examples) if split != "test" else "NOT_INSPECTED",
            "organ_support_eligible": sum(row.eligibility.organ_support_eligible for row in examples) if split != "test" else "NOT_INSPECTED",
        }
    return counts


def _missingness(bundle: RealDataBundle):
    result = {}
    sentinel = bundle.feature_schema.tslo_no_observation_value
    for split in ("train", "validation"):
        examples = [item for item in bundle.examples if item.split == split]
        total_dynamic = len(examples) * 8 * len(bundle.dynamic_feature_names)
        total_static = len(examples) * len(bundle.static_feature_names)
        observed = sum(
            value
            for item in examples
            for row in item.observation_mask
            for value in row
        )
        padding = sum(value for item in examples for value in item.padding_mask)
        tslo_sentinel = sum(
            value == sentinel
            for item in examples
            for row in (item.tslo_hours or ())
            for value in row
        )
        static_missing = sum(
            value is None
            for item in examples
            for value in (item.static_features or ())
        )
        result[split] = {
            "dynamic_cells": total_dynamic,
            "genuine_observation_cells": observed,
            "padding_bins": padding,
            "tslo_sentinel_cells": tslo_sentinel,
            "static_cells": total_static,
            "static_missing_cells": static_missing,
        }
    result["test"] = "STRUCTURAL_COUNTS_ONLY_FEATURE_DISTRIBUTION_NOT_INSPECTED"
    return result


def _loader_parity(bundle: RealDataBundle) -> None:
    for example in bundle.examples:
        view = loader_projection(example, bundle.dynamic_feature_names)
        if (
            view["history_values"] != example.history_values
            or view["dynamic_feature_names"] != bundle.dynamic_feature_names
            or view["observation_mask"] != example.observation_mask
            or view["padding_mask"] != example.padding_mask
            or view["tslo_hours"] != example.tslo_hours
            or view["static_features"] != example.static_features
            or view["eligibility"] != example.eligibility
            or view["targets"] != example.targets
        ):
            raise RealDataContractError("receiver loader projection changed canonical information")


def _check(name, status, detail):
    return {"name": name, "status": status, "detail": detail}


def audit_real_data(
    root: Path,
    *,
    manifest_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
):
    """Audit structural inputs only; no model or performance API is called."""

    root = root.resolve()
    config_path = config_path or root / "configs/data_acceptance_v1.json"
    config = _load_json(config_path)
    manifest_path = (manifest_path or root / str(config["input_manifest"])).resolve()
    checks = []
    blockers = []
    if not manifest_path.is_file():
        expected = config.get("expected_inputs", {})
        inventory = []
        for name, ref in sorted(expected.items()):
            path = root / ref
            if name == "support_handoff":
                owner = "Pulkit"
            elif name in ("split", "split_metadata", "tensor_contract", "timestamp_spec"):
                owner = "Vedant"
            else:
                owner = "Sanskruti"
            inventory.append(
                {
                    "name": name,
                    "ref": ref,
                    "owner": owner,
                    "sha256": sha256_file(path) if path.is_file() else None,
                    "status": "PRESENT_UNBOUND" if path.is_file() else "MISSING",
                }
            )
        messages = (
            "BLOCKED — INFORMATION REQUIRED: real-data input manifest",
            "BLOCKED — INFORMATION REQUIRED: Sanskruti canonical/schema/feature/SOFA/label artifacts",
            "BLOCKED — INFORMATION REQUIRED: frozen split_v1 artifact",
            "BLOCKED — PULKIT REAL ORGAN-SUPPORT LABEL ARTIFACT REQUIRED",
        )
        checks.extend(_check("upstream_inputs", BLOCKED, message) for message in messages)
        blockers.extend(messages)
        report = {
            "report_version": REPORT_VERSION,
            "overall_status": BLOCKED,
            "input_manifest": str(manifest_path.relative_to(root)),
            "input_inventory": inventory,
            "versions": {},
            "counts": {},
            "missingness_structural_audit": {},
            "checks": checks,
            "unresolved_blockers": blockers,
            "test_labels_accessed": False,
            "performance_inspected": False,
            "phase19_authorized": False,
        }
        report["acceptance_decision_sha256"] = canonical_sha256(report)
        return report
    try:
        bundle = load_real_bundle(root, manifest_path)
        checks.extend(
            (
                _check("schema_and_versions", "PASS", "all hash-bound versions match"),
                _check("canonical_key_uniqueness", "PASS", "stay_id+prediction_time unique"),
                _check("timestamp_parity", "PASS", "full structural index equals regenerated grid"),
                _check("tensor_shape", "PASS", "every development example is [8,F]"),
                _check("feature_order_and_dtypes", "PASS", "sidecar order and dtype declarations match"),
                _check("mask_tslo_padding", "PASS", "receiver contract validated"),
                _check("static_features", "PASS", "ordered upstream static contract validated"),
                _check("recovery_labels", "PASS", "independent baseline-relative formulas checked"),
                _check("icu_time_labels", "PASS", "outtime-cutoff and log1p formulas checked"),
                _check("split_subject_isolation", "PASS", "split_v1 mapping and isolation checked"),
                _check("preprocessing_provenance", "PASS", "train-only provenance/interface checked"),
                _check("structural_leakage", "PASS", "hash-bound upstream structural evidence checked"),
                _check("test_isolation", "PASS", "test structural index only; outcomes unopened"),
            )
        )
        assert_information_parity(bundle)
        checks.append(_check("xgb_gru_information_parity", "PASS", "XGBoost is row-major flattening only"))
        _loader_parity(bundle)
        checks.append(_check("loader_parity", "PASS", "lossless receiver projection checked for every development row"))
        if bundle.support_status != "PASS":
            blockers.append(bundle.support_status)
            checks.append(_check("support_handoff", BLOCKED, bundle.support_status))
        else:
            checks.append(_check("support_handoff", "PASS", "Pulkit evidence joined one-to-one and receiver checks passed"))
        inventory = [asdict(item) for item in bundle.inventory]
        report = {
            "report_version": REPORT_VERSION,
            "overall_status": ACCEPTED if not blockers else BLOCKED,
            "input_manifest": str(manifest_path.relative_to(root)),
            "input_inventory": inventory,
            "versions": dict(bundle.versions),
            "canonical_content_sha256": bundle.canonical_content_sha256,
            "structural_content_sha256": bundle.structural_content_sha256,
            "counts": _counts(bundle),
            "missingness_structural_audit": _missingness(bundle),
            "checks": checks,
            "manual_spot_check_policy": "FULL_FORMULA_CHECK_PLUS_DETERMINISTIC_KEY_ORDER_NO_IDS_REPORTED",
            "unresolved_blockers": blockers,
            "test_labels_accessed": False,
            "performance_inspected": False,
            "phase19_authorized": not blockers,
        }
    except (RealDataContractError, DataAcceptanceError, KeyError, TypeError, ValueError) as error:
        detail = str(error)
        blockers.append(detail)
        report = {
            "report_version": REPORT_VERSION,
            "overall_status": BLOCKED,
            "input_manifest": str(manifest_path.relative_to(root)),
            "input_inventory": [],
            "versions": {},
            "counts": {},
            "missingness_structural_audit": {},
            "checks": [_check("real_data_integration", BLOCKED, detail)],
            "unresolved_blockers": blockers,
            "test_labels_accessed": False,
            "performance_inspected": False,
            "phase19_authorized": False,
        }
    decision = dict(report)
    decision.pop("input_inventory", None)
    report["acceptance_decision_sha256"] = canonical_sha256(decision)
    return report


def write_acceptance_report(path: Path, report: Mapping[str, object]) -> Path:
    payload = dict(report)
    content = dict(payload)
    content.pop("report_content_sha256", None)
    payload["report_content_sha256"] = canonical_sha256(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def register_acceptance_snapshot(root: Path, report_path: Path) -> Path:
    """Register one immutable content-addressed report snapshot in Phase-16 lineage."""

    root = root.resolve()
    digest = sha256_file(report_path)
    relative = Path("artifacts/data_acceptance/history") / (
        "real_data_acceptance_v1_" + digest[:16] + ".json"
    )
    snapshot = root / relative
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if sha256_file(snapshot) != digest:
            raise DataAcceptanceError("acceptance snapshot path collision")
    else:
        shutil.copy2(report_path, snapshot)
    artifact_id = "phase18:data_acceptance:" + digest[:16]
    records = read_artifact_index(root / "experiments/artifacts.csv")
    matches = [item for item in records if item.artifact_id == artifact_id]
    if matches:
        if matches[0].artifact_sha256 != digest or matches[0].artifact_path != str(relative):
            raise DataAcceptanceError("registered acceptance snapshot conflicts")
        return snapshot
    record = ArtifactRecord(
        artifact_id=artifact_id,
        artifact_path=str(relative),
        artifact_type="data_acceptance_report",
        artifact_version=REPORT_VERSION,
        artifact_sha256=digest,
        producing_run_id="",
        config_hash=sha256_file(root / "configs/data_acceptance_v1.json"),
        generating_script="src/experiments/real_data_acceptance_cli.py",
        run_type="development",
        status="registered",
    )
    register_artifact(
        root / "experiments/artifacts.csv",
        record,
        read_run_registry(root / "experiments/registry.csv"),
        root,
    )
    return snapshot


def require_accepted_real_data(report_path: Path, expected_file_sha256: str):
    """Phase-19 precondition; validates content and exact frozen report bytes."""

    if not report_path.is_file() or sha256_file(report_path) != expected_file_sha256:
        raise DataAcceptanceError("real-data acceptance report missing or hash mismatch")
    report = _load_json(report_path)
    content = dict(report)
    supplied = content.pop("report_content_sha256", None)
    if supplied != canonical_sha256(content):
        raise DataAcceptanceError("real-data acceptance report content hash mismatch")
    if report.get("overall_status") != ACCEPTED or report.get("phase19_authorized") is not True:
        raise DataAcceptanceError("Phase 19 refused: real data are not ACCEPTED")
    if report.get("test_labels_accessed") is not False or report.get("performance_inspected") is not False:
        raise DataAcceptanceError("acceptance report records prohibited information access")
    return report
