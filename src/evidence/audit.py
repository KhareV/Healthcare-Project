"""Evidence consistency, traceability, terminology, and provenance audit."""

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.selection_validation import EXPLANATION_METHOD
from vedant_infra.hashing import is_sha256, sha256_file


class EvidenceAuditError(RuntimeError):
    """Raised when generated evidence is stale, misleading, or untraceable."""


ALLOWED_STATUSES = {"FINAL", "VALIDATION-ONLY", "BLOCKED", "DEFERRED"}
PROHIBITED_TERMS = (
    "hospital discharge ETA",
    "survival prediction",
    "clinically validated",
    "real-time clinical prediction",
    "treatment recommendation system",
    "patient-specific confidence interval",
    "fully reproducible",
)


def load_manifest(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest_version") != "vedant_evidence_manifest_v1":
        raise EvidenceAuditError("evidence manifest version mismatch")
    return value


def audit_manifest(root: Path, manifest: Mapping[str, object]):
    evidence_root = root / manifest["evidence_root"]
    declared = set()
    for item in manifest["items"]:
        if item.get("status") not in ALLOWED_STATUSES:
            raise EvidenceAuditError("evidence item has invalid status")
        reference = item.get("output_path")
        if not isinstance(reference, str) or Path(reference).is_absolute():
            raise EvidenceAuditError("evidence output path must be repository-relative")
        path = root / reference
        if not path.is_file() or sha256_file(path) != item.get("output_sha256"):
            raise EvidenceAuditError("evidence output hash mismatch: " + str(reference))
        if not item.get("generator") or not item.get("generated_at"):
            raise EvidenceAuditError("evidence provenance is incomplete")
        inputs = item.get("input_artifacts")
        hashes = item.get("input_hashes")
        if not isinstance(inputs, list) or not isinstance(hashes, list) or len(inputs) != len(hashes):
            raise EvidenceAuditError("evidence input provenance is malformed")
        for reference_value, digest in zip(inputs, hashes):
            source = root / reference_value
            if not source.is_file() or not is_sha256(digest) or sha256_file(source) != digest:
                raise EvidenceAuditError("evidence source hash mismatch")
        declared.add(path.resolve())
    manifest_path = evidence_root / "evidence_manifest_v1.json"
    allowed_unlisted = {manifest_path.resolve()}
    actual = {
        path.resolve()
        for path in evidence_root.rglob("*")
        if path.is_file()
    }
    orphans = actual - declared - allowed_unlisted
    if orphans:
        raise EvidenceAuditError("orphan evidence output: " + str(sorted(orphans)[0]))
    return {"status": "PASS", "evidence_items": len(declared), "orphan_outputs": 0}


def audit_terminology(paths: Sequence[Path]):
    findings = []
    for path in paths:
        if path.suffix.lower() not in {".md", ".csv", ".json", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for term in PROHIBITED_TERMS:
            if term.lower() in text.lower():
                findings.append(str(path) + ": " + term)
    if findings:
        raise EvidenceAuditError("prohibited terminology: " + "; ".join(findings))
    return {"status": "PASS", "findings": 0}


def validate_result_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        if row.get("split") not in ("validation", "test"):
            raise EvidenceAuditError("result row has ambiguous split")
        if row.get("task") == "recovery" and row.get("horizon") not in ("24h", "48h"):
            raise EvidenceAuditError("recovery evidence must keep horizons separate")
        if row.get("task") == "organ_support" and row.get("selection_criterion") != "validation_uncalibrated_auprc":
            raise EvidenceAuditError("support selection criterion is not frozen AUPRC")
        for count in ("N_examples", "N_ICU_stays"):
            if row.get(count) is None:
                raise EvidenceAuditError("result evidence lacks denominator counts")
        if row.get("ci_label") and row.get("ci_label") != "aggregate performance 95% CI":
            raise EvidenceAuditError("grouped bootstrap CI is mislabeled")


def validate_calibration_evidence(value: Mapping[str, object]) -> None:
    if (
        value.get("method") != "isotonic"
        or value.get("fit_partition") != "validation"
        or value.get("threshold_criterion") != "validation_f1"
        or value.get("selected_classifier_only") is not True
        or value.get("refit_during_evidence") is not False
    ):
        raise EvidenceAuditError("calibration evidence contradicts frozen method")


def validate_bootstrap_evidence(value: Mapping[str, object]) -> None:
    if (
        value.get("resampling_unit") != "stay_id"
        or value.get("rows_sampled_independently") is not False
        or value.get("ci_quantiles") != [0.025, 0.975]
        or value.get("uncertainty_scope") != "aggregate_performance"
    ):
        raise EvidenceAuditError("bootstrap evidence contradicts Phase-10 contract")


def validate_selected_routing(tasks: Mapping[str, object]) -> None:
    for entry in tasks.values():
        family = entry.get("family")
        if family not in EXPLANATION_METHOD or entry.get("explanation_method") != EXPLANATION_METHOD[family]:
            raise EvidenceAuditError("selected-family explanation routing mismatch")


def audit_generated_package(root: Path):
    manifest_path = root / "docs/evidence/models/evidence_manifest_v1.json"
    manifest = load_manifest(manifest_path)
    trace = audit_manifest(root, manifest)
    paths = tuple(root / item["output_path"] for item in manifest["items"])
    terms = audit_terminology(paths)
    values = json.loads((root / "docs/evidence/models/report_values_v1.json").read_text(encoding="utf-8"))
    if values.get("scientific_values"):
        raise EvidenceAuditError("scientific values require unavailable registered result sources")
    return {"status": "PASS", "traceability": trace, "terminology": terms}
