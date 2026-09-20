"""Freeze-gated raw-count reporting for organ-support labels.

The caller supplies the label loader.  Every governance check runs before that
callback, which keeps real labels unreachable while the endpoint is unfrozen.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from labels.endpoint_freeze import (
    EndpointFreezeError,
    ENDPOINT_VERSION,
    _write_immutable_json,
    utc_now,
    validate_freeze_record,
)
from vedant_infra.g3 import validate_g3_marker
from vedant_infra.hashing import is_sha256, sha256_file


GENERATOR_VERSION = "support_prevalence_generator_v1"
LABEL_ARTIFACT_VERSION = "support_label_artifact_v1"
REPORT_VERSION = "support_prevalence_raw_counts_v1"
POSITIVE = "INTERNAL_POSITIVE_INITIATION"
NEGATIVE = "INTERNAL_NEGATIVE_FULL_FOLLOWUP"
CENSORED = "INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION"
NOT_AT_RISK = "INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON"
ALLOWED_CASES = {POSITIVE, NEGATIVE, CENSORED, NOT_AT_RISK}


class SupportPrevalenceError(RuntimeError):
    """Raised before unsafe label access or for invalid label/report content."""


def _require_within_root(root: Path, path: Path, kind: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise SupportPrevalenceError(kind + " must be inside the repository root")
    return resolved


def _json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupportPrevalenceError("unreadable JSON: " + str(path)) from error
    if not isinstance(value, dict):
        raise SupportPrevalenceError("JSON artifact must be an object")
    return value


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise SupportPrevalenceError(field + " must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SupportPrevalenceError(field + " is malformed") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SupportPrevalenceError(field + " must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _authorized_population(config: Mapping[str, object], scope: str, population_id: str, partition: str) -> None:
    key = "real_authorized_populations" if scope == "real" else "synthetic_authorized_populations"
    populations = config.get(key)
    if not isinstance(populations, list):
        raise SupportPrevalenceError("authorized-population configuration is malformed")
    matches = [item for item in populations if isinstance(item, dict) and item.get("population_id") == population_id]
    if len(matches) != 1 or matches[0].get("scope") != scope:
        raise SupportPrevalenceError("population is not explicitly authorized for freeze scope")
    partitions = matches[0].get("allowed_partitions")
    if not isinstance(partitions, list) or partition not in partitions:
        raise SupportPrevalenceError("partition is not authorized for reporting population")


def _preload_guard(
    root: Path,
    *,
    freeze_path: Path,
    freeze_sha256: str,
    scope: str,
    population_id: str,
    partition: str,
    config_path: Path,
    g3_marker_path: Optional[Path],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    freeze_path = _require_within_root(root, freeze_path, "freeze artifact")
    config_path = _require_within_root(root, config_path, "governance config")
    if scope not in ("real", "synthetic"):
        raise SupportPrevalenceError("scope must be real or synthetic")
    try:
        freeze = validate_freeze_record(
            root, freeze_path, expected_sha256=freeze_sha256, requested_scope=scope
        )
    except EndpointFreezeError as error:
        raise SupportPrevalenceError(str(error)) from error
    if freeze.get("prevalence_access_authorized") is not True:
        raise SupportPrevalenceError("freeze does not authorize prevalence access")
    config = _json(config_path)
    if (
        config.get("schema_version") != "support_prevalence_governance_v1"
        or config.get("generator_version") != GENERATOR_VERSION
        or config.get("headline_prevalence_enabled") is not False
        or config.get("class_weight_fitting_in_scope") is not False
        or config.get("prevalence_range_acceptance_gate") is not False
    ):
        raise SupportPrevalenceError("prevalence governance contract is incompatible")
    _authorized_population(config, scope, population_id, partition)
    protected = config.get("protected_test_partition")
    if partition == protected:
        if g3_marker_path is None:
            raise SupportPrevalenceError("protected test partition rejected before label loading")
        try:
            validate_g3_marker(g3_marker_path, root, expected_scope=scope)
        except Exception as error:
            raise SupportPrevalenceError("invalid G3 marker; test labels remain protected") from error
        if freeze.get("test_label_access_authorized") is not True:
            raise SupportPrevalenceError("endpoint freeze does not authorize test-label access")
    return freeze, config


def create_synthetic_label_artifact(
    root: Path,
    fixture_path: Path,
    output_path: Path,
    *,
    freeze_path: Path,
    freeze_sha256: str,
) -> str:
    """Materialize a synthetic-only label envelope after an active synthetic freeze."""

    freeze, _ = _preload_guard(
        root,
        freeze_path=freeze_path,
        freeze_sha256=freeze_sha256,
        scope="synthetic",
        population_id="SYNTHETIC_PHASE4_FIXTURE_ONLY",
        partition="synthetic",
        config_path=root / "configs/support_prevalence_v1.json",
        g3_marker_path=None,
    )
    fixture = _json(fixture_path)
    if fixture.get("fixture_kind") != "SYNTHETIC_PHASE4_LABEL_ROWS_NOT_REAL_NOT_AUTHORIZING":
        raise SupportPrevalenceError("only the declared synthetic fixture may be materialized")
    payload = {
        "artifact_version": LABEL_ARTIFACT_VERSION,
        "artifact_scope": "synthetic",
        "synthetic_only": True,
        "generated_at_utc": utc_now(),
        "endpoint_version": ENDPOINT_VERSION,
        "generator_implementation_ref": "src/labels/support_prevalence.py",
        "generator_implementation_sha256": sha256_file(
            root / "src/labels/support_prevalence.py"
        ),
        "population_id": fixture.get("population_id"),
        "partition": fixture.get("partition"),
        "data_version": fixture.get("data_version"),
        "split_version": fixture.get("split_version"),
        "timestamp_spec_version": "timestamp_spec_v1",
        "freeze_ref": str(freeze_path.relative_to(root)),
        "freeze_sha256": freeze_sha256,
        "event_dictionary_sha256": freeze.get("event_dictionary_sha256"),
        "source_fixture_ref": str(fixture_path.relative_to(root)),
        "source_fixture_sha256": sha256_file(fixture_path),
        "records": fixture.get("records"),
    }
    return _write_immutable_json(output_path, payload)


def _validate_label_envelope(
    root: Path,
    envelope: Mapping[str, object],
    descriptor: Mapping[str, object],
    freeze: Mapping[str, object],
) -> Sequence[Mapping[str, object]]:
    exact = {
        "artifact_version": LABEL_ARTIFACT_VERSION,
        "artifact_scope": descriptor.get("scope"),
        "endpoint_version": ENDPOINT_VERSION,
        "population_id": descriptor.get("population_id"),
        "partition": descriptor.get("partition"),
        "data_version": descriptor.get("data_version"),
        "split_version": descriptor.get("split_version"),
        "timestamp_spec_version": descriptor.get("timestamp_spec_version"),
        "freeze_sha256": descriptor.get("freeze_sha256"),
        "event_dictionary_sha256": freeze.get("event_dictionary_sha256"),
        "generator_implementation_ref": "src/labels/support_prevalence.py",
        "generator_implementation_sha256": sha256_file(
            root / "src/labels/support_prevalence.py"
        ),
    }
    for field, expected in exact.items():
        if not expected or envelope.get(field) != expected:
            raise SupportPrevalenceError("label artifact binding mismatch: " + field)
    if descriptor.get("scope") == "real" and envelope.get("synthetic_only") is True:
        raise SupportPrevalenceError("synthetic labels cannot authorize real reporting")
    records = envelope.get("records")
    if not isinstance(records, list):
        raise SupportPrevalenceError("label records must be a list")
    return records


def _raw_counts(records: Sequence[Mapping[str, object]]) -> Mapping[str, int]:
    keys = set()
    stays = set()
    eligible_stays = set()
    positive_stays = set()
    counts = {"eligible_rows": 0, "positive_rows": 0, "negative_rows": 0, "censored_rows": 0, "not_at_risk_rows": 0}
    for row in records:
        if not isinstance(row, dict):
            raise SupportPrevalenceError("each label row must be an object")
        stay = row.get("stay_id")
        time = row.get("prediction_time")
        if not isinstance(stay, (str, int)) or isinstance(stay, bool) or not str(stay):
            raise SupportPrevalenceError("stay_id is missing")
        _parse_time(time, "prediction_time")
        key = (str(stay), str(time))
        if key in keys:
            raise SupportPrevalenceError("duplicate stay_id/prediction_time label row")
        keys.add(key)
        stays.add(str(stay))
        eligible, label, case = row.get("eligible"), row.get("label"), row.get("audit_case")
        if case not in ALLOWED_CASES:
            raise SupportPrevalenceError("unknown audit-case vocabulary")
        expected = {
            POSITIVE: (True, 1), NEGATIVE: (True, 0),
            CENSORED: (False, None), NOT_AT_RISK: (False, None),
        }[case]
        if (eligible, label) != expected:
            raise SupportPrevalenceError("eligibility/label contradicts audit case")
        if eligible:
            counts["eligible_rows"] += 1
            eligible_stays.add(str(stay))
        if case == POSITIVE:
            counts["positive_rows"] += 1
            positive_stays.add(str(stay))
        elif case == NEGATIVE:
            counts["negative_rows"] += 1
        elif case == CENSORED:
            counts["censored_rows"] += 1
        else:
            counts["not_at_risk_rows"] += 1
    return {
        "canonical_rows": len(records), **counts, "unique_stays": len(stays),
        "eligible_stays": len(eligible_stays), "positive_stays": len(positive_stays),
    }


def generate_raw_count_report(
    root: Path,
    *,
    freeze_path: Path,
    freeze_sha256: str,
    scope: str,
    population_id: str,
    partition: str,
    label_artifact_path: Path,
    label_artifact_sha256: str,
    label_loader: Callable[[Path], Mapping[str, object]],
    descriptor: Mapping[str, object],
    output_path: Path,
    config_path: Optional[Path] = None,
    g3_marker_path: Optional[Path] = None,
) -> str:
    """Generate raw counts; label_loader is invoked only after all access guards."""

    root = root.resolve()
    config_path = config_path or root / "configs/support_prevalence_v1.json"
    label_artifact_path = _require_within_root(root, label_artifact_path, "label artifact")
    freeze, config = _preload_guard(
        root, freeze_path=freeze_path, freeze_sha256=freeze_sha256, scope=scope,
        population_id=population_id, partition=partition, config_path=config_path,
        g3_marker_path=g3_marker_path,
    )
    required_descriptor = {
        "scope": scope, "population_id": population_id, "partition": partition,
        "freeze_sha256": freeze_sha256, "label_artifact_sha256": label_artifact_sha256,
    }
    for field, expected in required_descriptor.items():
        if descriptor.get(field) != expected:
            raise SupportPrevalenceError("label descriptor mismatch: " + field)
    if not is_sha256(label_artifact_sha256) or not label_artifact_path.is_file():
        raise SupportPrevalenceError("label artifact/hash is missing or malformed")
    if sha256_file(label_artifact_path) != label_artifact_sha256:
        raise SupportPrevalenceError("label artifact hash mismatch")

    envelope = label_loader(label_artifact_path)
    if not isinstance(envelope, Mapping):
        raise SupportPrevalenceError("label loader returned a malformed envelope")
    records = _validate_label_envelope(root, envelope, descriptor, freeze)
    counts = _raw_counts(records)
    required_counts = config.get("required_raw_counts")
    if not isinstance(required_counts, list) or set(required_counts) != set(counts):
        raise SupportPrevalenceError("raw-count output contract mismatch")
    generated = utc_now()
    if _parse_time(generated, "generated_at_utc") < _parse_time(freeze.get("freeze_timestamp_utc"), "freeze_timestamp_utc"):
        raise SupportPrevalenceError("report generation precedes endpoint freeze")
    report = {
        "report_version": REPORT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "report_scope": scope,
        "synthetic_only": scope == "synthetic",
        "generated_at_utc": generated,
        "endpoint_version": ENDPOINT_VERSION,
        "generator_implementation_ref": "src/labels/support_prevalence.py",
        "generator_implementation_sha256": sha256_file(
            root / "src/labels/support_prevalence.py"
        ),
        "population_id": population_id,
        "partition": partition,
        "data_version": descriptor.get("data_version"),
        "split_version": descriptor.get("split_version"),
        "timestamp_spec_version": descriptor.get("timestamp_spec_version"),
        "freeze_ref": str(freeze_path.relative_to(root)),
        "freeze_sha256": freeze_sha256,
        "event_dictionary_sha256": freeze.get("event_dictionary_sha256"),
        "label_artifact_ref": str(label_artifact_path.relative_to(root)),
        "label_artifact_sha256": label_artifact_sha256,
        "governance_config_ref": str(config_path.relative_to(root)),
        "governance_config_sha256": sha256_file(config_path),
        "raw_counts": counts,
        "headline_denominator_status": config.get("headline_denominator_status"),
        "headline_prevalence": None,
        "class_weights": None,
        "acceptance_gate_applied": False,
    }
    try:
        return _write_immutable_json(output_path, report)
    except EndpointFreezeError as error:
        raise SupportPrevalenceError(str(error)) from error


def json_label_loader(path: Path) -> Mapping[str, object]:
    return _json(path)
