"""Fail-closed Phase-20 evaluation orchestration with no training surface."""

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch

from evaluation.bootstrap import (
    bootstrap_icu_time,
    bootstrap_organ_support,
    bootstrap_recovery,
)
from evaluation.calibrate import apply_support_calibrator
from evaluation.metrics import (
    PredictionRecord,
    evaluate_icu_time,
    evaluate_organ_support,
    evaluate_recovery,
)
from evaluation.error_analysis import evaluate_error_analysis
from evaluation.sensitivity import (
    CompleteComponentRecord,
    evaluate_complete_component_sensitivity,
)
from experiments.audit import audit_registry_lineage
from experiments.lineage import (
    ArtifactRecord,
    read_artifact_index,
    read_run_registry,
    register_artifact,
)
from experiments.search_governance import canonical_sha256
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from vedant_infra.g3 import (
    ACCESS_STATE_RELATIVE_PATH,
    G3FreezeError,
    load_access_state,
    validate_g3_marker,
)
from vedant_infra.hashing import is_sha256, sha256_file


FINAL_TEST_CONFIG_VERSION = "final_test_evaluation_v1"
PRETEST_AUDIT_VERSION = "phase20_pretest_audit_v1"
BLOCKED_G3 = "BLOCKED — ACTIVE G3 FREEZE REQUIRED"


class FinalTestError(RuntimeError):
    """Raised before test access or on frozen-evaluation contract violations."""


@dataclass(frozen=True)
class PretestAuditItem:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class PretestAuditReport:
    report_version: str
    overall_status: str
    items: Tuple[PretestAuditItem, ...]
    blockers: Tuple[str, ...]
    test_loader_calls: int = 0
    test_data_accessed: bool = False

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ICULogPredictionRecord:
    stay_id: object
    prediction_time: str
    target_hours: float
    predicted_log1p_hours: float
    eligible: bool


@dataclass(frozen=True)
class FinalPredictionBundle:
    recovery24: Tuple[PredictionRecord, ...]
    recovery48: Tuple[PredictionRecord, ...]
    icu_time_log: Tuple[ICULogPredictionRecord, ...]
    support_raw: Tuple[PredictionRecord, ...]


@dataclass(frozen=True)
class ProcessedFinalEvaluation:
    recovery: Mapping[str, object]
    icu_time: object
    support_raw: object
    support_calibrated: object
    icu_time_hours_records: Tuple[PredictionRecord, ...]
    support_calibrated_records: Tuple[PredictionRecord, ...]
    raw_support_probabilities: Tuple[float, ...]
    calibrated_support_probabilities: Tuple[float, ...]
    fixed_support_threshold: float


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalTestError("unreadable JSON artifact: " + str(path)) from error
    if not isinstance(value, Mapping):
        raise FinalTestError("JSON artifact must contain an object")
    return value


def load_final_test_config(path: Path) -> Mapping[str, object]:
    return _load_json(path)


def _safe_ref(root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise FinalTestError("Phase-20 references must be repository-relative")
    path = (root / reference).resolve()
    if root.resolve() not in path.parents:
        raise FinalTestError("Phase-20 reference escapes repository root")
    return path


def _validate_bootstrap_config(root: Path, config: Mapping[str, object]) -> str:
    path = _safe_ref(root, config.get("bootstrap_config_ref"))
    expected = config.get("bootstrap_config_sha256")
    if not isinstance(expected, str) or not is_sha256(expected):
        raise FinalTestError("bootstrap configuration hash is malformed")
    if not path.is_file() or sha256_file(path) != expected:
        raise FinalTestError("bootstrap configuration hash mismatch")
    bootstrap = _load_json(path)
    count = bootstrap.get("n_bootstrap")
    seed = bootstrap.get("seed")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise FinalTestError("BLOCKED — BOOTSTRAP REPLICATE COUNT REQUIRED")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FinalTestError("BLOCKED — BOOTSTRAP SEED REQUIRED")
    if bootstrap.get("final_test_access_allowed") is not True:
        raise FinalTestError("bootstrap configuration is not frozen for final test")
    return expected


def pretest_audit(
    root: Path,
    *,
    config_path: Optional[Path] = None,
    marker_path: Optional[Path] = None,
    expected_scope: Optional[str] = None,
) -> PretestAuditReport:
    """Audit frozen inputs without opening or receiving a test artifact."""

    root = root.resolve()
    config = _load_json(config_path or root / "configs/final_test_v1.json")
    items = []

    def check(name, operation):
        try:
            detail = operation()
            items.append(PretestAuditItem(name, "PASS", str(detail or "validated")))
        except Exception as error:
            items.append(PretestAuditItem(name, "BLOCKED", str(error)))

    if config.get("config_version") != FINAL_TEST_CONFIG_VERSION:
        check("phase20_config", lambda: (_ for _ in ()).throw(
            FinalTestError("Phase-20 configuration version mismatch")
        ))
    else:
        check("phase20_config", lambda: FINAL_TEST_CONFIG_VERSION)

    scope = expected_scope or str(config.get("required_g3_scope"))
    marker = marker_path or _safe_ref(root, config.get("g3_marker_ref"))

    def active_g3():
        try:
            validated = validate_g3_marker(marker, root, expected_scope=scope)
        except Exception as error:
            raise FinalTestError(BLOCKED_G3 + ": " + str(error)) from error
        return validated["marker_sha256"]

    check("active_g3", active_g3)
    check("bootstrap", lambda: _validate_bootstrap_config(root, config))

    def retry_policy():
        policy = config.get("retry_policy")
        if not isinstance(policy, Mapping) or policy.get("approved_before_test") is not True:
            raise FinalTestError("BLOCKED — FINAL TEST RETRY POLICY REQUIRED")
        if not policy.get("policy_version"):
            raise FinalTestError("BLOCKED — FINAL TEST RETRY POLICY REQUIRED")
        return policy["policy_version"]

    check("retry_policy", retry_policy)

    def evaluation_plan():
        plan = config.get("final_evaluation_plan")
        if not isinstance(plan, Mapping) or plan.get("status") != "FROZEN_BEFORE_TEST":
            raise FinalTestError("BLOCKED — FROZEN FINAL EVALUATION PLAN REQUIRED")
        required = (
            "evaluate_naive",
            "evaluate_nonselected_families",
            "evaluate_lstm",
            "icu_error_percentiles",
        )
        if any(plan.get(field) is None for field in required):
            raise FinalTestError("final evaluation exposure plan is incomplete")
        return "selected/nonselected exposure fixed"

    check("final_evaluation_plan", evaluation_plan)

    def slices():
        payload = _load_json(_safe_ref(root, config.get("error_analysis_spec_ref")))
        if payload.get("status") != "FROZEN_FOR_FINAL_TEST":
            raise FinalTestError("BLOCKED — FROZEN FINAL ERROR-ANALYSIS SLICES REQUIRED")
        return payload.get("spec_version")

    check("error_analysis_slices", slices)

    def execution_entrypoints():
        values = (
            config.get("real_test_loader_entrypoint"),
            config.get("real_test_evaluator_entrypoint"),
        )
        if any(
            not isinstance(value, str)
            or value.count(":") != 1
            or not all(part for part in value.split(":"))
            for value in values
        ):
            raise FinalTestError(
                "BLOCKED — REAL FINAL-TEST LOADER/EVALUATOR ENTRYPOINTS REQUIRED"
            )
        return "loader and evaluator entrypoints frozen"

    check("execution_entrypoints", execution_entrypoints)

    def g4_schema():
        schema = config.get("g4_schema")
        if (
            not isinstance(schema, Mapping)
            or schema.get("status") != "FROZEN_BEFORE_TEST"
            or not schema.get("schema_version")
        ):
            raise FinalTestError(
                "UNLOCKED ENGINEERING PARAMETER — G4 EVALUATION FREEZE SCHEMA"
            )
        return schema["schema_version"]

    check("g4_schema", g4_schema)

    def lineage():
        summary = audit_registry_lineage(root)
        critical = (
            "duplicate_run_ids",
            "scientific_orphan_artifacts",
            "missing_artifacts",
            "hash_mismatches",
            "broken_parent_runs",
            "broken_artifact_parents",
            "unknown_artifact_producers",
            "run_reference_mismatches",
            "incompatible_dependencies",
        )
        if any(summary[name] for name in critical):
            raise FinalTestError("registry lineage has critical failures")
        return "critical lineage clean"

    check("registry_lineage", lineage)

    def access_state():
        state = load_access_state(root / ACCESS_STATE_RELATIVE_PATH)
        if state["state"] != "AUTHORIZED_NOT_RUN":
            raise FinalTestError("final-test access is not authorized for exactly one run")
        return state["state"]

    check("test_access_state", access_state)
    blockers = tuple(
        item.name + ": " + item.detail for item in items if item.status != "PASS"
    )
    return PretestAuditReport(
        report_version=PRETEST_AUDIT_VERSION,
        overall_status="PASS" if not blockers else "BLOCKED",
        items=tuple(items),
        blockers=blockers,
    )


def write_pretest_audit(
    root: Path, report: PretestAuditReport, *, config_path: Optional[Path] = None
) -> Path:
    config = _load_json(config_path or root / "configs/final_test_v1.json")
    path = _safe_ref(root.resolve(), config.get("pretest_audit_ref"))
    payload = report.as_dict()
    payload["report_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def register_pretest_snapshot(root: Path, report_path: Path) -> Path:
    """Register immutable blocked/pass evidence without claiming test execution."""

    root = root.resolve()
    digest = sha256_file(report_path)
    relative = Path("artifacts/governance/history") / (
        "phase20_pretest_audit_v1_" + digest[:16] + ".json"
    )
    snapshot = root / relative
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.is_file():
        if sha256_file(snapshot) != digest:
            raise FinalTestError("pre-test audit snapshot path collision")
    else:
        shutil.copy2(report_path, snapshot)
    artifact_id = "phase20:pretest:" + digest[:16]
    records = read_artifact_index(root / "experiments/artifacts.csv")
    existing = [item for item in records if item.artifact_id == artifact_id]
    if existing:
        if (
            existing[0].artifact_path != str(relative)
            or existing[0].artifact_sha256 != digest
        ):
            raise FinalTestError("registered pre-test audit snapshot conflicts")
        return snapshot
    register_artifact(
        root / "experiments/artifacts.csv",
        ArtifactRecord(
            artifact_id=artifact_id,
            artifact_path=str(relative),
            artifact_type="final_test_preflight_report",
            artifact_version=PRETEST_AUDIT_VERSION,
            artifact_sha256=digest,
            producing_run_id="",
            config_hash=sha256_file(root / "configs/final_test_v1.json"),
            generating_script="src/experiments/final_test_cli.py",
            run_type="development",
            status="registered",
        ),
        read_run_registry(root / "experiments/registry.csv"),
        root,
    )
    return snapshot


def process_icu_time_predictions(
    records: Sequence[ICULogPredictionRecord],
) -> Tuple[PredictionRecord, ...]:
    if not records:
        raise FinalTestError("ICU-time final evaluation requires predictions")
    raw = torch.tensor(
        [item.predicted_log1p_hours for item in records], dtype=torch.float64
    )
    hours = remaining_icu_hours_from_log_prediction(raw).tolist()
    return tuple(
        PredictionRecord(
            stay_id=item.stay_id,
            prediction_time=item.prediction_time,
            target=item.target_hours,
            prediction=float(prediction),
            eligible=item.eligible,
        )
        for item, prediction in zip(records, hours)
    )


def apply_frozen_support_postprocessing(
    records: Sequence[PredictionRecord], calibrator, *, threshold: float
) -> Tuple[Tuple[PredictionRecord, ...], Tuple[bool, ...]]:
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise FinalTestError("frozen support threshold must be within [0,1]")
    raw = tuple(float(item.prediction) for item in records)
    calibrated = apply_support_calibrator(calibrator, raw)
    converted = tuple(
        PredictionRecord(
            stay_id=item.stay_id,
            prediction_time=item.prediction_time,
            target=item.target,
            prediction=probability,
            eligible=item.eligible,
        )
        for item, probability in zip(records, calibrated)
    )
    return converted, tuple(value >= threshold_value for value in calibrated)


def evaluate_frozen_predictions(
    bundle: FinalPredictionBundle,
    *,
    support_calibrator,
    support_threshold: float,
    threshold_identifier: str,
    icu_percentile_levels: Sequence[float] = (),
) -> ProcessedFinalEvaluation:
    """Reuse frozen point estimators and transform-only postprocessing."""

    icu_records = process_icu_time_predictions(bundle.icu_time_log)
    calibrated, _ = apply_frozen_support_postprocessing(
        bundle.support_raw, support_calibrator, threshold=support_threshold
    )
    return ProcessedFinalEvaluation(
        recovery=evaluate_recovery(bundle.recovery24, bundle.recovery48),
        icu_time=evaluate_icu_time(
            icu_records, percentile_levels=icu_percentile_levels
        ),
        support_raw=evaluate_organ_support(
            bundle.support_raw, probability_type="raw"
        ),
        support_calibrated=evaluate_organ_support(
            calibrated,
            fixed_threshold=support_threshold,
            threshold_identifier=threshold_identifier,
            probability_type="calibrated",
        ),
        icu_time_hours_records=icu_records,
        support_calibrated_records=calibrated,
        raw_support_probabilities=tuple(
            float(item.prediction) for item in bundle.support_raw
        ),
        calibrated_support_probabilities=tuple(
            float(item.prediction) for item in calibrated
        ),
        fixed_support_threshold=float(support_threshold),
    )


def bootstrap_frozen_predictions(
    bundle: FinalPredictionBundle,
    processed: ProcessedFinalEvaluation,
    *,
    n_bootstrap: int,
    seed: int,
    threshold_identifier: str,
    icu_percentile_levels: Sequence[float] = (),
):
    """Reuse Phase-10 stay-copy bootstrap for each frozen point estimator."""

    return {
        "recovery": bootstrap_recovery(
            bundle.recovery24,
            bundle.recovery48,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "icu_stay_time": bootstrap_icu_time(
            processed.icu_time_hours_records,
            n_bootstrap=n_bootstrap,
            seed=seed,
            percentile_levels=icu_percentile_levels,
        ),
        "organ_support": bootstrap_organ_support(
            processed.support_calibrated_records,
            n_bootstrap=n_bootstrap,
            seed=seed,
            fixed_threshold=processed.fixed_support_threshold,
            threshold_identifier=threshold_identifier,
            probability_type="calibrated",
        ),
    }


def evaluate_test_complete_component(
    recovery24: Sequence[CompleteComponentRecord],
    recovery48: Sequence[CompleteComponentRecord],
):
    return evaluate_complete_component_sensitivity(
        recovery24, recovery48, expected_partition="test"
    )


def evaluate_prespecified_test_error_analysis(records, specification, binding):
    """Use the Phase-15 implementation with its explicit test authorization flag."""

    return evaluate_error_analysis(
        records, specification, binding, allow_test=True
    )


def validate_final_result_traceability(
    root: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    root = root.resolve()
    artifacts = {
        item.artifact_id: item
        for item in read_artifact_index(root / "experiments/artifacts.csv")
    }
    required_hashes = {
        "prediction_artifact_id": "prediction_sha256",
        "model_artifact_id": "model_sha256",
        "manifest_artifact_id": "manifest_sha256",
        "preprocessor_artifact_id": "preprocessor_sha256",
    }
    required_values = (
        "task",
        "model_run_id",
        "feature_version",
        "label_version",
        "code_commit",
    )
    for row in rows:
        if any(
            not is_sha256(str(row.get(field, "")))
            for field in tuple(required_hashes.values())
            + ("g3_marker_sha256", "split_hash")
        ):
            raise FinalTestError("final result row has incomplete hash lineage")
        if any(not row.get(field) for field in required_values):
            raise FinalTestError("final result row has incomplete provenance")
        resolved = {}
        for identifier_field, hash_field in required_hashes.items():
            artifact = artifacts.get(row.get(identifier_field))
            if artifact is None or artifact.artifact_sha256 != row.get(hash_field):
                raise FinalTestError("final result row is not registry-traceable")
            resolved[identifier_field] = artifact
        prediction = resolved["prediction_artifact_id"]
        model = resolved["model_artifact_id"]
        manifest = resolved["manifest_artifact_id"]
        preprocessor = resolved["preprocessor_artifact_id"]
        if (
            prediction.artifact_type != "prediction"
            or prediction.partition != "test"
            or prediction.model_sha256 != model.artifact_sha256
            or model.artifact_id not in prediction.parents
            or manifest.artifact_type != "selected_model_manifest"
            or model.artifact_id not in manifest.parents
            or preprocessor.artifact_type != "preprocessor"
            or row.get("split_hash") != prediction.split_hash
            or row.get("split_hash") != model.split_hash
            or row.get("model_run_id") != model.producing_run_id
        ):
            raise FinalTestError("final result artifact graph is incompatible")
        marker = _safe_ref(root, row.get("g3_marker_ref"))
        if (
            not marker.is_file()
            or sha256_file(marker) != row.get("g3_marker_sha256")
        ):
            raise FinalTestError("final result G3 binding is stale")


def run_final_test(
    root: Path,
    *,
    guarded_runner: Callable[[Callable[[], object]], object],
    test_loader: Callable[[], object],
    evaluator: Callable[[object], object],
    config_path: Optional[Path] = None,
    marker_path: Optional[Path] = None,
    expected_scope: Optional[str] = None,
):
    """Audit first, then perform exactly one guarded load-and-evaluate callback."""

    report = pretest_audit(
        root,
        config_path=config_path,
        marker_path=marker_path,
        expected_scope=expected_scope,
    )
    if report.overall_status != "PASS":
        g3_blocked = any(item.startswith("active_g3:") for item in report.blockers)
        prefix = BLOCKED_G3 if g3_blocked else "BLOCKED — PRE-TEST FREEZE AUDIT FAILED"
        raise FinalTestError(prefix + ": " + "; ".join(report.blockers))
    return guarded_runner(lambda: evaluator(test_loader()))
