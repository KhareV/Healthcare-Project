"""Fail-closed Phase-19 validation orchestration and accounting.

This module coordinates existing training/evaluation components. It contains no
model implementation and never imports or invokes a final-test loader.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

from data.acceptance import DataAcceptanceError, require_accepted_real_data
from evaluation.selection_validation import EXPLANATION_METHOD
from experiments.audit import audit_registry_lineage
from experiments.lineage import (
    ArtifactRecord,
    read_artifact_index,
    read_run_registry,
    register_artifact,
)
from experiments.search_governance import (
    SEARCH_BUDGET,
    TASKS,
    canonical_sha256,
    load_search_space,
    validate_candidate_list,
)
from vedant_infra.g3 import ACCESS_STATE_RELATIVE_PATH, load_access_state
from vedant_infra.hashing import is_sha256, sha256_file


SUITE_VERSION = "full_real_validation_suite_v1"
PREFLIGHT_VERSION = "phase19_preflight_v1"
STAGES = (
    "preprocessing",
    "naive_baselines",
    "searches",
    "within_family_selection",
    "selected_gru_reruns",
    "lstm_sensitivity",
    "cross_family_selection",
    "support_calibration",
    "support_threshold",
    "selected_manifest",
    "validation_comparison",
    "validation_ablations",
    "validation_error_analysis",
    "validation_freeze",
)
TASKS_ORDERED = ("recovery", "icu_stay_time", "organ_support")
FAMILIES_ORDERED = ("xgboost", "gru")


class ValidationSuiteError(RuntimeError):
    """Raised before a validation action can violate frozen governance."""


@dataclass(frozen=True)
class PreflightItem:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    report_version: str
    suite_version: str
    overall_status: str
    items: Tuple[PreflightItem, ...]
    blockers: Tuple[str, ...]
    scientific_commands_executed: int = 0
    test_accessed: bool = False

    def as_dict(self):
        return asdict(self)


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationSuiteError("unreadable JSON artifact: " + str(path)) from error
    if not isinstance(value, Mapping):
        raise ValidationSuiteError("JSON artifact must contain an object")
    return value


def _safe_path(root: Path, ref: object) -> Path:
    if not isinstance(ref, str) or not ref or Path(ref).is_absolute():
        raise ValidationSuiteError("suite reference must be repository-relative")
    path = (root / ref).resolve()
    if root.resolve() not in path.parents:
        raise ValidationSuiteError("suite reference escapes repository root")
    return path


def validate_scientific_candidate_manifest(payload: Mapping[str, object]) -> Tuple[Mapping[str, object], ...]:
    """Validate a predeclared real candidate list without generating candidates."""

    if payload.get("manifest_version") != "scientific_candidate_manifest_v1":
        raise ValidationSuiteError("scientific candidate manifest version mismatch")
    if payload.get("run_type") != "scientific" or payload.get("test_accessed") is not False:
        raise ValidationSuiteError("scientific candidate manifest scope is invalid")
    if payload.get("task") not in TASKS_ORDERED or payload.get("family") not in FAMILIES_ORDERED:
        raise ValidationSuiteError("scientific candidate task/family is invalid")
    if payload.get("candidate_count") != SEARCH_BUDGET:
        raise ValidationSuiteError("scientific candidate count must be exactly 30")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValidationSuiteError("scientific candidate list is missing")
    try:
        validate_candidate_list(candidates)
    except Exception as error:
        raise ValidationSuiteError(str(error)) from error
    if payload.get("candidate_list_hash") != canonical_sha256(candidates):
        raise ValidationSuiteError("scientific candidate-list hash mismatch")
    if not is_sha256(str(payload.get("search_space_hash", ""))):
        raise ValidationSuiteError("scientific search-space hash is missing")
    for field in ("acceptance_report_sha256", "split_hash"):
        if not is_sha256(str(payload.get(field, ""))):
            raise ValidationSuiteError(field + " is missing or malformed")
    if payload.get("allowed_partitions") != ["train", "validation"]:
        raise ValidationSuiteError("scientific candidates must be train+validation only")
    if not payload.get("feature_version") or not payload.get("label_version"):
        raise ValidationSuiteError("scientific feature/label versions are required")
    commit = payload.get("code_commit")
    if (
        not isinstance(commit, str)
        or len(commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValidationSuiteError("scientific candidate manifest requires exact code commit")
    return tuple(candidates)


def _validate_configured_candidate_manifest(
    root: Path,
    config: Mapping[str, object],
    key: str,
) -> str:
    task, family = key.split(":")
    manifest = _load_json(
        _safe_path(root, config["scientific_candidate_manifests"][key])
    )
    validate_scientific_candidate_manifest(manifest)
    if manifest.get("task") != task or manifest.get("family") != family:
        raise ValidationSuiteError("configured candidate manifest identity mismatch")
    _, space_hash = load_search_space(
        _safe_path(root, config["search_spaces"][family])
    )
    if manifest.get("search_space_hash") != space_hash:
        raise ValidationSuiteError("candidate manifest search-space hash mismatch")
    if manifest.get("acceptance_report_sha256") != config.get(
        "acceptance_report_sha256"
    ):
        raise ValidationSuiteError("candidate manifest accepted-data hash mismatch")
    return str(manifest["candidate_list_hash"])


def audit_search_budget(
    manifest: Mapping[str, object], attempts: Sequence[Mapping[str, object]]
) -> Mapping[str, object]:
    """Separate fixed scientific candidate identities from retry attempts."""

    candidates = validate_scientific_candidate_manifest(manifest)
    declared = {str(item["candidate_id"]) for item in candidates}
    unknown = sorted(
        {
            str(item.get("candidate_id"))
            for item in attempts
            if item.get("candidate_id") not in declared
        }
    )
    if unknown:
        raise ValidationSuiteError(
            "candidate outside frozen exact-30 list: " + ",".join(unknown)
        )
    by_candidate = {identifier: [] for identifier in declared}
    for attempt in attempts:
        by_candidate[str(attempt["candidate_id"])].append(attempt)
    complete = sum(
        any(item.get("status") == "completed" for item in values)
        for values in by_candidate.values()
    )
    failed = sum(
        bool(values)
        and not any(item.get("status") == "completed" for item in values)
        and values[-1].get("status") == "failed"
        for values in by_candidate.values()
    )
    retries = sum(max(0, len(values) - 1) for values in by_candidate.values())
    return {
        "scientific_candidate_count": len(declared),
        "complete_candidates": complete,
        "failed_candidates": failed,
        "not_started_candidates": len(declared) - complete - failed,
        "execution_attempts": len(attempts),
        "retry_attempts": retries,
        "budget_valid": len(declared) == SEARCH_BUDGET,
    }


def resume_action(
    candidate: Mapping[str, object],
    attempts: Sequence[Mapping[str, object]],
    root: Path,
) -> str:
    """Return SKIP_COMPLETE, RETRY_SAME_CANDIDATE, or RUN_SAME_CANDIDATE."""

    relevant = [
        item for item in attempts
        if item.get("candidate_id") == candidate.get("candidate_id")
    ]
    complete = [item for item in relevant if item.get("status") == "completed"]
    if complete:
        latest = complete[-1]
        if latest.get("config_hash") != candidate.get("config_hash"):
            raise ValidationSuiteError("complete run config hash differs from frozen candidate")
        for ref_field, hash_field in (
            ("model_artifact_ref", "model_sha256"),
            ("validation_prediction_ref", "validation_prediction_sha256"),
        ):
            path = _safe_path(root, latest.get(ref_field))
            if not path.is_file() or sha256_file(path) != latest.get(hash_field):
                raise ValidationSuiteError("complete run artifact is missing or incompatible")
        return "SKIP_COMPLETE"
    if relevant:
        return "RETRY_SAME_CANDIDATE"
    return "RUN_SAME_CANDIDATE"


def require_stage_order(state: Mapping[str, object], requested_stage: str) -> None:
    if requested_stage not in STAGES:
        raise ValidationSuiteError("unknown validation stage")
    statuses = state.get("stages")
    if not isinstance(statuses, Mapping):
        raise ValidationSuiteError("validation state lacks stages")
    index = STAGES.index(requested_stage)
    incomplete = [stage for stage in STAGES[:index] if statuses.get(stage) != "COMPLETE"]
    if incomplete:
        raise ValidationSuiteError(
            "validation stage order violation; incomplete: " + ", ".join(incomplete)
        )


def audit_selected_gru_reruns(
    records: Sequence[Mapping[str, object]],
    selected_search_runs: Mapping[str, str],
) -> None:
    """Require one fixed-seed GRU rerun per task without a new candidate."""

    for task in TASKS_ORDERED:
        matching = [
            row for row in records
            if row.get("task") == task
            and row.get("model_family") == "gru"
            and row.get("run_role") == "selected_config_rerun"
            and row.get("status") == "completed"
        ]
        if len(matching) != 1:
            raise ValidationSuiteError(
                "exactly one selected-GRU rerun is required for " + task
            )
        if (
            matching[0].get("parent_run_id") != selected_search_runs.get(task)
            or matching[0].get("candidate_id")
        ):
            raise ValidationSuiteError(
                "selected-GRU rerun must preserve parent config without candidate 31"
            )


def audit_lstm_sensitivity(
    records: Sequence[Mapping[str, object]],
    selected_gru_reruns: Mapping[str, str],
) -> None:
    """Require exactly one fixed, non-search LSTM sensitivity per task."""

    for task in TASKS_ORDERED:
        matching = [
            row for row in records
            if row.get("task") == task
            and row.get("model_family") == "lstm"
            and row.get("run_role") == "sensitivity"
            and row.get("status") == "completed"
        ]
        if len(matching) != 1:
            raise ValidationSuiteError(
                "exactly one LSTM sensitivity run is required for " + task
            )
        if (
            matching[0].get("parent_run_id")
            != selected_gru_reruns.get(task)
            or matching[0].get("candidate_id")
        ):
            raise ValidationSuiteError(
                "LSTM sensitivity must inherit the selected GRU rerun and remain outside search"
            )


def assert_validation_mutation_allowed(state: Mapping[str, object]) -> None:
    if state.get("status") == "VALIDATION_FROZEN":
        raise ValidationSuiteError("scientific reset required after validation freeze")


def validate_serving_manifest_roles(manifest: Mapping[str, object]) -> None:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASKS_ORDERED):
        raise ValidationSuiteError("selected manifest requires all three tasks")
    for task in TASKS_ORDERED:
        entry = tasks[task]
        if not isinstance(entry, Mapping) or entry.get("family") not in FAMILIES_ORDERED:
            raise ValidationSuiteError("only XGBoost or GRU may be selected")
        if entry.get("explanation_method") != EXPLANATION_METHOD[entry["family"]]:
            raise ValidationSuiteError("selected explanation routing is incompatible")


def validate_postprocessing_bindings(
    selected_support: Mapping[str, object],
    calibrator: Mapping[str, object],
    threshold: Mapping[str, object],
) -> None:
    if calibrator.get("selected_model_sha256") != selected_support.get("artifact_sha256"):
        raise ValidationSuiteError("calibrator hash binding does not match selected support model")
    if calibrator.get("partition") != "validation":
        raise ValidationSuiteError("support calibration must be validation-only")
    if threshold.get("calibrator_sha256") != calibrator.get("artifact_sha256"):
        raise ValidationSuiteError("threshold hash binding does not match calibrator")
    if threshold.get("criterion") != "validation_f1" or threshold.get("partition") != "validation":
        raise ValidationSuiteError("support threshold must be validation-F1 only")


def validate_comparison_traceability(
    rows: Sequence[Mapping[str, object]], root: Path
) -> None:
    runs = {row.get("run_id"): row for row in read_run_registry(root / "experiments/registry.csv")}
    artifacts = {item.artifact_path: item for item in read_artifact_index(root / "experiments/artifacts.csv")}
    for row in rows:
        run = runs.get(row.get("run_id"))
        artifact = artifacts.get(row.get("model_artifact_ref"))
        if not run or not artifact:
            raise ValidationSuiteError("validation comparison row is unregistered")
        if run.get("model_sha256") != row.get("model_sha256") or artifact.artifact_sha256 != row.get("model_sha256"):
            raise ValidationSuiteError("validation comparison row hash is untraceable")


def _acceptance_check(root: Path, config: Mapping[str, object]) -> str:
    ref = config.get("acceptance_report_ref")
    expected = config.get("acceptance_report_sha256")
    if not isinstance(expected, str) or not is_sha256(expected):
        raise ValidationSuiteError("acceptance report SHA-256 is malformed")
    artifacts = {item.artifact_id: item for item in read_artifact_index(root / "experiments/artifacts.csv")}
    record = artifacts.get(config.get("acceptance_artifact_id"))
    if not record or record.artifact_path != ref or record.artifact_sha256 != expected:
        raise ValidationSuiteError("Phase-16 registry does not bind the acceptance report")
    try:
        report = require_accepted_real_data(_safe_path(root, ref), expected)
    except DataAcceptanceError as error:
        raise ValidationSuiteError(str(error)) from error
    for item in report.get("input_inventory", ()):
        if not isinstance(item, Mapping):
            raise ValidationSuiteError("accepted-data inventory is malformed")
        if item.get("status") == "PROTECTED_NOT_OPENED":
            continue
        if item.get("status") != "HASH_VERIFIED":
            raise ValidationSuiteError("accepted-data dependency is not hash-verified")
        path = _safe_path(root, item.get("ref"))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValidationSuiteError("accepted upstream artifact changed after acceptance")
    return expected


def preflight_validation_suite(root: Path, config_path: Optional[Path] = None) -> PreflightReport:
    root = root.resolve()
    config_path = config_path or root / "configs/validation_suite_v1.json"
    config = _load_json(config_path)
    if config.get("suite_version") != SUITE_VERSION:
        raise ValidationSuiteError("validation suite config version mismatch")
    items = []

    def check(name, operation):
        try:
            detail = operation()
            items.append(PreflightItem(name, "PASS", str(detail or "validated")))
        except Exception as error:
            items.append(PreflightItem(name, "BLOCKED", str(error)))

    check("real_data_acceptance", lambda: _acceptance_check(root, config))

    def test_lock():
        state = load_access_state(root / ACCESS_STATE_RELATIVE_PATH)
        if state["state"] != "NEVER_OPENED":
            raise ValidationSuiteError("final-test state is not NEVER_OPENED")
        if (root / "artifacts/governance/g3_freeze.json").exists():
            raise ValidationSuiteError("active G3 marker must not pre-authorize Phase 19")
        if config.get("allowed_partitions") != ["train", "validation"] or config.get("test_partition_allowed") is not False:
            raise ValidationSuiteError("validation suite partition policy is unsafe")
        return "NEVER_OPENED; train+validation only"

    check("test_lock", test_lock)
    for family in FAMILIES_ORDERED:
        check(
            "search_space:" + family,
            lambda family=family: load_search_space(
                _safe_path(root, config["search_spaces"][family])
            )[1],
        )
    manifests = config.get("scientific_candidate_manifests", {})
    for task in TASKS_ORDERED:
        for family in FAMILIES_ORDERED:
            key = task + ":" + family
            check(
                "candidate_manifest:" + key,
                lambda key=key: _validate_configured_candidate_manifest(
                    root, config, key
                ),
            )

    def seed_policy():
        policy = config.get("seed_policy")
        if not isinstance(policy, Mapping) or policy.get("approved_for_real") is not True:
            raise ValidationSuiteError("BLOCKED — PROJECT SEED POLICY REQUIRED")
        if not policy.get("policy_version") or not isinstance(policy.get("seeds"), list):
            raise ValidationSuiteError("BLOCKED — PROJECT SEED POLICY REQUIRED")
        return policy["policy_version"]

    check("project_seed_policy", seed_policy)

    def threshold_policy():
        policy = config.get("threshold_policy")
        if not isinstance(policy, Mapping) or policy.get("approved_for_real") is not True:
            raise ValidationSuiteError("BLOCKED — REAL THRESHOLD POLICY REQUIRED")
        return "approved"

    check("threshold_policy", threshold_policy)
    check(
        "xgboost_entrypoint",
        lambda: config.get("xgboost_entrypoint")
        or (_ for _ in ()).throw(
            ValidationSuiteError(
                "BLOCKED — REQUIRED VALIDATION DEPENDENCY MISSING: Sanskruti XGBoost entrypoint"
            )
        ),
    )

    def lineage():
        summary = audit_registry_lineage(root)
        keys = (
            "missing_artifacts",
            "hash_mismatches",
            "unknown_artifact_producers",
            "incompatible_dependencies",
            "scientific_orphan_artifacts",
        )
        if any(summary[key] for key in keys):
            raise ValidationSuiteError("registry lineage has critical failures")
        return "critical lineage clean"

    check("registry_lineage", lineage)
    blockers = tuple(
        item.name + ": " + item.detail
        for item in items
        if item.status != "PASS"
    )
    return PreflightReport(
        report_version=PREFLIGHT_VERSION,
        suite_version=SUITE_VERSION,
        overall_status="PASS" if not blockers else "BLOCKED",
        items=tuple(items),
        blockers=blockers,
    )


def write_preflight_report(root: Path, report: PreflightReport) -> Path:
    config = _load_json(root / "configs/validation_suite_v1.json")
    path = _safe_path(root.resolve(), config["preflight_report"])
    payload = report.as_dict()
    payload["report_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def register_preflight_snapshot(root: Path, report_path: Path) -> Path:
    """Register immutable blocked/pass preflight evidence without a run claim."""

    root = root.resolve()
    digest = sha256_file(report_path)
    relative = Path("artifacts/validation/history") / (
        "phase19_preflight_v1_" + digest[:16] + ".json"
    )
    snapshot = root / relative
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.is_file():
        if sha256_file(snapshot) != digest:
            raise ValidationSuiteError("preflight snapshot path collision")
    else:
        shutil.copy2(report_path, snapshot)
    artifact_id = "phase19:preflight:" + digest[:16]
    records = read_artifact_index(root / "experiments/artifacts.csv")
    existing = [item for item in records if item.artifact_id == artifact_id]
    if existing:
        if (
            existing[0].artifact_path != str(relative)
            or existing[0].artifact_sha256 != digest
        ):
            raise ValidationSuiteError("registered preflight snapshot conflicts")
        return snapshot
    register_artifact(
        root / "experiments/artifacts.csv",
        ArtifactRecord(
            artifact_id=artifact_id,
            artifact_path=str(relative),
            artifact_type="validation_preflight_report",
            artifact_version=PREFLIGHT_VERSION,
            artifact_sha256=digest,
            producing_run_id="",
            config_hash=sha256_file(root / "configs/validation_suite_v1.json"),
            generating_script="src/experiments/validation_suite_cli.py",
            run_type="development",
            status="registered",
        ),
        read_run_registry(root / "experiments/registry.csv"),
        root,
    )
    return snapshot


def run_validation_suite(
    root: Path,
    scientific_executor: Callable[[], object],
    *,
    config_path: Optional[Path] = None,
):
    """Call a configured scientific executor only after every preflight passes."""

    report = preflight_validation_suite(root, config_path)
    if report.overall_status != "PASS":
        raise ValidationSuiteError(
            "BLOCKED — REQUIRED VALIDATION DEPENDENCY MISSING: "
            + "; ".join(report.blockers)
        )
    return scientific_executor()
