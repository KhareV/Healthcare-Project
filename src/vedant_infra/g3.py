"""Fail-closed G3 audit, freeze marker, and access-state governance."""

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

from data.split import load_split_csv, verify_split_artifacts
from evaluation.selection_validation import EXPLANATION_METHOD
from experiments.audit import audit_registry_lineage
from experiments.lineage import read_artifact_index, read_run_registry
from experiments.search_governance import SEARCH_BUDGET, canonical_sha256
from vedant_infra.hashing import is_sha256, sha256_file


MARKER_VERSION = "g3_freeze_marker_v1"
AUDIT_VERSION = "g3_prerequisite_audit_v1"
ACCESS_STATE_VERSION = "final_test_access_state_v1"
REAL_MARKER_RELATIVE_PATH = Path("artifacts/governance/g3_freeze.json")
ACCESS_STATE_RELATIVE_PATH = Path("artifacts/governance/test_access_state.json")
TASKS = ("recovery", "icu_stay_time", "organ_support")
FAMILIES = ("xgboost", "gru")
ACCESS_STATES = {
    "NEVER_OPENED",
    "AUTHORIZED_NOT_RUN",
    "FINAL_RUN_COMPLETED",
    "INVALIDATED_BY_RESET",
}
BLOCKED_PREFIX = "BLOCKED — G3 PREREQUISITE MISSING"


class G3FreezeError(RuntimeError):
    """Raised when freeze creation or final-test access is unsafe."""


@dataclass(frozen=True)
class AuditItem:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class G3AuditReport:
    audit_version: str
    freeze_scope: str
    items: Tuple[AuditItem, ...]
    overall: str
    blockers: Tuple[str, ...]
    test_data_accessed: bool = False

    def as_dict(self):
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G3FreezeError("unreadable JSON artifact: " + str(path)) from error
    if not isinstance(value, Mapping):
        raise G3FreezeError("JSON artifact must be an object: " + str(path))
    return value


def _safe_ref(root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise G3FreezeError("governance artifact reference must be repository-relative")
    path = (root / reference).resolve()
    if root.resolve() not in path.parents:
        raise G3FreezeError("governance artifact reference escapes repository root")
    return path


def _verify_ref(root: Path, reference: object, expected_hash: object) -> Path:
    path = _safe_ref(root, reference)
    if not isinstance(expected_hash, str) or not is_sha256(expected_hash):
        raise G3FreezeError("governance dependency hash is malformed")
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise G3FreezeError(
            "governance dependency missing or hash mismatch: " + str(reference)
        )
    return path


def _reject_test_performance_fields(value, path="g3_input") -> None:
    forbidden = {
        "test_mae",
        "test_auprc",
        "test_prevalence",
        "test_metrics",
        "test_labels",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise G3FreezeError(
                    "test performance cannot influence G3: "
                    + path
                    + "."
                    + str(key)
                )
            _reject_test_performance_fields(child, path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_test_performance_fields(child, "{}[{}]".format(path, index))


def _manifest_state(root: Path, scope: str):
    manifest_ref = "artifacts/models/selected_models_v1.json"
    manifest_path = root / manifest_ref
    manifest = _load_json(manifest_path)
    _reject_test_performance_fields(manifest)
    expected_mode = "real" if scope == "real" else "synthetic"
    if manifest.get("manifest_version") != "selected_models_calibrated_v1":
        raise G3FreezeError("selected manifest version is not calibrated v1")
    if (
        manifest.get("selection_mode") != expected_mode
        or manifest.get("test_accessed") is not False
    ):
        raise G3FreezeError("selected manifest scope or test-nonuse state is invalid")
    governance_bindings = manifest.get("governance_bindings", {})
    if scope == "real":
        required = {
            "g1", "phase14_handoff", "stage2_family_selection", "lstm_sensitivity",
            "feature_schema", "split", "preprocessor", "recovery_target_scaler",
            "support_class_weight",
        }
        if not isinstance(governance_bindings, Mapping) or set(governance_bindings) != required:
            raise G3FreezeError("selected manifest governance bindings are incomplete")
        for name, binding in governance_bindings.items():
            if not isinstance(binding, Mapping):
                raise G3FreezeError("selected manifest governance binding is malformed: " + name)
            _verify_ref(root, binding.get("ref"), binding.get("sha256"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASKS):
        raise G3FreezeError("selected manifest requires all three tasks")
    runs = read_run_registry(root / "experiments/registry.csv")
    artifacts = read_artifact_index(root / "experiments/artifacts.csv")
    runs_by_id = {row.get("run_id", ""): row for row in runs}
    artifacts_by_path = {record.artifact_path: record for record in artifacts}
    task_bindings = {}
    split_hashes = set()
    for task in TASKS:
        entry = tasks[task]
        if not isinstance(entry, Mapping):
            raise G3FreezeError("selected task entry is malformed")
        family = entry.get("family")
        if (
            family not in FAMILIES
            or entry.get("explanation_method") != EXPLANATION_METHOD[family]
        ):
            raise G3FreezeError(
                "selected model family/explanation routing mismatch"
            )
        artifact = _verify_ref(
            root, entry.get("artifact_ref"), entry.get("artifact_sha256")
        )
        run_id = entry.get("run_id")
        run = runs_by_id.get(run_id)
        if (
            not run
            or run.get("task") != task
            or run.get("model_family") != family
        ):
            raise G3FreezeError(
                "selected run is absent or incompatible in registry"
            )
        if scope == "real" and run.get("run_type") != "scientific":
            raise G3FreezeError(
                "real selected model depends on non-scientific run"
            )
        if scope != "real" and run.get("run_type") not in (
            "synthetic",
            "sensitivity_smoke",
            "test",
        ):
            raise G3FreezeError("synthetic selected model run type is invalid")
        indexed = artifacts_by_path.get(str(entry.get("artifact_ref")))
        if (
            not indexed
            or indexed.artifact_sha256 != entry.get("artifact_sha256")
            or indexed.producing_run_id != run_id
        ):
            raise G3FreezeError(
                "selected checkpoint lacks compatible artifact lineage"
            )
        for field in ("split_hash", "feature_version", "label_version"):
            if not entry.get(field) or entry.get(field) != run.get(field):
                raise G3FreezeError(
                    "selected model {} mismatch".format(field)
                )
        split_hashes.add(entry["split_hash"])
        preprocessor = _verify_ref(
            root,
            entry.get("preprocessor_ref"),
            entry.get("preprocessor_sha256"),
        )
        if entry.get("preprocessing_fit_partition") != "train":
            raise G3FreezeError(
                "selected preprocessing lacks training-only fit provenance"
            )
        if (
            not isinstance(entry.get("selected_gru_run_id"), str)
            or not entry["selected_gru_run_id"]
        ):
            raise G3FreezeError(
                "selected GRU lineage is required for LSTM sensitivity"
            )
        task_bindings[task] = {
            "run_id": run_id,
            "family": family,
            "code_commit": run["code_commit"],
            "model_ref": str(entry["artifact_ref"]),
            "model_sha256": sha256_file(artifact),
            "feature_version": entry["feature_version"],
            "label_version": entry["label_version"],
            "preprocessor_ref": str(entry["preprocessor_ref"]),
            "preprocessor_sha256": sha256_file(preprocessor),
            "explanation_method": entry["explanation_method"],
            "selected_gru_run_id": entry["selected_gru_run_id"],
        }
    if len(split_hashes) != 1:
        raise G3FreezeError("selected tasks do not share one split hash")
    support = tasks["organ_support"]
    calibrator = support.get("calibrator")
    threshold = support.get("threshold")
    if not isinstance(calibrator, Mapping) or not isinstance(
        threshold, Mapping
    ):
        raise G3FreezeError("support calibrator and threshold are required")
    calibrator_path = _verify_ref(
        root,
        calibrator.get("artifact_ref"),
        calibrator.get("artifact_sha256"),
    )
    threshold_path = _verify_ref(
        root,
        threshold.get("artifact_ref"),
        threshold.get("artifact_sha256"),
    )
    if (
        calibrator.get("selected_model_sha256")
        != support.get("artifact_sha256")
        or calibrator.get("partition") != "validation"
    ):
        raise G3FreezeError(
            "support calibrator/model or validation provenance mismatch"
        )
    if (
        threshold.get("calibrator_sha256")
        != calibrator.get("artifact_sha256")
        or threshold.get("criterion") != "validation_f1"
        or threshold.get("partition") != "validation"
    ):
        raise G3FreezeError(
            "support threshold/calibrator or validation-F1 provenance mismatch"
        )
    cal_index = artifacts_by_path.get(str(calibrator["artifact_ref"]))
    threshold_index = artifacts_by_path.get(str(threshold["artifact_ref"]))
    if not cal_index or not threshold_index:
        raise G3FreezeError(
            "support post-processing lacks artifact lineage"
        )
    if cal_index.model_sha256 != support.get("artifact_sha256"):
        raise G3FreezeError(
            "registered calibrator is bound to another model"
        )
    if (
        threshold_index.calibrator_sha256
        != calibrator.get("artifact_sha256")
    ):
        raise G3FreezeError(
            "registered threshold is bound to another calibrator"
        )
    content = dict(manifest)
    supplied_hash = content.pop("manifest_sha256", None)
    if supplied_hash != canonical_sha256(content):
        raise G3FreezeError("selected manifest canonical hash mismatch")
    return {
        "manifest_ref": manifest_ref,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_content_hash": supplied_hash,
        "tasks": task_bindings,
        "split_hash": next(iter(split_hashes)),
        "support_calibrator": {
            "ref": str(calibrator["artifact_ref"]),
            "sha256": sha256_file(calibrator_path),
        },
        "support_threshold": {
            "ref": str(threshold["artifact_ref"]),
            "sha256": sha256_file(threshold_path),
            "criterion": "validation_f1",
        },
        "governance_bindings": dict(governance_bindings),
        "code_commit": manifest.get("code_commit"),
        "split_ref": governance_bindings.get("split", {}).get("ref", "artifacts/splits/split_v1.csv"),
    }


def _audit_searches(runs, scope: str):
    allowed = "scientific" if scope == "real" else ("synthetic", "test")
    evidence = {}
    for task in TASKS:
        for family in FAMILIES:
            authoritative_version = None
            if scope == "real":
                authoritative_version = (
                    "synthetic_xgb_phase12_validation_search_v1"
                    if family == "xgboost"
                    else "vedant_final_gru_validation_search_v2"
                )
            matching = [
                row
                for row in runs
                if row.get("task") == task
                and row.get("model_family") == family
                and row.get("status") == "completed"
                and row.get("candidate_id")
                and (authoritative_version is None or row.get("search_version") == authoritative_version)
                and (
                    row.get("run_type") == allowed
                    if isinstance(allowed, str)
                    else row.get("run_type") in allowed
                )
            ]
            candidate_ids = {row.get("candidate_id") for row in matching}
            if len(candidate_ids) != SEARCH_BUDGET:
                raise G3FreezeError(
                    "{} {} search lacks exact-30 completed candidates".format(
                        task, family
                    )
                )
            config_hashes = {row.get("config_hash") for row in matching}
            if len(config_hashes) != SEARCH_BUDGET:
                raise G3FreezeError(
                    "{} {} search lacks 30 unique configurations".format(
                        task, family
                    )
                )
            if any(
                not row.get("search_space_hash")
                or not row.get("candidate_list_hash")
                for row in matching
            ):
                raise G3FreezeError("search hashes are incomplete")
            space_hashes = {row["search_space_hash"] for row in matching}
            list_hashes = {row["candidate_list_hash"] for row in matching}
            versions = {row.get("search_version") for row in matching}
            if (
                len(space_hashes) != 1
                or len(list_hashes) != 1
                or len(versions) != 1
                or not next(iter(versions))
            ):
                raise G3FreezeError(
                    "{} {} search identity is not frozen".format(task, family)
                )
            evidence[task + ":" + family] = {
                "search_version": next(iter(versions)),
                "search_space_hash": next(iter(space_hashes)),
                "candidate_list_hash": next(iter(list_hashes)),
                "scientific_candidate_count": SEARCH_BUDGET,
            }
    return evidence


def _audit_lstm(runs, manifest_state, scope: str):
    allowed = ("scientific_sensitivity",) if scope == "real" else ("synthetic", "sensitivity_smoke", "test")
    evidence = {}
    for task in TASKS:
        matching = [
            row
            for row in runs
            if row.get("task") == task
            and row.get("model_family") == "lstm"
            and row.get("status") == "completed"
            and row.get("run_type") in allowed
        ]
        if len(matching) != 1:
            raise G3FreezeError(
                "exactly one fixed LSTM sensitivity run is required for "
                + task
            )
        if (
            matching[0].get("parent_run_id")
            != manifest_state["tasks"][task]["selected_gru_run_id"]
        ):
            raise G3FreezeError(
                "LSTM sensitivity does not reference selected GRU settings"
            )
        if scope == "real":
            try:
                notes = json.loads(matching[0].get("notes") or "{}")
            except json.JSONDecodeError as error:
                raise G3FreezeError("LSTM sensitivity governance notes are invalid") from error
            if (matching[0].get("search_version") != "vedant_finalization_stage2_v1"
                    or notes.get("participates_in_search") is not False):
                raise G3FreezeError("LSTM sensitivity must not participate in hyperparameter search")
        elif matching[0].get("candidate_id"):
            raise G3FreezeError("LSTM sensitivity must not participate in hyperparameter search")
        evidence[task] = {
            "run_id": matching[0]["run_id"],
            "parent_selected_gru_run_id": matching[0]["parent_run_id"],
            "model_ref": matching[0]["model_artifact_ref"],
            "model_sha256": matching[0]["model_sha256"],
            "code_commit": matching[0]["code_commit"],
        }
    return evidence


def load_access_state(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {
            "state_version": ACCESS_STATE_VERSION,
            "state": "NEVER_OPENED",
            "history": [],
        }
    payload = _load_json(path)
    if (
        payload.get("state_version") != ACCESS_STATE_VERSION
        or payload.get("state") not in ACCESS_STATES
    ):
        raise G3FreezeError("test-access state is malformed")
    if not isinstance(payload.get("history"), list):
        raise G3FreezeError("test-access history is malformed")
    return payload


def _write_json_atomic(
    path: Path, payload: Mapping[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _audit_explanations(root: Path, scope: str) -> None:
    try:
        adapters = _load_json(
            root / "artifacts/governance/explanation_adapters_review.json"
        )
    except G3FreezeError as error:
        raise G3FreezeError(
            "BLOCKED — PULKIT EXPLANATION ADAPTER G3 DEPENDENCY"
        ) from error
    if (
        adapters.get("scope") != scope
        or adapters.get("tree_shap_implemented") is not True
        or adapters.get("integrated_gradients_implemented") is not True
        or adapters.get("reviewed") is not True
    ):
        raise G3FreezeError(
            "BLOCKED — PULKIT EXPLANATION ADAPTER G3 DEPENDENCY"
        )


def _audit_reviews(root: Path, scope: str) -> None:
    reviews = _load_json(
        root / "artifacts/governance/g3_review_signoffs.json"
    )
    if reviews.get("scope") != scope:
        raise G3FreezeError("G3 approval scope mismatch")
    if scope == "real":
        if (reviews.get("decision_authority") != "USER_DELEGATED_AI_PROJECT_DECISION"
                or reviews.get("authorization") != "STAGE3_G3_FREEZE_REQUESTED"
                or reviews.get("human_member_signoff_claimed") is not False):
            raise G3FreezeError("explicit delegated Stage-3 G3 authorization is required")
    elif set(reviews.get("approved_reviewers", ())) != {"sanskruti", "vedant", "pulkit"}:
        raise G3FreezeError("all-member G3 signoff is required")


def audit_g3(
    root: Path,
    *,
    scope: str = "real",
    allow_authorized_state: bool = False,
) -> G3AuditReport:
    if scope not in ("real", "synthetic_test"):
        raise G3FreezeError(
            "freeze scope must be real or synthetic_test"
        )
    root = root.resolve()
    items = []

    def check(name, operation):
        try:
            detail = operation()
            items.append(
                AuditItem(name, "PASS", str(detail or "validated"))
            )
        except Exception as error:
            items.append(AuditItem(name, "BLOCKED", str(error)))

    if scope == "real" and (root / "artifacts/splits/synthetic_split_v2.csv").is_file():
        split_path = root / "artifacts/splits/synthetic_split_v2.csv"
        split_metadata = root / "artifacts/splits/synthetic_split_v2.metadata.json"
    else:
        split_path = root / "artifacts/splits/split_v1.csv"
        split_metadata = root / "artifacts/splits/split_v1.metadata.json"

    def split_check():
        if scope == "real":
            if split_path.name == "synthetic_split_v2.csv":
                metadata = _load_json(split_metadata)
                if (metadata.get("artifact_version") != "synthetic_subject_split_v2"
                        or metadata.get("status") != "FROZEN_SYNTHETIC_AUTHORIZED"
                        or metadata.get("split_path") != "artifacts/splits/synthetic_split_v2.csv"
                        or metadata.get("split_sha256") != sha256_file(split_path)
                        or metadata.get("labels_or_performance_used") is not False):
                    raise G3FreezeError("accepted synthetic split metadata is invalid")
                with split_path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if tuple(reader.fieldnames or ()) != ("subject_id", "split", "clone_fingerprint_sha256"):
                        raise G3FreezeError("accepted synthetic split schema mismatch")
                    rows = list(reader)
                if (not rows or len({row["subject_id"] for row in rows}) != len(rows)
                        or any(row["split"] not in ("train", "validation", "test") for row in rows)):
                    raise G3FreezeError("accepted synthetic split subject isolation failed")
                declared = metadata.get("subject_counts", {})
                actual = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
                if declared != actual or metadata.get("total_subjects") != len(rows):
                    raise G3FreezeError("accepted synthetic split count mismatch")
                fingerprint_splits = {}
                for row in rows:
                    fingerprint_splits.setdefault(row["clone_fingerprint_sha256"], set()).add(row["split"])
                if any(len(values) != 1 for values in fingerprint_splits.values()):
                    raise G3FreezeError("accepted synthetic split clone isolation failed")
                return metadata["split_sha256"]
            metadata = verify_split_artifacts(
                split_path, split_metadata
            )
            return metadata.get("split_file_sha256")
        load_split_csv(split_path)
        return sha256_file(split_path)

    state_holder = {}
    check("split", split_check)

    def manifest_check():
        state_holder.update(_manifest_state(root, scope))
        if sha256_file(split_path) != state_holder["split_hash"]:
            raise G3FreezeError(
                "selected manifest split hash differs from split artifact"
            )
        return state_holder["manifest_sha256"]

    check("selected_models_manifest", manifest_check)
    runs = ()
    try:
        runs = read_run_registry(root / "experiments/registry.csv")
    except Exception:
        pass
    check(
        "validation_searches",
        lambda: _audit_searches(runs, scope),
    )

    def lstm_check():
        if not state_holder:
            raise G3FreezeError(
                "selected manifest prerequisite failed"
            )
        return _audit_lstm(runs, state_holder, scope)

    check("lstm_sensitivity", lstm_check)

    def support_check(key):
        if not state_holder:
            raise G3FreezeError(
                "selected manifest prerequisite failed"
            )
        return state_holder[key]

    check(
        "support_calibrator",
        lambda: support_check("support_calibrator"),
    )
    check(
        "support_threshold",
        lambda: support_check("support_threshold"),
    )

    def registry_check():
        summary = audit_registry_lineage(root)
        critical = sum(
            len(summary[name])
            if not isinstance(summary[name], int)
            else summary[name]
            for name in (
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
        )
        if critical:
            raise G3FreezeError(
                "critical Phase-16 registry audit failures"
            )
        return (
            "critical lineage clean; development-history issues "
            "are noncritical"
        )

    check("registry_lineage", registry_check)
    check(
        "explanation_adapters",
        lambda: _audit_explanations(root, scope),
    )
    check(
        "review_signoffs",
        lambda: _audit_reviews(root, scope),
    )

    def nonuse_check():
        state = load_access_state(
            root / ACCESS_STATE_RELATIVE_PATH
        )
        allowed_states = {
            "NEVER_OPENED",
            "INVALIDATED_BY_RESET",
        }
        if allow_authorized_state:
            allowed_states.add("AUTHORIZED_NOT_RUN")
        if state["state"] not in allowed_states:
            raise G3FreezeError(
                "test was already authorized or consumed"
            )
        if scope == "real" and any(
            row.get("run_type") == "test" for row in runs
        ):
            raise G3FreezeError(
                "test-labelled development run exists"
            )
        return (
            "state never opened; manifest/search/calibration inputs "
            "are train/validation only"
        )

    check("test_nonuse", nonuse_check)
    blockers = tuple(
        item.name + ": " + item.detail
        for item in items
        if item.status != "PASS"
    )
    return G3AuditReport(
        audit_version=AUDIT_VERSION,
        freeze_scope=scope,
        items=tuple(items),
        overall="PASS" if not blockers else "BLOCKED",
        blockers=blockers,
    )


def write_audit_report(
    root: Path, report: G3AuditReport
) -> Path:
    path = root / "artifacts/governance/g3_audit_report.json"
    payload = report.as_dict()
    payload.update(
        {"authorization": False, "created_at_utc": utc_now()}
    )
    _write_json_atomic(path, payload)
    return path


def _dependency_bindings(
    root: Path, state: Mapping[str, object]
) -> Tuple[Mapping[str, str], ...]:
    refs = [
        (state["manifest_ref"], state["manifest_sha256"]),
        (state.get("split_ref", "artifacts/splits/split_v1.csv"), state["split_hash"]),
        (
            "experiments/registry.csv",
            sha256_file(root / "experiments/registry.csv"),
        ),
        (
            "experiments/artifacts.csv",
            sha256_file(root / "experiments/artifacts.csv"),
        ),
        (
            state["support_calibrator"]["ref"],
            state["support_calibrator"]["sha256"],
        ),
        (
            state["support_threshold"]["ref"],
            state["support_threshold"]["sha256"],
        ),
        (
            "artifacts/governance/explanation_adapters_review.json",
            sha256_file(
                root
                / "artifacts/governance/explanation_adapters_review.json"
            ),
        ),
        (
            "artifacts/governance/g3_review_signoffs.json",
            sha256_file(root / "artifacts/governance/g3_review_signoffs.json"),
        ),
    ]
    for binding in state.get("governance_bindings", {}).values():
        refs.append((binding["ref"], binding["sha256"]))
    split_metadata = root / (state.get("split_ref", "artifacts/splits/split_v1.csv") + ".metadata.json")
    if not split_metadata.is_file() and state.get("split_ref", "").endswith(".csv"):
        split_metadata = root / state["split_ref"].replace(".csv", ".metadata.json")
    if split_metadata.is_file():
        refs.append(
            (
                str(split_metadata.relative_to(root)),
                sha256_file(split_metadata),
            )
        )
    for task in TASKS:
        binding = state["tasks"][task]
        refs.extend(
            (
                (binding["model_ref"], binding["model_sha256"]),
                (
                    binding["preprocessor_ref"],
                    binding["preprocessor_sha256"],
                ),
            )
        )
    unique = {}
    for reference, digest in refs:
        unique[reference] = digest
    return tuple(
        {"ref": ref, "sha256": digest}
        for ref, digest in sorted(unique.items())
    )


def freeze_g3(
    root: Path,
    *,
    marker_path: Optional[Path] = None,
    scope: str = "real",
    created_at_utc: Optional[str] = None,
) -> Path:
    root = root.resolve()
    target = (
        marker_path or (root / REAL_MARKER_RELATIVE_PATH)
    ).resolve()
    if target.exists():
        raise G3FreezeError(
            "active marker path already exists; overwrite is forbidden"
        )
    if (
        scope == "synthetic_test"
        and target == (root / REAL_MARKER_RELATIVE_PATH).resolve()
    ):
        raise G3FreezeError(
            "synthetic marker cannot be written to real G3 path"
        )
    report = audit_g3(root, scope=scope)
    if report.overall != "PASS":
        raise G3FreezeError(
            BLOCKED_PREFIX + ": " + "; ".join(report.blockers)
        )
    state = _manifest_state(root, scope)
    runs = read_run_registry(root / "experiments/registry.csv")
    search_evidence = _audit_searches(runs, scope)
    lstm_evidence = _audit_lstm(runs, state, scope)
    reviews = _load_json(
        root / "artifacts/governance/g3_review_signoffs.json"
    )
    marker = {
        "marker_version": MARKER_VERSION,
        "gate": "G3",
        "freeze_status": "ACTIVE",
        "freeze_scope": scope,
        "created_at_utc": created_at_utc or utc_now(),
        "split_hash": state["split_hash"],
        "selected_models_manifest": state["manifest_ref"],
        "selected_models_sha256": state["manifest_sha256"],
        "tasks": state["tasks"],
        "support_calibrator": state["support_calibrator"],
        "support_threshold": state["support_threshold"],
        "validation_searches": search_evidence,
        "lstm_sensitivity_runs": lstm_evidence,
        "dependencies": _dependency_bindings(root, state),
        "reviewers": reviews.get("approved_reviewers", reviews.get("authorized_roles", ["project_owner"])),
        "test_accessed_before_freeze": False,
        "test_accessed": False,
        "status": "G3_ACTIVE",
        "all_model_choices_frozen": True,
        "calibration_frozen": True,
        "threshold_frozen": True,
        "explanation_routing_frozen": True,
        "code_commit": state.get("code_commit"),
        "governance_bindings": state.get("governance_bindings", {}),
        "audit_version": AUDIT_VERSION,
    }
    marker["marker_sha256"] = canonical_sha256(marker)
    _write_json_atomic(target, marker)
    access_path = root / ACCESS_STATE_RELATIVE_PATH
    access = dict(load_access_state(access_path))
    if access["state"] not in (
        "NEVER_OPENED",
        "INVALIDATED_BY_RESET",
    ):
        target.unlink(missing_ok=True)
        raise G3FreezeError(
            "test access state does not permit a new freeze"
        )
    history = list(access["history"])
    history.append(
        {
            "event": "G3_FREEZE_CREATED",
            "at_utc": marker["created_at_utc"],
            "marker_ref": str(target.relative_to(root)),
            "marker_sha256": sha256_file(target),
            "scope": scope,
        }
    )
    _write_json_atomic(
        access_path,
        {
            "state_version": ACCESS_STATE_VERSION,
            "state": "AUTHORIZED_NOT_RUN",
            "history": history,
        },
    )
    return target


def validate_g3_marker(
    marker_path: Path,
    root: Path,
    *,
    expected_scope: str,
) -> Mapping[str, object]:
    root = root.resolve()
    marker = _load_json(marker_path)
    if (
        marker.get("marker_version") != MARKER_VERSION
        or marker.get("gate") != "G3"
        or marker.get("freeze_status") != "ACTIVE"
    ):
        raise G3FreezeError(
            "G3 marker is inactive or has wrong schema"
        )
    if marker.get("freeze_scope") != expected_scope:
        raise G3FreezeError(
            "G3 marker scope cannot authorize this test path"
        )
    content = dict(marker)
    supplied = content.pop("marker_sha256", None)
    if supplied != canonical_sha256(content):
        raise G3FreezeError("G3 marker integrity hash mismatch")
    if marker.get("test_accessed_before_freeze") is not False:
        raise G3FreezeError("G3 marker records prior test access")
    dependencies = marker.get("dependencies")
    if not isinstance(dependencies, list):
        raise G3FreezeError("G3 dependency list is malformed")
    for dependency in dependencies:
        if not isinstance(dependency, Mapping):
            raise G3FreezeError(
                "G3 dependency record is malformed"
            )
        _verify_ref(
            root,
            dependency.get("ref"),
            dependency.get("sha256"),
        )
    report = audit_g3(
        root,
        scope=expected_scope,
        allow_authorized_state=True,
    )
    if report.overall != "PASS":
        raise G3FreezeError(
            "G3 dependencies no longer satisfy prerequisite audit"
        )
    current = _manifest_state(root, expected_scope)
    expected_fields = {
        "split_hash": current["split_hash"],
        "selected_models_manifest": current["manifest_ref"],
        "selected_models_sha256": current["manifest_sha256"],
        "tasks": current["tasks"],
        "support_calibrator": current["support_calibrator"],
        "support_threshold": current["support_threshold"],
        "governance_bindings": current.get("governance_bindings", {}),
        "code_commit": current.get("code_commit"),
        "status": "G3_ACTIVE",
        "test_accessed": False,
        "all_model_choices_frozen": True,
        "calibration_frozen": True,
        "threshold_frozen": True,
        "explanation_routing_frozen": True,
    }
    if any(marker.get(key) != value for key, value in expected_fields.items()):
        raise G3FreezeError("G3 marker declarations differ from frozen state")
    expected_dependencies = _dependency_bindings(root, current)
    if tuple(dependencies) != expected_dependencies:
        raise G3FreezeError("G3 marker dependency set is incomplete or stale")
    runs = read_run_registry(root / "experiments/registry.csv")
    if marker.get("validation_searches") != _audit_searches(
        runs, expected_scope
    ) or marker.get("lstm_sensitivity_runs") != _audit_lstm(
        runs, current, expected_scope
    ):
        raise G3FreezeError("G3 search or LSTM declarations are stale")
    return marker


def guarded_test_access(
    root: Path,
    loader: Callable[[], object],
    *,
    marker_path: Optional[Path] = None,
    expected_scope: str = "real",
) -> object:
    root = root.resolve()
    marker = marker_path or (root / REAL_MARKER_RELATIVE_PATH)
    validate_g3_marker(
        marker, root, expected_scope=expected_scope
    )
    access_path = root / ACCESS_STATE_RELATIVE_PATH
    state = dict(load_access_state(access_path))
    if state["state"] != "AUTHORIZED_NOT_RUN":
        raise G3FreezeError(
            "single final evaluation already consumed or not authorized"
        )
    history = list(state["history"])
    history.append(
        {
            "event": "FINAL_TEST_ACCESS_CONSUMED",
            "at_utc": utc_now(),
            "marker_sha256": sha256_file(marker),
            "scope": expected_scope,
        }
    )
    _write_json_atomic(
        access_path,
        {
            "state_version": ACCESS_STATE_VERSION,
            "state": "FINAL_RUN_COMPLETED",
            "history": history,
        },
    )
    return loader()


def invalidate_for_reset(
    root: Path,
    *,
    reason: str,
    marker_path: Optional[Path] = None,
) -> Path:
    if not reason.strip():
        raise G3FreezeError("scientific reset requires a reason")
    root = root.resolve()
    marker = marker_path or (root / REAL_MARKER_RELATIVE_PATH)
    if not marker.is_file():
        raise G3FreezeError(
            "scientific reset requires the prior marker"
        )
    state_path = root / ACCESS_STATE_RELATIVE_PATH
    state = dict(load_access_state(state_path))
    if state["state"] != "FINAL_RUN_COMPLETED":
        raise G3FreezeError(
            "only a consumed final run can be formally reset"
        )
    archive = root / "artifacts/governance/history"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / (
        "invalidated_" + sha256_file(marker)[:16] + ".json"
    )
    shutil.copy2(marker, destination)
    marker.unlink()
    history = list(state["history"])
    history.append(
        {
            "event": "SCIENTIFIC_RESET_INVALIDATED",
            "at_utc": utc_now(),
            "reason": reason,
            "archived_marker_ref": str(
                destination.relative_to(root)
            ),
        }
    )
    _write_json_atomic(
        state_path,
        {
            "state_version": ACCESS_STATE_VERSION,
            "state": "INVALIDATED_BY_RESET",
            "history": history,
        },
    )
    return destination


def invalidate_invalid_pretest_marker(
    root: Path,
    *,
    reason: str,
    marker_path: Optional[Path] = None,
    expected_scope: str = "real",
) -> Path:
    """Archive only an invalid marker before final-test access is consumed."""
    if not reason.strip():
        raise G3FreezeError("pre-test marker correction requires a reason")
    root = root.resolve()
    marker = marker_path or (root / REAL_MARKER_RELATIVE_PATH)
    if not marker.is_file():
        raise G3FreezeError("pre-test marker correction requires the prior marker")
    state_path = root / ACCESS_STATE_RELATIVE_PATH
    state = dict(load_access_state(state_path))
    if state["state"] != "AUTHORIZED_NOT_RUN" or any(
        row.get("event") == "FINAL_TEST_ACCESS_CONSUMED" for row in state["history"]
    ):
        raise G3FreezeError("pre-test correction is forbidden after final-test access")
    try:
        validate_g3_marker(marker, root, expected_scope=expected_scope)
    except G3FreezeError as validation_error:
        failure = str(validation_error)
    else:
        raise G3FreezeError("valid active G3 cannot use pre-test correction")
    archive = root / "artifacts/governance/history"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / ("invalid_pretest_" + sha256_file(marker)[:16] + ".json")
    shutil.copy2(marker, destination)
    marker.unlink()
    history = list(state["history"])
    history.append({
        "event": "INVALID_PRETEST_G3_CORRECTED",
        "at_utc": utc_now(),
        "reason": reason,
        "validation_failure": failure,
        "archived_marker_ref": str(destination.relative_to(root)),
        "test_accessed": False,
    })
    _write_json_atomic(state_path, {
        "state_version": ACCESS_STATE_VERSION,
        "state": "INVALIDATED_BY_RESET",
        "history": history,
    })
    return destination


def require_development_change_allowed(
    root: Path, change_type: str
) -> None:
    if change_type not in (
        "model_selection",
        "threshold",
        "calibration",
        "features",
        "labels",
    ):
        raise G3FreezeError(
            "unknown result-affecting change type"
        )
    state = load_access_state(
        root / ACCESS_STATE_RELATIVE_PATH
    )
    if state["state"] in (
        "AUTHORIZED_NOT_RUN",
        "FINAL_RUN_COMPLETED",
    ):
        raise G3FreezeError(
            "formal reset required before post-freeze scientific change"
        )
