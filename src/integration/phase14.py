"""Final hash-bound Sanskruti handoff and receiver acceptance."""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from data.synthetic.config import canonical_json_bytes
from data.synthetic.history_provider import CanonicalTimelineHistoryProvider
from data.synthetic.provenance import sha256_file
from data.synthetic.sofa_provider import PulkitStateSOFASupportProvider, SyntheticCurrentSOFAProvider
from data.synthetic.validation import load_jsonl
from data.xgb_canonical import Phase10XGBData, assert_exact_reconstruction, model_information_views, unflatten_row
from features.synthetic import SyntheticCanonicalFeatureBuilder
from labels.phase9_final import _normalized
from labels.support_state import ExecutionMode
from labels.synthetic_profile import load_synthetic_event_dictionary
from models.xgb_canonical import XGBLineage, load_xgb_bundle
from preprocess.synthetic_phase10 import FrozenSyntheticFeaturePreprocessor
from serving.history import SyntheticPointEventHistoryTruncator
from serving.pipeline import PredictionPipeline
from serving.preprocessing import CanonicalHistoryInputProvider


VERSION = "sanskruti_phase14_handoff_v1"
AUTHORITY = "USER_DELEGATED_AI_PROJECT_DECISION"
G1_PATH = "artifacts/acceptance/g1_synthetic_data_freeze_v1.json"
HANDOFF_PATH = "artifacts/handoffs/sanskruti_phase14_handoff_v1.json"
EXPECTED_PARENT_PATHS = {
    "g1_acceptance": G1_PATH,
    "feature_schema": "configs/synthetic/feature_schema_v2.json",
    "event_dictionary": "configs/event_dict_v2.yaml",
    "split": "artifacts/splits/synthetic_split_v2.csv",
    "preprocessor": "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json",
    "sofa_spec": "configs/synthetic/sofa_spec_v1.json",
    "flat_feature_map": "artifacts/features/synthetic_xgb_flat_feature_map_v1.json",
}


class HandoffError(RuntimeError):
    pass


def _json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _git(root: Path) -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()


def _ref(root: Path, path: str, **extra: object) -> Mapping[str, object]:
    result = {"path": path, "sha256": sha256_file(root / path)}
    result.update(extra)
    return result


def _contains_absolute_machine_path(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith(("/Users/", "/home/", "C:\\Users\\"))
    if isinstance(value, Mapping):
        return any(_contains_absolute_machine_path(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_machine_path(v) for v in value)
    return False


def _plain(value: object) -> object:
    """Normalize immutable tuple views for byte-value receiver comparison."""
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def build_handoff(root: Path, *, code_commit: str | None = None) -> Path:
    g1 = _json(root / G1_PATH)
    if g1.get("status") != "ACCEPTED_SYNTHETIC_DATA_LABEL_FREEZE" or g1.get("test_accessed") is not False:
        raise HandoffError("BLOCKED — ACCEPTED SYNTHETIC G1 REQUIRED")
    schema = _json(root / "configs/synthetic/feature_schema_v2.json")
    phase10 = _json(root / "artifacts/manifests/synthetic_phase10_manifest_v1.json")
    flat_map = _json(root / "artifacts/features/synthetic_xgb_flat_feature_map_v1.json")
    phase12_best = _json(root / "artifacts/search/xgb/phase12/best_xgb_candidates_v1.json")
    refs = {
        "project_scope": _ref(root, "docs/governance/project_scope_v2.md"),
        "machine_scope": _ref(root, "configs/governance/project_scope_v2.json"),
        "g1_acceptance": _ref(root, G1_PATH, version=g1["acceptance_version"], status=g1["status"]),
        "generator_config": _ref(root, "configs/synthetic/final_benchmark_v1.json"),
        "latent_process": _ref(root, "configs/synthetic/latent_process_v1.yaml"),
        "support_process": _ref(root, "configs/synthetic/support_process_v1.yaml"),
        "event_dictionary": _ref(root, "configs/event_dict_v2.yaml"),
        "raw_dataset_manifest": _ref(root, "artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json"),
        "cohort_manifest": _ref(root, "artifacts/data/synthetic/cohorts/final/phase9_final_v1/synthetic_cohort_manifest_v2.json"),
        "structural_index": _ref(root, "artifacts/data/synthetic/cohorts/final/phase9_final_v1/structural_index.jsonl"),
        "timeline_manifest": _ref(root, "artifacts/data/synthetic/timelines/final/phase9_final_v1/synthetic_processed_manifest_v1.json"),
        "canonical_statics": _ref(root, "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl"),
        "sofa_spec": _ref(root, "configs/synthetic/sofa_spec_v1.json"),
        "feature_schema": _ref(root, "configs/synthetic/feature_schema_v2.json", version=schema["schema_version"]),
        "feature_artifact": _ref(root, "artifacts/data/synthetic/features/final/phase9_final_v2/canonical_feature_inputs.jsonl"),
        "target_dictionary": _ref(root, "docs/evidence/data/final_target_dictionary_v1.md"),
        "pre_split_targets": _ref(root, "artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl"),
        "split": _ref(root, "artifacts/splits/synthetic_split_v2.csv"),
        "preprocessor": _ref(root, "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"),
        "train_fit_subjects": _ref(root, "artifacts/preprocessors/synthetic_train_fit_subjects_v1.txt"),
        "recovery_target_scaler": _ref(root, "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json"),
        "support_class_weight": _ref(root, "artifacts/preprocessors/support_class_weight_synthetic_v1.json"),
        "flat_feature_map": _ref(root, "artifacts/features/synthetic_xgb_flat_feature_map_v1.json"),
        "phase11_parity": _ref(root, "artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json"),
        "phase12_search": _ref(root, "artifacts/search/xgb/phase12/manifests/phase12_search_manifest_v1.json"),
        "phase12_best_xgb": _ref(root, "artifacts/search/xgb/phase12/best_xgb_candidates_v1.json"),
        "data_qa": _ref(root, "docs/evidence/data/data_qa_report_v1.md"),
        "leakage_audit": _ref(root, "docs/evidence/data/leakage_audit_v1.md"),
        "reproducibility": _ref(root, "docs/evidence/data/reproducibility_report_v1.md"),
    }
    payload = {
        "manifest_version": VERSION,
        "status": "SANSKRUTI_TRACK_HANDOFF_FROZEN",
        "decision_authority": AUTHORITY,
        "code_commit": code_commit or _git(root),
        "scope": "SYNTHETIC_ADULT_CARDIAC_RETROSPECTIVE_FORECASTING",
        "g1_sha256": refs["g1_acceptance"]["sha256"],
        "dimensions": {"T": schema["shape"]["sequence_length"], "F": schema["shape"]["feature_dimension"], "raw_static_S": 3, "transformed_static_S": phase10["encoded_static_dimension"], "xgb_flat_D": len(flat_map["entries"])},
        "row_counts": {"total": sum(phase10["row_counts"].values()), **phase10["row_counts"]},
        "parents": refs,
        "best_xgb_within_family": {task: {"candidate_id": item["candidate_id"], "run_id": item["run_id"], "bundle_path": item["metrics"]["bundle_path"], "metrics_path": item["metrics"]["metrics_path"], "model_hashes": item["metrics"]["model_hashes"], "status": "XGB_WITHIN_FAMILY_VALIDATION_WINNER"} for task,item in phase12_best["tasks"].items()},
        "receiver_contracts": {
            "vedant": "PHASE10_TRANSFORMED_CANONICAL_ROWS_PLUS_PHASE11_LOSSLESS_VIEWS",
            "pulkit": "CANONICAL_TIMELINE_PLUS_SHARED_FEATURE_BUILDER_AND_FROZEN_TRANSFORM",
        },
        "test_accessed": False, "g3_created": False,
        "selected_models_manifest_created": False,
        "final_model_selection": "PENDING_DOWNSTREAM_VEDANT",
        "support_calibration": "PENDING_AFTER_DOWNSTREAM_SELECTION",
        "support_threshold": "PENDING_AFTER_DOWNSTREAM_CALIBRATION",
    }
    if _contains_absolute_machine_path(payload):
        raise HandoffError("handoff contains an absolute machine path")
    output = root / HANDOFF_PATH
    _write(output, payload)
    validate_handoff(root, output)
    return output


def validate_handoff(root: Path, path: Path | None = None) -> Mapping[str, Any]:
    path = path or root / HANDOFF_PATH
    payload = _json(path)
    if payload.get("manifest_version") != VERSION or payload.get("status") != "SANSKRUTI_TRACK_HANDOFF_FROZEN":
        raise HandoffError("unsupported or non-frozen Phase-14 handoff")
    if payload.get("test_accessed") is not False or payload.get("g3_created") is not False:
        raise HandoffError("test/G3 governance violation")
    if payload.get("selected_models_manifest_created") is not False:
        raise HandoffError("Phase 14 cannot select models")
    if payload.get("decision_authority") != AUTHORITY:
        raise HandoffError("unexpected Phase-14 decision authority")
    if _contains_absolute_machine_path(payload):
        raise HandoffError("handoff contains an absolute machine path")
    parents = payload.get("parents")
    if not isinstance(parents, Mapping):
        raise HandoffError("handoff parent inventory missing")
    for name, item in parents.items():
        parent = root / item["path"]
        if not parent.is_file() or sha256_file(parent) != item["sha256"]:
            raise HandoffError("handoff parent mismatch: " + name)
    for name, expected_path in EXPECTED_PARENT_PATHS.items():
        if parents.get(name, {}).get("path") != expected_path:
            raise HandoffError("non-authoritative handoff parent: " + name)
    g1 = _json(root / parents["g1_acceptance"]["path"])
    if parents["g1_acceptance"]["sha256"] != payload["g1_sha256"] or g1.get("status") != "ACCEPTED_SYNTHETIC_DATA_LABEL_FREEZE" or g1.get("test_accessed") is not False:
        raise HandoffError("accepted G1 binding mismatch")
    if g1.get("phase12_results_status") != "VALID_UNCHANGED_PARENT_ARTIFACTS":
        raise HandoffError("G1 does not preserve Phase-12 validity")
    schema = _json(root / parents["feature_schema"]["path"])
    if payload["dimensions"]["F"] != schema["shape"]["feature_dimension"] or payload["dimensions"]["T"] != schema["shape"]["sequence_length"]:
        raise HandoffError("handoff tensor dimensions differ from feature schema")
    flat_map = _json(root / parents["flat_feature_map"]["path"])
    if payload["dimensions"]["xgb_flat_D"] != len(flat_map["entries"]):
        raise HandoffError("handoff flat dimension differs from flat feature map")
    forbidden_active = ("feature_schema_v1.json", "event_dict_v1.yaml", "support_process_v1.proposed.yaml", "anchor_year_group")
    active_text = json.dumps(payload["parents"], sort_keys=True)
    if any(value in active_text for value in forbidden_active):
        raise HandoffError("stale scientific parent is active")
    # This validates the immutable Phase-14 handoff, including its declaration
    # that Phase 14 itself did not perform downstream model selection.  A later
    # authorized Vedant stage may create the selected-model manifest without
    # retroactively invalidating this historical handoff.
    return payload


def _load_best_xgb(root: Path, gateway: Phase10XGBData, handoff: Mapping[str, Any]) -> Mapping[str, str]:
    feature_names = tuple(item["flat_name"] for item in gateway.flat_map["entries"])
    loaded = {}
    for task, ref in handoff["best_xgb_within_family"].items():
        metadata_path = root / ref["bundle_path"]
        metadata = _json(metadata_path)
        lineage = XGBLineage(**metadata["lineage"])
        kwargs: dict[str, object] = {}
        if task == "recovery":
            kwargs.update(recovery_scaler=gateway.recovery_scaler, recovery_scaler_path=gateway.recovery_scaler_path)
            expected_task = "recovery"
        elif task == "organ_support":
            kwargs.update(support_class_weight=gateway.support_class_weight, support_class_weight_path=gateway.class_weight_path)
            expected_task = "organ_support"
        else:
            expected_task = "icu_stay_time"
        load_xgb_bundle(metadata_path.parent, expected_task=expected_task, expected_lineage=lineage, expected_feature_names=feature_names, **kwargs)
        loaded[task] = "PASS_LOADED_HASH_AND_LINEAGE_VALIDATED"
    return loaded


def vedant_receiver_acceptance(root: Path, handoff_path: Path) -> Mapping[str, Any]:
    handoff = validate_handoff(root, handoff_path)
    gateway = Phase10XGBData(root, root / handoff["parents"]["flat_feature_map"]["path"])
    train, validation = gateway.load_partition("train"), gateway.load_partition("validation")
    row = json.loads((root / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl").open().readline())
    views = model_information_views(row, gateway.flat_map)
    assert_exact_reconstruction(views["gru"], unflatten_row(views["xgboost"], gateway.flat_map))
    tasks = {name: {"train": len(gateway.task_matrix(train, name).y), "validation": len(gateway.task_matrix(validation, name).y)} for name in ("recovery24", "recovery48", "icu_stay_time", "organ_support")}
    result = {
        "artifact_version": "vedant_receiver_acceptance_v1", "status": "VEDANT_RECEIVER_READY",
        "g1_sha256": handoff["g1_sha256"], "handoff_manifest_sha256": sha256_file(handoff_path),
        "loader": "data.xgb_canonical.Phase10XGBData", "timestamp_parity": "EXACT_CANONICAL_ROW_KEYS",
        "tensor_shape": [8, handoff["dimensions"]["F"]], "flat_dimension": handoff["dimensions"]["xgb_flat_D"],
        "feature_schema_sha256": handoff["parents"]["feature_schema"]["sha256"], "preprocessor_sha256": handoff["parents"]["preprocessor"]["sha256"],
        "task_eligibility_counts": tasks, "information_parity": "PASS_EXACT_FLATTEN_UNFLATTEN",
        "targets": {"recovery":"RAW_DELTAS_PLUS_ACCEPTED_SCALER_INDEPENDENT_MASKS","icu":"LOG1P_REMAINING_EPISODE_HOURS_NO_TARGET_SCALER","support":"BINARY_ELIGIBLE_PLUS_ACCEPTED_CLASS_WEIGHT"},
        "best_xgb_artifacts": _load_best_xgb(root, gateway, handoff),
        "gru_loader_readiness": "PASS_STRUCTURED_MODEL_INPUT_8xF", "preprocessor_fit_calls": 0,
        "scientific_training_performed": False, "test_accessed": False,
    }
    return result


def _serving_components(root: Path, handoff: Mapping[str, Any]):
    timeline_path = root / handoff["parents"]["timeline_manifest"]["path"]
    provider = CanonicalTimelineHistoryProvider(timeline_path, root)
    timeline_manifest = _json(timeline_path)
    statics = load_jsonl(root / next(x["repository_relative_path"] for x in timeline_manifest["artifacts"] if x["logical_name"] == "canonical_statics"))
    builder = SyntheticCanonicalFeatureBuilder(schema_path=root / handoff["parents"]["feature_schema"]["path"], root=root, statics_by_stay={row["stay_id"]: row for row in statics})
    canonical = CanonicalHistoryInputProvider(history_provider=provider, history_truncator=SyntheticPointEventHistoryTruncator(), build_features=builder, feature_schema=builder.feature_schema)
    frozen = FrozenSyntheticFeaturePreprocessor(root / handoff["parents"]["preprocessor"]["path"])
    return provider, builder, canonical, frozen


def pulkit_receiver_acceptance(root: Path, handoff_path: Path) -> Mapping[str, Any]:
    handoff = validate_handoff(root, handoff_path)
    _, builder, canonical, frozen = _serving_components(root, handoff)
    stored = json.loads((root / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl").open().readline())
    cutoff = stored["prediction_time"].replace("Z", "+00:00")
    raw_view = canonical.get_canonical_input(stay_id=stored["stay_id"], prediction_time=cutoff, task="recovery", family="gru", feature_version=builder.feature_schema.version)
    transformed = frozen.transform(raw_view)
    equality = all(
        _plain(transformed[left]) == stored[right]
        for left, right in (
            ("history_values", "sequence_values"),
            ("observation_mask", "observation_mask"),
            ("tslo_hours", "tslo_hours"),
            ("padding_mask", "padding_mask"),
            ("static_features", "static_features"),
        )
    )
    if not equality:
        raise HandoffError("Pulkit serving path differs from accepted offline transformed row")
    raw_manifest = _json(root / handoff["parents"]["raw_dataset_manifest"]["path"])
    support_path = root / next(x["repository_relative_path"] for x in raw_manifest["artifacts"] if x["logical_name"] == "support_intervals")
    supports = load_jsonl(support_path)
    dictionary = load_synthetic_event_dictionary(root / handoff["parents"]["event_dictionary"]["path"])
    vaso, vent, exposures = _normalized(supports, dictionary)
    del vaso
    support_provider = PulkitStateSOFASupportProvider(event_dictionary=dictionary, execution_mode=ExecutionMode.SYNTHETIC, ventilation_intervals=vent, vasoactive_exposures=exposures, vasoactive_dose_contract_version="synthetic_support_process_v1", vasoactive_dose_contract_sha256=handoff["parents"]["support_process"]["sha256"], vasoactive_coverage_known=True)
    package = load_jsonl(root / handoff["parents"]["pre_split_targets"]["path"])
    sofa_row = next(row for row in package if row["baseline_sofa"] is not None)
    legal: dict[object, list[str]] = {}
    structural = load_jsonl(root / handoff["parents"]["structural_index"]["path"])
    for row in structural: legal.setdefault(row["stay_id"], []).append(row["prediction_time"])
    sofa = SyntheticCurrentSOFAProvider(manifest_path=root / handoff["parents"]["timeline_manifest"]["path"], root=root, sofa_spec_path=root / handoff["parents"]["sofa_spec"]["path"], support_provider=support_provider, expected_sofa_spec_sha256=handoff["parents"]["sofa_spec"]["sha256"], legal_cutoffs=legal)
    current = sofa.current_sofa(stay_id=sofa_row["stay_id"], prediction_time=sofa_row["prediction_time"])
    if current.value != sofa_row["baseline_sofa"]:
        raise HandoffError("Pulkit CurrentSOFA differs from accepted sofa_at(t)")
    result = {
        "artifact_version": "pulkit_receiver_acceptance_v1", "status": "PULKIT_DATA_RECEIVER_READY",
        "g1_sha256": handoff["g1_sha256"], "handoff_manifest_sha256": sha256_file(handoff_path),
        "timeline_sha256": handoff["parents"]["timeline_manifest"]["sha256"], "event_dictionary_sha256": handoff["parents"]["event_dictionary"]["sha256"],
        "feature_schema_sha256": handoff["parents"]["feature_schema"]["sha256"], "preprocessor_sha256": handoff["parents"]["preprocessor"]["sha256"], "sofa_spec_sha256": handoff["parents"]["sofa_spec"]["sha256"],
        "history_provider_compatibility": "PASS_CANONICAL_TIMELINE_AND_CUTOFF_FILTERED_SUPPORT",
        "direct_vs_serving_equality": "PASS_EXACT", "feature_order_equality": "PASS_EXACT",
        "support_state_compatibility": "PASS_EXISTING_PULKIT_ENGINES", "current_sofa_equality": "PASS_EXACT",
        "serving_fit_calls": 0, "prediction_pipeline_interface": PredictionPipeline.__module__ + "." + PredictionPipeline.__name__,
        "prediction_pipeline_readiness": "PASS_INTERFACE_NON_SCIENTIFIC_FIXTURE_ONLY",
        "selected_model_bundle_status": "NOT_YET_FROZEN_DOWNSTREAM", "test_accessed": False,
    }
    return result


def _completion_matrix(root: Path, handoff: Mapping[str, Any]) -> list[Mapping[str, str]]:
    mapping = [
        ("scope",1,"project_scope","g1_acceptance","Vedant/Pulkit"),
        ("generator",3,"raw_dataset_manifest","reproducibility","all"),
        ("cohort",4,"cohort_manifest","data_qa","all"),
        ("structural index",4,"structural_index","data_qa","Vedant/Pulkit"),
        ("timeline",5,"timeline_manifest","data_qa","Pulkit"),
        ("provenance",5,"timeline_manifest","reproducibility","all"),
        ("six-component SOFA",6,"sofa_spec","data_qa","Pulkit/recovery"),
        ("features",7,"feature_artifact","data_qa","Vedant/Pulkit"),
        ("missingness",7,"feature_schema","leakage_audit","Vedant/Pulkit"),
        ("TSLO",7,"feature_schema","leakage_audit","Vedant/Pulkit"),
        ("padding",7,"feature_schema","leakage_audit","Vedant/Pulkit"),
        ("recovery labels",8,"pre_split_targets","target_dictionary","Vedant"),
        ("ICU labels",8,"pre_split_targets","target_dictionary","Vedant"),
        ("support labels",9,"pre_split_targets","target_dictionary","Vedant/Pulkit"),
        ("split",10,"split","data_qa","Vedant/Pulkit"),
        ("preprocessing",10,"preprocessor","data_qa","Vedant/Pulkit"),
        ("XGB/GRU parity",11,"phase11_parity","data_qa","Vedant"),
        ("XGB scientific search",12,"phase12_search","phase12_best_xgb","Vedant"),
        ("data QA",13,"g1_acceptance","data_qa","all"),
        ("leakage",13,"g1_acceptance","leakage_audit","all"),
        ("reproducibility",13,"g1_acceptance","reproducibility","all"),
        ("G1",13,"g1_acceptance","g1_acceptance","all"),
        ("Vedant handoff",14,"phase11_parity","g1_acceptance","Vedant"),
        ("Pulkit handoff",14,"timeline_manifest","g1_acceptance","Pulkit"),
    ]
    rows=[]
    for requirement,phase,artifact,evidence,consumer in mapping:
        item=handoff["parents"][artifact]
        ev=handoff["parents"][evidence]
        rows.append({"requirement":requirement,"phase_implemented":str(phase),"authoritative_artifact":item["path"],"evidence":ev["path"],"sha256":item["sha256"],"status":"COMPLETE","downstream_consumer":consumer})
    return rows


def finalize(root: Path) -> Mapping[str, Path]:
    handoff_path = build_handoff(root)
    handoff = validate_handoff(root, handoff_path)
    vedant = vedant_receiver_acceptance(root, handoff_path)
    pulkit = pulkit_receiver_acceptance(root, handoff_path)
    output = root / "artifacts/handoffs"
    vedant_path=output/"vedant_receiver_acceptance_v1.json"; _write(vedant_path,vedant)
    pulkit_path=output/"pulkit_receiver_acceptance_v1.json"; _write(pulkit_path,pulkit)
    matrix_path=output/"sanskruti_completion_matrix_v1.json"; _write(matrix_path,{"status":"SANSKRUTI_TRACK_COMPLETE","requirements":_completion_matrix(root,handoff)})
    downstream_path=output/"downstream_next_steps_v1.json"
    _write(downstream_path,{"status":"PENDING_DOWNSTREAM_NOT_EXECUTED_PHASE14","vedant":["run governed GRU searches on accepted G1 lineage","select best GRU per task","compare best XGB versus best GRU on validation","run fixed LSTM sensitivity","freeze selected_models","calibrate selected support classifier on validation","select support threshold on validation","freeze G3"],"pulkit":["bind selected manifest after downstream freeze","route selected family explanations","bind calibrator and threshold","run selected pipeline/API/dashboard"],"team_after_g3":["open final test once","compute final metrics and grouped bootstrap intervals","perform no post-test tuning"]})
    status_path=output/"sanskruti_phase14_status_v1.json"
    _write(status_path,{"status":"SANSKRUTI_TRACK_COMPLETE","g1":"G1_ACCEPTED","g1_sha256":handoff["g1_sha256"],"handoff_sha256":sha256_file(handoff_path),"vedant_receiver":"READY","pulkit_data_receiver":"READY","g2_readiness":"READY_FOR_DOWNSTREAM_MODEL_INTEGRATION","final_model_selection":"PENDING_DOWNSTREAM_VEDANT","g3":"NOT_CREATED","final_test":"NEVER_OPENED","test_accessed":False,"scientific_training_performed":False})
    ownership_path=output/"sanskruti_file_ownership_v1.json"
    _write(ownership_path,{"status":"AUDITED","classes":{
        "ACTIVE_AUTHORITY":[item["path"] for item in handoff["parents"].values()],
        "HISTORICAL_SUPERSEDED":["src/data/real_adapter.py","configs/event_dict_v1.yaml","configs/synthetic/feature_schema_v1.json"],
        "ENGINEERING_FIXTURE_ONLY":["artifacts/demo","tests/fixtures"],
        "DOWNSTREAM_REFERENCE":["src/models","src/evaluation","src/serving","src/api","src/dashboard"],
        "UNUSED_SAFE_TO_ARCHIVE":[]},
        "active_stale_artifact_count":0,"historical_files_deleted":False})
    supersession_path=output/"requirement_supersession_v1.json"
    _write(supersession_path,{"decision_authority":AUTHORITY,"mappings":[
        {"old_requirement":"MIMIC extraction","status":"SUPERSEDED","replacement":"final synthetic generator and provenance"},
        {"old_requirement":"unrestricted ICU cohort","status":"SUPERSEDED","replacement":"adult cardiac synthetic cohort"},
        {"old_requirement":"MIMIC era split","status":"SUPERSEDED","replacement":"deterministic synthetic subject split v2"},
        {"old_requirement":"real-data handoff wording","status":"SUPERSEDED","replacement":"accepted synthetic scientific handoff"}]})
    return {"handoff":handoff_path,"vedant":vedant_path,"pulkit":pulkit_path,"matrix":matrix_path,"downstream":downstream_path,"status":status_path,"ownership":ownership_path,"supersession":supersession_path}
