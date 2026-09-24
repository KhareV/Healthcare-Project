"""Stage-5 one-time final-test loader and evaluator.

``stage5_test_loader`` is the single authorized point where the sealed test
partition is ever materialized and fed through the three frozen G3-selected
models — the exact action ``vedant_infra.g3.guarded_test_access`` marks as
consumed. It performs zero training, zero refitting, and zero calibration.

``stage5_evaluator`` performs only downstream analysis (metrics, bootstrap,
sensitivity, error analysis) on the already-obtained frozen predictions using
the existing Phase-9/14/15/20 evaluation modules, then persists and registers
every result artifact and freezes G4.

Both are designed to be dry-run against the ``validation`` partition (never
``test``) via ``load_and_infer(root, partition="validation")`` for testing,
completely separately from the guarded final-test entrypoint.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

from data.synthetic.config import canonical_json_bytes
from data.synthetic.sofa_provider import PulkitStateSOFASupportProvider, SyntheticCurrentSOFAProvider
from data.synthetic.validation import load_jsonl
from data.xgb_canonical import model_information_views
from evaluation.bootstrap import bootstrap_icu_time, bootstrap_organ_support, bootstrap_recovery, grouped_bootstrap
from evaluation.error_analysis import (
    ErrorAnalysisRecord,
    ErrorAnalysisTableRow,
    SelectedModelBinding,
    evaluate_error_analysis,
    write_error_analysis_table,
)
from evaluation.final_test import (
    FinalPredictionBundle,
    FinalTestError,
    ICULogPredictionRecord,
    apply_frozen_support_postprocessing,
    bootstrap_frozen_predictions,
    evaluate_frozen_predictions,
    evaluate_prespecified_test_error_analysis,
    evaluate_test_complete_component,
    process_icu_time_predictions,
    validate_final_result_traceability,
)
from evaluation.metrics import (
    EvaluationResult,
    PredictionRecord,
    evaluate_icu_time,
    evaluate_organ_support,
    evaluate_recovery_horizon,
)
from evaluation.naive_baseline import NaiveBaselineArtifact, load_or_fit_naive_baselines
from evaluation.sensitivity import CompleteComponentRecord, evaluate_complete_component_sensitivity
from evaluation.slices import SliceDefinition, SliceGroup, SliceSpecification
from experiments.lineage import ArtifactRecord, read_artifact_index, read_run_registry, register_artifact
from experiments.search_governance import canonical_sha256
from labels.phase9_final import _normalized
from labels.support_state import ExecutionMode
from labels.synthetic_profile import load_synthetic_event_dictionary
from models.gru_icu_time import load_icu_time_bundle
from models.icu_time_postprocess import remaining_icu_hours_from_log_prediction
from models.xgb_canonical import XGBLineage, load_xgb_bundle
from preprocess.synthetic_phase10 import transform_row
from preprocess.target_scaler import RecoveryTargetScaler
from training.class_weights import SupportClassWeight
from vedant_infra.g3 import ACCESS_STATE_RELATIVE_PATH, load_access_state, utc_now, validate_g3_marker
from vedant_infra.hashing import is_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PRE_SPLIT_PATH = "artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl"
SPLIT_PATH = "artifacts/splits/synthetic_split_v2.csv"
PREPROCESSOR_PATH = "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"
FLAT_MAP_PATH = "artifacts/features/synthetic_xgb_flat_feature_map_v1.json"
SELECTED_MODELS_PATH = "artifacts/models/selected_models_v1.json"
G3_PATH = "artifacts/governance/g3_freeze.json"
CALIBRATOR_PATH = "artifacts/calibration/isotonic_support_v1.json"
THRESHOLD_PATH = "artifacts/thresholds/support_threshold_v1.json"
NAIVE_BASELINE_PATH = "artifacts/final_test/naive_baseline/naive_baseline_v1.json"
TRAIN_PATH = "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/train.jsonl"
TIMELINE_MANIFEST_PATH = "artifacts/data/synthetic/timelines/final/phase9_final_v1/synthetic_processed_manifest_v1.json"
RAW_DATASET_MANIFEST_PATH = "artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json"
EVENT_DICT_PATH = "configs/event_dict_v2.yaml"
SUPPORT_PROCESS_PATH = "configs/synthetic/support_process_v1.yaml"
SOFA_SPEC_PATH = "configs/synthetic/sofa_spec_v1.json"


class Stage5Error(RuntimeError):
    pass


def _json(root: Path, ref: str) -> Mapping[str, object]:
    return json.loads((root / ref).read_text(encoding="utf-8"))


def _split_assignments(root: Path) -> Mapping[str, str]:
    with (root / SPLIT_PATH).open(newline="", encoding="utf-8") as handle:
        return {row["subject_id"]: row["split"] for row in csv.DictReader(handle)}


def materialize_partition_rows(root: Path, partition: str, preprocessor: Mapping[str, object]) -> Tuple[Mapping[str, object], ...]:
    """Apply the frozen transform to one partition's rows for the first time.

    Reuses ``preprocess.synthetic_phase10.transform_row`` exactly as Phase 10
    used it for train/validation — no refitting, same frozen preprocessor
    payload. This is the one place ``test`` rows are ever transformed.
    """

    assignments = _split_assignments(root)
    rows = load_jsonl(root / PRE_SPLIT_PATH)
    selected = tuple(row for row in rows if assignments.get(str(row["subject_id"])) == partition)
    if not selected:
        raise Stage5Error("no rows found for partition: " + partition)
    return tuple(transform_row(row, partition, preprocessor) for row in selected)


def _canonical_batch(structured, *, feature_names: Sequence[str], feature_schema_version: str, prediction_time: str):
    from data.collate import CanonicalBatch

    return CanonicalBatch(
        identifiers={"prediction_time": (prediction_time,)},
        sequence=torch.tensor(structured.values[None, ...], dtype=torch.float32),
        padding_mask=torch.tensor(structured.padding_mask[None, ...], dtype=torch.bool),
        observation_mask=torch.tensor(structured.observation_mask[None, ...], dtype=torch.bool),
        tslo=torch.tensor(structured.tslo_hours[None, ...], dtype=torch.float32),
        static_features=torch.tensor(structured.statics[None, ...], dtype=torch.float32),
        targets={},
        eligibility={},
        versions={"feature_schema_version": feature_schema_version},
        feature_names=tuple(feature_names),
    )


@dataclass(frozen=True)
class LoadedModels:
    recovery_bundle: object
    recovery_lineage: XGBLineage
    icu_model: object
    support_bundle: object
    support_lineage: XGBLineage
    calibrator: object
    threshold: float
    flat_map: Mapping[str, object]
    flat_map_sha256: str
    temporal_feature_names: Tuple[str, ...]
    feature_schema_version: str
    manifest: Mapping[str, object]
    manifest_sha256: str
    g3_marker: Mapping[str, object]
    g3_sha256: str


def load_frozen_models(root: Path) -> LoadedModels:
    """Re-verify every dependency hash live, then load read-only, exactly once."""

    marker_path = root / G3_PATH
    marker = validate_g3_marker(marker_path, root, expected_scope="real")
    manifest_path = root / SELECTED_MODELS_PATH
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != marker["selected_models_sha256"]:
        raise Stage5Error("selected_models_v1 hash differs from the G3-bound hash")
    manifest = _json(root, SELECTED_MODELS_PATH)
    if manifest.get("serving_ready") is not False or manifest.get("test_accessed") is not False:
        raise Stage5Error("selected_models_v1 is not in the expected pre-test frozen state")
    tasks = manifest["tasks"]
    for task, entry in tasks.items():
        if entry.get("family") == "lstm":
            raise Stage5Error("LSTM is not final-test eligible")

    flat_map_path = root / FLAT_MAP_PATH
    flat_map = json.loads(flat_map_path.read_text(encoding="utf-8"))
    flat_map_sha256 = sha256_file(flat_map_path)
    if flat_map_sha256 != tasks["recovery"]["flat_feature_map_sha256"]:
        raise Stage5Error("flat feature map hash mismatch")

    recovery_entry = tasks["recovery"]
    recovery_dir = (root / recovery_entry["artifact_ref"]).resolve().parent
    recovery_metadata = json.loads((recovery_dir / "bundle.metadata.json").read_text(encoding="utf-8"))
    recovery_lineage = XGBLineage(**recovery_metadata["lineage"])
    scaler_path = root / manifest["governance_bindings"]["recovery_target_scaler"]["ref"]
    scaler = RecoveryTargetScaler.load(scaler_path)
    recovery_bundle = load_xgb_bundle(
        recovery_dir, expected_task="recovery", expected_lineage=recovery_lineage,
        expected_feature_names=tuple(item["flat_name"] for item in flat_map["entries"]),
        recovery_scaler=scaler, recovery_scaler_path=scaler_path,
    )
    for name, item in recovery_entry["model_artifacts"].items():
        path = recovery_dir / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise Stage5Error("recovery " + name + " model hash mismatch")

    support_entry = tasks["organ_support"]
    support_dir = (root / support_entry["artifact_ref"]).resolve().parent
    support_metadata = json.loads((support_dir / "bundle.metadata.json").read_text(encoding="utf-8"))
    support_lineage = XGBLineage(**support_metadata["lineage"])
    class_weight_path = root / manifest["governance_bindings"]["support_class_weight"]["ref"]
    class_weight = SupportClassWeight.load(class_weight_path)
    support_bundle = load_xgb_bundle(
        support_dir, expected_task="organ_support", expected_lineage=support_lineage,
        expected_feature_names=tuple(item["flat_name"] for item in flat_map["entries"]),
        support_class_weight=class_weight, support_class_weight_path=class_weight_path,
    )
    for name, item in support_entry["model_artifacts"].items():
        path = support_dir / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise Stage5Error("support " + name + " model hash mismatch")

    icu_entry = tasks["icu_stay_time"]
    icu_checkpoint = root / icu_entry["artifact_ref"]
    if sha256_file(icu_checkpoint) != icu_entry["artifact_sha256"]:
        raise Stage5Error("ICU checkpoint hash mismatch")
    icu_sidecar = json.loads((icu_checkpoint.with_name(icu_checkpoint.name + ".metadata.json")).read_text(encoding="utf-8"))
    icu_model, _icu_meta = load_icu_time_bundle(
        icu_checkpoint,
        expected_tensor_contract_version=icu_sidecar["tensor_contract_version"],
        expected_feature_schema_version=icu_sidecar["feature_schema_version"],
    )
    icu_model.eval()

    calibrator_path = root / support_entry["calibrator"]["artifact_ref"]
    if sha256_file(calibrator_path) != support_entry["calibrator"]["artifact_sha256"]:
        raise Stage5Error("support calibrator hash mismatch")
    from evaluation.calibrate import load_calibrator_artifact

    calibrator = load_calibrator_artifact(calibrator_path, expected_sha256=support_entry["calibrator"]["artifact_sha256"])
    threshold_path = root / support_entry["threshold"]["artifact_ref"]
    if sha256_file(threshold_path) != support_entry["threshold"]["artifact_sha256"]:
        raise Stage5Error("support threshold hash mismatch")
    threshold_value = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
    if threshold_value != support_entry["threshold"]["value"]:
        raise Stage5Error("support threshold value mismatch")

    feature_schema_version = recovery_entry["feature_version"]
    temporal_feature_names = tuple(
        item["canonical_feature_name"] for item in flat_map["entries"][: flat_map["temporal_feature_count"]]
    )

    return LoadedModels(
        recovery_bundle=recovery_bundle, recovery_lineage=recovery_lineage,
        icu_model=icu_model, support_bundle=support_bundle, support_lineage=support_lineage,
        calibrator=calibrator, threshold=threshold_value,
        flat_map=flat_map, flat_map_sha256=flat_map_sha256,
        temporal_feature_names=temporal_feature_names, feature_schema_version=feature_schema_version,
        manifest=manifest, manifest_sha256=manifest_sha256,
        g3_marker=marker, g3_sha256=sha256_file(marker_path),
    )


@dataclass(frozen=True)
class RowSliceMetadata:
    stay_id: str
    prediction_time: str
    grid_index: int
    baseline_sofa: object
    observation_density: float
    cardiac_condition_group: object


def _observation_density(row: Mapping[str, object]) -> float:
    mask = row["observation_mask"]
    padding = row["padding_mask"]
    total = 0
    observed = 0
    for bin_index, bin_values in enumerate(mask):
        if padding[bin_index]:
            continue
        total += len(bin_values)
        observed += sum(1 for value in bin_values if value)
    return observed / total if total else float("nan")


def load_and_infer(root: Path, *, partition: str) -> Mapping[str, object]:
    """Materialize one partition and run the three frozen models exactly once.

    ``partition`` is caller-controlled so this can be exercised against
    ``validation`` for development without ever touching ``test``; the
    guarded Stage-5 entrypoint always calls it with ``partition="test"``.
    """

    root = root.resolve()
    models = load_frozen_models(root)
    rows = materialize_partition_rows(root, partition, json.loads((root / PREPROCESSOR_PATH).read_text(encoding="utf-8")))

    # baseline_sofa for slicing, joined from the pre-split package.
    assignments = _split_assignments(root)
    pre_split_baseline = {
        (row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"]): row["baseline_sofa"]
        for row in load_jsonl(root / PRE_SPLIT_PATH)
        if assignments.get(str(row["subject_id"])) == partition
    }
    statics_by_stay = {
        row["stay_id"]: row
        for row in load_jsonl(root / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl")
    }

    recovery24, recovery48, support_raw, icu_log = [], [], [], []
    slice_meta_by_key: dict = {}
    for row in rows:
        views = model_information_views(row, models.flat_map)
        flat = views["xgboost"].reshape(1, -1)
        structured = views["gru"]
        key = (row["subject_id"], row["stay_id"], row["prediction_time"], row["grid_index"])
        density = _observation_density(row)
        slice_meta_by_key[key] = RowSliceMetadata(
            stay_id=row["stay_id"], prediction_time=row["prediction_time"], grid_index=int(row["grid_index"]),
            baseline_sofa=pre_split_baseline.get(key), observation_density=density,
            cardiac_condition_group=statics_by_stay.get(row["stay_id"], {}).get("cardiac_condition_group"),
        )

        recovery_pred = models.recovery_bundle.predict(flat)[0]
        recovery24.append(
            PredictionRecord(
                stay_id=row["stay_id"], prediction_time=row["prediction_time"],
                target=float(row["delta_sofa_24"]) if row["recovery24_eligible"] else 0.0,
                prediction=float(recovery_pred[0]), eligible=bool(row["recovery24_eligible"]),
            )
        )
        recovery48.append(
            PredictionRecord(
                stay_id=row["stay_id"], prediction_time=row["prediction_time"],
                target=float(row["delta_sofa_48"]) if row["recovery48_eligible"] else 0.0,
                prediction=float(recovery_pred[1]), eligible=bool(row["recovery48_eligible"]),
            )
        )

        support_pred = float(models.support_bundle.predict(flat)[0])
        support_raw.append(
            PredictionRecord(
                stay_id=row["stay_id"], prediction_time=row["prediction_time"],
                target=float(row["organ_support_label"]) if row["organ_support_eligible"] else 0.0,
                prediction=support_pred, eligible=bool(row["organ_support_eligible"]),
            )
        )

        batch = _canonical_batch(
            structured, feature_names=models.temporal_feature_names,
            feature_schema_version=models.feature_schema_version, prediction_time=row["prediction_time"],
        )
        with torch.no_grad():
            icu_raw = float(models.icu_model(batch).reshape(()).item())
        icu_log.append(
            ICULogPredictionRecord(
                stay_id=row["stay_id"], prediction_time=row["prediction_time"],
                target_hours=float(math.expm1(row["icu_time_log1p"])) if row["icu_time_eligible"] else 0.0,
                predicted_log1p_hours=icu_raw, eligible=bool(row["icu_time_eligible"]),
            )
        )

    recovery24_by_key = {(item.stay_id, item.prediction_time): item for item in recovery24}
    recovery48_by_key = {(item.stay_id, item.prediction_time): item for item in recovery48}
    cc24, cc48 = compute_complete_component_records(
        root, rows, partition, recovery24_by_key, recovery48_by_key
    )

    naive, naive_path, naive_sha256 = load_or_fit_naive_baselines(
        root, artifact_ref=NAIVE_BASELINE_PATH, train_ref=TRAIN_PATH
    )
    naive_recovery24 = tuple(replace(item, prediction=naive.recovery24_constant_delta) for item in recovery24)
    naive_recovery48 = tuple(replace(item, prediction=naive.recovery48_constant_delta) for item in recovery48)
    naive_icu_log = tuple(replace(item, predicted_log1p_hours=naive.icu_time_log1p_median) for item in icu_log)
    naive_support = tuple(replace(item, prediction=naive.organ_support_prevalence) for item in support_raw)

    return {
        "partition": partition,
        "models": models,
        "rows": rows,
        "slice_meta_by_key": slice_meta_by_key,
        "bundle": FinalPredictionBundle(
            recovery24=tuple(recovery24), recovery48=tuple(recovery48),
            icu_time_log=tuple(icu_log), support_raw=tuple(support_raw),
        ),
        "naive_bundle": FinalPredictionBundle(
            recovery24=naive_recovery24, recovery48=naive_recovery48,
            icu_time_log=naive_icu_log, support_raw=naive_support,
        ),
        "naive_artifact": naive,
        "naive_artifact_path": naive_path,
        "naive_artifact_sha256": naive_sha256,
        "n_rows": len(rows),
        "complete_component_24": cc24,
        "complete_component_48": cc48,
    }


@dataclass(frozen=True)
class SofaScoringContext:
    provider: SyntheticCurrentSOFAProvider
    history_by_stay: Mapping[object, tuple]
    ventilation_by_stay: Mapping[object, tuple]
    vasoactive_by_stay: Mapping[object, tuple]
    event_dictionary: object
    dose_contract_sha256: str


def build_sofa_scoring_context(root: Path) -> SofaScoringContext:
    """Exact reconstruction of serving.real.bundle's SOFA provider assembly,
    without the test-partition history guard: this module is itself the one
    authorized place that may score sealed-test-partition cutoffs.

    Also pre-groups history/ventilation/vasoactive-exposure rows by stay_id:
    ``sofa_at``/``query_invasive_ventilation_state`` each scan their full
    input per call (built for one-off verification, not a full-partition
    sweep), and pre-filtering the input list per stay is mathematically
    identical to letting them scan the whole dataset every time (both
    already discard non-matching stay_id rows internally) but tractable at
    partition scale.
    """

    timeline_manifest_path = root / TIMELINE_MANIFEST_PATH
    raw_manifest = _json(root, RAW_DATASET_MANIFEST_PATH)
    support_path = root / next(
        item["repository_relative_path"] for item in raw_manifest["artifacts"] if item["logical_name"] == "support_intervals"
    )
    supports = load_jsonl(support_path)
    dictionary = load_synthetic_event_dictionary(root / EVENT_DICT_PATH)
    _vaso, vent, exposures = _normalized(supports, dictionary)
    dose_contract_sha256 = sha256_file(root / SUPPORT_PROCESS_PATH)
    full_support_provider = PulkitStateSOFASupportProvider(
        event_dictionary=dictionary, execution_mode=ExecutionMode.SYNTHETIC,
        ventilation_intervals=vent, vasoactive_exposures=exposures,
        vasoactive_dose_contract_version="synthetic_support_process_v1",
        vasoactive_dose_contract_sha256=dose_contract_sha256,
        vasoactive_coverage_known=True,
    )
    provider = SyntheticCurrentSOFAProvider(
        manifest_path=timeline_manifest_path, root=root, sofa_spec_path=root / SOFA_SPEC_PATH,
        support_provider=full_support_provider,
    )
    history_by_stay: dict = {}
    for event in provider.history:
        history_by_stay.setdefault(event.get("stay_id"), []).append(event)
    ventilation_by_stay: dict = {}
    for interval in vent:
        ventilation_by_stay.setdefault(interval.stay_id, []).append(interval)
    vasoactive_by_stay: dict = {}
    for exposure in exposures:
        vasoactive_by_stay.setdefault(exposure.stay_id, []).append(exposure)
    return SofaScoringContext(
        provider=provider,
        history_by_stay={key: tuple(value) for key, value in history_by_stay.items()},
        ventilation_by_stay={key: tuple(value) for key, value in ventilation_by_stay.items()},
        vasoactive_by_stay={key: tuple(value) for key, value in vasoactive_by_stay.items()},
        event_dictionary=dictionary, dose_contract_sha256=dose_contract_sha256,
    )


def _stay_support_provider(context: SofaScoringContext, stay_id: object) -> PulkitStateSOFASupportProvider:
    return PulkitStateSOFASupportProvider(
        event_dictionary=context.event_dictionary, execution_mode=ExecutionMode.SYNTHETIC,
        ventilation_intervals=context.ventilation_by_stay.get(stay_id, ()),
        vasoactive_exposures=context.vasoactive_by_stay.get(stay_id, ()),
        vasoactive_dose_contract_version="synthetic_support_process_v1",
        vasoactive_dose_contract_sha256=context.dose_contract_sha256,
        vasoactive_coverage_known=True,
    )


def _z(instant: datetime) -> str:
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def compute_complete_component_records(
    root: Path,
    rows: Sequence[Mapping[str, object]],
    partition: str,
    recovery24_by_key: Mapping[tuple, "PredictionRecord"],
    recovery48_by_key: Mapping[tuple, "PredictionRecord"],
) -> Tuple[Tuple[CompleteComponentRecord, ...], Tuple[CompleteComponentRecord, ...]]:
    """Baseline (t) and future (t+24h / t+48h) SOFA component-observed flags
    for every recovery-eligible row, via the frozen, unmodified sofa_at()
    scorer only — no model retraining, purely a deterministic recomputation
    from already accepted raw events. Reuses the exact already-computed
    model predictions (never recomputed here) for each row's target/
    prediction pair.

    ``sofa_at`` itself scans its full ``history`` argument per call; calling
    it once per stay across the whole multi-subject timeline (as
    ``SyntheticCurrentSOFAProvider.score`` does) is O(rows x total_events)
    and too slow for a full-partition sweep. Pre-grouping history by stay_id
    once here and passing sofa_at() only that stay's rows is mathematically
    identical (sofa_at itself already discards non-matching stay_id rows)
    and turns this into a per-partition-tractable computation.
    """

    from data.synthetic.sofa import sofa_at

    context = build_sofa_scoring_context(root)
    provider = context.provider
    stay_support_cache: dict = {}

    def support_provider_for(stay_id):
        if stay_id not in stay_support_cache:
            stay_support_cache[stay_id] = _stay_support_provider(context, stay_id)
        return stay_support_cache[stay_id]

    cc24, cc48 = [], []
    for row in rows:
        if not (row["recovery24_eligible"] or row["recovery48_eligible"]):
            continue
        stay_id = row["stay_id"]
        stay_record = provider.stays[stay_id]
        stay_history = context.history_by_stay.get(stay_id, ())
        stay_support = support_provider_for(stay_id)
        key = (row["stay_id"], row["prediction_time"])
        t = datetime.fromisoformat(str(row["prediction_time"]).replace("Z", "+00:00"))
        try:
            baseline = sofa_at(
                stay_history, stay_record, _z(t),
                spec=provider.spec, bindings=provider.bindings, support_provider=stay_support,
            )
        except Exception:
            continue
        if row["recovery24_eligible"]:
            try:
                future = sofa_at(
                    stay_history, stay_record, _z(t + timedelta(hours=24)),
                    spec=provider.spec, bindings=provider.bindings, support_provider=stay_support,
                )
                source = recovery24_by_key[key]
                cc24.append(
                    CompleteComponentRecord(
                        stay_id=stay_id, prediction_time=row["prediction_time"], partition=partition,
                        target=source.target, prediction=source.prediction, eligible=True,
                        baseline_component_observed=baseline.component_observed,
                        future_component_observed=future.component_observed,
                    )
                )
            except Exception:
                pass
        if row["recovery48_eligible"]:
            try:
                future = sofa_at(
                    stay_history, stay_record, _z(t + timedelta(hours=48)),
                    spec=provider.spec, bindings=provider.bindings, support_provider=stay_support,
                )
                source = recovery48_by_key[key]
                cc48.append(
                    CompleteComponentRecord(
                        stay_id=stay_id, prediction_time=row["prediction_time"], partition=partition,
                        target=source.target, prediction=source.prediction, eligible=True,
                        baseline_component_observed=baseline.component_observed,
                        future_component_observed=future.component_observed,
                    )
                )
            except Exception:
                pass
    return tuple(cc24), tuple(cc48)


ERROR_ANALYSIS_CONFIG_PATH = "configs/error_analysis/error_analysis_v1.json"
ACTIVE_SLICE_NAMES = (
    "baseline_sofa", "remaining_icu_duration", "observation_density",
    "cutoff_timing", "cardiac_subtype", "complete_component_sofa",
)


def build_slice_specification(root: Path) -> Tuple[SliceSpecification, str]:
    """Convert the frozen, human-authored slice config into the typed
    ``evaluation.slices`` dataclasses it validates against. Only slices
    explicitly marked FROZEN are included; medical_surgical and
    baseline_support stay NOT_REPORTED (no authoritative frozen mapping)."""

    config_path = root / ERROR_ANALYSIS_CONFIG_PATH
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_FOR_FINAL_TEST":
        raise Stage5Error("error-analysis slice spec is not frozen for final test")
    applicability = payload["task_slice_applicability"]
    definitions = []
    for item in payload["slices"]:
        if item["slice_name"] not in ACTIVE_SLICE_NAMES or item["status"] != "FROZEN":
            continue
        groups = tuple(
            SliceGroup(
                name=group["name"], lower=group.get("lower"), upper=group.get("upper"),
                categories=tuple(group.get("categories", ())),
            )
            for group in item["groups"]
        )
        definitions.append(
            SliceDefinition(
                slice_name=item["slice_name"], source_variable=item["source_variable"],
                allowed_tasks=tuple(applicability[item["slice_name"]]),
                groups=groups, missing_value_policy=item["missing_value_policy"],
                missing_group=item.get("missing_group"), mapping_version=item["mapping_version"],
                provenance=item["provenance"], status="FROZEN",
                assignment_kind=item.get("assignment_kind", "numeric"),
            )
        )
    spec = SliceSpecification(
        spec_version=payload["spec_version"], slices=tuple(definitions),
        final_test_authorized=True, minimum_sample_size=payload["minimum_sample_size"], mode="real",
    )
    return spec, sha256_file(config_path)


def _slice_metadata_for(meta: RowSliceMetadata) -> Mapping[str, object]:
    return {
        "baseline_sofa_at_t": meta.baseline_sofa,
        "remaining_current_icu_hours_at_t": None,  # filled per-row only for icu-eligible rows below
        "canonical_observation_density": meta.observation_density if math.isfinite(meta.observation_density) else None,
        "canonical_cutoff_index": meta.grid_index,
        "frozen_cardiac_condition_group": meta.cardiac_condition_group,
    }


def build_error_analysis_records(
    loaded: Mapping[str, object],
    manifest_bindings: Mapping[str, Mapping[str, object]],
) -> Tuple[ErrorAnalysisRecord, ...]:
    """One ErrorAnalysisRecord per (task[, horizon]) prediction row, carrying
    only frozen pre-test slice metadata — never a derived error/residual."""

    partition = loaded["partition"]
    slice_meta_by_key = loaded["slice_meta_by_key"]
    bundle: FinalPredictionBundle = loaded["bundle"]
    cc24_by_key = {(item.stay_id, item.prediction_time): item for item in loaded["complete_component_24"]}
    cc48_by_key = {(item.stay_id, item.prediction_time): item for item in loaded["complete_component_48"]}
    records = []

    meta_by_stay_time = {
        (meta.stay_id, meta.prediction_time): meta for meta in slice_meta_by_key.values()
    }

    def base_slice_metadata(stay_id, prediction_time, *, icu_hours=None):
        meta = meta_by_stay_time.get((stay_id, prediction_time))
        if meta is None:
            return {}
        payload = _slice_metadata_for(meta)
        if icu_hours is not None:
            payload["remaining_current_icu_hours_at_t"] = icu_hours
        return payload

    recovery_binding = manifest_bindings["recovery"]
    for horizon, items, cc_by_key in (("24h", bundle.recovery24, cc24_by_key), ("48h", bundle.recovery48, cc48_by_key)):
        for item in items:
            records.append(
                ErrorAnalysisRecord(
                    row_id=str((item.stay_id, item.prediction_time, horizon)),
                    stay_id=item.stay_id, prediction_time=item.prediction_time, split=partition,
                    task="recovery", horizon=horizon, target=item.target, prediction=item.prediction,
                    eligible=item.eligible, model_run_id=recovery_binding["model_run_id"],
                    model_hash=recovery_binding["model_hash"], feature_version=recovery_binding["feature_version"],
                    label_version=recovery_binding["label_version"], split_hash=recovery_binding["split_hash"],
                    slice_metadata=base_slice_metadata(item.stay_id, item.prediction_time),
                    complete_component_record=cc_by_key.get((item.stay_id, item.prediction_time)),
                )
            )

    icu_binding = manifest_bindings["icu_stay_time"]
    for raw, processed in zip(
        bundle.icu_time_log,
        process_icu_time_predictions(bundle.icu_time_log) if bundle.icu_time_log else (),
    ):
        records.append(
            ErrorAnalysisRecord(
                row_id=str((raw.stay_id, raw.prediction_time, "icu")),
                stay_id=raw.stay_id, prediction_time=raw.prediction_time, split=partition,
                task="icu_stay_time", horizon=None, target=processed.target, prediction=processed.prediction,
                eligible=raw.eligible, model_run_id=icu_binding["model_run_id"], model_hash=icu_binding["model_hash"],
                feature_version=icu_binding["feature_version"], label_version=icu_binding["label_version"],
                split_hash=icu_binding["split_hash"],
                slice_metadata=base_slice_metadata(raw.stay_id, raw.prediction_time, icu_hours=processed.target),
            )
        )

    support_binding = manifest_bindings["organ_support"]
    calibrated, _ = None, None
    for item in bundle.support_raw:
        records.append(
            ErrorAnalysisRecord(
                row_id=str((item.stay_id, item.prediction_time, "support")),
                stay_id=item.stay_id, prediction_time=item.prediction_time, split=partition,
                task="organ_support", horizon=None, target=item.target, prediction=item.prediction,
                eligible=item.eligible, model_run_id=support_binding["model_run_id"],
                model_hash=support_binding["model_hash"], feature_version=support_binding["feature_version"],
                label_version=support_binding["label_version"], split_hash=support_binding["split_hash"],
                slice_metadata=base_slice_metadata(item.stay_id, item.prediction_time),
                probability_type="raw",
            )
        )
    return tuple(records)


def _to_jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_to_jsonable(payload)))
    return sha256_file(path)


def build_manifest_bindings(models: LoadedModels) -> Mapping[str, Mapping[str, str]]:
    tasks = models.manifest["tasks"]
    bindings = {}
    for task, family in (("recovery", "xgboost"), ("icu_stay_time", "gru"), ("organ_support", "xgboost")):
        entry = tasks[task]
        bindings[task] = {
            "model_run_id": entry["run_id"],
            "model_hash": entry["artifact_sha256"],
            "model_family": family,
            "feature_version": entry["feature_version"],
            "label_version": entry["label_version"],
            "split_hash": entry["split_hash"],
        }
    return bindings


FINAL_TEST_OUTPUT_ROOT = "artifacts/final_test"


def evaluate_naive_bundle(naive_bundle: FinalPredictionBundle):
    icu_processed = process_icu_time_predictions(naive_bundle.icu_time_log)
    return {
        "recovery": {
            "24h": evaluate_recovery_horizon(naive_bundle.recovery24, horizon="24h"),
            "48h": evaluate_recovery_horizon(naive_bundle.recovery48, horizon="48h"),
        },
        "icu_stay_time": evaluate_icu_time(icu_processed),
        "organ_support": evaluate_organ_support(naive_bundle.support_raw, probability_type="raw"),
        "_icu_processed": icu_processed,
    }


def bootstrap_naive_bundle(naive_bundle: FinalPredictionBundle, icu_processed, *, n_bootstrap: int, seed: int):
    return {
        "recovery": bootstrap_recovery(naive_bundle.recovery24, naive_bundle.recovery48, n_bootstrap=n_bootstrap, seed=seed),
        "icu_stay_time": bootstrap_icu_time(icu_processed, n_bootstrap=n_bootstrap, seed=seed),
        "organ_support": bootstrap_organ_support(
            naive_bundle.support_raw, n_bootstrap=n_bootstrap, seed=seed, probability_type="raw"
        ),
    }


def stage5_evaluate(loaded: Mapping[str, object], *, root: Path = ROOT, n_bootstrap: int = 2000, seed: int = 20260921) -> Mapping[str, object]:
    """Pure downstream analysis on already-obtained frozen predictions: no
    further test-partition access of any kind occurs below this point."""

    root = root.resolve()
    partition = loaded["partition"]
    models: LoadedModels = loaded["models"]
    bundle: FinalPredictionBundle = loaded["bundle"]
    naive_bundle: FinalPredictionBundle = loaded["naive_bundle"]
    threshold_identifier = "support_threshold_v1"

    processed = evaluate_frozen_predictions(
        bundle, support_calibrator=models.calibrator, support_threshold=models.threshold,
        threshold_identifier=threshold_identifier,
    )
    naive_processed = evaluate_naive_bundle(naive_bundle)
    icu_naive_processed = naive_processed.pop("_icu_processed")

    bootstrap_results = bootstrap_frozen_predictions(
        bundle, processed, n_bootstrap=n_bootstrap, seed=seed, threshold_identifier=threshold_identifier,
    )
    naive_bootstrap = bootstrap_naive_bundle(naive_bundle, icu_naive_processed, n_bootstrap=n_bootstrap, seed=seed)

    if partition == "test":
        sensitivity = evaluate_test_complete_component(loaded["complete_component_24"], loaded["complete_component_48"])
    else:
        sensitivity = evaluate_complete_component_sensitivity(
            loaded["complete_component_24"], loaded["complete_component_48"], expected_partition=partition
        )

    spec, spec_hash = build_slice_specification(root)
    bindings = build_manifest_bindings(models)
    error_records = build_error_analysis_records(loaded, bindings)
    error_rows: list = []
    for task in ("recovery", "icu_stay_time", "organ_support"):
        task_records = tuple(item for item in error_records if item.task == task)
        binding = SelectedModelBinding(
            task=task, model_family=bindings[task]["model_family"], model_run_id=bindings[task]["model_run_id"],
            model_hash=bindings[task]["model_hash"], feature_version=bindings[task]["feature_version"],
            label_version=bindings[task]["label_version"], split_hash=bindings[task]["split_hash"], selected=True,
            probability_type="raw" if task == "organ_support" else None,
        )
        if partition == "test":
            error_rows.extend(evaluate_prespecified_test_error_analysis(task_records, spec, binding))
        else:
            error_rows.extend(evaluate_error_analysis(task_records, spec, binding, allow_test=False))

    return {
        "partition": partition,
        "processed": processed,
        "naive_processed": naive_processed,
        "bootstrap": bootstrap_results,
        "naive_bootstrap": naive_bootstrap,
        "sensitivity": sensitivity,
        "error_rows": tuple(error_rows),
        "slice_spec_hash": spec_hash,
        "bindings": bindings,
    }


def _row_identifiers(rows: Sequence[Mapping[str, object]]) -> Mapping[tuple, Mapping[str, object]]:
    return {
        (row["stay_id"], row["prediction_time"]): {
            "subject_id": row["subject_id"], "stay_id": row["stay_id"],
            "prediction_time": row["prediction_time"], "grid_index": int(row["grid_index"]),
        }
        for row in rows
    }


def persist_predictions(loaded: Mapping[str, object], output_dir: Path) -> Mapping[str, Tuple[Path, str]]:
    ids = _row_identifiers(loaded["rows"])
    bundle: FinalPredictionBundle = loaded["bundle"]
    calibrated, alerts = apply_frozen_support_postprocessing(
        bundle.support_raw, loaded["models"].calibrator, threshold=loaded["models"].threshold
    )
    calibrated_by_key = {(item.stay_id, item.prediction_time): item for item in calibrated}
    alerts_by_key = {
        (item.stay_id, item.prediction_time): alert for item, alert in zip(bundle.support_raw, alerts)
    }
    r24_by_key = {(item.stay_id, item.prediction_time): item for item in bundle.recovery24}
    r48_by_key = {(item.stay_id, item.prediction_time): item for item in bundle.recovery48}

    # Registered separately per horizon (recovery24_v1.json / recovery48_v1.json):
    # the registry's "prediction" artifact class requires exactly one
    # model_checkpoint parent, and each horizon depends on its own frozen
    # booster (recovery_24h.json / recovery_48h.json).
    recovery24_rows, recovery48_rows = [], []
    for key, ident in ids.items():
        r24 = r24_by_key.get(key)
        r48 = r48_by_key.get(key)
        if r24 is not None:
            recovery24_rows.append({
                **ident, "delta_sofa_24_target": r24.target, "delta_sofa_24_predicted": r24.prediction,
                "recovery24_eligible": bool(r24.eligible),
            })
        if r48 is not None:
            recovery48_rows.append({
                **ident, "delta_sofa_48_target": r48.target, "delta_sofa_48_predicted": r48.prediction,
                "recovery48_eligible": bool(r48.eligible),
            })
    icu_rows = []
    for item in bundle.icu_time_log:
        ident = ids.get((item.stay_id, item.prediction_time), {})
        hours = float(remaining_icu_hours_from_log_prediction(torch.tensor(item.predicted_log1p_hours, dtype=torch.float64)).item())
        icu_rows.append({
            **ident, "target_hours": item.target_hours if item.eligible else None,
            "predicted_log1p_hours": item.predicted_log1p_hours, "predicted_hours": hours, "eligible": bool(item.eligible),
        })
    support_rows = []
    for item in bundle.support_raw:
        key = (item.stay_id, item.prediction_time)
        ident = ids.get(key, {})
        cal = calibrated_by_key.get(key)
        support_rows.append({
            **ident, "label": item.target if item.eligible else None, "raw_probability": item.prediction,
            "calibrated_probability": cal.prediction if cal else None,
            "alert_state": bool(alerts_by_key.get(key)) if key in alerts_by_key else None,
            "eligible": bool(item.eligible),
        })

    lineage_header = {
        "partition": loaded["partition"], "manifest_sha256": loaded["models"].manifest_sha256,
        "g3_sha256": loaded["models"].g3_sha256,
        "split_hash": loaded["models"].manifest["tasks"]["recovery"]["split_hash"],
    }
    written = {}
    for name, rows in (
        ("recovery24", recovery24_rows), ("recovery48", recovery48_rows),
        ("icu", icu_rows), ("support", support_rows),
    ):
        path = output_dir / "predictions" / (name + "_v1.json")
        digest = _write_json(path, {"schema": "stage5_final_test_prediction_v1", **lineage_header, "rows": rows})
        written[name] = (path, digest)
    return written


def persist_metrics(loaded, evaluated, output_dir: Path) -> Tuple[Path, str]:
    processed = evaluated["processed"]
    naive = evaluated["naive_processed"]
    payload = {
        "schema": "stage5_final_test_metrics_v1",
        "partition": evaluated["partition"],
        "evaluator_version": "phase9_final_evaluator_v1",
        "metric_implementation_version": "stay_balanced_metrics_v1",
        "selected": {
            "recovery": {horizon: asdict(result) for horizon, result in processed.recovery.items()},
            "icu_stay_time": asdict(processed.icu_time),
            "organ_support_raw": asdict(processed.support_raw),
            "organ_support_calibrated": asdict(processed.support_calibrated),
        },
        "naive_baseline": {
            "recovery": {horizon: asdict(result) for horizon, result in naive["recovery"].items()},
            "icu_stay_time": asdict(naive["icu_stay_time"]),
            "organ_support_raw": asdict(naive["organ_support"]),
        },
        "naive_baseline_artifact": {
            "ref": NAIVE_BASELINE_PATH, "sha256": loaded["naive_artifact_sha256"],
        },
    }
    path = output_dir / "metrics" / "metrics_v1.json"
    digest = _write_json(path, payload)
    return path, digest


def persist_bootstrap(evaluated, output_dir: Path) -> Tuple[Path, str]:
    payload = {
        "schema": "stage5_final_test_bootstrap_v1",
        "partition": evaluated["partition"],
        "n_bootstrap": 2000, "seed": 20260921, "ci_level": 0.95,
        "percentile_convention": "linear_type7_index_equals_q_times_n_minus_one_v1",
        "selected": {
            "recovery": {
                horizon: {name: asdict(result) for name, result in metrics.items()}
                for horizon, metrics in evaluated["bootstrap"]["recovery"].items()
            },
            "icu_stay_time": {name: asdict(result) for name, result in evaluated["bootstrap"]["icu_stay_time"].items()},
            "organ_support": {name: asdict(result) for name, result in evaluated["bootstrap"]["organ_support"].items()},
        },
        "naive_baseline": {
            "recovery": {
                horizon: {name: asdict(result) for name, result in metrics.items()}
                for horizon, metrics in evaluated["naive_bootstrap"]["recovery"].items()
            },
            "icu_stay_time": {name: asdict(result) for name, result in evaluated["naive_bootstrap"]["icu_stay_time"].items()},
            "organ_support": {name: asdict(result) for name, result in evaluated["naive_bootstrap"]["organ_support"].items()},
        },
    }
    path = output_dir / "bootstrap" / "bootstrap_v1.json"
    digest = _write_json(path, payload)
    return path, digest


def persist_sensitivity(evaluated, output_dir: Path) -> Tuple[Path, str]:
    sensitivity = evaluated["sensitivity"]
    payload = {
        "schema": "stage5_complete_component_sensitivity_v1",
        "partition": evaluated["partition"],
        "analysis_type": sensitivity.analysis_type,
        "test_accessed": sensitivity.test_accessed,
        "recovery24": asdict(sensitivity.recovery24),
        "recovery48": asdict(sensitivity.recovery48),
    }
    path = output_dir / "sensitivity" / "complete_component_v1.json"
    digest = _write_json(path, payload)
    return path, digest


def persist_error_analysis(evaluated, output_dir: Path) -> Mapping[str, Tuple[Path, str]]:
    rows = evaluated["error_rows"]
    csv_path = output_dir / "error_analysis" / "error_analysis_v1.csv"
    write_error_analysis_table(rows, csv_path)
    csv_sha256 = sha256_file(csv_path)
    json_path = output_dir / "error_analysis" / "error_analysis_v1.json"
    json_sha256 = _write_json(
        json_path,
        {
            "schema": "stage5_error_analysis_summary_v1", "partition": evaluated["partition"],
            "slice_spec_hash": evaluated["slice_spec_hash"], "row_count": len(rows),
            "rows": [asdict(row) for row in rows],
        },
    )
    return {"csv": (csv_path, csv_sha256), "json": (json_path, json_sha256)}


def build_comparison_table(evaluated) -> Tuple[Mapping[str, object], ...]:
    processed = evaluated["processed"]
    naive = evaluated["naive_processed"]
    bootstrap = evaluated["bootstrap"]
    naive_bootstrap = evaluated["naive_bootstrap"]

    def ci(entry):
        return [entry.ci_lower, entry.ci_upper] if entry else [None, None]

    rows = []
    for horizon in ("24h", "48h"):
        rows.append({
            "task": "recovery_" + horizon, "naive_baseline": naive["recovery"][horizon].metrics["mae"],
            "selected_family": "xgboost", "selected_candidate": "xgb-recovery-014",
            "primary_metric": "mae", "primary_point_estimate": processed.recovery[horizon].metrics["mae"],
            "ci_95": ci(bootstrap["recovery"][horizon]["mae"]),
            "naive_ci_95": ci(naive_bootstrap["recovery"][horizon]["mae"]),
        })
    rows.append({
        "task": "icu_stay_time", "naive_baseline": naive["icu_stay_time"].metrics["median_absolute_error"],
        "selected_family": "gru", "selected_candidate": "gru-icu-time-026",
        "primary_metric": "median_absolute_error",
        "primary_point_estimate": processed.icu_time.metrics["median_absolute_error"],
        "ci_95": ci(bootstrap["icu_stay_time"]["median_absolute_error"]),
        "naive_ci_95": ci(naive_bootstrap["icu_stay_time"]["median_absolute_error"]),
    })
    rows.append({
        "task": "organ_support", "naive_baseline": naive["organ_support"].metrics["auprc"],
        "selected_family": "xgboost", "selected_candidate": "xgb-support-024",
        "primary_metric": "auprc_raw", "primary_point_estimate": processed.support_raw.metrics["auprc"],
        "ci_95": ci(bootstrap["organ_support"]["auprc"]),
        "naive_ci_95": ci(naive_bootstrap["organ_support"]["auprc"]),
    })
    return tuple(rows)


MODEL_ARTIFACT_IDS = {
    "recovery24": "phase12-xgb-recovery-014-attempt-1:model:recovery24",
    "recovery48": "phase12-xgb-recovery-014-attempt-1:model:recovery48",
    "icu": "final-v2-gru-icu-time-026-attempt-1:model",
    "support": "phase12-xgb-support-024-attempt-1:model:organ_support",
}
MANIFEST_ARTIFACT_ID = "stage3-selected-models-manifest"
PREPROCESSOR_ARTIFACT_ID = "stage5-registered-synthetic-feature-preprocessor-v1"
STAGE5_RUN_ID = "stage5-final-test-evaluation-v1"


def _population_hash(rows: Sequence[Mapping[str, object]]) -> str:
    return canonical_sha256(sorted([str(row.get("stay_id")), str(row.get("prediction_time"))] for row in rows))


def register_stage5_results(
    root: Path,
    loaded: Mapping[str, object],
    evaluated: Mapping[str, object],
    prediction_paths: Mapping[str, Tuple[Path, str]],
    metric_path: Tuple[Path, str],
    error_analysis_paths: Mapping[str, Tuple[Path, str]],
    naive_path: Path,
    naive_sha256: str,
    *,
    code_commit: str,
) -> Tuple[str, ...]:
    """Register every new final-test artifact with exact lineage. Runs only
    after every G3 check in this process has already completed — registering
    here legitimately extends experiments/artifacts.csv beyond what G3's own
    bound hash of that file recorded at Stage-3 freeze time (expected; see
    the Stage-5 review for why that is not G3 corruption)."""

    models: LoadedModels = loaded["models"]
    bindings = evaluated["bindings"]
    split_hash = bindings["recovery"]["split_hash"]
    index_path = root / "experiments/artifacts.csv"
    registry_rows = read_run_registry(root / "experiments/registry.csv")
    existing_ids = {item.artifact_id for item in read_artifact_index(index_path)}
    registered = []

    def _register(record: ArtifactRecord) -> None:
        if record.artifact_id in existing_ids:
            return
        register_artifact(index_path, record, registry_rows, root)
        existing_ids.add(record.artifact_id)
        registered.append(record.artifact_id)

    if MANIFEST_ARTIFACT_ID not in existing_ids:
        raise Stage5Error("selected-models manifest is not registered; cannot bind lineage")

    _register(ArtifactRecord(
        artifact_id="stage5-naive-baseline-v1", artifact_path=str(Path(NAIVE_BASELINE_PATH)),
        artifact_type="naive_baseline", artifact_version="naive_baseline_v1", artifact_sha256=naive_sha256,
        producing_run_id="", generating_script="src/evaluation/naive_baseline.py", status="registered",
    ))
    # The real, G3-frozen synthetic_feature_preprocessor_v1.json was never
    # itself registered with artifact_type="preprocessor" (only Phase-5/7
    # smoke-fixture scalers were); validate_final_result_traceability
    # requires exactly that type for its preprocessor binding.
    _register(ArtifactRecord(
        artifact_id=PREPROCESSOR_ARTIFACT_ID, artifact_path=PREPROCESSOR_PATH,
        artifact_type="preprocessor", artifact_version="synthetic_feature_preprocessor_v1",
        artifact_sha256=models.manifest["tasks"]["recovery"]["preprocessor_sha256"],
        # This artifact was created during (pre-registered) Phase-10 fitting;
        # Stage 5 is simply the first place it needed a formal registry row
        # (as opposed to being embedded-by-hash in bundle metadata). Binding
        # its producing_run_id to the Stage-5 run honestly records when it
        # was registered, not when it was originally created.
        producing_run_id=STAGE5_RUN_ID, parent_artifact_ids=MANIFEST_ARTIFACT_ID,
        preprocessor_sha256=models.manifest["tasks"]["recovery"]["preprocessor_sha256"],
        generating_script="src/preprocess/synthetic_phase10.py", status="registered",
    ))

    task_by_name = {
        "recovery24": ("recovery", "xgboost"), "recovery48": ("recovery", "xgboost"),
        "icu": ("icu_stay_time", "gru"), "support": ("organ_support", "xgboost"),
    }
    model_hash_by_name = {
        "recovery24": models.manifest["tasks"]["recovery"]["model_artifacts"]["recovery24"]["sha256"],
        "recovery48": models.manifest["tasks"]["recovery"]["model_artifacts"]["recovery48"]["sha256"],
        "icu": models.manifest["tasks"]["icu_stay_time"]["artifact_sha256"],
        "support": models.manifest["tasks"]["organ_support"]["artifact_sha256"],
    }
    prediction_ids = {}
    for name, (path, digest) in prediction_paths.items():
        task, family = task_by_name[name]
        model_artifact_id = MODEL_ARTIFACT_IDS[name]
        binding = bindings[task]
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        artifact_id = "stage5-final-test-prediction-" + name + "-v1"
        prediction_ids[name] = artifact_id
        _register(ArtifactRecord(
            artifact_id=artifact_id, artifact_path=str(path.relative_to(root)) if path.is_absolute() else str(path),
            artifact_type="prediction", artifact_version="stage5_final_test_prediction_v1", artifact_sha256=digest,
            producing_run_id=STAGE5_RUN_ID, parent_artifact_ids=";".join((model_artifact_id, MANIFEST_ARTIFACT_ID)),
            task=task, model_family=family, split_hash=split_hash, feature_version=binding["feature_version"],
            label_version=binding["label_version"], model_sha256=model_hash_by_name[name],
            creation_commit=code_commit, partition=evaluated["partition"],
            prediction_population_hash=_population_hash(rows), run_type="scientific", status="registered",
        ))

    _register(ArtifactRecord(
        artifact_id="stage5-final-test-metrics-v1", artifact_path=str(metric_path[0].relative_to(root)),
        artifact_type="metric_table", artifact_version="stage5_final_test_metrics_v1", artifact_sha256=metric_path[1],
        producing_run_id=STAGE5_RUN_ID, parent_artifact_ids=";".join(prediction_ids.values()),
        evaluator_version="phase9_final_evaluator_v1", metric_implementation_version="stay_balanced_metrics_v1",
        partition=evaluated["partition"], creation_commit=code_commit, run_type="scientific", status="registered",
    ))
    _register(ArtifactRecord(
        artifact_id="stage5-final-test-error-analysis-v1",
        artifact_path=str(error_analysis_paths["csv"][0].relative_to(root)),
        artifact_type="error_analysis_table", artifact_version="stage5_error_analysis_v1",
        artifact_sha256=error_analysis_paths["csv"][1], producing_run_id=STAGE5_RUN_ID,
        parent_artifact_ids=";".join(prediction_ids.values()), evaluator_version="prespecified_error_analysis_v1",
        metric_implementation_version="stay_balanced_metrics_v1", slice_spec_hash=evaluated["slice_spec_hash"],
        partition=evaluated["partition"], creation_commit=code_commit, run_type="scientific", status="registered",
    ))
    return tuple(registered)


def ensure_stage5_run_registered(root: Path, *, code_commit: str, split_hash: str, preprocessor_sha256: str) -> None:
    registry_path = root / "experiments/registry.csv"
    rows = read_run_registry(registry_path)
    if any(row.get("run_id") == STAGE5_RUN_ID for row in rows):
        return
    fieldnames = list(rows[0].keys()) if rows else [
        "run_id", "timestamp_utc", "task", "model_family", "seed", "code_commit", "config_ref", "config_hash",
        "search_space_hash", "split_hash", "feature_version", "label_version", "model_artifact_ref", "model_sha256",
        "metrics_ref", "status", "parent_run_id", "notes", "candidate_id", "search_version", "candidate_list_hash",
        "attempt_number", "retry_of_run_id", "attempt_status_detail", "validation_objective", "run_type", "finalized",
        "metrics_sha256", "preprocessor_ref", "preprocessor_sha256", "event_dict_version", "parent_feature_version",
        "derived_feature_hash", "environment_ref", "environment_sha256", "dirty_worktree", "dataset_version",
        "mimic_code_commit", "extraction_ref", "extraction_sha256", "cohort_version", "feature_dictionary_ref",
        "feature_dictionary_sha256", "label_spec_ref", "split_ref",
    ]
    new_row = {name: "" for name in fieldnames}
    new_row.update({
        "run_id": STAGE5_RUN_ID, "timestamp_utc": utc_now(), "code_commit": code_commit,
        "split_hash": split_hash, "preprocessor_ref": PREPROCESSOR_PATH, "preprocessor_sha256": preprocessor_sha256,
        "split_ref": SPLIT_PATH, "status": "completed", "run_type": "scientific", "finalized": "true",
        "notes": "Stage 5 one-time final-test evaluation of the three frozen G3-selected models plus naive baselines",
    })
    with registry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(new_row)


G1_PATH = "artifacts/acceptance/g1_synthetic_data_freeze_v1.json"
PHASE14_HANDOFF_PATH = "artifacts/handoffs/sanskruti_phase14_handoff_v1.json"
STAGE2_SELECTION_PATH = "artifacts/selection/validation_family_selection_v1.json"
STAGE4_ACCEPTANCE_PATH = "artifacts/acceptance/stage4_serving_integration_freeze_v1.json"
STAGE4_BUNDLE_PATH = "artifacts/serving/final_serving_bundle_v1.json"
STAGE4_ENV_LOCK_PATH = "artifacts/serving/stage4_environment_lock_v1.txt"
STAGE5_PLAN_PATH = "artifacts/governance/stage5_final_test_plan_v1.json"
G4_PATH = "artifacts/governance/g4_test_evaluation_freeze_v1.json"
G4_VERSION = "g4_test_evaluation_freeze_v1"


def build_g4(
    root: Path,
    loaded: Mapping[str, object],
    evaluated: Mapping[str, object],
    *,
    prediction_paths: Mapping[str, Tuple[Path, str]],
    metric_path: Tuple[Path, str],
    bootstrap_path: Tuple[Path, str],
    sensitivity_path: Tuple[Path, str],
    error_analysis_paths: Mapping[str, Tuple[Path, str]],
    naive_path: Path,
    naive_sha256: str,
    plan_path: Path,
    plan_sha256: str,
    code_commit: str,
    registered_artifact_ids: Sequence[str],
) -> Tuple[Path, str]:
    """Freeze G4 — the post-test evaluation freeze. Never touches G3."""

    models: LoadedModels = loaded["models"]
    access_state = load_access_state(root / ACCESS_STATE_RELATIVE_PATH)
    payload = {
        "manifest_version": G4_VERSION,
        "status": "FINAL_TEST_EVALUATION_FROZEN",
        "gate": "G4",
        "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION",
        "authorization": "STAGE5_G4_FREEZE_REQUESTED",
        "human_member_signoff_claimed": False,
        "parents": {
            "g1": {"ref": G1_PATH, "sha256": sha256_file(root / G1_PATH)},
            "phase14_handoff": {"ref": PHASE14_HANDOFF_PATH, "sha256": sha256_file(root / PHASE14_HANDOFF_PATH)},
            "stage2_family_selection": {"ref": STAGE2_SELECTION_PATH, "sha256": sha256_file(root / STAGE2_SELECTION_PATH)},
            "selected_models_v1": {"ref": SELECTED_MODELS_PATH, "sha256": models.manifest_sha256},
            "g3_freeze": {"ref": G3_PATH, "sha256": models.g3_sha256},
            "stage4_acceptance": {"ref": STAGE4_ACCEPTANCE_PATH, "sha256": sha256_file(root / STAGE4_ACCEPTANCE_PATH)},
            "stage4_serving_bundle": {"ref": STAGE4_BUNDLE_PATH, "sha256": sha256_file(root / STAGE4_BUNDLE_PATH)},
            "stage5_final_test_plan": {"ref": str(plan_path.relative_to(root)), "sha256": plan_sha256},
            "naive_baseline": {"ref": str(naive_path.relative_to(root)) if naive_path.is_absolute() else str(naive_path), "sha256": naive_sha256},
        },
        "final_test_predictions": {
            name: {"ref": str(path.relative_to(root)), "sha256": digest}
            for name, (path, digest) in prediction_paths.items()
        },
        "final_metric_artifact": {"ref": str(metric_path[0].relative_to(root)), "sha256": metric_path[1]},
        "bootstrap_artifact": {"ref": str(bootstrap_path[0].relative_to(root)), "sha256": bootstrap_path[1]},
        "sensitivity_artifact": {"ref": str(sensitivity_path[0].relative_to(root)), "sha256": sensitivity_path[1]},
        "error_analysis_artifact": {
            "csv_ref": str(error_analysis_paths["csv"][0].relative_to(root)), "csv_sha256": error_analysis_paths["csv"][1],
            "json_ref": str(error_analysis_paths["json"][0].relative_to(root)), "json_sha256": error_analysis_paths["json"][1],
        },
        "registry": {
            "artifacts_ref": "experiments/artifacts.csv", "artifacts_sha256": sha256_file(root / "experiments/artifacts.csv"),
            "registry_ref": "experiments/registry.csv", "registry_sha256": sha256_file(root / "experiments/registry.csv"),
            "newly_registered_artifact_ids": list(registered_artifact_ids),
        },
        "code_commit": code_commit,
        "environment_lock": {"ref": STAGE4_ENV_LOCK_PATH, "sha256": sha256_file(root / STAGE4_ENV_LOCK_PATH)},
        "final_test_access": {
            "state": access_state["state"], "history": list(access_state["history"]),
        },
        "test_accessed": True,
        "final_test_access_consumed_count": sum(
            1 for event in access_state["history"] if event.get("event") == "FINAL_TEST_ACCESS_CONSUMED"
        ),
        "no_post_test_tuning": True,
        "bootstrap_config": {"n_bootstrap": 2000, "seed": 20260921, "ci_level": 0.95},
    }
    path = root / G4_PATH
    digest = _write_json(path, payload)
    return path, digest


def stage5_test_loader() -> Mapping[str, object]:
    """The single authorized final-test access point (module:attribute
    entrypoint for experiments.final_test_cli). Materializes the sealed test
    partition for the first time and runs the three frozen G3-selected
    models exactly once. No metrics, bootstrap, or persistence happen here."""

    return load_and_infer(ROOT, partition="test")


def stage5_evaluator(loaded: Mapping[str, object]) -> Mapping[str, object]:
    """Downstream analysis + persistence + registration + G4 (module:attribute
    entrypoint for experiments.final_test_cli). Runs immediately after
    stage5_test_loader inside the same guarded callback; performs zero
    further test-partition access."""

    if loaded["partition"] != "test":
        raise Stage5Error("stage5_evaluator must only run against the test partition")
    root = ROOT
    evaluated = stage5_evaluate(loaded, root=root, n_bootstrap=2000, seed=20260921)

    output_dir = root / FINAL_TEST_OUTPUT_ROOT
    prediction_paths = persist_predictions(loaded, output_dir)
    metric_path = persist_metrics(loaded, evaluated, output_dir)
    bootstrap_path = persist_bootstrap(evaluated, output_dir)
    sensitivity_path = persist_sensitivity(evaluated, output_dir)
    error_analysis_paths = persist_error_analysis(evaluated, output_dir)
    comparison_table = build_comparison_table(evaluated)
    comparison_path = output_dir / "comparison" / "comparison_table_v1.json"
    comparison_sha256 = _write_json(comparison_path, {"schema": "stage5_comparison_table_v1", "rows": comparison_table})

    code_commit = loaded["models"].manifest.get("code_commit")
    if not code_commit:
        import subprocess

        code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()

    # Every G3 check for this run has already completed (guarded_test_access,
    # then load_frozen_models inside stage5_test_loader). Registering here
    # legitimately extends experiments/artifacts.csv/registry.csv beyond
    # G3's own bound hash of those files at Stage-3 freeze time.
    ensure_stage5_run_registered(
        root, code_commit=code_commit, split_hash=evaluated["bindings"]["recovery"]["split_hash"],
        preprocessor_sha256=loaded["models"].manifest["tasks"]["recovery"]["preprocessor_sha256"],
    )
    registered = register_stage5_results(
        root, loaded, evaluated, prediction_paths, metric_path, error_analysis_paths,
        Path(loaded["naive_artifact_path"]), loaded["naive_artifact_sha256"], code_commit=code_commit,
    )

    plan_path = root / STAGE5_PLAN_PATH
    plan_sha256 = sha256_file(plan_path) if plan_path.is_file() else ""

    g4_path, g4_sha256 = build_g4(
        root, loaded, evaluated, prediction_paths=prediction_paths, metric_path=metric_path,
        bootstrap_path=bootstrap_path, sensitivity_path=sensitivity_path, error_analysis_paths=error_analysis_paths,
        naive_path=Path(loaded["naive_artifact_path"]), naive_sha256=loaded["naive_artifact_sha256"],
        plan_path=plan_path, plan_sha256=plan_sha256, code_commit=code_commit, registered_artifact_ids=registered,
    )

    task_by_name = {"recovery24": "recovery", "recovery48": "recovery", "icu": "icu_stay_time", "support": "organ_support"}
    model_run_id_by_name = {
        "recovery24": "phase12-xgb-recovery-014-attempt-1", "recovery48": "phase12-xgb-recovery-014-attempt-1",
        "icu": "final-v2-gru-icu-time-026-attempt-1", "support": "phase12-xgb-support-024-attempt-1",
    }
    model_sha256_by_name = {
        "recovery24": loaded["models"].manifest["tasks"]["recovery"]["model_artifacts"]["recovery24"]["sha256"],
        "recovery48": loaded["models"].manifest["tasks"]["recovery"]["model_artifacts"]["recovery48"]["sha256"],
        "icu": loaded["models"].manifest["tasks"]["icu_stay_time"]["artifact_sha256"],
        "support": loaded["models"].manifest["tasks"]["organ_support"]["artifact_sha256"],
    }
    label_version_by_name = {
        "recovery24": loaded["models"].manifest["tasks"]["recovery"]["label_version"],
        "recovery48": loaded["models"].manifest["tasks"]["recovery"]["label_version"],
        "icu": loaded["models"].manifest["tasks"]["icu_stay_time"]["label_version"],
        "support": loaded["models"].manifest["tasks"]["organ_support"]["label_version"],
    }
    validate_final_result_traceability(root, [
        {
            "prediction_artifact_id": "stage5-final-test-prediction-" + name + "-v1",
            "prediction_sha256": digest,
            "model_artifact_id": MODEL_ARTIFACT_IDS[name],
            "model_sha256": model_sha256_by_name[name],
            "manifest_artifact_id": MANIFEST_ARTIFACT_ID,
            "manifest_sha256": loaded["models"].manifest_sha256,
            "preprocessor_artifact_id": PREPROCESSOR_ARTIFACT_ID,
            "preprocessor_sha256": loaded["models"].manifest["tasks"]["recovery"]["preprocessor_sha256"],
            "g3_marker_ref": G3_PATH,
            "g3_marker_sha256": loaded["models"].g3_sha256,
            "split_hash": evaluated["bindings"]["recovery"]["split_hash"],
            "task": task_by_name[name],
            "model_run_id": model_run_id_by_name[name],
            "feature_version": loaded["models"].manifest["tasks"]["recovery"]["feature_version"],
            "label_version": label_version_by_name[name],
            "code_commit": code_commit,
        }
        for name, (_, digest) in prediction_paths.items()
    ])

    return {
        "status": "STAGE5_FINAL_TEST_EVALUATED",
        "g4_path": str(g4_path.relative_to(root)),
        "g4_sha256": g4_sha256,
        "prediction_paths": {name: (str(path.relative_to(root)), digest) for name, (path, digest) in prediction_paths.items()},
        "metric_path": (str(metric_path[0].relative_to(root)), metric_path[1]),
        "bootstrap_path": (str(bootstrap_path[0].relative_to(root)), bootstrap_path[1]),
        "sensitivity_path": (str(sensitivity_path[0].relative_to(root)), sensitivity_path[1]),
        "error_analysis_paths": {key: (str(path.relative_to(root)), digest) for key, (path, digest) in error_analysis_paths.items()},
        "comparison_table_path": (str(comparison_path.relative_to(root)), comparison_sha256),
        "registered_artifact_ids": list(registered),
        "comparison_table": comparison_table,
    }
