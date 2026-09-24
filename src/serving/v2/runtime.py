"""V2 real, model-backed serving runtime: raw-history truncation -> V2
canonical features -> frozen Phase-3 XGBoost models -> frozen Phase-3
postprocessing/calibration -> TreeSHAP explanation.

Reuses, unmodified:
  - features.synthetic.SyntheticCanonicalFeatureBuilder (the same single-
    cutoff, cutoff-aware canonical feature builder Phase-1/3's batch
    pipeline used to build every DEV/fresh-cohort row -- genuine raw-history
    truncation and feature construction, not a lookup).
  - data.synthetic.sofa_provider.SyntheticCurrentSOFAProvider /
    PulkitStateSOFASupportProvider (the same current-SOFA computation used
    throughout Phase 1-3).
  - data.timestamps.generate_prediction_rows_for_stay (the canonical legal-
    cutoff generator).
  - performance_v2.v2_features / context_normalization / final_evaluation
    (Phase-3's frozen Group-A + context feature assembly, reconstructed
    exactly as Phase 4 did for the one-time fresh-test run).
  - models.recovery_output-equivalent clip(0,24) reconstruction, expm1
    postprocess, and the frozen isotonic_support_v2 calibrator.

No model, feature, calibration, or threshold decision is made here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
import xgboost as xgb

from data.synthetic.sofa import load_sofa_spec
from data.synthetic.sofa_provider import PulkitStateSOFASupportProvider, SyntheticCurrentSOFAProvider
from data.synthetic.validation import load_jsonl
from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay
from features.synthetic import FeatureBuildContext, SyntheticCanonicalFeatureBuilder
from labels.support_state import ExecutionMode, NormalizedActiveInterval, query_vasopressor_state
from labels.synthetic_profile import load_synthetic_event_dictionary
from labels.ventilation_state import NormalizedVentilationInterval, query_invasive_ventilation_state
from performance_v2.context_normalization import apply_normalization
from performance_v2.final_evaluation import TASK_MODEL_FILE, TASK_VARIANT, isotonic_apply, load_frozen_normalization, postprocess_icu_hours, reconstruct_group_a_names
from performance_v2.v2_features import build_group_a_matrix
from serving.v2.guard import UnknownDemoStayError, guard_demo_stay
from vedant_infra.hashing import sha256_file

RAW_DIR = "artifacts/data/synthetic/final/phase9_final_v1"
TIMELINE_DIR = "artifacts/data/synthetic/timelines/final/phase9_final_v1"
PROCESSED_MANIFEST = TIMELINE_DIR + "/synthetic_processed_manifest_v1.json"
FEATURE_SCHEMA_PATH = "configs/synthetic/feature_schema_v2.json"
SOFA_SPEC_PATH = "configs/synthetic/sofa_spec_v1.json"
SUPPORT_PROCESS_PATH = "configs/synthetic/support_process_v1.yaml"
EVENT_DICT_PATH = "configs/event_dict_v2.yaml"

THRESHOLD_VALUE = 0.39781983118092895


class IllegalCutoffError(ValueError):
    """Raised when a requested prediction_time is not a legal cutoff."""


class V2ServingError(RuntimeError):
    """Raised for any V2 serving artifact/verification failure."""


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass(frozen=True)
class V2Prediction:
    stay_id: str
    prediction_time: str
    grid_index: int
    icu_elapsed_hours: int
    current_sofa: float
    recovery24_delta: float
    recovery48_delta: float
    recovery24_sofa: float
    recovery48_sofa: float
    icu_raw_log_prediction: float
    icu_remaining_hours: float
    support_raw_probability: float
    support_calibrated_probability: float
    support_threshold: float
    support_alert: bool
    model_hashes: Mapping[str, str]
    feature_row: Mapping[str, object]
    matrices: Mapping[str, np.ndarray]
    feature_names: Mapping[str, Tuple[str, ...]]


class V2ServingRuntime:
    """One process-lifetime, hash-verified V2 serving runtime."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

        self.model_hashes = {task: sha256_file(self.root / TASK_MODEL_FILE[task]) for task in TASK_VARIANT}
        expected_model_hashes = {
            "recovery24": "b355c26339271f0b84c123eb46cfb61c8df0cc9c9c6fcfcbb86c29047d10ad3c",
            "recovery48": "5dda6bb0830428b44b3a7cf554dc7f6d2adc278c7b218045d1df8ff960f6c701",
            "icu_stay_time": "918c81eef86bd89e91e02bb06c254774375f3f2371c4f3c5200c270dea24f7e6",
            "organ_support": "2fa653a0d2c644d8a1de806146f8e8cba46df87e6bd037868122276b4a440671",
        }
        if self.model_hashes != expected_model_hashes:
            raise V2ServingError("V2 model file hash mismatch: refusing to serve")

        self.models = {}
        for task in TASK_VARIANT:
            model = xgb.XGBClassifier() if task == "organ_support" else xgb.XGBRegressor()
            model.load_model(str(self.root / TASK_MODEL_FILE[task]))
            self.models[task] = model

        self.group_a_names = {task: reconstruct_group_a_names(self.root, task) for task in TASK_VARIANT}
        self.normalizations = {variant: load_frozen_normalization(self.root, variant) for variant in set(TASK_VARIANT.values())}

        calibrator_payload = json.loads((self.root / "artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json").read_text())
        self.calibration_knots = calibrator_payload["calibration_knots"]

        threshold_payload = json.loads((self.root / "artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json").read_text())
        if abs(threshold_payload["threshold_value"] - THRESHOLD_VALUE) > 1e-12:
            raise V2ServingError("support threshold value mismatch: refusing to serve")

        self.feature_builder = self._build_feature_builder()
        self.sofa_provider = self._build_sofa_provider()

        statics_rows = load_jsonl(self.root / TIMELINE_DIR / "canonical_statics.jsonl")
        self._statics_by_stay = {row["stay_id"]: row for row in statics_rows}

        events = load_jsonl(self.root / TIMELINE_DIR / "canonical_timeline.jsonl")
        self._events_by_stay: dict = {}
        for row in events:
            self._events_by_stay.setdefault(row["stay_id"], []).append(row)

        supports = load_jsonl(self.root / RAW_DIR / "support_intervals.jsonl")
        self._supports_by_stay: dict = {}
        for row in supports:
            self._supports_by_stay.setdefault(row["stay_id"], []).append(row)

    def _build_feature_builder(self) -> SyntheticCanonicalFeatureBuilder:
        statics_rows = load_jsonl(self.root / TIMELINE_DIR / "canonical_statics.jsonl")
        statics_by_stay = {row["stay_id"]: row for row in statics_rows}
        return SyntheticCanonicalFeatureBuilder(
            schema_path=self.root / FEATURE_SCHEMA_PATH, root=self.root, statics_by_stay=statics_by_stay,
        )

    def _build_sofa_provider(self) -> SyntheticCurrentSOFAProvider:
        dictionary = load_synthetic_event_dictionary(self.root / EVENT_DICT_PATH)
        support_hash = sha256_file(self.root / SUPPORT_PROCESS_PATH)
        supports = load_jsonl(self.root / RAW_DIR / "support_intervals.jsonl")
        vaso_rows = [r for r in supports if r["support_type"] == "VASOPRESSOR"]
        vent_rows = [r for r in supports if r["support_type"] == "RESPIRATORY"]
        vaso = tuple(
            NormalizedActiveInterval.from_mapping({
                "stay_id": r["stay_id"], "agent_key": r["agent_key"], "interval_start": r["interval_start"],
                "interval_end": r["interval_end"], "source_event_ref": r["support_event_id"],
                "normalization_provenance_version": dictionary.synthetic_mapping_provenance_version,
            })
            for r in vaso_rows
        )
        vent = tuple(
            NormalizedVentilationInterval.from_mapping({
                "stay_id": r["stay_id"], "category": r["respiratory_category"], "interval_start": r["interval_start"],
                "interval_end": r["interval_end"], "source_state_ref": r["support_event_id"],
                "concept_version": dictionary.ventilation.synthetic_concept_version,
                "adapter_version": dictionary.ventilation.synthetic_adapter_version,
                "normalization_provenance_ref": r["normalization_provenance_ref"],
            })
            for r in vent_rows
        )
        from data.synthetic.sofa import VasoactiveExposure

        exposures = tuple(
            VasoactiveExposure(
                stay_id=r["stay_id"], agent=r["agent_key"], rate=float(r["rate_value"]), unit=r["rate_unit"],
                interval_start=_dt(r["interval_start"]), interval_end=_dt(r["interval_end"]), source_ref=r["support_event_id"],
            )
            for r in vaso_rows
        )
        support_provider = PulkitStateSOFASupportProvider(
            event_dictionary=dictionary, execution_mode=ExecutionMode.SYNTHETIC,
            ventilation_intervals=vent, vasoactive_exposures=exposures,
            vasoactive_dose_contract_version="synthetic_support_process_v1",
            vasoactive_dose_contract_sha256=support_hash, vasoactive_coverage_known=True,
        )
        return SyntheticCurrentSOFAProvider(
            manifest_path=self.root / PROCESSED_MANIFEST, root=self.root,
            sofa_spec_path=self.root / SOFA_SPEC_PATH, support_provider=support_provider,
        )

    def legal_cutoffs(self, stay_id: str) -> Tuple[str, ...]:
        entry = guard_demo_stay(self.root, stay_id)
        return tuple(entry["legal_cutoffs"])

    def predict(self, stay_id: str, prediction_time: str) -> V2Prediction:
        entry = guard_demo_stay(self.root, stay_id)  # UnknownDemoStayError on anything else, incl. fresh-test subjects
        subject_id = entry["subject_id"]
        legal = self.legal_cutoffs(stay_id)
        if prediction_time not in legal:
            raise IllegalCutoffError(f"prediction_time {prediction_time!r} is not a legal cutoff for {stay_id}")

        static = self._statics_by_stay[stay_id]
        intime = _dt(static["intime"])
        outtime = _dt(static["outtime"])
        t = _dt(prediction_time)
        icu_elapsed_hours = round((t - intime).total_seconds() / 3600.0)
        grid_index = (icu_elapsed_hours - 24) // 6

        context = FeatureBuildContext(
            subject_id=subject_id, stay_id=stay_id, intime=intime, outtime=outtime,
            events=tuple(self._events_by_stay.get(stay_id, ())),
            prediction_time=t, prediction_time_text=prediction_time,
            grid_index=int(grid_index), icu_elapsed_hours=int(icu_elapsed_hours),
            support_intervals=tuple(self._supports_by_stay.get(stay_id, ())),
        )
        canonical = self.feature_builder.build(context)

        sofa_result = self.sofa_provider.score(stay_id=stay_id, cutoff=prediction_time)
        current_sofa = float(sofa_result.total_score)

        row = {
            "subject_id": subject_id, "stay_id": stay_id, "prediction_time": prediction_time,
            "grid_index": int(grid_index), "temporal_feature_names": canonical.temporal_feature_names,
            "history_values": canonical.history_values, "observation_mask": canonical.observation_mask,
            "padding_mask": canonical.padding_mask,
            "tslo_hours": canonical.tslo_hours, "static_features": canonical.static_features,
            "static_feature_names": canonical.static_feature_names,
            "elapsed_episode_hours_at_t": float(icu_elapsed_hours), "cutoff_index": int(grid_index),
            "hours_since_first_eligible_cutoff": float(icu_elapsed_hours) - 24.0,
            "baseline_sofa": current_sofa,
        }

        matrices = {}
        feature_names = {}
        for task, variant in TASK_VARIANT.items():
            matrix_a, names_a = build_group_a_matrix([row], names=self.group_a_names[task])
            context_matrix = apply_normalization([row], self.normalizations[variant])
            matrix = np.concatenate([matrix_a, context_matrix], axis=1)
            matrices[task] = matrix
            feature_names[task] = names_a + tuple(self.normalizations[variant]["feature_names"])

        recovery24_delta = float(self.models["recovery24"].predict(matrices["recovery24"])[0])
        recovery48_delta = float(self.models["recovery48"].predict(matrices["recovery48"])[0])
        recovery24_sofa = float(np.clip(current_sofa + recovery24_delta, 0.0, 24.0))
        recovery48_sofa = float(np.clip(current_sofa + recovery48_delta, 0.0, 24.0))

        icu_raw_log = float(self.models["icu_stay_time"].predict(matrices["icu_stay_time"])[0])
        icu_hours = float(postprocess_icu_hours(np.asarray([icu_raw_log]))[0])

        support_raw = float(self.models["organ_support"].predict_proba(matrices["organ_support"])[0, 1])
        support_cal = float(isotonic_apply([support_raw], self.calibration_knots)[0])
        support_alert = bool(support_cal >= THRESHOLD_VALUE)

        return V2Prediction(
            stay_id=stay_id, prediction_time=prediction_time, grid_index=int(grid_index),
            icu_elapsed_hours=int(icu_elapsed_hours), current_sofa=current_sofa,
            recovery24_delta=recovery24_delta, recovery48_delta=recovery48_delta,
            recovery24_sofa=recovery24_sofa, recovery48_sofa=recovery48_sofa,
            icu_raw_log_prediction=icu_raw_log, icu_remaining_hours=icu_hours,
            support_raw_probability=support_raw, support_calibrated_probability=support_cal,
            support_threshold=THRESHOLD_VALUE, support_alert=support_alert,
            model_hashes=self.model_hashes, feature_row=row, matrices=matrices, feature_names=feature_names,
        )
