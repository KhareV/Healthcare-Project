"""Pure dashboard bindings; no prediction, calibration, or explanation logic."""

from dataclasses import dataclass
from typing import Mapping, Tuple

from dashboard.catalog import TimelineEvent
from dashboard.replay import ReplaySnapshot
from evaluation.selection_validation import TASKS
from serving.recovery_display import RecoveryDisplayData, recovery_display_data


REPLAY_BANNER = "RETROSPECTIVE SEQUENTIAL REPLAY - NOT REAL-TIME CLINICAL PREDICTION"
SYNTHETIC_BANNER = "SYNTHETIC DEMO / NON-SCIENTIFIC"
ICU_TITLE = "Remaining ICU stay time"
ICU_DEFINITION = "Remaining time until current ICU stay ends"
SUPPORT_TITLE = "New Organ-Support Initiation Risk"
SUPPORT_DEFINITION = (
    "Calibrated probability of eligible OFF-to-ON initiation in the next 24 hours"
)
MONITORED_SUPPORTS = (
    "Qualifying vasopressor support",
    "Invasive mechanical ventilation",
)
EXPLANATION_TITLE = "Top contributors to this model prediction"


@dataclass(frozen=True)
class ExplanationView:
    task: str
    family: str
    method: str
    items: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class TaskMetadataView:
    task: str
    family: str
    model_version: str
    artifact_sha256: str
    preprocessor_sha256: str
    feature_version: str
    label_version: str
    split_hash: str
    explanation_method: str


@dataclass(frozen=True)
class DashboardView:
    stay_id: str
    prediction_time: str
    icu_elapsed_hours: float
    timeline: Tuple[TimelineEvent, ...]
    recovery: RecoveryDisplayData
    icu_stay_time_hours: float
    support_probability: float
    support_threshold: float
    explanations: Tuple[ExplanationView, ...]
    quality: Mapping[str, int]
    task_metadata: Tuple[TaskMetadataView, ...]
    manifest_version: str
    manifest_sha256: str
    split_version: str
    calibrator_sha256: str
    threshold_sha256: str


def build_dashboard_view(snapshot: ReplaySnapshot, *, synthetic: bool) -> DashboardView:
    """Bind one validated snapshot to all panels without deriving new science."""

    response = snapshot.prediction
    metadata = snapshot.model_metadata
    recovery = recovery_display_data(response, snapshot.current_sofa, synthetic=synthetic)
    quality = response["data_quality"]
    required_quality = (
        "total_bins",
        "observed_bins",
        "padding_bins",
        "total_feature_values",
        "observed_feature_values",
        "missing_feature_values",
    )
    if set(quality) != set(required_quality):
        raise ValueError("data-quality contract is incomplete")
    explanations = tuple(
        ExplanationView(
            task=task,
            family=response["explanation_features"][task]["family"],
            method=response["explanation_features"][task]["explanation_method"],
            items=tuple(
                (item["feature_name"], float(item["attribution"]))
                for item in response["explanation_features"][task]["items"]
            ),
        )
        for task in TASKS
    )
    task_metadata = tuple(
        TaskMetadataView(
            task=task,
            family=metadata["tasks"][task]["family"],
            model_version=metadata["tasks"][task]["model_version"],
            artifact_sha256=metadata["tasks"][task]["artifact_sha256"],
            preprocessor_sha256=metadata["tasks"][task]["preprocessor_sha256"],
            feature_version=metadata["tasks"][task]["feature_version"],
            label_version=metadata["tasks"][task]["label_version"],
            split_hash=metadata["tasks"][task]["split_hash"],
            explanation_method=metadata["tasks"][task]["explanation_method"],
        )
        for task in TASKS
    )
    return DashboardView(
        stay_id=snapshot.stay_id,
        prediction_time=snapshot.prediction_time,
        icu_elapsed_hours=snapshot.icu_elapsed_hours,
        timeline=snapshot.timeline,
        recovery=recovery,
        icu_stay_time_hours=float(response["icu_stay_time_hours"]),
        support_probability=float(response["organ_support_probability_calibrated"]),
        support_threshold=float(metadata["organ_support"]["threshold"]),
        explanations=explanations,
        quality={key: int(quality[key]) for key in required_quality},
        task_metadata=task_metadata,
        manifest_version=metadata["manifest_version"],
        manifest_sha256=metadata["manifest_sha256"],
        split_version=metadata["split_version"],
        calibrator_sha256=metadata["organ_support"]["calibrator_sha256"],
        threshold_sha256=metadata["organ_support"]["threshold_sha256"],
    )
