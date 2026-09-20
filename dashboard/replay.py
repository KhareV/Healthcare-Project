"""Synchronous retrospective-replay state controller."""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from pydantic import ValidationError

from api.schemas import ModelMetadataResponse, PredictionResponse
from evaluation.selection_validation import EXPLANATION_METHOD, TASKS
from dashboard.api_client import DashboardAPI, DashboardAPIError
from dashboard.catalog import DashboardCatalog, DashboardCatalogError, TimelineEvent
from serving.prediction_schema import validate_response
from serving.recovery import CurrentSOFAState


class ReplayContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplaySnapshot:
    stay_id: str
    prediction_time: str
    icu_elapsed_hours: float
    timeline: Tuple[TimelineEvent, ...]
    current_sofa: CurrentSOFAState
    prediction: Mapping[str, object]
    model_metadata: Mapping[str, object]


def _validate_response_identity(
    prediction: Mapping[str, object], metadata: Mapping[str, object], cutoff: str
) -> None:
    if prediction["prediction_time"] != cutoff:
        raise ReplayContractError("prediction response is bound to another cutoff")
    if prediction["model_metadata"]["manifest_version"] != metadata["manifest_version"]:
        raise ReplayContractError("prediction and metadata manifests differ")
    if prediction["model_metadata"]["split_version"] != metadata["split_version"]:
        raise ReplayContractError("prediction and metadata split versions differ")
    for task in TASKS:
        predicted = prediction["model_versions"][task]
        reported = metadata["tasks"][task]
        explanation = prediction["explanation_features"][task]
        if (
            predicted["family"] != reported["family"]
            or predicted["model_version"] != reported["model_version"]
            or predicted["artifact_sha256"] != reported["artifact_sha256"]
            or predicted["preprocessor_sha256"] != reported["preprocessor_sha256"]
            or prediction["model_metadata"]["feature_version"][task]
            != reported["feature_version"]
            or prediction["model_metadata"]["label_version"][task]
            != reported["label_version"]
            or explanation["family"] != reported["family"]
            or explanation["explanation_method"] != reported["explanation_method"]
            or reported["explanation_method"]
            != EXPLANATION_METHOD[reported["family"]]
        ):
            raise ReplayContractError("task prediction metadata is incompatible")
    if (
        prediction["model_versions"]["organ_support"]["calibrator_sha256"]
        != metadata["organ_support"]["calibrator_sha256"]
    ):
        raise ReplayContractError("support calibration metadata is incompatible")


class ReplayController:
    def __init__(self, catalog: DashboardCatalog, api: DashboardAPI) -> None:
        self._catalog = catalog
        self._api = api
        self.selected_stay_id: Optional[str] = None
        self.selected_cutoff: Optional[str] = None
        self.snapshot: Optional[ReplaySnapshot] = None
        self.last_error: Optional[str] = None

    @property
    def available_stays(self) -> Tuple[str, ...]:
        return self._catalog.stay_ids

    @property
    def available_cutoffs(self) -> Tuple[str, ...]:
        if self.selected_stay_id is None:
            return ()
        return self._catalog.stay(self.selected_stay_id).legal_cutoffs

    def select_stay(self, stay_id: str) -> None:
        self._catalog.stay(stay_id)
        self.selected_stay_id = stay_id
        self.selected_cutoff = None
        self.snapshot = None
        self.last_error = None

    def select_cutoff(self, cutoff: str) -> ReplaySnapshot:
        if self.selected_stay_id is None:
            raise ReplayContractError("select a demo stay before choosing a cutoff")
        if cutoff not in self.available_cutoffs:
            raise ReplayContractError("selected cutoff is not legal for this stay")
        self.selected_cutoff = cutoff
        self.snapshot = None
        self.last_error = None
        try:
            metadata = ModelMetadataResponse.model_validate(
                self._api.get_model_metadata()
            ).model_dump()
            prediction = PredictionResponse.model_validate(
                self._api.predict(self.selected_stay_id, cutoff)
            ).model_dump()
            prediction = validate_response(
                prediction, synthetic=self._catalog.scope == "synthetic"
            )
            if metadata["serving_scope"] != self._catalog.scope:
                raise ReplayContractError("catalog and serving scopes differ")
            _validate_response_identity(prediction, metadata, cutoff)
            timeline = self._catalog.timeline_through(self.selected_stay_id, cutoff)
            current = self._catalog.current_sofa(self.selected_stay_id, cutoff)
            stay = self._catalog.stay(self.selected_stay_id)
            snapshot = ReplaySnapshot(
                stay_id=self.selected_stay_id,
                prediction_time=cutoff,
                icu_elapsed_hours=stay.elapsed_hours(cutoff),
                timeline=timeline,
                current_sofa=current,
                prediction=prediction,
                model_metadata=metadata,
            )
        except (
            DashboardAPIError,
            DashboardCatalogError,
            ReplayContractError,
            ValidationError,
            ValueError,
            KeyError,
            TypeError,
        ) as error:
            self.last_error = (
                error.safe_message
                if isinstance(error, DashboardAPIError)
                else "replay data is unavailable or incompatible"
            )
            self.snapshot = None
            if isinstance(error, (DashboardAPIError, DashboardCatalogError, ReplayContractError)):
                raise
            raise ReplayContractError("replay response contract is invalid") from error
        self.snapshot = snapshot
        return snapshot
