"""Thin canonical-feature and frozen-preprocessor integration boundary.

No scientific feature or preprocessing logic lives here.  The canonical
builder is injected and its validated output is adapted with the existing
tensor-contract parity utility.  The Phase-6 pipeline remains responsible for
calling the selected serialized preprocessor's ``transform`` method.
"""

from datetime import datetime
from typing import Callable, Mapping, Optional

from data.schema import (
    CanonicalFeatureInput,
    CanonicalExample,
    FeatureSchemaReference,
    TaskEligibility,
    TaskTargets,
    temporal_information_views,
    validate_canonical_example,
    validate_canonical_feature_input,
)
from serving.history import HistoryProvider, HistoryTruncator, HistoryTrace


class ServingFeatureContractError(RuntimeError):
    """Canonical feature output is incompatible with serving/model identity."""


CanonicalFeatureBuilder = Callable[[object], object]


_EMPTY_ELIGIBILITY = TaskEligibility(False, False, False, False)
_EMPTY_TARGETS = TaskTargets(None, None, None, None)


class CanonicalHistoryInputProvider:
    """Compose history isolation/truncation with one injected canonical builder."""

    def __init__(
        self,
        *,
        history_provider: HistoryProvider,
        history_truncator: HistoryTruncator,
        build_features: CanonicalFeatureBuilder,
        feature_schema: FeatureSchemaReference,
    ) -> None:
        self._history_provider = history_provider
        self._history_truncator = history_truncator
        self._build_features = build_features
        self._feature_schema = feature_schema
        builder_schema = getattr(build_features, "feature_schema", None)
        if builder_schema is None:
            builder_schema = getattr(build_features, "schema", None)
        if builder_schema != feature_schema:
            raise ServingFeatureContractError(
                "canonical builder feature schema/order binding mismatch"
            )
        self._last_trace: Optional[HistoryTrace] = None
        self._last_example: Optional[object] = None

    def get_canonical_input(
        self,
        *,
        stay_id: object,
        prediction_time: str,
        task: str,
        family: str,
        feature_version: str,
    ) -> object:
        timeline = self._history_provider.get_stay(stay_id)
        truncated = self._history_truncator.truncate(timeline, prediction_time)
        example = self._build_features(truncated)
        if isinstance(example, CanonicalFeatureInput):
            validate_canonical_feature_input(example, self._feature_schema)
        elif isinstance(example, CanonicalExample):
            validate_canonical_example(example, self._feature_schema)
        else:
            raise ServingFeatureContractError("canonical builder returned an unsupported type")
        if feature_version != self._feature_schema.version:
            raise ServingFeatureContractError("selected model feature version mismatch")
        requested_time = datetime.fromisoformat(prediction_time.replace("Z", "+00:00"))
        returned_time = datetime.fromisoformat(example.prediction_time.replace("Z", "+00:00"))
        if example.stay_id != stay_id or returned_time != requested_time:
            raise ServingFeatureContractError("canonical builder returned the wrong example")
        if isinstance(example, CanonicalExample) and (example.eligibility != _EMPTY_ELIGIBILITY or example.targets != _EMPTY_TARGETS):
            raise ServingFeatureContractError(
                "serving feature output must not contain targets or outcome eligibility"
            )
        views = temporal_information_views(example)
        if family not in views:
            raise ServingFeatureContractError("unsupported model-family representation")
        self._last_trace = truncated.trace
        self._last_example = example
        return views[family]

    def data_quality(self, *, stay_id: object, prediction_time: str) -> Mapping[str, int]:
        example = self._last_example
        trace = self._last_trace
        if example is None or trace is None:
            raise ServingFeatureContractError("data quality requested before feature construction")
        requested_time = datetime.fromisoformat(prediction_time.replace("Z", "+00:00"))
        returned_time = datetime.fromisoformat(example.prediction_time.replace("Z", "+00:00"))
        if example.stay_id != stay_id or returned_time != requested_time:
            raise ServingFeatureContractError("data quality request does not match latest input")
        total_bins = len(example.padding_mask)
        padding_bins = sum(example.padding_mask)
        feature_dim = self._feature_schema.feature_dim
        observed_values = sum(sum(row) for row in example.observation_mask)
        observed_bins = sum(any(row) for row in example.observation_mask)
        return {
            "total_bins": total_bins,
            "observed_bins": observed_bins,
            "padding_bins": padding_bins,
            "total_feature_values": total_bins * feature_dim,
            "observed_feature_values": observed_values,
            # Phase-5 defines this as all tensor positions without a genuine
            # observation.  Structural padding remains separately visible via
            # padding_bins and must not be silently merged at presentation.
            "missing_feature_values": total_bins * feature_dim - observed_values,
        }
