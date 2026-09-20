"""Phase-14 synthetic cross-layer integration composition helpers."""

from dataclasses import dataclass
from types import MappingProxyType

from api_helpers import client_for
from dashboard.api_client import DashboardAPIError
from dashboard.catalog import DashboardCatalog, ReplayStay, TimelineEvent
from dashboard.replay import ReplayController
from phase7_helpers import EARLY_CUTOFF, LATER_CUTOFF, MAIN_CUTOFF, MAIN_STAY, build_provider, load_timelines
from serving.pipeline import PredictionPipeline
from models.gru_recovery import RECOVERY_HORIZON_ORDER
from serving.recovery import (
    CurrentSOFAState,
    ExplicitOriginalUnitDeltaAdapter,
    OriginalUnitRecoveryDeltas,
    RecoveryServingPostprocessor,
)
from serving_helpers import FixtureRuntime, build_kwargs, copy_serving_fixture


SOFA_VERSION = "SYNTHETIC_SOFA_AT_T_PHASE14_V1"
LEGAL_CUTOFFS = (EARLY_CUTOFF, MAIN_CUTOFF, LATER_CUTOFF)


class SyntheticCurrentSOFAProvider:
    def current_sofa(self, *, stay_id, prediction_time):
        return CurrentSOFAState(
            stay_id=stay_id,
            prediction_time=prediction_time,
            value=4.0,
            sofa_version=SOFA_VERSION,
            source_version="SYNTHETIC_PHASE14_CURRENT_SOFA_NOT_REAL",
            source_sha256="7" * 64,
            component_observed=(True, True, False, True, False, True),
        )


class IntegratedPostprocessor:
    """Phase-12 recovery plus canonical Phase-11 ICU-time postprocessing."""

    def __init__(self, current_sofa_provider=None, expected_sofa_version=SOFA_VERSION):
        from serving.postprocessing import CanonicalICUTimeServingPostprocessor

        recovery = RecoveryServingPostprocessor(
            current_sofa_provider=current_sofa_provider or SyntheticCurrentSOFAProvider(),
            delta_adapter=ExplicitOriginalUnitDeltaAdapter(),
            expected_sofa_version=expected_sofa_version,
        )
        self._delegate = CanonicalICUTimeServingPostprocessor(recovery)

    def recovery(self, raw_output, prepared_input, **context):
        return self._delegate.recovery(raw_output, prepared_input, **context)

    def icu_stay_time_hours(self, raw_output):
        return self._delegate.icu_stay_time_hours(raw_output)


class TestClientDashboardAPI:
    """Dashboard protocol adapter over an actual in-process FastAPI client."""

    def __init__(self, client):
        self.client = client
        self.calls = []

    @staticmethod
    def _value(response):
        payload = response.json()
        if response.status_code >= 400:
            detail = payload.get("error", {})
            raise DashboardAPIError(
                response.status_code,
                detail.get("code", "api_error"),
                detail.get("message", "serving request failed"),
            )
        return payload

    def get_health(self):
        self.calls.append(("GET", "/health"))
        return self._value(self.client.get("/health"))

    def get_model_metadata(self):
        self.calls.append(("GET", "/model-metadata"))
        return self._value(self.client.get("/model-metadata"))

    def predict(self, stay_id, prediction_time):
        self.calls.append(("POST", "/predict", stay_id, prediction_time))
        return self._value(
            self.client.post(
                "/predict",
                json={"stay_id": stay_id, "prediction_time": prediction_time},
            )
        )


@dataclass
class IntegratedSystem:
    pipeline: PredictionPipeline
    runtime: FixtureRuntime
    builder: object
    api_client: object
    dashboard_api: TestClientDashboardAPI
    dashboard_catalog: DashboardCatalog

    def controller(self):
        return ReplayController(self.dashboard_catalog, self.dashboard_api)


def dashboard_catalog(timelines=None):
    timeline = (timelines or load_timelines())[0]
    events = tuple(
        TimelineEvent(
            prediction_available_time=row["synthetic_available_time"],
            category="lab",
            label=row["synthetic_feature"] + " #" + str(row["synthetic_sequence"]),
            value=row["synthetic_value"],
            observed=True,
        )
        for row in timeline.events
        if row["synthetic_available_time"] >= timeline.intime.isoformat()
    )
    stay = ReplayStay(
        subject_id=str(timeline.subject_id),
        stay_id=str(timeline.stay_id),
        intime=timeline.intime.isoformat(),
        legal_cutoffs=LEGAL_CUTOFFS,
        events=events,
        current_sofa_by_cutoff=MappingProxyType(
            {cutoff: 4.0 for cutoff in LEGAL_CUTOFFS}
        ),
        sofa_version=SOFA_VERSION,
    )
    return DashboardCatalog((stay,), scope="synthetic")


def build_integrated_system(
    tmp_path,
    *,
    timelines=None,
    current_sofa_provider=None,
    expected_sofa_version=SOFA_VERSION,
    dashboard_catalog_override=None,
):
    root = copy_serving_fixture(tmp_path)
    runtime = FixtureRuntime()
    provider, builder = build_provider(timelines=timelines)
    runtime.input_provider = provider
    pipeline = PredictionPipeline.build(
        **build_kwargs(root, runtime),
        input_provider=provider,
        postprocessor=IntegratedPostprocessor(
            current_sofa_provider, expected_sofa_version
        ),
        explanation_adapters=runtime.explanation_adapters,
    )
    recovery_predictor = runtime.predictors["recovery"]

    def recovery_predict(prepared_input):
        recovery_predictor.predict_calls += 1
        recovery_predictor.last_prepared_input = prepared_input
        return OriginalUnitRecoveryDeltas(
            delta_24h=0.25,
            delta_48h=-0.5,
            output_domain="original_sofa_delta_points",
            horizon_order=RECOVERY_HORIZON_ORDER,
            transform_provenance="model_native_original_units",
        )

    recovery_predictor.predict = recovery_predict
    api_client = client_for(pipeline)
    dashboard_api = TestClientDashboardAPI(api_client)
    return IntegratedSystem(
        pipeline=pipeline,
        runtime=runtime,
        builder=builder,
        api_client=api_client,
        dashboard_api=dashboard_api,
        dashboard_catalog=dashboard_catalog_override or dashboard_catalog(timelines),
    )
