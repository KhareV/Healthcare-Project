from dataclasses import replace

from models.gru_recovery import RECOVERY_HORIZON_ORDER
from serving.interfaces import ModelIdentity
from serving.recovery import (
    ORIGINAL_DELTA_DOMAIN,
    CurrentSOFAState,
    OriginalUnitRecoveryDeltas,
    RecoveryServingPostprocessor,
)


SOFA_VERSION = "SYNTHETIC_SOFA_AT_T_PHASE12_V1"
SOURCE_HASH = "5" * 64
CUTOFF = "2026-01-02T12:00:00+00:00"
STAY_ID = "SYNTHETIC_PHASE11_STAY"


def identity(family="gru"):
    return ModelIdentity(
        task="recovery",
        family=family,
        model_version="SYNTHETIC_RECOVERY_PHASE12_NOT_REAL",
        artifact_ref="synthetic/recovery.mock",
        artifact_sha256="1" * 64,
        preprocessor_ref="synthetic/preprocessor.mock",
        preprocessor_sha256="2" * 64,
        feature_version="SYNTHETIC_FEATURES_PHASE6_NOT_REAL",
        label_version="SYNTHETIC_RECOVERY_LABELS_PHASE6_NOT_REAL",
        split_hash="3" * 64,
        explanation_method=("integrated_gradients" if family == "gru" else "tree_shap"),
        target_transform_ref=None,
        target_transform_sha256=None,
    )


def state(
    value=10.0,
    *,
    stay_id=STAY_ID,
    prediction_time=CUTOFF,
    sofa_version=SOFA_VERSION,
    component_observed=(True, True, True, True, True, True),
):
    return CurrentSOFAState(
        stay_id=stay_id,
        prediction_time=prediction_time,
        value=value,
        sofa_version=sofa_version,
        source_version="SYNTHETIC_CURRENT_SOFA_SOURCE_PHASE12_V1",
        source_sha256=SOURCE_HASH,
        component_observed=component_observed,
    )


class SpyCurrentSOFAProvider:
    def __init__(self, supplied_state=None, by_cutoff=None):
        self.supplied_state = supplied_state or state()
        self.by_cutoff = dict(by_cutoff or {})
        self.calls = []

    def current_sofa(self, *, stay_id, prediction_time):
        self.calls.append((stay_id, prediction_time))
        return self.by_cutoff.get(prediction_time, self.supplied_state)


class SpyOriginalUnitDeltaAdapter:
    def __init__(self, *, provenance="frozen_inverse_transform_applied_once"):
        self.provenance = provenance
        self.calls = []

    def to_original_units(self, raw_output, prepared_input, model_identity):
        self.calls.append((raw_output, prepared_input, model_identity))
        return OriginalUnitRecoveryDeltas(
            delta_24h=float(raw_output[0]),
            delta_48h=float(raw_output[1]),
            output_domain=ORIGINAL_DELTA_DOMAIN,
            horizon_order=RECOVERY_HORIZON_ORDER,
            transform_provenance=self.provenance,
        )


def processor(current=None, adapter=None):
    current = current or SpyCurrentSOFAProvider()
    adapter = adapter or SpyOriginalUnitDeltaAdapter()
    return (
        RecoveryServingPostprocessor(
            current_sofa_provider=current,
            delta_adapter=adapter,
            expected_sofa_version=SOFA_VERSION,
        ),
        current,
        adapter,
    )
