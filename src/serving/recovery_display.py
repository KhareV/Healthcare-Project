"""Pure UI-framework-independent recovery view-data binding."""

from dataclasses import dataclass
from typing import Mapping

from serving.prediction_schema import validate_response
from serving.recovery import CurrentSOFAState, RecoveryServingError


@dataclass(frozen=True)
class RecoveryDisplayData:
    prediction_time: str
    current_sofa: float
    delta_24h: float
    delta_48h: float
    predicted_sofa_24h: float
    predicted_sofa_48h: float
    current_label: str = "Observed current SOFA at cutoff"
    predicted_24h_label: str = "Predicted SOFA +24h"
    predicted_48h_label: str = "Predicted SOFA +48h"
    display_minimum: float = 0.0
    display_maximum: float = 24.0


def recovery_display_data(
    response: Mapping[str, object],
    current_sofa: CurrentSOFAState,
    *,
    synthetic: bool,
) -> RecoveryDisplayData:
    """Bind validated API values without repeating reconstruction arithmetic."""

    validated = validate_response(response, synthetic=synthetic)
    current_sofa.validate()
    if validated["prediction_time"] != current_sofa.prediction_time:
        raise RecoveryServingError(
            "display current SOFA is bound to another prediction cutoff"
        )
    recovery = validated["recovery"]
    return RecoveryDisplayData(
        prediction_time=str(validated["prediction_time"]),
        current_sofa=float(current_sofa.value),
        delta_24h=float(recovery["delta_24h"]),
        delta_48h=float(recovery["delta_48h"]),
        predicted_sofa_24h=float(recovery["reconstructed_sofa_24h"]),
        predicted_sofa_48h=float(recovery["reconstructed_sofa_48h"]),
    )
