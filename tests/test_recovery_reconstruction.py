from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.recovery_output import reconstruct_absolute_sofa
from serving.recovery import (
    ORIGINAL_DELTA_DOMAIN,
    ExplicitOriginalUnitDeltaAdapter,
    OriginalUnitRecoveryDeltas,
    RecoveryServingError,
    RecoveryServingPostprocessor,
)
from recovery_helpers import (
    CUTOFF,
    SOFA_VERSION,
    STAY_ID,
    SpyCurrentSOFAProvider,
    SpyOriginalUnitDeltaAdapter,
    identity,
    processor,
    state,
)


def reconstruct(current, delta24, delta48, *, clip=True):
    return reconstruct_absolute_sofa(
        torch.tensor([current], dtype=torch.float64),
        torch.tensor([[delta24, delta48]], dtype=torch.float64),
        clip_for_display=clip,
    )[0]


def test_canonical_double_counting_fixture_uses_same_baseline_independently():
    result = reconstruct(10.0, 6.0, -3.0)
    assert result.tolist() == [16.0, 7.0]
    assert result[1].item() != 13.0


def test_horizon_independence_and_swap_properties():
    baseline = reconstruct(10.0, 6.0, -3.0)
    changed24 = reconstruct(10.0, 100.0, -3.0)
    changed48 = reconstruct(10.0, 6.0, -100.0)
    assert changed24[1] == baseline[1]
    assert changed48[0] == baseline[0]
    swapped = reconstruct(10.0, -3.0, 6.0)
    assert swapped.tolist() == [7.0, 16.0]


def test_display_only_clipping_preserves_raw_deltas():
    raw = torch.tensor([[8.0, -30.0]], dtype=torch.float64)
    before = raw.clone()
    display = reconstruct_absolute_sofa(
        torch.tensor([22.0], dtype=torch.float64), raw, clip_for_display=True
    )
    unbounded = reconstruct_absolute_sofa(
        torch.tensor([22.0], dtype=torch.float64), raw, clip_for_display=False
    )
    assert display.tolist() == [[24.0, 0.0]]
    assert unbounded.tolist() == [[30.0, -8.0]]
    assert torch.equal(raw, before)


def test_exact_boundaries_in_range_and_continuous_values_are_not_rounded():
    assert reconstruct(3.0, -3.0, 21.0).tolist() == [0.0, 24.0]
    assert reconstruct(10.0, -3.0, 5.0).tolist() == [7.0, 15.0]
    assert reconstruct(10.25, 0.5, -0.125).tolist() == [10.75, 10.125]


@pytest.mark.parametrize(
    "current,delta24,delta48,match",
    (
        (float("nan"), 0.0, 0.0, "baseline_sofa must be finite"),
        (float("inf"), 0.0, 0.0, "baseline_sofa must be finite"),
        (30.0, 0.0, 0.0, "within"),
        (-1.0, 0.0, 0.0, "within"),
        (10.0, float("nan"), 0.0, "predictions must be finite"),
        (10.0, 0.0, float("inf"), "predictions must be finite"),
    ),
)
def test_nonfinite_and_invalid_current_sofa_fail_without_sanitizing(current, delta24, delta48, match):
    with pytest.raises(ValueError, match=match):
        reconstruct(current, delta24, delta48)


def test_untyped_standardized_output_cannot_be_reconstructed_directly():
    adapter = ExplicitOriginalUnitDeltaAdapter()
    with pytest.raises(RecoveryServingError, match="standardized"):
        adapter.to_original_units((0.0, 0.0), object(), identity())


@pytest.mark.parametrize(
    "changes,match",
    (
        ({"output_domain": "standardized_gru_output"}, "original SOFA-delta"),
        ({"horizon_order": ("delta_sofa_48h", "delta_sofa_24h")}, "horizon order"),
        ({"transform_provenance": "inverse_transform_applied_twice"}, "provenance"),
        ({"delta_24h": float("nan")}, "finite"),
        ({"delta_48h": float("inf")}, "finite"),
    ),
)
def test_output_domain_horizon_transform_and_finiteness_are_explicit(changes, match):
    value = OriginalUnitRecoveryDeltas(
        delta_24h=1.0,
        delta_48h=-1.0,
        output_domain=ORIGINAL_DELTA_DOMAIN,
        horizon_order=("delta_sofa_24h", "delta_sofa_48h"),
        transform_provenance="frozen_inverse_transform_applied_once",
    )
    with pytest.raises(RecoveryServingError, match=match):
        replace(value, **changes).validate()


def test_current_sofa_interface_is_called_once_at_t_and_inverse_adapter_once():
    service, current, adapter = processor()
    result = service.recovery(
        (6.0, -3.0),
        {"model_input": True},
        stay_id=STAY_ID,
        prediction_time=CUTOFF,
        model_identity=identity(),
    )
    assert result == {
        "delta_24h": 6.0,
        "delta_48h": -3.0,
        "reconstructed_sofa_24h": 16.0,
        "reconstructed_sofa_48h": 7.0,
    }
    assert current.calls == [(STAY_ID, CUTOFF)]
    assert len(adapter.calls) == 1


def test_recovery_reconstruction_is_identical_for_gru_and_xgboost_identities():
    results = []
    for family in ("gru", "xgboost"):
        service, _, _ = processor()
        results.append(
            service.recovery(
                (6.0, -3.0), object(), stay_id=STAY_ID,
                prediction_time=CUTOFF, model_identity=identity(family)
            )
        )
    assert results[0] == results[1]


def test_sofa_version_cutoff_stay_and_component_flag_contracts_fail_closed():
    attacked = (
        state(sofa_version="OTHER"),
        state(stay_id="OTHER"),
        state(prediction_time="2026-01-03T12:00:00+00:00"),
        state(component_observed=(True,)),
    )
    for supplied in attacked:
        service, _, _ = processor(SpyCurrentSOFAProvider(supplied))
        with pytest.raises(RecoveryServingError):
            service.recovery(
                (0.0, 0.0), object(), stay_id=STAY_ID,
                prediction_time=CUTOFF, model_identity=identity()
            )


def test_missing_component_flags_do_not_create_a_completeness_heuristic():
    supplied = state(component_observed=(False, False, False, False, False, False))
    service, _, _ = processor(SpyCurrentSOFAProvider(supplied))
    assert service.recovery(
        (1.0, -1.0), object(), stay_id=STAY_ID,
        prediction_time=CUTOFF, model_identity=identity()
    )["reconstructed_sofa_48h"] == 9.0


def test_later_cutoff_uses_its_own_verified_current_state_without_equality_requirement():
    later = "2026-01-02T18:00:00+00:00"
    current = SpyCurrentSOFAProvider(
        by_cutoff={
            CUTOFF: state(10.0),
            later: state(12.0, prediction_time=later),
        }
    )
    service, _, _ = processor(current)
    first = service.recovery(
        (1.0, 1.0), object(), stay_id=STAY_ID,
        prediction_time=CUTOFF, model_identity=identity()
    )
    second = service.recovery(
        (1.0, 1.0), object(), stay_id=STAY_ID,
        prediction_time=later, model_identity=identity()
    )
    assert first["reconstructed_sofa_24h"] == 11.0
    assert second["reconstructed_sofa_24h"] == 13.0
    assert current.calls == [(STAY_ID, CUTOFF), (STAY_ID, later)]


def test_metric_training_and_selection_paths_do_not_import_display_reconstruction():
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "src/evaluation/metrics.py",
        root / "src/evaluation/selection.py",
        root / "src/training/tasks/recovery.py",
    )
    for path in paths:
        if path.exists():
            source = path.read_text()
            assert "reconstruct_absolute_sofa" not in source
            assert "serving.recovery" not in source


def test_recovery_serving_module_has_no_training_selection_calibration_or_threshold_logic():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/serving/recovery.py").read_text().lower()
    for forbidden in (
        ".fit(", "optimizer", "loss.backward", "choose_best", "isotonic",
        "support_calibrator", "support_threshold", "sofa_at(t+24", "sofa_at(t+48",
    ):
        assert forbidden not in source
