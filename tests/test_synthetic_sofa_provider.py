from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import replace
import pytest

from data.synthetic.sofa import SOFADependencyUnavailable, SOFAError
from data.synthetic.sofa_provider import InMemorySOFASupportProvider, PulkitStateSOFASupportProvider, SyntheticCurrentSOFAProvider
from labels.event_dict import load_event_dictionary
from labels.support_state import ExecutionMode
from labels.ventilation_state import NormalizedVentilationInterval, RespiratoryCategory

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/data/synthetic/timelines/fixtures/phase5_fixture_v1/synthetic_processed_manifest_v1.json"
SPEC = ROOT / "configs/synthetic/sofa_spec_v1.json"
CONTRACT_HASH = "a" * 64


def make_provider(**kwargs):
    support = InMemorySOFASupportProvider(
        contract_version="ENGINEERING_SYNTHETIC_FIXTURE_SUPPORT_V1",
        contract_sha256=CONTRACT_HASH,
    )
    return SyntheticCurrentSOFAProvider(
        manifest_path=MANIFEST, root=ROOT, sofa_spec_path=SPEC,
        support_provider=support, **kwargs,
    )


def first_identity(provider):
    stay_id = sorted(provider.stays)[0]
    cutoff = provider.stays[stay_id]["intime"]
    return stay_id, cutoff


def test_provider_matches_direct_sofa_at_exactly():
    provider = make_provider()
    stay_id, cutoff = first_identity(provider)
    direct = provider.score(stay_id=stay_id, cutoff=cutoff)
    current = provider.current_sofa(stay_id=stay_id, prediction_time=cutoff)
    assert current.value == direct.total_score
    assert current.component_observed == direct.component_observed
    assert current.sofa_version == direct.sofa_spec_version
    assert current.source_sha256 == direct.source_identity_sha256


def test_provider_rejects_wrong_expected_spec_hash():
    with pytest.raises(SOFAError, match="SOFA-spec hash"):
        make_provider(expected_sofa_spec_sha256="0" * 64)


def test_provider_optional_legal_cutoff_validation():
    provider = make_provider(legal_cutoffs={})
    stay_id, cutoff = first_identity(provider)
    with pytest.raises(SOFAError, match="legal project cutoff"):
        provider.current_sofa(stay_id=stay_id, prediction_time=cutoff)


def test_provider_stay_isolation():
    provider = make_provider()
    stay_id, cutoff = first_identity(provider)
    result = provider.score(stay_id=stay_id, cutoff=cutoff)
    assert result.stay_id == stay_id
    assert all(row["stay_id"] == stay_id for row in provider.history if row["event_id"] in {ref for part in result.components for ref in part.evidence_refs})


def test_pulkit_adapter_reuses_existing_ventilation_state_engine():
    dictionary = load_event_dictionary(ROOT / "configs/event_dict_v1.yaml")
    cutoff = datetime(2100, 1, 2, tzinfo=timezone.utc)
    interval = NormalizedVentilationInterval(
        stay_id="A", category=RespiratoryCategory.INVASIVE,
        interval_start=cutoff - timedelta(hours=2),
        interval_end=cutoff - timedelta(minutes=30), source_state_ref="vent-1",
        concept_version=dictionary.ventilation.synthetic_concept_version,
        adapter_version=dictionary.ventilation.synthetic_adapter_version,
        normalization_provenance_ref="fixture",
    )
    adapter = PulkitStateSOFASupportProvider(
        event_dictionary=dictionary, execution_mode=ExecutionMode.SYNTHETIC,
        ventilation_intervals=(interval,), vasoactive_exposures=(),
        vasoactive_dose_contract_version="fixture-dose-v1",
        vasoactive_dose_contract_sha256="f" * 64,
        vasoactive_coverage_known=True,
    )
    evidence = adapter.evidence(
        stay_id="A", window_start=cutoff - timedelta(hours=24), cutoff=cutoff,
        respiratory_times=(cutoff - timedelta(hours=1),),
    )
    assert evidence.ventilation[0].state.value == "ACTIVE"
    assert evidence.ventilation[0].source_ref == "vent-1"

    leaking_adapter = PulkitStateSOFASupportProvider(
        event_dictionary=dictionary, execution_mode=ExecutionMode.SYNTHETIC,
        ventilation_intervals=(replace(interval, interval_end=cutoff + timedelta(hours=1)),),
        vasoactive_exposures=(), vasoactive_dose_contract_version="fixture-dose-v1",
        vasoactive_dose_contract_sha256="f" * 64,
        vasoactive_coverage_known=True,
    )
    cutoff_safe = leaking_adapter.evidence(
        stay_id="A", window_start=cutoff - timedelta(hours=24), cutoff=cutoff,
        respiratory_times=(cutoff,),
    )
    assert cutoff_safe.ventilation[0].state.value == "ACTIVE"
    assert cutoff_safe.ventilation[0].source_ref == "vent-1"
    assert not hasattr(cutoff_safe.ventilation[0], "interval_end")
