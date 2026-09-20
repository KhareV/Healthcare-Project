from dataclasses import replace
import pytest

from data.synthetic.sofa import SOFADependencyUnavailable, SOFAError, sofa_at
from synthetic_sofa_helpers import BINDINGS, CUTOFF, SPEC, STAY, event, provider


def run(rows, support=None, bindings=BINDINGS):
    return sofa_at(rows, STAY, CUTOFF, spec=SPEC, bindings=bindings, support_provider=support)


def test_all_missing_is_zero_with_six_false_flags_given_known_support_coverage():
    result = run([], provider())
    assert result.total_score == 0
    assert result.component_observed == (False,) * 6


def test_mixed_missing_and_observed_zero_are_distinct():
    result = run([event("platelet_count", 200), event("glasgow_coma_scale", 15)], provider())
    assert result.component_scores == (0, 0, 0, 0, 0, 0)
    assert result.component_observed == (False, True, False, False, True, False)


def test_missing_provider_is_pipeline_block_not_clinical_missingness():
    with pytest.raises(SOFADependencyUnavailable, match="REQUIRED SOFA DEPENDENCY"):
        run([], None)


def test_unknown_provider_coverage_is_not_coerced_to_inactive():
    with pytest.raises(SOFADependencyUnavailable):
        run([], provider(known=False))


def test_working_provider_and_no_map_is_clinically_unobserved_cardiovascular():
    result = run([], provider())
    assert (result.components[3].score, result.components[3].observed) == (0, False)


@pytest.mark.parametrize("concept,value,bad_unit", [("platelet_count",100,"mg/dL"),("bilirubin_total",2,"mmHg"),("creatinine",2,"score"),("fio2",.5,"percent")])
def test_invalid_units_fail_closed(concept, value, bad_unit):
    with pytest.raises(SOFAError, match="unit"):
        run([event(concept, value, unit=bad_unit)], provider())


@pytest.mark.parametrize("concept,value", [("creatinine",float("nan")),("platelet_count",float("inf")),("fio2",0)])
def test_invalid_values_fail_closed(concept, value):
    with pytest.raises(SOFAError):
        run([event(concept, value)], provider())


def test_support_contract_hash_mismatch_fails_closed():
    with pytest.raises(SOFAError, match="support contract"):
        run([], provider(contract_hash="f" * 64))


def test_binding_hash_shape_validation_fails_closed():
    with pytest.raises(SOFAError, match="processed_schema_sha256"):
        run([], provider(), replace(BINDINGS, processed_schema_sha256="wrong"))
