import pytest

from reproducibility.compare import ReproducibilityComparisonError, validate_reproduction_identity
from test_repro_frozen_inference import identity


@pytest.mark.parametrize("field", ("config_hash", "split_hash", "preprocessor_sha256"))
def test_changed_reproduction_identity_fails_before_comparison(field):
    expected = identity()
    reproduced = dict(expected)
    reproduced[field] = "f" * 64
    with pytest.raises(ReproducibilityComparisonError, match=field):
        validate_reproduction_identity(expected, reproduced)
