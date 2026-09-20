from dataclasses import replace

import numpy as np
import pytest

from explainability.tree_shap import (
    FlattenedFeatureContractError,
    PreparedTreeInput,
    aggregate_absolute_shap,
)
from tree_shap_helpers import identities, prepared


def test_structured_reverse_mapping_preserves_bins_static_mask_tslo_and_padding():
    mapped = identities()
    assert [(item.bin_index, item.relative_start_hours, item.relative_end_hours) for item in mapped[:3]] == [
        (0, -48, -42), (7, -6, 0), (3, -30, -24)
    ]
    assert mapped[3].channel_type == "observation_mask"
    assert mapped[4].channel_type == "tslo"
    assert mapped[5].channel_type == "padding"
    assert mapped[6].channel_type == "static"
    assert mapped[6].bin_index is None


def test_duplicate_names_count_and_bin_interval_fail_closed():
    base = prepared()
    duplicate = base.feature_identities[:-1] + (replace(base.feature_identities[-1], flat_name=base.feature_identities[0].flat_name),)
    with pytest.raises(FlattenedFeatureContractError, match="unique"):
        replace(base, feature_identities=duplicate).validate()
    with pytest.raises(FlattenedFeatureContractError, match="count"):
        replace(base, feature_identities=base.feature_identities[:-1]).validate()
    bad_bin = (replace(base.feature_identities[0], relative_start_hours=-47),) + base.feature_identities[1:]
    with pytest.raises(FlattenedFeatureContractError, match="interval"):
        replace(base, feature_identities=bad_bin).validate()


def test_signed_values_remain_available_and_absolute_bin_aggregation_is_exact():
    signed = np.asarray([-2.0, 3.0, -4.0, 0.5, -0.25, 1.0, -5.0])
    aggregate = aggregate_absolute_shap(signed, identities())
    a_latest = next(row for row in aggregate if row["base_feature"] == "A" and row["channel_type"] == "value")
    assert a_latest["absolute_attribution"] == 5.0
    assert signed.tolist() == [-2.0, 3.0, -4.0, 0.5, -0.25, 1.0, -5.0]
    assert len(aggregate) == 6


def test_prepared_input_rejects_nonfinite_values_without_new_missing_policy():
    base = prepared()
    attacked = base.values.copy()
    attacked[0, 2] = np.nan
    with pytest.raises(Exception, match="finite"):
        replace(base, values=attacked).validate()
