from data.real_adapter import assert_information_parity, load_real_bundle
from real_data_helpers import build_real_handoff_fixture


def test_xgboost_is_only_row_major_flattening_of_gru_information(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    bundle = load_real_bundle(tmp_path, manifest)
    assert_information_parity(bundle)
    example = bundle.examples[0]
    assert len(example.history_values) == 8
    assert len(example.history_values[0]) == len(bundle.dynamic_feature_names)
