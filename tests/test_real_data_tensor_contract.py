import json

import pytest

from data.real_adapter import RealDataContractError, load_real_bundle
from real_data_helpers import build_real_handoff_fixture, refresh_manifest_hash, write_json


def _mutate_canonical(root, operation):
    path = root / "artifacts/data/feature_dataset_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    operation(payload["rows"][0])
    write_json(path, payload)
    refresh_manifest_hash(root, "canonical_dataset")


@pytest.mark.parametrize("length", (7, 9))
def test_seven_or_nine_bins_fail(tmp_path, length):
    manifest = build_real_handoff_fixture(tmp_path)

    def attack(row):
        for field in ("sequence", "padding", "observed", "tslo"):
            row[field] = row[field][:length] if length == 7 else row[field] + [row[field][-1]]

    _mutate_canonical(tmp_path, attack)
    with pytest.raises(RealDataContractError, match="exactly 8|shape"):
        load_real_bundle(tmp_path, manifest)


def test_padding_and_observation_mask_confusion_fails(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)

    def attack(row):
        row["observed"][0][0] = True
        row["sequence"][0][0] = 3.0

    _mutate_canonical(tmp_path, attack)
    with pytest.raises(RealDataContractError, match="padded bins"):
        load_real_bundle(tmp_path, manifest)


def test_tslo_sentinel_mismatch_fails(tmp_path):
    manifest = build_real_handoff_fixture(tmp_path)
    _mutate_canonical(tmp_path, lambda row: row["tslo"][0].__setitem__(0, -999.0))
    with pytest.raises(RealDataContractError, match="TSLO"):
        load_real_bundle(tmp_path, manifest)
