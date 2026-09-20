import json
from pathlib import Path

from data.dataset import CanonicalTensorDataset, SyntheticTensorizationPolicy
from data.schema import FeatureSchemaReference, deserialize_canonical_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FIXTURE = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "canonical"
    / "canonical_synthetic_v1.json"
)
SYNTHETIC_SPLIT = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "canonical"
    / "canonical_synthetic_split_v1.csv"
)


def feature_schema():
    payload = json.loads(CANONICAL_FIXTURE.read_text(encoding="utf-8"))
    raw = payload["synthetic_feature_schema"]
    return FeatureSchemaReference(
        version=raw["version"],
        feature_names=tuple(raw["feature_names"]),
        status=raw["status"],
    )


def canonical_dataset():
    schema = feature_schema()
    return deserialize_canonical_dataset(CANONICAL_FIXTURE.read_bytes(), schema)


def tensor_dataset(partition):
    return CanonicalTensorDataset.from_fixture(
        CANONICAL_FIXTURE,
        feature_schema(),
        partition,
        SyntheticTensorizationPolicy(0.0),
        SYNTHETIC_SPLIT,
    )

