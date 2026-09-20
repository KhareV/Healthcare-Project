from pathlib import Path

from reproducibility.audit import load_config
from reproducibility.reproduce import synthetic_reproduction


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_bootstrap_regenerates_with_same_seed_and_b():
    result = synthetic_reproduction(
        load_config(ROOT / "configs/reproducibility_v1.json")
    )["bootstrap_regeneration"]
    assert result["status"] == "PASS"
    assert result["requested_replicates"] == 8
    assert result["valid_replicates"] + result["invalid_replicates"] == 8
