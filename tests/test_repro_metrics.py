from pathlib import Path

from reproducibility.audit import load_config
from reproducibility.reproduce import synthetic_reproduction


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_metric_regeneration_is_exact():
    result = synthetic_reproduction(
        load_config(ROOT / "configs/reproducibility_v1.json")
    )
    assert result["metric_regeneration"]["status"] == "PASS"
    assert result["metric_regeneration"]["expected"] == result["metric_regeneration"]["reproduced"]
