"""Precompute the past-cutoff-only SOFA trend for every recovery-eligible
TRAIN/VALIDATION row, once, and cache it. DEVELOPMENT_DIAGNOSTIC_ONLY;
internal cache, not a Phase-1 findings artifact. Reused by the trend/delta
feature group (C) and recovery baseline R1.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from performance_v2.data_loading import load_dev_rows  # noqa: E402
from performance_v2.sofa_trend import compute_sofa_trends  # noqa: E402

CACHE_DIR = ROOT / "artifacts/performance_v2/phase1/_cache"


def main() -> None:
    rows = load_dev_rows(ROOT)
    print("all dev rows:", len(rows))
    t0 = time.time()
    trends = compute_sofa_trends(ROOT, rows)
    print("elapsed seconds:", time.time() - t0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {stay_id + "||" + prediction_time: value for (stay_id, prediction_time), value in trends.items()}
    out_path = CACHE_DIR / "sofa_trend_cache_v1.json"
    out_path.write_text(json.dumps(serializable), encoding="utf-8")
    print("wrote", out_path, "entries", len(serializable))


if __name__ == "__main__":
    main()
