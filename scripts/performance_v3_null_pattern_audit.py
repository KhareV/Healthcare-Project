#!/usr/bin/env python3
"""Run the D1 availability-linked-missingness audit on the corrected V3 feature caches."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from performance_v3.features import load_feature_cache  # noqa: E402
from performance_v3.null_audit import audit_cache  # noqa: E402

CACHE = ROOT / "artifacts/performance_v3/cache"
OUT = ROOT / "artifacts/performance_v3/data_characterization/null_pattern_audit_v1.json"


def main() -> None:
    report = {"artifact_version": "performance_v3_null_pattern_audit_v1", "amendment": "performance_v3_protocol_amendment_v1_1", "caches": {}}
    for name in ("dev_pool", "features_LCE"):
        report["caches"][name] = audit_cache(load_feature_cache(CACHE / name))
    report["verdict"] = "PASS" if all(c["verdict"] == "PASS" for c in report["caches"].values()) else "FAIL"
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, c in report["caches"].items():
        print(name, c["verdict"], "max|gap|=%.4f" % c["max_abs_gap"], "current_sofa_nan=%.3f" % c["current_sofa_nan_fraction"], "flagged", len(c["flagged_columns"]))
    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
