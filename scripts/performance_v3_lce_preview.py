#!/usr/bin/env python3
"""One-time DEVELOPMENT sanity preview on LCE (not used in selection): V3 vs CONTROL vs frozen V2 (as-served / as-evaluated)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from performance_v3.data import CohortDir, load_cohort_rows
from performance_v3.serving import V2Comparator, V3Bundle
from performance_v3 import paired_bootstrap as pb, task_metrics as tm
import performance_v3_v2_modes_check as mc

def main():
    rows = list(load_cohort_rows(ROOT, CohortDir(ROOT / "artifacts/performance_v3/cohorts/LCE"), cohort_id="SYN-V3-LCE", role="lce"))
    tr = json.loads((ROOT / "artifacts/performance_v3/cache/sofa_LCE.json").read_text())
    trends = {tuple(k.split("||", 1)): v for k, v in tr["trends"].items()}
    comps = {tuple(k.split("||", 1)): v for k, v in tr["components"].items()}
    v2 = V2Comparator.load(ROOT)
    preds = {"V2_AS_EVALUATED": v2.predict_rows(rows), "V2_AS_SERVED": v2.predict_rows(rows, current_sofa_trends=trends)}
    for name, kind in (("V3", "v3"), ("CONTROL", "control")):
        b = V3Bundle.load(ROOT, f"artifacts/performance_v3/phase3/{kind}/bundle_manifest.json")
        preds[name] = b.predict_rows(rows, sofa_trends=trends, sofa_components=comps)
    keys = {"recovery24": "recovery24_delta", "recovery48": "recovery48_delta"}
    out = {"cohort": "LCE (development, not used in selection)", "rows": len(rows), "models": {}}
    for name, p in preds.items():
        out["models"][name] = {t: mc.metrics(t, rows, p, keys) for t in ("recovery24", "recovery48", "icu_stay_time", "organ_support")}
    Path(ROOT / "artifacts/performance_v3/phase2/lce_preview_v1.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for t in ("recovery24", "recovery48", "icu_stay_time", "organ_support"):
        print(t)
        for name in preds:
            m = out["models"][name][t]
            print("  %-16s" % name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})

main()
