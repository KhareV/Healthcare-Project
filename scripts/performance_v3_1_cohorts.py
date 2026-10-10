#!/usr/bin/env python3
"""plan | build V3.1 diagnostic cohorts with the frozen generator chain (same chain as V3)."""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from performance_v3.protocol import GENERATION_TIMESTAMP
from performance_v3_1 import protocol as P

def run(stage, argv, log):
    t = time.time(); r = subprocess.run(argv, cwd=ROOT, env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    log.append({"stage": stage, "seconds": round(time.time() - t, 1), "rc": r.returncode})
    if r.returncode: sys.stderr.write(r.stderr[-3000:]); raise SystemExit("STOP " + stage)
    print(f"[{log[-1]['seconds']:7.1f}s] {stage}", flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("plan", "build")); ap.add_argument("cohorts", nargs="*", default=list(P.SPECS))
    ap.add_argument("--python", default=sys.executable); a = ap.parse_args()
    (ROOT / P.CONFIG_DIR_REL).mkdir(parents=True, exist_ok=True)
    for cid in a.cohorts:
        ident = P.identity(ROOT, cid); cfg_path = ROOT / P.CONFIG_DIR_REL / f"{cid.lower()}_v1.json"
        if a.cmd == "plan":
            cfg_path.write_text(json.dumps(P.runtime_config(ROOT, cid), indent=2, sort_keys=True) + "\n")
            print(f'    ({ident["seed"]}, {ident["n_subjects"]}): "{ident["namespace"]}",  # {cid}'); continue
        out = P.cohort_dir(ROOT, cid); rel = lambda p: str(p.relative_to(ROOT)); st = ["--generation-timestamp", GENERATION_TIMESTAMP]; log = []
        if (out / "pre_split/phase9_pre_split_manifest_v1.json").exists(): print("skip", cid); continue
        py = a.python
        run("generate", [py, "scripts/generate_synthetic_dataset.py", "--config", rel(cfg_path), "--output", rel(out / "raw"), *st], log)
        run("cohort", [py, "scripts/build_synthetic_cohort.py", "--source-manifest", rel(out / "raw/synthetic_dataset_manifest_v1.json"), "--config", "configs/synthetic/cohort_spec_v2.yaml", "--output", rel(out / "cohort"), "--mode", "final", *st], log)
        run("timeline", [py, "scripts/build_synthetic_timeline.py", "--cohort-manifest", rel(out / "cohort/synthetic_cohort_manifest_v2.json"), "--concept-map", "configs/synthetic/concept_map_v1.json", "--processed-schema", "configs/synthetic/processed_schema_v1.json", "--output", rel(out / "timeline"), "--mode", "final", *st], log)
        run("features", [py, "scripts/build_synthetic_features.py", "--processed-manifest", rel(out / "timeline/synthetic_processed_manifest_v1.json"), "--structural-index", rel(out / "cohort/structural_index.jsonl"), "--feature-schema", "configs/synthetic/feature_schema_v2.json", "--output", rel(out / "features"), "--mode", "final", *st], log)
        run("pre_split", [py, "scripts/build_phase9_final_package.py", "--cohort-manifest", rel(out / "cohort/synthetic_cohort_manifest_v2.json"), "--processed-manifest", rel(out / "timeline/synthetic_processed_manifest_v1.json"), "--feature-manifest", rel(out / "features/synthetic_feature_input_manifest_v1.json"), "--output", rel(out / "pre_split"), "--generation-timestamp", GENERATION_TIMESTAMP], log)
        (out / "build_log.json").write_text(json.dumps({"identity": ident, "stages": log}, indent=2) + "\n")
main()
