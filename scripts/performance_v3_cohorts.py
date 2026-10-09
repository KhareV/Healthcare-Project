#!/usr/bin/env python3
"""Plan / build the Performance-V3 synthetic cohorts with the frozen generator chain.

  plan   : write per-cohort runtime configs + print FINAL_IDENTITIES entries to register
  build  : generate -> cohort -> timeline -> features -> pre-split package (final mode)

The chain is exactly the one recorded in the V2 fresh-cohort manifests; no generator
code or scientific parameter is touched here.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from performance_v3.protocol import (  # noqa: E402
    COHORT_CONFIG_DIR_REL, COHORT_SPECS, GENERATION_TIMESTAMP, cohort_config, cohort_dir,
    cohort_identity, protocol_sha256, sha256_file,
)


def _run(stage: str, argv, log: list) -> None:
    started = time.time()
    result = subprocess.run(argv, cwd=ROOT, env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    elapsed = round(time.time() - started, 1)
    log.append({"stage": stage, "seconds": elapsed, "returncode": result.returncode, "command": " ".join(argv[1:])})
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-4000:])
        raise SystemExit(f"STOP: stage {stage} failed")
    print(f"[{elapsed:8.1f}s] {stage}", flush=True)


def cmd_plan(ids) -> None:
    out_dir = ROOT / COHORT_CONFIG_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    print("protocol_sha256", protocol_sha256(ROOT))
    for cohort_id in ids:
        identity = cohort_identity(ROOT, cohort_id)
        config = cohort_config(ROOT, cohort_id)
        path = out_dir / f"{cohort_id.lower()}_v1.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"({identity['seed']}, {identity['n_subjects']}): \"{identity['namespace']}\",  # {cohort_id} -> {path.relative_to(ROOT)}")


def cmd_build(ids, python: str, force: bool) -> None:
    for cohort_id in ids:
        identity = cohort_identity(ROOT, cohort_id)
        cfg = ROOT / COHORT_CONFIG_DIR_REL / f"{cohort_id.lower()}_v1.json"
        if not cfg.exists():
            raise SystemExit(f"missing config for {cohort_id}; run `plan` first")
        out = cohort_dir(ROOT, cohort_id)
        done = out / "pre_split" / "phase9_pre_split_manifest_v1.json"
        if done.exists() and not force:
            print(f"skip {cohort_id}: already built")
            continue
        rel = lambda p: str(p.relative_to(ROOT))
        log: list = []
        stamp = ["--generation-timestamp", GENERATION_TIMESTAMP]
        _run("generate", [python, "scripts/generate_synthetic_dataset.py", "--config", rel(cfg), "--output", rel(out / "raw"), *stamp], log)
        _run("cohort", [python, "scripts/build_synthetic_cohort.py", "--source-manifest", rel(out / "raw/synthetic_dataset_manifest_v1.json"),
                        "--config", "configs/synthetic/cohort_spec_v2.yaml", "--output", rel(out / "cohort"), "--mode", "final", *stamp], log)
        _run("timeline", [python, "scripts/build_synthetic_timeline.py", "--cohort-manifest", rel(out / "cohort/synthetic_cohort_manifest_v2.json"),
                          "--concept-map", "configs/synthetic/concept_map_v1.json", "--processed-schema", "configs/synthetic/processed_schema_v1.json",
                          "--output", rel(out / "timeline"), "--mode", "final", *stamp], log)
        _run("features", [python, "scripts/build_synthetic_features.py", "--processed-manifest", rel(out / "timeline/synthetic_processed_manifest_v1.json"),
                          "--structural-index", rel(out / "cohort/structural_index.jsonl"), "--feature-schema", "configs/synthetic/feature_schema_v2.json",
                          "--output", rel(out / "features"), "--mode", "final", *stamp], log)
        _run("pre_split", [python, "scripts/build_phase9_final_package.py", "--cohort-manifest", rel(out / "cohort/synthetic_cohort_manifest_v2.json"),
                           "--processed-manifest", rel(out / "timeline/synthetic_processed_manifest_v1.json"),
                           "--feature-manifest", rel(out / "features/synthetic_feature_input_manifest_v1.json"),
                           "--output", rel(out / "pre_split"), "--generation-timestamp", GENERATION_TIMESTAMP], log)
        record = {"cohort_id": cohort_id, "identity": identity, "config_sha256": sha256_file(cfg), "stages": log,
                  "protocol_sha256": protocol_sha256(ROOT) if cohort_id != "T" else None}
        (out / "build_log.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("built", cohort_id, "->", rel(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build"):
        p = sub.add_parser(name)
        p.add_argument("cohorts", nargs="*", default=["D1", "D2", "D3", "LCE"], choices=sorted(COHORT_SPECS))
        if name == "build":
            p.add_argument("--python", default=sys.executable)
            p.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.cmd == "plan":
        cmd_plan(args.cohorts)
    else:
        cmd_build(args.cohorts, args.python, args.force)


if __name__ == "__main__":
    main()
