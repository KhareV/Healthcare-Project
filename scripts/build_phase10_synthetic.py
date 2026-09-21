#!/usr/bin/env python3
"""Build the frozen synthetic split and training-fit preprocessing artifacts."""
import argparse
from pathlib import Path

from preprocess.synthetic_phase10 import build_phase10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic/split_preprocessing_v1.yaml"))
    parser.add_argument("--phase9-manifest", type=Path, default=Path("artifacts/data/synthetic/pre_split/final/phase9_final_v1/phase9_pre_split_manifest_v1.json"))
    parser.add_argument("--raw-manifest", type=Path, default=Path("artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/synthetic/phase10/final/synthetic_phase10_v1"))
    parser.add_argument("--generation-timestamp", default="2026-09-21T00:00:00Z")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build_phase10(
        repo_root=root, config_path=root / args.config, phase9_manifest_path=root / args.phase9_manifest,
        raw_manifest_path=root / args.raw_manifest, output_directory=root / args.output,
        generation_timestamp=args.generation_timestamp,
    )
    print(result)


if __name__ == "__main__":
    main()
