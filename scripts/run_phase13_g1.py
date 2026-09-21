#!/usr/bin/env python3
"""Run the full Phase-13 synthetic G1 acceptance gate."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acceptance.phase13 import finalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-junit", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, default=Path("artifacts/audit/phase13_full_replay"))
    parser.add_argument("--keep-replay", action="store_true")
    args = parser.parse_args()
    junit = args.focused_junit if args.focused_junit.is_absolute() else ROOT / args.focused_junit
    replay = args.replay_root if args.replay_root.is_absolute() else ROOT / args.replay_root
    try:
        marker = finalize(ROOT, junit, replay)
        print(marker.relative_to(ROOT))
    finally:
        if replay.exists() and not args.keep_replay:
            shutil.rmtree(replay)


if __name__ == "__main__":
    main()
