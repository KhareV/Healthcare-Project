#!/usr/bin/env python3
"""Generate the V3 research figures, tables, plot data, registry and index (read-only on all artifacts)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from performance_v3.research_export import run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/performance_v3/research")
    args = parser.parse_args()
    ex = run_all(ROOT, args.out)
    print(json.dumps({"exports": len(ex.entries), "skipped": ex.skipped}, indent=1))


if __name__ == "__main__":
    main()
