"""Metadata-only audit and provenance trace command for Phase 16."""

import argparse
import json
from pathlib import Path

from experiments.audit import audit_registry_lineage
from experiments.lineage import (
    read_artifact_index,
    read_run_registry,
    trace_artifact,
    trace_run,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "trace-run", "trace-artifact"))
    parser.add_argument("identifier", nargs="?")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_registry_lineage(args.root)
    else:
        if not args.identifier:
            parser.error("trace commands require an identifier")
        runs = read_run_registry(args.root / "experiments/registry.csv")
        artifacts = read_artifact_index(args.root / "experiments/artifacts.csv")
        result = (
            trace_run(args.identifier, artifacts, runs)
            if args.command == "trace-run"
            else trace_artifact(args.identifier, artifacts, runs)
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
