"""Phase-19 plan/audit/run/finalize command surface."""

import argparse
import json
from pathlib import Path

from experiments.validation_suite import (
    ValidationSuiteError,
    preflight_validation_suite,
    register_preflight_snapshot,
    run_validation_suite,
    write_preflight_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "audit", "run", "finalize"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    report = preflight_validation_suite(root)
    output = write_preflight_report(root, report)
    snapshot = register_preflight_snapshot(root, output)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    print("preflight_report=" + str(output))
    print("registered_snapshot=" + str(snapshot))
    if args.command in ("run", "finalize"):
        try:
            run_validation_suite(
                root,
                lambda: (_ for _ in ()).throw(
                    ValidationSuiteError("scientific executor is not configured")
                ),
            )
        except ValidationSuiteError as error:
            parser.exit(2, "validation suite refused: " + str(error) + "\n")
    if report.overall_status != "PASS":
        parser.exit(2, "validation suite preflight is BLOCKED\n")


if __name__ == "__main__":
    main()
