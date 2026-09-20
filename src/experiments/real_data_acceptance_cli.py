"""Phase-18 structural real-data audit; no training or performance evaluation."""

import argparse
import json
from pathlib import Path

from data.acceptance import (
    audit_real_data,
    register_acceptance_snapshot,
    write_acceptance_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit_real_data(root, manifest_path=args.manifest)
    output = root / "artifacts/data_acceptance/real_data_acceptance_v1.json"
    write_acceptance_report(output, report)
    snapshot = register_acceptance_snapshot(root, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("acceptance_report=" + str(output))
    print("registered_snapshot=" + str(snapshot))
    raise SystemExit(0 if report["overall_status"] == "ACCEPTED" else 2)


if __name__ == "__main__":
    main()
