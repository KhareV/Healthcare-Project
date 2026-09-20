"""Audit or attempt the real G3 freeze; no bypass flags exist."""

import argparse
import json
from pathlib import Path

from vedant_infra.g3 import (
    G3FreezeError,
    audit_g3,
    freeze_g3,
    write_audit_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "freeze"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "audit":
        report = audit_g3(args.root, scope="real")
        path = write_audit_report(args.root, report)
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        print("audit_report=" + str(path))
        return
    try:
        path = freeze_g3(args.root, scope="real")
    except G3FreezeError as error:
        parser.exit(2, "G3 freeze refused: " + str(error) + "\n")
    print(path)


if __name__ == "__main__":
    main()
