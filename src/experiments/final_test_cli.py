"""Audit the one canonical real Phase-20 path without bypass flags."""

import argparse
import importlib
import json
from pathlib import Path

from evaluation.final_test import (
    load_final_test_config,
    pretest_audit,
    register_pretest_snapshot,
    run_final_test,
    write_pretest_audit,
)
from experiments.final_test_entrypoint import run_guarded_final_test


def _load_callable(reference: str):
    module_name, attribute = reference.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError("configured Phase-20 entrypoint is not callable")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    report = pretest_audit(root)
    output = write_pretest_audit(root, report)
    snapshot = register_pretest_snapshot(root, output)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    print("pretest_audit=" + str(output))
    print("registered_snapshot=" + str(snapshot))
    if report.overall_status != "PASS":
        parser.exit(2, "BLOCKED — ACTIVE G3 FREEZE REQUIRED\n")
    if args.command == "run":
        config = load_final_test_config(root / "configs/final_test_v1.json")
        loader = _load_callable(config["real_test_loader_entrypoint"])
        evaluator = _load_callable(config["real_test_evaluator_entrypoint"])
        run_final_test(
            root,
            guarded_runner=lambda callback: run_guarded_final_test(root, callback),
            test_loader=loader,
            evaluator=evaluator,
        )


if __name__ == "__main__":
    main()
