#!/usr/bin/env python3
"""Governed V3 pre-test freeze, seal, preflight, ONE-TIME paired final evaluation and guard.

  freeze            bind every frozen input by SHA-256; write selected_models_v3.json, freeze record, plan
  identity          print the sealed-cohort FINAL_IDENTITIES line and whether it is registered
  seal              structural-only audit of the generated sealed cohort; initialise the ledger
  preflight         development-only checks (never opens the sealed cohort); exit 1 on failure
  run               THE ONE-TIME OPERATION (authorize -> consume -> score -> save -> evidence)
  second-run-guard  prove a repeat run is BLOCKED_ALREADY_CONSUMED
  recompute         metrics / bootstrap / plots from the saved predictions only
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from performance_v3 import final_eval as fe  # noqa: E402


def _paths(args) -> fe.Paths:
    p = fe.Paths(ROOT, Path(args.base) if args.base else None)
    if getattr(args, "v3_bundle", None):
        p.v3_bundle = Path(args.v3_bundle)
    if getattr(args, "control_bundle", None):
        p.control_bundle = Path(args.control_bundle)
    for name in ("v3_bundle", "control_bundle"):
        value = getattr(p, name)
        if not value.is_absolute():
            setattr(p, name, ROOT / value)
    return p


def _dev_dry_run_rows(n: int):
    from performance_v3.data import CohortDir, iter_cohort_rows
    import itertools
    rows = list(itertools.islice(iter_cohort_rows(ROOT, CohortDir(ROOT / "artifacts/performance_v3/cohorts/LCE"), cohort_id="SYN-V3-LCE", role="lce"), n))
    trends = json.loads((ROOT / "artifacts/performance_v3/cache/sofa_LCE.json").read_text())
    tmap = {tuple(k.split("||", 1)): v for k, v in trends["trends"].items()}
    comps = {tuple(k.split("||", 1)): {c: tuple(v) for c, v in d.items()} for k, d in trends["components"].items()} if isinstance(trends.get("components"), dict) else {}
    return rows, (tmap, comps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=None, help="override artifacts/performance_v3 (tests)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--v3-bundle", default="artifacts/performance_v3/phase3/v3/bundle_manifest.json")
    f.add_argument("--control-bundle", default="artifacts/performance_v3/phase3/control/bundle_manifest.json")
    sub.add_parser("identity")
    s = sub.add_parser("seal"); s.add_argument("--skip-generator-check", action="store_true")
    pf = sub.add_parser("preflight"); pf.add_argument("--dry-run-rows", type=int, default=300); pf.add_argument("--v2-identity-n", type=int, default=300)
    r = sub.add_parser("run"); r.add_argument("--authorized-by", default="coordinator")
    sub.add_parser("second-run-guard")
    rc = sub.add_parser("recompute"); rc.add_argument("--from-predictions", action="store_true", required=True); rc.add_argument("--bootstrap-n", type=int, default=None)
    args = parser.parse_args()
    paths = _paths(args)
    if args.cmd == "freeze":
        print(json.dumps(fe.freeze(ROOT, paths), indent=1))
    elif args.cmd == "identity":
        print(json.dumps(fe.identity(ROOT), indent=1))
    elif args.cmd == "seal":
        out = fe.seal(ROOT, paths, check_generator=not args.skip_generator_check)
        print(json.dumps(out, indent=1))
        return 0 if out["verdict"] == "PASS" else 1
    elif args.cmd == "preflight":
        rows, sofa = _dev_dry_run_rows(args.dry_run_rows)
        out = fe.preflight(ROOT, paths, dry_run_rows=rows, dry_run_sofa=sofa)
        print(json.dumps({k: v["pass"] for k, v in out["checks"].items()}, indent=1), out["verdict"])
        return 0 if out["verdict"] == "PASS" else 1
    elif args.cmd == "run":
        print(json.dumps(fe.run_final(ROOT, paths, authorized_by=args.authorized_by), indent=1))
    elif args.cmd == "second-run-guard":
        out = fe.second_run_guard(ROOT, paths)
        print(json.dumps(out, indent=1))
        return 0 if out["blocked"] else 1
    elif args.cmd == "recompute":
        print(json.dumps(fe.recompute_from_predictions(ROOT, paths, n_bootstrap=args.bootstrap_n), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
