"""Stage-5 FINAL CLOSURE (not Stage 6): governance-truthfulness pass.

This script does exactly two things to
artifacts/governance/g4_test_evaluation_freeze_v1.json, both non-scientific:

1. Refreshes registry.artifacts_sha256 / registry.registry_sha256 to match
   the current experiments/artifacts.csv / experiments/registry.csv, since
   the post-Part-B docs/evidence/models/ regeneration (fixing a genuine,
   current-vs-current evidence-staleness test failure) legitimately added
   development-scoped rows to experiments/artifacts.csv.
2. Adds explicit access-history clarity fields so G4 cannot be misread as
   claiming only one consumption event ever happened:
   - total_historical_consumption_events: 2 (both FINAL_TEST_ACCESS_CONSUMED
     events across this project's full history)
   - successful_scientific_evaluation_count: 1 (the second event; the first
     crashed before any test row was read -- zero science exposure)
   - consumption_events_since_current_g3_freeze: 1 (the current G3, created
     at the formal reset, has been consumed exactly once)

No prediction, metric, bootstrap, sensitivity, error-analysis, naive
baseline, comparison, model, split, calibrator, or threshold artifact is
touched by this script. It prints every field that changed so the change
can be reviewed directly, and independently re-verifies (by re-reading from
disk) that every final-test scientific-output hash referenced by G4 is
unchanged from before this script ran.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

SCIENTIFIC_HASH_FIELDS = (
    ("final_test_predictions", "recovery24"),
    ("final_test_predictions", "recovery48"),
    ("final_test_predictions", "icu"),
    ("final_test_predictions", "support"),
    ("final_metric_artifact",),
    ("bootstrap_artifact",),
    ("sensitivity_artifact",),
)


def _get(payload, path):
    node = payload
    for key in path:
        node = node[key]
    return node


def main() -> None:
    g4_path = ROOT / "artifacts/governance/g4_test_evaluation_freeze_v1.json"
    before = json.loads(g4_path.read_text())
    before_scientific = {path: _get(before, path)["sha256"] for path in SCIENTIFIC_HASH_FIELDS}
    before_error_analysis = (before["error_analysis_artifact"]["csv_sha256"], before["error_analysis_artifact"]["json_sha256"])
    before_g3 = before["parents"]["g3_freeze"]["sha256"]
    before_selected_models = before["parents"]["selected_models_v1"]["sha256"]
    before_naive = before["parents"]["naive_baseline"]["sha256"]

    payload = dict(before)
    payload["registry"] = dict(payload["registry"])
    payload["registry"]["artifacts_sha256"] = sha256_file(ROOT / "experiments/artifacts.csv")
    payload["registry"]["registry_sha256"] = sha256_file(ROOT / "experiments/registry.csv")
    payload["total_historical_consumption_events"] = 2
    payload["successful_scientific_evaluation_count"] = 1
    payload["consumption_events_since_current_g3_freeze"] = 1
    payload["access_history_note"] = (
        "Two FINAL_TEST_ACCESS_CONSUMED events exist in the full project "
        "history (see final_test_access.history below and "
        "artifacts/governance/stage5_final_test_access_incident_v1.json). "
        "The first (2026-09-24T06:55:44Z) was consumed under the original "
        "G3 freeze and crashed before any test-partition row was read, "
        "before any inference, and before any metric/bootstrap computation "
        "-- zero scientific test information was exposed. The project owner "
        "then explicitly authorized fixing the defect and formally "
        "resetting G3 (vedant_infra.g3.invalidate_for_reset, reason logged "
        "in final_test_access.history), after which a replacement G3 "
        "(byte-identical scientific dependency bindings, new freeze "
        "timestamp) was created. The second event "
        "(2026-09-24T07:08:39Z), consumed under that replacement G3, is the "
        "one successful scientific final-test evaluation reported "
        "throughout this artifact and docs/vedant/FINAL_TEST_EVALUATION_REVIEW.md."
    )

    g4_path.write_bytes(canonical_json_bytes(payload))
    after = json.loads(g4_path.read_text())

    print("=== fields changed ===")
    print("registry.artifacts_sha256: " + before["registry"]["artifacts_sha256"] + " -> " + after["registry"]["artifacts_sha256"])
    print("registry.registry_sha256: " + before["registry"]["registry_sha256"] + " -> " + after["registry"]["registry_sha256"])
    print("added: total_historical_consumption_events=" + str(after["total_historical_consumption_events"]))
    print("added: successful_scientific_evaluation_count=" + str(after["successful_scientific_evaluation_count"]))
    print("added: consumption_events_since_current_g3_freeze=" + str(after["consumption_events_since_current_g3_freeze"]))

    print("=== scientific-output hash immutability re-verification (from disk, post-write) ===")
    all_unchanged = True
    for path in SCIENTIFIC_HASH_FIELDS:
        after_hash = _get(after, path)["sha256"]
        ok = after_hash == before_scientific[path]
        all_unchanged = all_unchanged and ok
        print(str(path) + ": unchanged=" + str(ok))
    ea_ok = (after["error_analysis_artifact"]["csv_sha256"], after["error_analysis_artifact"]["json_sha256"]) == before_error_analysis
    print("error_analysis_artifact: unchanged=" + str(ea_ok))
    g3_ok = after["parents"]["g3_freeze"]["sha256"] == before_g3
    print("parents.g3_freeze: unchanged=" + str(g3_ok))
    sm_ok = after["parents"]["selected_models_v1"]["sha256"] == before_selected_models
    print("parents.selected_models_v1: unchanged=" + str(sm_ok))
    naive_ok = after["parents"]["naive_baseline"]["sha256"] == before_naive
    print("parents.naive_baseline: unchanged=" + str(naive_ok))
    all_unchanged = all_unchanged and ea_ok and g3_ok and sm_ok and naive_ok
    print("ALL_SCIENTIFIC_HASHES_UNCHANGED=" + str(all_unchanged))
    print("g4_file_sha256=" + sha256_file(g4_path))
    if not all_unchanged:
        raise SystemExit("REFUSING: a scientific hash changed unexpectedly")


if __name__ == "__main__":
    main()
