"""Performance-v2 Phase 4 preflight: validate EVERYTHING possible without
opening fresh-test row content. Only structural/hash/load checks. Must be
PASS on every item before access is authorized.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from evaluation.bootstrap import grouped_bootstrap  # noqa: E402
from evaluation.metrics import PredictionRecord, evaluate_recovery_horizon  # noqa: E402
from performance_v2.final_evaluation import load_xgb_model, verify_feature_dimension  # noqa: E402
from performance_v2.fresh_test_access import (  # noqa: E402
    SEALED_NOT_ACCESSED,
    current_state,
    has_ever_consumed_access,
)
from vedant_infra.hashing import sha256_file  # noqa: E402

PHASE3_DIR = ROOT / "artifacts/performance_v2/phase3"
GOV_DIR = ROOT / "artifacts/performance_v2/governance"
PHASE4_DIR = ROOT / "artifacts/performance_v2/phase4"
FRESH_DIR = PHASE3_DIR / "fresh_test_cohort"

EXPECTED_HASHES = {
    "artifacts/performance_v2/phase3/selected_models_v2.json": "6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8",
    "artifacts/performance_v2/governance/v2_model_freeze_v1.json": "f5feb29cab2e25d6e8bff860af52ac505f40951578aa73547ec420ba8fcda570",
    "artifacts/performance_v2/phase3/calibration/isotonic_support_v2.json": "1ef311fd8a81a3e392db8a4cb7668eb4f404b0d21d818a94fc5ce308fdfe0b4f",
    "artifacts/performance_v2/phase3/thresholds/support_threshold_v2.json": "7d14ac249c3ee6b7dcb04110b6aa7ba274b27c73f3ced6bb7aae614d68a9d77c",
    "artifacts/performance_v2/phase3/fresh_test_cohort_spec_v1.json": "b25789da0af513e71e9ca03040f305cae4e09b52b8ea1a2fee1abab916219377",
    "artifacts/performance_v2/governance/v2_fresh_test_freeze_v1.json": "f70d4cc7fca69981861c8d679101176529f036579ba1849f742e6c3fc17916ce",
    "artifacts/performance_v2/governance/generator_equivalence_audit_v1.json": "bd95e38f16eba014ef881864005ce3936a91518204e9a502080f6b74458578f8",
}

MODEL_HASHES = {
    "artifacts/performance_v2/phase3/models/recovery24_xgb_v2.json": "b355c26339271f0b84c123eb46cfb61c8df0cc9c9c6fcfcbb86c29047d10ad3c",
    "artifacts/performance_v2/phase3/models/recovery48_xgb_v2.json": "5dda6bb0830428b44b3a7cf554dc7f6d2adc278c7b218045d1df8ff960f6c701",
    "artifacts/performance_v2/phase3/models/icu_time_xgb_v2.json": "918c81eef86bd89e91e02bb06c254774375f3f2371c4f3c5200c270dea24f7e6",
    "artifacts/performance_v2/phase3/models/organ_support_xgb_v2.json": "2fa653a0d2c644d8a1de806146f8e8cba46df87e6bd037868122276b4a440671",
}

PIPELINE_MANIFESTS = [
    FRESH_DIR / "raw/synthetic_dataset_manifest_v1.json",
    FRESH_DIR / "cohort/synthetic_cohort_manifest_v2.json",
    FRESH_DIR / "timeline/synthetic_processed_manifest_v1.json",
    FRESH_DIR / "features/synthetic_feature_input_manifest_v1.json",
    FRESH_DIR / "pre_split/phase9_pre_split_manifest_v1.json",
]


class PreflightError(RuntimeError):
    pass


def main() -> None:
    items = []

    def check(name, fn):
        try:
            detail = fn()
            items.append({"name": name, "status": "PASS", "detail": str(detail)})
        except Exception as error:  # noqa: BLE001
            items.append({"name": name, "status": "BLOCKED", "detail": f"{type(error).__name__}: {error}"})

    def top_level_hashes():
        for path, expected in EXPECTED_HASHES.items():
            full = ROOT / path
            if not full.is_file():
                raise PreflightError(f"missing: {path}")
            actual = sha256_file(full)
            if actual != expected:
                raise PreflightError(f"hash mismatch: {path}")
        return f"{len(EXPECTED_HASHES)} artifacts hash-verified"

    check("parent_artifact_paths_and_hashes", top_level_hashes)

    def model_hashes():
        for path, expected in MODEL_HASHES.items():
            full = ROOT / path
            if not full.is_file():
                raise PreflightError(f"missing model: {path}")
            if sha256_file(full) != expected:
                raise PreflightError(f"model hash mismatch: {path}")
        return f"{len(MODEL_HASHES)} model files hash-verified"

    check("model_file_hashes", model_hashes)

    def selected_models_validates():
        sm = json.loads((PHASE3_DIR / "selected_models_v2.json").read_text())
        if sm["status"] != "V2_SELECTED_MODELS_FROZEN_PRE_TEST":
            raise PreflightError("selected_models_v2 status unexpected: " + sm["status"])
        if sm["tasks"]["recovery48"]["feature_variant"] != "B_MIN":
            raise PreflightError("recovery48 feature_variant is not B_MIN")
        return sm["status"]

    check("selected_models_v2_validates", selected_models_validates)

    def model_freeze_validates():
        mf = json.loads((GOV_DIR / "v2_model_freeze_v1.json").read_text())
        if mf["status"] != "V2_MODEL_FREEZE_ACTIVE":
            raise PreflightError("model freeze status unexpected: " + mf["status"])
        return mf["status"]

    check("model_freeze_validates", model_freeze_validates)

    def fresh_test_freeze_validates():
        ftf = json.loads((GOV_DIR / "v2_fresh_test_freeze_v1.json").read_text())
        if ftf["status"] != "V2_FRESH_TEST_FROZEN":
            raise PreflightError("fresh-test freeze status unexpected: " + ftf["status"])
        return ftf["status"]

    check("fresh_test_freeze_validates", fresh_test_freeze_validates)

    def bulk_files_exist_and_hash_match():
        n_checked = 0
        for manifest_path in PIPELINE_MANIFESTS:
            if not manifest_path.is_file():
                raise PreflightError("missing pipeline manifest: " + str(manifest_path))
            manifest = json.loads(manifest_path.read_text())
            for artifact in manifest["artifacts"]:
                relative = artifact.get("path") or artifact.get("repository_relative_path")
                expected = artifact.get("sha256")
                full = ROOT / relative
                if not full.is_file():
                    raise PreflightError("missing bulk file: " + relative)
                actual = sha256_file(full)
                if actual != expected:
                    raise PreflightError("bulk file hash mismatch: " + relative)
                n_checked += 1
        return f"{n_checked} bulk files hash-verified against their frozen pipeline manifests"

    check("fresh_cohort_bulk_files_hash_match", bulk_files_exist_and_hash_match)

    def access_state_sealed():
        state = current_state(ROOT)
        if state != SEALED_NOT_ACCESSED:
            raise PreflightError("access state is not SEALED_NOT_ACCESSED: " + state)
        if has_ever_consumed_access(ROOT):
            raise PreflightError("a prior FINAL_V2_TEST_ACCESS_CONSUMED event already exists")
        return state

    check("access_state_sealed_not_accessed", access_state_sealed)

    def no_prior_predictions():
        pred_dir = PHASE4_DIR / "predictions"
        if pred_dir.exists() and any(pred_dir.iterdir()):
            raise PreflightError("prediction artifacts already exist")
        return "no prior prediction artifacts"

    check("no_prior_final_v2_prediction_artifact", no_prior_predictions)

    def no_prior_metrics():
        for name in ("metrics", "bootstrap"):
            d = PHASE4_DIR / name
            if d.exists() and any(d.iterdir()):
                raise PreflightError(f"{name} artifacts already exist")
        eval_freeze = GOV_DIR / "v2_final_evaluation_freeze_v1.json"
        if eval_freeze.exists():
            raise PreflightError("v2_final_evaluation_freeze_v1.json already exists")
        return "no prior metric/bootstrap/final-freeze artifacts"

    check("no_prior_final_v2_metric_artifact", no_prior_metrics)

    def models_load():
        for task in ("recovery24", "recovery48", "icu_stay_time", "organ_support"):
            load_xgb_model(ROOT, task)
        return "all 4 model files load"

    check("model_files_load", models_load)

    def calibrator_loads():
        payload = json.loads((PHASE3_DIR / "calibration/isotonic_support_v2.json").read_text())
        knots = payload["calibration_knots"]
        if not knots["x_thresholds"] or len(knots["x_thresholds"]) != len(knots["y_thresholds"]):
            raise PreflightError("calibrator knots malformed")
        return f"{len(knots['x_thresholds'])} knots"

    check("calibrator_loads", calibrator_loads)

    def threshold_loads():
        payload = json.loads((PHASE3_DIR / "thresholds/support_threshold_v2.json").read_text())
        value = payload["threshold_value"]
        if not (0.0 <= value <= 1.0) or abs(value - 0.39781983118092895) > 1e-12:
            raise PreflightError("threshold value unexpected")
        return value

    check("threshold_loads", threshold_loads)

    def preprocessing_loads():
        for variant in ("B_MIN", "B_FULL", "B_PLUS_F"):
            path = PHASE3_DIR / f"preprocessing/context_normalization_v2_{variant}.json"
            payload = json.loads(path.read_text())
            if "mean" not in payload or "std" not in payload:
                raise PreflightError(f"{variant} preprocessing malformed")
        return "3 variants load"

    check("preprocessing_artifacts_load", preprocessing_loads)

    def adapters_initialize():
        results = [verify_feature_dimension(ROOT, task) for task in ("recovery24", "recovery48", "icu_stay_time", "organ_support")]
        return results

    check("prediction_adapters_initialize", adapters_initialize)

    def evaluation_code_initializes():
        import evaluation.bootstrap  # noqa: F401
        import evaluation.metrics  # noqa: F401
        import evaluation.weighted_stats  # noqa: F401
        import evaluation.weights  # noqa: F401

        return "evaluation modules import cleanly"

    check("evaluation_code_initializes", evaluation_code_initializes)

    def output_dirs_writable():
        for sub in ("predictions", "metrics", "bootstrap"):
            d = PHASE4_DIR / sub
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
        return "phase4 output subdirectories writable"

    check("output_directories_writable", output_dirs_writable)

    def bootstrap_dummy_fixture():
        dummy = tuple(
            PredictionRecord(stay_id=f"DUMMY-{i}", prediction_time=f"2026-01-01T00:0{i}:00Z", target=float(i % 3), prediction=float(i % 3) + 0.1, eligible=True)
            for i in range(12)
        )

        def metric_fn(rows):
            return evaluate_recovery_horizon(rows, horizon="24h").metrics["mae"]

        result = grouped_bootstrap(dummy, metric_fn=metric_fn, metric_name="mae", task="recovery", n_bootstrap=25, seed=1, horizon="24h")
        if result.n_valid_replicates == 0:
            raise PreflightError("dummy bootstrap produced zero valid replicates")
        return f"dummy bootstrap OK: {result.n_valid_replicates}/{result.n_requested_replicates} valid"

    check("bootstrap_dummy_fixture_works", bootstrap_dummy_fixture)

    def atomic_transaction_probe():
        with tempfile.TemporaryDirectory(dir=PHASE4_DIR) as tmp:
            tmp_path = Path(tmp) / "probe.json"
            tmp_path.write_bytes(canonical_json_bytes({"probe": True}))
            if not tmp_path.is_file():
                raise PreflightError("temporary artifact write failed")
        return "temporary output directory write/cleanup OK"

    check("final_run_transaction_can_write_temp_artifacts", atomic_transaction_probe)

    def state_machine_design_check():
        import inspect

        import performance_v2.fresh_test_access as fta

        source = inspect.getsource(fta.consume_access)
        if "SEALED_NOT_ACCESSED" in source:
            raise PreflightError("consume_access incorrectly re-checks SEALED_NOT_ACCESSED after authorization")
        return "consume_access requires only AUTHORIZED_FOR_ONE_FINAL_RUN, no post-consumption SEALED_NOT_ACCESSED re-check"

    check("state_machine_transition_design", state_machine_design_check)

    blockers = [item for item in items if item["status"] != "PASS"]
    report = {
        "status": "PREFLIGHT_PASS" if not blockers else "PREFLIGHT_BLOCKED",
        "artifact_version": "performance_v2_phase4_preflight_v1",
        "items": items,
        "n_items": len(items),
        "n_pass": len(items) - len(blockers),
        "n_blocked": len(blockers),
        "fresh_test_rows_read": False,
    }
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PHASE4_DIR / "preflight_report_v1.json"
    report_path.write_bytes(canonical_json_bytes(report))
    print("wrote", report_path, sha256_file(report_path))
    for item in items:
        print(f"  [{item['status']}] {item['name']}: {item['detail']}")
    print(report["status"])
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
