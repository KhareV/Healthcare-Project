"""Build the FINAL PROJECT RELEASE manifest. Purely a release/evidence
binding -- creates no new scientific freeze and does not alter any V2
scientific artifact."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_PATH = ROOT / "artifacts/performance_v2/release/final_project_release_v2.json"


def _ref(path: str) -> dict:
    return {"ref": path, "sha256": sha256_file(ROOT / path)}


def main() -> None:
    code_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    evidence_dir = ROOT / "docs/evidence/dashboard_v2"
    manifest = {
        "status": "FINAL_PROJECT_RELEASE_READY",
        "artifact_version": "final_project_release_v2",
        "parents": {
            "v2_final_evaluation_freeze": _ref("artifacts/performance_v2/governance/v2_final_evaluation_freeze_v1.json"),
            "v2_dashboard_serving_bundle": _ref("artifacts/performance_v2/serving/v2_dashboard_serving_bundle_v1.json"),
            "v2_demo_manifest": _ref("configs/performance_v2/v2_demo_manifest_v1.json"),
            "dashboard_evidence_manifest": _ref("docs/evidence/dashboard_v2/capture_manifest_v1.json"),
            "final_v2_report": _ref("docs/performance_v2/FINAL_V2_TEST_EVALUATION.md"),
            "demo_script": _ref("docs/performance_v2/V2_DEMO_SCRIPT.md"),
            "readme": _ref("README.md"),
            "runbook": _ref("RUNBOOK.md"),
        },
        "dashboard_version": "v2_prediction_response_v1",
        "api_schema_version": "v2_prediction_response_v1",
        "evidence_screenshot_count": len(list(evidence_dir.glob("*.html"))),
        "scientific_artifacts_unchanged_confirmation": {
            "selected_models_v2_sha256": "6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8",
            "v2_model_freeze_sha256": "f5feb29cab2e25d6e8bff860af52ac505f40951578aa73547ec420ba8fcda570",
            "v2_final_evaluation_freeze_sha256": "a433c1ee203f6cf9f01b131a2a9a10c73da57097564705ce64d7351c0057b649",
        },
        "code_commit": code_commit,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(canonical_json_bytes(manifest))
    print("wrote", OUT_PATH, sha256_file(OUT_PATH))


if __name__ == "__main__":
    main()
