"""Capture release evidence from the ACTUAL running V2 dashboard (a real
request/response cycle through the real FastAPI app, via TestClient -- not
a mockup, not hand-written HTML). Saved as rendered HTML pages, matching the
project's existing evidence convention
(docs/evidence/system/dashboard/dashboard_render_synthetic_v1.html).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from data.synthetic.validation import load_jsonl  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "docs/evidence/dashboard_v2"


def main() -> None:
    from fastapi.testclient import TestClient

    from api.v2_app import build_v2_app
    from dashboard.v2_api_client import InProcessV2DashboardAPIClient
    from dashboard.v2_app import build_v2_dashboard_app

    api_client_http = TestClient(build_v2_app(ROOT))
    api_client = InProcessV2DashboardAPIClient(api_client_http)

    demo = json.loads((ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").read_text())
    demo_stays = {item["stay_id"] for item in demo["demo_subjects"]}
    events = load_jsonl(ROOT / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_timeline.jsonl")
    raw_events_by_stay = {}
    for row in events:
        if row["stay_id"] in demo_stays:
            raw_events_by_stay.setdefault(row["stay_id"], []).append(row)

    dashboard_app = build_v2_dashboard_app(root=ROOT, api_client=api_client, raw_events_by_stay=raw_events_by_stay)
    client = TestClient(dashboard_app)

    stay = demo["demo_subjects"][0]
    stay_id = stay["stay_id"]
    cutoffs = stay["legal_cutoffs"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captures = []

    def capture(name: str, params: dict):
        response = client.get("/", params=params)
        path = OUT_DIR / f"{name}.html"
        path.write_text(response.text, encoding="utf-8")
        captures.append({"name": name, "path": str(path.relative_to(ROOT)), "status_code": response.status_code, "sha256": sha256_file(path)})
        print(name, response.status_code, path)

    capture("01_landing_patient_replay", {"view": "replay", "stay_id": stay_id, "prediction_time": cutoffs[0]})
    capture("02_recovery_trajectory_mid_replay", {"view": "replay", "stay_id": stay_id, "prediction_time": cutoffs[len(cutoffs) // 2]})
    capture("03_organ_support_risk_panel", {"view": "replay", "stay_id": stay_id, "prediction_time": cutoffs[-1]})
    capture("04_retrospective_replay_cutoff_t0", {"view": "replay", "stay_id": stay_id, "prediction_time": cutoffs[0]})
    capture("05_retrospective_replay_cutoff_t_last", {"view": "replay", "stay_id": stay_id, "prediction_time": cutoffs[-1]})
    capture("06_forecast_details", {"view": "forecast", "stay_id": stay_id, "prediction_time": cutoffs[-1]})
    capture("07_explainability_panel", {"view": "explainability", "stay_id": stay_id, "prediction_time": cutoffs[-1]})
    capture("08_model_performance_page", {"view": "performance"})
    capture("09_data_quality_provenance_page", {"view": "dataquality", "stay_id": stay_id, "prediction_time": cutoffs[-1]})

    manifest = {
        "status": "V2_DASHBOARD_EVIDENCE_CAPTURED",
        "artifact_version": "v2_dashboard_evidence_manifest_v1",
        "capture_method": "fastapi.testclient.TestClient against the real V2 API and V2 dashboard apps (genuine request/response cycle, not a mockup)",
        "demo_stay_id": stay_id,
        "captures": captures,
    }
    manifest_path = OUT_DIR / "capture_manifest_v1.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print("wrote", manifest_path, sha256_file(manifest_path))


if __name__ == "__main__":
    main()
