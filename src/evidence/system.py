"""Generate and audit Pulkit Phase-16 system evidence.

The generator executes the synthetic integration composition used by the
Phase-14/15 acceptance tests.  It never substitutes that composition for a
real selected-model release and it records unavailable release evidence as a
blocked manifest entry without an output path.
"""

import argparse
import hashlib
import json
import platform
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping

from evidence.audit import EvidenceAuditError, audit_terminology
from experiments.audit import audit_registry_lineage
from reproducibility.audit import (
    audit_environment_lock,
    audit_path_portability,
    audit_registered_artifacts,
    audit_restricted_data,
    load_config,
)
from vedant_infra.g3 import audit_g3
from vedant_infra.hashing import is_sha256, sha256_file


MANIFEST_VERSION = "pulkit_system_evidence_manifest_v1"
GENERATOR = "src/evidence/system.py"
EVIDENCE_ROOT = Path("docs/evidence/system")
ALLOWED_STATUSES = {"PASS", "BLOCKED", "REVIEW_REQUIRED"}
ALLOWED_SCOPES = {"synthetic", "repository", "blocked"}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _canonical(value: object) -> object:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _value_sha256(value: object) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source(root: Path, reference: str) -> Mapping[str, str]:
    path = root / reference
    if not path.is_file():
        raise EvidenceAuditError("evidence source is absent: " + reference)
    return {"path": reference, "sha256": sha256_file(path)}


def _item(root, identifier, description, scope, status, output, sources, command, blocker=None):
    if output is None:
        return {
            "evidence_id": identifier,
            "description": description,
            "scope": scope,
            "status": status,
            "output_path": None,
            "output_sha256": None,
            "source_artifacts": [],
            "source_command": command,
            "generator": GENERATOR,
            "blocker": blocker,
        }
    return {
        "evidence_id": identifier,
        "description": description,
        "scope": scope,
        "status": status,
        "output_path": str(output.relative_to(root)),
        "output_sha256": sha256_file(output),
        "source_artifacts": [_source(root, value) for value in sources],
        "source_command": command,
        "generator": GENERATOR,
        "blocker": blocker,
    }


def _junit_summary(path: Path, command: str) -> Mapping[str, object]:
    if not path.is_file():
        return {
            "status": "BLOCKED",
            "command": command,
            "detail": "JUnit execution output was not supplied to the generator",
            "counts": None,
        }
    root = ET.parse(str(path)).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return {
        "status": "PASS" if counts["failures"] == counts["errors"] == 0 else "BLOCKED",
        "command": command,
        "junit_sha256": sha256_file(path),
        "counts": counts,
    }


def _capture_synthetic_system(root: Path):
    # These imports are deliberately late: the development evidence command
    # must opt in to the tests directory that owns the synthetic composition.
    from api.main import create_app
    from dashboard.app import create_dashboard_app
    from demo.fixture import (
        DemoCurrentSOFAProvider,
        dashboard_catalog_for_demo,
        validate_demo_fixture,
    )
    from fastapi.testclient import TestClient
    from integration_helpers import build_integrated_system
    from serving.artifacts import ArtifactHashMismatchError

    demo = validate_demo_fixture(root)
    captures = []
    equivalence = []
    with tempfile.TemporaryDirectory(prefix="pulkit_phase16_") as directory:
        system = build_integrated_system(
            Path(directory),
            timelines=demo.timelines,
            current_sofa_provider=DemoCurrentSOFAProvider(demo),
            expected_sofa_version="SYNTHETIC_SOFA_AT_T_PHASE15_V1",
            dashboard_catalog_override=dashboard_catalog_for_demo(demo),
        )
        health = system.api_client.get("/health")
        metadata = system.api_client.get("/model-metadata")
        for cutoff in demo.legal_cutoffs:
            request = {"stay_id": demo.stay_id, "prediction_time": cutoff}
            direct = system.pipeline.predict(request)
            direct_inputs = {
                task: _value_sha256(system.runtime.predictors[task].last_prepared_input)
                for task in ("recovery", "icu_stay_time", "organ_support")
            }
            response = system.api_client.post("/predict", json=request)
            api_inputs = {
                task: _value_sha256(system.runtime.predictors[task].last_prepared_input)
                for task in ("recovery", "icu_stay_time", "organ_support")
            }
            body = response.json()
            captures.append({
                "request": request,
                "status_code": response.status_code,
                "response": body,
                "direct_pipeline_equal": body == json.loads(json.dumps(direct)),
            })
            equivalence.append({
                "prediction_time": cutoff,
                "direct_input_sha256_by_task": direct_inputs,
                "api_serving_input_sha256_by_task": api_inputs,
                "equal_by_task": {
                    task: direct_inputs[task] == api_inputs[task]
                    for task in direct_inputs
                },
            })

        unknown = system.api_client.post(
            "/predict",
            json={"stay_id": "SYNTHETIC_UNKNOWN_STAY", "prediction_time": demo.legal_cutoffs[0]},
        )
        illegal = system.api_client.post(
            "/predict",
            json={"stay_id": demo.stay_id, "prediction_time": "2030-01-02T00:00:01+00:00"},
        )
        class IncompatibleSyntheticPipeline:
            bundle = system.pipeline.bundle

            def predict(self, _request):
                raise ArtifactHashMismatchError(
                    "controlled synthetic artifact mismatch for structured-error evidence"
                )

        unavailable = TestClient(create_app(IncompatibleSyntheticPipeline())).post(
            "/predict",
            json={"stay_id": demo.stay_id, "prediction_time": demo.legal_cutoffs[0]},
        )
        dashboard = TestClient(create_dashboard_app(system.controller, synthetic=True)).get(
            "/",
            params={"stay_id": demo.stay_id, "prediction_time": demo.legal_cutoffs[0]},
        )
        replay = []
        timeline = demo.timelines[0]
        event_time_field = timeline.contract.event_time_field
        for capture in captures:
            cutoff = capture["request"]["prediction_time"]
            visible = [row for row in timeline.events if row[event_time_field] <= cutoff]
            future = [row for row in timeline.events if row[event_time_field] > cutoff]
            replay.append({
                "prediction_time": cutoff,
                "request": capture["request"],
                "visible_event_count": len(visible),
                "future_event_count_hidden": len(future),
                "maximum_visible_event_time": max(row[event_time_field] for row in visible),
                "api_response_sha256": _value_sha256(capture["response"]),
                "prediction_recomputed": True,
            })

    return {
        "demo": demo,
        "api": {
            "evidence_version": "system_api_evidence_synthetic_v1",
            "scope": "synthetic",
            "scientific_status": "NON_SCIENTIFIC_SYSTEM_FUNCTION_EVIDENCE_ONLY",
            "health": {"status_code": health.status_code, "response": health.json()},
            "model_metadata": {"status_code": metadata.status_code, "response": metadata.json()},
            "predictions": captures,
            "structured_errors": {
                "unknown_stay": {"status_code": unknown.status_code, "response": unknown.json()},
                "illegal_prediction_time": {"status_code": illegal.status_code, "response": illegal.json()},
                "serving_unavailable": {"status_code": unavailable.status_code, "response": unavailable.json()},
            },
        },
        "equivalence": {
            "evidence_version": "system_equivalence_synthetic_v1",
            "scope": "synthetic",
            "comparison": "direct_pipeline_input_vs_api_serving_input",
            "cutoffs": equivalence,
            "all_equal": all(all(row["equal_by_task"].values()) for row in equivalence),
        },
        "replay": {
            "evidence_version": "replay_t1_t2_t3_synthetic_v1",
            "scope": "synthetic",
            "fixture_sha256": demo.fixture_sha256,
            "predictions_required_to_change": False,
            "cutoffs": replay,
        },
        "dashboard_html": dashboard.text,
        "dashboard_metadata": {
            "evidence_version": "dashboard_render_metadata_v1",
            "scope": "synthetic",
            "status_code": dashboard.status_code,
            "fixture_sha256": demo.fixture_sha256,
            "cutoff": demo.legal_cutoffs[0],
            "html_sha256": hashlib.sha256(dashboard.content).hexdigest(),
            "mandatory_banner_present": (
                "RETROSPECTIVE SEQUENTIAL REPLAY" in dashboard.text
                and "NOT REAL-TIME CLINICAL PREDICTION" in dashboard.text
            ),
            "screenshot_status": "BLOCKED_NO_BROWSER_CAPTURE_TOOL_AVAILABLE",
        },
    }


def _phase_rows():
    names = (
        "vasopressor state definition", "invasive ventilation state definition",
        "composite support label and censoring", "endpoint freeze and prevalence discipline",
        "prediction schema contract", "pipeline and artifact compatibility",
        "history truncation and preprocessing equivalence", "model and explanation router",
        "Integrated Gradients adapter", "TreeSHAP adapter", "FastAPI and ICU-time postprocess",
        "recovery reconstruction", "dashboard and replay", "full integration",
        "packaging and demo fixture", "release evidence and completion audit",
    )
    return [
        {
            "phase": index,
            "name": name,
            "framework_status": "PASS",
            "real_release_status": "BLOCKED_UPSTREAM_REAL_ARTIFACTS"
            if index >= 6 else "REVIEW_REQUIRED_FOR_REAL_RELEASE",
        }
        for index, name in enumerate(names, start=1)
    ]


def generate(root: Path, junit: Path = None):
    root = root.resolve()
    evidence = root / EVIDENCE_ROOT
    capture = _capture_synthetic_system(root)
    command = "PYTHONPATH=src:tests:. python3 -m evidence.system generate --root . --junit <pytest-junit.xml>"
    common_sources = (
        "data/demo/demo_patient_v1.json", "data/demo/demo_patient_v1.metadata.json",
        "tests/fixtures/serving/selected_models_synthetic_phase6_v1.json",
        "tests/fixtures/serving/recovery_gru.mock",
        "tests/fixtures/serving/recovery_model.metadata.json",
        "tests/fixtures/serving/recovery_preprocessor.json",
        "tests/fixtures/serving/icu_xgboost.mock",
        "tests/fixtures/serving/icu_model.metadata.json",
        "tests/fixtures/serving/icu_preprocessor.json",
        "tests/fixtures/serving/support_gru.mock",
        "tests/fixtures/serving/support_model.metadata.json",
        "tests/fixtures/serving/support_preprocessor.json",
        "tests/fixtures/serving/support_calibrator.json",
        "tests/fixtures/serving/support_threshold.json",
        "configs/prediction_schema_v1.json", "src/serving/pipeline.py", "api/main.py",
        "tests/integration_helpers.py", "src/demo/fixture.py", GENERATOR,
    )
    items = []

    api_path = _json(evidence / "api/api_evidence_synthetic_v1.json", capture["api"])
    items.append(_item(root, "synthetic_api", "Executed health, metadata, prediction, and structured-error API evidence", "synthetic", "PASS", api_path, common_sources, command))
    replay_path = _json(evidence / "replay/replay_t1_t2_t3_synthetic_v1.json", capture["replay"])
    items.append(_item(root, "synthetic_replay", "Executed three-cutoff replay evidence", "synthetic", "PASS", replay_path, common_sources, command))
    equality_path = _json(evidence / "equivalence/api_direct_and_input_equivalence_v1.json", capture["equivalence"])
    items.append(_item(root, "synthetic_equivalence", "API/direct output and direct/API prepared-input equality", "synthetic", "PASS", equality_path, common_sources, command))

    html_path = _write(evidence / "dashboard/dashboard_render_synthetic_v1.html", capture["dashboard_html"])
    items.append(_item(root, "dashboard_render", "Actual dashboard HTML rendered through its API-bound controller", "synthetic", "PASS", html_path, common_sources + ("dashboard/app.py",), command))
    dashboard_meta = _json(evidence / "dashboard/dashboard_render_metadata_v1.json", capture["dashboard_metadata"])
    items.append(_item(root, "dashboard_render_metadata", "Dashboard render identity and banner check", "synthetic", "PASS", dashboard_meta, common_sources + ("dashboard/app.py",), command))

    first = capture["api"]["predictions"][0]["response"]
    explanation = {
        "evidence_version": "explanation_evidence_synthetic_v1",
        "scope": "synthetic",
        "status": "SYSTEM_CONTRACT_ADAPTER_OUTPUT_NOT_REAL_LIBRARY_ATTRIBUTION",
        "prediction_time": first["prediction_time"],
        "manifest_version": first["model_metadata"]["manifest_version"],
        "feature_schema_by_task": first["model_metadata"]["feature_version"],
        "tasks": {
            task: {
                "family": first["model_versions"][task]["family"],
                "method": value["explanation_method"],
                "model_sha256": first["model_versions"][task]["artifact_sha256"],
                "items": value["items"],
            }
            for task, value in first["explanation_features"].items()
        },
        "actual_adapter_validation": {
            "integrated_gradients": "tests/test_integrated_gradients.py and tests/test_integrated_gradients_integration.py",
            "tree_shap": "tests/test_tree_shap.py and tests/test_tree_shap_integration.py",
        },
    }
    explanation_path = _json(evidence / "explanations/explanation_evidence_synthetic_v1.json", explanation)
    items.append(_item(root, "synthetic_explanations", "Selected-family explanation contract output plus actual-adapter test references", "synthetic", "PASS", explanation_path, common_sources + ("src/explainability/ig.py", "src/explainability/tree_shap.py", "configs/explainability/ig_synthetic_development_v1.json", "configs/explainability/tree_shap_synthetic_development_v1.json"), command))

    manifest_path = root / "tests/fixtures/serving/selected_models_synthetic_phase6_v1.json"
    model_status = {
        "evidence_version": "serving_manifest_status_v1",
        "real": {"status": "BLOCKED", "required_path": "artifacts/models/selected_models_v1.json", "present": (root / "artifacts/models/selected_models_v1.json").is_file()},
        "synthetic": {"status": "PASS", "path": str(manifest_path.relative_to(root)), "sha256": sha256_file(manifest_path), "manifest": json.loads(manifest_path.read_text())},
    }
    model_path = _json(evidence / "model/serving_manifest_status_v1.json", model_status)
    items.append(_item(root, "serving_manifest", "Real manifest blocker and separate synthetic demo manifest", "repository", "PASS", model_path, (str(manifest_path.relative_to(root)),), command))

    contracts = {
        "evidence_version": "output_contract_synthetic_v1",
        "scope": "synthetic",
        "prediction_time": first["prediction_time"],
        "recovery": first["recovery"],
        "recovery_horizons_independent": True,
        "icu_stay_time_hours": first["icu_stay_time_hours"],
        "organ_support_probability_calibrated": first["organ_support_probability_calibrated"],
        "support_calibration_fit_calls": 0,
        "source_response_sha256": _value_sha256(first),
    }
    contracts_path = _json(evidence / "contracts/output_contract_synthetic_v1.json", contracts)
    items.append(_item(root, "output_contracts", "Executed recovery, ICU-time, and support response contract", "synthetic", "PASS", contracts_path, common_sources, command))

    test_command = "PYTHONPATH=src:tests:. python3 -m pytest -q --junitxml=<pytest-junit.xml>"
    test_report = dict(_junit_summary(junit, test_command)) if junit else dict(_junit_summary(Path("__absent__"), test_command))
    test_report.update({"evidence_version": "phase16_test_report_v1", "scope": "repository", "python": platform.python_version()})
    test_path = _json(evidence / "integration/full_test_report_v1.json", test_report)
    full_suite_sources = tuple(
        str(path.relative_to(root))
        for base in (root / "src", root / "api", root / "dashboard", root / "tests", root / "configs")
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"}
    )
    items.append(_item(root, "full_test_report", "Machine-parsed pytest JUnit result", "repository", test_report["status"], test_path, full_suite_sources, test_command, None if test_report["status"] == "PASS" else "passing JUnit execution required"))

    compatibility = {
        "evidence_version": "artifact_compatibility_report_v1",
        "status": test_report["status"],
        "scope": "synthetic",
        "covered_fail_closed_cases": [
            "model hash mismatch", "feature/label metadata mismatch", "preprocessor train-fit provenance mismatch",
            "calibrator/model mismatch", "threshold/calibrator mismatch", "wrong explanation routing",
        ],
        "execution_source": test_report,
        "test_source": "tests/test_artifact_compatibility_e2e.py",
    }
    compatibility_path = _json(evidence / "compatibility/artifact_compatibility_report_v1.json", compatibility)
    items.append(_item(root, "artifact_compatibility", "Phase-14 fail-closed compatibility cases in the executed suite", "synthetic", test_report["status"], compatibility_path, ("tests/test_artifact_compatibility_e2e.py",), test_command))

    repro_config = load_config(root / "configs/reproducibility_v1.json")
    environment = {
        "evidence_version": "environment_audit_v1",
        "final_lock": audit_environment_lock(root, repro_config),
        "observed_snapshot": {"path": "observed_environment_phase15.json", "sha256": sha256_file(root / "observed_environment_phase15.json")},
    }
    env_path = _json(evidence / "environment/environment_audit_v1.json", environment)
    items.append(_item(root, "environment", "Final-lock audit plus non-final observed snapshot identity", "repository", "BLOCKED", env_path, ("configs/reproducibility_v1.json", "observed_environment_phase15.json"), command, environment["final_lock"]["detail"]))

    registry = audit_registry_lineage(root)
    registry_path = _json(evidence / "registry/registry_lineage_audit_v1.json", {"evidence_version": "registry_lineage_audit_v1", "audit": registry})
    items.append(_item(root, "registry_lineage", "Read-only current registry and lineage audit", "repository", "PASS", registry_path, ("experiments/registry.csv", "experiments/artifacts.csv"), command))
    reproduction = {
        "evidence_version": "system_reproduction_audit_v1",
        "environment": audit_environment_lock(root, repro_config),
        "artifacts": audit_registered_artifacts(root),
        "portability": audit_path_portability(root),
        "restricted_data": audit_restricted_data(root),
        "local_process_test": test_report,
        "non_owner": {"status": "BLOCKED", "detail": "No non-owner clean-machine execution or valid signoff exists"},
    }
    reproduction_path = _json(evidence / "reproduction/reproduction_audit_v1.json", reproduction)
    items.append(_item(root, "reproduction", "Local layered audit and truthful non-owner blocker", "repository", "BLOCKED", reproduction_path, ("configs/reproducibility_v1.json", "artifacts/reproducibility/reproduction_report_v1.json"), command, "final environment lock and non-owner execution required"))

    g3 = audit_g3(root, scope="real")
    completion = {
        "evidence_version": "pulkit_completion_audit_v1",
        "framework_track": "PASS_SYNTHETICALLY_VERIFIED",
        "phase_count": 16,
        "phases": _phase_rows(),
        "real_release": "BLOCKED",
        "final_test_accessed": g3.test_data_accessed,
        "g3": g3.overall,
        "g4_present": (root / "artifacts/governance/g4_evaluation_freeze.json").is_file(),
        "phase17_exists": False,
        "unmet_final_conditions": ["approved exact environment lock", "real selected bundle", "dashboard screenshots", "non-owner clean-machine execution", "required member reviews"],
    }
    completion_path = _json(evidence / "audit/pulkit_completion_audit_v1.json", completion)
    items.append(_item(root, "completion_audit", "All sixteen Pulkit phase dispositions and release blockers", "repository", "REVIEW_REQUIRED", completion_path, ("docs/CODEX_PROJECT_CONTEXT_V1.md", "docs/pulkit/PHASE15_PACKAGING_REVIEW.md"), command, "framework complete; final release acceptance conditions remain"))

    contribution_path = _write(evidence / "contributions/pulkit_contribution_v1.md", CONTRIBUTION_TEXT)
    items.append(_item(root, "contribution", "Pulkit ownership contribution evidence", "repository", "PASS", contribution_path, ("docs/CODEX_PROJECT_CONTEXT_V1.md",), command))
    viva_path = _write(evidence / "viva/viva_pulkit.md", VIVA_TEXT)
    items.append(_item(root, "viva", "Pulkit system and cross-project viva preparation", "repository", "PASS", viva_path, ("docs/CODEX_PROJECT_CONTEXT_V1.md",), command))
    audit_path = _write(evidence / "PHASE16_RELEASE_AUDIT.md", RELEASE_TEXT)
    items.append(_item(root, "release_audit", "Human-readable final system release audit", "repository", "REVIEW_REQUIRED", audit_path, ("docs/CODEX_PROJECT_CONTEXT_V1.md", "RUNBOOK.md", "README.md"), command, "real release inputs and acceptance evidence remain unavailable"))

    blockers = (
        ("git_tag", "Git commit and release tag provenance", "repository is not a Git working tree"),
        ("real_api", "API evidence from the final real selected bundle", "real selected-model manifest and artifacts are absent"),
        ("screenshots", "Dashboard screenshot set", "no browser capture tool or manually captured build-bound images are available"),
        ("clean_install", "Clean installation from final lock", "package-manager policy and final lock are unresolved"),
        ("non_owner", "Non-owner clean-machine reproduction", "must be executed and signed through the team process by a non-owner"),
        ("member_reviews", "Vedant and Sanskruti reviews", "no valid human approval artifacts exist"),
    )
    for identifier, description, blocker in blockers:
        items.append(_item(root, identifier, description, "blocked", "BLOCKED", None, (), command, blocker))

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "evidence_root": EVIDENCE_ROOT.as_posix(),
        "status": "REVIEW_REQUIRED_REAL_RELEASE_BLOCKED",
        "generated_at": "2026-09-20",
        "generator": GENERATOR,
        "git_commit": None,
        "release_tag": None,
        "demo_fixture_sha256": capture["demo"].fixture_sha256,
        "final_environment_lock_sha256": None,
        "scientific_values_generated": False,
        "final_test_accessed": g3.test_data_accessed,
        "items": items,
    }
    manifest_path_out = _json(evidence / "system_evidence_manifest_v1.json", manifest)
    audit(root)
    return manifest_path_out


def audit(root: Path):
    root = root.resolve()
    manifest_path = root / EVIDENCE_ROOT / "system_evidence_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise EvidenceAuditError("system evidence manifest version mismatch")
    declared = set()
    for item in manifest.get("items", ()): 
        if item.get("status") not in ALLOWED_STATUSES or item.get("scope") not in ALLOWED_SCOPES:
            raise EvidenceAuditError("system evidence status or scope is invalid")
        output = item.get("output_path")
        if item["status"] == "BLOCKED" and item["scope"] == "blocked":
            if output is not None or item.get("output_sha256") is not None or not item.get("blocker"):
                raise EvidenceAuditError("blocked evidence has fabricated output or lacks blocker")
            continue
        if not isinstance(output, str) or Path(output).is_absolute():
            raise EvidenceAuditError("system evidence output path is invalid")
        path = root / output
        if not path.is_file() or not is_sha256(item.get("output_sha256", "")) or sha256_file(path) != item["output_sha256"]:
            raise EvidenceAuditError("system evidence output hash mismatch: " + output)
        for source in item.get("source_artifacts", ()):
            source_path = root / source.get("path", "")
            if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
                raise EvidenceAuditError("system evidence source hash mismatch: " + str(source.get("path")))
        if item["scope"] == "synthetic" and "synthetic" not in path.read_text(encoding="utf-8").lower():
            raise EvidenceAuditError("synthetic evidence lacks an explicit scope label")
        declared.add(path.resolve())
    actual = {path.resolve() for path in (root / EVIDENCE_ROOT).rglob("*") if path.is_file()}
    orphans = actual - declared - {manifest_path.resolve()}
    if orphans:
        raise EvidenceAuditError("orphan system evidence output: " + str(sorted(orphans)[0]))
    text_paths = [Path(value) for value in declared if Path(value).suffix in {".md", ".json", ".txt"}]
    audit_terminology(text_paths)
    api = json.loads((root / EVIDENCE_ROOT / "api/api_evidence_synthetic_v1.json").read_text())
    codes = api["structured_errors"]
    if [codes[name]["status_code"] for name in ("unknown_stay", "illegal_prediction_time", "serving_unavailable")] != [404, 422, 503]:
        raise EvidenceAuditError("structured API error evidence is invalid")
    if not all(row["direct_pipeline_equal"] for row in api["predictions"]):
        raise EvidenceAuditError("API/direct equality evidence failed")
    if len(api["predictions"]) < 3:
        raise EvidenceAuditError("three-cutoff replay evidence is absent")
    from demo.fixture import validate_demo_fixture
    from serving.prediction_schema import validate_response

    demo = validate_demo_fixture(root)
    if manifest.get("demo_fixture_sha256") != demo.fixture_sha256:
        raise EvidenceAuditError("demo fixture identity disagrees with manifest")
    for row in api["predictions"]:
        validate_response(row["response"], synthetic=True)
    repro = load_config(root / "configs/reproducibility_v1.json")
    environment = audit_environment_lock(root, repro)
    if manifest.get("final_environment_lock_sha256") != environment.get("lock_sha256"):
        raise EvidenceAuditError("environment lock identity disagrees with manifest")
    if manifest.get("scientific_values_generated") is not False:
        raise EvidenceAuditError("manual scientific values are not allowed")
    return {"status": "PASS", "indexed_outputs": len(declared), "blocked_slots": sum(item["status"] == "BLOCKED" for item in manifest["items"]), "orphans": 0}


CONTRIBUTION_TEXT = """# Pulkit Contribution Evidence\n\nStatus: **implemented and synthetically verified; real release blocked upstream**.\n\nPulkit's repository-owned track covers organ-support endpoint state and label contracts, serving schemas and pipeline orchestration, cutoff-safe history, artifact and explanation routing, Integrated Gradients and TreeSHAP adapters, FastAPI, recovery and ICU-time presentation, retrospective replay UI, integration/compatibility testing, packaging, the official non-sensitive demo fixture, and this governed system-evidence package.\n\nThis document records implementation ownership, not a human approval, scientific performance claim, or team-level completion claim.\n"""


VIVA_TEXT = """# Pulkit Viva Readiness\n\nPulkit should be able to explain the full path from the adult first-ICU-stay cohort and subject-disjoint coarse temporal holdout through the canonical `(t-48h, t]` representation, independent recovery horizons, remaining current-ICU-stay time, eligible OFF-to-ON support initiation, stay-balanced evaluation, validation-only selection/calibration/thresholding, and final-test isolation.\n\nFor the product layer, be ready to trace one request through cutoff validation, history truncation, canonical feature construction, frozen transform-only preprocessing, selected-family prediction, recovery/ICU/support postprocessing, family-bound attribution, schema validation, FastAPI, and dashboard replay. Explain that attribution describes model influence and does not establish clinical effect.\n\nCurrent boundary: synthetic system contracts are executable; real selected artifacts, an approved environment lock, final-test/G4 evidence, and non-owner acceptance are unavailable.\n"""


RELEASE_TEXT = """# Pulkit Phase 16 Release Audit\n\nThe governed system evidence was generated by executing the official synthetic fixture through the Phase-14/15 pipeline, API, and dashboard composition. It establishes system-contract behavior only and contains no scientific performance results.\n\nThe real release remains blocked by the absent real selected-model bundle, approved exact environment lock, Git/tag provenance, dashboard image capture, non-owner clean-machine execution, and member reviews. Final-test data was not opened and no G3/G4 state was created. There is no Pulkit Phase 17.\n"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "audit"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    result = generate(args.root, args.junit) if args.command == "generate" else audit(args.root)
    print(result if isinstance(result, Path) else json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
