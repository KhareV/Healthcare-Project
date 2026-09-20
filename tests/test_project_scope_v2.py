import csv
import hashlib
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCOPE_JSON = REPOSITORY_ROOT / "configs" / "governance" / "project_scope_v2.json"
SCOPE_DOC = REPOSITORY_ROOT / "docs" / "governance" / "project_scope_v2.md"
DELTA_CSV = REPOSITORY_ROOT / "docs" / "governance" / "project_scope_v2_delta.csv"
IMPACT_DOC = REPOSITORY_ROOT / "docs" / "governance" / "project_scope_v2_impact.md"
CONTEXT_V2 = REPOSITORY_ROOT / "docs" / "CODEX_PROJECT_CONTEXT_V2.md"
HASH_MANIFEST = REPOSITORY_ROOT / "docs" / "governance" / "project_scope_v2_hashes.json"


def _scope():
    with SCOPE_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ProjectScopeV2IdentityTests(unittest.TestCase):
    def test_required_governance_artifacts_exist(self):
        for path in (SCOPE_JSON, SCOPE_DOC, DELTA_CSV, IMPACT_DOC, CONTEXT_V2, HASH_MANIFEST):
            self.assertTrue(path.is_file(), str(path))

    def test_scope_package_hash_manifest_matches_files(self):
        with HASH_MANIFEST.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["algorithm"], "SHA-256")
        self.assertEqual(manifest["scope_version"], "PROJECT_SCOPE_V2")
        expected = {
            "configs/governance/project_scope_v2.json",
            "docs/governance/project_scope_v2.md",
            "docs/governance/project_scope_v2_delta.csv",
            "docs/governance/project_scope_v2_impact.md",
            "docs/CODEX_PROJECT_CONTEXT_V2.md",
        }
        self.assertEqual(set(manifest["files"]), expected)
        for relative_path, expected_hash in manifest["files"].items():
            content = (REPOSITORY_ROOT / relative_path).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected_hash)

    def test_scope_is_draft_and_does_not_fabricate_approval(self):
        scope = _scope()
        self.assertEqual(scope["scope_version"], "PROJECT_SCOPE_V2")
        self.assertEqual(scope["status"], "DRAFT_COMPLETE_TEAM_FREEZE_REQUIRED")
        self.assertEqual(scope["approval_status"], "REVIEW_REQUIRED")
        self.assertEqual(scope["review"]["approved_by"], [])
        self.assertTrue(scope["review"]["team_freeze_required"])

    def test_final_data_and_fixture_scopes_are_distinct(self):
        data = _scope()["data"]
        self.assertEqual(data["final_scope"], "AUTHORIZED_FINAL_SYNTHETIC_DATA")
        self.assertEqual(data["engineering_fixture_scope"], "ENGINEERING_SYNTHETIC_FIXTURE")
        self.assertFalse(data["mimic_patient_data_used"])
        self.assertFalse(data["mimic_experimental_results_claimed"])
        self.assertTrue(data["fixture_promotion_without_acceptance_prohibited"])

    def test_population_is_adult_cardiac_and_not_silently_hf_only(self):
        population = _scope()["population"]
        self.assertEqual(population["domain"], "CARDIAC_HEART_DISEASE")
        self.assertTrue(population["adult_only"])
        self.assertFalse(population["heart_failure_only"])
        self.assertEqual(
            population["exact_subtype_taxonomy_status"],
            "DEFERRED_TO_PHASE_2_GENERATOR_SPECIFICATION",
        )


class ProjectScopeV2ScientificContractTests(unittest.TestCase):
    def test_retains_prediction_grid(self):
        time = _scope()["time"]
        self.assertEqual(time["candidate_formula"], "t_k = intime + 24h + 6h*k")
        self.assertEqual((time["k_min"], time["k_max"]), (0, 11))
        self.assertEqual(time["retention_rule"], "t_k <= outtime - 6h")
        self.assertEqual(time["max_prediction_rows_per_stay"], 12)

    def test_retains_exact_history_and_missingness_semantics(self):
        history = _scope()["history"]
        self.assertEqual(history["interval"], "(t-48h, t]")
        self.assertEqual(history["number_of_bins"], 8)
        self.assertEqual(history["bin_width_hours"], 6)
        self.assertEqual(history["bin_closure"], "LEFT_OPEN_RIGHT_CLOSED")
        self.assertEqual(history["pre_episode_policy"], "STRUCTURAL_PADDING")
        self.assertTrue(history["missingness_distinct_from_padding"])
        self.assertEqual(history["learned_preprocessing_fit_scope"], "TRAINING_SUBJECTS_ONLY")

    def test_retains_three_task_definitions(self):
        tasks = _scope()["tasks"]
        self.assertEqual(set(tasks), {"recovery", "remaining_episode_time", "organ_support"})
        self.assertEqual(tasks["recovery"]["delta_24h"], "SOFA(t+24h) - SOFA(t)")
        self.assertEqual(tasks["recovery"]["delta_48h"], "SOFA(t+48h) - SOFA(t)")
        self.assertFalse(tasks["recovery"]["chaining_allowed"])
        self.assertEqual(
            tasks["remaining_episode_time"]["model_target"],
            "log1p((outtime - t) hours)",
        )
        self.assertEqual(tasks["organ_support"]["window"], "(t, t+24h]")
        self.assertEqual(tasks["organ_support"]["partial_negative_followup"], "CENSORED")

    def test_retains_information_parity_and_governance(self):
        scope = _scope()
        self.assertEqual(scope["models"]["core_candidates"], ["naive", "xgboost", "gru"])
        self.assertTrue(scope["models"]["xgboost_gru_information_parity_required"])
        self.assertEqual(scope["models"]["configurations_per_task"], {"xgboost": 30, "gru": 30})
        self.assertEqual(scope["evaluation"]["selection_partition"], "validation")
        self.assertEqual(
            scope["evaluation"]["final_test_status"],
            "SEALED_UNTIL_G3_EQUIVALENT_FREEZE",
        )

    def test_mimic_split_is_superseded_without_inventing_replacement(self):
        split = _scope()["split"]
        self.assertEqual(split["mimic_anchor_era_policy"], "SUPERSEDED")
        self.assertEqual(
            split["policy_status"],
            "UNLOCKED_SCIENTIFIC_PARAMETER_SYNTHETIC_SUBJECT_SPLIT_POLICY",
        )
        self.assertIsNone(split["percentages"])
        self.assertIsNone(split["seed"])
        self.assertTrue(split["subject_level_separation"])
        self.assertFalse(split["duplicate_or_clone_cross_split_allowed"])

    def test_retains_replay_and_rejects_clinical_claims(self):
        scope = _scope()
        self.assertEqual(
            scope["serving"]["product_mode"],
            "RETROSPECTIVE_SEQUENTIAL_REPLAY_NOT_REAL_TIME_CLINICAL_PREDICTION",
        )
        prohibited = set(scope["claims"]["prohibited"])
        self.assertIn("MIMIC_VALIDATION", prohibited)
        self.assertIn("EXTERNAL_CLINICAL_VALIDATION", prohibited)
        self.assertIn("REAL_TIME_CLINICAL_PREDICTION", prohibited)
        self.assertEqual(scope["claims"]["clinical_validation_status"], "NOT_CLINICALLY_VALIDATED")

    def test_phase_boundary_prohibits_later_work(self):
        boundary = _scope()["phase_boundary"]
        self.assertEqual(boundary["phase"], "SANSKRUTI_REBUILD_PHASE_1_OF_14")
        self.assertFalse(boundary["dataset_generation_authorized"])
        self.assertFalse(boundary["model_training_authorized"])
        self.assertFalse(boundary["final_test_access_authorized"])
        self.assertFalse(boundary["downstream_runtime_contract_modification_authorized"])


class ProjectScopeV2TraceabilityTests(unittest.TestCase):
    def test_scope_delta_is_complete_and_uses_allowed_classifications(self):
        with DELTA_CSV.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 46)
        required_topics = {
            "dataset_source", "population", "adult_policy", "episode_boundary",
            "retained_episode", "subject_id", "stay_id", "intime_outtime",
            "event_time", "prediction_grid", "history_window", "bins", "bin_closure",
            "dynamic_feature_governance", "static_feature_governance", "observation_masks",
            "tslo", "padding", "sofa_severity", "recovery_target", "recovery_horizons",
            "remaining_stay", "support_endpoint", "censoring", "task_eligibility", "split",
            "naive", "xgboost", "gru", "lstm", "search_budget", "metrics", "bootstrap",
            "selection", "calibration", "threshold", "explanations", "prediction_pipeline",
            "api", "dashboard", "replay", "final_test", "artifact_lineage",
            "reproducibility", "evidence", "clinical_limitations",
        }
        by_topic = {row["topic"]: row for row in rows}
        self.assertEqual(required_topics - set(by_topic), set())
        allowed = {
            "RETAINED", "SUPERSEDED", "SYNTHETIC_EQUIVALENT",
            "DEFERRED_PARAMETER", "NOT_APPLICABLE",
        }
        self.assertTrue(all(row["classification"] in allowed for row in rows))
        self.assertTrue(all(row["review_status"] == "REVIEW_REQUIRED" for row in rows))

    def test_impact_matrix_covers_both_owners_and_key_components(self):
        text = IMPACT_DOC.read_text(encoding="utf-8")
        for token in (
            "## Vedant-owned systems", "Prediction timestamps", "Subject split",
            "Tensor schema", "Recovery GRU", "Remaining-time GRU", "Support GRU",
            "LSTM sensitivity", "Metrics", "Bootstrap", "Search", "Selection",
            "Calibration/threshold", "Registry/lineage", "G3/final-test guard",
            "Data acceptance", "Reproducibility", "Evidence",
            "## Pulkit-owned systems", "Vasopressor state", "Ventilation state",
            "Composite support label", "Prediction schema", "PredictionPipeline",
            "Artifact resolver", "History", "Preprocessing seam", "Explanation router",
            "Integrated Gradients", "TreeSHAP", "FastAPI", "Current SOFA/recovery",
            "Dashboard", "Replay", "Integration", "Packaging", "Release evidence",
        ):
            self.assertIn(token, text)

    def test_human_and_machine_scope_agree(self):
        text = SCOPE_DOC.read_text(encoding="utf-8")
        scope = _scope()
        self.assertEqual(scope["status"], "DRAFT_COMPLETE_TEAM_FREEZE_REQUIRED")
        self.assertIn("DRAFT COMPLETE — TEAM FREEZE REQUIRED", text)
        self.assertIn("AUTHORIZED_FINAL_SYNTHETIC_DATA", text)
        self.assertIn("adult cardiac/heart-disease", text)
        self.assertIn(scope["time"]["candidate_formula"], text)
        self.assertIn(scope["history"]["interval"], text)
        self.assertIn(scope["tasks"]["recovery"]["delta_24h"], text)
        self.assertIn(scope["tasks"]["remaining_episode_time"]["model_target"], text)
        self.assertIn(scope["tasks"]["organ_support"]["window"], text)
        self.assertIn("RETROSPECTIVE SEQUENTIAL REPLAY", text)


if __name__ == "__main__":
    unittest.main()
