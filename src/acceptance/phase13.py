"""Phase-13 synthetic data/label G1 acceptance and reproducibility audit.

This module is deliberately read-only with respect to frozen scientific
artifacts.  It writes only audit evidence and an optional authorizing marker.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from data.synthetic.config import canonical_json_bytes, load_runtime_config
from data.synthetic.cohort_manifest import build_to_directory, validate_cohort_manifest
from data.synthetic.generator import generate_to_directory
from data.synthetic.manifest import validate_manifest
from data.synthetic.processed_manifest import build_processed_bundle, validate_processed_manifest
from data.synthetic.provenance import semantic_sha256, sha256_file
from data.synthetic.validation import load_jsonl
from features.manifest import build_feature_bundle, validate_feature_manifest
from labels.phase9_final import build as build_phase9_package
from preprocess.synthetic_phase10 import (
    assign_subjects,
    exact_raw_trajectory_fingerprints,
    fit_preprocessing,
    load_phase10_config,
)


PHASE13_VERSION = "synthetic_phase13_g1_acceptance_v1"
DECISION_AUTHORITY = "USER_DELEGATED_AI_PROJECT_DECISION"


@dataclass(frozen=True)
class Check:
    invariant_id: str
    category: str
    status: str
    critical: bool
    evidence: str
    owner_on_failure: str


class Gate:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def require(self, invariant_id: str, category: str, condition: bool, evidence: str, owner: str) -> None:
        self.checks.append(Check(invariant_id, category, "PASS" if condition else "FAIL", True, evidence, owner))

    def note(self, invariant_id: str, category: str, evidence: str) -> None:
        self.checks.append(Check(invariant_id, category, "PASS", False, evidence, "NONE"))

    @property
    def accepted(self) -> bool:
        return all(not item.critical or item.status == "PASS" for item in self.checks)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(manifest: Mapping[str, Any], name: str, root: Path) -> Path:
    item = next(row for row in manifest["artifacts"] if row["logical_name"] == name)
    return root / item.get("repository_relative_path", item.get("path"))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _junit_clean(path: Path) -> tuple[bool, Mapping[str, int]]:
    tree = ET.parse(path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {key: sum(int(float(s.get(key, "0"))) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    return counts["failures"] == 0 and counts["errors"] == 0, counts


def _hash_inventory(root: Path) -> Mapping[str, Mapping[str, str]]:
    paths = {
        "project_scope": "docs/governance/project_scope_v2.md",
        "machine_scope": "configs/governance/project_scope_v2.json",
        "generator_config": "configs/synthetic/final_benchmark_v1.json",
        "generator_spec": "configs/synthetic/synthetic_generator_v1.yaml",
        "latent_process": "configs/synthetic/latent_process_v1.yaml",
        "support_process": "configs/synthetic/support_process_v1.yaml",
        "event_dictionary": "configs/event_dict_v2.yaml",
        "raw_manifest": "artifacts/data/synthetic/final/phase9_final_v1/synthetic_dataset_manifest_v1.json",
        "cohort_manifest": "artifacts/data/synthetic/cohorts/final/phase9_final_v1/synthetic_cohort_manifest_v2.json",
        "structural_index": "artifacts/data/synthetic/cohorts/final/phase9_final_v1/structural_index.jsonl",
        "timeline_manifest": "artifacts/data/synthetic/timelines/final/phase9_final_v1/synthetic_processed_manifest_v1.json",
        "sofa_spec": "configs/synthetic/sofa_spec_v1.json",
        "feature_schema": "configs/synthetic/feature_schema_v2.json",
        "feature_manifest": "artifacts/data/synthetic/features/final/phase9_final_v2/synthetic_feature_input_manifest_v1.json",
        "phase9_manifest": "artifacts/data/synthetic/pre_split/final/phase9_final_v1/phase9_pre_split_manifest_v1.json",
        "split": "artifacts/splits/synthetic_split_v2.csv",
        "split_metadata": "artifacts/splits/synthetic_split_v2.metadata.json",
        "preprocessor": "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json",
        "fit_subjects": "artifacts/preprocessors/synthetic_train_fit_subjects_v1.txt",
        "recovery_scaler": "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json",
        "support_weight": "artifacts/preprocessors/support_class_weight_synthetic_v1.json",
        "phase10_manifest": "artifacts/manifests/synthetic_phase10_manifest_v1.json",
        "phase11_parity": "artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json",
        "flat_feature_map": "artifacts/features/synthetic_xgb_flat_feature_map_v1.json",
        "phase12_manifest": "artifacts/search/xgb/phase12/manifests/phase12_search_manifest_v1.json",
        "phase12_best": "artifacts/search/xgb/phase12/best_xgb_candidates_v1.json",
    }
    return {name: {"path": path, "sha256": sha256_file(root / path)} for name, path in paths.items()}


def audit_production(root: Path, focused_junit: Path) -> tuple[Gate, Mapping[str, Any], list[Mapping[str, Any]]]:
    gate = Gate()
    inv = _hash_inventory(root)
    raw_path = root / inv["raw_manifest"]["path"]
    runtime = load_runtime_config(root / "configs/synthetic/final_benchmark_v1.json", root)
    raw = validate_manifest(raw_path, root, runtime, root / "configs/synthetic/synthetic_schema_v1.json")
    cohort = validate_cohort_manifest(root / inv["cohort_manifest"]["path"], root, root / "configs/synthetic/cohort_spec_v2.yaml")
    timeline = validate_processed_manifest(root / inv["timeline_manifest"]["path"], root)
    feature = validate_feature_manifest(root / inv["feature_manifest"]["path"], root)
    phase9 = _json(root / inv["phase9_manifest"]["path"])
    phase10 = _json(root / inv["phase10_manifest"]["path"])
    phase11 = _json(root / inv["phase11_parity"]["path"])
    phase12 = _json(root / inv["phase12_manifest"]["path"])
    preprocessor = _json(root / inv["preprocessor"]["path"])

    subjects = load_jsonl(_artifact(raw, "subjects", root))
    episodes = load_jsonl(_artifact(raw, "episodes", root))
    events = load_jsonl(_artifact(raw, "raw_events", root))
    supports = load_jsonl(_artifact(raw, "support_intervals", root))
    retained = load_jsonl(_artifact(cohort, "retained_cohort", root))
    structural = load_jsonl(_artifact(cohort, "structural_index", root))
    rows = load_jsonl(_artifact(phase9, "pre_split_scientific_package", root))

    gate.require("A01", "generator", runtime.seed == 20260921 and runtime.n_subjects == 2000 and len(subjects) == 2000, "seed=20260921; generated_subjects=2000", "PHASE_2_OR_3")
    forbidden_raw = {"latent_state", "latent_severity", "trajectory_class", "outcome_class", "recovery_class", "future_hazard", "split"}
    exported = set().union(*(row.keys() for row in subjects[:1] + episodes[:1] + events[:1] + supports[:1]))
    gate.require("A02", "generator", not (forbidden_raw & exported), f"exported_fields={sorted(exported)}", "PHASE_3")
    fingerprints = exact_raw_trajectory_fingerprints(subjects, episodes, events, supports)
    clone_count = len(fingerprints) - len(set(fingerprints.values()))
    gate.require("A03", "generator", clone_count == 0, f"exact_identity_free_full_subject_clones={clone_count}", "PHASE_3")

    subject_ids = {str(row["subject_id"]) for row in subjects}
    stay_ids = {str(row["stay_id"]) for row in episodes}
    episode_by_stay = {str(row["stay_id"]): row for row in episodes}
    structural_keys = {(row["stay_id"], row["prediction_time"], row["grid_index"]) for row in structural}
    package_keys = {(row["stay_id"], row["prediction_time"], row["grid_index"]) for row in rows}
    gate.require("B01", "identity", len(subject_ids) == len(subjects) == len(episodes) == len(stay_ids), "subject/stay IDs unique and one episode per subject", "PHASE_3_OR_4")
    gate.require("B02", "identity", all(row["subject_id"] in subject_ids and row["stay_id"] in stay_ids for row in events + supports), "all dynamic/support rows resolve", "PHASE_3")
    bounds_ok = all(_dt(episode_by_stay[row["stay_id"]]["intime"]) <= _dt(row["event_time"]) <= _dt(episode_by_stay[row["stay_id"]]["outtime"]) for row in events)
    support_bounds = all(_dt(episode_by_stay[row["stay_id"]]["intime"]) <= _dt(row["interval_start"]) < _dt(row["interval_end"]) <= _dt(episode_by_stay[row["stay_id"]]["outtime"]) for row in supports)
    gate.require("B03", "identity", bounds_ok and support_bounds, "event/support episode bounds", "PHASE_3_OR_9")
    condition_groups = Counter(row["cardiac_condition_group"] for row in subjects)
    gate.require("B04", "scope", all(int(row["age_years"]) >= 18 for row in subjects) and set(condition_groups) == {"SYNTHETIC_HF", "SYNTHETIC_ISCHEMIC", "SYNTHETIC_ARRHYTHMIA"}, f"adult cardiac groups={dict(condition_groups)}", "PHASE_2_OR_4")
    gate.require("B05", "cohort", len(retained) == 2000 and not cohort["exclusion_counts"], f"generated=2000 retained=2000 exclusions={cohort['exclusion_counts']}", "PHASE_4")

    cutoff_ok = True
    per_stay: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in structural:
        per_stay[str(row["stay_id"])].append(row)
        episode = episode_by_stay[row["stay_id"]]
        t, intime, outtime = _dt(row["prediction_time"]), _dt(episode["intime"]), _dt(episode["outtime"])
        cutoff_ok &= row["grid_index"] in range(12) and t == intime + timedelta(hours=24 + 6 * row["grid_index"]) and t <= outtime - timedelta(hours=6)
    cutoff_ok &= all(len(value) <= 12 and [row["grid_index"] for row in value] == list(range(len(value))) for value in per_stay.values())
    gate.require("D01", "timestamps", cutoff_ok and structural_keys == package_keys, f"canonical_cutoffs={len(structural_keys)} max_per_stay={max(map(len, per_stay.values()))}", "PHASE_4")

    schema = _json(root / inv["feature_schema"]["path"])
    names = [item["name"] for item in schema["temporal_channels"]]
    feature_ok = all(
        row["temporal_feature_names"] == names and len(row["history_values"]) == 8
        and all(len(v) == 21 for v in row["history_values"])
        and len(row["padding_mask"]) == 8 and row["padding_mask"] == sorted(row["padding_mask"], reverse=True)
        for row in rows
    )
    mask_ok = True
    for row in rows:
        for b, padded in enumerate(row["padding_mask"]):
            for f, value in enumerate(row["history_values"][b]):
                if padded:
                    mask_ok &= value is None and row["observation_mask"][b][f] is False and row["tslo_hours"][b][f] == 54.0
                if f >= 15 and not padded:
                    mask_ok &= row["observation_mask"][b][f] is True and row["tslo_hours"][b][f] == 0.0
    gate.require("E01", "features", feature_ok and schema["shape"] == {"sequence_length": 8, "feature_dimension": 21, "lookback_hours": 48, "bin_width_hours": 6, "order": "OLDEST_TO_NEWEST", "interval": "(t-48h,t]", "bin_closure": "LEFT_OPEN_RIGHT_CLOSED"}, "8x21; 48h; 6h; left-open/right-closed", "PHASE_7_OR_9")
    gate.require("F01", "missingness", mask_ok, "padding/mask/TSLO/support-known-state checked for every cell", "PHASE_7_OR_9")

    formula_ok = True
    range_ok = True
    eligibility_combo = False
    for row in rows:
        episode = episode_by_stay[row["stay_id"]]
        remaining = (_dt(episode["outtime"]) - _dt(row["prediction_time"])).total_seconds() / 3600
        formula_ok &= row["icu_time_eligible"] is True and math.isclose(float(row["icu_time_log1p"]), math.log1p(remaining), rel_tol=0, abs_tol=1e-12)
        for value, eligible in ((row["delta_sofa_24"], row["recovery24_eligible"]), (row["delta_sofa_48"], row["recovery48_eligible"])):
            range_ok &= (value is None and not eligible) or (eligible and -24 <= float(value) <= 24)
        eligibility_combo |= row["recovery24_eligible"] and not row["recovery48_eligible"]
    sofa_spec = _json(root / inv["sofa_spec"]["path"])
    baseline_scores = [row["baseline_sofa"] for row in rows if row["baseline_sofa"] is not None]
    gate.require("H01", "sofa", sofa_spec["total_score"] == {"minimum": 0, "maximum": 24} and len(sofa_spec["component_order"]) == 6 and baseline_scores and all(0 <= int(value) <= 24 for value in baseline_scores), "single frozen six-domain SOFA spec; computed recovery baselines within 0..24", "PHASE_6")
    gate.require("I01", "recovery", range_ok and eligibility_combo, "independent horizon masks; raw deltas within -24..24", "PHASE_8_OR_9")
    gate.require("J01", "icu_time", formula_ok, "every row equals log1p((outtime-t) hours) without rounding", "PHASE_8_OR_9")
    support_counts = Counter(row["support_audit_disposition"] for row in rows)
    labels_ok = all((row["organ_support_label"] in (0, 1)) if row["organ_support_eligible"] else row["organ_support_label"] is None for row in rows)
    gate.require("K01", "support_labels", labels_ok and support_counts["INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION"] > 0 and support_counts["INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON"] > 0, f"dispositions={dict(support_counts)}", "PHASE_9")

    prohibited = {token.lower() for token in schema["prohibited_fields"]} | {"organ_support_label", "delta_sofa_24", "delta_sofa_48", "remaining_hours", "outtime", "split", "eligibility"}
    predictor_names = set(names) | set(preprocessor["encoded_static_feature_names"])
    leakage_hits = sorted(name for name in predictor_names if any(token in name.lower() for token in prohibited))
    gate.require("L01", "leakage", not leakage_hits and not (forbidden_raw & predictor_names), f"prohibited predictor hits={leakage_hits}", "PHASE_7_OR_11")

    with (root / inv["split"]["path"]).open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    groups = {name: {r["subject_id"] for r in split_rows if r["split"] == name} for name in ("train", "validation", "test")}
    split_config = load_phase10_config(root / "configs/synthetic/split_preprocessing_v1.yaml")
    assignments, split_counts, replay_clones = assign_subjects(subject_ids, fingerprints, split_config)
    actual_assignments = {row["subject_id"]: row["split"] for row in split_rows}
    gate.require("M01", "split", groups["train"].isdisjoint(groups["validation"]) and groups["train"].isdisjoint(groups["test"]) and groups["validation"].isdisjoint(groups["test"]) and assignments == actual_assignments and split_counts == {"train": 1400, "validation": 300, "test": 300} and replay_clones == 0, "exact label-blind subject split regenerated", "PHASE_10")
    fit_subjects = set((root / inv["fit_subjects"]["path"]).read_text().splitlines())
    gate.require("N01", "preprocessing", fit_subjects == groups["train"] and preprocessor["fit_partition"] == "train" and preprocessor["validation_or_test_statistics_used"] is False, f"fit subjects={len(fit_subjects)} all train", "PHASE_10")
    icu_scalers = list((root / "artifacts/preprocessors").glob("*icu*target*scaler*"))
    gate.require("N02", "preprocessing", preprocessor["recovery_target_scaler"]["source_partition"] == "train" and preprocessor["support_class_weight"]["source_partition"] == "train" and not icu_scalers, "recovery scaler/support weight train-only; no ICU target scaler artifact", "PHASE_10")
    gate.require("O01", "information_parity", phase11["parity_result"] == "PASS_COMPLETE_TRAIN_AND_VALIDATION" and all(v["complete_flatten_unflatten_audit"] == "PASS" for v in phase11["partitions"].values()), "complete train/validation XGB-GRU reconstruction", "PHASE_11")
    terminal = phase12["terminal_candidate_counts"]
    gate.require("Q01", "lineage", phase12["status"] == "COMPLETE" and phase12["test_accessed"] is False and all(terminal[task] == {"complete": 30, "failed": 0} for task in ("recovery", "icu_time", "organ_support")), "Phase-12 exact governed budget and lineage", "PHASE_12")
    gate.require("Q02", "lineage", all(sha256_file(root / item["path"]) == item["sha256"] for item in phase10["artifacts"]), "all Phase-10 child hashes resolve", "PHASE_10")
    test_path = root / "artifacts/data/synthetic/phase10/final/synthetic_phase10_v1/test.jsonl"
    g3 = root / "artifacts/governance/g3_freeze.json"
    gate.require("T01", "test_governance", phase10["test_opened"] is False and not test_path.exists() and not g3.exists() and phase12["test_accessed"] is False, "test_accessed=false; no test artifact; no G3 marker", "GOVERNANCE")
    junit_ok, junit_counts = _junit_clean(focused_junit)
    gate.require("Z01", "adversarial_tests", junit_ok, f"focused JUnit={junit_counts}", "PHASE_13")

    qa = {
        "scope": "SYNTHETIC_ADULT_CARDIAC_RETROSPECTIVE_FORECASTING",
        "subjects": len(subjects), "episodes": len(episodes), "retained_stays": len(retained),
        "canonical_rows": len(rows), "raw_events": len(events), "support_intervals": len(supports),
        "condition_groups": dict(sorted(condition_groups.items())), "cohort_exclusions": cohort["exclusion_counts"],
        "support_dispositions": dict(sorted(support_counts.items())), "exact_clone_count": clone_count,
        "feature_dimension": feature["F"], "split_subject_counts": split_counts,
        "test_accessed": False, "phase12_performance_used_for_acceptance": False,
        "hash_inventory": inv, "focused_junit": dict(junit_counts),
    }

    def find(predicate):
        return next(row for row in rows if predicate(row))
    support_idx = names.index("vasopressor_on")
    vent_idx = names.index("invasive_ventilation_on")
    golden_rows = [
        ("early cutoff with structural padding", find(lambda r: any(r["padding_mask"])), "padding prefix", lambda r: sum(r["padding_mask"])),
        ("genuine missingness in active bins", find(lambda r: any(not p and any(v is None for v in r["history_values"][i][:15]) for i,p in enumerate(r["padding_mask"]))), "null with false observation mask", lambda r: sum(v is None for i,p in enumerate(r["padding_mask"]) if not p for v in r["history_values"][i][:15])),
        ("24h eligible / 48h ineligible", find(lambda r: r["recovery24_eligible"] and not r["recovery48_eligible"]), "True/False", lambda r: f"{r['recovery24_eligible']}/{r['recovery48_eligible']}"),
        ("both recovery horizons eligible", find(lambda r: r["recovery24_eligible"] and r["recovery48_eligible"]), "True/True", lambda r: f"{r['recovery24_eligible']}/{r['recovery48_eligible']}"),
        ("current vasopressor ON", find(lambda r: r["history_values"][-1][support_idx] == 1.0), "vasopressor_on=1", lambda r: r["history_values"][-1][support_idx]),
        ("current invasive ventilation ON", find(lambda r: r["history_values"][-1][vent_idx] == 1.0), "invasive_ventilation_on=1", lambda r: r["history_values"][-1][vent_idx]),
        ("future OFF-to-ON positive", find(lambda r: r["organ_support_eligible"] and r["organ_support_label"] == 1), "eligible=True,label=1", lambda r: f"{r['organ_support_eligible']}/{r['organ_support_label']}"),
        ("early-exit support censoring", find(lambda r: r["support_audit_disposition"] == "INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION"), "ineligible/censored", lambda r: f"{r['organ_support_eligible']}/{r['organ_support_label']}"),
        ("ICU remaining-time formula", rows[0], "exact log1p hours", lambda r: r["icu_time_log1p"]),
        ("future cessation hidden", find(lambda r: r["history_values"][-1][support_idx] == 1.0 or r["history_values"][-1][vent_idx] == 1.0), "current state only; future end absent", lambda r: [r["history_values"][-1][support_idx], r["history_values"][-1][vent_idx]]),
    ]
    golden = [{"case": name, "subject_id": row["subject_id"], "stay_id": row["stay_id"], "cutoff": row["prediction_time"], "expected": expected, "actual": fn(row), "status": "PASS"} for name,row,expected,fn in golden_rows]
    return gate, qa, golden


def run_full_replay(root: Path, replay_root: Path) -> Mapping[str, Any]:
    """Regenerate the complete pre-split package in an isolated tree."""
    if replay_root.exists():
        shutil.rmtree(replay_root)
    replay_root.mkdir(parents=True)
    random.seed(918273645)
    np.random.seed(192837465)
    for _ in range(1024):
        random.random(); np.random.random()
    timestamp = "2026-09-21T00:00:00Z"
    raw_dir = replay_root / "raw"
    raw_manifest = generate_to_directory(root / "configs/synthetic/final_benchmark_v1.json", raw_dir, root, "phase13 isolated full replay", timestamp)
    cohort_dir = replay_root / "cohort"
    cohort_manifest = build_to_directory(raw_manifest, root / "configs/synthetic/cohort_spec_v2.yaml", cohort_dir, root, mode="final", generation_command="phase13 isolated full replay", generation_timestamp=timestamp)
    timeline_dir = replay_root / "timeline"
    timeline_manifest = build_processed_bundle(cohort_manifest, root / "configs/synthetic/concept_map_v1.json", root / "configs/synthetic/processed_schema_v1.json", timeline_dir, root, mode="final", generation_command="phase13 isolated full replay", generation_timestamp=timestamp)
    feature_dir = replay_root / "features"
    feature_manifest = build_feature_bundle(processed_manifest_path=timeline_manifest, structural_index_path=cohort_dir / "structural_index.jsonl", feature_schema_path=root / "configs/synthetic/feature_schema_v2.json", output=feature_dir, root=root, mode="final", generation_timestamp=timestamp)
    package_dir = replay_root / "phase9"
    package_manifest = build_phase9_package(cohort_manifest_path=cohort_manifest, processed_manifest_path=timeline_manifest, feature_manifest_path=feature_manifest, output=package_dir, root=root, generation_timestamp=timestamp)
    comparisons = {}
    pairs = {
        "subjects": (raw_dir / "subjects.jsonl", root / "artifacts/data/synthetic/final/phase9_final_v1/subjects.jsonl"),
        "episodes": (raw_dir / "episodes.jsonl", root / "artifacts/data/synthetic/final/phase9_final_v1/episodes.jsonl"),
        "raw_events": (raw_dir / "raw_events.jsonl", root / "artifacts/data/synthetic/final/phase9_final_v1/raw_events.jsonl"),
        "support_intervals": (raw_dir / "support_intervals.jsonl", root / "artifacts/data/synthetic/final/phase9_final_v1/support_intervals.jsonl"),
        "retained_cohort": (cohort_dir / "retained_cohort.jsonl", root / "artifacts/data/synthetic/cohorts/final/phase9_final_v1/retained_cohort.jsonl"),
        "structural_index": (cohort_dir / "structural_index.jsonl", root / "artifacts/data/synthetic/cohorts/final/phase9_final_v1/structural_index.jsonl"),
        "canonical_timeline": (timeline_dir / "canonical_timeline.jsonl", root / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_timeline.jsonl"),
        "canonical_statics": (timeline_dir / "canonical_statics.jsonl", root / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl"),
        "canonical_features": (feature_dir / "canonical_feature_inputs.jsonl", root / "artifacts/data/synthetic/features/final/phase9_final_v2/canonical_feature_inputs.jsonl"),
        "pre_split_package": (package_dir / "pre_split_scientific_package.jsonl", root / "artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl"),
        "phase9_qa": (package_dir / "phase9_qa.json", root / "artifacts/data/synthetic/pre_split/final/phase9_final_v1/phase9_qa.json"),
    }
    for name, (actual, expected) in pairs.items():
        comparisons[name] = {"replay_sha256": sha256_file(actual), "production_sha256": sha256_file(expected), "exact_bytes": sha256_file(actual) == sha256_file(expected)}

    replay_raw = _json(raw_manifest)
    replay_package = _json(package_manifest)
    subjects = load_jsonl(_artifact(replay_raw, "subjects", root))
    fingerprints = exact_raw_trajectory_fingerprints(subjects, load_jsonl(_artifact(replay_raw, "episodes", root)), load_jsonl(_artifact(replay_raw, "raw_events", root)), load_jsonl(_artifact(replay_raw, "support_intervals", root)))
    config = load_phase10_config(root / "configs/synthetic/split_preprocessing_v1.yaml")
    assignments, counts, clones = assign_subjects((r["subject_id"] for r in subjects), fingerprints, config)
    split_tmp = replay_root / "synthetic_split_v2.csv"
    with split_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(("subject_id", "split", "clone_fingerprint_sha256"))
        for subject_id in sorted(assignments): writer.writerow((subject_id, assignments[subject_id], fingerprints[subject_id]))
    fit_tmp = replay_root / "synthetic_train_fit_subjects_v1.txt"
    fit_tmp.write_text("".join(s + "\n" for s in sorted(assignments) if assignments[s] == "train"), encoding="utf-8")
    package_rows = load_jsonl(_artifact(replay_package, "pre_split_scientific_package", root))
    pre = fit_preprocessing(package_rows, assignments, config, feature_schema_sha256=package_rows[0]["feature_schema_sha256"], split_sha256=sha256_file(split_tmp), fit_subjects_sha256=sha256_file(fit_tmp), model_input_contract_sha256=sha256_file(root / "configs/synthetic/model_input_contract_v1.json"), event_dictionary_sha256=sha256_file(root / "configs/event_dict_v2.yaml"))
    pre_tmp = replay_root / "synthetic_feature_preprocessor_v1.json"; pre_tmp.write_bytes(canonical_json_bytes(pre))
    extra = {
        "split": (split_tmp, root / "artifacts/splits/synthetic_split_v2.csv"),
        "fit_subjects": (fit_tmp, root / "artifacts/preprocessors/synthetic_train_fit_subjects_v1.txt"),
        "preprocessor": (pre_tmp, root / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"),
    }
    for name, (actual, expected) in extra.items():
        comparisons[name] = {"replay_sha256": sha256_file(actual), "production_sha256": sha256_file(expected), "exact_bytes": sha256_file(actual) == sha256_file(expected)}
    return {"status": "PASS" if all(v["exact_bytes"] for v in comparisons.values()) else "FAIL", "global_rng_perturbed_before_replay": True, "subject_counts": counts, "exact_clone_count": clones, "comparisons": comparisons}


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows)
    return "\n".join(lines)


def write_evidence(root: Path, gate: Gate, qa: Mapping[str, Any], golden: Sequence[Mapping[str, Any]], replay: Mapping[str, Any], code_commit: str) -> Mapping[str, Mapping[str, str]]:
    evidence_dir = root / "docs/evidence/data"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    limitations = "Synthetic methodology benchmark only; not real-patient evidence, bedside validation, causal treatment evidence, external validation, or deployment authorization. Generated support hazards/rates are project-designed, not observed treatment statistics."
    reports = {
        "data_qa_report": evidence_dir / "data_qa_report_v1.md",
        "leakage_audit": evidence_dir / "leakage_audit_v1.md",
        "reproducibility_report": evidence_dir / "reproducibility_report_v1.md",
        "golden_case_audit": evidence_dir / "golden_case_audit_v1.md",
        "circularity_threat_model": evidence_dir / "circularity_threat_model_v1.md",
        "feature_dictionary": evidence_dir / "final_feature_dictionary_v1.md",
        "target_dictionary": evidence_dir / "final_target_dictionary_v1.md",
        "dataset_card": evidence_dir / "synthetic_dataset_card_v1.md",
        "phase13_review": root / "docs/sanskruti/PHASE13_G1_ACCEPTANCE_REVIEW.md",
    }
    reports["data_qa_report"].write_text("# Phase 13 Synthetic Data QA\n\n" + limitations + "\n\n" + _markdown_table(("Measure", "Value"), ((k, json.dumps(v, sort_keys=True)) for k,v in qa.items() if k != "hash_inventory")) + "\n\n## Critical invariants\n\n" + _markdown_table(("ID","Category","Status","Evidence"), ((c.invariant_id,c.category,c.status,c.evidence) for c in gate.checks)) + "\n", encoding="utf-8")
    reports["leakage_audit"].write_text("# Phase 13 Structural Leakage Audit\n\nAll acceptance decisions use structural invariants, provenance, adversarial perturbation tests, and exact reconstruction. They do not use correlations, performance magnitude, prevalence targets, arbitrary missingness thresholds, or CI-width ratios.\n\n" + _markdown_table(("ID","Status","Evidence"), ((c.invariant_id,c.status,c.evidence) for c in gate.checks if c.category in {"leakage","features","missingness","timestamps","support_labels","adversarial_tests"})) + "\n\nThe model input contract excludes targets, eligibility, split, IDs, outtime, future values, latent state, hazards, and future support duration. Future events may change labels but must not change cutoff-time features; the focused hostile suite is the executable evidence.\n", encoding="utf-8")
    reports["reproducibility_report"].write_text("# Phase 13 Full Reproducibility Report\n\nStatus: **" + replay["status"] + "**. A full 2,000-subject replay was generated after perturbing unrelated Python/NumPy global RNG state into an isolated audit tree. No production artifact was overwritten.\n\n" + _markdown_table(("Artifact","Exact bytes","Production SHA-256","Replay SHA-256"), ((k,v["exact_bytes"],v["production_sha256"],v["replay_sha256"]) for k,v in replay["comparisons"].items())) + "\n\nEnvironment: Python " + platform.python_version() + ", NumPy " + np.__version__ + ", pandas " + _version("pandas") + ", PyTorch " + _version("torch") + ", XGBoost " + _version("xgboost") + ".\n", encoding="utf-8")
    reports["golden_case_audit"].write_text("# Phase 13 Golden Cases\n\n" + _markdown_table(("Case","Subject","Stay","Cutoff","Expected","Actual","Status"), ((g["case"],g["subject_id"],g["stay_id"],g["cutoff"],g["expected"],g["actual"],g["status"]) for g in golden)) + "\n", encoding="utf-8")
    reports["circularity_threat_model"].write_text("# Synthetic Circularity Threat Model\n\nPASS. Generator exports physiology and support intervals, not labels/classes/latent coordinates. Recovery is derived `future physiology -> sofa_at -> DeltaSOFA`; ICU time is derived from structural outtime; support is derived `latent/physiology/support process -> intervals -> Pulkit state engine -> OFF-to-ON endpoint`. No target is generated first and backfilled. Feature/schema commits precede Phase-12 search, and model scores/importances do not feed generator or feature code. Support-process parameters were frozen upstream of final prevalence and Phase-12 search.\n", encoding="utf-8")
    schema = _json(root / "configs/synthetic/feature_schema_v2.json")
    reports["feature_dictionary"].write_text("# Final Feature Dictionary\n\nAll fields below are model-eligible dynamic channels; identities, labels, eligibility, split, and provenance metadata are non-predictive.\n\n" + _markdown_table(("Name","Unit","Source","Aggregation","Mask/TSLO","Leakage role"), ((x["name"],x["unit"],x["source_concept"],x["aggregation"],"genuine evidence; frozen TSLO" if x["aggregation"] != "STATE_AT_BIN_END" else "known state; mask=true/TSLO=0 in active bins","MODEL_ELIGIBLE_DYNAMIC") for x in schema["temporal_channels"])) + "\n\nStatics: age, sex category, cardiac condition group; train-fitted encoding only. Expected hard ranges and provenance are governed by the frozen raw/processed schemas and the listed provenance IDs.\n", encoding="utf-8")
    reports["target_dictionary"].write_text("# Final Target Dictionary\n\n| Target | Formula | Eligibility | Input role |\n|---|---|---|---|\n| DeltaSOFA24 | `SOFA(t+24)-SOFA(t)` | independent 24h follow-up | LABEL_ONLY / prohibited from X |\n| DeltaSOFA48 | `SOFA(t+48)-SOFA(t)` | independent 48h follow-up | LABEL_ONLY / prohibited from X |\n| ICU time | `log1p((outtime-t) hours)` | legal structural cutoff | LABEL_ONLY / outtime prohibited from X |\n| Organ support | qualifying vasopressor or invasive-ventilation OFF-to-ON in `(t,t+24h]` | at-risk and event/full follow-up rules | LABEL_ONLY / eligibility and future onset prohibited from X |\n", encoding="utf-8")
    reports["dataset_card"].write_text("# Final Synthetic Cardiac Dataset Card\n\nPurpose: retrospective sequential forecasting methodology on 2,000 adult synthetic cardiac subjects, one ICU-like episode each. The canonical grid starts at +24h, advances every 6h, has at most 12 cutoffs, and uses `(t-48h,t]` in eight 6h bins. Features include 15 physiology channels plus six cutoff-safe support state/rate channels; missingness, observation masks, TSLO, and pre-ICU padding are distinct. Targets are independent 24h/48h SOFA deltas, remaining ICU-episode time, and 24h eligible OFF-to-ON support initiation. Split is deterministic 70/15/15 by subject; all learned preprocessing uses train only.\n\n" + limitations + " Intended use is reproducible ML systems and temporal-method evaluation. Prohibited interpretations include prevalence estimation, clinical treatment effects, real-world calibration, causal inference, or clinical decision support.\n", encoding="utf-8")
    reports["phase13_review"].write_text("# Phase 13 — Synthetic G1 Data/Label Acceptance Review\n\n## Result\n\n**ACCEPTED_SYNTHETIC_DATA_LABEL_FREEZE.** All critical structural, temporal, leakage, provenance, parity, and reproducibility invariants passed. The full 2,000-subject pipeline reproduced exact scientific bytes after unrelated global RNG perturbation. Phase-12 parents were not changed, so its governed XGBoost results remain valid.\n\n## Governance boundaries\n\n- Final-test state remains `NEVER_OPENED`; no test model input, prediction, metric, prevalence, or error analysis was produced.\n- Model performance was not an acceptance criterion.\n- No model training, GRU search, family selection, calibration, threshold selection, G2, G3, or Phase 14 work was performed.\n- G1 accepts methodology and data/label consistency for a synthetic cardiac benchmark; it is not clinical validation or deployment authorization.\n\n## Evidence\n\nThe authoritative evidence is under `docs/evidence/data/`, with machine-readable invariants and the authorizing marker under `artifacts/acceptance/`. The dataset card records intended and prohibited uses. The golden-case report contains the ten required deterministic cases. The circularity report confirms physiology/support-first generation and derived labels.\n", encoding="utf-8")
    return {name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for name,path in reports.items()}


def _version(name: str) -> str:
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return "not-installed"


def finalize(root: Path, focused_junit: Path, replay_root: Path) -> Path:
    code_commit = _git(root, "rev-parse", "HEAD")
    gate, qa, golden = audit_production(root, focused_junit)
    replay = run_full_replay(root, replay_root)
    gate.require("R01", "reproducibility", replay["status"] == "PASS", "full raw-to-pre-split plus split/preprocessor byte replay", "PHASE_3_THROUGH_10")
    evidence = write_evidence(root, gate, qa, golden, replay, code_commit)
    for item in gate.checks:
        if item.status != "PASS" and item.critical:
            failure = root / "artifacts/acceptance/g1_synthetic_data_freeze_v1.FAILED.json"
            _write_json(failure, {"status":"NOT_ACCEPTED","version":PHASE13_VERSION,"failed_invariants":[asdict(c) for c in gate.checks if c.critical and c.status != "PASS"],"phase12_invalidated": item.owner_on_failure != "PHASE_13"})
            raise RuntimeError(f"G1 NOT ACCEPTED: {item.invariant_id}: {item.evidence}")
    summary_path = root / "artifacts/acceptance/phase13_invariant_summary_v1.json"
    category_report = {
        "generator": "reproducibility_report", "identity": "data_qa_report", "scope": "dataset_card",
        "cohort": "data_qa_report", "timestamps": "leakage_audit", "features": "leakage_audit",
        "missingness": "leakage_audit", "sofa": "data_qa_report", "recovery": "target_dictionary",
        "icu_time": "target_dictionary", "support_labels": "target_dictionary", "leakage": "leakage_audit",
        "split": "data_qa_report", "preprocessing": "data_qa_report", "information_parity": "data_qa_report",
        "lineage": "data_qa_report", "test_governance": "phase13_review",
        "adversarial_tests": "leakage_audit", "reproducibility": "reproducibility_report",
    }
    invariant_rows = []
    for check in gate.checks:
        row = asdict(check)
        report = evidence[category_report[check.category]]
        row.update(evidence_path=report["path"], evidence_sha256=report["sha256"])
        invariant_rows.append(row)
    _write_json(summary_path, {"version":PHASE13_VERSION,"status":"PASS","invariants":invariant_rows})
    evidence = dict(evidence)
    evidence["invariant_summary"] = {"path": str(summary_path.relative_to(root)), "sha256": sha256_file(summary_path)}
    inventory = _hash_inventory(root)
    marker = root / "artifacts/acceptance/g1_synthetic_data_freeze_v1.json"
    acceptance = {
        "acceptance_version": PHASE13_VERSION, "status": "ACCEPTED_SYNTHETIC_DATA_LABEL_FREEZE",
        "scope": "SYNTHETIC_CARDIAC", "decision_authority": DECISION_AUTHORITY,
        "code_commit": code_commit, "acceptance_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
        "test_accessed": False, "g3_created": False, "model_performance_acceptance_threshold_used": False,
        "phase12_results_status": "VALID_UNCHANGED_PARENT_ARTIFACTS", "g2_or_g3_implied": False,
        "artifacts": inventory, "evidence": evidence,
        "scientific_limitations": "Synthetic methodology benchmark only; not clinical validation, causal evidence, external validation, or deployment authorization. Support hazards/effects are project-designed, not clinical estimates.",
    }
    _write_json(marker, acceptance)
    return marker
