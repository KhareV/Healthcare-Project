"""Governed V3 pre-test freeze, sealed-cohort seal, one-time paired evaluation and evidence.

Library behind scripts/performance_v3_final_evaluation.py. Design rules:
  * every frozen input is bound by SHA-256 (freeze record + evaluation plan) and re-verified
    before authorization;
  * the sealed cohort is only reachable through performance_v3.test_access.SealedCohortAccess
    (state is CONSUMED before any row is returned; a second run is blocked);
  * V3, V2_AS_SERVED (primary comparator), V2_AS_EVALUATED and CONTROL are scored on the SAME
    rows; metrics, paired bootstrap and plots are computed only from the saved prediction file;
  * `seal` is structural only: it never opens a label, eligibility or QA-count field.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluation.metrics import PredictionRecord
from performance_v3 import calibration as cal
from performance_v3 import task_metrics as tm
from performance_v3.paired_bootstrap import ci_excludes_zero, paired_stay_cluster_bootstrap
from performance_v3.serving import params_sha256
from performance_v3.test_access import (
    BLOCKED_ALREADY_CONSUMED, FINAL_RUN_COMPLETED, SEALED_NOT_ACCESSED, SealedCohortAccess, SealedCohortAccessError,
)

FINAL_EVAL_VERSION = "performance_v3_final_eval_v1"
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
MODELS = ("V3", "V2_AS_SERVED", "V2_AS_EVALUATED", "CONTROL")
PRIMARY_COMPARATOR = "V2_AS_SERVED"
COMPARATORS = ("V2_AS_SERVED", "V2_AS_EVALUATED", "CONTROL")
PRIMARY_METRIC = {"recovery24": "mae", "recovery48": "mae", "icu_stay_time": "weighted_median_ae_hours", "organ_support": "auprc"}
SECONDARY = {
    "recovery24": ("rmse", "median_ae"), "recovery48": ("rmse", "median_ae"),
    "icu_stay_time": ("mae", "rmse"), "organ_support": ("auroc", "brier", "log_loss", "f1"),
}
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261012
CI_LEVELS = (0.95, 0.9875)
PROMOTION_CI_LEVEL = 0.9875
BRIER_TOLERANCE = 0.003
COHORT_ID = "SYN-V3-T"
V2_SELECTED_SHA = "6fed05f3c71ce648658b2f864e69fa9f80f3ec8538f53a4e370fa8a4eaf176a8"
V2_FINAL_FREEZE_SHA = "a433c1ee203f6cf9f01b131a2a9a10c73da57097564705ce64d7351c0057b649"
SCIENTIFIC_CODE_FILES = (
    "src/data/synthetic/latent_state.py", "src/data/synthetic/observations.py", "src/data/synthetic/support.py",
    "src/data/synthetic/records.py", "src/data/synthetic/rng.py", "src/data/synthetic/validation.py",
    "src/data/synthetic/sofa.py", "src/data/synthetic/sofa_provider.py", "src/labels/recovery.py",
    "src/labels/icu_time.py", "src/labels/organ_support.py", "src/labels/support_state.py",
    "src/labels/ventilation_state.py", "src/features/synthetic.py", "src/data/synthetic/generator.py",
    "src/labels/phase9_final.py",
)
SCIENTIFIC_CONFIG_KEYS = (
    "population", "calendar", "duration", "latent", "observation", "variables", "latent_process_path",
    "support_process_path", "event_dictionary_path", "minimum_accepted_episode_hours",
)


class FrozenArtifactError(RuntimeError):
    """A frozen, hash-bound input changed or is missing."""


class RowIdentityError(RuntimeError):
    """Models were not scored on identical eligible row identities."""


# ---------------------------------------------------------------------------
# paths / hashing
# ---------------------------------------------------------------------------
def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default) + "\n").encode()


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(type(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_canonical(path: Path, value: object) -> str:
    """Write canonical JSON + a `<name>.sha256` sidecar; returns the file sha256."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def resolve(root: Path, ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else Path(root) / p


def _rel(root: Path, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


class Paths:
    """All artifact locations; `base` defaults to <root>/artifacts/performance_v3 (overridable for tests)."""

    def __init__(self, root: Path, base: Optional[Path] = None, cohort_dir: Optional[Path] = None):
        self.root = Path(root)
        self.base = Path(base) if base is not None else self.root / "artifacts/performance_v3"
        self.cohort_dir = Path(cohort_dir) if cohort_dir is not None else self.base / "cohorts/T"
        g = self.base / "governance"
        self.v3_bundle = self.base / "phase3/v3/bundle_manifest.json"
        self.control_bundle = self.base / "phase3/control/bundle_manifest.json"
        self.selected_models = self.base / "phase3/selected_models_v3.json"
        self.freeze = g / "v3_model_freeze_v1.json"
        self.plan = g / "v3_pretest_evaluation_plan_v1.json"
        self.seal = g / "v3_test_cohort_seal_v1.json"
        self.state = g / "v3_final_test_access_state.json"
        self.preflight = g / "v3_pretest_preflight_v1.json"
        self.final_freeze = g / "v3_final_evaluation_freeze_v1.json"
        self.second_run = g / "v3_second_run_guard_v1.json"
        self.incident = g / "v3_final_test_incident_v1.json"
        p4 = self.base / "phase4"
        self.predictions = p4 / "predictions/test_predictions_v1.jsonl"
        self.predictions_manifest = p4 / "predictions/predictions_manifest_v1.json"
        self.final_metrics = p4 / "final_metrics_v3.json"
        self.paired_metrics = p4 / "paired_v2_v3_metrics.json"
        self.paired_bootstrap = p4 / "paired_bootstrap_v1.json"
        self.calibration = p4 / "calibration_evidence_v1.json"
        self.error_analysis = p4 / "error_analysis_v1.json"
        self.plots = p4 / "plots"
        self.protocol = self.root / "configs/performance_v3/scientific_protocol_v1.yaml"
        self.amendment = self.root / "configs/performance_v3/protocol_amendment_v1_1.yaml"
        self.v2_selected = self.root / "artifacts/performance_v2/phase3/selected_models_v2.json"
        self.v2_final_freeze = self.root / "artifacts/performance_v2/governance/v2_final_evaluation_freeze_v1.json"
        self.dev_pool_cache = self.base / "cache/dev_pool"
        self.search_ledger = self.base / "phase2/trial_ledger_v1.jsonl"
        self.learning_curve = self.base / "data_characterization/learning_curve_metrics.json"


# ---------------------------------------------------------------------------
# bundle hash collection / verification
# ---------------------------------------------------------------------------
def bundle_files(root: Path, manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """Every file a V3Bundle manifest depends on, with its CURRENT sha256; verifies stored hashes."""

    manifest_path = resolve(root, str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {"manifest": {"path": _rel(root, manifest_path), "sha256": sha256_file(manifest_path)}}
    for task, entry in manifest["tasks"].items():
        meta_path = resolve(root, entry["model_ref"])
        names_path = resolve(root, entry["feature_names_ref"])
        for label, path, expected in ((f"{task}:model_meta", meta_path, entry["model_sha256"]),
                                      (f"{task}:feature_names", names_path, entry["feature_names_sha256"])):
            actual = sha256_file(path)
            if actual != expected:
                raise FrozenArtifactError(f"{label} hash mismatch: manifest {expected} != file {actual}")
            out[label] = {"path": _rel(root, path), "sha256": actual}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for f in meta.get("files", []):
            fp = meta_path.parent / f["path"]
            actual = sha256_file(fp)
            if actual != f["sha256"]:
                raise FrozenArtifactError(f"{task}:{f['path']} hash mismatch")
            out[f"{task}:model_file:{f['path']}"] = {"path": _rel(root, fp), "sha256": actual}
    cal_block = manifest["calibrator"]
    if params_sha256(cal_block["params"]) != cal_block["sha256"]:
        raise FrozenArtifactError("calibrator params sha256 mismatch")
    out["calibrator_params"] = {"path": _rel(root, manifest_path) + "#calibrator.params", "sha256": cal_block["sha256"]}
    return out


def verify_bound_files(root: Path, bound: Mapping[str, Mapping[str, str]]) -> None:
    for label, rec in bound.items():
        if rec["path"].endswith("#calibrator.params"):
            continue
        path = resolve(root, rec["path"])
        if not path.is_file():
            raise FrozenArtifactError(f"frozen file missing: {label} -> {rec['path']}")
        if sha256_file(path) != rec["sha256"]:
            raise FrozenArtifactError(f"frozen file changed: {label} -> {rec['path']}")


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # pragma: no cover
        return "UNKNOWN"


def environment_record() -> Dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "sklearn", "xgboost", "lightgbm", "catboost", "torch", "scipy", "pandas"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "absent"
    return out


# ---------------------------------------------------------------------------
# slices (cut-points from the development pool only)
# ---------------------------------------------------------------------------
def row_observation_density(row: Mapping[str, object]) -> float:
    cells = [bool(m) for bin_row in row["observation_mask"] for m in bin_row]
    return float(sum(cells) / len(cells)) if cells else 0.0


def cache_observation_density(names: Sequence[str], matrix: np.ndarray) -> np.ndarray:
    columns = [i for i, n in enumerate(names) if n.startswith("A__bin") and n.endswith("__mask")]
    if not columns:
        raise FrozenArtifactError("no observation-mask columns in cache")
    return np.asarray(matrix[:, columns], dtype=np.float64).mean(axis=1)


def slice_cutpoints_from_cache(cache) -> Dict[str, object]:
    density = cache_observation_density(cache.names, cache.matrix)
    hours = np.expm1(np.asarray(cache.meta["icu_time_log1p"], dtype=np.float64))
    hours = hours[np.isfinite(hours)]
    return {
        "source": "development pool feature cache ONLY",
        "observation_density_tertile_edges": [float(np.percentile(density, 100 / 3)), float(np.percentile(density, 200 / 3))],
        "remaining_time_quartile_edges_hours": [float(np.percentile(hours, q)) for q in (25, 50, 75)],
        "cutoff_bands": {"0-1": [0, 1], "2-4": [2, 4], ">=5": [5, None]},
        "n_cache_rows": int(cache.matrix.shape[0]),
    }


def _band(grid_index: int) -> str:
    return "0-1" if grid_index <= 1 else ("2-4" if grid_index <= 4 else ">=5")


def _bucket(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def slice_labels(line: Mapping[str, object], cut: Mapping[str, object], task: str) -> Dict[str, str]:
    lab = line["labels"]
    out = {
        "cutoff_band": _band(int(line["grid_index"])),
        "cardiac_subtype": str(line.get("cardiac_subtype") or "UNKNOWN"),
        "observation_density_tertile": _bucket(float(line["observation_density"]), cut["observation_density_tertile_edges"], ("T1_low", "T2_mid", "T3_high")),
    }
    if lab.get("icu_time_log1p") is not None:
        hours = math.expm1(float(lab["icu_time_log1p"]))
        out["remaining_time_quartile"] = _bucket(hours, cut["remaining_time_quartile_edges_hours"], ("Q1", "Q2", "Q3", "Q4"))
    if task == "recovery24" and lab.get("delta_sofa_24") is not None:
        d = float(lab["delta_sofa_24"]); out["recovery_target_sign"] = "negative" if d < 0 else ("zero" if d == 0 else "positive")
    if task == "recovery48" and lab.get("delta_sofa_48") is not None:
        d = float(lab["delta_sofa_48"]); out["recovery_target_sign"] = "negative" if d < 0 else ("zero" if d == 0 else "positive")
    if task == "organ_support" and lab.get("organ_support_label") is not None:
        out["support_outcome"] = "positive" if float(lab["organ_support_label"]) >= 0.5 else "negative"
    return out


# ---------------------------------------------------------------------------
# freeze / identity
# ---------------------------------------------------------------------------
def make_plan(root: Path, paths: Paths, cutpoints: Mapping[str, object]) -> Dict[str, object]:
    return {
        "artifact_version": "performance_v3_pretest_evaluation_plan_v1",
        "status": "FROZEN_BEFORE_TEST_ACCESS",
        "metrics": {
            "recovery24": {"primary": "stay_balanced_mae_delta_sofa_24", "secondary": list(SECONDARY["recovery24"])},
            "recovery48": {"primary": "stay_balanced_mae_delta_sofa_48", "secondary": list(SECONDARY["recovery48"])},
            "icu_stay_time": {"primary": "stay_balanced_weighted_median_ae_hours", "secondary": list(SECONDARY["icu_stay_time"]), "units": "hours, expm1(max(raw,0))"},
            "organ_support": {"primary": "stay_balanced_calibrated_auprc", "secondary": ["raw_auprc", "raw_auroc", "calibrated_auroc", "brier", "log_loss", "f1_at_frozen_threshold", "precision", "recall", "specificity", "confusion_matrix"]},
        },
        "models": {
            "V3": "frozen V3 bundle (selected_models_v3.json)",
            "V2_AS_SERVED": "frozen V2 models fed SOFA(t) for every row (PRIMARY comparator; behaviour of the V2 serving runtime)",
            "V2_AS_EVALUATED": "frozen V2 models fed pre-split baseline_sofa with nulls preserved (V2 offline harness)",
            "CONTROL": "V2 recipe refit on the 8,000-subject pool with corrected features (secondary)",
        },
        "primary_comparator": PRIMARY_COMPARATOR,
        "row_identity": "all four models scored on IDENTICAL rows; any eligible-row mismatch is a pipeline investigation, never a row drop",
        "paired_bootstrap": {"type": "stay_cluster_percentile_paired", "replicates": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                             "ci_levels_reporting": [0.95], "ci_level_promotion": PROMOTION_CI_LEVEL, "difference": "V3 - comparator",
                             "same_resamples_for_both_models": True},
        "prespecified_slices": ["cutoff_band", "cardiac_subtype", "observation_density_tertile", "recovery_target_sign",
                                "remaining_time_quartile", "support_outcome"],
        "slice_cutpoints": cutpoints,
        "exploratory_slices_policy": "must be labelled exploratory; may not affect any decision",
        "clean_subset": "rows with >=24h follow-up available (recovery24-eligible, non-null baseline_sofa)",
        "promotion_rule": {
            "unit": "per task", "basis_comparator": PRIMARY_COMPARATOR,
            "promote_v3_if": "paired primary-metric difference favours V3 AND the 98.75% paired CI excludes 0",
            "support_additional": f"delta Brier point estimate <= +{BRIER_TOLERANCE}",
            "otherwise": "retain frozen V2 for that task; mixed bundle allowed per task",
            "no_post_test_selection": True,
        },
        "test_access_procedure": ["freeze verified", "authorize(freeze_sha, plan_sha)", "consume_and_load_rows (marks consumed first)",
                                  "score four models on identical rows", "save predictions immutably", "complete()", "metrics from saved predictions only"],
        "failure_policy": {"before_access": "fix development path, rerun preflight", "after_access": "incident record; no scientific retry; recompute from saved predictions only"},
    }


def freeze(root: Path, paths: Paths, *, cutpoints: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """Verify every hash, write selected_models_v3.json, the freeze record and the evaluation plan."""

    root = Path(root)
    bound: Dict[str, Dict[str, str]] = {}
    for label, manifest in (("v3", paths.v3_bundle), ("control", paths.control_bundle)):
        for key, rec in bundle_files(root, manifest).items():
            bound[f"{label}:{key}"] = rec
    for label, path, expected in (("v2_selected_models", paths.v2_selected, V2_SELECTED_SHA),
                                  ("v2_final_evaluation_freeze", paths.v2_final_freeze, V2_FINAL_FREEZE_SHA)):
        actual = sha256_file(path)
        if actual != expected:
            raise FrozenArtifactError(f"{label} sha256 {actual} != expected {expected}")
        bound[label] = {"path": _rel(root, path), "sha256": actual}
    for label, path in (("protocol", paths.protocol), ("amendment_v1_1", paths.amendment)):
        bound[label] = {"path": _rel(root, path), "sha256": sha256_file(path)}
    for label, path in (("search_ledger", paths.search_ledger), ("learning_curve", paths.learning_curve),
                        ("dev_pool_cache_meta", paths.dev_pool_cache.with_suffix(".json"))):
        if Path(path).is_file():
            bound[label] = {"path": _rel(root, path), "sha256": sha256_file(path)}
    if cutpoints is None:
        from performance_v3.features import load_feature_cache
        cutpoints = slice_cutpoints_from_cache(load_feature_cache(paths.dev_pool_cache))

    v3_manifest = json.loads(paths.v3_bundle.read_text(encoding="utf-8"))
    v3_manifest = dict(v3_manifest, status="FROZEN_PRETEST")
    selected_sha = write_canonical(paths.selected_models, v3_manifest)
    bound["selected_models_v3"] = {"path": _rel(root, paths.selected_models), "sha256": selected_sha}

    plan = make_plan(root, paths, cutpoints)
    plan_sha = write_canonical(paths.plan, plan)
    freeze_record = {
        "artifact_version": "performance_v3_model_freeze_v1",
        "status": "V3_MODELS_FROZEN_PRETEST",
        "test_accessed": False,
        "code_commit": _git_head(root),
        "environment": environment_record(),
        "bound_files": bound,
        "plan_ref": _rel(root, paths.plan),
        "plan_sha256": plan_sha,
        "selected_models_v3_sha256": selected_sha,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    freeze_sha = write_canonical(paths.freeze, freeze_record)
    return {"freeze_sha256": freeze_sha, "plan_sha256": plan_sha, "selected_models_v3_sha256": selected_sha, "bound_files": len(bound)}


def verify_frozen(root: Path, paths: Paths) -> Tuple[str, str]:
    """Re-verify the freeze record, the plan and every bound file; returns (freeze_sha, plan_sha)."""

    if not paths.freeze.is_file() or not paths.plan.is_file():
        raise FrozenArtifactError("freeze record or evaluation plan missing")
    record = json.loads(paths.freeze.read_text(encoding="utf-8"))
    plan_sha = sha256_file(paths.plan)
    if plan_sha != record["plan_sha256"]:
        raise FrozenArtifactError("evaluation plan changed after freeze")
    verify_bound_files(root, record["bound_files"])
    return sha256_file(paths.freeze), plan_sha


def identity(root: Path) -> Dict[str, object]:
    from data.synthetic.config import FINAL_IDENTITIES
    from performance_v3.protocol import cohort_identity

    ident = cohort_identity(Path(root), "T")
    key = (ident["seed"], ident["n_subjects"])
    registered = FINAL_IDENTITIES.get(key) == ident["namespace"]
    return {
        **ident,
        "registered": bool(registered),
        "final_identities_line": f'    ({ident["seed"]}, {ident["n_subjects"]}): "{ident["namespace"]}",',
    }


# ---------------------------------------------------------------------------
# seal (structural only)
# ---------------------------------------------------------------------------
def seal(root: Path, paths: Paths, *, peer_raw_dirs: Optional[Mapping[str, Path]] = None,
         check_generator: bool = True, init_ledger: bool = True) -> Dict[str, object]:
    from data.synthetic.validation import load_jsonl
    from performance_v3.audit import overlap_report
    from preprocess.synthetic_phase10 import exact_raw_trajectory_fingerprints

    root = Path(root)
    t = paths.cohort_dir
    raw = t / "raw"
    needed = {
        "raw_manifest": raw / "synthetic_dataset_manifest_v1.json", "subjects": raw / "subjects.jsonl", "episodes": raw / "episodes.jsonl",
        "events": raw / "raw_events.jsonl", "supports": raw / "support_intervals.jsonl",
        "structural_index": t / "cohort/structural_index.jsonl", "timeline_manifest": t / "timeline/synthetic_processed_manifest_v1.json",
        "pre_split_exists": t / "pre_split/pre_split_scientific_package.jsonl",
    }
    missing = [k for k, p in needed.items() if not p.is_file()]
    checks: Dict[str, Dict[str, object]] = {"files_exist": {"pass": not missing, "missing": missing}}
    if missing:
        return {"verdict": "FAIL", "checks": checks}
    ident = identity(root)
    manifest = json.loads(needed["raw_manifest"].read_text(encoding="utf-8"))
    checks["manifest_identity"] = {
        "pass": manifest.get("manifest_status") == "AUTHORIZED_FINAL_SYNTHETIC_DATA"
        and manifest.get("primary_seed_identity", {}).get("value") == ident["seed"]
        and int(manifest.get("generated_subject_count", -1)) == int(ident["n_subjects"]),
        "seed": ident["seed"], "n_subjects": ident["n_subjects"], "namespace": ident["namespace"],
    }
    subjects = list(load_jsonl(needed["subjects"]))
    episodes = list(load_jsonl(needed["episodes"]))
    ns = str(ident["namespace"])
    subject_ids = [r["subject_id"] for r in subjects]
    stay_ids = [r["stay_id"] for r in episodes]
    checks["uniqueness_and_namespace"] = {
        "pass": len(set(subject_ids)) == len(subject_ids) == len(set(stay_ids)) == len(stay_ids)
        and all(s.startswith(ns + "-S-") for s in subject_ids) and all(s.startswith(ns + "-E-") for s in stay_ids),
        "subjects": len(subject_ids), "stays": len(stay_ids),
    }
    index_stays = [r["stay_id"] for r in load_jsonl(needed["structural_index"])]
    checks["structural_index_counts"] = {
        "pass": len(index_stays) > 0 and set(index_stays) <= set(stay_ids),
        "rows": len(index_stays), "stays_with_rows": len(set(index_stays)),
        "max_rows_per_stay": int(max(np.bincount(np.unique(index_stays, return_inverse=True)[1]))) if index_stays else 0,
    }
    events = load_jsonl(needed["events"])
    supports = load_jsonl(needed["supports"])
    entries = {"T": {"subject_ids": set(subject_ids), "stay_ids": set(stay_ids),
                     "fingerprints": dict(exact_raw_trajectory_fingerprints(subjects, episodes, events, supports))}}
    peers = peer_raw_dirs
    if peers is None:
        base = root / "artifacts/performance_v3/cohorts"
        peers = {"V2_FRESH_TEST": root / "artifacts/performance_v2/phase3/fresh_test_cohort/raw",
                 "V1": root / "artifacts/data/synthetic/final/phase9_final_v1"}
        peers.update({c: base / c / "raw" for c in ("D1", "D2", "D3", "LCE")})
    for name, directory in peers.items():
        directory = Path(directory)
        if not (directory / "subjects.jsonl").is_file():
            continue
        s = list(load_jsonl(directory / "subjects.jsonl")); e = list(load_jsonl(directory / "episodes.jsonl"))
        entries[name] = {"subject_ids": {r["subject_id"] for r in s}, "stay_ids": {r["stay_id"] for r in e},
                         "fingerprints": dict(exact_raw_trajectory_fingerprints(s, e, load_jsonl(directory / "raw_events.jsonl"), load_jsonl(directory / "support_intervals.jsonl")))}
    overlap = overlap_report(entries, test_cohorts=[n for n in entries if n.startswith("V1") or n.startswith("V2")], v3_cohorts=["T"])
    t_pairs = {k: v for k, v in overlap["pairwise"].items() if "T" in k.split("|")}
    checks["overlap"] = {
        "pass": not overlap["blockers"] and all(v["subject_intersection"] == 0 and v["stay_intersection"] == 0 and v["fingerprint_intersection"] == 0 for v in t_pairs.values())
        and overlap["within_cohort_duplicate_fingerprints"].get("T", 0) == 0,
        "peers_compared": sorted(n for n in entries if n != "T"), "blockers": overlap["blockers"], "pairs_with_T": t_pairs,
    }
    if check_generator:
        checks["generator_equivalence"] = _generator_equivalence(root, ident)
    checks["registered_final_identity"] = {"pass": bool(ident["registered"])}
    verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    record = {
        "artifact_version": "performance_v3_test_cohort_seal_v1", "verdict": verdict, "structural_only": True,
        "labels_eligibility_or_qa_counts_read": False, "cohort_id": COHORT_ID, "identity": {k: ident[k] for k in ("seed", "n_subjects", "namespace", "seed_material", "seed_material_sha256")},
        "checks": checks, "file_hashes": {k: sha256_file(p) for k, p in needed.items() if k != "pre_split_exists"},
    }
    sha = write_canonical(paths.seal, record)
    if verdict == "PASS" and init_ledger:
        access = SealedCohortAccess(root, COHORT_ID, paths.state, lambda r: (), expected_paths=(), sealed_paths=(t,))
        if not paths.state.exists():
            access.initialize()
    return {"verdict": verdict, "seal_sha256": sha, "checks": {k: v["pass"] for k, v in checks.items()}}


def _generator_equivalence(root: Path, ident: Mapping[str, object]) -> Dict[str, object]:
    from data.synthetic.config import load_runtime_config

    cfg_path = root / "configs/performance_v3/cohorts/t_v1.json"
    out: Dict[str, object] = {"pass": False}
    if not cfg_path.is_file():
        out["error"] = "configs/performance_v3/cohorts/t_v1.json missing (run performance_v3_cohorts.py plan T)"
        return out
    values = load_runtime_config(cfg_path, root).values
    base = load_runtime_config(root / "configs/synthetic/final_benchmark_v1.json", root).values
    sci = [k for k in SCIENTIFIC_CONFIG_KEYS if values.get(k) != base.get(k)]
    inventory = root / "artifacts/performance_v3/governance/pre_phase_hash_inventory_v1.sha256"
    inv = {}
    for line in inventory.read_text().splitlines():
        digest, path = line.split("  ", 1)
        inv[path] = digest
    code = {p: (inv.get(p) == sha256_file(root / p)) for p in SCIENTIFIC_CODE_FILES}
    out.update({
        "scientific_key_diffs": sci, "seed_matches_derivation": values["primary_seed"] == ident["seed"],
        "scientific_code_unchanged": all(code.values()), "code_changed_files": sorted(p for p, ok in code.items() if not ok),
        "pass": not sci and values["primary_seed"] == ident["seed"] and all(code.values()) and values.get("mode") == "final",
    })
    return out


# ---------------------------------------------------------------------------
# prediction building (single operation) and persistence
# ---------------------------------------------------------------------------
_LABEL_FIELDS = ("delta_sofa_24", "delta_sofa_48", "icu_time_log1p", "organ_support_label", "baseline_sofa",
                 "recovery24_eligible", "recovery48_eligible", "icu_time_eligible", "organ_support_eligible")
_PRED_FIELDS = ("recovery24_delta", "recovery48_delta", "icu_remaining_hours", "icu_raw_log_prediction", "support_raw_probability",
                "support_calibrated_probability", "support_threshold", "support_alert", "current_sofa", "recovery24_sofa", "recovery48_sofa")


def _static_group(row: Mapping[str, object]) -> str:
    if row.get("cardiac_condition_group"):
        return str(row["cardiac_condition_group"])
    names, values = row.get("static_feature_names") or (), row.get("static_features") or ()
    for n, v in zip(names, values):
        if n == "cardiac_condition_group":
            return str(v)
    return "UNKNOWN"


def build_prediction_lines(rows: Sequence[Mapping[str, object]], outputs: Mapping[str, Sequence[Mapping[str, object]]]) -> List[Dict[str, object]]:
    lines: List[Dict[str, object]] = []
    for model, preds in outputs.items():
        if len(preds) != len(rows):
            raise RowIdentityError(f"{model}: {len(preds)} predictions for {len(rows)} rows")
    for i, row in enumerate(rows):
        models = {}
        for model, preds in outputs.items():
            p = preds[i]
            if p["stay_id"] != row["stay_id"] or p["prediction_time"] != row["prediction_time"]:
                raise RowIdentityError(f"{model}: row {i} identity differs from the loaded row")
            models[model] = {k: p.get(k) for k in _PRED_FIELDS}
        lines.append({
            "stay_id": row["stay_id"], "prediction_time": row["prediction_time"], "subject_id": row.get("subject_id"),
            "grid_index": int(row["grid_index"]), "cardiac_subtype": _static_group(row),
            "observation_density": row_observation_density(row),
            "labels": {k: (None if row.get(k) is None else (bool(row[k]) if k.endswith("_eligible") else float(row[k]))) for k in _LABEL_FIELDS},
            "models": models,
        })
    return lines


def save_predictions(paths: Paths, lines: Sequence[Mapping[str, object]]) -> str:
    target = paths.predictions
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FrozenArtifactError("predictions file already exists; refusing to overwrite an immutable result")
    with target.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, sort_keys=True, separators=(",", ":"), default=_json_default) + "\n")
    target.chmod(0o444)
    digest = sha256_file(target)
    write_canonical(paths.predictions_manifest, {"artifact_version": "performance_v3_test_predictions_v1", "rows": len(lines),
                                                "models": list(MODELS), "predictions_sha256": digest, "file": target.name})
    return digest


def load_predictions(paths: Paths) -> List[Dict[str, object]]:
    manifest = json.loads(paths.predictions_manifest.read_text(encoding="utf-8"))
    if sha256_file(paths.predictions) != manifest["predictions_sha256"]:
        raise FrozenArtifactError("saved predictions changed after being written")
    return [json.loads(l) for l in paths.predictions.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# metrics from saved predictions
# ---------------------------------------------------------------------------
def _target(task: str, lab: Mapping[str, object]) -> Optional[float]:
    if task == "recovery24":
        return float(lab["delta_sofa_24"]) if lab.get("recovery24_eligible") and lab.get("delta_sofa_24") is not None else None
    if task == "recovery48":
        return float(lab["delta_sofa_48"]) if lab.get("recovery48_eligible") and lab.get("delta_sofa_48") is not None else None
    if task == "icu_stay_time":
        return float(math.expm1(float(lab["icu_time_log1p"]))) if lab.get("icu_time_eligible") and lab.get("icu_time_log1p") is not None else None
    return float(lab["organ_support_label"]) if lab.get("organ_support_eligible") and lab.get("organ_support_label") is not None else None


def _pred(task: str, p: Mapping[str, object], kind: str = "calibrated") -> float:
    if task == "recovery24":
        return float(p["recovery24_delta"])
    if task == "recovery48":
        return float(p["recovery48_delta"])
    if task == "icu_stay_time":
        return float(p["icu_remaining_hours"])
    return float(p["support_raw_probability"] if kind == "raw" else p["support_calibrated_probability"])


def make_task_records(lines: Sequence[Mapping[str, object]], model: str, task: str, *, kind: str = "calibrated",
                      subset: Optional[Callable[[Mapping[str, object]], bool]] = None) -> List[PredictionRecord]:
    out = []
    for line in lines:
        if subset is not None and not subset(line):
            continue
        y = _target(task, line["labels"])
        if y is None:
            continue
        out.append(PredictionRecord(stay_id=line["stay_id"], prediction_time=line["prediction_time"], target=y,
                                    prediction=_pred(task, line["models"][model], kind), eligible=True))
    return out


def eligible_identity_sets(lines: Sequence[Mapping[str, object]]) -> Dict[str, List[Tuple[str, str]]]:
    """Per task eligible (stay, time) identities; identical for every model by construction, asserted."""

    out = {}
    for task in TASKS:
        base = [(l["stay_id"], l["prediction_time"]) for l in lines if _target(task, l["labels"]) is not None]
        for model in MODELS:
            mine = [(l["stay_id"], l["prediction_time"]) for l in lines if _target(task, l["labels"]) is not None
                    and all(l["models"][model].get(k) is not None for k in _needed_pred_fields(task))]
            if mine != base:
                raise RowIdentityError(f"{task}/{model}: eligible-row identity set differs ({len(mine)} vs {len(base)})")
        out[task] = base
    return out


def _needed_pred_fields(task: str) -> Tuple[str, ...]:
    return {"recovery24": ("recovery24_delta",), "recovery48": ("recovery48_delta",), "icu_stay_time": ("icu_remaining_hours",),
            "organ_support": ("support_raw_probability", "support_calibrated_probability")}[task]


def point_metrics(task: str, records: Sequence[PredictionRecord], threshold: Optional[float] = None) -> Dict[str, Optional[float]]:
    names = [PRIMARY_METRIC[task], *SECONDARY[task]]
    if task == "organ_support":
        names += ["precision", "recall", "specificity"]
    out: Dict[str, Optional[float]] = {}
    for name in dict.fromkeys(names):
        if name in ("f1", "precision", "recall", "specificity") and threshold is None:
            continue
        v = tm.record_metric_fn(task, name, threshold=threshold)(records)
        out[name] = None if v is None or not math.isfinite(v) else float(v)
    out["n_rows"] = len(records)
    out["n_stays"] = len({r.stay_id for r in records})
    return out


def _threshold(lines: Sequence[Mapping[str, object]], model: str) -> float:
    return float(lines[0]["models"][model]["support_threshold"])


def _cis_from_replicates(values: np.ndarray, levels: Sequence[float]) -> Dict[str, Optional[List[float]]]:
    values = values[np.isfinite(values)]
    out: Dict[str, Optional[List[float]]] = {}
    for level in levels:
        if len(values) == 0:
            out["%g" % level] = None
        else:
            a = (1 - level) / 2
            out["%g" % level] = [float(np.quantile(values, a)), float(np.quantile(values, 1 - a))]
    return out


def evaluate_predictions(lines: Sequence[Mapping[str, object]], plan: Mapping[str, object], *, n_bootstrap: Optional[int] = None,
                         seed: Optional[int] = None) -> Dict[str, object]:
    """All reported numbers, computed ONLY from saved prediction lines (no model inference)."""

    boot = plan["paired_bootstrap"]
    n = int(n_bootstrap if n_bootstrap is not None else boot["replicates"])
    seed = int(seed if seed is not None else boot["seed"])
    levels = (0.95, PROMOTION_CI_LEVEL)
    identities = eligible_identity_sets(lines)

    thresholds = {m: _threshold(lines, m) for m in MODELS}
    models_out: Dict[str, Dict[str, object]] = {m: {} for m in MODELS}
    for model in MODELS:
        for task in TASKS:
            recs = make_task_records(lines, model, task)
            models_out[model][task] = point_metrics(task, recs, thresholds[model] if task == "organ_support" else None)
        raw = make_task_records(lines, model, "organ_support", kind="raw")
        models_out[model]["organ_support_raw_scores"] = {k: tm.record_metric_fn("organ_support", k)(raw) for k in ("auprc", "auroc", "brier", "log_loss")}

    paired: Dict[str, Dict[str, object]] = {}
    arrays: Dict[str, Dict[str, object]] = {}
    for comparator in COMPARATORS:
        for task in TASKS:
            metric = PRIMARY_METRIC[task]
            a = make_task_records(lines, comparator, task)
            b = make_task_records(lines, "V3", task)
            res = paired_stay_cluster_bootstrap(a, b, tm.record_metric_fn(task, metric), n=n, seed=seed, ci_levels=levels, metric_name=metric, task=task)
            paired[f"{task}|{metric}|V3_vs_{comparator}"] = _strip(res)
            arrays[f"{task}|{metric}|V3_vs_{comparator}"] = {"a": res["per_replicate_all_a"], "b": res["per_replicate_all_b"]}
    for task in TASKS:
        for name in SECONDARY[task]:
            kinds = ["calibrated"] + (["raw"] if task == "organ_support" and name in ("auroc", "auprc") else [])
            for kind in kinds:
                a = make_task_records(lines, PRIMARY_COMPARATOR, task, kind=kind)
                b = make_task_records(lines, "V3", task, kind=kind)
                threshold = None
                if name == "f1":
                    # each model uses its own frozen threshold: bootstrap on a shared-threshold estimator is undefined, so
                    # report F1 through the point estimates only
                    continue
                res = paired_stay_cluster_bootstrap(a, b, tm.record_metric_fn(task, name, threshold=threshold), n=n, seed=seed,
                                                    ci_levels=levels, metric_name=name + ("_raw" if kind == "raw" else ""), task=task)
                paired[f"{task}|{name}{'_raw' if kind == 'raw' else ''}|V3_vs_{PRIMARY_COMPARATOR}"] = _strip(res)
    # additional raw-score AUPRC/AUROC paired difference (support) so discrimination effects are separable from calibration
    for name in ("auprc",):
        a = make_task_records(lines, PRIMARY_COMPARATOR, "organ_support", kind="raw")
        b = make_task_records(lines, "V3", "organ_support", kind="raw")
        res = paired_stay_cluster_bootstrap(a, b, tm.record_metric_fn("organ_support", name), n=n, seed=seed, ci_levels=levels, metric_name=name + "_raw", task="organ_support")
        paired[f"organ_support|{name}_raw|V3_vs_{PRIMARY_COMPARATOR}"] = _strip(res)

    # per-model primary-metric CIs from the replicate arrays of the primary-comparator pairings
    for task in TASKS:
        key = f"{task}|{PRIMARY_METRIC[task]}|V3_vs_{PRIMARY_COMPARATOR}"
        models_out["V3"][task]["primary_ci"] = _cis_from_replicates(arrays[key]["b"], (0.95,))
        for comparator in COMPARATORS:
            k = f"{task}|{PRIMARY_METRIC[task]}|V3_vs_{comparator}"
            models_out[comparator][task]["primary_ci"] = _cis_from_replicates(arrays[k]["a"], (0.95,))

    promotion = _promotion(paired, models_out)
    clean = _clean_subset(lines, thresholds)
    return {"models": models_out, "paired": paired, "promotion": promotion, "clean_subset": clean, "identities": {t: len(v) for t, v in identities.items()},
            "thresholds": thresholds, "bootstrap": {"n": n, "seed": seed, "ci_levels": list(levels)}}


def _strip(res: Mapping[str, object]) -> Dict[str, object]:
    keep = ("version", "metric_name", "task", "difference_definition", "seed", "n_requested", "n_valid_joint", "n_invalid",
            "n_stays", "n_rows", "point_a", "point_b", "diff_point", "ci")
    out = {k: res[k] for k in keep}
    out["ci_excludes_zero_95"] = bool(ci_excludes_zero(res, 0.95))
    out["ci_excludes_zero_9875"] = bool(ci_excludes_zero(res, PROMOTION_CI_LEVEL))
    out["per_replicate_diff"] = [round(float(v), 8) for v in res["per_replicate_diff"]]
    return out


def _promotion(paired: Mapping[str, Mapping[str, object]], models: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    out = {}
    for task in TASKS:
        metric = PRIMARY_METRIC[task]
        res = paired[f"{task}|{metric}|V3_vs_{PRIMARY_COMPARATOR}"]
        diff = res["diff_point"]
        higher_better = task == "organ_support"
        favours = diff is not None and ((diff > 0) if higher_better else (diff < 0))
        promote = bool(favours and res["ci_excludes_zero_9875"])
        reason = "primary delta favours V3 and 98.75% paired CI excludes 0" if promote else "primary delta does not satisfy the pre-registered rule"
        entry = {"primary_metric": metric, "delta_v3_minus_comparator": diff, "ci_9875": res["ci"].get("%g" % PROMOTION_CI_LEVEL),
                 "favours_v3": bool(favours), "ci_excludes_zero_9875": res["ci_excludes_zero_9875"]}
        if task == "organ_support":
            brier = paired.get(f"organ_support|brier|V3_vs_{PRIMARY_COMPARATOR}")
            ok = brier is not None and brier["diff_point"] is not None and brier["diff_point"] <= BRIER_TOLERANCE
            entry["delta_brier"] = None if brier is None else brier["diff_point"]
            entry["brier_condition_met"] = bool(ok)
            promote = bool(promote and ok)
            if not ok:
                reason = "delta Brier exceeds +%.3f" % BRIER_TOLERANCE if favours else reason
        entry["decision"] = "PROMOTE_V3" if promote else "RETAIN_V2"
        entry["reason"] = reason
        out[task] = entry
    return out


def _clean_subset(lines: Sequence[Mapping[str, object]], thresholds: Mapping[str, float]) -> Dict[str, object]:
    def subset(line):
        return bool(line["labels"].get("recovery24_eligible"))
    out = {"definition": ">=24h follow-up available (recovery24-eligible); V2_AS_EVALUATED == V2_AS_SERVED on these rows", "rows": sum(1 for l in lines if subset(l)), "models": {}}
    for model in MODELS:
        out["models"][model] = {}
        for task in TASKS:
            recs = make_task_records(lines, model, task, subset=subset)
            out["models"][model][task] = point_metrics(task, recs, thresholds[model] if task == "organ_support" else None) if recs else None
    return out


def calibration_evidence(lines: Sequence[Mapping[str, object]], max_curve_points: int = 200) -> Dict[str, object]:
    out: Dict[str, object] = {"artifact_version": "performance_v3_calibration_evidence_v1", "models": {}}
    for model in MODELS:
        recs_c = make_task_records(lines, model, "organ_support", kind="calibrated")
        recs_r = make_task_records(lines, model, "organ_support", kind="raw")
        y = np.asarray([r.target for r in recs_c]); w = tm.stay_weights([r.stay_id for r in recs_c])
        pc = np.asarray([r.prediction for r in recs_c]); pr = np.asarray([r.prediction for r in recs_r])
        thr = _threshold(lines, model)
        pred_pos = pc >= thr
        conf_w = {"TP": float(w[(y == 1) & pred_pos].sum()), "FP": float(w[(y == 0) & pred_pos].sum()),
                  "TN": float(w[(y == 0) & ~pred_pos].sum()), "FN": float(w[(y == 1) & ~pred_pos].sum())}
        conf_n = {"TP": int(((y == 1) & pred_pos).sum()), "FP": int(((y == 0) & pred_pos).sum()),
                  "TN": int(((y == 0) & ~pred_pos).sum()), "FN": int(((y == 1) & ~pred_pos).sum())}
        out["models"][model] = {
            "n_rows": int(len(y)), "n_stays": len({r.stay_id for r in recs_c}), "prevalence_stay_balanced": float(np.sum(w * y) / np.sum(w)),
            "calibrated": cal.calibration_metrics(pc, y, w), "raw": cal.calibration_metrics(pr, y, w),
            "reliability_calibrated": _reliability(pc, y, w), "reliability_raw": _reliability(pr, y, w),
            "roc_calibrated": _curve(y, pc, w, "roc", max_curve_points), "pr_calibrated": _curve(y, pc, w, "pr", max_curve_points),
            "roc_raw": _curve(y, pr, w, "roc", max_curve_points), "pr_raw": _curve(y, pr, w, "pr", max_curve_points),
            "frozen_threshold": thr, "confusion_weighted": conf_w, "confusion_rows": conf_n,
        }
    return out


def _reliability(p: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    order = np.argsort(p, kind="mergesort")
    cum = np.cumsum(w[order]) / np.sum(w)
    edges = np.minimum((cum * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        idx = order[edges == b]
        if len(idx) == 0:
            continue
        ww = w[idx]
        out.append({"bin": b, "n_rows": int(len(idx)), "mean_predicted": float(np.sum(ww * p[idx]) / np.sum(ww)),
                    "observed_rate": float(np.sum(ww * y[idx]) / np.sum(ww)), "weight": float(np.sum(ww))})
    return out


def _curve(y: np.ndarray, s: np.ndarray, w: np.ndarray, kind: str, max_points: int) -> Dict[str, List[float]]:
    order = np.argsort(-s, kind="mergesort")
    ys, ws, ss = y[order], w[order], s[order]
    tp = np.cumsum(ws * (ys == 1)); fp = np.cumsum(ws * (ys == 0))
    last = np.r_[ss[1:] != ss[:-1], True]  # one point per distinct threshold
    tp, fp = tp[last], fp[last]
    P, N = float(np.sum(w * (y == 1))), float(np.sum(w * (y == 0)))
    if kind == "roc":
        xs, ys_ = np.r_[0.0, fp / N], np.r_[0.0, tp / P]
    else:
        xs, ys_ = np.r_[0.0, tp / P], np.r_[1.0, tp / np.maximum(tp + fp, 1e-12)]
    if len(xs) > max_points:
        keep = np.unique(np.linspace(0, len(xs) - 1, max_points).astype(int))
        xs, ys_ = xs[keep], ys_[keep]
    return {"x": [float(v) for v in xs], "y": [float(v) for v in ys_]}


def error_analysis(lines: Sequence[Mapping[str, object]], plan: Mapping[str, object]) -> Dict[str, object]:
    cut = plan["slice_cutpoints"]
    out: Dict[str, object] = {"artifact_version": "performance_v3_error_analysis_v1", "prespecified": True,
                              "slice_cutpoints": cut, "tasks": {}}
    thresholds = {m: _threshold(lines, m) for m in MODELS}
    for task in TASKS:
        families: Dict[str, Dict[str, List[Mapping[str, object]]]] = {}
        for line in lines:
            if _target(task, line["labels"]) is None:
                continue
            for family, label in slice_labels(line, cut, task).items():
                families.setdefault(family, {}).setdefault(label, []).append(line)
        task_out = {}
        for family, groups in sorted(families.items()):
            task_out[family] = {}
            for label, members in sorted(groups.items()):
                ids = {(m["stay_id"], m["prediction_time"]) for m in members}
                entry = {"n_rows": len(members), "n_stays": len({m["stay_id"] for m in members}), "models": {}}
                for model in MODELS:
                    recs = make_task_records(lines, model, task, subset=lambda l, ids=ids: (l["stay_id"], l["prediction_time"]) in ids)
                    v = tm.record_metric_fn(task, PRIMARY_METRIC[task])(recs) if recs else None
                    entry["models"][model] = None if v is None or not math.isfinite(v) else float(v)
                task_out[family][label] = entry
        out["tasks"][task] = task_out
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def make_plots(lines: Sequence[Mapping[str, object]], evaluation: Mapping[str, object], calib: Mapping[str, object], out_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    colors = {"V3": "#1f77b4", "V2_AS_SERVED": "#d62728", "V2_AS_EVALUATED": "#ff7f0e", "CONTROL": "#2ca02c"}

    def save(fig, name):
        path = out_dir / name
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); written.append(path.name)

    for kind, title, xl, yl in (("roc", "ROC (calibrated)", "False positive rate", "True positive rate"),
                                ("pr", "Precision-recall (calibrated)", "Recall", "Precision")):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for m in MODELS:
            c = calib["models"][m][f"{kind}_calibrated"]; ax.plot(c["x"], c["y"], label=m, color=colors[m])
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=7)
        save(fig, f"support_{kind}.png")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for m in MODELS:
        rel = calib["models"][m]["reliability_calibrated"]
        ax.plot([r["mean_predicted"] for r in rel], [r["observed_rate"] for r in rel], marker="o", ms=3, label=m, color=colors[m])
    ax.set_title("Reliability (stay-balanced, equal-mass bins)"); ax.set_xlabel("Mean predicted"); ax.set_ylabel("Observed rate"); ax.legend(fontsize=7)
    save(fig, "support_reliability.png")
    for task in ("recovery24", "recovery48", "icu_stay_time"):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for m in ("V3", PRIMARY_COMPARATOR):
            recs = make_task_records(lines, m, task)
            err = np.asarray([r.prediction - r.target for r in recs])
            ax.hist(err, bins=60, alpha=0.5, label=m, color=colors[m])
        ax.set_title(f"{task}: signed error (prediction - target)"); ax.legend(fontsize=7)
        save(fig, f"{task}_residuals.png")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels, points, lows, highs = [], [], [], []
    for task in TASKS:
        res = evaluation["paired"][f"{task}|{PRIMARY_METRIC[task]}|V3_vs_{PRIMARY_COMPARATOR}"]
        ci = res["ci"].get("0.95") or [np.nan, np.nan]
        labels.append(f"{task}\n({PRIMARY_METRIC[task]})"); points.append(res["diff_point"]); lows.append(ci[0]); highs.append(ci[1])
    y = np.arange(len(labels))
    ax.errorbar(points, y, xerr=[np.asarray(points) - np.asarray(lows), np.asarray(highs) - np.asarray(points)], fmt="o", color="#1f77b4")
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Paired difference V3 - V2_AS_SERVED (95% CI)")
    save(fig, "paired_differences_forest.png")
    return written


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def write_evidence(root: Path, paths: Paths, lines: Sequence[Mapping[str, object]], plan: Mapping[str, object], *,
                   n_bootstrap: Optional[int] = None, extra: Optional[Mapping[str, object]] = None) -> Dict[str, str]:
    evaluation = evaluate_predictions(lines, plan, n_bootstrap=n_bootstrap)
    calib = calibration_evidence(lines)
    errors = error_analysis(lines, plan)
    shas = {}
    shas["final_metrics_v3"] = write_canonical(paths.final_metrics, {
        "artifact_version": "performance_v3_final_metrics_v1", "models": evaluation["models"], "thresholds": evaluation["thresholds"],
        "denominators": evaluation["identities"], "clean_subset": evaluation["clean_subset"], "bootstrap": evaluation["bootstrap"],
        "predictions_sha256": json.loads(paths.predictions_manifest.read_text())["predictions_sha256"], **(extra or {})})
    shas["paired_v2_v3_metrics"] = write_canonical(paths.paired_metrics, {
        "artifact_version": "performance_v3_paired_v2_v3_metrics_v1", "primary_comparator": PRIMARY_COMPARATOR,
        "promotion": evaluation["promotion"],
        "point_differences": {k: {kk: v[kk] for kk in ("diff_point", "point_a", "point_b", "n_rows", "n_stays", "ci", "ci_excludes_zero_95", "ci_excludes_zero_9875")}
                              for k, v in evaluation["paired"].items()},
        "clean_subset": evaluation["clean_subset"]})
    shas["paired_bootstrap"] = write_canonical(paths.paired_bootstrap, {"artifact_version": "performance_v3_paired_bootstrap_v1", **evaluation["bootstrap"],
                                                                         "comparisons": evaluation["paired"]})
    shas["calibration_evidence"] = write_canonical(paths.calibration, calib)
    shas["error_analysis"] = write_canonical(paths.error_analysis, errors)
    plots = make_plots(lines, evaluation, calib, paths.plots)
    shas["plots"] = {name: sha256_file(paths.plots / name) for name in plots}  # type: ignore[assignment]
    return shas


def write_incident(paths: Paths, stage: str, error: BaseException) -> None:
    write_canonical(paths.incident, {
        "artifact_version": "performance_v3_final_test_incident_v1", "stage": stage, "error_type": type(error).__name__,
        "error": str(error)[:2000], "traceback": traceback.format_exc()[-6000:],
        "policy": "access was consumed; no scientific retry; metrics may be recomputed from saved predictions only",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})


@dataclass
class Injectables:
    """Overridable collaborators (tests supply light stand-ins; the CLI uses the real implementations)."""

    rows_loader: Optional[Callable[[Path], Sequence[Mapping[str, object]]]] = None
    sofa_fn: Optional[Callable[[Sequence[Mapping[str, object]]], Tuple[Mapping, Mapping]]] = None
    v3_loader: Optional[Callable[[], object]] = None
    control_loader: Optional[Callable[[], object]] = None
    v2_loader: Optional[Callable[[], object]] = None
    n_bootstrap: Optional[int] = None


def _default_rows_loader(paths: Paths) -> Callable[[Path], Sequence[Mapping[str, object]]]:
    def load(root: Path):
        from data.synthetic.validation import load_jsonl
        from performance_v3.data import _enrich

        statics = {r["stay_id"]: r for r in load_jsonl(paths.cohort_dir / "timeline/canonical_statics.jsonl")}
        rows = []
        for row in load_jsonl(paths.cohort_dir / "pre_split/pre_split_scientific_package.jsonl"):
            enriched = _enrich(row, statics[row["stay_id"]])
            enriched["split"] = "sealed_test"; enriched["cohort_id"] = COHORT_ID
            rows.append(enriched)
        return rows
    return load


def _default_sofa_fn(root: Path, paths: Paths):
    def compute(rows):
        from performance_v3.features import cohort_scoring_context, compute_sofa_trends_and_components

        raw = _rel(root, paths.cohort_dir / "raw/synthetic_dataset_manifest_v1.json")
        timeline = _rel(root, paths.cohort_dir / "timeline/synthetic_processed_manifest_v1.json")
        with cohort_scoring_context(root, raw_manifest_ref=raw, timeline_manifest_ref=timeline) as ctx:
            return compute_sofa_trends_and_components(rows, ctx)
    return compute


def make_access(root: Path, paths: Paths, rows_loader) -> SealedCohortAccess:
    return SealedCohortAccess(root, COHORT_ID, paths.state, rows_loader,
                              expected_paths=(paths.cohort_dir / "pre_split/pre_split_scientific_package.jsonl",
                                              paths.cohort_dir / "timeline/canonical_statics.jsonl"),
                              sealed_paths=(paths.cohort_dir,))


def run_final(root: Path, paths: Paths, inj: Optional[Injectables] = None, *, authorized_by: str = "coordinator") -> Dict[str, object]:
    """THE ONE-TIME OPERATION."""

    inj = inj or Injectables()
    root = Path(root)
    freeze_sha, plan_sha = verify_frozen(root, paths)          # fails BEFORE authorization if anything changed
    from performance_v3.serving import V2Comparator, V3Bundle

    v3 = inj.v3_loader() if inj.v3_loader else V3Bundle.load(root, paths.selected_models)
    control = inj.control_loader() if inj.control_loader else V3Bundle.load(root, paths.control_bundle)
    v2 = inj.v2_loader() if inj.v2_loader else V2Comparator.load(root)
    loader = inj.rows_loader or _default_rows_loader(paths)
    access = make_access(root, paths, loader)
    access.authorize(freeze_sha, plan_sha, authorized_by=authorized_by)
    stage = "consume_and_load"
    try:
        rows = access.consume_and_load_rows()                   # CONSUMED before rows return
        stage = "sofa"
        sofa_fn = inj.sofa_fn or _default_sofa_fn(root, paths)
        trends, components = sofa_fn(rows)
        missing = [r["stay_id"] for r in rows if "t" not in (trends.get((r["stay_id"], r["prediction_time"])) or {})]
        if missing:
            raise RowIdentityError(f"SOFA(t) unavailable for {len(missing)} rows; refusing a null current-SOFA feature")
        stage = "inference"
        outputs = {
            "V3": v3.predict_rows(rows, sofa_trends=trends, sofa_components=components),
            "V2_AS_SERVED": v2.predict_rows(rows, current_sofa_trends=trends),
            "V2_AS_EVALUATED": v2.predict_rows(rows),
            "CONTROL": control.predict_rows(rows, sofa_trends=trends, sofa_components=components),
        }
        stage = "persist"
        lines = build_prediction_lines(rows, outputs)
        eligible_identity_sets(lines)
        predictions_sha = save_predictions(paths, lines)
        access.complete(result_sha256=predictions_sha)
        stage = "evidence"
        plan = json.loads(paths.plan.read_text(encoding="utf-8"))
        shas = write_evidence(root, paths, lines, plan, n_bootstrap=inj.n_bootstrap)
        record = {
            "artifact_version": "performance_v3_final_evaluation_freeze_v1", "status": "V3_FINAL_EVALUATION_FROZEN",
            "test_accessed": True, "access_state": FINAL_RUN_COMPLETED, "successful_scientific_evaluation_count": 1,
            "freeze_sha256": freeze_sha, "plan_sha256": plan_sha, "predictions_sha256": predictions_sha,
            "rows": len(lines), "stays": len({l["stay_id"] for l in lines}), "outputs": shas, "no_post_test_tuning": True,
            "code_commit": _git_head(root), "environment": environment_record(),
        }
        final_sha = write_canonical(paths.final_freeze, record)
        return {"final_freeze_sha256": final_sha, "predictions_sha256": predictions_sha, "rows": len(lines)}
    except Exception as error:
        write_incident(paths, stage, error)
        raise


def recompute_from_predictions(root: Path, paths: Paths, *, n_bootstrap: Optional[int] = None) -> Dict[str, object]:
    lines = load_predictions(paths)
    plan = json.loads(paths.plan.read_text(encoding="utf-8"))
    return {"outputs": write_evidence(Path(root), paths, lines, plan, n_bootstrap=n_bootstrap, extra={"recomputed_from_saved_predictions": True})}


def second_run_guard(root: Path, paths: Paths) -> Dict[str, object]:
    access = make_access(Path(root), paths, lambda r: (_ for _ in ()).throw(AssertionError("rows must never be loaded by the guard")))
    before = sha256_file(paths.predictions) if paths.predictions.exists() else None
    outcome = {"authorize_refused": False, "consume_refused": False}
    try:
        access.authorize("0" * 64, "0" * 64)
    except SealedCohortAccessError:
        outcome["authorize_refused"] = True
    try:
        access.consume_and_load_rows()
    except SealedCohortAccessError as error:
        outcome["consume_refused"] = True
        outcome["consume_error_names_block"] = BLOCKED_ALREADY_CONSUMED in str(error)
    record = {
        "artifact_version": "performance_v3_second_run_guard_v1", "effective_state": access.effective_state(),
        "blocked": access.effective_state() == BLOCKED_ALREADY_CONSUMED and outcome["authorize_refused"] and outcome["consume_refused"],
        "outcomes": outcome, "predictions_sha256_unchanged": (before == sha256_file(paths.predictions)) if before else None,
    }
    write_canonical(paths.second_run, record)
    return record


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def preflight(root: Path, paths: Paths, *, dry_run_rows: Optional[Sequence[Mapping[str, object]]] = None,
              dry_run_sofa: Optional[Tuple[Mapping, Mapping]] = None, inj: Optional[Injectables] = None,
              v2_identity: Optional[Callable[[], Dict[str, object]]] = None) -> Dict[str, object]:
    """Development-only checks; never reads the sealed cohort."""

    import tempfile

    from performance_v3.data import CohortAccessError, CohortDir, load_cohort_rows
    from performance_v3.serving import V2Comparator, V3Bundle
    from performance_v3.test_access import assert_not_sealed

    inj = inj or Injectables()
    root = Path(root)
    checks: Dict[str, Dict[str, object]] = {}

    def run(name, fn):
        try:
            detail = fn()
            checks[name] = {"pass": True, **(detail or {})}
        except Exception as error:
            checks[name] = {"pass": False, "error": f"{type(error).__name__}: {error}"[:500]}

    def frozen():
        f, p = verify_frozen(root, paths)
        return {"freeze_sha256": f, "plan_sha256": p}
    run("frozen_hashes", frozen)

    def ledger():
        access = make_access(root, paths, lambda r: ())
        state = access.state()
        if state != SEALED_NOT_ACCESSED:
            raise AssertionError(f"ledger state {state}, expected {SEALED_NOT_ACCESSED}")
        if not paths.seal.is_file() or json.loads(paths.seal.read_text())["verdict"] != "PASS":
            raise AssertionError("seal record missing or not PASS")
        return {"state": state, "seal_sha256": sha256_file(paths.seal)}
    run("ledger_and_seal", ledger)

    def loaders_refuse():
        refused = []
        try:
            load_cohort_rows(root, CohortDir(paths.cohort_dir), cohort_id=COHORT_ID, role="dev")
        except CohortAccessError:
            refused.append("load_cohort_rows(cohort_id=SYN-V3-T)")
        try:
            load_cohort_rows(root, CohortDir(paths.cohort_dir), cohort_id="SYN-V3-LCE", role="sealed_test")
        except CohortAccessError:
            refused.append("load_cohort_rows(role=sealed_test)")
        make_access(root, paths, lambda r: ())  # registers the sealed path
        try:
            assert_not_sealed(str(paths.cohort_dir / "pre_split"))
        except SealedCohortAccessError:
            refused.append("assert_not_sealed(path)")
        if len(refused) != 3:
            raise AssertionError("development loaders did not refuse the sealed cohort: " + repr(refused))
        return {"refused": refused}
    run("development_loaders_refuse_sealed", loaders_refuse)

    def guard_dry_run():
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "ledger.json"
            acc = SealedCohortAccess(tmp, "DRYRUN-COHORT", state, lambda r: [{"x": 1}], expected_paths=(), sealed_paths=())
            acc.initialize(); acc.authorize("a" * 64, "b" * 64); acc.consume_and_load_rows(); acc.complete()
            for call in (lambda: acc.consume_and_load_rows(), lambda: acc.authorize("a" * 64, "b" * 64)):
                try:
                    call()
                except SealedCohortAccessError:
                    continue
                raise AssertionError("second access was not blocked")
            return {"effective_state": acc.effective_state()}
    run("second_run_guard_dry_run", guard_dry_run)

    def dry_run():
        if dry_run_rows is None:
            raise AssertionError("no development dry-run rows supplied")
        v3 = inj.v3_loader() if inj.v3_loader else V3Bundle.load(root, paths.selected_models if paths.selected_models.exists() else paths.v3_bundle)
        control = inj.control_loader() if inj.control_loader else V3Bundle.load(root, paths.control_bundle)
        v2 = inj.v2_loader() if inj.v2_loader else V2Comparator.load(root)
        trends, components = dry_run_sofa if dry_run_sofa is not None else _default_sofa_fn(root, paths)(dry_run_rows)
        rows = list(dry_run_rows)
        outputs = {"V3": v3.predict_rows(rows, sofa_trends=trends, sofa_components=components),
                   "V2_AS_SERVED": v2.predict_rows(rows, current_sofa_trends=trends), "V2_AS_EVALUATED": v2.predict_rows(rows),
                   "CONTROL": control.predict_rows(rows, sofa_trends=trends, sofa_components=components)}
        lines = build_prediction_lines(rows, outputs)
        eligible_identity_sets(lines)
        plan = json.loads(paths.plan.read_text(encoding="utf-8"))
        ev = evaluate_predictions(lines, plan, n_bootstrap=20)
        finite = all(np.isfinite(float(p[k])) for m in outputs.values() for p in m for k in ("recovery24_delta", "recovery48_delta", "icu_remaining_hours", "support_calibrated_probability"))
        if not finite:
            raise AssertionError("non-finite dry-run predictions")
        return {"rows": len(rows), "promotion_keys": sorted(ev["promotion"])}
    run("development_dry_run_full_pipeline", dry_run)

    if v2_identity is not None:
        run("v2_comparator_identity", v2_identity)
    verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    record = {"artifact_version": "performance_v3_pretest_preflight_v1", "verdict": verdict, "checks": checks,
              "sealed_cohort_rows_opened": False, "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_canonical(paths.preflight, record)
    return record
