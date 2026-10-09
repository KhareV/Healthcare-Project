#!/usr/bin/env python3
"""V3 data audits: identity/trajectory overlap, generator equivalence, structural invariants.

Read-only on every historical artifact; writes JSON under
artifacts/performance_v3/data_characterization/. No label distribution is read for any
sealed cohort (none exists; the SYN-V3-T cohort is intentionally not part of this audit).
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes, load_runtime_config  # noqa: E402
from data.synthetic.validation import load_jsonl  # noqa: E402
from performance_v3.audit import overlap_report, structural_checks  # noqa: E402
from performance_v3.protocol import cohort_identity, protocol_sha256, sha256_file  # noqa: E402
from preprocess.synthetic_phase10 import exact_raw_trajectory_fingerprints  # noqa: E402

OUT = ROOT / "artifacts/performance_v3/data_characterization"
V3 = ("D1", "D2", "D3", "LCE")
BASE_COMMIT = "2218cbd"
V1_RAW = ROOT / "artifacts/data/synthetic/final/phase9_final_v1"
V2_RAW = ROOT / "artifacts/performance_v2/phase3/fresh_test_cohort/raw"
V1_SPLIT = ROOT / "artifacts/splits/synthetic_split_v2.csv"
DEV_MANIFEST = ROOT / "artifacts/performance_v2/phase3/dev_subject_manifest_v1.json"
INVENTORY = ROOT / "artifacts/performance_v3/governance/pre_phase_hash_inventory_v1.sha256"

SCIENTIFIC_CODE_FILES = [
    "src/data/synthetic/latent_state.py", "src/data/synthetic/observations.py", "src/data/synthetic/support.py",
    "src/data/synthetic/records.py", "src/data/synthetic/rng.py", "src/data/synthetic/validation.py",
    "src/data/synthetic/sofa.py", "src/data/synthetic/sofa_provider.py", "src/labels/recovery.py",
    "src/labels/icu_time.py", "src/labels/organ_support.py", "src/labels/support_state.py",
    "src/labels/ventilation_state.py", "src/features/synthetic.py",
    "src/data/synthetic/generator.py", "src/labels/phase9_final.py",
]
SCIENTIFIC_CONFIG_KEYS = (
    "population", "calendar", "duration", "latent", "observation", "variables", "latent_process_path",
    "support_process_path", "event_dictionary_path", "minimum_accepted_episode_hours",
)
ALLOWED_TOP_LEVEL_DIFFS = {
    "primary_seed", "n_subjects", "runtime_config_version", "subject_id_namespace", "accepted_subject_policy",
    "purpose", "not_a_new_development_split", "seed_derivation", "scientific_equivalence_note", "inherits",
}
PROHIBITED_TOKENS = ("subject_id", "stay_id", "prediction_time", "grid_index", "split", "eligible", "target",
                     "delta_sofa", "future_sofa", "outtime", "remaining_hours", "organ_support_label", "censor", "latent")


def _write(name: str, value) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT), sha256_file(path)[:12])
    return path


def _raw(directory: Path):
    return (load_jsonl(directory / "subjects.jsonl"), load_jsonl(directory / "episodes.jsonl"),
            load_jsonl(directory / "raw_events.jsonl"), load_jsonl(directory / "support_intervals.jsonl"))


def _cohort_entry(subjects, episodes, events, supports, subject_filter=None):
    if subject_filter is not None:
        keep = set(subject_filter)
        subjects = [r for r in subjects if r["subject_id"] in keep]
        episodes = [r for r in episodes if r["subject_id"] in keep]
        events = [r for r in events if r["subject_id"] in keep]
        supports = [r for r in supports if r["subject_id"] in keep]
    return {
        "subject_ids": {r["subject_id"] for r in subjects},
        "stay_ids": {r["stay_id"] for r in episodes},
        "fingerprints": dict(exact_raw_trajectory_fingerprints(subjects, episodes, events, supports)),
    }


def overlap_audit() -> dict:
    with V1_SPLIT.open(newline="", encoding="utf-8") as handle:
        split = {r["subject_id"]: r["split"] for r in csv.DictReader(handle)}
    dev_manifest = json.loads(DEV_MANIFEST.read_text())
    dev_ids = set(dev_manifest["dev_subjects"])
    assert dev_ids == {s for s, v in split.items() if v in ("train", "validation")}
    entries = {}
    s, e, ev, sp = _raw(V1_RAW)
    for name, members in (("V1_TRAIN", {k for k, v in split.items() if v == "train"}),
                          ("V1_VALIDATION", {k for k, v in split.items() if v == "validation"}),
                          ("V1_TEST", {k for k, v in split.items() if v == "test"})):
        entries[name] = _cohort_entry(s, e, ev, sp, members)
    del s, e, ev, sp
    entries["V2_FRESH_TEST"] = _cohort_entry(*_raw(V2_RAW))
    for cid in V3:
        entries[cid] = _cohort_entry(*_raw(ROOT / "artifacts/performance_v3/cohorts" / cid / "raw"))
    report = overlap_report(entries, test_cohorts=("V1_TEST", "V2_FRESH_TEST"), v3_cohorts=V3)
    report.update({
        "artifact_version": "performance_v3_overlap_audit_v1",
        "v1_dev_manifest_subjects": len(dev_ids),
        "v1_dev_equals_train_plus_validation": True,
        "v1_total_subjects": len(split),
        "sealed_v3_test_cohort_included": False,
        "verdict": "PASS" if not report["blockers"] and report["all_cross_cohort_subject_and_stay_intersections_zero"]
        and report["clone_group_count"] == 0 else "REVIEW_REQUIRED",
    })
    return report


def _inventory() -> dict:
    out = {}
    for line in INVENTORY.read_text().splitlines():
        digest, path = line.split("  ", 1)
        out[path] = digest
    return out


def equivalence_audit() -> dict:
    v1 = load_runtime_config(ROOT / "configs/synthetic/final_benchmark_v1.json", ROOT).values
    v2_file = json.loads((ROOT / "configs/performance_v2/fresh_test_cohort_v1.json").read_text())
    base = json.loads((ROOT / "configs/synthetic/final_benchmark_v1.json").read_text())
    cohorts = {}
    all_ok = True
    for cid in V3:
        path = ROOT / f"configs/performance_v3/cohorts/{cid.lower()}_v1.json"
        values = load_runtime_config(path, ROOT).values
        raw = json.loads(path.read_text())
        sci_diffs = {k: 1 for k in SCIENTIFIC_CONFIG_KEYS if v1.get(k) != values.get(k)}
        other = sorted(k for k in set(v1) | set(values) if v1.get(k) != values.get(k) and k not in ALLOWED_TOP_LEVEL_DIFFS)
        # raw-file keys that differ from the V2 fresh config beyond the allowed set
        raw_other = sorted(k for k in set(raw) | set(v2_file) if raw.get(k) != v2_file.get(k) and k not in ALLOWED_TOP_LEVEL_DIFFS)
        ident = cohort_identity(ROOT, cid)
        ok = (not sci_diffs and not other and not raw_other and values["primary_seed"] == ident["seed"]
              and values["n_subjects"] == ident["n_subjects"] and values["subject_id_namespace"] == ident["namespace"]
              and values["mode"] == "final")
        all_ok &= ok
        cohorts[cid] = {"config_ref": str(path.relative_to(ROOT)), "config_sha256": sha256_file(path),
                        "seed": values["primary_seed"], "n_subjects": values["n_subjects"],
                        "namespace": values["subject_id_namespace"],
                        "scientific_key_diffs_vs_final_benchmark_v1": sorted(sci_diffs),
                        "non_allowed_top_level_diffs_vs_final_benchmark_v1": other,
                        "non_allowed_diffs_vs_v2_fresh_config": raw_other,
                        "seed_matches_protocol_derivation": values["primary_seed"] == ident["seed"], "pass": ok}
    inv = _inventory()
    code = {}
    for path in SCIENTIFIC_CODE_FILES:
        now = sha256_file(ROOT / path)
        code[path] = {"sha256": now, "equals_pre_phase_inventory": inv.get(path) == now}
    code_ok = all(v["equals_pre_phase_inventory"] for v in code.values())
    diff = subprocess.run(["git", "diff", BASE_COMMIT, "--", "src/data/synthetic/config.py"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    ident_lines = [l for l in added if re.match(r'^\s*\(\d+, \d+\): "SYN-V3-[A-Z0-9]+",\s*$', l)]
    other_added = [l for l in added if l not in ident_lines and not l.strip().startswith("#")]
    config_ok = (not removed and len(ident_lines) == 4 and not other_added)
    src_diff = subprocess.run(["git", "diff", BASE_COMMIT, "--name-only", "--", "src", "scripts", "api", "configs"],
                              cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    unexpected = sorted(p for p in src_diff if not p.startswith(("src/performance_v3/", "scripts/performance_v3_",
                                                                "configs/performance_v3/")) and p != "src/data/synthetic/config.py")
    repo_ok = not unexpected
    return {
        "artifact_version": "performance_v3_generator_equivalence_audit_v1",
        "protocol_sha256": protocol_sha256(ROOT),
        "cohort_configs": cohorts,
        "scientific_code_hashes": code,
        "scientific_code_unchanged_vs_pre_phase_inventory": code_ok,
        "config_py_diff": {"removed_lines": removed, "added_identity_lines": ident_lines,
                           "other_added_non_comment_lines": other_added,
                           "only_four_final_identities_added": config_ok},
        "tracked_changes_outside_v3_namespace_vs_base_commit": unexpected,
        "generator_modified": False if (code_ok and config_ok and repo_ok) else True,
        "verdict": "PASS" if all_ok and code_ok and config_ok and repo_ok else "FAIL",
    }


def structural_audit() -> dict:
    schema = json.loads((ROOT / "configs/synthetic/feature_schema_v2.json").read_text())
    names = [c["name"] for c in schema["temporal_channels"]]
    out = {}
    all_pass = True
    for cid in V3:
        base = ROOT / "artifacts/performance_v3/cohorts" / cid
        episodes = {r["stay_id"]: r for r in load_jsonl(base / "raw/episodes.jsonl")}
        rows = list(load_jsonl(base / "pre_split/pre_split_scientific_package.jsonl"))
        checks = structural_checks(rows, episodes, names, PROHIBITED_TOKENS)
        retained = list(load_jsonl(base / "cohort/retained_cohort.jsonl"))
        generated = sum(1 for _ in load_jsonl(base / "raw/subjects.jsonl"))
        checks["identity_one_episode_per_subject_all_retained"] = {
            "pass": len(episodes) == generated == len(retained), "generated": generated, "retained": len(retained),
            "stays_with_rows": len({r["stay_id"] for r in rows})}
        pass_ = all(c["pass"] for c in checks.values())
        all_pass &= pass_
        out[cid] = {"rows": len(rows), "checks": checks, "pass": pass_}
    return {"artifact_version": "performance_v3_structural_audit_v1", "reuses": "src/acceptance/phase13.py invariants D01,E01,F01,H01,I01,J01,K01,L01",
            "cohorts": out, "verdict": "PASS" if all_pass else "FAIL", "sealed_cohort_labels_read": False}


def main() -> None:
    which = sys.argv[1:] or ["overlap", "equivalence", "structural"]
    if "overlap" in which:
        _write("overlap_audit_v1.json", overlap_audit())
    if "equivalence" in which:
        _write("generator_equivalence_audit_v1.json", equivalence_audit())
    if "structural" in which:
        _write("structural_audit_v1.json", structural_audit())


if __name__ == "__main__":
    main()
