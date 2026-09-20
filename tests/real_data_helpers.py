import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.split import (
    CohortSubjectRecord,
    generate_subject_split,
    write_split_artifacts,
)
from data.timestamps import RetainedICUStay, generate_prediction_timestamps
from vedant_infra.hashing import sha256_file


FIELD_MAP = {
    "subject_id": "sid",
    "stay_id": "icu",
    "prediction_time": "cutoff",
    "grid_index": "k",
    "intime": "icu_in",
    "outtime": "icu_out",
    "anchor_year_group": "era",
    "split": "partition",
    "history_values": "sequence",
    "padding_mask": "padding",
    "observation_mask": "observed",
    "tslo_hours": "tslo",
    "static_features": "static",
    "recovery24_eligible": "rec24_ok",
    "recovery48_eligible": "rec48_ok",
    "icu_time_eligible": "icu_ok",
    "recovery24_delta_sofa": "delta24",
    "recovery48_delta_sofa": "delta48",
    "sofa_t": "sofa_now",
    "sofa_t24": "sofa_24",
    "sofa_t48": "sofa_48",
    "sofa_component_observed_flags": "sofa_observed",
    "icu_remaining_hours": "remaining_h",
    "icu_time_log1p_hours": "remaining_log",
}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def refresh_manifest_hash(root, name):
    manifest_path = root / "artifacts/data/real_data_inputs_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ref = manifest["artifacts"][name]["ref"]
    manifest["artifacts"][name]["sha256"] = sha256_file(root / ref)
    write_json(manifest_path, manifest)


def _artifact(ref, digest, owner, kind, version, schema):
    return {
        "ref": ref,
        "sha256": digest,
        "owner": owner,
        "artifact_type": kind,
        "artifact_version": version,
        "schema_version": schema,
        "creation_commit": "1" * 40,
        "consumer_contract": "phase18_real_data_acceptance_v1",
    }


def build_real_handoff_fixture(root: Path, *, include_support=True):
    artifacts = root / "artifacts/data"
    split_path = root / "artifacts/splits/split_v1.csv"
    split_metadata = root / "artifacts/splits/split_v1.metadata.json"
    cohort = (
        CohortSubjectRecord("TRAIN-A", "2008-2010", "STAY-A"),
        CohortSubjectRecord("VAL-B", "2014-2016", "STAY-B"),
        CohortSubjectRecord("TEST-C", "2017-2019", "STAY-C"),
    )
    split_result = generate_subject_split(cohort)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    write_split_artifacts(
        split_result,
        split_path,
        split_metadata,
        source_cohort_version="synthetic_real_handoff_fixture_v1",
        source_cohort_sha256="a" * 64,
        split_spec_sha256="b" * 64,
        created_at_utc="2026-09-18T00:00:00Z",
        code_commit="1" * 40,
    )
    split_hash = sha256_file(split_path)
    subjects = (
        ("TRAIN-A", "STAY-A", "2008-2010", "train"),
        ("VAL-B", "STAY-B", "2014-2016", "validation"),
        ("TEST-C", "STAY-C", "2017-2019", "test"),
    )
    structural_rows = []
    canonical_rows = []
    support_rows = []
    for subject, stay, era, split in subjects:
        intime = datetime(2020, 1, 1, tzinfo=timezone.utc)
        outtime = intime + timedelta(hours=78)
        predictions = generate_prediction_timestamps(
            (RetainedICUStay(subject, stay, intime, outtime),)
        )
        for prediction in predictions:
            common = {
                "sid": subject,
                "icu": stay,
                "cutoff": prediction.prediction_time.isoformat(),
                "k": prediction.grid_index,
                "icu_in": intime.isoformat(),
                "icu_out": outtime.isoformat(),
                "era": era,
                "partition": split,
            }
            structural_rows.append(common)
            if split == "test":
                continue
            padding_count = max(0, 4 - prediction.grid_index)
            sequence = []
            observed = []
            tslo = []
            last_feature0 = None
            for bin_index in range(8):
                if bin_index < padding_count:
                    sequence.append([None, None])
                    observed.append([False, False])
                    tslo.append([-1.0, -1.0])
                elif last_feature0 is None:
                    sequence.append([float(prediction.grid_index + 1), None])
                    observed.append([True, False])
                    tslo.append([1.0, -1.0])
                    last_feature0 = 1.0
                else:
                    sequence.append([None, None])
                    observed.append([False, False])
                    last_feature0 += 6.0
                    tslo.append([last_feature0, -1.0])
            hours = (outtime - prediction.prediction_time).total_seconds() / 3600
            rec24 = prediction.prediction_time + timedelta(hours=24) <= outtime
            rec48 = prediction.prediction_time + timedelta(hours=48) <= outtime
            sofa_now = float(5 + prediction.grid_index)
            canonical_rows.append(
                {
                    **common,
                    "sequence": sequence,
                    "padding": [index < padding_count for index in range(8)],
                    "observed": observed,
                    "tslo": tslo,
                    "static": [60.0],
                    "rec24_ok": rec24,
                    "rec48_ok": rec48,
                    "icu_ok": True,
                    "delta24": 1.0 if rec24 else None,
                    "delta48": 2.0 if rec48 else None,
                    "sofa_now": sofa_now,
                    "sofa_24": sofa_now + 1.0 if rec24 else None,
                    "sofa_48": sofa_now + 2.0 if rec48 else None,
                    "sofa_observed": [True, True, True, True, True, True],
                    "remaining_h": hours,
                    "remaining_log": math.log1p(hours),
                }
            )
            if prediction.grid_index == 0:
                support = {
                    **common,
                    "organ_support_eligible": True,
                    "organ_support_target": 1,
                    "vasopressor_on_at_t": False,
                    "ventilation_on_at_t": False,
                    "vasopressor_onset_time": (
                        prediction.prediction_time + timedelta(hours=3)
                    ).isoformat(),
                    "ventilation_onset_time": None,
                    "censor_reason": None,
                }
            elif prediction.grid_index == 1:
                support = {
                    **common,
                    "organ_support_eligible": False,
                    "organ_support_target": None,
                    "vasopressor_on_at_t": True,
                    "ventilation_on_at_t": True,
                    "vasopressor_onset_time": None,
                    "ventilation_onset_time": None,
                    "censor_reason": "already_on_both",
                }
            else:
                full = prediction.prediction_time + timedelta(hours=24) <= outtime
                support = {
                    **common,
                    "organ_support_eligible": full,
                    "organ_support_target": 0 if full else None,
                    "vasopressor_on_at_t": False,
                    "ventilation_on_at_t": False,
                    "vasopressor_onset_time": None,
                    "ventilation_onset_time": None,
                    "censor_reason": None if full else "icu_exit_before_24h",
                }
            support_rows.append(support)

    canonical_path = artifacts / "feature_dataset_v1.json"
    canonical_hash = write_json(
        canonical_path,
        {
            "storage_format": "canonical_json_v1",
            "dataset_version": "feature_dataset_v1",
            "schema_version": "processed_schema_v1",
            "feature_schema_version": "feature_schema_v1",
            "label_spec_version": "label_spec_v1",
            "tensor_contract_version": "tensor_contract_v1",
            "timestamp_spec_version": "timestamp_spec_v1",
            "dynamic_feature_names": ["dynamic_a", "dynamic_b"],
            "static_feature_names": ["age_years"],
            "dtypes": {
                "history_values": "float32",
                "observation_mask": "bool",
                "padding_mask": "bool",
                "tslo_hours": "float32",
                "static_features": "float32",
                "eligibility": "bool",
                "timestamps": "iso8601",
            },
            "rows": canonical_rows,
        },
    )
    structural_path = artifacts / "canonical_structural_index_v1.json"
    structural_hash = write_json(
        structural_path,
        {"index_version": "canonical_structural_index_v1", "rows": structural_rows},
    )
    feature_path = artifacts / "feature_schema_v1.json"
    feature_hash = write_json(
        feature_path,
        {
            "feature_schema_version": "feature_schema_v1",
            "dynamic_feature_names": ["dynamic_a", "dynamic_b"],
            "static_feature_names": ["age_years"],
            "tslo_no_observation_value": -1.0,
            "valid_ranges": {"dynamic_a": [-100.0, 100.0], "dynamic_b": [-100.0, 100.0]},
        },
    )
    dictionary_path = artifacts / "feature_provenance_v1.json"
    dictionary_hash = write_json(
        dictionary_path,
        {
            "dictionary_version": "feature_provenance_v1",
            "features": ["dynamic_a", "dynamic_b", "age_years"],
            "all_model_fields_documented": True,
        },
    )
    sofa_path = artifacts / "sofa_provenance_v1.json"
    sofa_hash = write_json(
        sofa_path,
        {
            "provenance_version": "sofa_provenance_v1",
            "same_function_all_cutoffs": True,
            "mimic_code_commit": "2" * 40,
            "six_component_flags_present": True,
        },
    )
    label_path = artifacts / "label_spec_v1.json"
    label_hash = write_json(
        label_path,
        {
            "label_spec_version": "label_spec_v1",
            "recovery24_formula": "SOFA(t+24)-SOFA(t)",
            "recovery48_formula": "SOFA(t+48)-SOFA(t)",
            "icu_time_formula": "log1p((outtime-t)_hours)",
        },
    )
    tensor_path = artifacts / "tensor_contract_v1.json"
    tensor_hash = write_json(tensor_path, {"contract_version": "tensor_contract_v1"})
    timestamp_path = artifacts / "timestamp_spec_v1.yaml"
    timestamp_path.write_text("spec_version: timestamp_spec_v1\n", encoding="utf-8")
    timestamp_hash = sha256_file(timestamp_path)
    sidecar_path = artifacts / "processed_schema_v1.json"
    sidecar_hash = write_json(
        sidecar_path,
        {
            "schema_version": "processed_schema_v1",
            "field_map": FIELD_MAP,
            "dtypes": {
                "history_values": "float32",
                "observation_mask": "bool",
                "padding_mask": "bool",
                "tslo_hours": "float32",
                "static_features": "float32",
                "eligibility": "bool",
                "timestamps": "iso8601",
            },
            "split_sha256": split_hash,
            "preprocessing": {
                "status": "NOT_FIT_PHASE19_INTERFACE_READY",
                "fit_partition": "train",
                "validation_or_test_statistics_used": False,
            },
            "leakage_evidence": {
                "no_post_outtime_events": True,
                "no_post_cutoff_features": True,
                "no_prelookback_forward_fill": True,
                "no_future_duration_or_discharge_features": True,
            },
        },
    )
    records = {
        "canonical_dataset": _artifact("artifacts/data/feature_dataset_v1.json", canonical_hash, "Sanskruti", "canonical_dataset", "feature_dataset_v1", "processed_schema_v1"),
        "schema_sidecar": _artifact("artifacts/data/processed_schema_v1.json", sidecar_hash, "Sanskruti", "schema_sidecar", "processed_schema_v1", "processed_schema_v1"),
        "feature_schema": _artifact("artifacts/data/feature_schema_v1.json", feature_hash, "Sanskruti", "feature_schema", "feature_schema_v1", "feature_schema_v1"),
        "feature_dictionary": _artifact("artifacts/data/feature_provenance_v1.json", dictionary_hash, "Sanskruti", "feature_dictionary", "feature_provenance_v1", "feature_schema_v1"),
        "structural_index": _artifact("artifacts/data/canonical_structural_index_v1.json", structural_hash, "Sanskruti", "structural_index", "canonical_structural_index_v1", "processed_schema_v1"),
        "split": _artifact("artifacts/splits/split_v1.csv", split_hash, "Vedant", "subject_split", "split_v1", "split_spec_v1"),
        "split_metadata": _artifact("artifacts/splits/split_v1.metadata.json", sha256_file(split_metadata), "Vedant", "split_metadata", "split_v1", "split_spec_v1"),
        "sofa_provenance": _artifact("artifacts/data/sofa_provenance_v1.json", sofa_hash, "Sanskruti", "sofa_provenance", "sofa_provenance_v1", "sofa_spec_v1"),
        "label_spec": _artifact("artifacts/data/label_spec_v1.json", label_hash, "Sanskruti", "label_spec", "label_spec_v1", "label_spec_v1"),
        "tensor_contract": _artifact("artifacts/data/tensor_contract_v1.json", tensor_hash, "Vedant", "tensor_contract", "tensor_contract_v1", "tensor_contract_v1"),
        "timestamp_spec": _artifact("artifacts/data/timestamp_spec_v1.yaml", timestamp_hash, "Vedant", "timestamp_contract", "timestamp_spec_v1", "timestamp_spec_v1"),
        "protected_test_labels": {
            **_artifact("artifacts/data/protected/test_labels_v1.parquet", "f" * 64, "Sanskruti", "protected_test_labels", "labels_v1", "label_spec_v1"),
            "access_policy": "PROTECTED_NOT_OPENED_PHASE18",
        },
    }
    if include_support:
        support_path = artifacts / "organ_support_handoff_v1.json"
        support_hash = write_json(
            support_path,
            {"handoff_version": "organ_support_handoff_v1", "rows": support_rows},
        )
        records["support_handoff"] = _artifact("artifacts/data/organ_support_handoff_v1.json", support_hash, "Pulkit", "support_handoff", "organ_support_handoff_v1", "event_dict_v1")
    manifest_path = artifacts / "real_data_inputs_v1.json"
    write_json(
        manifest_path,
        {"manifest_version": "real_data_input_manifest_v1", "artifacts": records},
    )
    return manifest_path
