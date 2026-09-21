"""Synthetic Phase-10 split and train-only preprocessing implementation.

The split is label-blind and deterministic.  Every learned statistic is fitted
only after subject assignment and only from training-subject rows.  Validation
and sealed-test labels are never summarized by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Iterable, Mapping, Sequence

import yaml

from data.synthetic.config import canonical_json_bytes
from data.synthetic.provenance import sha256_file
from data.synthetic.validation import load_jsonl


SPLITS = ("train", "validation", "test")


class Phase10ContractError(ValueError):
    """Raised when Phase-10 input or configuration violates the contract."""


@dataclass(frozen=True)
class Phase10Config:
    version: str
    seed: int
    namespace: str
    fractions: Mapping[str, float]
    unknown_token: str
    tslo_sentinel: float
    padding_output_value: float


def load_phase10_config(path: Path) -> Phase10Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_SYNTHETIC_AUTHORIZED":
        raise Phase10ContractError("Phase-10 contract is not frozen")
    split = payload["split"]
    preprocessing = payload["feature_preprocessing"]
    fractions = split["fractions"]
    if tuple(fractions) != SPLITS or not math.isclose(sum(fractions.values()), 1.0):
        raise Phase10ContractError("split fractions must be ordered train/validation/test and sum to one")
    if split.get("stratification") != "NONE" or split.get("labels_or_performance_used") is not False:
        raise Phase10ContractError("synthetic split must remain label-blind and unstratified")
    if split.get("fake_calendar_fields_allowed") is not False:
        raise Phase10ContractError("fake synthetic calendar split fields are prohibited")
    if preprocessing.get("validation_or_test_statistics_used") is not False:
        raise Phase10ContractError("preprocessing must be training-only")
    return Phase10Config(
        version=str(payload["contract_version"]),
        seed=int(split["seed"]),
        namespace=str(split["hash_namespace"]),
        fractions={name: float(fractions[name]) for name in SPLITS},
        unknown_token=str(preprocessing["unknown_category_token"]),
        tslo_sentinel=float(preprocessing["tslo_no_observation_sentinel"]),
        padding_output_value=float(preprocessing["padding_output_value"]),
    )


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def exact_raw_trajectory_fingerprints(
    subjects: Sequence[Mapping[str, object]],
    episodes: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    supports: Sequence[Mapping[str, object]],
) -> Mapping[str, str]:
    """Hash identity-free raw trajectories, including support and duration."""
    subject_rows = {str(row["subject_id"]): row for row in subjects}
    episode_rows = {str(row["subject_id"]): row for row in episodes}
    if set(subject_rows) != set(episode_rows):
        raise Phase10ContractError("raw subject and episode identities differ")
    stay_to_subject = {str(row["stay_id"]): str(row["subject_id"]) for row in episodes}
    hashers = {}
    starts = {}
    for subject_id in sorted(subject_rows):
        subject = subject_rows[subject_id]
        episode = episode_rows[subject_id]
        start = _parse_time(str(episode["intime"]))
        starts[subject_id] = start
        duration = (_parse_time(str(episode["outtime"])) - start).total_seconds()
        initial = [subject["age_years"], subject["sex_category"], subject["cardiac_condition_group"], duration]
        digest = hashlib.sha256()
        digest.update(canonical_json_bytes(initial))
        hashers[subject_id] = digest
    for row in events:
        subject_id = stay_to_subject.get(str(row["stay_id"]))
        if subject_id is None or str(row["subject_id"]) != subject_id:
            raise Phase10ContractError("raw event identity mismatch")
        start = starts[subject_id]
        normalized = [
            "EVENT", row["concept_code"],
            (_parse_time(str(row["event_time"])) - start).total_seconds(),
            row["value_numeric"], row["value_text"], row["unit"],
            None if row["interval_start"] is None else (_parse_time(str(row["interval_start"])) - start).total_seconds(),
            None if row["interval_end"] is None else (_parse_time(str(row["interval_end"])) - start).total_seconds(),
        ]
        hashers[subject_id].update(canonical_json_bytes(normalized))
    for row in supports:
        subject_id = stay_to_subject.get(str(row["stay_id"]))
        if subject_id is None or str(row["subject_id"]) != subject_id:
            raise Phase10ContractError("support interval identity mismatch")
        start = starts[subject_id]
        normalized = [
            "SUPPORT", row["support_type"], row["support_category"], row["agent_key"],
            row["respiratory_category"], row["rate_value"], row["rate_unit"],
            (_parse_time(str(row["interval_start"])) - start).total_seconds(),
            (_parse_time(str(row["interval_end"])) - start).total_seconds(),
        ]
        hashers[subject_id].update(canonical_json_bytes(normalized))
    return {subject_id: digest.hexdigest() for subject_id, digest in hashers.items()}


def _largest_remainder_counts(total: int, fractions: Mapping[str, float]) -> Mapping[str, int]:
    raw = {name: total * fractions[name] for name in SPLITS}
    counts = {name: int(math.floor(raw[name])) for name in SPLITS}
    remainder = total - sum(counts.values())
    ranked = sorted(SPLITS, key=lambda name: (-(raw[name] - counts[name]), SPLITS.index(name)))
    for name in ranked[:remainder]:
        counts[name] += 1
    return counts


def assign_subjects(
    subject_ids: Iterable[str],
    clone_fingerprints: Mapping[str, str],
    config: Phase10Config,
) -> tuple[Mapping[str, str], Mapping[str, int], int]:
    subjects = tuple(sorted(set(str(value) for value in subject_ids)))
    if not subjects or set(subjects) != set(clone_fingerprints):
        raise Phase10ContractError("clone inventory must exactly cover split subjects")
    groups = defaultdict(list)
    for subject_id in subjects:
        groups[clone_fingerprints[subject_id]].append(subject_id)
    ranked_groups = sorted(
        groups.values(),
        key=lambda group: hashlib.sha256(
            (config.namespace + "\0" + str(config.seed) + "\0" + "\0".join(group)).encode("utf-8")
        ).hexdigest(),
    )
    targets = _largest_remainder_counts(len(subjects), config.fractions)
    assignments = {}
    counts = Counter()
    for group in ranked_groups:
        candidates = [name for name in SPLITS if counts[name] < targets[name]] or list(SPLITS)
        split = min(
            candidates,
            key=lambda name: (
                abs((counts[name] + len(group)) - targets[name]),
                -(targets[name] - counts[name]),
                SPLITS.index(name),
            ),
        )
        for subject_id in group:
            assignments[subject_id] = split
        counts[split] += len(group)
    for members in groups.values():
        if len({assignments[subject_id] for subject_id in members}) != 1:
            raise Phase10ContractError("an exact clone group crossed split boundaries")
    return assignments, {name: counts[name] for name in SPLITS}, sum(len(v) - 1 for v in groups.values())


def _population(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise Phase10ContractError("cannot fit a statistic from zero training values")
    mean = sum(values) / len(values)
    scale = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, 1.0 if scale == 0.0 else scale


def fit_preprocessing(
    rows: Sequence[Mapping[str, object]],
    assignments: Mapping[str, str],
    config: Phase10Config,
    *,
    feature_schema_sha256: str,
    split_sha256: str,
    fit_subjects_sha256: str,
    model_input_contract_sha256: str = "0" * 64,
    event_dictionary_sha256: str = "0" * 64,
) -> Mapping[str, object]:
    training = [row for row in rows if assignments[str(row["subject_id"])] == "train"]
    if not training:
        raise Phase10ContractError("training partition has no rows")
    names = tuple(training[0]["temporal_feature_names"])
    static_names = tuple(training[0]["static_feature_names"])
    observed = [[] for _ in names]
    active_cells = [[] for _ in names]
    age_values = []
    category_values = {name: set() for name in static_names[1:]}
    train_subjects = set()
    for row in training:
        if tuple(row["temporal_feature_names"]) != names or tuple(row["static_feature_names"]) != static_names:
            raise Phase10ContractError("Phase-9 feature order changed within training rows")
        if row.get("split") is not None or row.get("preprocessor") is not None:
            raise Phase10ContractError("Phase-9 input is not genuinely pre-split")
        train_subjects.add(str(row["subject_id"]))
        for bin_index, padded in enumerate(row["padding_mask"]):
            if padded:
                continue
            for feature_index, value in enumerate(row["history_values"][bin_index]):
                if value is not None:
                    observed[feature_index].append(float(value))
        statics = row["static_features"]
        age_values.append(float(statics[0]))
        for index, name in enumerate(static_names[1:], start=1):
            category_values[name].add(str(statics[index]))
    medians = [float(statistics.median(values)) if values else 0.0 for values in observed]
    if any(not values for values in observed):
        missing = [names[index] for index, values in enumerate(observed) if not values]
        raise Phase10ContractError("training has no genuine observations for: " + ", ".join(missing))
    for row in training:
        for bin_index, padded in enumerate(row["padding_mask"]):
            if padded:
                continue
            for feature_index, value in enumerate(row["history_values"][bin_index]):
                active_cells[feature_index].append(medians[feature_index] if value is None else float(value))
    dynamic_means_scales = [_population(values) for values in active_cells]
    age_mean, age_scale = _population(age_values)
    categories = {
        name: sorted(values) + ([config.unknown_token] if config.unknown_token not in values else [])
        for name, values in category_values.items()
    }
    recovery24 = [float(row["delta_sofa_24"]) for row in training if row["recovery24_eligible"]]
    recovery48 = [float(row["delta_sofa_48"]) for row in training if row["recovery48_eligible"]]
    mean24, scale24 = _population(recovery24)
    mean48, scale48 = _population(recovery48)
    support = [int(row["organ_support_label"]) for row in training if row["organ_support_eligible"]]
    positives = sum(value == 1 for value in support)
    negatives = sum(value == 0 for value in support)
    if positives == 0 or positives + negatives != len(support):
        raise Phase10ContractError("training support labels cannot produce a class weight")
    encoded_static_names = ["age_years__z"]
    for name in static_names[1:]:
        encoded_static_names.extend(f"{name}=={value}" for value in categories[name])
    return {
        "artifact_version": "synthetic_feature_preprocessor_v1",
        "contract_version": config.version,
        "status": "FROZEN_TRAIN_FIT_TRANSFORM",
        "fit_partition": "train",
        "validation_or_test_statistics_used": False,
        "split_sha256": split_sha256,
        "fit_subjects_sha256": fit_subjects_sha256,
        "fit_subject_count": len(train_subjects),
        "fit_row_count": len(training),
        "feature_schema_sha256": feature_schema_sha256,
        "model_input_contract_version": "synthetic_model_input_contract_v1",
        "model_input_contract_sha256": model_input_contract_sha256,
        "temporal_feature_names": list(names),
        "dynamic_imputation": {
            "method": "TRAIN_GENUINE_OBSERVATION_MEDIAN_PER_CHANNEL",
            "values": medians,
            "genuine_observation_counts": [len(values) for values in observed],
        },
        "dynamic_scaling": {
            "method": "TRAIN_ACTIVE_BIN_POPULATION_ZSCORE_AFTER_IMPUTATION",
            "means": [item[0] for item in dynamic_means_scales],
            "scales": [item[1] for item in dynamic_means_scales],
            "active_cell_counts": [len(values) for values in active_cells],
            "zero_variance_policy": "UNIT_SCALE",
        },
        "padding": {"output_value": config.padding_output_value, "mask_preserved": True},
        "observation_mask": "PRESERVED_UNCHANGED",
        "tslo": {"transform": "IDENTITY_FROZEN_HOURS", "no_observation_sentinel": config.tslo_sentinel},
        "raw_static_feature_names": list(static_names),
        "encoded_static_feature_names": encoded_static_names,
        "static_age": {"method": "TRAIN_POPULATION_ZSCORE", "mean": age_mean, "scale": age_scale},
        "static_categories": {"method": "TRAIN_VOCABULARY_FULL_ONE_HOT_WITH_UNKNOWN_BUCKET", "unknown_token": config.unknown_token, "vocabularies": categories},
        "recovery_target_scaler": {
            "scaler_version": "recovery_target_scaler_v1",
            "source_partition": "train",
            "split_hash": split_sha256,
            "mean_24": mean24, "scale_24": scale24, "eligible_count_24": len(recovery24),
            "mean_48": mean48, "scale_48": scale48, "eligible_count_48": len(recovery48),
            "tensor_contract_version": "synthetic_model_input_contract_v1",
            "feature_schema_version": "synthetic_feature_schema_v2",
            "variance_convention": "population_std_ddof_0", "zero_variance_policy": "unit_scale",
            "provenance": "TRAIN_PARTITION_ELIGIBLE_TARGETS_ONLY",
        },
        "support_class_weight": {
            "formula_version": "eligible_train_negative_over_positive_v1",
            "source_partition": "train", "split_hash": split_sha256,
            "eligible_train_count": len(support), "positive_count": positives, "negative_count": negatives,
            "pos_weight": negatives / positives, "formula": "N_negative / N_positive",
            "label_contract_version": "organ_support_label_rule_v1_unchanged",
            "event_dictionary_version": "synthetic_event_dict_v2",
            "source_config_hash": event_dictionary_sha256,
            "provenance": "ELIGIBLE_TRAIN_ROWS_ONLY",
        },
    }


def transform_row(row: Mapping[str, object], split: str, preprocessor: Mapping[str, object]) -> Mapping[str, object]:
    transformed = transform_feature_view(
        {
            "history_values": row["history_values"],
            "observation_mask": row["observation_mask"],
            "tslo_hours": row["tslo_hours"],
            "padding_mask": row["padding_mask"],
            "static_features": row["static_features"],
            "static_feature_names": row["static_feature_names"],
        },
        preprocessor,
    )
    return {
        "subject_id": row["subject_id"], "stay_id": row["stay_id"],
        "prediction_time": row["prediction_time"], "grid_index": row["grid_index"],
        "split": split,
        "feature_schema_version": row["feature_schema_version"],
        "feature_schema_sha256": row["feature_schema_sha256"],
        "preprocessor_version": preprocessor["artifact_version"],
        "temporal_feature_names": row["temporal_feature_names"],
        "sequence_values": transformed["history_values"],
        "observation_mask": transformed["observation_mask"], "tslo_hours": transformed["tslo_hours"],
        "padding_mask": transformed["padding_mask"],
        "static_features": transformed["static_features"],
        "static_feature_names": transformed["static_feature_names"],
        "delta_sofa_24": row["delta_sofa_24"], "recovery24_eligible": row["recovery24_eligible"],
        "delta_sofa_48": row["delta_sofa_48"], "recovery48_eligible": row["recovery48_eligible"],
        "icu_time_log1p": row["icu_time_log1p"], "icu_time_eligible": row["icu_time_eligible"],
        "organ_support_label": row["organ_support_label"], "organ_support_eligible": row["organ_support_eligible"],
    }


def transform_feature_view(view: Mapping[str, object], preprocessor: Mapping[str, object]) -> Mapping[str, object]:
    """Apply the exact frozen transform to a GRU or flattened XGBoost view."""
    medians = preprocessor["dynamic_imputation"]["values"]
    means = preprocessor["dynamic_scaling"]["means"]
    scales = preprocessor["dynamic_scaling"]["scales"]
    padding_value = float(preprocessor["padding"]["output_value"])
    feature_count = len(medians)
    history = view["history_values"]
    flat = bool(history) and not isinstance(history[0], (list, tuple))
    if flat:
        if len(history) != 8 * feature_count:
            raise Phase10ContractError("flattened history has the wrong Phase-10 dimension")
        history = [history[index * feature_count:(index + 1) * feature_count] for index in range(8)]
    elif len(history) != 8 or any(len(values) != feature_count for values in history):
        raise Phase10ContractError("history must have shape [8,F]")
    padding_mask = view["padding_mask"]
    if len(padding_mask) != 8:
        raise Phase10ContractError("padding mask must have length eight")
    sequence = []
    for bin_index, padded in enumerate(padding_mask):
        transformed = []
        for feature_index, value in enumerate(history[bin_index]):
            if padded:
                transformed.append(padding_value)
            else:
                dense = medians[feature_index] if value is None else float(value)
                transformed.append((dense - means[feature_index]) / scales[feature_index])
        sequence.append(transformed)
    static = view["static_features"]
    raw_static_names = tuple(preprocessor["raw_static_feature_names"])
    supplied_names = tuple(view.get("static_feature_names", raw_static_names))
    if supplied_names != raw_static_names or len(static) != len(raw_static_names):
        raise Phase10ContractError("raw static feature order does not match the preprocessor")
    age = preprocessor["static_age"]
    encoded = [(float(static[0]) - age["mean"]) / age["scale"]]
    vocabularies = preprocessor["static_categories"]["vocabularies"]
    unknown = preprocessor["static_categories"]["unknown_token"]
    for index, name in enumerate(raw_static_names[1:], start=1):
        vocabulary = vocabularies[name]
        value = str(static[index])
        selected = value if value in vocabulary else unknown
        encoded.extend(1.0 if category == selected else 0.0 for category in vocabulary)
    return {
        "history_values": tuple(value for row in sequence for value in row) if flat else sequence,
        "observation_mask": view["observation_mask"], "tslo_hours": view["tslo_hours"],
        "padding_mask": padding_mask,
        "static_features": encoded,
        "static_feature_names": preprocessor["encoded_static_feature_names"],
    }


class FrozenSyntheticFeaturePreprocessor:
    """Transform-only serving adapter backed by the Phase-10 JSON artifact."""

    def __init__(self, artifact_path: Path, *, task: str = "shared") -> None:
        self.task = task
        self.artifact_sha256 = sha256_file(artifact_path)
        self._payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if self._payload.get("status") != "FROZEN_TRAIN_FIT_TRANSFORM":
            raise Phase10ContractError("preprocessor artifact is not frozen")
        if self._payload.get("fit_partition") != "train" or self._payload.get("validation_or_test_statistics_used") is not False:
            raise Phase10ContractError("preprocessor artifact lacks training-only provenance")

    def transform(self, prepared_input: object) -> object:
        if not isinstance(prepared_input, Mapping):
            raise Phase10ContractError("prepared serving input must be a mapping")
        return transform_feature_view(prepared_input, self._payload)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _write_split(path: Path, assignments: Mapping[str, str], fingerprints: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("subject_id", "split", "clone_fingerprint_sha256"))
        for subject_id in sorted(assignments):
            writer.writerow((subject_id, assignments[subject_id], fingerprints[subject_id]))


def build_phase10(
    *, repo_root: Path, config_path: Path, phase9_manifest_path: Path,
    raw_manifest_path: Path, output_directory: Path, generation_timestamp: str,
) -> Path:
    config = load_phase10_config(config_path)
    model_input_contract_path = repo_root / "configs/synthetic/model_input_contract_v1.json"
    model_input_contract = json.loads(model_input_contract_path.read_text(encoding="utf-8"))
    if model_input_contract.get("status") != "FROZEN_SYNTHETIC_AUTHORIZED":
        raise Phase10ContractError("model-input successor contract is not frozen")
    phase9 = json.loads(phase9_manifest_path.read_text(encoding="utf-8"))
    artifact = next(item for item in phase9["artifacts"] if item["logical_name"] == "pre_split_scientific_package")
    package_path = repo_root / artifact["path"]
    if sha256_file(package_path) != artifact["sha256"]:
        raise Phase10ContractError("Phase-9 package hash mismatch")
    rows = load_jsonl(package_path)
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_paths = {}
    for item in raw_manifest["artifacts"]:
        path = repo_root / item["repository_relative_path"]
        if sha256_file(path) != item["sha256"]:
            raise Phase10ContractError("raw Phase-9 artifact hash mismatch: " + item["logical_name"])
        raw_paths[item["logical_name"]] = path
    subjects = load_jsonl(raw_paths["subjects"])
    fingerprints = exact_raw_trajectory_fingerprints(
        subjects, load_jsonl(raw_paths["episodes"]), load_jsonl(raw_paths["raw_events"]), load_jsonl(raw_paths["support_intervals"])
    )
    assignments, subject_counts, clone_count = assign_subjects((row["subject_id"] for row in subjects), fingerprints, config)
    row_subjects = {str(row["subject_id"]) for row in rows}
    if row_subjects != set(assignments):
        raise Phase10ContractError("Phase-9 rows and raw split subjects differ")
    temp_root = Path(tempfile.mkdtemp(prefix=".phase10-", dir=repo_root))
    try:
        split_path = temp_root / "synthetic_split_v2.csv"
        _write_split(split_path, assignments, fingerprints)
        split_hash = sha256_file(split_path)
        fit_ids_path = temp_root / "synthetic_train_fit_subjects_v1.txt"
        fit_ids_path.write_text("".join(subject_id + "\n" for subject_id in sorted(assignments) if assignments[subject_id] == "train"), encoding="utf-8")
        fit_hash = sha256_file(fit_ids_path)
        feature_schema_hashes = {str(row["feature_schema_sha256"]) for row in rows}
        if len(feature_schema_hashes) != 1:
            raise Phase10ContractError("Phase-9 feature schema hash is inconsistent")
        preprocessor = fit_preprocessing(
            rows, assignments, config, feature_schema_sha256=next(iter(feature_schema_hashes)),
            split_sha256=split_hash, fit_subjects_sha256=fit_hash,
            model_input_contract_sha256=sha256_file(model_input_contract_path),
            event_dictionary_sha256=phase9["dependencies"]["event_dictionary"]["sha256"],
        )
        if preprocessor["encoded_static_feature_names"] != model_input_contract["static_features"]["ordered_names"]:
            raise Phase10ContractError("fitted static vocabulary differs from model-input contract")
        preprocessor_path = temp_root / "synthetic_feature_preprocessor_v1.json"
        _write_json(preprocessor_path, preprocessor)
        preprocessor_hash = sha256_file(preprocessor_path)
        recovery_path = temp_root / "recovery_target_scaler_synthetic_v1.json"
        _write_json(recovery_path, preprocessor["recovery_target_scaler"])
        support_path = temp_root / "support_class_weight_synthetic_v1.json"
        _write_json(support_path, preprocessor["support_class_weight"])
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        if output_directory.exists():
            raise FileExistsError(output_directory)
        local_output = temp_root / "model_ready"
        local_output.mkdir()
        handles = {name: (local_output / f"{name}.jsonl").open("wb") for name in ("train", "validation")}
        row_counts = Counter()
        try:
            for row in rows:
                split = assignments[str(row["subject_id"])]
                row_counts[split] += 1
                if split == "test":
                    continue
                handles[split].write(canonical_json_bytes(transform_row(row, split, preprocessor)))
        finally:
            for handle in handles.values():
                handle.close()
        split_final = repo_root / "artifacts/splits/synthetic_split_v2.csv"
        metadata_final = repo_root / "artifacts/splits/synthetic_split_v2.metadata.json"
        fit_final = repo_root / "artifacts/preprocessors/synthetic_train_fit_subjects_v1.txt"
        preprocessor_final = repo_root / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json"
        recovery_final = repo_root / "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json"
        support_final = repo_root / "artifacts/preprocessors/support_class_weight_synthetic_v1.json"
        manifest_final = repo_root / "artifacts/manifests/synthetic_phase10_manifest_v1.json"
        for final in (split_final, metadata_final, fit_final, preprocessor_final, recovery_final, support_final, manifest_final):
            if final.exists():
                raise FileExistsError(final)
            final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(split_path, split_final)
        shutil.copyfile(fit_ids_path, fit_final)
        shutil.copyfile(preprocessor_path, preprocessor_final)
        shutil.copyfile(recovery_path, recovery_final)
        shutil.copyfile(support_path, support_final)
        os.replace(local_output, output_directory)
        metadata = {
            "artifact_version": "synthetic_subject_split_v2", "status": "FROZEN_SYNTHETIC_AUTHORIZED",
            "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION", "method": "SEEDED_SHA256_RANK_EXACT_COUNT",
            "seed": config.seed, "hash_namespace": config.namespace, "fractions": config.fractions,
            "subject_counts": subject_counts, "total_subjects": len(assignments), "exact_clone_count": clone_count,
            "clone_policy": "GROUP_EXACT_RAW_TRAJECTORY_FINGERPRINTS_BEFORE_ASSIGNMENT",
            "labels_or_performance_used": False, "fake_calendar_fields_used": False,
            "final_test_access": "SEALED_UNTIL_G3_EQUIVALENT_FREEZE",
            "split_path": str(split_final.relative_to(repo_root)), "split_sha256": split_hash,
            "source_phase9_manifest_path": str(phase9_manifest_path.relative_to(repo_root)),
            "source_phase9_manifest_sha256": sha256_file(phase9_manifest_path),
            "generation_timestamp_utc": generation_timestamp,
        }
        _write_json(metadata_final, metadata)
        artifacts = []
        for logical, final in (
            ("subject_split", split_final), ("split_metadata", metadata_final), ("train_fit_subjects", fit_final),
            ("feature_preprocessor", preprocessor_final), ("recovery_target_scaler", recovery_final),
            ("support_class_weight", support_final),
        ):
            artifacts.append({"logical_name": logical, "path": str(final.relative_to(repo_root)), "sha256": sha256_file(final)})
        for split in ("train", "validation"):
            final = output_directory / f"{split}.jsonl"
            artifacts.append({"logical_name": f"model_ready_{split}", "path": str(final.relative_to(repo_root)), "sha256": sha256_file(final), "record_count": row_counts[split], "access_status": "AVAILABLE"})
        manifest = {
            "manifest_version": "synthetic_phase10_manifest_v1", "status": "PHASE10_COMPLETE",
            "decision_authority": "USER_DELEGATED_AI_PROJECT_DECISION", "contract_path": str(config_path.relative_to(repo_root)),
            "contract_sha256": sha256_file(config_path), "phase9_manifest_path": str(phase9_manifest_path.relative_to(repo_root)),
            "phase9_manifest_sha256": sha256_file(phase9_manifest_path), "split_sha256": split_hash,
            "preprocessor_sha256": preprocessor_hash, "subject_counts": subject_counts,
            "model_input_contract_path": str(model_input_contract_path.relative_to(repo_root)),
            "model_input_contract_sha256": sha256_file(model_input_contract_path),
            "row_counts": {name: row_counts[name] for name in SPLITS}, "feature_dimension": len(preprocessor["temporal_feature_names"]),
            "encoded_static_dimension": len(preprocessor["encoded_static_feature_names"]),
            "test_label_statistics_computed": False, "test_opened": False, "g3_freeze_active": False,
            "test_model_ready_artifact": "NOT_MATERIALIZED_UNTIL_G3_EQUIVALENT_FREEZE",
            "model_training_performed": False, "hyperparameter_search_performed": False,
            "calibration_performed": False, "threshold_selection_performed": False,
            "generation_timestamp_utc": generation_timestamp, "artifacts": artifacts,
        }
        _write_json(manifest_final, manifest)
        return manifest_final
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
