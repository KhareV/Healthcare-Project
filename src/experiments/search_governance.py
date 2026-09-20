"""Frozen-budget, validation-only search governance for GRU and XGBoost."""

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple


SEARCH_BUDGET = 30
TASKS = ("recovery", "icu_stay_time", "organ_support")
FAMILIES = ("gru", "xgboost")
TEST_SAMPLER = "deterministic_seeded_synthetic_dry_run_v1"
STARTED_STATUSES = {"RUNNING", "COMPLETE", "FAILED", "CANCELLED"}


class GovernanceError(ValueError):
    """Raised when search governance, isolation, or frozen contracts are violated."""


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_search_space(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_search_space(payload)
    return payload, canonical_sha256(payload)


def validate_search_space(space: Mapping[str, object]) -> None:
    family = space.get("family")
    if family not in FAMILIES or space.get("budget_per_task") != SEARCH_BUDGET:
        raise GovernanceError("search family and exact budget must match the frozen plan")
    parameters = space.get("parameters")
    if not isinstance(parameters, Mapping):
        raise GovernanceError("search parameters must be a mapping")
    if family == "gru":
        expected = {
            "hidden_dim": {"type": "categorical", "values": [64, 128, 192]},
            "num_layers": {"type": "categorical", "values": [1, 2]},
            "dropout": {"type": "continuous", "minimum": 0.1, "maximum": 0.4, "distribution": "UNRESOLVED"},
            "learning_rate": {"type": "continuous", "minimum": 0.0001, "maximum": 0.003, "distribution": "log_uniform"},
            "batch_size": {"type": "categorical", "values": [64, 128]},
            "weight_decay": {"type": "continuous", "minimum": 0.000001, "maximum": 0.001, "distribution": "log_uniform"},
        }
        training = space.get("training", {})
        if training.get("maximum_epochs") != 60 or training.get("early_stopping_patience") != 8:
            raise GovernanceError("GRU training cap/patience conflict with Master Plan")
        if training.get("optimizer") != "adamw" or training.get("restore_best_epoch") is not True:
            raise GovernanceError("GRU optimizer/restore policy conflict with Master Plan")
    else:
        expected = {
            "max_depth": {"minimum": 3, "maximum": 8, "type": "integer", "distribution": "UNRESOLVED"},
            "learning_rate": {"minimum": 0.01, "maximum": 0.2, "type": "continuous", "distribution": "log_uniform"},
            "subsample": {"minimum": 0.6, "maximum": 1.0, "type": "continuous", "distribution": "UNRESOLVED"},
            "colsample_bytree": {"minimum": 0.6, "maximum": 1.0, "type": "continuous", "distribution": "UNRESOLVED"},
            "min_child_weight": {"minimum": 1, "maximum": 10, "type": "numeric", "distribution": "UNRESOLVED"},
            "reg_alpha": {"minimum": 0.0, "maximum": 1.0, "type": "continuous", "distribution": "UNRESOLVED"},
            "reg_lambda": {"minimum": 0.1, "maximum": 10.0, "type": "continuous", "distribution": "UNRESOLVED"},
        }
        training = space.get("training", {})
        if training.get("maximum_estimators") != 1000 or training.get("retain_best_iteration") is not True:
            raise GovernanceError("XGBoost estimator/restore policy conflict with Master Plan")
    if parameters != expected:
        raise GovernanceError("SCIENTIFIC CONFIGURATION CONFLICT: search ranges differ")


def _log_value(low, high, fraction):
    return math.exp(math.log(low) + fraction * (math.log(high) - math.log(low)))


def materialize_candidates(space, *, task, seed, run_type):
    """Generate exact-30 candidates only for a declared synthetic dry run."""

    validate_search_space(space)
    if task not in TASKS:
        raise GovernanceError("unsupported task")
    if run_type != "synthetic_dry_run":
        raise GovernanceError("scientific sampler is unresolved; real candidates cannot be generated")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GovernanceError("dry-run sampler requires an explicit TEST seed")
    rng = random.Random(seed)
    fractions = [index / (SEARCH_BUDGET - 1) for index in range(SEARCH_BUDGET)]
    rng.shuffle(fractions)
    candidates = []
    family = space["family"]
    for index, fraction in enumerate(fractions, start=1):
        if family == "gru":
            config = {
                "hidden_dim": (64, 128, 192)[(index - 1) % 3],
                "num_layers": (1, 2)[((index - 1) // 3) % 2],
                "dropout": 0.1 + 0.3 * fraction,
                "learning_rate": _log_value(0.0001, 0.003, fraction),
                "batch_size": (64, 128)[((index - 1) // 6) % 2],
                "weight_decay": _log_value(0.000001, 0.001, 1.0 - fraction),
            }
        else:
            config = {
                "max_depth": 3 + ((index - 1) % 6),
                "learning_rate": _log_value(0.01, 0.2, fraction),
                "subsample": 0.6 + 0.4 * fraction,
                "colsample_bytree": 0.6 + 0.4 * (1.0 - fraction),
                "min_child_weight": 1.0 + 9.0 * fraction,
                "reg_alpha": fraction,
                "reg_lambda": 0.1 + 9.9 * (1.0 - fraction),
                "n_estimators": 1000,
            }
        candidates.append(
            {
                "candidate_id": "dryrun_{}_{}_{:02d}".format(family, task, index),
                "config": config,
                "config_hash": canonical_sha256(config),
            }
        )
    validate_candidate_list(candidates)
    return tuple(candidates)


def validate_candidate_list(candidates) -> None:
    if len(candidates) != SEARCH_BUDGET:
        raise GovernanceError("candidate list must contain exactly 30 configurations")
    ids = [candidate["candidate_id"] for candidate in candidates]
    hashes = [candidate["config_hash"] for candidate in candidates]
    if len(set(ids)) != SEARCH_BUDGET or len(set(hashes)) != SEARCH_BUDGET:
        raise GovernanceError("all 30 candidate IDs and hyperparameter sets must be unique")
    for candidate in candidates:
        if canonical_sha256(candidate["config"]) != candidate["config_hash"]:
            raise GovernanceError("candidate config hash mismatch")


def load_candidate_manifest(path, space):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("run_type") != "synthetic_dry_run":
        raise GovernanceError("only synthetic dry-run manifests are currently authorized")
    if manifest.get("candidate_count") != SEARCH_BUDGET:
        raise GovernanceError("manifest must declare exactly 30 candidates")
    if manifest.get("sampler") != TEST_SAMPLER:
        raise GovernanceError("unapproved candidate sampler")
    candidates = materialize_candidates(
        space,
        task=manifest["task"],
        seed=manifest["test_sampler_seed"],
        run_type=manifest["run_type"],
    )
    digest = canonical_sha256(candidates)
    if manifest.get("candidate_list_hash") != digest:
        raise GovernanceError("candidate-list hash mismatch")
    return manifest, candidates


def assert_immutable_transition(before, after) -> None:
    if before.get("status") not in STARTED_STATUSES:
        return
    immutable = (
        "search_version",
        "search_space_hash",
        "candidate_list_hash",
        "validation_objective",
        "split_hash",
        "feature_version",
        "label_version",
    )
    changed = [field for field in immutable if before.get(field) != after.get(field)]
    if changed:
        raise GovernanceError("started search cannot mutate: " + ", ".join(changed))


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    run_id: str
    status: str
    metrics: Mapping[str, float]
    probability_type: Optional[str] = None


def choose_best_within_family(results: Sequence[CandidateResult], task: str):
    complete = [result for result in results if result.status == "COMPLETE"]
    if not complete:
        raise GovernanceError("no valid COMPLETE candidates to rank")
    if task == "recovery":
        key, direction = "validation_recovery24_mae", "min"
    elif task == "icu_stay_time":
        key, direction = "validation_median_absolute_error_hours", "min"
    elif task == "organ_support":
        key, direction = "validation_auprc", "max"
        if any(result.probability_type != "raw_uncalibrated" for result in complete):
            raise GovernanceError("support search ranking requires raw uncalibrated probabilities")
    else:
        raise GovernanceError("unsupported task")
    for result in complete:
        value = result.metrics.get(key)
        if value is None or not math.isfinite(float(value)):
            raise GovernanceError("COMPLETE candidate lacks finite primary validation metric")
    ordered = sorted(complete, key=lambda item: item.metrics[key], reverse=direction == "max")
    if len(ordered) > 1 and ordered[0].metrics[key] == ordered[1].metrics[key]:
        raise GovernanceError("UNLOCKED PARAMETER — WITHIN-FAMILY VALIDATION TIE-BREAK POLICY")
    return ordered[0]


class ValidationOnlyAccess:
    ALLOWED = {"train", "validation"}

    @classmethod
    def require(cls, partition):
        if partition not in cls.ALLOWED:
            raise GovernanceError("HARD FAIL — PHASE 11 TEST ACCESS FORBIDDEN")
        return partition


def execute_synthetic_candidate(candidate, *, partition_provider, runner):
    """Exercise a dry-run candidate while requesting only train/validation data."""

    if not candidate.get("candidate_id", "").startswith("dryrun_"):
        raise GovernanceError("synthetic runner accepts only dry-run candidate IDs")
    train_data = partition_provider(ValidationOnlyAccess.require("train"))
    validation_data = partition_provider(ValidationOnlyAccess.require("validation"))
    result = runner(candidate["config"], train_data, validation_data)
    if not isinstance(result, Mapping):
        raise GovernanceError("candidate runner must return validation result metadata")
    if result.get("test_accessed") is not False:
        raise GovernanceError("dry-run result must attest test_accessed=false")
    return result


@dataclass(frozen=True)
class InformationManifest:
    family: str
    row_keys_hash: str
    feature_schema_version: str
    temporal_bins: int
    dynamic_features: Tuple[str, ...]
    observation_masks: Tuple[str, ...]
    tslo_features: Tuple[str, ...]
    static_features: Tuple[str, ...]
    preprocessing_hash: str
    split_hash: str
    eligibility_hash: str
    representation: str


def validate_information_parity(gru: InformationManifest, xgb: InformationManifest) -> None:
    if gru.family != "gru" or xgb.family != "xgboost":
        raise GovernanceError("information parity requires GRU and XGBoost manifests")
    for field in (
        "row_keys_hash",
        "feature_schema_version",
        "temporal_bins",
        "dynamic_features",
        "observation_masks",
        "tslo_features",
        "static_features",
        "preprocessing_hash",
        "split_hash",
        "eligibility_hash",
    ):
        if getattr(gru, field) != getattr(xgb, field):
            raise GovernanceError("XGB/GRU information parity failure: " + field)
    if gru.representation != "sequence_8xf" or xgb.representation != "flattened_8xf":
        raise GovernanceError("only canonical sequence-versus-flattened representation may differ")
