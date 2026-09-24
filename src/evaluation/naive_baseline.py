"""Prespecified, mathematically-defined naive baselines, fit on TRAIN only.

These are not learned models: recovery predicts an exact constant zero,
ICU-time predicts the frozen train-set stay-balanced weighted median of the
model's own target domain (log1p remaining hours, then the same frozen
postprocess everywhere else uses), and organ-support predicts the frozen
train-set stay-balanced eligible prevalence as one constant probability.

Fitting reads only ``model_ready_train`` rows (the same file the accepted
XGB/GRU searches were trained from) — never validation or test outcomes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from data.synthetic.config import canonical_json_bytes
from evaluation.weighted_stats import weighted_mean, weighted_median
from evaluation.weights import compute_stay_weights
from vedant_infra.hashing import sha256_file


NAIVE_BASELINE_VERSION = "naive_baseline_v1"
RECOVERY_CONSTANT_DELTA = 0.0


class NaiveBaselineError(RuntimeError):
    pass


@dataclass(frozen=True)
class NaiveBaselineArtifact:
    artifact_version: str
    source_partition: str
    source_ref: str
    source_sha256: str
    recovery24_constant_delta: float
    recovery48_constant_delta: float
    icu_time_log1p_median: float
    icu_time_train_n_examples: int
    icu_time_train_n_stays: int
    organ_support_prevalence: float
    organ_support_train_n_examples: int
    organ_support_train_n_stays: int
    organ_support_train_n_positive_examples: int
    weighting: str = "stay_balanced_eligible_snapshot_weight_1_over_stay_count"

    def to_mapping(self) -> Mapping[str, object]:
        return asdict(self)

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(self.to_mapping()))
        return sha256_file(path)

    @classmethod
    def load(cls, path: Path) -> "NaiveBaselineArtifact":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("artifact_version") != NAIVE_BASELINE_VERSION:
            raise NaiveBaselineError("naive baseline artifact version mismatch")
        return cls(**payload)


def _load_train_rows(path: Path) -> Tuple[Mapping[str, object], ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise NaiveBaselineError("naive baseline fitting requires train-only rows")
            rows.append(row)
    if not rows:
        raise NaiveBaselineError("no train rows available for naive baseline fitting")
    return tuple(rows)


def fit_naive_baselines(root: Path, *, train_ref: str) -> NaiveBaselineArtifact:
    root = root.resolve()
    train_path = root / train_ref
    rows = _load_train_rows(train_path)

    icu_stays = [row["stay_id"] for row in rows]
    icu_eligible = [bool(row["icu_time_eligible"]) for row in rows]
    icu_weights_result = compute_stay_weights(icu_stays, icu_eligible)
    icu_values = tuple(
        float(rows[index]["icu_time_log1p"]) for index in icu_weights_result.eligible_indices
    )
    icu_weights = tuple(
        icu_weights_result.weights[index] for index in icu_weights_result.eligible_indices
    )
    icu_median = weighted_median(icu_values, icu_weights)

    support_stays = [row["stay_id"] for row in rows]
    support_eligible = [bool(row["organ_support_eligible"]) for row in rows]
    support_weights_result = compute_stay_weights(support_stays, support_eligible)
    support_labels = tuple(
        float(rows[index]["organ_support_label"]) for index in support_weights_result.eligible_indices
    )
    support_weights = tuple(
        support_weights_result.weights[index] for index in support_weights_result.eligible_indices
    )
    prevalence = weighted_mean(support_labels, support_weights)

    return NaiveBaselineArtifact(
        artifact_version=NAIVE_BASELINE_VERSION,
        source_partition="train",
        source_ref=train_ref,
        source_sha256=sha256_file(train_path),
        recovery24_constant_delta=RECOVERY_CONSTANT_DELTA,
        recovery48_constant_delta=RECOVERY_CONSTANT_DELTA,
        icu_time_log1p_median=icu_median,
        icu_time_train_n_examples=icu_weights_result.n_examples,
        icu_time_train_n_stays=icu_weights_result.n_icu_stays,
        organ_support_prevalence=prevalence,
        organ_support_train_n_examples=support_weights_result.n_examples,
        organ_support_train_n_stays=support_weights_result.n_icu_stays,
        organ_support_train_n_positive_examples=int(sum(support_labels)),
    )


def load_or_fit_naive_baselines(
    root: Path, *, artifact_ref: str, train_ref: str
) -> Tuple[NaiveBaselineArtifact, Path, str]:
    """Validate and reuse an existing frozen artifact, or fit+freeze one now."""

    root = root.resolve()
    path = root / artifact_ref
    if path.is_file():
        artifact = NaiveBaselineArtifact.load(path)
        train_path = root / artifact.source_ref
        if not train_path.is_file() or sha256_file(train_path) != artifact.source_sha256:
            raise NaiveBaselineError("frozen naive baseline artifact source hash is stale")
        return artifact, path, sha256_file(path)
    artifact = fit_naive_baselines(root, train_ref=train_ref)
    digest = artifact.save(path)
    return artifact, path, digest
