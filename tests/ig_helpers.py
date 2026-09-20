from dataclasses import replace

import torch

from data.collate import CanonicalBatch
from explainability.ig import AttributionTarget, IGConfig
from explainability.router import ExplanationContext


CUTOFF = "2026-01-04T12:00:00+00:00"
FEATURE_VERSION = "SYNTHETIC_FEATURES_PHASE6_NOT_REAL"
MODEL_HASH = "9" * 64


class SyntheticTwoOutputGRU(torch.nn.Module):
    task = "recovery"
    family = "gru"
    artifact_sha256 = MODEL_HASH
    horizon_order = ("delta_sofa_24h", "delta_sofa_48h")

    def __init__(self):
        super().__init__()
        self.output_weights = torch.nn.Parameter(torch.ones(2))

    def forward(self, batch):
        valid = (~batch.padding_mask).unsqueeze(-1).to(batch.sequence.dtype)
        values = batch.sequence * valid
        return torch.stack(
            (
                values[:, :, 0].sum(1) * self.output_weights[0],
                values[:, :, 1].sum(1) * self.output_weights[1],
            ),
            dim=1,
        )


class SyntheticScalarGRU(torch.nn.Module):
    family = "gru"
    artifact_sha256 = MODEL_HASH

    def __init__(self, task):
        super().__init__()
        self.task = task

    def forward(self, batch):
        valid = (~batch.padding_mask).unsqueeze(-1).to(batch.sequence.dtype)
        return (batch.sequence * valid).sum(dim=(1, 2)).unsqueeze(1)


def batch():
    sequence = torch.tensor(
        [[[50.0, -50.0], [40.0, -40.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]]
    )
    return CanonicalBatch(
        identifiers={
            "subject_id": ("SYNTHETIC",),
            "stay_id": ("SYNTHETIC_STAY",),
            "prediction_time": (CUTOFF,),
            "grid_index": (8,),
            "split": ("validation",),
        },
        sequence=sequence,
        padding_mask=torch.tensor([[True, True, False, False, False, False, False, False]]),
        observation_mask=torch.ones_like(sequence, dtype=torch.bool),
        tslo=None,
        static_features=None,
        targets={},
        eligibility={},
        versions={"feature_schema_version": FEATURE_VERSION},
        feature_names=("SYNTHETIC_A", "SYNTHETIC_B"),
    )


def target(task="recovery", name="delta_sofa_24h", index=0):
    return AttributionTarget(
        task=task,
        output_name=name,
        output_index=index,
        output_domain_version="SYNTHETIC_MODEL_NATIVE_OUTPUT_V1",
        scientific_scope="synthetic_development_only",
    )


def config():
    return IGConfig(
        version="SYNTHETIC_IG_CONFIG_V1",
        n_steps=32,
        method="gausslegendre",
        internal_batch_size=None,
        return_convergence_delta=True,
        scientific_scope="synthetic_development_only",
    )


def context(model, explanation_target="delta_sofa_24h", prepared=None):
    prepared = prepared or batch()
    return ExplanationContext(
        task=model.task,
        input_task=model.task,
        family="gru",
        input_family="gru",
        model_sha256=MODEL_HASH,
        prediction_time=CUTOFF,
        input_prediction_time=CUTOFF,
        manifest_version="selected_models_synthetic_phase6_v1",
        manifest_sha256="8" * 64,
        feature_schema_version=FEATURE_VERSION,
        feature_schema_sha256=None,
        model=model,
        prepared_input=prepared,
        raw_output=model(prepared).detach(),
        explanation_target=explanation_target,
        feature_metadata={"feature_schema_version": FEATURE_VERSION},
        synthetic=True,
    )
