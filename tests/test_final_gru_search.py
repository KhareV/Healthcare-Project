import copy
import json
from pathlib import Path

import pytest
import torch

from data.gru_canonical import GRUCanonicalDataError, Phase10GRUDataset, TaskEligibleGRUDataset
from experiments.gru_final import (
    BUDGET, GRUSearchError, MASTER_SEED, TASKS, candidate_manifest,
    canonical_sha256, choose_best, derived_seed, materialize_candidates,
    validate_candidates, validate_checkpoint_lineage, validate_search_space,
)
from models.gru import GRUEncoderConfig
from training.checkpoint import save_checkpoint
from training.smoke import SmokeTestModel
from training.tasks.icu_time import masked_icu_time_huber_loss
from training.tasks.organ_support import masked_support_bce_with_logits_loss
from training.tasks.recovery import masked_recovery_huber_loss
from preprocess.target_scaler import RecoveryTargetScaler


ROOT = Path(__file__).resolve().parents[1]
SPACE = json.loads((ROOT / "configs/search_space_gru_final_v1.json").read_text())


def test_final_space_and_exact_deterministic_lhs_budget():
    validate_search_space(SPACE)
    for task in TASKS:
        first = materialize_candidates(SPACE, task)
        second = materialize_candidates(SPACE, task)
        assert first == second
        assert len(first) == BUDGET
        assert len({row["config_hash"] for row in first}) == BUDGET
        assert not any(row["candidate_id"].endswith("031") for row in first)


def test_exact_seed_derivation_examples():
    import hashlib
    expected = int.from_bytes(hashlib.sha256(b"20260921|vedant_final_gru|recovery|sampler").digest()[:4], "big")
    assert MASTER_SEED == 20260921
    assert derived_seed("recovery", "sampler") == expected


def test_candidate_manifest_is_pretraining_frozen_and_scientific():
    payload = candidate_manifest(SPACE, "recovery", "a" * 40)
    assert payload["candidate_count"] == 30
    assert payload["status"] == "FROZEN_BEFORE_TRAINING"
    assert payload["run_type"] == "scientific"
    assert payload["test_accessed"] is False


def test_duplicate_or_candidate_31_is_rejected():
    rows = list(materialize_candidates(SPACE, "recovery"))
    rows[-1] = dict(rows[-1], candidate_id="gru-recovery-031")
    with pytest.raises(GRUSearchError):
        validate_candidates(rows, "recovery")


def test_test_partition_is_structurally_forbidden():
    with pytest.raises(GRUCanonicalDataError, match="TEST ACCESS FORBIDDEN"):
        Phase10GRUDataset(ROOT, "test")


def test_accepted_phase10_dataset_shape_and_information():
    row = Phase10GRUDataset(ROOT, "validation")[0]
    assert row["sequence"].shape == (8, 21)
    assert row["observation_mask"].shape == (8, 21)
    assert row["tslo"].shape == (8, 21)
    assert row["padding_mask"].shape == (8,)
    assert row["static_features"].shape == (8,)


def test_training_task_views_contain_only_eligible_rows():
    base = Phase10GRUDataset(ROOT, "train")
    for task in TASKS:
        view = TaskEligibleGRUDataset(base, task)
        for index in (0, len(view) - 1):
            masks = view[index]["eligibility"]
            assert (masks["recovery24"] or masks["recovery48"]) if task == "recovery" else masks[task]


def test_gru_config_requires_mask_tslo_static_and_unidirectional():
    config = GRUEncoderConfig(feature_dim=21, hidden_dim=64, num_layers=1, dropout=.2,
                              include_observation_mask=True, include_tslo=True, static_dim=8)
    config.validate()


def test_task_losses_preserve_frozen_masking_and_weighting():
    scaler = RecoveryTargetScaler.load(ROOT / "artifacts/preprocessors/recovery_target_scaler_synthetic_v1.json")
    recovery = masked_recovery_huber_loss(torch.zeros(2, 2), torch.tensor([[1., 2.], [3., float("nan")]]),
                                          torch.tensor([[True, True], [True, False]]), scaler)
    assert recovery.loss is not None and recovery.eligible_count == 3
    icu = masked_icu_time_huber_loss(torch.zeros(2), torch.tensor([1., float("nan")]), torch.tensor([True, False]))
    assert icu.loss is not None and icu.eligible_count == 1
    support = masked_support_bce_with_logits_loss(torch.zeros(2), torch.tensor([1., float("nan")]),
                                                  torch.tensor([True, False]), pos_weight=1.5)
    assert support.loss is not None and support.eligible_count == 1


@pytest.mark.parametrize("task,rows,winner", [
    ("recovery", [dict(candidate_id="b", status="COMPLETE", mae24=1., mae48=2., rmse24=3.),
                  dict(candidate_id="a", status="COMPLETE", mae24=1., mae48=2., rmse24=3.)], "a"),
    ("icu_time", [dict(candidate_id="b", status="COMPLETE", median_absolute_error_hours=2., mae_hours=2., rmse_hours=3.),
                  dict(candidate_id="a", status="COMPLETE", median_absolute_error_hours=1., mae_hours=3., rmse_hours=4.)], "a"),
    ("organ_support", [dict(candidate_id="b", status="COMPLETE", auprc=.7, brier=.2, auroc=.8, probability_type="raw_uncalibrated"),
                       dict(candidate_id="a", status="COMPLETE", auprc=.7, brier=.1, auroc=.7, probability_type="raw_uncalibrated")], "a"),
])
def test_task_specific_scientific_ranking(task, rows, winner):
    assert choose_best(task, rows)["candidate_id"] == winner


def test_support_calibrated_candidate_cannot_be_ranked():
    with pytest.raises(GRUSearchError, match="uncalibrated"):
        choose_best("organ_support", [dict(candidate_id="a", status="COMPLETE", auprc=.7, brier=.2, auroc=.8, probability_type="calibrated")])


def test_checkpoint_lineage_fails_closed(tmp_path):
    model = SmokeTestModel(2); optimizer = torch.optim.AdamW(model.parameters())
    metadata = {"run_id":"scientific","task":"recovery","model_family":"gru","seed":1,"epoch":1,
                "config_hash":"a"*64,"tensor_contract_version":"synthetic_model_input_contract_v1",
                "feature_schema_version":"synthetic_feature_schema_v2","split_hash":"b"*64,
                "code_commit":"c"*40,"validation_value":1.0,"synthetic_smoke_test":False,
                "synthetic_scientific_search":True,"g1_sha256":"d"*64}
    path = tmp_path / "model.pt"; save_checkpoint(path, model, optimizer, metadata)
    assert validate_checkpoint_lineage(path, {"g1_sha256":"d"*64})["run_id"] == "scientific"
    with pytest.raises(GRUSearchError, match="lineage mismatch"):
        validate_checkpoint_lineage(path, {"g1_sha256":"e"*64})


def test_search_space_mutation_fails_closed():
    changed = copy.deepcopy(SPACE); changed["fixed"]["max_epochs"] = 61
    with pytest.raises(GRUSearchError):
        validate_search_space(changed)
