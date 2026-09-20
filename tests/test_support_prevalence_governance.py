import json
import shutil
from pathlib import Path

import pytest

from labels.endpoint_freeze import create_synthetic_endpoint_freeze
from labels.support_prevalence import (
    SupportPrevalenceError,
    create_synthetic_label_artifact,
    generate_raw_count_report,
    json_label_loader,
)
from vedant_infra.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in (
        "configs/event_dict_v1.yaml",
        "configs/timestamp_spec_v1.yaml",
        "configs/support_prevalence_v1.json",
        "src/labels/organ_support.py",
        "src/labels/support_prevalence.py",
        "tests/fixtures/support_endpoint_freeze/synthetic_label_rows_v1.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return root


def _chain(tmp_path: Path):
    root = _root(tmp_path)
    freeze_path = root / "artifacts/labels/synthetic/freeze.json"
    freeze_hash = create_synthetic_endpoint_freeze(root, freeze_path)
    label_path = root / "artifacts/labels/synthetic/labels.json"
    label_hash = create_synthetic_label_artifact(
        root,
        root / "tests/fixtures/support_endpoint_freeze/synthetic_label_rows_v1.json",
        label_path,
        freeze_path=freeze_path,
        freeze_sha256=freeze_hash,
    )
    descriptor = {
        "scope": "synthetic",
        "population_id": "SYNTHETIC_PHASE4_FIXTURE_ONLY",
        "partition": "synthetic",
        "freeze_sha256": freeze_hash,
        "label_artifact_sha256": label_hash,
        "data_version": "synthetic_phase4_population_v1",
        "split_version": "synthetic_partition_only_v1",
        "timestamp_spec_version": "timestamp_spec_v1",
    }
    return root, freeze_path, freeze_hash, label_path, label_hash, descriptor


def _generate(chain, output=None, loader=json_label_loader, **overrides):
    root, freeze_path, freeze_hash, label_path, label_hash, descriptor = chain
    args = dict(
        root=root, freeze_path=freeze_path, freeze_sha256=freeze_hash,
        scope="synthetic", population_id="SYNTHETIC_PHASE4_FIXTURE_ONLY",
        partition="synthetic", label_artifact_path=label_path,
        label_artifact_sha256=label_hash, label_loader=loader,
        descriptor=descriptor,
        output_path=output or root / "artifacts/labels/synthetic/report.json",
    )
    args.update(overrides)
    return generate_raw_count_report(**args)


def test_synthetic_freeze_label_report_chain_has_raw_counts_only(tmp_path):
    chain = _chain(tmp_path)
    digest = _generate(chain)
    report_path = chain[0] / "artifacts/labels/synthetic/report.json"
    report = json.loads(report_path.read_text())
    assert digest == sha256_file(report_path)
    assert report["raw_counts"] == {
        "canonical_rows": 5, "eligible_rows": 3, "positive_rows": 2,
        "negative_rows": 1, "censored_rows": 1, "not_at_risk_rows": 1,
        "unique_stays": 4, "eligible_stays": 2, "positive_stays": 2,
    }
    assert report["headline_prevalence"] is None
    assert report["class_weights"] is None
    assert report["acceptance_gate_applied"] is False
    assert report["freeze_sha256"] == chain[2]
    assert report["label_artifact_sha256"] == chain[4]


@pytest.mark.parametrize(
    "override,match",
    [
        ({"freeze_sha256": "0" * 64}, "hash mismatch"),
        ({"scope": "real"}, "scope"),
        ({"population_id": "AMBIGUOUS"}, "population"),
        ({"partition": "test"}, "partition"),
        ({"label_artifact_sha256": "0" * 64}, "descriptor mismatch|hash mismatch"),
    ],
)
def test_access_failures_happen_before_loader(tmp_path, override, match):
    chain = _chain(tmp_path)
    calls = []
    def hostile(_path):
        calls.append(True)
        raise AssertionError("loader must remain unreachable")
    with pytest.raises(SupportPrevalenceError, match=match):
        _generate(chain, loader=hostile, **override)
    assert calls == []


def test_descriptor_mismatch_happens_before_loader(tmp_path):
    chain = _chain(tmp_path)
    descriptor = dict(chain[5])
    descriptor["data_version"] = "wrong"
    calls = []
    def hostile(_path):
        calls.append(True)
        return {}
    with pytest.raises(SupportPrevalenceError, match="binding mismatch"):
        _generate(chain, loader=hostile, descriptor=descriptor)
    assert calls == [True]  # Access was authorized; envelope validation then rejects drift.


@pytest.mark.parametrize(
    "case,eligible,label",
    [
        ("INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION", True, 0),
        ("INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON", True, 0),
    ],
)
def test_censoring_and_not_at_risk_can_never_become_negative(tmp_path, case, eligible, label):
    chain = _chain(tmp_path)
    envelope = json.loads(chain[3].read_text())
    row = next(item for item in envelope["records"] if item["audit_case"] == case)
    row["eligible"], row["label"] = eligible, label
    with pytest.raises(SupportPrevalenceError, match="contradicts"):
        _generate(chain, loader=lambda _: envelope)


def test_duplicate_cutoff_is_rejected(tmp_path):
    chain = _chain(tmp_path)
    envelope = json.loads(chain[3].read_text())
    envelope["records"].append(dict(envelope["records"][0]))
    with pytest.raises(SupportPrevalenceError, match="duplicate"):
        _generate(chain, loader=lambda _: envelope)


def test_dependency_mutation_blocks_before_loader(tmp_path):
    chain = _chain(tmp_path)
    event = chain[0] / "configs/event_dict_v1.yaml"
    event.write_text(event.read_text() + "\n# stale freeze attack\n")
    calls = []
    with pytest.raises(SupportPrevalenceError, match="changed after endpoint freeze"):
        _generate(chain, loader=lambda _: calls.append(True))
    assert calls == []


def test_report_path_is_immutable(tmp_path):
    chain = _chain(tmp_path)
    output = chain[0] / "artifacts/labels/synthetic/report.json"
    _generate(chain, output=output)
    payload = json.loads(output.read_text())
    payload["raw_counts"]["positive_rows"] = 99
    output.write_text(json.dumps(payload))
    with pytest.raises(SupportPrevalenceError, match="different bytes"):
        _generate(chain, output=output)


def test_governance_disables_prevalence_gate_and_class_weight_fit():
    config = json.loads((ROOT / "configs/support_prevalence_v1.json").read_text())
    assert config["real_authorized_populations"] == []
    assert config["headline_prevalence_enabled"] is False
    assert config["prevalence_range_acceptance_gate"] is False
    assert config["class_weight_fitting_in_scope"] is False


def test_label_artifact_outside_repository_is_rejected_before_loader(tmp_path):
    chain = _chain(tmp_path)
    outside = tmp_path / "outside-labels.json"
    shutil.copy2(chain[3], outside)
    calls = []
    with pytest.raises(SupportPrevalenceError, match="inside the repository"):
        _generate(
            chain,
            label_artifact_path=outside,
            label_artifact_sha256=sha256_file(outside),
            loader=lambda _: calls.append(True),
        )
    assert calls == []


def test_reporting_source_has_no_endpoint_mutation_or_prevalence_threshold():
    source = (ROOT / "src/labels/support_prevalence.py").read_text().lower()
    assert "write_event_dictionary" not in source
    assert "min_prevalence" not in source
    assert "max_prevalence" not in source
