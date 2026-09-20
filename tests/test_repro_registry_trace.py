from pathlib import Path

from experiments.lineage import read_artifact_index, read_run_registry, trace_artifact
from reproducibility.audit import audit_registered_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_registered_artifacts_rehash_and_trace_without_owner_knowledge():
    assert audit_registered_artifacts(ROOT)["status"] == "PASS"
    artifacts = read_artifact_index(ROOT / "experiments/artifacts.csv")
    target = next(item for item in artifacts if item.artifact_id.startswith("phase20:pretest:"))
    traced = trace_artifact(
        target.artifact_id,
        artifacts,
        read_run_registry(ROOT / "experiments/registry.csv"),
    )
    assert traced["artifact"]["artifact_sha256"] == target.artifact_sha256
