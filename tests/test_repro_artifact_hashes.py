import pytest

from reproducibility.compare import ReproducibilityComparisonError, verify_file_hash
from vedant_infra.hashing import sha256_file


def test_artifact_reload_requires_exact_hash(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"frozen-model")
    digest = sha256_file(artifact)
    assert verify_file_hash(artifact, digest) == digest
    artifact.write_bytes(b"mutated")
    with pytest.raises(ReproducibilityComparisonError, match="hash mismatch"):
        verify_file_hash(artifact, digest)
