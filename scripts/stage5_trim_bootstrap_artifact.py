"""One-off correction, applied before the Part-B push (git push was rejected:
GitHub's 100MB file-size limit; the artifact was 176MB).

persist_bootstrap() originally serialized the full BootstrapResult
dataclass per metric, including bootstrap_distribution (2000 floats) and
sampled_cluster_sequences (2000 tuples of resampled stay ids, each up to
~300 entries) -- exactly reproducible from the frozen prediction artifacts
plus the frozen seed (n_bootstrap=2000, seed=20260921), and never referenced
by point estimate, CI, or valid/invalid-replicate reporting.
src/experiments/stage5_final_test.py's persist_bootstrap() is fixed to omit
both fields going forward. This applies the identical trim to the
already-written artifacts/final_test/bootstrap/bootstrap_v1.json (no
predictions, metrics, point estimates, or CIs are touched -- only the two
oversized, regenerable fields are dropped) and refreshes G4's
bootstrap_artifact hash to match. No test-partition access.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

LARGE_FIELDS = ("bootstrap_distribution", "sampled_cluster_sequences")


def _trim(node):
    if isinstance(node, dict):
        if any(key in node for key in LARGE_FIELDS) and "point_estimate" in node:
            return {key: _trim(value) for key, value in node.items() if key not in LARGE_FIELDS}
        return {key: _trim(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_trim(item) for item in node]
    return node


def main() -> None:
    bootstrap_path = ROOT / "artifacts/final_test/bootstrap/bootstrap_v1.json"
    before_size = bootstrap_path.stat().st_size
    payload = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    payload["omitted_from_this_artifact"] = list(LARGE_FIELDS)
    payload["omitted_field_reproducibility"] = (
        "Exactly reproducible from the frozen prediction artifacts in "
        "artifacts/final_test/predictions/ using this same n_bootstrap and "
        "seed; not stored here to keep this artifact a reasonable size."
    )
    trimmed = _trim(payload)
    bootstrap_path.write_bytes(canonical_json_bytes(trimmed))
    after_size = bootstrap_path.stat().st_size
    new_hash = sha256_file(bootstrap_path)
    print("before_bytes=" + str(before_size))
    print("after_bytes=" + str(after_size))
    print("new_bootstrap_sha256=" + new_hash)

    g4_path = ROOT / "artifacts/governance/g4_test_evaluation_freeze_v1.json"
    g4_payload = json.loads(g4_path.read_text())
    g4_payload["bootstrap_artifact"]["sha256"] = new_hash
    g4_payload["registry"]["artifacts_sha256"] = sha256_file(ROOT / "experiments/artifacts.csv")
    g4_payload["registry"]["registry_sha256"] = sha256_file(ROOT / "experiments/registry.csv")
    g4_path.write_bytes(canonical_json_bytes(g4_payload))
    print("g4_bootstrap_sha256=" + g4_payload["bootstrap_artifact"]["sha256"])
    print("g4_file_sha256=" + sha256_file(g4_path))


if __name__ == "__main__":
    main()
