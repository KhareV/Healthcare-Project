import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from reproducibility.audit import audit_environment_lock, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_unresolved_manager_blocks_final_lock_truthfully():
    manager_files = (
        "pyproject.toml", "poetry.lock", "uv.lock", "requirements.txt",
        "requirements.in", "environment.yml", "setup.cfg", "setup.py",
    )
    assert not any((ROOT / name).exists() for name in manager_files)
    assert not tuple(ROOT.glob("environment.lock.*"))
    result = audit_environment_lock(
        ROOT, load_config(ROOT / "configs/reproducibility_v1.json")
    )
    assert result["status"] == "BLOCKED"
    assert result["lock_ref"] is None


def test_observed_snapshot_matches_runtime_but_is_explicitly_nonfinal():
    snapshot = json.loads((ROOT / "observed_environment_phase15.json").read_text())
    assert snapshot["status"] == "NON_FINAL_DEVELOPMENT_SNAPSHOT"
    assert snapshot["package_manager_status"] == "BLOCKED_UNRESOLVED"
    assert snapshot["final_environment_lock"] is None
    assert snapshot["python"]["implementation"] == platform.python_implementation()
    assert snapshot["python"]["version"] == platform.python_version()
    for distribution, expected in snapshot["packages"].items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        assert actual == expected, distribution
    assert snapshot["platform"]["architecture"] == platform.machine()
    assert snapshot["platform"]["torch_cuda_build"] is None
    assert snapshot["platform"]["cuda_available"] is False


def test_real_stack_missing_dependencies_are_not_hidden():
    snapshot = json.loads((ROOT / "observed_environment_phase15.json").read_text())
    # Stage 4 installed the previously-missing real-stack dependencies (captum
    # for Integrated Gradients, shap for TreeSHAP) into this development
    # environment; the observed snapshot now truthfully records both present.
    assert set(snapshot["missing_required_real_stack_packages"]) == set()
    for distribution in ("captum", "shap"):
        assert snapshot["packages"][distribution] is not None
