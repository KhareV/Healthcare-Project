from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_runbook_reference_existing_entrypoints_and_docs():
    readme = (ROOT / "README.md").read_text()
    runbook = (ROOT / "RUNBOOK.md").read_text()
    for path in (
        "observed_environment_phase15.json",
        "docs/REPRODUCIBILITY.md",
        "docs/pulkit/PHASE15_PACKAGING_REVIEW.md",
        "data/demo/demo_patient_v1.json",
        "api/main.py",
        "dashboard/app.py",
    ):
        assert (ROOT / path).exists(), path
    for required in (
        "demo.fixture validate", "pytest", "api.main:app", "dashboard.app:app",
        "registry_cli audit", "Ctrl-C", "MIMIC-IV v2.2",
    ):
        assert required in readme + runbook
    assert "blocked" in (readme + runbook).lower()


def test_runbook_contains_no_claimed_final_install_or_live_demo_success():
    text = (ROOT / "RUNBOOK.md").read_text().lower()
    assert "no final dependency lock exists" in text
    assert "live prediction smoke is blocked" in text
    assert "packaged t1/t2/t3 live replay remains blocked" in text
    assert "pip install -r" not in text


def test_gitignore_protects_local_sensitive_and_environment_paths():
    values = set((ROOT / ".gitignore").read_text().splitlines())
    for expected in (".venv/", "venv/", ".env", "data/raw/", "data/mimic*/", ".cache/"):
        assert expected in values
    assert "data/demo/" not in values
