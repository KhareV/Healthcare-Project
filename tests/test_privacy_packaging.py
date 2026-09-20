import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ROOT / "README.md",
    ROOT / "RUNBOOK.md",
    ROOT / "observed_environment_phase15.json",
    ROOT / "data/demo/demo_patient_v1.json",
    ROOT / "data/demo/demo_patient_v1.metadata.json",
)


def test_packaging_operational_files_have_no_personal_absolute_paths():
    patterns = (
        re.compile(r"/Users/[^/\s]+/"),
        re.compile(r"/home/[^/\s]+/"),
        re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    )
    for path in FILES:
        text = path.read_text()
        assert not any(pattern.search(text) for pattern in patterns), path


def test_demo_and_packaging_files_have_no_credential_assignments():
    secret = re.compile(
        r"(?i)(password|api[_-]?token|access[_-]?token|api[_-]?key)\s*[=:]\s*[^<\s]+"
    )
    for path in FILES:
        assert not secret.search(path.read_text()), path


def test_demo_has_no_raw_mimic_field_names_or_binary_exports():
    demo = ROOT / "data/demo"
    assert {path.suffix for path in demo.iterdir()} == {".json"}
    text = "\n".join(path.read_text().lower() for path in demo.iterdir())
    for prohibited in ('"itemid"', '"hadm_id"', '"caregiver_id"', '"patient_name"'):
        assert prohibited not in text
