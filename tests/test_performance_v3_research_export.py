import hashlib
import json
import shutil
from pathlib import Path

import pytest

from performance_v3 import research_export as R

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts/performance_v3/phase4/final_metrics_v3.json").exists(),
                                reason="V3 result artifacts absent")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    out = []
    for name in ("a", "b"):
        rel = f"artifacts/performance_v3/_research_test_{name}"
        ex = R.run_all(ROOT, rel, write_docs=False)
        out.append((rel, ex))
    yield out
    for rel, _ in out:
        shutil.rmtree(ROOT / rel, ignore_errors=True)


def test_runs_are_deterministic(two_runs):
    (ra, ea), (rb, eb) = two_runs
    side = lambda ex, rel: {e["file"].replace(rel, "X"): (e["file_sha256"], [s["sha256"] for s in e["sources"]]) for e in ex.entries}
    assert side(ea, ra) == side(eb, rb)
    assert len(ea.entries) > 40


def test_every_figure_and_table_has_sidecar_with_sources(two_runs):
    rel, ex = two_runs[0]
    base = ROOT / rel
    for path in list((base / "figures").iterdir()) + list((base / "tables").iterdir()) + list((base / "plot_data").iterdir()):
        if path.name.endswith(".meta.json"):
            continue
        side = path.with_name(path.name + ".meta.json")
        assert side.exists(), path
        meta = json.loads(side.read_text())
        assert meta["file_sha256"] == _sha(path)
        assert meta["model_bundle_id"] == R.BUNDLE_ID and meta["code_commit"] and meta["sources"]
        for s in meta["sources"]:
            assert _sha(ROOT / s["path"]) == s["sha256"]


def test_registry_hashes_match_files_and_invalidated_are_flagged():
    entries = R.registry_entries(ROOT)
    assert entries
    for e in entries:
        assert _sha(ROOT / e["artifact_ref"]) == e["sha256"]
        assert e["research_question"] in R.RQ and e["code_commit"] and "experiment_id" in e
    invalid = [e for e in entries if e["status"] == "INVALIDATED_D1"]
    assert invalid and all("INVALIDATED" in e["artifact_ref"] or "invalidated" in e["artifact_ref"] for e in invalid)
    assert {e["research_question"] for e in entries} >= {"RQ1", "RQ3", "RQ4", "RQ5"}


def test_historical_values_are_labelled_and_unmodified():
    assert R.HISTORICAL["V2_final_test"]["icu_median_ae_hours"][0] == 5.2060
    assert R.HISTORICAL["V1_final_test"]["support_calibrated_auprc"][0] == 0.6337
