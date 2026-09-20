import pytest

from evaluation.final_test import BLOCKED_G3, FinalTestError, pretest_audit, run_final_test
from final_test_helpers import write_phase20_config
from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import freeze_g3


def test_real_repository_refuses_before_loader_opens():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    opened = []
    report = pretest_audit(root)
    assert report.overall_status == "BLOCKED"
    assert any(BLOCKED_G3 in item for item in report.blockers)
    with pytest.raises(FinalTestError, match="ACTIVE G3"):
        run_final_test(
            root,
            guarded_runner=lambda callback: callback(),
            test_loader=lambda: opened.append(True),
            evaluator=lambda value: value,
        )
    assert opened == []


@pytest.mark.parametrize("target", ("manifest", "model", "calibrator", "threshold"))
def test_frozen_artifact_mutation_blocks_before_test_access(tmp_path, target):
    paths = build_complete_g3_fixture(tmp_path)
    marker = freeze_g3(
        tmp_path,
        marker_path=tmp_path / "synthetic-marker.json",
        scope="synthetic_test",
    )
    config = write_phase20_config(tmp_path, marker)
    artifact = paths["models"]["recovery"] if target == "model" else paths[target]
    artifact.write_bytes(artifact.read_bytes() + b"mutation")
    opened = []

    with pytest.raises(FinalTestError, match="ACTIVE G3"):
        run_final_test(
            tmp_path,
            guarded_runner=lambda callback: callback(),
            test_loader=lambda: opened.append(True),
            evaluator=lambda value: value,
            config_path=config,
            marker_path=marker,
            expected_scope="synthetic_test",
        )
    assert opened == []
