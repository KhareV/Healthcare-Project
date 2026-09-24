import pytest

from evaluation.final_test import BLOCKED_G3, FinalTestError, pretest_audit, run_final_test
from final_test_helpers import write_phase20_config
from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import freeze_g3


def test_real_repository_refuses_before_loader_opens():
    """The real repository's actual current state decides the expected
    outcome here, in three legitimate phases of this same project:

    - before any G3 freeze existed: BLOCKED on active_g3.
    - after Stage-5 Part A froze the pre-test plan/config but before the
      one-time test access is consumed: PASS — the guard now legitimately
      allows the *dummy* loader/evaluator below to run (they touch no real
      data and mutate no real state), proving the framework is unblocked.
    - after the one-time final-test access is consumed: BLOCKED again,
      always on test_access_state (an ordinary second run is refused), and
      legitimately also on active_g3 once any final-test result has been
      registered into experiments/artifacts.csv/registry.csv, since those
      files are G3-bound dependencies and registering the scientific result
      is exactly what the frozen final-test evaluator does. This "test
      nonuse readiness no longer passing" is expected post-test, not
      corruption — G4 (artifacts/governance/g4_test_evaluation_freeze_v1.json)
      is the authoritative governance record from that point forward.

    In every phase, the dummy loader/evaluator below never open real test
    data and the dummy guarded_runner never touches the real access-state
    file, so running this test is always safe regardless of phase.
    """
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    opened = []
    report = pretest_audit(root)
    marker_active = (root / "artifacts/governance/g3_freeze.json").exists()
    access_state_path = root / "artifacts/governance/test_access_state.json"
    access_consumed = (
        access_state_path.is_file()
        and __import__("json").loads(access_state_path.read_text()).get("state") != "AUTHORIZED_NOT_RUN"
    )

    if not marker_active:
        assert report.overall_status == "BLOCKED"
        assert any(BLOCKED_G3 in item for item in report.blockers)
    elif access_consumed:
        assert report.overall_status == "BLOCKED"
        assert any("test_access_state" in item for item in report.blockers)
    else:
        assert report.overall_status == "PASS"
        assert not report.blockers

    def invoke():
        return run_final_test(
            root,
            guarded_runner=lambda callback: callback(),
            test_loader=lambda: opened.append(True) or "dummy-loaded",
            evaluator=lambda value: value,
        )

    if report.overall_status == "PASS":
        assert invoke() == "dummy-loaded"
        assert opened == [True]
    else:
        with pytest.raises(FinalTestError, match="BLOCKED"):
            invoke()
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
