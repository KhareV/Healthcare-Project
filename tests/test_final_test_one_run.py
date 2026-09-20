import pytest

from evaluation.final_test import FinalTestError, run_final_test
from final_test_helpers import write_phase20_config
from g3_helpers import build_complete_g3_fixture
from vedant_infra.g3 import freeze_g3, guarded_test_access


def test_one_guarded_final_run_and_no_training_callback(tmp_path):
    build_complete_g3_fixture(tmp_path)
    marker = freeze_g3(
        tmp_path,
        marker_path=tmp_path / "synthetic-marker.json",
        scope="synthetic_test",
    )
    config = write_phase20_config(tmp_path, marker)
    calls = {"loader": 0, "evaluator": 0, "training": 0}

    def loader():
        calls["loader"] += 1
        return "frozen-predictions"

    def evaluator(value):
        calls["evaluator"] += 1
        return value

    runner = lambda callback: guarded_test_access(
        tmp_path,
        callback,
        marker_path=marker,
        expected_scope="synthetic_test",
    )
    assert run_final_test(
        tmp_path,
        guarded_runner=runner,
        test_loader=loader,
        evaluator=evaluator,
        config_path=config,
        marker_path=marker,
        expected_scope="synthetic_test",
    ) == "frozen-predictions"
    assert calls == {"loader": 1, "evaluator": 1, "training": 0}
    with pytest.raises(FinalTestError, match="ACTIVE G3"):
        run_final_test(
            tmp_path,
            guarded_runner=runner,
            test_loader=loader,
            evaluator=evaluator,
            config_path=config,
            marker_path=marker,
            expected_scope="synthetic_test",
        )
    assert calls["loader"] == 1
