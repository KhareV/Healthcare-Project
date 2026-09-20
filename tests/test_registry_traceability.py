from experiments.lineage import artifact_descendants, trace_result, trace_run
from registry_helpers import chain, run


def test_result_to_run_and_reverse_trace(tmp_path):
    selected = run()
    records = chain(tmp_path, selected)
    result = trace_result("metric", records, (selected,))
    assert result["ancestor_artifact_ids"] == ("prediction", "model")
    assert result["producing_run"]["run_id"] == "run-A"
    assert result["configuration_hash"] == selected["config_hash"]
    reverse = trace_run("run-A", records, (selected,))
    assert set(reverse["direct_artifact_ids"]) == {"model", "prediction", "metric"}
    assert artifact_descendants(records, "model") == ("metric", "prediction")
