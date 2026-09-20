from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_one_sofa_function_and_no_horizon_specific_scoring_api():
    tree = ast.parse((ROOT / "src/data/synthetic/sofa.py").read_text())
    names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert names.count("sofa_at") == 1
    assert not {"baseline_sofa", "future_sofa", "dashboard_sofa", "serving_sofa", "label_sofa"} & set(names)


def test_phase6_does_not_create_later_phase_artifacts_or_dependencies():
    source = "\n".join((ROOT / path).read_text().lower() for path in (
        "src/data/synthetic/sofa.py", "src/data/synthetic/sofa_provider.py",
    ))
    for forbidden in ("deltasofa24", "deltasofa48", "xgboost", "gru", "lstm", "isotonic", "standardscaler", "train_test_split"):
        assert forbidden not in source


def test_old_proxy_is_not_imported():
    source = (ROOT / "src/data/synthetic/sofa.py").read_text()
    assert "Heart-Failure-Recovery-and-Deterioration-Trajectory-Prediction" not in source
    assert "preprocessing.sofa" not in source
