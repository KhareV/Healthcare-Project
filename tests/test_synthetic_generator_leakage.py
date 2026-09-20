import copy, json
from pathlib import Path
import pytest, yaml
from data.synthetic.config import SyntheticConfigError, load_runtime_config
from data.synthetic.generator import generate
from data.synthetic.validation import SyntheticValidationError, validate_dataset

ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/"configs/synthetic/fixtures/phase3_fixture_v1.json"

def bundle():
    config=load_runtime_config(CONFIG,ROOT); rows=generate(config,ROOT)
    schema=json.loads((ROOT/"configs/synthetic/synthetic_schema_v1.json").read_text()); inventory=yaml.safe_load((ROOT/"configs/synthetic/synthetic_generator_v1.yaml").read_text())["raw_variable_inventory"]
    return [copy.deepcopy(x) for x in rows],schema,inventory

@pytest.mark.parametrize("field",["trajectory","split","delta_sofa_24","target","latent_state"])
def test_forbidden_or_unclassified_subject_fields_fail_closed(field):
    rows,schema,inventory=bundle(); rows[0][0][field]="hostile"
    with pytest.raises(SyntheticValidationError,match="forbidden/unclassified"): validate_dataset(*rows,schema,inventory)

def test_event_after_outtime_fails_closed():
    rows,schema,inventory=bundle(); rows[2][0]["event_time"]="2199-01-01T00:00:00.000000Z"
    with pytest.raises(SyntheticValidationError,match="outside episode"): validate_dataset(*rows,schema,inventory)

def test_invalid_equal_episode_boundary_fails_closed():
    rows,schema,inventory=bundle(); rows[1][0]["outtime"]=rows[1][0]["intime"]
    with pytest.raises(SyntheticValidationError,match="strictly after"): validate_dataset(*rows,schema,inventory)

def test_duplicate_stay_fails_closed():
    rows,schema,inventory=bundle(); rows[1][1]["stay_id"]=rows[1][0]["stay_id"]
    with pytest.raises(SyntheticValidationError,match="duplicate stay_id"): validate_dataset(*rows,schema,inventory)

def test_outtime_role_hostile_change_fails_closed():
    rows,schema,inventory=bundle(); field=next(x for x in schema["tables"]["episodes"]["fields"] if x["name"]=="outtime"); field["role"]="MODEL_ELIGIBLE_RAW"; field["model_eligible"]=True
    with pytest.raises(SyntheticValidationError,match="outtime"): validate_dataset(*rows,schema,inventory)

def test_final_mode_never_falls_back(tmp_path):
    payload=json.loads(CONFIG.read_text()); payload["mode"]="final"; target=tmp_path/"final.json"; target.write_text(json.dumps(payload))
    with pytest.raises(SyntheticConfigError,match="FINAL mode refused"): load_runtime_config(target,ROOT)

def test_generator_source_has_no_model_or_preprocessor_imports():
    source="\n".join(p.read_text() for p in (ROOT/"src/data/synthetic").glob("*.py"))
    assert "import torch" not in source and "import xgboost" not in source and "import sklearn" not in source
