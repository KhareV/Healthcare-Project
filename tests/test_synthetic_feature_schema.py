import json
from dataclasses import replace
import pytest

from data.schema import TensorContractError, validate_canonical_feature_input
from features.manifest import validate_feature_manifest
from features.synthetic import FeatureBuilderError, load_feature_schema
from synthetic_feature_helpers import ROOT, SCHEMA, builder, context

def test_schema_freezes_exact_f_s_order_and_sentinel():
    payload,ref=load_feature_schema(SCHEMA,ROOT)
    assert ref.feature_dim==15==payload["shape"]["feature_dimension"]
    assert ref.feature_names[0]=="pao2__latest" and ref.feature_names[-1]=="lactate__latest"
    assert ref.static_feature_names==("age_years","sex_category","cardiac_condition_group")
    assert ref.tslo_no_observation_value==54.0
    assert payload["support_features"]["included"] is False and payload["sofa_feature"]["included"] is False

def test_feature_schema_hash_and_order_attacks_fail():
    item=builder().build(context())
    schema=builder().feature_schema
    with pytest.raises(TensorContractError,match="hash"):
        validate_canonical_feature_input(replace(item,feature_schema_sha256="0"*64),schema)
    with pytest.raises(TensorContractError):
        validate_canonical_feature_input(item,replace(schema,feature_names=tuple(reversed(schema.feature_names))))

def test_prohibited_static_model_field_fails_closed(tmp_path):
    payload=json.loads(SCHEMA.read_text()); payload["static_contract"]["ordered_names"]=["age_years","outtime","cardiac_condition_group"]
    target=tmp_path/"schema.json"; target.write_text(json.dumps(payload))
    with pytest.raises(FeatureBuilderError,match="prohibited"):
        builder_instance=__import__("features.synthetic",fromlist=["SyntheticCanonicalFeatureBuilder"]).SyntheticCanonicalFeatureBuilder(schema_path=target,root=ROOT,statics_by_stay={"A":{}}); builder_instance.build(context())

def test_embedded_order_missing_extra_and_padding_tslo_attacks_fail():
    item=builder().build(context())
    schema=builder().feature_schema
    with pytest.raises(TensorContractError,match="order"):
        validate_canonical_feature_input(replace(item,temporal_feature_names=item.temporal_feature_names[:-1]),schema)
    hostile=[list(row) for row in item.tslo_hours]; hostile[0][0]=0.0
    padded=builder().build(context(intime=__import__("synthetic_feature_helpers").T-__import__("datetime").timedelta(hours=24)))
    hostile=[list(row) for row in padded.tslo_hours]; hostile[0][0]=0.0
    with pytest.raises(TensorContractError,match="padded"):
        validate_canonical_feature_input(replace(padded,tslo_hours=tuple(tuple(x) for x in hostile)),schema)

@pytest.mark.parametrize("kind",["fixtures","smoke"])
def test_generated_manifest_hashes_and_receiver_records_validate(kind):
    path=ROOT/f"artifacts/data/synthetic/features/{kind}/phase7_{'fixture' if kind=='fixtures' else 'smoke'}_v1/synthetic_feature_input_manifest_v1.json"
    manifest=validate_feature_manifest(path,ROOT)
    assert manifest["F"]==15 and manifest["raw_S"]==3
    assert manifest["labels"]=="NOT_CREATED" and manifest["split"]=="NOT_ASSIGNED" and manifest["preprocessor"]=="NOT_FIT"
