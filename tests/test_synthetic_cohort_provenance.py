import json, shutil, tempfile
from pathlib import Path
import pytest, yaml
from data.synthetic.cohort import SyntheticCohortError
from data.synthetic.cohort_manifest import build_to_directory, validate_cohort_manifest

ROOT=Path(__file__).resolve().parents[1]; SPEC=ROOT/"configs/synthetic/cohort_spec_v2.yaml"; SOURCE=ROOT/"artifacts/data/synthetic/fixtures/phase3_fixture_v1/synthetic_dataset_manifest_v1.json"

def make_output(source=SOURCE,spec=SPEC,mode="engineering",stamp="2026-09-20T17:00:00Z"):
    parent=Path(tempfile.mkdtemp(prefix="phase4-test-",dir=ROOT)); output=parent/"cohort"
    manifest=build_to_directory(source,spec,output,ROOT,mode=mode,generation_command="pytest phase4",generation_timestamp=stamp)
    return parent,output,manifest

def test_engineering_manifest_and_artifacts_validate():
    parent,output,manifest=make_output()
    try:
        value=validate_cohort_manifest(manifest,ROOT,SPEC)
        assert value["manifest_status"]=="ENGINEERING_COHORT_FIXTURE" and value["structural_index_status"]=="ENGINEERING_OUTCOME_FREE_STRUCTURAL_INDEX_FIXTURE"
        assert value["split_status"]=="NOT_ASSIGNED_PHASE_10"
    finally: shutil.rmtree(parent)

def test_rebuild_has_identical_data_artifact_hashes():
    first=make_output(); second=make_output()
    try:
        a=json.loads(first[2].read_text()); b=json.loads(second[2].read_text())
        chosen={"retained_cohort","structural_index","exclusion_summary","cohort_summary"}
        assert [(x["logical_name"],x["sha256"],x["semantic_sha256"]) for x in a["artifacts"] if x["logical_name"] in chosen]==[(x["logical_name"],x["sha256"],x["semantic_sha256"]) for x in b["artifacts"] if x["logical_name"] in chosen]
    finally: shutil.rmtree(first[0]); shutil.rmtree(second[0])

def test_final_mode_rejects_fixture_source_without_output():
    parent=Path(tempfile.mkdtemp(prefix="phase4-final-test-",dir=ROOT)); output=parent/"cohort"
    try:
        with pytest.raises(SyntheticCohortError,match="AUTHORIZED_FINAL_SYNTHETIC_DATA"): build_to_directory(SOURCE,SPEC,output,ROOT,mode="final",generation_command="pytest")
        assert not output.exists()
    finally: shutil.rmtree(parent)

def test_source_manifest_tamper_is_rejected():
    parent=Path(tempfile.mkdtemp(prefix="phase4-source-test-",dir=ROOT)); altered=parent/"source.json"; payload=json.loads(SOURCE.read_text()); payload["artifacts"][0]["sha256"]="0"*64; altered.write_text(json.dumps(payload)); output=parent/"cohort"
    try:
        with pytest.raises(Exception,match="hash mismatch"): build_to_directory(altered,SPEC,output,ROOT,mode="engineering",generation_command="pytest")
    finally: shutil.rmtree(parent)

def test_cohort_spec_tamper_invalidates_existing_manifest():
    parent,output,manifest=make_output(); altered=parent/"cohort.yaml"; value=yaml.safe_load(SPEC.read_text()); value["population"]["minimum_age_years"]=19; altered.write_text(yaml.safe_dump(value))
    try:
        with pytest.raises(SyntheticCohortError,match="cohort_spec_sha256 mismatch"): validate_cohort_manifest(manifest,ROOT,altered)
    finally: shutil.rmtree(parent)

def test_timestamp_spec_tamper_is_rejected():
    parent=Path(tempfile.mkdtemp(prefix="phase4-clock-test-",dir=ROOT)); altered=parent/"cohort.yaml"; value=yaml.safe_load(SPEC.read_text()); value["timestamp_contract"]["sha256"]="0"*64; altered.write_text(yaml.safe_dump(value)); output=parent/"cohort"
    try:
        with pytest.raises(SyntheticCohortError,match="timestamp spec hash mismatch"): build_to_directory(SOURCE,altered,output,ROOT,mode="engineering",generation_command="pytest")
    finally: shutil.rmtree(parent)

def test_artifact_tamper_invalidates_cohort_manifest():
    parent,output,manifest=make_output()
    try:
        with (output/"structural_index.jsonl").open("a") as handle: handle.write("{}\n")
        with pytest.raises(SyntheticCohortError,match="artifact missing or hash mismatch"): validate_cohort_manifest(manifest,ROOT,SPEC)
    finally: shutil.rmtree(parent)

def test_output_refuses_overwrite():
    parent,output,manifest=make_output()
    try:
        with pytest.raises(FileExistsError): build_to_directory(SOURCE,SPEC,output,ROOT,mode="engineering",generation_command="pytest")
    finally: shutil.rmtree(parent)
