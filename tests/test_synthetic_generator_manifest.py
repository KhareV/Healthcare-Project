import copy, json, shutil, tempfile
from pathlib import Path
import pytest
from data.synthetic.config import RuntimeConfig, load_runtime_config
from data.synthetic.generator import generate_to_directory
from data.synthetic.manifest import validate_manifest
from data.synthetic.validation import SyntheticValidationError

ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/"configs/synthetic/fixtures/phase3_fixture_v1.json"; SCHEMA=ROOT/"configs/synthetic/synthetic_schema_v1.json"

def make_output():
    parent=Path(tempfile.mkdtemp(prefix="phase3-test-",dir=ROOT)); output=parent/"dataset"
    manifest=generate_to_directory(CONFIG,output,ROOT,"pytest phase3 fixture","2026-09-20T16:45:00Z")
    return parent,output,manifest

def test_manifest_verifies_all_artifacts_and_counts():
    parent,output,manifest=make_output()
    try:
        value=validate_manifest(manifest,ROOT,load_runtime_config(CONFIG,ROOT),SCHEMA)
        assert value["manifest_status"]=="FIXTURE" and value["record_counts"]["support_intervals"]==0
    finally: shutil.rmtree(parent)

def test_regeneration_produces_equal_table_bytes_and_semantic_hashes():
    first=make_output(); second=make_output()
    try:
        a=json.loads(first[2].read_text()); b=json.loads(second[2].read_text())
        assert [(x["sha256"],x["semantic_sha256"]) for x in a["artifacts"]] == [(x["sha256"],x["semantic_sha256"]) for x in b["artifacts"]]
    finally: shutil.rmtree(first[0]); shutil.rmtree(second[0])

def test_manifest_detects_artifact_tampering():
    parent,output,manifest=make_output()
    try:
        with (output/"subjects.jsonl").open("a") as handle: handle.write("{}\n")
        with pytest.raises(SyntheticValidationError,match="hash mismatch"): validate_manifest(manifest,ROOT)
    finally: shutil.rmtree(parent)

def test_manifest_detects_config_hash_mismatch():
    parent,output,manifest=make_output(); config=load_runtime_config(CONFIG,ROOT); changed=RuntimeConfig(config.path,copy.deepcopy(config.values),"0"*64)
    try:
        with pytest.raises(SyntheticValidationError,match="config hash mismatch"): validate_manifest(manifest,ROOT,changed,SCHEMA)
    finally: shutil.rmtree(parent)

def test_manifest_detects_schema_hash_mismatch():
    parent,output,manifest=make_output(); altered=parent/"schema.json"; altered.write_text(SCHEMA.read_text()+"\n")
    try:
        with pytest.raises(SyntheticValidationError,match="schema hash mismatch"): validate_manifest(manifest,ROOT,load_runtime_config(CONFIG,ROOT),altered)
    finally: shutil.rmtree(parent)

def test_output_refuses_overwrite():
    parent,output,manifest=make_output()
    try:
        with pytest.raises(FileExistsError): generate_to_directory(CONFIG,output,ROOT,"pytest","2026-09-20T16:45:00Z")
    finally: shutil.rmtree(parent)
