import json,shutil,tempfile
from pathlib import Path
import pytest
from data.synthetic.normalization import SyntheticTimelineError
from data.synthetic.processed_manifest import build_processed_bundle,validate_processed_manifest
from synthetic_timeline_helpers import COHORT,MAP,ROOT,SCHEMA

def make(mode="engineering"):
    parent=Path(tempfile.mkdtemp(prefix="phase5-test-",dir=ROOT)); out=parent/"bundle"; manifest=build_processed_bundle(COHORT,MAP,SCHEMA,out,ROOT,mode=mode,generation_command="pytest phase5",generation_timestamp="2026-09-20T17:20:00Z"); return parent,out,manifest
def test_processed_manifest_validates():
    p,o,m=make()
    try: assert validate_processed_manifest(m,ROOT)["manifest_status"]=="ENGINEERING_CANONICAL_TIMELINE_FIXTURE"
    finally: shutil.rmtree(p)
def test_repeated_build_has_identical_data_hashes():
    a=make(); b=make()
    try:
        x=json.loads(a[2].read_text()); y=json.loads(b[2].read_text()); assert [(r["sha256"],r["semantic_sha256"]) for r in x["artifacts"]]==[(r["sha256"],r["semantic_sha256"]) for r in y["artifacts"]]
    finally: shutil.rmtree(a[0]); shutil.rmtree(b[0])
def test_final_mode_rejects_engineering_cohort():
    parent=Path(tempfile.mkdtemp(prefix="phase5-final-",dir=ROOT)); out=parent/"bundle"
    try:
        with pytest.raises(SyntheticTimelineError,match="authorized final"): build_processed_bundle(COHORT,MAP,SCHEMA,out,ROOT,mode="final",generation_command="pytest")
        assert not out.exists()
    finally: shutil.rmtree(parent)
def test_artifact_tamper_rejected():
    p,o,m=make()
    try:
        with (o/"canonical_timeline.jsonl").open("a") as h:h.write("{}\n")
        with pytest.raises(SyntheticTimelineError,match="hash mismatch"): validate_processed_manifest(m,ROOT)
    finally: shutil.rmtree(p)
@pytest.mark.parametrize("field",["concept_map_sha256","processed_schema_sha256","provenance_sha256"])
def test_bound_contract_hash_attack_rejected(field):
    p,o,m=make()
    try:
        payload=json.loads(m.read_text()); payload[field]="0"*64; m.write_text(json.dumps(payload))
        with pytest.raises(SyntheticTimelineError,match=field): validate_processed_manifest(m,ROOT)
    finally: shutil.rmtree(p)
def test_output_refuses_overwrite():
    p,o,m=make()
    try:
        with pytest.raises(FileExistsError): build_processed_bundle(COHORT,MAP,SCHEMA,o,ROOT,mode="engineering",generation_command="pytest")
    finally: shutil.rmtree(p)
