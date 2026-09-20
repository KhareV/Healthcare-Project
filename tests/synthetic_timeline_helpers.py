import json
from pathlib import Path
from data.synthetic.processed_manifest import _load_upstream
from data.synthetic.normalization import normalize_events
ROOT=Path(__file__).resolve().parents[1]
COHORT=ROOT/"artifacts/data/synthetic/cohorts/fixtures/phase4_fixture_v2/synthetic_cohort_manifest_v2.json"
MAP=ROOT/"configs/synthetic/concept_map_v1.json"; SCHEMA=ROOT/"configs/synthetic/processed_schema_v1.json"
def upstream(): return _load_upstream(COHORT,ROOT)
def normalized():
    cohort,phase3,ct,rt=upstream(); mapping=json.loads(MAP.read_text())
    return normalize_events(rt["raw_events"],ct["retained_cohort"],mapping,phase3["raw_schema_version"])[0]
