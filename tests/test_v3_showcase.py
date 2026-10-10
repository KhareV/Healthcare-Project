import copy
import hashlib
import json
from pathlib import Path

import pytest

from serving.v3 import showcase as S

ROOT = Path(__file__).resolve().parents[1]
HAVE = (ROOT / S.MANIFEST_REL).exists() and S.cohort_paths(ROOT)["timeline"].exists()
needs = pytest.mark.skipif(not HAVE, reason="LCE cohort / showcase manifest (gitignored artifact) absent")


@pytest.fixture(scope="module")
def manifest():
    return S.load_showcase(ROOT)


@needs
def test_manifest_integrity_and_counts(manifest):
    path = ROOT / S.MANIFEST_REL
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".json.sha256").read_text().split()[0]
    scn = manifest["scenarios"]
    assert manifest["n_scenarios"] == len(scn) == 8
    assert len({s["stay_id"] for s in scn}) == 8
    assert all(s["status"] == S.STATUS and s["n_cutoffs"] >= 5 and s["source_cohort"] == "LCE" for s in scn)
    assert not any("prediction" in k for s in scn for k in s)
    assert (ROOT / "configs/performance_v2/v2_demo_manifest_v1.json").exists()


@needs
def test_every_stay_is_lce_and_not_forbidden(manifest):
    for s in manifest["scenarios"]:
        S.assert_allowed(s["stay_id"], s["subject_id"])


@pytest.mark.parametrize("stay,subj", [("SYN-E-00000010", "SYN-S-00000010"), ("SYN-V2-E-00000001", "SYN-V2-S-00000001"),
                                       ("SYN-V3-T-E-00000001", "SYN-V3-T-S-00000001"), ("SYN-V3-D1-E-00000001", "SYN-V3-D1-S-00000001"),
                                       ("CUSTOM-ABC", "CUSTOM-SUBJ-ABC")])
def test_exclusion_guard_refuses_other_cohorts(stay, subj):
    with pytest.raises(S.ExclusionError):
        S.assert_allowed(stay, subj)


@needs
def test_loader_rejects_injected_forbidden_stay(manifest, tmp_path):
    bad = copy.deepcopy({k: v for k, v in manifest.items() if k != "_sha256"})
    bad["scenarios"][0]["stay_id"] = "SYN-V3-T-E-00000001"
    bad["scenarios"][0]["subject_id"] = "SYN-V3-T-S-00000001"
    root = tmp_path
    p = root / S.MANIFEST_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(bad))
    p.with_suffix(".json.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "  x\n")
    S._load_cached.cache_clear()
    with pytest.raises(S.ExclusionError):
        S.load_showcase(root, verify_cutoffs=False)
    S._load_cached.cache_clear()


@needs
def test_tampered_manifest_is_rejected(tmp_path):
    p = tmp_path / S.MANIFEST_REL
    p.parent.mkdir(parents=True)
    p.write_bytes((ROOT / S.MANIFEST_REL).read_bytes() + b" ")
    p.with_suffix(".json.sha256").write_text((ROOT / S.MANIFEST_REL).with_suffix(".json.sha256").read_text())
    S._load_cached.cache_clear()
    with pytest.raises(S.ShowcaseError):
        S.load_showcase(tmp_path, verify_cutoffs=False)
    S._load_cached.cache_clear()


@needs
def test_cutoffs_equal_canonical_generator(manifest):
    from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay

    for s in manifest["scenarios"]:
        stay = RetainedICUStay(subject_id=s["subject_id"], stay_id=s["stay_id"], intime=S._dt(s["intime"]), outtime=S._dt(s["outtime"]))
        assert [S.iso_s(r.prediction_time) for r in generate_prediction_rows_for_stay(stay)] == s["legal_cutoffs"]
        assert all(S._dt(c) <= S._dt(s["outtime"]) for c in s["legal_cutoffs"])


@needs
def test_timeline_visibility_never_leaks_future(manifest):
    for s in manifest["scenarios"][:3]:
        tl = S.scenario_timeline(ROOT, s["scenario_id"])
        cuts = tl["cutoff_hours"]
        for ser in tl["series"]:
            for p in ser["points"]:
                k = p["first_visible_cutoff"]
                if k is not None:
                    assert p["hours_since_admission"] <= cuts[k] + 1e-6
                    assert k == 0 or p["hours_since_admission"] > cuts[k - 1] - 1e-6
                else:
                    assert p["hours_since_admission"] > cuts[-1]
        for r in tl["support_runs"]:
            k = r["first_visible_cutoff"]
            assert k is None or r["start_hour"] <= cuts[k] + 1e-6
            e = r["end_first_visible_cutoff"]
            assert e is None or r["end_hour"] <= cuts[e] + 1e-6


@needs
def test_story_facts_use_only_data_up_to_cutoff_and_hindsight_is_labelled(manifest):
    s = manifest["scenarios"][5]
    story = S.scenario_story(ROOT, s["scenario_id"])
    assert len(story["steps"]) == s["n_cutoffs"]
    for step in story["steps"]:
        assert all("hindsight" not in f for f in step["facts"])
        assert all(r["hindsight"] is True for r in step["revealed_next"])
        txt = " ".join(f["text"] for f in step["facts"]).lower()
        assert "treat" not in txt and "recommend" not in txt
    cf = s["cutoff_facts"][0]
    assert f"{cf['sofa_t']:g}" in " ".join(f["text"] for f in story["steps"][0]["facts"])


@needs
def test_support_initiation_scenario_has_qualifying_onset_after_early_cutoff(manifest):
    s = next(x for x in manifest["scenarios"] if x["category"] == "support_initiation")
    onsets = s["measured_criteria"]["qualifying_onsets"]
    assert onsets and onsets[0]["hours_since_admission"] > s["cutoff_facts"][1]["icu_elapsed_hours"]


class StubRuntime:
    def __init__(self):
        self.stays, self.support = {}, {}

    def ephemeral_entry(self, stay_id):
        return self.stays.get(stay_id)

    def register_ephemeral_stay(self, *, stay_id, subject_id, statics_row, events, legal_cutoffs, owner_user_id):
        assert owner_user_id is None
        self.stays[stay_id] = {"stay_id": stay_id, "subject_id": subject_id, "legal_cutoffs": list(legal_cutoffs), "events": len(events), "statics": statics_row}

    def register_ephemeral_support(self, *, stay_id, support_rows):
        self.support[stay_id] = len(support_rows)


@needs
def test_registration_with_stub_runtime_is_idempotent_and_complete(manifest):
    rt = StubRuntime()
    ids = S.register_showcase(rt, ROOT)
    assert len(ids) == 8 and set(rt.stays) == {s["stay_id"] for s in manifest["scenarios"]}
    assert all(v["events"] > 50 for v in rt.stays.values())
    assert all(rt.stays[s["stay_id"]]["legal_cutoffs"] == s["legal_cutoffs"] for s in manifest["scenarios"])
    assert S.register_showcase(rt, ROOT) == ids


@needs
def test_router_endpoints_and_compare():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.showcase_router import build_showcase_router

    rt = StubRuntime()
    S.register_showcase(rt, ROOT)
    app = FastAPI()
    app.include_router(build_showcase_router(ROOT, rt))
    c = TestClient(app)
    r = c.get("/showcase/scenarios").json()
    assert len(r["scenarios"]) == 8 and all(s["registered_for_inference"] for s in r["scenarios"])
    assert c.get("/showcase/scenarios/SCN-99").status_code == 404
    assert c.get("/showcase/scenarios/SCN-01/timeline").status_code == 200
    assert c.get("/showcase/scenarios/SCN-01/story").status_code == 200
    cmp_ = c.get("/showcase/compare", params={"a": "SCN-01", "b": "SCN-02"}).json()
    assert cmp_["aligned_cutoffs"] == min(r["scenarios"][0]["n_cutoffs"], r["scenarios"][1]["n_cutoffs"])
    assert c.get("/showcase/compare", params={"a": "SCN-01", "b": "SCN-01"}).status_code == 422
