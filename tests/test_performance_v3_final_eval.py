import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from performance_v3 import final_eval as fe
from performance_v3.data import CohortAccessError, CohortDir, load_cohort_rows
from performance_v3.serving import V3Bundle
from performance_v3.test_access import (
    BLOCKED_ALREADY_CONSUMED, FINAL_RUN_COMPLETED, FINAL_TEST_ACCESS_CONSUMED, SEALED_NOT_ACCESSED,
    SealedCohortAccess, SealedCohortAccessError, _reset_registry_for_tests,
)
from test_performance_v3_features import synthetic_rows
from test_performance_v3_serving import GROUPS, _features, make_bundle_dir

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows(n_stays=30, cutoffs=4, seed=21)


@pytest.fixture(scope="module")
def names(rows):
    return tuple(sorted({n for r in rows for n in _features(r) if n.split("__", 1)[0] in GROUPS}))


def _absolutize(bundle_root: Path, manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["tasks"].values():
        for key in ("model_ref", "feature_names_ref"):
            entry[key] = str((bundle_root / entry[key]).resolve())
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


@pytest.fixture(scope="module")
def bundles(tmp_path_factory, rows, names):
    base = tmp_path_factory.mktemp("bundles")
    out = {}
    for label, seed in (("v3", None), ("control", 5), ("v2stub", 9)):
        d = base / label
        d.mkdir()
        manifest, _ = make_bundle_dir(d, rows, names, seed=1 if label == "v3" else 2 + (seed or 0), recovery24_seed=seed)
        out[label] = (d, _absolutize(d, manifest))
    return out


class V2Stub:
    """Stand-in for V2Comparator: same API, two input modes (None => as evaluated)."""

    def __init__(self, bundle, reverse=False):
        self.bundle = bundle
        self.reverse = reverse

    def predict_rows(self, rows, *, current_sofa_trends=None):
        out = self.bundle.predict_rows(rows, sofa_trends=current_sofa_trends)
        return out[::-1] if self.reverse else out


def _trends(rows):
    return {(r["stay_id"], r["prediction_time"]): {"t": float(r["baseline_sofa"])} for r in rows}


CUT = {"source": "test", "observation_density_tertile_edges": [0.3, 0.5], "remaining_time_quartile_edges_hours": [5.0, 15.0, 40.0],
       "cutoff_bands": {"0-1": [0, 1], "2-4": [2, 4], ">=5": [5, None]}, "n_cache_rows": 0}


class Env:
    def __init__(self, tmp_path, rows, bundles, *, v2_reverse=False):
        self.rows = rows
        self.paths = fe.Paths(ROOT, base=tmp_path / "v3", cohort_dir=tmp_path / "v3/cohorts/T")
        self.paths.v3_bundle = bundles["v3"][1]
        self.paths.control_bundle = bundles["control"][1]
        for rel in ("pre_split/pre_split_scientific_package.jsonl", "timeline/canonical_statics.jsonl"):
            f = self.paths.cohort_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("{}\n")
        self.bundles = bundles
        self.load_states = []
        self.inj = fe.Injectables(rows_loader=self._loader, sofa_fn=lambda rs: (_trends(rs), {}),
                                  v3_loader=lambda: V3Bundle.load(ROOT, self.paths.selected_models),
                                  control_loader=lambda: V3Bundle.load(ROOT, self.paths.control_bundle),
                                  v2_loader=lambda: V2Stub(V3Bundle.load(ROOT, bundles["v2stub"][1]), reverse=v2_reverse),
                                  n_bootstrap=20)

    def _loader(self, root):
        self.load_states.append(json.loads(self.paths.state.read_text())["status"])
        return list(self.rows)

    def freeze(self):
        return fe.freeze(ROOT, self.paths, cutpoints=CUT)

    def init_ledger(self):
        fe.make_access(ROOT, self.paths, lambda r: ()).initialize()


@pytest.fixture()
def env(tmp_path, rows, bundles):
    return Env(tmp_path, rows, bundles)


# ------------------------------------------------------------------ freeze
def test_freeze_binds_every_hash_and_v2_identity(env):
    out = env.freeze()
    record = json.loads(env.paths.freeze.read_text())
    assert record["test_accessed"] is False and record["status"] == "V3_MODELS_FROZEN_PRETEST"
    bound = record["bound_files"]
    assert bound["v2_selected_models"]["sha256"] == fe.V2_SELECTED_SHA
    assert bound["v2_final_evaluation_freeze"]["sha256"] == fe.V2_FINAL_FREEZE_SHA
    assert any(k.startswith("control:") for k in bound) and any(k.startswith("v3:") for k in bound)
    assert "protocol" in bound and "amendment_v1_1" in bound and "selected_models_v3" in bound
    plan = json.loads(env.paths.plan.read_text())
    assert plan["primary_comparator"] == "V2_AS_SERVED"
    assert plan["paired_bootstrap"]["replicates"] == 2000 and plan["paired_bootstrap"]["seed"] == 20261012
    assert json.loads(env.paths.selected_models.read_text())["status"] == "FROZEN_PRETEST"
    assert fe.verify_frozen(ROOT, env.paths) == (out["freeze_sha256"], out["plan_sha256"])
    assert env.paths.freeze.with_suffix(".json.sha256").is_file()


def test_changed_frozen_file_blocks_run_before_any_authorization(env):
    env.freeze()
    env.init_ledger()
    victim = env.bundles["v3"][0] / "organ_support" / "feature_names.json"
    victim.write_text(victim.read_text() + " ")
    try:
        with pytest.raises(fe.FrozenArtifactError):
            fe.run_final(ROOT, env.paths, env.inj)
        assert env.load_states == []
        assert json.loads(env.paths.state.read_text())["status"] == SEALED_NOT_ACCESSED
    finally:
        victim.write_text(victim.read_text().rstrip(" "))


def test_changed_plan_is_detected(env):
    env.freeze()
    env.paths.plan.write_bytes(env.paths.plan.read_bytes().replace(b"FROZEN_BEFORE_TEST_ACCESS", b"EDITED_AFTER_FREEZE_XXXX"))
    with pytest.raises(fe.FrozenArtifactError):
        fe.verify_frozen(ROOT, env.paths)


# ------------------------------------------------------------------ one-time run
def test_run_final_state_machine_artifacts_and_second_run_block(env):
    env.freeze()
    env.init_ledger()
    result = fe.run_final(ROOT, env.paths, env.inj)
    # consumption is recorded BEFORE rows are handed out
    assert env.load_states == [FINAL_TEST_ACCESS_CONSUMED]
    access = fe.make_access(ROOT, env.paths, lambda r: ())
    assert access.state() == FINAL_RUN_COMPLETED and access.effective_state() == BLOCKED_ALREADY_CONSUMED
    for path in (env.paths.final_metrics, env.paths.paired_metrics, env.paths.paired_bootstrap, env.paths.calibration,
                 env.paths.error_analysis, env.paths.final_freeze, env.paths.predictions):
        assert path.is_file(), path
    assert {p.name for p in env.paths.plots.iterdir()} >= {"support_roc.png", "support_pr.png", "support_reliability.png", "paired_differences_forest.png"}
    assert (env.paths.predictions.stat().st_mode & 0o222) == 0  # immutable
    freeze = json.loads(env.paths.final_freeze.read_text())
    assert freeze["successful_scientific_evaluation_count"] == 1 and freeze["predictions_sha256"] == result["predictions_sha256"]
    paired = json.loads(env.paths.paired_metrics.read_text())
    assert set(paired["promotion"]) == set(fe.TASKS)
    assert all(v["decision"] in ("PROMOTE_V3", "RETAIN_V2") for v in paired["promotion"].values())

    guard = fe.second_run_guard(ROOT, env.paths)
    assert guard["blocked"] is True and guard["effective_state"] == BLOCKED_ALREADY_CONSUMED
    assert guard["predictions_sha256_unchanged"] is True
    with pytest.raises(SealedCohortAccessError):
        fe.run_final(ROOT, env.paths, env.inj)
    assert env.load_states == [FINAL_TEST_ACCESS_CONSUMED]  # the second attempt never loaded rows


def test_row_identity_mismatch_writes_incident_and_never_resets(tmp_path, rows, bundles):
    env = Env(tmp_path, rows, bundles, v2_reverse=True)
    env.freeze()
    env.init_ledger()
    with pytest.raises(fe.RowIdentityError):
        fe.run_final(ROOT, env.paths, env.inj)
    incident = json.loads(env.paths.incident.read_text())
    assert incident["stage"] == "persist" and "no scientific retry" in incident["policy"]
    access = fe.make_access(ROOT, env.paths, lambda r: ())
    assert access.state() == FINAL_TEST_ACCESS_CONSUMED and access.effective_state() == BLOCKED_ALREADY_CONSUMED
    assert not env.paths.predictions.exists()
    with pytest.raises(SealedCohortAccessError):
        fe.run_final(ROOT, env.paths, env.inj)


def test_eligible_identity_assertion_fires_when_a_model_lacks_a_prediction(rows, bundles):
    bundle = V3Bundle.load(ROOT, bundles["v3"][1])
    out = bundle.predict_rows(rows, sofa_trends=_trends(rows))
    lines = fe.build_prediction_lines(rows, {m: out for m in fe.MODELS})
    fe.eligible_identity_sets(lines)
    victim = next(l for l in lines if l["labels"]["organ_support_eligible"])
    victim["models"]["CONTROL"]["support_raw_probability"] = None
    with pytest.raises(fe.RowIdentityError):
        fe.eligible_identity_sets(lines)


def test_paired_bootstrap_shares_draws_identical_models_have_zero_difference(rows, bundles):
    bundle = V3Bundle.load(ROOT, bundles["v3"][1])
    out = bundle.predict_rows(rows, sofa_trends=_trends(rows))
    lines = fe.build_prediction_lines(rows, {m: out for m in fe.MODELS})
    plan = fe.make_plan(ROOT, fe.Paths(ROOT), CUT)
    ev = fe.evaluate_predictions(lines, plan, n_bootstrap=25)
    for key, res in ev["paired"].items():
        assert res["diff_point"] == pytest.approx(0.0, abs=1e-12), key
        assert all(abs(v) < 1e-9 for v in res["per_replicate_diff"]), key
        assert res["n_valid_joint"] > 0
    assert all(p["decision"] == "RETAIN_V2" for p in ev["promotion"].values())
    # clean subset only contains horizon-available rows
    assert ev["clean_subset"]["rows"] == sum(1 for l in lines if l["labels"]["recovery24_eligible"])


def test_promotion_rule_requires_ci_exclusion_and_brier_condition():
    paired = {}
    models = {}
    for task in fe.TASKS:
        paired[f"{task}|{fe.PRIMARY_METRIC[task]}|V3_vs_V2_AS_SERVED"] = {
            "diff_point": -0.3 if task != "organ_support" else 0.05, "ci": {"0.9875": [-0.5, -0.1] if task != "organ_support" else [0.01, 0.09]},
            "ci_excludes_zero_9875": True}
    paired["organ_support|brier|V3_vs_V2_AS_SERVED"] = {"diff_point": 0.01, "ci": {}, "ci_excludes_zero_9875": False}
    out = fe._promotion(paired, models)
    assert out["recovery24"]["decision"] == "PROMOTE_V3"
    assert out["organ_support"]["decision"] == "RETAIN_V2" and out["organ_support"]["brier_condition_met"] is False
    paired["organ_support|brier|V3_vs_V2_AS_SERVED"]["diff_point"] = -0.002
    assert fe._promotion(paired, models)["organ_support"]["decision"] == "PROMOTE_V3"
    paired["recovery48|mae|V3_vs_V2_AS_SERVED"]["ci_excludes_zero_9875"] = False
    assert fe._promotion(paired, models)["recovery48"]["decision"] == "RETAIN_V2"


def test_recompute_from_saved_predictions_reproduces_metrics(env):
    env.freeze()
    env.init_ledger()
    fe.run_final(ROOT, env.paths, env.inj)
    first = json.loads(env.paths.final_metrics.read_text())["models"]
    fe.recompute_from_predictions(ROOT, env.paths, n_bootstrap=20)
    assert json.loads(env.paths.final_metrics.read_text())["models"] == first
    env.paths.predictions.chmod(0o644)
    env.paths.predictions.write_text(env.paths.predictions.read_text() + "\n")
    with pytest.raises(fe.FrozenArtifactError):
        fe.load_predictions(env.paths)


# ------------------------------------------------------------------ preflight / guards
def test_preflight_passes_on_dev_fixtures_and_never_opens_sealed_rows(env, rows):
    env.freeze()
    env.init_ledger()
    env.paths.seal.write_text(json.dumps({"verdict": "PASS"}))
    env.load_states.clear()
    record = fe.preflight(ROOT, env.paths, dry_run_rows=rows[:40], dry_run_sofa=(_trends(rows[:40]), {}), inj=env.inj)
    failed = {k: v for k, v in record["checks"].items() if not v["pass"]}
    assert record["verdict"] == "PASS", failed
    assert record["sealed_cohort_rows_opened"] is False and env.load_states == []
    assert json.loads(env.paths.state.read_text())["status"] == SEALED_NOT_ACCESSED


def test_preflight_fails_when_ledger_already_consumed(env, rows):
    env.freeze()
    env.init_ledger()
    fe.run_final(ROOT, env.paths, env.inj)
    env.paths.seal.write_text(json.dumps({"verdict": "PASS"}))
    record = fe.preflight(ROOT, env.paths, dry_run_rows=rows[:20], dry_run_sofa=(_trends(rows[:20]), {}), inj=env.inj)
    assert record["verdict"] == "FAIL" and not record["checks"]["ledger_and_seal"]["pass"]


def test_development_loaders_refuse_the_sealed_cohort(tmp_path):
    t = tmp_path / "cohorts" / "T"
    (t / "pre_split").mkdir(parents=True); (t / "timeline").mkdir(parents=True)
    (t / "timeline/canonical_statics.jsonl").write_text(json.dumps({"stay_id": "SYN-V3-T-E-00000001", "intime": "2110-01-01T00:00:00Z"}) + "\n")
    (t / "pre_split/pre_split_scientific_package.jsonl").write_text(
        json.dumps({"stay_id": "SYN-V3-T-E-00000001", "subject_id": "SYN-V3-T-S-00000001", "prediction_time": "2110-01-02T00:00:00Z", "grid_index": 0}) + "\n")
    with pytest.raises(CohortAccessError):
        load_cohort_rows(ROOT, CohortDir(t), cohort_id="SYN-V3-T", role="dev")
    with pytest.raises(CohortAccessError):
        load_cohort_rows(ROOT, CohortDir(t), cohort_id="SYN-V3-LCE", role="sealed_test")
    with pytest.raises(CohortAccessError):  # even with a development id and role, the namespace prefix is refused
        load_cohort_rows(ROOT, CohortDir(t), cohort_id="SYN-V3-LCE", role="lce")
    fe.make_access(ROOT, fe.Paths(ROOT, base=tmp_path, cohort_dir=t), lambda r: ())
    from performance_v3.test_access import assert_not_sealed
    with pytest.raises(SealedCohortAccessError):
        assert_not_sealed(str(t / "pre_split"))


def test_identity_reports_registration_state(monkeypatch):
    fake = {"cohort_id": "T", "namespace": "SYN-V3-T", "n_subjects": 1500, "role": "sealed_test", "seed": 123456789,
            "seed_material": "m", "seed_material_sha256": "x"}
    monkeypatch.setattr("performance_v3.protocol.cohort_identity", lambda root, cid: dict(fake))
    out = fe.identity(ROOT)
    assert out["registered"] is False and out["final_identities_line"] == '    (123456789, 1500): "SYN-V3-T",'


# ------------------------------------------------------------------ seal (structural only)
def _slice_raw(src: Path, dst: Path, subject_ids):
    dst.mkdir(parents=True, exist_ok=True)
    pattern = re.compile("|".join(re.escape('"' + s + '"') for s in subject_ids))
    for name in ("subjects.jsonl", "episodes.jsonl", "raw_events.jsonl", "support_intervals.jsonl"):
        with (src / name).open() as fin, (dst / name).open("w") as fout:
            for line in fin:
                if pattern.search(line):
                    fout.write(line)


@pytest.mark.skipif(not (ROOT / "artifacts/performance_v3/cohorts/LCE/raw/subjects.jsonl").exists(), reason="LCE dev cohort not built")
def test_seal_is_structural_only_and_detects_overlap(tmp_path, monkeypatch):
    lce = ROOT / "artifacts/performance_v3/cohorts/LCE"
    subjects = [json.loads(l)["subject_id"] for _, l in zip(range(40), (lce / "raw/subjects.jsonl").open())]
    t_ids, peer_ids = subjects[:20], subjects[20:40]
    t = tmp_path / "v3/cohorts/T"
    _slice_raw(lce / "raw", t / "raw", t_ids)
    _slice_raw(lce / "raw", tmp_path / "peer_raw", peer_ids)
    stays = {json.loads(l)["stay_id"] for l in (t / "raw/episodes.jsonl").open()}
    (t / "cohort").mkdir(); (t / "timeline").mkdir(); (t / "pre_split").mkdir()
    with (lce / "cohort/structural_index.jsonl").open() as fin, (t / "cohort/structural_index.jsonl").open("w") as fout:
        for line in fin:
            if json.loads(line)["stay_id"] in stays:
                fout.write(line)
    manifest = json.loads((lce / "raw/synthetic_dataset_manifest_v1.json").read_text())
    manifest["generated_subject_count"] = 20
    (t / "raw/synthetic_dataset_manifest_v1.json").write_text(json.dumps(manifest))
    (t / "timeline/synthetic_processed_manifest_v1.json").write_text("{}")
    (t / "pre_split/pre_split_scientific_package.jsonl").write_text("SEALED_LABELS_MUST_NOT_BE_READ\n")  # never parsed
    ident = {"cohort_id": "T", "namespace": "SYN-V3-LCE", "n_subjects": 20, "role": "sealed_test", "registered": True,
             "seed": manifest["primary_seed_identity"]["value"], "seed_material": "m", "seed_material_sha256": "x"}
    monkeypatch.setattr(fe, "identity", lambda root: dict(ident))
    paths = fe.Paths(ROOT, base=tmp_path / "v3", cohort_dir=t)
    out = fe.seal(ROOT, paths, peer_raw_dirs={"PEER_DISJOINT": tmp_path / "peer_raw"}, check_generator=False)
    assert out["verdict"] == "PASS", out
    record = json.loads(paths.seal.read_text())
    assert record["structural_only"] is True and record["labels_eligibility_or_qa_counts_read"] is False
    assert json.loads(paths.state.read_text())["status"] == SEALED_NOT_ACCESSED
    # an overlapping peer (the same trajectories) must fail the seal
    paths2 = fe.Paths(ROOT, base=tmp_path / "v3b", cohort_dir=t)
    _slice_raw(lce / "raw", tmp_path / "peer_dup", t_ids[:5])
    bad = fe.seal(ROOT, paths2, peer_raw_dirs={"PEER_DUP": tmp_path / "peer_dup"}, check_generator=False, init_ledger=False)
    assert bad["verdict"] == "FAIL" and bad["checks"]["overlap"] is False
