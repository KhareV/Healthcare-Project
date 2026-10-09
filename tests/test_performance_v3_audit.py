"""Fast tests for performance_v3.audit helpers (tiny synthetic inputs)."""

import math

import numpy as np

from performance_v3.audit import icc1, overlap_report, percentiles, structural_checks, support_prevalence, target_denominators


def _cohort(prefix, n, fp_offset=0):
    subs = {f"{prefix}-S{i}" for i in range(n)}
    return {"subject_ids": subs, "stay_ids": {s.replace("-S", "-E") for s in subs},
            "fingerprints": {f"{prefix}-S{i}": f"fp-{prefix}-{i + fp_offset}" for i in range(n)}}


def test_overlap_clean_cohorts_pass():
    r = overlap_report({"A": _cohort("A", 5), "B": _cohort("B", 4)}, test_cohorts=("A",), v3_cohorts=("B",))
    assert r["blockers"] == [] and r["clone_group_count"] == 0
    assert r["pairwise"]["A|B"] == {"subject_intersection": 0, "stay_intersection": 0, "fingerprint_intersection": 0}
    assert r["all_cross_cohort_subject_and_stay_intersections_zero"]
    assert r["cohorts"]["A"]["subjects"] == 5 and r["cohorts"]["B"]["unique_fingerprints"] == 4


def test_overlap_detects_injected_cross_cohort_clone_as_blocker():
    a, b = _cohort("A", 3), _cohort("B", 3)
    b["fingerprints"]["B-S1"] = a["fingerprints"]["A-S2"]
    r = overlap_report({"A": a, "B": b}, test_cohorts=("A",), v3_cohorts=("B",))
    assert r["clone_group_count"] == 1
    assert r["clone_groups"][0] == ["A:A-S2", "B:B-S1"]
    assert r["pairwise"]["A|B"]["fingerprint_intersection"] == 1
    assert len(r["blockers"]) == 1 and "TEST" in r["blockers"][0]


def test_overlap_within_cohort_duplicates_counted_not_independent():
    a = _cohort("A", 4)
    a["fingerprints"]["A-S3"] = a["fingerprints"]["A-S0"]
    r = overlap_report({"A": a})
    assert r["within_cohort_duplicate_fingerprints"]["A"] == 1
    assert r["cohorts"]["A"]["subjects"] == 4 and r["cohorts"]["A"]["unique_fingerprints"] == 3
    assert r["blockers"] == []  # no V3/test relation declared


def test_overlap_identity_intersection_is_blocker():
    a, b = _cohort("A", 2), _cohort("A", 2)
    r = overlap_report({"T": a, "V": b}, test_cohorts=("T",), v3_cohorts=("V",))
    assert not r["all_cross_cohort_subject_and_stay_intersections_zero"]
    assert any("identity overlap" in x for x in r["blockers"])


def _row(stay, k, intime="2100-01-01T00:00:00Z", dur_h=60.0, **over):
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2100, 1, 1, tzinfo=timezone.utc)
    t = t0 + timedelta(hours=24 + 6 * k)
    remaining = dur_h - (24 + 6 * k)
    row = {
        "stay_id": stay, "subject_id": stay.replace("E", "S"), "grid_index": k,
        "prediction_time": t.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
        "temporal_feature_names": ["a", "b"] + [f"c{i}" for i in range(19)],
        "history_values": [[None] * 21 for _ in range(8)], "observation_mask": [[False] * 21 for _ in range(8)],
        "tslo_hours": [[54.0] * 21 for _ in range(8)], "padding_mask": [True] * 3 + [False] * 5,
        "icu_time_eligible": True, "icu_time_log1p": math.log1p(remaining),
        "delta_sofa_24": None, "delta_sofa_48": None, "recovery24_eligible": False, "recovery48_eligible": False,
        "organ_support_eligible": True, "organ_support_label": 0, "support_audit_disposition": "INTERNAL_NEGATIVE_FULL_FOLLOWUP",
    }
    for b in range(3, 8):  # non-padded: support channels always observed, tslo 0
        for f in range(15, 21):
            row["observation_mask"][b][f] = True
            row["tslo_hours"][b][f] = 0.0
    row.update(over)
    return row


def _episodes():
    return {"E1": {"intime": "2100-01-01T00:00:00Z", "outtime": "2100-01-03T12:00:00Z"}}  # 60h


def test_structural_checks_pass_on_valid_rows():
    rows = [_row("E1", 0), _row("E1", 1, support_audit_disposition="INTERNAL_POSITIVE_INITIATION", organ_support_label=1),
            _row("E1", 2, organ_support_eligible=False, organ_support_label=None,
                 support_audit_disposition="INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION")]
    checks = structural_checks(rows, _episodes(), rows[0]["temporal_feature_names"], ("outtime", "latent"))
    assert all(c["pass"] for c in checks.values()), {k: v for k, v in checks.items() if not v["pass"]}
    assert checks["max_12_cutoffs_per_stay"]["max_observed"] == 3


def test_structural_checks_flag_violations():
    bad_cutoff = _row("E1", 0, prediction_time="2100-01-02T01:00:00.000000Z")
    assert not structural_checks([bad_cutoff], _episodes(), bad_cutoff["temporal_feature_names"], ())["first_cutoff_is_intime_plus_24h"]["pass"]
    late = _row("E1", 5)  # t = 54h > outtime(60h) - 6h? 54 <= 54 ok -> push beyond
    ep = {"E1": {"intime": "2100-01-01T00:00:00Z", "outtime": "2100-01-03T07:00:00Z"}}  # 55h
    assert not structural_checks([late], ep, late["temporal_feature_names"], ())["cutoff_not_later_than_outtime_minus_6h"]["pass"]
    leak = structural_checks([_row("E1", 0)], _episodes(), ["delta_sofa_24x"] + ["c"] * 20, ("delta_sofa",))
    assert not leak["no_prohibited_predictor_names"]["pass"]
    wrong_icu = _row("E1", 0, icu_time_log1p=1.0)
    assert not structural_checks([wrong_icu], _episodes(), wrong_icu["temporal_feature_names"], ())["icu_time_log1p_exact"]["pass"]
    mismatch = _row("E1", 0, organ_support_label=None)
    assert not structural_checks([mismatch], _episodes(), mismatch["temporal_feature_names"], ())["support_label_eligibility_consistent"]["pass"]


def test_target_denominators_and_support_prevalence():
    rows = [
        {"stay_id": "E1", "recovery24_eligible": True, "delta_sofa_24": 1.0, "recovery48_eligible": True, "delta_sofa_48": 0.0,
         "icu_time_eligible": True, "icu_time_log1p": 1.0, "organ_support_eligible": True, "organ_support_label": 1,
         "support_audit_disposition": "INTERNAL_POSITIVE_INITIATION"},
        {"stay_id": "E1", "recovery24_eligible": True, "delta_sofa_24": 0.0, "recovery48_eligible": False, "delta_sofa_48": None,
         "icu_time_eligible": True, "icu_time_log1p": 1.0, "organ_support_eligible": True, "organ_support_label": 0,
         "support_audit_disposition": "INTERNAL_NEGATIVE_FULL_FOLLOWUP"},
        {"stay_id": "E2", "recovery24_eligible": False, "delta_sofa_24": None, "recovery48_eligible": False, "delta_sofa_48": None,
         "icu_time_eligible": True, "icu_time_log1p": 1.0, "organ_support_eligible": False, "organ_support_label": None,
         "support_audit_disposition": "INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION"},
    ]
    d = target_denominators(rows)
    assert d["recovery24"] == {"eligible_rows": 2, "unique_stays": 1}
    assert d["recovery48"] == {"eligible_rows": 1, "unique_stays": 1}
    assert d["icu_time"] == {"eligible_rows": 3, "unique_stays": 2}
    assert d["organ_support"] == {"eligible_rows": 2, "unique_stays": 1}
    p = support_prevalence(rows)
    assert (p["positive_rows"], p["negative_rows"], p["censored_rows"]) == (1, 1, 1)
    assert p["row_prevalence"] == 0.5 and p["stay_balanced_prevalence"] == 0.5 and p["stays_with_any_positive"] == 1


def test_icc_and_percentiles():
    rng = np.random.default_rng(0)
    groups = [rng.normal(m, 0.1, 6) for m in rng.normal(0, 5, 40)]
    assert icc1(groups)["icc1"] > 0.95
    noise = [rng.normal(0, 1, 6) for _ in range(60)]
    assert abs(icc1(noise)["icc1"]) < 0.15
    assert icc1([np.array([1.0])])["icc1"] is None
    p = percentiles(np.arange(101, dtype=float))
    assert p["p50"] == 50.0 and p["n"] == 101 and percentiles(np.array([]))["n"] == 0
