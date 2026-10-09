"""Pure audit / characterization helpers for the V3 cohorts (read-only, deterministic)."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def r6(value: Optional[float]) -> Optional[float]:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), 6)


# ---------------------------------------------------------------- overlap audit
def overlap_report(
    cohorts: Mapping[str, Mapping[str, object]],
    test_cohorts: Iterable[str] = (),
    v3_cohorts: Iterable[str] = (),
) -> Dict[str, object]:
    """cohorts[name] = {"subject_ids": set, "stay_ids": set, "fingerprints": {subject_id: fp}}.

    Pairwise subject/stay intersections, within-cohort duplicate fingerprints,
    cross-cohort exact-trajectory overlap, and BLOCKER if a V3 subject
    duplicates a trajectory of any cohort listed in ``test_cohorts``.
    """

    names = sorted(cohorts)
    test_set, v3_set = set(test_cohorts), set(v3_cohorts)
    sizes = {n: {"subjects": len(cohorts[n]["subject_ids"]), "stays": len(cohorts[n]["stay_ids"]),
                 "unique_fingerprints": len(set(cohorts[n]["fingerprints"].values()))} for n in names}
    pairs: Dict[str, object] = {}
    for a, b in itertools.combinations(names, 2):
        fa, fb = set(cohorts[a]["fingerprints"].values()), set(cohorts[b]["fingerprints"].values())
        pairs[f"{a}|{b}"] = {
            "subject_intersection": len(set(cohorts[a]["subject_ids"]) & set(cohorts[b]["subject_ids"])),
            "stay_intersection": len(set(cohorts[a]["stay_ids"]) & set(cohorts[b]["stay_ids"])),
            "fingerprint_intersection": len(fa & fb),
        }
    by_fp: Dict[str, List[str]] = defaultdict(list)
    for n in names:
        for subject, fp in cohorts[n]["fingerprints"].items():
            by_fp[fp].append(f"{n}:{subject}")
    clone_groups = sorted((sorted(m) for m in by_fp.values() if len(m) > 1), key=lambda m: m[0])
    within = {n: len(cohorts[n]["fingerprints"]) - len(set(cohorts[n]["fingerprints"].values())) for n in names}
    blockers: List[str] = []
    for group in clone_groups:
        owners = {m.split(":", 1)[0] for m in group}
        if owners & test_set and owners & v3_set:
            blockers.append("V3 trajectory clones historical TEST subject: " + ", ".join(group))
    for pair, rec in pairs.items():
        a, b = pair.split("|")
        if (a in test_set and b in v3_set) or (b in test_set and a in v3_set):
            if rec["subject_intersection"] or rec["stay_intersection"]:
                blockers.append(f"identity overlap between V3 and historical test: {pair}")
    return {
        "cohorts": sizes,
        "pairwise": pairs,
        "within_cohort_duplicate_fingerprints": within,
        "clone_groups": clone_groups,
        "clone_group_count": len(clone_groups),
        "blockers": blockers,
        "all_cross_cohort_subject_and_stay_intersections_zero": all(
            r["subject_intersection"] == 0 and r["stay_intersection"] == 0 for r in pairs.values()),
        "fold_assignment_policy": (
            "every member of an exact-fingerprint clone group is assigned the fold of its lexicographically "
            "first member (performance_v3.splits.clone_safe_folds); clones are never counted as independent subjects"),
    }


# ---------------------------------------------------------------- structural audit
def structural_checks(rows: Sequence[Mapping[str, object]], episodes: Mapping[str, Mapping[str, object]],
                      temporal_names: Sequence[str], prohibited_tokens: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Phase-13-style structural invariants on pre-split package rows. Returns check -> {pass, n}."""

    per_stay: Dict[str, List[int]] = defaultdict(list)
    cutoff_ok = first_ok = bound_ok = True
    shape_ok = pad_ok = mask_ok = names_ok = True
    elig_ok = icu_ok = rec_ok = sup_ok = True
    for row in rows:
        ep = episodes[row["stay_id"]]
        t, intime, outtime = _dt(row["prediction_time"]), _dt(ep["intime"]), _dt(ep["outtime"])
        k = int(row["grid_index"])
        per_stay[row["stay_id"]].append(k)
        cutoff_ok &= 0 <= k < 12 and t == intime + timedelta(hours=24 + 6 * k)
        bound_ok &= t <= outtime - timedelta(hours=6)
        names_ok &= list(row["temporal_feature_names"]) == list(temporal_names)
        hv, om, pm, ts = row["history_values"], row["observation_mask"], row["padding_mask"], row["tslo_hours"]
        shape_ok &= len(hv) == 8 and len(om) == 8 and len(pm) == 8 and len(ts) == 8 \
            and all(len(x) == 21 for x in hv) and all(len(x) == 21 for x in om) and all(len(x) == 21 for x in ts)
        pad_ok &= list(pm) == sorted(pm, reverse=True)
        for b, padded in enumerate(pm):
            for f in range(21):
                if padded:
                    mask_ok &= hv[b][f] is None and om[b][f] is False and ts[b][f] == 54.0
                elif f >= 15:
                    mask_ok &= om[b][f] is True and ts[b][f] == 0.0
        remaining = (outtime - t).total_seconds() / 3600.0
        icu_ok &= row["icu_time_eligible"] is True and math.isclose(float(row["icu_time_log1p"]), math.log1p(remaining), abs_tol=1e-12)
        for val, el in ((row["delta_sofa_24"], row["recovery24_eligible"]), (row["delta_sofa_48"], row["recovery48_eligible"])):
            rec_ok &= (val is None and not el) or (bool(el) and -24 <= float(val) <= 24)
        rec_ok &= (not row["recovery48_eligible"]) or bool(row["recovery24_eligible"])
        sup_ok &= (row["organ_support_label"] in (0, 1)) if row["organ_support_eligible"] else row["organ_support_label"] is None
        disp = row.get("support_audit_disposition")
        elig_ok &= (disp in ("INTERNAL_POSITIVE_INITIATION", "INTERNAL_NEGATIVE_FULL_FOLLOWUP")) == bool(row["organ_support_eligible"])
        if disp == "INTERNAL_POSITIVE_INITIATION":
            sup_ok &= row["organ_support_label"] == 1
        if disp == "INTERNAL_NEGATIVE_FULL_FOLLOWUP":
            sup_ok &= row["organ_support_label"] == 0
    max_per_stay = max((len(v) for v in per_stay.values()), default=0)
    grid_ok = all(sorted(v) == list(range(len(v))) for v in per_stay.values())
    first_ok = all(0 in v for v in per_stay.values())
    lowered = [n.lower() for n in temporal_names]
    hits = sorted({t for t in prohibited_tokens for n in lowered if t in n})
    disp_counts = Counter(r.get("support_audit_disposition") for r in rows)
    return {
        "max_12_cutoffs_per_stay": {"pass": max_per_stay <= 12 and grid_ok, "max_observed": max_per_stay, "stays": len(per_stay)},
        "first_cutoff_is_intime_plus_24h": {"pass": first_ok and cutoff_ok, "rows": len(rows)},
        "cutoff_not_later_than_outtime_minus_6h": {"pass": bound_ok, "rows": len(rows)},
        "history_shape_8x21_and_names": {"pass": shape_ok and names_ok, "rows": len(rows)},
        "padding_monotone_and_padded_cells_unobserved": {"pass": pad_ok and mask_ok, "rows": len(rows)},
        "icu_time_log1p_exact": {"pass": icu_ok, "rows": len(rows)},
        "recovery_masks_and_ranges": {"pass": rec_ok, "rows": len(rows)},
        "support_label_eligibility_consistent": {"pass": sup_ok and elig_ok, "rows": len(rows)},
        "no_prohibited_predictor_names": {"pass": not hits, "hits": hits},
        "support_disposition_counts": {"pass": disp_counts["INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION"] > 0
                                       and disp_counts["INTERNAL_POSITIVE_INITIATION"] > 0,
                                       "counts": dict(sorted((str(k), v) for k, v in disp_counts.items()))},
    }


# ---------------------------------------------------------------- characterization helpers
def percentiles(values: np.ndarray, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)) -> Dict[str, Optional[float]]:
    if values.size == 0:
        return {f"p{q}": None for q in qs} | {"n": 0}
    out = {f"p{q}": r6(np.percentile(values, q)) for q in qs}
    out["n"] = int(values.size)
    out["mean"] = r6(values.mean())
    out["sd"] = r6(values.std(ddof=1)) if values.size > 1 else None
    return out


def target_denominators(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, int]]:
    def stat(pred):
        sel = [r for r in rows if pred(r)]
        return {"eligible_rows": len(sel), "unique_stays": len({r["stay_id"] for r in sel})}
    return {
        "legal_prediction_rows": stat(lambda r: True),
        "recovery24": stat(lambda r: bool(r["recovery24_eligible"]) and r["delta_sofa_24"] is not None),
        "recovery48": stat(lambda r: bool(r["recovery48_eligible"]) and r["delta_sofa_48"] is not None),
        "icu_time": stat(lambda r: bool(r["icu_time_eligible"]) and r["icu_time_log1p"] is not None),
        "organ_support": stat(lambda r: bool(r["organ_support_eligible"]) and r["organ_support_label"] is not None),
    }


def support_prevalence(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    disp = Counter(str(r.get("support_audit_disposition")) for r in rows)
    elig = [r for r in rows if r["organ_support_eligible"] and r["organ_support_label"] is not None]
    pos = [r for r in elig if r["organ_support_label"] == 1]
    per_stay: Dict[str, List[int]] = defaultdict(list)
    for r in elig:
        per_stay[r["stay_id"]].append(int(r["organ_support_label"]))
    w_pos = sum(sum(v) / len(v) for v in per_stay.values())
    return {
        "disposition_rows": dict(sorted(disp.items())),
        "eligible_rows": len(elig), "positive_rows": len(pos), "negative_rows": len(elig) - len(pos),
        "censored_rows": disp.get("INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION", 0),
        "not_at_risk_rows": disp.get("INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON", 0),
        "eligible_stays": len(per_stay),
        "stays_with_any_positive": sum(1 for v in per_stay.values() if any(v)),
        "row_prevalence": r6(len(pos) / len(elig)) if elig else None,
        "stay_balanced_prevalence": r6(w_pos / len(per_stay)) if per_stay else None,
    }


def icc1(groups: Sequence[np.ndarray]) -> Dict[str, Optional[float]]:
    """One-way random-effects ICC(1) for unbalanced groups (ANOVA estimator)."""

    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if g.size >= 1]
    k = len(groups)
    n = sum(g.size for g in groups)
    if k < 2 or n <= k:
        return {"icc1": None, "groups": k, "n": n}
    grand = sum(g.sum() for g in groups) / n
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - sum(g.size ** 2 for g in groups) / n) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else None
    return {"icc1": r6(icc), "groups": k, "n": int(n), "between_var": r6(max((msb - msw) / n0, 0.0)), "within_var": r6(msw)}
