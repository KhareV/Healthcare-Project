#!/usr/bin/env python3
"""Descriptive characterization of the V3 development pool (V1 train+validation + D1-D3) and LCE.

Real data only, deterministic, DEVELOPMENT-tagged. V1 TEST, V2 fresh test and the (not yet
generated) sealed V3 test cohort are never read.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.synthetic.validation import load_jsonl  # noqa: E402
from performance_v3.audit import _dt, icc1, percentiles, r6, support_prevalence, target_denominators  # noqa: E402
from performance_v3.protocol import cohort_identity, protocol_sha256, sha256_file  # noqa: E402

OUT = ROOT / "artifacts/performance_v3/data_characterization"
FIG = OUT / "figures"
V1_SPLIT = ROOT / "artifacts/splits/synthetic_split_v2.csv"
V1_PRE = ROOT / "artifacts/data/synthetic/pre_split/final/phase9_final_v1/pre_split_scientific_package.jsonl"
V1_STATICS = ROOT / "artifacts/data/synthetic/timelines/final/phase9_final_v1/canonical_statics.jsonl"
V1_RAW_SUBJECTS = ROOT / "artifacts/data/synthetic/final/phase9_final_v1/subjects.jsonl"
CHANNELS = [c["name"] for c in json.loads((ROOT / "configs/synthetic/feature_schema_v2.json").read_text())["temporal_channels"]]
CLINICAL = 15  # channels 0..14 are observed clinical variables; 15..20 are support state/rate channels
SHORT = [c.split("__")[0] for c in CHANNELS]
PAIRS = [("mean_arterial_pressure", "systolic_blood_pressure"), ("systolic_blood_pressure", "diastolic_blood_pressure"),
         ("pao2", "fio2"), ("pao2", "oxygen_saturation"), ("creatinine", "urine_output_volume"),
         ("lactate", "mean_arterial_pressure"), ("heart_rate", "lactate"), ("respiratory_rate", "oxygen_saturation"),
         ("creatinine", "bilirubin_total"), ("platelet_count", "bilirubin_total"), ("glasgow_coma_scale", "lactate")]
ICC_CHANNELS = ["heart_rate", "mean_arterial_pressure", "creatinine", "lactate", "oxygen_saturation",
                "platelet_count", "glasgow_coma_scale", "temperature"]


def _write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", name)


def _csv(name, header, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(header)
        w.writerows(rows)


def load_group(name):
    """-> dict(rows, statics{stay:row}, generated, namespace)."""
    if name == "V1_DEV":
        with V1_SPLIT.open(newline="", encoding="utf-8") as h:
            dev = {r["subject_id"] for r in csv.DictReader(h) if r["split"] in ("train", "validation")}
        rows = [r for r in load_jsonl(V1_PRE) if r["subject_id"] in dev]
        statics = {r["stay_id"]: r for r in load_jsonl(V1_STATICS) if r["subject_id"] in dev}
        generated = len(dev)
    else:
        base = ROOT / "artifacts/performance_v3/cohorts" / name
        rows = list(load_jsonl(base / "pre_split/pre_split_scientific_package.jsonl"))
        statics = {r["stay_id"]: r for r in load_jsonl(base / "timeline/canonical_statics.jsonl")}
        generated = sum(1 for _ in load_jsonl(base / "raw/subjects.jsonl"))
    return {"rows": rows, "statics": statics, "generated": generated}


def arrays(rows):
    n = len(rows)
    hv = np.array([r["history_values"] for r in rows], dtype=float).reshape(n, 8, 21)
    om = np.array([r["observation_mask"] for r in rows], dtype=bool).reshape(n, 8, 21)
    pm = np.array([r["padding_mask"] for r in rows], dtype=bool).reshape(n, 8)
    return hv, om, pm


def merge(groups):
    rows, statics, generated = [], {}, 0
    for g in groups:
        rows += g["rows"]
        statics.update(g["statics"])
        generated += g["generated"]
    return {"rows": rows, "statics": statics, "generated": generated}


def describe(g):
    rows, statics = g["rows"], g["statics"]
    n = len(rows)
    hv, om, pm = arrays(rows)
    stays = sorted({r["stay_id"] for r in rows})
    den = target_denominators(rows)
    out = {"subjects_generated": g["generated"], "retained_stays": len(statics), "stays_with_legal_cutoff": len(stays),
           "rows": n, "denominators": den, "support": support_prevalence(rows)}
    # subtype / demographics (stays with at least one row)
    st = [statics[s] for s in stays]
    sub = Counter(s["cardiac_condition_group"] for s in st)
    out["subtype"] = {k: {"stays": v, "fraction": r6(v / len(st))} for k, v in sorted(sub.items())}
    sex = Counter(s["sex_category"] for s in st)
    ages = np.array([s["age_years"] for s in st], dtype=float)
    out["age_sex"] = {"sex": {k: {"stays": v, "fraction": r6(v / len(st))} for k, v in sorted(sex.items())},
                      "age_years": percentiles(ages),
                      "age_by_subtype_mean": {k: r6(np.mean([s["age_years"] for s in st if s["cardiac_condition_group"] == k])) for k in sorted(sub)}}
    hrs = np.array([(_dt(s["outtime"]) - _dt(s["intime"])).total_seconds() / 3600 for s in st])
    out["episode_duration_hours"] = {"all": percentiles(hrs), "by_subtype": {
        k: percentiles(np.array([(_dt(s["outtime"]) - _dt(s["intime"])).total_seconds() / 3600 for s in st
                                 if s["cardiac_condition_group"] == k])) for k in sorted(sub)}}
    per_stay = Counter(r["stay_id"] for r in rows)
    hist = Counter(per_stay.values())
    out["legal_cutoffs_per_stay"] = {"histogram": {str(k): hist[k] for k in sorted(hist)}, "mean": r6(np.mean(list(per_stay.values()))),
                                     "max": max(per_stay.values()), "stays": len(per_stay)}
    # missingness (non-padded cells)
    nonpad = ~pm
    cell_n = nonpad.sum()
    out["padding"] = {"padded_bin_fraction": r6(pm.mean()), "padded_cells": int(pm.sum()), "nonpadded_cells": int(cell_n)}
    by_feature = {}
    for f, name in enumerate(CHANNELS):
        obs = (om[:, :, f] & nonpad).sum()
        by_feature[name] = {"observed_fraction_nonpadded_bins": r6(obs / cell_n), "missing_fraction_nonpadded_bins": r6(1 - obs / cell_n),
                            "channel_type": "clinical_observation" if f < CLINICAL else "support_state_always_known"}
    out["missingness_by_feature"] = by_feature
    by_bin = {}
    for b in range(8):
        nb = nonpad[:, b].sum()
        by_bin[str(b)] = {"nonpadded_rows": int(nb), "padded_fraction": r6(1 - nb / n),
                          "missing_fraction_by_clinical_channel": {SHORT[f]: r6(1 - (om[:, b, f] & nonpad[:, b]).sum() / nb) if nb else None
                                                                   for f in range(CLINICAL)}}
    out["missingness_by_bin"] = by_bin
    # SOFA
    base = np.array([np.nan if r["baseline_sofa"] is None else float(r["baseline_sofa"]) for r in rows])
    gi = np.array([int(r["grid_index"]) for r in rows])
    out["sofa"] = {"baseline_sofa_available_rows": int((~np.isnan(base)).sum()), "rows": n,
                   "overall": percentiles(base[~np.isnan(base)]),
                   "by_cutoff_index": {str(k): percentiles(base[(gi == k) & ~np.isnan(base)]) for k in range(12) if ((gi == k) & ~np.isnan(base)).any()},
                   "components": "UNAVAILABLE: the pre-split package stores only total baseline_sofa (component scores are not persisted); "
                                 "not recomputed here to avoid a second SOFA implementation"}
    # variable distributions from newest bin (disjoint across a stay's cutoffs)
    newest = hv[:, 7, :]
    nmask = om[:, 7, :] & ~pm[:, 7][:, None]
    out["variable_distributions_newest_bin"] = {CHANNELS[f]: percentiles(newest[nmask[:, f], f]) for f in range(CLINICAL)}
    out["_arrays"] = {"newest": newest, "nmask": nmask, "stay": np.array([r["stay_id"] for r in rows]), "gi": gi, "base": base,
                      "hours": hrs, "subtype_stays": sub}
    return out


def correlations(g_desc):
    a = g_desc["_arrays"]
    newest, nmask = a["newest"], a["nmask"]
    idx = {c.split("__")[0]: i for i, c in enumerate(CHANNELS)}
    def corr(x, y):
        m = nmask[:, x] & nmask[:, y]
        if m.sum() < 200:
            return None, int(m.sum())
        return r6(np.corrcoef(newest[m, x], newest[m, y])[0, 1]), int(m.sum())
    named = []
    for p, q in PAIRS:
        r, n = corr(idx[p], idx[q])
        named.append({"a": p, "b": q, "pearson_r": r, "n_rows": n})
    allp = []
    for i in range(CLINICAL):
        for j in range(i + 1, CLINICAL):
            r, n = corr(i, j)
            if r is not None:
                allp.append({"a": SHORT[i], "b": SHORT[j], "pearson_r": r, "n_rows": n})
    allp.sort(key=lambda d: (-abs(d["pearson_r"]), d["a"], d["b"]))
    return {"clinically_motivated_pairs": named, "top_abs_pairs": allp[:15], "pair_count_evaluated": len(allp),
            "basis": "newest 6h bin of each cutoff row, genuine observations only, pairwise-complete, n>=200"}


def icc_report(g_desc):
    a = g_desc["_arrays"]
    out = {}
    for ch in ICC_CHANNELS:
        f = [c.split("__")[0] for c in CHANNELS].index(ch)
        groups = defaultdict(list)
        for s, v, ok in zip(a["stay"], a["newest"][:, f], a["nmask"][:, f]):
            if ok:
                groups[s].append(v)
        out[ch] = icc1([np.array(v, dtype=float) for v in groups.values() if len(v) >= 1])
    return {"method": "one-way random-effects ICC(1), unbalanced ANOVA estimator; groups=stays, observations=newest-bin values "
                      "of that stay's cutoff rows (disjoint 6h bins)", "channels": out}


def figures(names, descs):
    FIG.mkdir(parents=True, exist_ok=True)
    cols = ["#2bb8b0", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f"]
    fig, ax = plt.subplots(figsize=(7, 4))
    subs = sorted({k for d in descs.values() for k in d["subtype"]})
    for i, k in enumerate(subs):
        ax.bar(np.arange(len(names)) + i * 0.25, [descs[n]["subtype"].get(k, {"fraction": 0})["fraction"] for n in names], 0.25, label=k, color=cols[i])
    ax.set_xticks(np.arange(len(names)) + 0.25); ax.set_xticklabels(names); ax.set_ylabel("fraction of stays"); ax.legend(fontsize=7)
    ax.set_title("Cardiac subtype mix (development / LCE cohorts)")
    fig.tight_layout(); fig.savefig(FIG / "subtype_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for n, c in (("DEV_POOL", cols[0]), ("LCE", cols[1])):
        ax.hist(descs[n]["_arrays"]["hours"], bins=40, alpha=0.6, density=True, label=n, color=c)
    ax.set_xlabel("episode duration (h)"); ax.set_ylabel("density"); ax.legend(); ax.set_title("ICU-like episode duration")
    fig.tight_layout(); fig.savefig(FIG / "episode_duration.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    tasks = ["recovery24", "recovery48", "icu_time", "organ_support"]
    den = descs["DEV_POOL"]["denominators"]
    ax.bar(np.arange(4) - 0.2, [den[t]["eligible_rows"] for t in tasks], 0.4, label="eligible rows", color=cols[0])
    ax.bar(np.arange(4) + 0.2, [den[t]["unique_stays"] for t in tasks], 0.4, label="unique stays", color=cols[1])
    ax.set_xticks(range(4)); ax.set_xticklabels(tasks); ax.legend(); ax.set_title("Task eligibility, development pool")
    fig.tight_layout(); fig.savefig(FIG / "target_eligibility.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    m = np.array([[descs["DEV_POOL"]["missingness_by_bin"][str(b)]["missing_fraction_by_clinical_channel"][SHORT[f]] or 0 for b in range(8)]
                  for f in range(CLINICAL)])
    im = ax.imshow(m, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(CLINICAL)); ax.set_yticklabels(SHORT[:CLINICAL], fontsize=7); ax.set_xticks(range(8))
    ax.set_xlabel("bin (0 = oldest, 7 = newest); padded rows excluded"); fig.colorbar(im, label="missing fraction")
    ax.set_title("Missingness by channel and bin, development pool")
    fig.tight_layout(); fig.savefig(FIG / "missingness_heatmap.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    s = descs["DEV_POOL"]["sofa"]["by_cutoff_index"]
    ks = sorted(int(k) for k in s)
    ax.errorbar(ks, [s[str(k)]["mean"] for k in ks], yerr=[s[str(k)]["sd"] or 0 for k in ks], marker="o", color=cols[0], capsize=3)
    ax.set_xlabel("cutoff index k"); ax.set_ylabel("baseline SOFA at t (mean +/- sd)"); ax.set_title("SOFA by cutoff, development pool")
    fig.tight_layout(); fig.savefig(FIG / "sofa_by_cutoff.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["positive_rows", "negative_rows", "censored_rows", "not_at_risk_rows"]
    bottom = np.zeros(len(names))
    for i, k in enumerate(keys):
        vals = np.array([descs[n]["support"][k] / descs[n]["rows"] for n in names])
        ax.bar(names, vals, bottom=bottom, label=k, color=cols[i]); bottom += vals
    ax.legend(fontsize=7); ax.set_ylabel("fraction of legal rows"); ax.set_title("Organ-support row dispositions")
    fig.tight_layout(); fig.savefig(FIG / "support_dispositions.png", dpi=160); plt.close(fig)


def lineage():
    out = {"protocol_sha256": protocol_sha256(ROOT), "cohorts": {}}
    for cid in ("D1", "D2", "D3", "LCE"):
        base = ROOT / "artifacts/performance_v3/cohorts" / cid
        ident = cohort_identity(ROOT, cid)
        log = json.loads((base / "build_log.json").read_text())
        manifests = {"raw": "raw/synthetic_dataset_manifest_v1.json", "cohort": "cohort/synthetic_cohort_manifest_v2.json",
                     "timeline": "timeline/synthetic_processed_manifest_v1.json",
                     "features": "features/synthetic_feature_input_manifest_v1.json", "pre_split": "pre_split/phase9_pre_split_manifest_v1.json"}
        cmds = {}
        for k, p in manifests.items():
            txt = (base / p).read_text()
            i = txt.find('"generation_command"')
            cmds[k] = json.loads(txt)["generation_command"] if i >= 0 and k != "pre_split" else None
        out["cohorts"][cid] = {
            "namespace": ident["namespace"], "n_subjects": ident["n_subjects"], "seed": ident["seed"],
            "seed_material": ident["seed_material"], "seed_material_sha256": ident["seed_material_sha256"],
            "config_ref": f"configs/performance_v3/cohorts/{cid.lower()}_v1.json",
            "config_sha256": sha256_file(ROOT / f"configs/performance_v3/cohorts/{cid.lower()}_v1.json"),
            "manifest_sha256": {k: sha256_file(base / p) for k, p in manifests.items()},
            "generation_commands": cmds, "build_stage_seconds": {s["stage"]: s["seconds"] for s in log["stages"]},
            "build_commands": [s["command"] for s in log["stages"]]}
    out["v1_development_subjects"] = {
        "source_config": "configs/synthetic/final_benchmark_v1.json", "seed": 20260921, "namespace": "SYN",
        "dev_manifest_ref": "artifacts/performance_v2/phase3/dev_subject_manifest_v1.json",
        "dev_manifest_sha256": sha256_file(ROOT / "artifacts/performance_v2/phase3/dev_subject_manifest_v1.json")}
    return out


def main():
    groups = {"V1_DEV": load_group("V1_DEV")}
    for c in ("D1", "D2", "D3", "LCE"):
        groups[c] = load_group(c)
    groups["DEV_POOL"] = merge([groups[k] for k in ("V1_DEV", "D1", "D2", "D3")])
    names = ["V1_DEV", "D1", "D2", "D3", "DEV_POOL", "LCE"]
    descs = {n: describe(groups[n]) for n in names}
    for n in names:
        print(n, descs[n]["subjects_generated"], descs[n]["rows"], flush=True)
    tag = {"artifact_status": "DEVELOPMENT_CHARACTERIZATION_NOT_FINAL_TEST", "protocol_sha256": protocol_sha256(ROOT),
           "partitions": {"DEV_POOL": "V1 train+validation (1700) + D1+D2+D3", "LCE": "learning-curve evaluation cohort",
                          "V1_DEV": "V1 train+validation", "D1": "SYN-V3-D1", "D2": "SYN-V3-D2", "D3": "SYN-V3-D3"},
           "excluded_never_read": ["V1_TEST", "V2_FRESH_TEST", "SYN-V3-T"]}
    strip = lambda d, *keys: {k: d[k] for k in keys}
    _write("cohort_summary.json", {**tag, "cohorts": {n: {**strip(descs[n], "subjects_generated", "retained_stays", "stays_with_legal_cutoff", "rows", "padding"),
                                                      "legal_cutoffs_per_stay_mean": descs[n]["legal_cutoffs_per_stay"]["mean"]} for n in names}})
    _write("cohort_flow.json", {**tag, "cohorts": {n: {"subjects_generated": descs[n]["subjects_generated"], "retained_stays": descs[n]["retained_stays"],
                                                      "stays_with_legal_cutoff": descs[n]["stays_with_legal_cutoff"], "legal_rows": descs[n]["rows"],
                                                      **{f"{t}_stays": descs[n]["denominators"][t]["unique_stays"] for t in ("recovery24", "recovery48", "icu_time", "organ_support")}}
                                                  for n in names}})
    _write("subtype_distribution.json", {**tag, "cohorts": {n: descs[n]["subtype"] for n in names}})
    _write("target_eligibility.json", {**tag, "cohorts": {n: descs[n]["denominators"] for n in names}})
    _write("outcome_prevalence.json", {**tag, "cohorts": {n: descs[n]["support"] for n in names}})
    _write("missingness_by_feature.json", {**tag, "cohorts": {n: descs[n]["missingness_by_feature"] for n in names}})
    _write("missingness_by_bin.json", {**tag, "cohorts": {n: descs[n]["missingness_by_bin"] for n in names}})
    _write("sofa_distributions.json", {**tag, "cohorts": {n: descs[n]["sofa"] for n in names}})
    _write("episode_duration_summary.json", {**tag, "cohorts": {n: descs[n]["episode_duration_hours"] for n in names}})
    _write("legal_cutoffs_per_stay.json", {**tag, "cohorts": {n: descs[n]["legal_cutoffs_per_stay"] for n in names}})
    _write("age_sex_composition.json", {**tag, "cohorts": {n: descs[n]["age_sex"] for n in names}})
    _write("variable_distributions.json", {**tag, "basis": "newest 6h bin of each cutoff row, genuine observations",
                                           "cohorts": {n: descs[n]["variable_distributions_newest_bin"] for n in ("DEV_POOL", "LCE")}})
    _write("correlations.json", {**tag, "cohorts": {n: correlations(descs[n]) for n in ("DEV_POOL", "LCE")}})
    _write("between_subject_variability.json", {**tag, "cohorts": {n: icc_report(descs[n]) for n in ("DEV_POOL", "LCE")}})
    _write("generation_lineage.json", lineage())

    _csv("subtype_distribution.csv", ["cohort", "subtype", "stays", "fraction"],
         [[n, k, v["stays"], v["fraction"]] for n in names for k, v in descs[n]["subtype"].items()])
    _csv("target_eligibility.csv", ["cohort", "task", "eligible_rows", "unique_stays"],
         [[n, t, v["eligible_rows"], v["unique_stays"]] for n in names for t, v in descs[n]["denominators"].items()])
    _csv("outcome_prevalence.csv", ["cohort", "eligible_rows", "positive_rows", "negative_rows", "censored_rows", "not_at_risk_rows",
                                    "eligible_stays", "stays_with_any_positive", "row_prevalence", "stay_balanced_prevalence"],
         [[n] + [descs[n]["support"][k] for k in ("eligible_rows", "positive_rows", "negative_rows", "censored_rows", "not_at_risk_rows",
                                                   "eligible_stays", "stays_with_any_positive", "row_prevalence", "stay_balanced_prevalence")] for n in names])
    _csv("missingness_by_feature.csv", ["cohort", "channel", "channel_type", "observed_fraction_nonpadded_bins"],
         [[n, c, v["channel_type"], v["observed_fraction_nonpadded_bins"]] for n in names for c, v in descs[n]["missingness_by_feature"].items()])
    _csv("missingness_by_bin.csv", ["cohort", "bin", "channel", "missing_fraction", "nonpadded_rows"],
         [[n, b, c, v, descs[n]["missingness_by_bin"][b]["nonpadded_rows"]] for n in names
          for b in map(str, range(8)) for c, v in descs[n]["missingness_by_bin"][b]["missing_fraction_by_clinical_channel"].items()])
    _csv("sofa_by_cutoff.csv", ["cohort", "cutoff_index", "n", "mean", "sd", "p50"],
         [[n, k, v["n"], v["mean"], v["sd"], v["p50"]] for n in names for k, v in descs[n]["sofa"]["by_cutoff_index"].items()])
    figures(["V1_DEV", "D1", "D2", "D3", "DEV_POOL", "LCE"], descs)
    print("done")


if __name__ == "__main__":
    main()
