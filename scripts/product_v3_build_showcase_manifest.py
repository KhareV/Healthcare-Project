#!/usr/bin/env python3
"""Build the additive V3 showcase manifest (8 scenarios) from the LCE development-evaluation cohort.

Selection uses only structural / observed-trajectory criteria (SOFA(t) from the cache, support intervals,
subtype, observation density, canonical cutoff count). No model prediction or error is consulted.
"""
import hashlib
import json
import sys
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay  # noqa: E402
from serving.v3 import showcase as S  # noqa: E402

SOFA_CACHE = ROOT / "artifacts/performance_v3/cache/sofa_LCE.json"
OUT = ROOT / S.MANIFEST_REL
MIN_CUTOFFS = 5


def rank(stay_id: str) -> str:
    return hashlib.sha256(stay_id.encode()).hexdigest()


def main() -> None:
    paths = S.cohort_paths(ROOT)
    statics = {r["stay_id"]: r for r in S.stream_jsonl(paths["statics"])}
    index: dict = {}
    for r in S.stream_jsonl(paths["structural_index"]):
        index.setdefault(r["stay_id"], []).append(r)
    trends = json.loads(SOFA_CACHE.read_text())["trends"]
    masks: dict = {}
    for r in S.stream_jsonl(paths["pre_split"]):
        mask, pad = r["observation_mask"], r["padding_mask"]
        real = [i for i, p in enumerate(pad) if not p]
        observed_bins = sum(1 for i in real if any(mask[i]))
        cells = sum(sum(1 for v in mask[i] if v) for i in real)
        masks[(r["stay_id"], r["prediction_time"])] = {
            "observed_bins": observed_bins, "padding_bins": int(sum(1 for p in pad if p)),
            "observation_density": round(cells / (max(len(real), 1) * len(mask[0])), 6)}
    supports: dict = {}
    for r in S.stream_jsonl(paths["support"]):
        supports.setdefault(r["stay_id"], []).append(r)

    cat = {}
    for sid, st in statics.items():
        S.assert_allowed(sid, st["subject_id"])
        rows = sorted(index.get(sid, []), key=lambda r: r["grid_index"])
        if len(rows) < MIN_CUTOFFS:
            continue
        intime, outtime = S._dt(st["intime"]), S._dt(st["outtime"])
        canon = generate_prediction_rows_for_stay(RetainedICUStay(subject_id=st["subject_id"], stay_id=sid, intime=intime, outtime=outtime))
        canon_t = [c.prediction_time for c in canon]
        if canon_t != [S._dt(r["prediction_time"]) for r in rows]:
            raise SystemExit(f"cutoff mismatch for {sid}")
        facts = []
        for r in rows:
            m = masks[(sid, r["prediction_time"])]
            tr = trends.get(f"{sid}||{r['prediction_time']}")
            if tr is None:
                break
            facts.append({"cutoff_index": r["grid_index"], "prediction_time": S.iso_s(S._dt(r["prediction_time"])), "icu_elapsed_hours": r["icu_elapsed_hours"],
                          "sofa_t": float(tr["t"]), **m})
        if len(facts) != len(rows):
            continue
        sofa = [f["sofa_t"] for f in facts]
        onsets = S.qualifying_onsets(supports.get(sid, []))
        cut_dt = [S._dt(f["prediction_time"]) for f in facts]
        post_early = [(c, t) for c, t in onsets if t > cut_dt[1] and t < cut_dt[-1]]
        cat[sid] = {"st": st, "facts": facts, "sofa": sofa, "n": len(facts), "onsets": onsets, "post_early_onsets": post_early,
                    "net": sofa[-1] - sofa[0], "max_rise": max([b - a for a, b in zip(sofa, sofa[1:])] or [0]),
                    "max_fall": max([a - b for a, b in zip(sofa, sofa[1:])] or [0]),
                    "density": sum(f["observation_density"] for f in facts) / len(facts),
                    "cut_dt": cut_dt, "intime": intime, "outtime": outtime, "rank": rank(sid)}

    chosen: dict = {}
    taken = set()
    notes = []

    def pick(key, title, subtype_label, objective, candidates, score, focus, rationale, substitute=None):
        pool = [sid for sid in candidates if sid not in taken]
        if not pool:
            if substitute is None:
                notes.append({"category": key, "status": "UNSATISFIED_NO_SUBSTITUTE"})
                return
            pool = [sid for sid in substitute["candidates"] if sid not in taken]
            notes.append({"category": key, "status": "SUBSTITUTED", "reason": substitute["reason"]})
            rationale = rationale + " [SUBSTITUTE: " + substitute["reason"] + "]"
            score = substitute["score"]
        best = sorted(pool, key=lambda s: (-score(cat[s]), cat[s]["rank"]))[0]
        taken.add(best)
        chosen[key] = (best, title, objective, focus, rationale, score(cat[best]))

    allc = list(cat)
    onset_ok = [s for s in allc if cat[s]["post_early_onsets"]]
    pick("support_initiation", "New organ-support initiation after stable early course", None,
         "Show how a new qualifying vasopressor or invasive-ventilation start appears in the observed record and in the forecast of its risk.",
         onset_ok, lambda c: c["n"],
         "Support-risk forecast before a qualifying onset",
         "qualifying OFF->ON onset after the second cutoff and before the last; score = number of legal cutoffs; tie-break sha256(stay_id)")
    pick("sparse_observations", "Sparse observation history", None,
         "Show forecasts and data-quality signals when few measurements are available in the 48 h window.",
         [s for s in allc], lambda c: -c["density"],
         "Data completeness versus forecast stability",
         "lowest mean observation density among stays with >=5 cutoffs; tie-break sha256(stay_id)")
    pick("long_episode", "Long ICU-like episode", None,
         "Show how remaining-time and risk forecasts evolve across many legal cutoffs.",
         [s for s in allc if cat[s]["n"] >= 9], lambda c: c["n"],
         "Remaining-time forecast over a long course",
         "legal cutoffs >= 9; score = number of cutoffs; tie-break sha256(stay_id)",
         substitute={"candidates": allc, "score": lambda c: c["n"], "reason": "no stay with >=9 legal cutoffs; the stay with the most legal cutoffs was used"})
    pick("improvement", "Gradual improvement", None,
         "Show recovery forecasts when observed severity falls steadily.",
         [s for s in allc if cat[s]["net"] <= -2 and cat[s]["max_rise"] <= 1 and not cat[s]["onsets"]],
         lambda c: -c["net"] * 10 + c["n"],
         "Falling SOFA with no organ-support start",
         "SOFA(t) falls >=2 first-to-last cutoff, no consecutive rise >1, no qualifying support onset; score = net fall; tie-break sha256(stay_id)",
         substitute={"candidates": [s for s in allc if cat[s]["net"] <= -1 and not cat[s]["post_early_onsets"]], "score": lambda c: -c["net"] * 10 + c["n"],
                     "reason": "no stay met the >=2 point criterion; the largest net fall without a qualifying onset was used"})
    pick("deterioration", "Progressive deterioration", None,
         "Show recovery and support-risk forecasts when observed severity rises.",
         [s for s in allc if cat[s]["net"] >= 2], lambda c: c["net"] * 10 + c["n"],
         "Rising SOFA across cutoffs",
         "SOFA(t) rises >=2 first-to-last cutoff; score = net rise; tie-break sha256(stay_id)",
         substitute={"candidates": [s for s in allc if cat[s]["net"] >= 1], "score": lambda c: c["net"] * 10 + c["n"],
                     "reason": "no stay met the >=2 point rise; the largest net rise was used"})
    for key, group, label in (("heart_failure", "SYNTHETIC_HF", "Heart-failure-like episode"), ("ischemic", "SYNTHETIC_ISCHEMIC", "Ischemic episode"), ("arrhythmia", "SYNTHETIC_ARRHYTHMIA", "Arrhythmia-dominant episode")):
        pick(key, label, None, f"Illustrate a {group.split('_')[1].lower()} cardiac-subtype trajectory with a non-flat observed SOFA course.",
             [s for s in allc if cat[s]["st"]["cardiac_condition_group"] == group and cat[s]["n"] >= 6 and cat[s]["max_fall"] + cat[s]["max_rise"] >= 1],
             lambda c: (min(c["n"], 8)) * 10 + abs(c["net"]),
             "Subtype-specific observed trajectory",
             f"subtype {group}, >=6 legal cutoffs, observed SOFA changes between cutoffs; score = min(cutoffs,8)*10+|net SOFA change|; tie-break sha256(stay_id)")

    order = ["improvement", "deterioration", "heart_failure", "ischemic", "arrhythmia", "support_initiation", "sparse_observations", "long_episode"]
    scenarios = []
    for i, key in enumerate([k for k in order if k in chosen], 1):
        sid, title, objective, focus, rationale, score = chosen[key]
        c = cat[sid]
        st = c["st"]
        if key == "support_initiation":
            comp, when = c["post_early_onsets"][0]
            ev = {"type": "support_onset", "hours_since_admission": S._hours(c["intime"], when), "description": f"First qualifying {comp.replace('_', ' ')} onset in the observed record."}
        elif key == "sparse_observations":
            f = min(c["facts"], key=lambda f: f["observation_density"])
            ev = {"type": "lowest_observation_density", "hours_since_admission": f["icu_elapsed_hours"], "description": f"Lowest observed-cell density ({f['observation_density']:.3f}) at cutoff {f['cutoff_index']}."}
        elif key == "long_episode":
            ev = {"type": "episode_end", "hours_since_admission": S._hours(c["intime"], c["outtime"]), "description": f"Episode ends at hour {S._hours(c['intime'], c['outtime']):.1f} after {c['n']} legal cutoffs."}
        else:
            k = max(range(c["n"] - 1), key=lambda j: abs(c["sofa"][j + 1] - c["sofa"][j]))
            ev = {"type": "largest_sofa_step", "hours_since_admission": c["facts"][k + 1]["icu_elapsed_hours"],
                  "description": f"Largest SOFA change between consecutive cutoffs ({c['sofa'][k + 1] - c['sofa'][k]:+g})."}
        scenarios.append({
            "scenario_id": f"SCN-{i:02d}", "patient_alias": f"DEMO-V3-{i:02d}", "category": key, "title": title,
            "subtype": st["cardiac_condition_group"], "objective": objective, "source_cohort": S.SOURCE_COHORT,
            "stay_id": sid, "subject_id": st["subject_id"], "age_years": st["age_years"], "sex_category": st["sex_category"],
            "intime": st["intime"], "outtime": st["outtime"],
            "legal_cutoffs": [f["prediction_time"] for f in c["facts"]], "n_cutoffs": c["n"],
            "selection_rationale": rationale,
            "measured_criteria": {"net_sofa_change": c["net"], "max_consecutive_rise": c["max_rise"], "mean_observation_density": round(c["density"], 6),
                                  "qualifying_onsets": [{"component": comp, "hours_since_admission": S._hours(c["intime"], t)} for comp, t in c["onsets"]],
                                  "selection_score": score},
            "key_event": ev, "analytical_focus": focus, "status": S.STATUS, "cutoff_facts": c["facts"],
        })
    lineage = {name: S.sha256_file(p) for name, p in paths.items()}
    lineage["sofa_cache_sofa_LCE_json"] = S.sha256_file(SOFA_CACHE)
    manifest = {
        "artifact_version": "performance_v3_product_showcase_manifest_v1", "status": S.STATUS, "source_cohort": S.SOURCE_COHORT,
        "selection_policy": {
            "independent_of_model_performance": True, "min_legal_cutoffs": MIN_CUTOFFS, "tie_break": "sha256(stay_id) ascending",
            "population": "LCE: held-out development evaluation cohort; never used for V3 training, selection or any sealed final test",
            "excluded": "V1 corpus incl. V1 test split, V2 fresh test, V3 sealed test, other development cohorts, custom records",
            "cutoff_source": "data.timestamps.generate_prediction_rows_for_stay (recomputed and asserted equal)",
            "sofa_source": "frozen sofa_at(t) via artifacts/performance_v3/cache/sofa_LCE.json trends['t'] (observed state, not a prediction)",
        },
        "substitutions": notes, "n_scenarios": len(scenarios), "scenarios": scenarios, "lineage": lineage,
        "does_not_modify": "configs/performance_v2/v2_demo_manifest_v1.json",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (OUT.parent / (OUT.name + ".sha256")).write_text(S.sha256_file(OUT) + "  " + OUT.name + "\n")
    for s in scenarios:
        print(s["scenario_id"], s["patient_alias"], s["subtype"], s["n_cutoffs"], s["category"], s["measured_criteria"]["net_sofa_change"], s["key_event"]["type"])
    print("substitutions:", notes)


if __name__ == "__main__":
    main()
