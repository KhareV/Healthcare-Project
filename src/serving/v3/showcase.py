"""V3 patient-scenario showcase: manifest loading, runtime registration, observed timelines and grounded stories.

Additive and non-scientific. The frozen V2 demo manifest is never touched. Scenarios are drawn from the
LCE cohort (held-out development evaluation; never in V3 training/selection and never in any sealed test).
No model prediction is stored or used here; predictions always come from the live V3 runtime.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

MANIFEST_REL = "configs/performance_v3_product/showcase_manifest_v1.json"
COHORT_REL = "artifacts/performance_v3/cohorts/LCE"
SOURCE_COHORT = "LCE"
STATUS = "RESEARCH_DEMO_DEVELOPMENT_COHORT_NOT_TRAINING_NOT_FINAL_TEST"
ALLOWED_STAY_PREFIX = "SYN-V3-LCE-E-"
ALLOWED_SUBJECT_PREFIX = "SYN-V3-LCE-S-"
# V1 corpus ("SYN-E-"/"SYN-S-", includes the V1 test split), V2 fresh test, V3 sealed test, every other cohort.
FORBIDDEN_PREFIXES = ("SYN-E-", "SYN-S-", "SYN-V2-", "SYN-V3-T-", "SYN-V3-D", "SYN-V31-", "CUSTOM-")
QUALIFYING_EXCLUDED_AGENTS = {"dobutamine"}
KEY_CHANNELS = ("heart_rate", "mean_arterial_pressure", "oxygen_saturation", "respiratory_rate", "lactate", "creatinine", "glasgow_coma_scale", "temperature", "platelet_count")


class ShowcaseError(RuntimeError):
    pass


class ExclusionError(ShowcaseError):
    """A stay/subject outside the permitted development showcase cohort was supplied."""


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_s(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_allowed(stay_id: str, subject_id: str) -> None:
    for value in (stay_id, subject_id):
        if str(value).startswith(FORBIDDEN_PREFIXES):
            raise ExclusionError(f"{value!r} belongs to a forbidden cohort (V1/V2/V3 test, other development, or custom)")
    if not str(stay_id).startswith(ALLOWED_STAY_PREFIX) or not str(subject_id).startswith(ALLOWED_SUBJECT_PREFIX):
        raise ExclusionError(f"{stay_id!r}/{subject_id!r} is not an LCE showcase-cohort identity")


def cohort_paths(root: Path) -> Dict[str, Path]:
    base = Path(root) / COHORT_REL
    return {
        "statics": base / "timeline/canonical_statics.jsonl",
        "timeline": base / "timeline/canonical_timeline.jsonl",
        "support": base / "raw/support_intervals.jsonl",
        "structural_index": base / "cohort/structural_index.jsonl",
        "pre_split": base / "pre_split/pre_split_scientific_package.jsonl",
    }


def stream_jsonl(path: Path, wanted: Optional[Set[str]] = None) -> Iterator[dict]:
    key = '"stay_id":"'
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if wanted is not None:
                i = line.find(key)
                if i < 0:
                    continue
                j = line.find('"', i + len(key))
                if line[i + len(key):j] not in wanted:
                    continue
            line = line.strip()
            if line:
                yield json.loads(line)


def qualifying_onsets(support_rows: Sequence[Mapping[str, object]]) -> List[Tuple[str, datetime]]:
    """OFF->ON component onsets from closed-open segment unions (same qualification as event_dict_v2)."""

    out: List[Tuple[str, datetime]] = []
    comps = (
        ("vasopressor", [r for r in support_rows if r["support_type"] == "VASOPRESSOR" and r.get("agent_key") not in QUALIFYING_EXCLUDED_AGENTS]),
        ("invasive_ventilation", [r for r in support_rows if r["support_type"] == "RESPIRATORY" and r.get("respiratory_category") == "INVASIVE"]),
    )
    for name, rows in comps:
        spans = [(_dt(r["interval_start"]), _dt(r["interval_end"])) for r in rows]
        for start in sorted({s for s, _ in spans}):
            before = any(s < start <= e for s, e in spans)
            if not before:
                out.append((name, start))
    return sorted(out, key=lambda x: x[1])


def support_runs(support_rows: Sequence[Mapping[str, object]]) -> List[dict]:
    """Merge contiguous hourly segments of the same (type, category) into runs (rate = mean)."""

    groups: Dict[Tuple[str, str], List[Tuple[datetime, datetime, Optional[float]]]] = {}
    for r in support_rows:
        cat = r.get("agent_key") or r.get("respiratory_category") or r.get("support_category")
        groups.setdefault((r["support_type"], str(cat)), []).append((_dt(r["interval_start"]), _dt(r["interval_end"]), r.get("rate_value")))
    runs: List[dict] = []
    for (stype, cat), segs in groups.items():
        segs.sort()
        cur = None
        for s, e, rate in segs:
            if cur and s <= cur["end"]:
                cur["end"] = max(cur["end"], e)
                if rate is not None:
                    cur["rates"].append(float(rate))
            else:
                if cur:
                    runs.append(cur)
                cur = {"support_type": stype, "category": cat, "start": s, "end": e, "rates": [float(rate)] if rate is not None else []}
        if cur:
            runs.append(cur)
    runs.sort(key=lambda r: (r["start"], r["support_type"], r["category"]))
    return runs


def qualifying(run: Mapping[str, object]) -> bool:
    if run["support_type"] == "VASOPRESSOR":
        return run["category"] not in QUALIFYING_EXCLUDED_AGENTS
    return run["category"] == "INVASIVE"


# ----------------------------------------------------------------- manifest
def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_REL


@lru_cache(maxsize=4)
def _load_cached(root_s: str, mtime: float) -> dict:
    root = Path(root_s)
    path = manifest_path(root)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise ShowcaseError("showcase manifest or its .sha256 sidecar is missing")
    actual = sha256_file(path)
    if actual != sidecar.read_text().split()[0]:
        raise ShowcaseError("showcase manifest sha256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_cohort") != SOURCE_COHORT or manifest.get("status") != STATUS:
        raise ShowcaseError("showcase manifest has an unexpected source cohort or status")
    paths = cohort_paths(root)
    for name, digest in manifest["lineage"].items():
        if name in paths and paths[name].exists() and sha256_file(paths[name]) != digest:
            raise ShowcaseError(f"showcase lineage hash mismatch for {name}")
    seen = set()
    for scn in manifest["scenarios"]:
        assert_allowed(scn["stay_id"], scn["subject_id"])
        if scn["stay_id"] in seen:
            raise ShowcaseError("duplicate stay in showcase")
        seen.add(scn["stay_id"])
        if scn["n_cutoffs"] != len(scn["legal_cutoffs"]) or scn["n_cutoffs"] < 5:
            raise ShowcaseError(f"{scn['scenario_id']}: invalid cutoff count")
    manifest["_sha256"] = actual
    return manifest


def load_showcase(root: Path, *, verify_cutoffs: bool = True) -> dict:
    root = Path(root)
    path = manifest_path(root)
    manifest = _load_cached(str(root.resolve()), path.stat().st_mtime if path.exists() else 0.0)
    if verify_cutoffs:
        _verify_canonical_cutoffs(root, manifest)
    return manifest


@lru_cache(maxsize=4)
def _verified(root_s: str, sha: str) -> bool:
    return True


def _verify_canonical_cutoffs(root: Path, manifest: Mapping[str, object]) -> None:
    from data.timestamps import RetainedICUStay, generate_prediction_rows_for_stay

    key = (str(Path(root).resolve()), manifest["_sha256"])
    if key in _CUTOFF_OK:
        return
    for scn in manifest["scenarios"]:
        stay = RetainedICUStay(subject_id=scn["subject_id"], stay_id=scn["stay_id"], intime=_dt(scn["intime"]), outtime=_dt(scn["outtime"]))
        canon = [iso_s(r.prediction_time) for r in generate_prediction_rows_for_stay(stay)]
        if canon != list(scn["legal_cutoffs"]):
            raise ShowcaseError(f"{scn['scenario_id']}: legal cutoffs differ from the canonical generator")
    _CUTOFF_OK.add(key)


_CUTOFF_OK: Set[Tuple[str, str]] = set()


def get_scenario(root: Path, scenario_id: str) -> dict:
    for scn in load_showcase(root)["scenarios"]:
        if scn["scenario_id"] == scenario_id or scn["patient_alias"] == scenario_id:
            return scn
    raise KeyError(scenario_id)


# ----------------------------------------------------------------- runtime registration
def register_showcase(runtime, root: Path) -> List[str]:
    """Register every scenario stay through the runtime's demo API.

    Streams only the selected stays from the LCE timeline/raw files. Idempotent."""

    manifest = load_showcase(root)
    wanted = {s["stay_id"] for s in manifest["scenarios"]}
    paths = cohort_paths(root)
    statics = {r["stay_id"]: r for r in stream_jsonl(paths["statics"], wanted)}
    events: Dict[str, list] = {k: [] for k in wanted}
    for r in stream_jsonl(paths["timeline"], wanted):
        events[r["stay_id"]].append(r)
    supports: Dict[str, list] = {k: [] for k in wanted}
    for r in stream_jsonl(paths["support"], wanted):
        supports[r["stay_id"]].append(r)
    registered: List[str] = []
    for scn in manifest["scenarios"]:
        sid = scn["stay_id"]
        if sid in getattr(runtime, "_demo_extra", {}):
            registered.append(sid)
            continue
        row = dict(statics[sid])
        if row["subject_id"] != scn["subject_id"]:
            raise ShowcaseError("statics subject mismatch")
        row["patient_alias"] = scn["patient_alias"]
        if hasattr(runtime, "register_demo_stay"):
            runtime.register_demo_stay(
                stay_id=sid, subject_id=scn["subject_id"], statics_row=row,
                events=events[sid], support_rows=supports[sid],
                legal_cutoffs=tuple(scn["legal_cutoffs"]),
            )
        else:
            # Small test doubles may only expose the earlier ephemeral interface.
            if runtime.ephemeral_entry(sid) is not None:
                registered.append(sid)
                continue
            runtime.register_ephemeral_stay(
                stay_id=sid, subject_id=scn["subject_id"], statics_row=row, events=events[sid],
                legal_cutoffs=tuple(scn["legal_cutoffs"]), owner_user_id=None,
            )
            if supports[sid]:
                runtime.register_ephemeral_support(stay_id=sid, support_rows=supports[sid])
        registered.append(sid)
    return registered


# ----------------------------------------------------------------- timelines
def _hours(intime: datetime, when: datetime) -> float:
    return round((when - intime).total_seconds() / 3600.0, 4)


def _first_visible(cut_times: Sequence[datetime], when: datetime) -> Optional[int]:
    for i, t in enumerate(cut_times):
        if when <= t:
            return i
    return None


def _scenario_raw(root: Path, scn: Mapping[str, object]):
    wanted = {scn["stay_id"]}
    paths = cohort_paths(root)
    events = list(stream_jsonl(paths["timeline"], wanted))
    supports = list(stream_jsonl(paths["support"], wanted))
    return events, supports


def scenario_timeline(root: Path, scenario_id: str) -> dict:
    scn = get_scenario(root, scenario_id)
    intime = _dt(scn["intime"])
    cuts = [_dt(c) for c in scn["legal_cutoffs"]]
    events, supports = _scenario_raw(root, scn)
    series: Dict[str, dict] = {}
    for e in events:
        if e.get("value_numeric") is None:
            continue
        concept = e["canonical_concept"]
        when = _dt(e["event_time"] if e.get("event_time") else e["interval_end"])
        entry = series.setdefault(concept, {"concept": concept, "unit": e.get("unit"), "points": []})
        entry["points"].append({"hours_since_admission": _hours(intime, when), "value": float(e["value_numeric"]), "first_visible_cutoff": _first_visible(cuts, when)})
    for s in series.values():
        s["points"].sort(key=lambda p: p["hours_since_admission"])
    runs = []
    for r in support_runs(supports):
        runs.append({
            "support_type": r["support_type"], "category": r["category"], "qualifying_for_initiation": qualifying(r),
            "start_hour": _hours(intime, r["start"]), "end_hour": _hours(intime, r["end"]),
            "mean_rate": (sum(r["rates"]) / len(r["rates"])) if r["rates"] else None, "rate_unit": "ug/kg/min" if r["rates"] else None,
            "first_visible_cutoff": _first_visible(cuts, r["start"]),
            "end_first_visible_cutoff": next((i for i, t in enumerate(cuts) if r["end"] <= t), None),
        })
    return {
        "scenario_id": scn["scenario_id"], "stay_id": scn["stay_id"], "intime": scn["intime"], "outtime_hours": _hours(intime, _dt(scn["outtime"])),
        "cutoff_hours": [_hours(intime, c) for c in cuts], "legal_cutoffs": scn["legal_cutoffs"],
        "visibility_rule": "a point/interval with first_visible_cutoff == k is visible to the model only at cutoff index >= k; end_first_visible_cutoff gives the first cutoff at which an interval's end is known",
        "series": [series[k] for k in sorted(series)], "support_runs": runs,
        "observed_sofa_by_cutoff": [c["sofa_t"] for c in scn["cutoff_facts"]],
    }


def scenario_story(root: Path, scenario_id: str) -> dict:
    scn = get_scenario(root, scenario_id)
    intime = _dt(scn["intime"])
    cuts = [_dt(c) for c in scn["legal_cutoffs"]]
    outtime = _dt(scn["outtime"])
    events, supports = _scenario_raw(root, scn)
    runs = support_runs(supports)
    by_concept: Dict[str, List[Tuple[datetime, float, str]]] = {}
    for e in events:
        if e.get("value_numeric") is None:
            continue
        when = _dt(e["event_time"] if e.get("event_time") else e["interval_end"])
        by_concept.setdefault(e["canonical_concept"], []).append((when, float(e["value_numeric"]), e.get("unit") or ""))
    for v in by_concept.values():
        v.sort()
    steps = []
    for k, t in enumerate(cuts):
        facts: List[dict] = []
        cf = scn["cutoff_facts"][k]
        facts.append({"kind": "elapsed", "text": f"{cf['icu_elapsed_hours']} h after ICU admission; {k + 1} of {len(cuts)} legal cutoffs."})
        facts.append({"kind": "sofa", "text": f"SOFA computed at this cutoff: {cf['sofa_t']:g} (from data up to the cutoff)."})
        facts.append({"kind": "window", "text": f"{cf['observed_bins']} of 8 six-hour bins in the 48 h window contain at least one observation; {cf['padding_bins']} bins precede ICU admission (structural padding)."})
        for ch in KEY_CHANNELS:
            vals = [x for x in by_concept.get(ch, []) if x[0] <= t]
            if vals:
                when, v, unit = vals[-1]
                facts.append({"kind": "latest_value", "concept": ch, "text": f"Latest {ch.replace('_', ' ')}: {v:.3g} {unit} ({_hours(t, when) * -1:.1f} h before this cutoff)."})
        active = [r for r in runs if r["start"] <= t < r["end"] and qualifying(r)]
        facts.append({"kind": "support_state", "text": ("Qualifying support active: " + ", ".join(sorted({a['category'].lower() for a in active})) + ".") if active
                      else "No qualifying vasopressor or invasive ventilation active."})
        nxt = cuts[k + 1] if k + 1 < len(cuts) else outtime
        revealed: List[dict] = []
        for r in runs:
            if t < r["start"] <= nxt:
                revealed.append({"kind": "support_started", "hindsight": True, "hours_since_admission": _hours(intime, r["start"]),
                                 "text": f"{r['category'].title()} ({r['support_type'].lower()}) began at hour {_hours(intime, r['start']):.1f}."})
            if t < r["end"] <= nxt and r["end"] < outtime:
                revealed.append({"kind": "support_stopped", "hindsight": True, "hours_since_admission": _hours(intime, r["end"]),
                                 "text": f"{r['category'].title()} ({r['support_type'].lower()}) ended at hour {_hours(intime, r['end']):.1f}."})
        if k + 1 < len(cuts):
            d = scn["cutoff_facts"][k + 1]["sofa_t"] - cf["sofa_t"]
            if abs(d) >= 2:
                revealed.append({"kind": "sofa_change", "hindsight": True, "text": f"SOFA changed by {d:+g} by the next cutoff."})
        else:
            revealed.append({"kind": "episode_end", "hindsight": True, "text": f"The episode ended at hour {_hours(intime, outtime):.1f}."})
        steps.append({"cutoff_index": k, "prediction_time": scn["legal_cutoffs"][k], "facts": facts, "revealed_next": revealed,
                      "revealed_next_label": "HINDSIGHT: not available to the model at this cutoff"})
    return {"scenario_id": scn["scenario_id"], "title": scn["title"], "objective": scn["objective"], "steps": steps,
            "note": "Facts are derived only from structured data; no treatment advice or diagnosis is implied."}


def compare(root: Path, a: str, b: str) -> dict:
    sa, sb = get_scenario(root, a), get_scenario(root, b)
    n = min(sa["n_cutoffs"], sb["n_cutoffs"])
    rows = []
    for i in range(n):
        fa, fb = sa["cutoff_facts"][i], sb["cutoff_facts"][i]
        rows.append({"cutoff_index": i, "a": {k: fa[k] for k in ("prediction_time", "icu_elapsed_hours", "sofa_t", "observed_bins", "observation_density")},
                     "b": {k: fb[k] for k in ("prediction_time", "icu_elapsed_hours", "sofa_t", "observed_bins", "observation_density")}})
    meta = lambda s: {k: s[k] for k in ("scenario_id", "patient_alias", "title", "subtype", "stay_id", "n_cutoffs")}
    return {"a": meta(sa), "b": meta(sb), "aligned_cutoffs": n, "rows": rows,
            "note": "Two hand-picked demonstration episodes are not an evaluation sample. Predictions come from /predict."}
