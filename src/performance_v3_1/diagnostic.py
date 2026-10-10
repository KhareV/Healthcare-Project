"""Privileged latent-state diagnostic machinery (DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY).

Nothing here is a serving input. Oracle features come from an exact replay of the frozen
generator's hidden state AT the cutoff; labels/eligibility come from the canonical pre-split
package and are used only as targets / to define the evaluation population.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.synthetic.config import load_runtime_config
from data.synthetic.support import load_support_contract
from performance_v3_1 import protocol as P
from performance_v3_1.replay import QUALIFYING_VASO_EXCLUDED, candidate_map, continuation_label, replay_stay

FORBIDDEN_NAME_TOKENS = ("outtime", "duration", "label", "eligib", "future", "recovery", "seed", "rng")
A_NAMES = tuple(f"z_t_{i}" for i in range(5)) + ("vaso_on_t", "invasive_on_t")
B_EXTRA_NAMES = (
    tuple(f"recent6_mean_z{i}" for i in range(3)) + tuple(f"equilibrium_{i}" for i in range(5)) + ("support_propensity",)
    + tuple(f"drift_{i}" for i in range(5)) + ("shock_remaining",) + tuple(f"shock_pulse_{i}" for i in range(5))
    + ("subject_effect_vaso", "subject_effect_resp", "response_vaso", "response_resp", "vaso_on_hours", "invasive_on_hours",
       "active_agent_count", "elapsed_hours") + tuple(f"z_admission_{i}" for i in range(5))
)
B_NAMES = A_NAMES + B_EXTRA_NAMES


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class CohortContext:
    def __init__(self, root: Path, cohort_id: str, *, cohort_dir: Optional[Path] = None, config_path: Optional[Path] = None):
        self.root = Path(root)
        self.cohort_id = cohort_id
        self.dir = Path(cohort_dir) if cohort_dir else P.cohort_dir(self.root, cohort_id)
        self.config = load_runtime_config(Path(config_path) if config_path else self.root / P.CONFIG_DIR_REL / f"{cohort_id.lower()}_v1.json", self.root)
        self.contract = load_support_contract(self.root / self.config.values["support_process_path"], self.root / self.config.values["latent_process_path"])
        self.cands = candidate_map(self.config, self.config.n_subjects)
        ns = self.config.subject_id_namespace
        self.by_stay = {f"{ns}-E-{c['subject_ordinal']:08d}": c for c in self.cands}
        self.episodes = {r["stay_id"]: r for r in jsonl(self.dir / "raw/episodes.jsonl")}
        self.raw_support: Dict[str, list] = {}
        for r in jsonl(self.dir / "raw/support_intervals.jsonl"):
            self.raw_support.setdefault(r["stay_id"], []).append(r)

    def intime_outtime(self, stay_id: str):
        e = self.episodes[stay_id]
        return _dt(e["intime"]), _dt(e["outtime"])


def verify_replay(ctx: CohortContext, max_stays: Optional[int] = None) -> Dict[str, object]:
    """Bitwise check: replica reproduces raw support rows (and implied duration) for every stay."""

    checked = mismatched = 0
    first_bad = None
    for stay_id, cand in list(ctx.by_stay.items())[: max_stays]:
        intime, outtime = ctx.intime_outtime(stay_id)
        duration = (outtime - intime).total_seconds() / 3600.0
        hours = int(math.ceil(duration))
        if abs(duration - cand["duration"]) > 1e-4:
            mismatched += 1; first_bad = first_bad or (stay_id, "duration"); continue
        _, rows, _, _ = replay_stay(ctx.config, ctx.contract, cand, intime, hours)
        got = []
        for r in rows:
            if r["start"] >= outtime:
                continue
            end = min(r["end"], outtime)
            got.append((r["support_type"], r["name"], r["start"].isoformat(timespec="microseconds").replace("+00:00", "Z"),
                        end.isoformat(timespec="microseconds").replace("+00:00", "Z"), r["rate"]))
        want = sorted((r["support_type"], r["support_category"], r["interval_start"], r["interval_end"], r["rate_value"]) for r in ctx.raw_support.get(stay_id, []))
        got.sort(key=lambda x: (x[2], x[0], x[1]))
        want = sorted(want, key=lambda x: (x[2], x[0], x[1]))
        checked += 1
        if got != want:
            mismatched += 1
            first_bad = first_bad or (stay_id, "support_rows")
    return {"stays_checked": checked, "mismatched": mismatched, "first_mismatch": first_bad, "bitwise_identical": mismatched == 0}


def _state_features(snapshot: Mapping[str, object], cand: Mapping[str, object], h: int) -> Dict[str, float]:
    z = snapshot["z"]
    agents = snapshot["active_agents"]; resp = snapshot["active_resp"]
    qual = {n: s for n, s in agents.items() if n not in QUALIFYING_VASO_EXCLUDED}
    f: Dict[str, float] = {}
    for i in range(5):
        f[f"z_t_{i}"] = float(z[i])
    f["vaso_on_t"] = float(bool(qual)); f["invasive_on_t"] = float("INVASIVE" in resp)
    rec = snapshot["recent"]
    for i in range(3):
        f[f"recent6_mean_z{i}"] = float(rec[:, i].mean())
    for i in range(5):
        f[f"equilibrium_{i}"] = float(snapshot["equilibrium"][i])
        f[f"drift_{i}"] = float(snapshot["drift"][i])
        f[f"z_admission_{i}"] = float(cand["z0"][i])
    f["support_propensity"] = float(cand["propensity"])
    rem = int(snapshot["shock_remaining"])
    f["shock_remaining"] = float(max(rem, 0))
    pulse = snapshot["shock"] / max(rem, 1) if rem > 0 else np.zeros(5)
    for i in range(5):
        f[f"shock_pulse_{i}"] = float(pulse[i])
    f["subject_effect_vaso"] = float(snapshot["subject_effect"]["vaso"]); f["subject_effect_resp"] = float(snapshot["subject_effect"]["resp"])
    f["response_vaso"] = float(snapshot["response"]["vaso"]); f["response_resp"] = float(snapshot["response"]["resp"])
    f["vaso_on_hours"] = float(h - min(s["start"] for s in qual.values())) if qual else 0.0
    f["invasive_on_hours"] = float(h - resp["INVASIVE"]["start"]) if "INVASIVE" in resp else 0.0
    f["active_agent_count"] = float(len(agents))
    f["elapsed_hours"] = float(h)
    return f


def snapshot_table(ctx: CohortContext, rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], dict]:
    """(stay_id, prediction_time) -> {'features': {...}, 'snapshot': {...}, 'cand': {...}, 'h': int}."""

    by_stay: Dict[str, List[Mapping[str, object]]] = {}
    for r in rows:
        by_stay.setdefault(r["stay_id"], []).append(r)
    out: Dict[Tuple[str, str], dict] = {}
    for stay_id, srows in by_stay.items():
        cand = ctx.by_stay[stay_id]
        intime, outtime = ctx.intime_outtime(stay_id)
        hours = int(math.ceil((outtime - intime).total_seconds() / 3600.0))
        hs = {}
        for r in srows:
            h = (_dt(r["prediction_time"]) - intime).total_seconds() / 3600.0
            if abs(h - round(h)) > 1e-6:
                raise ValueError("prediction_time is not on the hourly latent grid")
            hs[(stay_id, r["prediction_time"])] = int(round(h))
        _, _, snaps, _ = replay_stay(ctx.config, ctx.contract, cand, intime, hours, snapshot_hours=set(hs.values()))
        for key, h in hs.items():
            if h not in snaps:
                raise ValueError(f"no snapshot at hour {h} for {stay_id}")
            out[key] = {"features": _state_features(snaps[h], cand, h), "snapshot": snaps[h], "cand": cand, "h": h}
    return out


def sample_duration(rng, cand: Mapping[str, object], config, elapsed_h: float) -> float:
    """Admission duration conditioned on: accepted (>= minimum_accepted) and legal cutoff exists (>= elapsed+6h)."""

    from scipy.stats import norm

    d = config.values["duration"]
    sigma = float(d["noise_scale"])
    lower = max(float(config.values["minimum_accepted_episode_hours"]), elapsed_h + 6.0)
    lower = min(lower, float(d["maximum_hours"]))
    mean = float(cand["log_hours_mean"])
    z_low = (math.log(lower) - mean) / sigma
    u = rng.uniform(norm.cdf(z_low), 1.0 - 1e-12)
    log_h = mean + sigma * float(norm.ppf(min(max(u, 1e-12), 1 - 1e-12)))
    return round(max(float(d["minimum_hours"]), min(float(d["maximum_hours"]), math.exp(log_h))), 6)


def oracle_c_probability(ctx: CohortContext, entry: Mapping[str, object], m: int, seed_parts: Sequence[int]) -> Tuple[float, int, int]:
    """Monte Carlo P(positive | state, eligible). Returns (p, n_pos, n_neg)."""

    pos = neg = 0
    for j in range(m):
        rng = np.random.default_rng(np.random.SeedSequence(list(seed_parts) + [j]))
        duration = sample_duration(rng, entry["cand"], ctx.config, float(entry["h"]))
        a, b = continuation_label(entry["snapshot"], ctx.config, ctx.contract, rng, duration)
        pos += a; neg += b
    p = pos / (pos + neg) if pos + neg else float("nan")
    return p, pos, neg


def forbidden_token_scan(names: Iterable[str]) -> List[str]:
    return [n for n in names if any(t in n.lower() for t in FORBIDDEN_NAME_TOKENS)]
