"""PRIVILEGED, NON-DEPLOYABLE replay of the frozen synthetic support/latent process.

DEVELOPMENT_PRIVILEGED_DIAGNOSTIC_ONLY. Re-implements the hour loop of
data.synthetic.support.simulate_with_support as ONE class so that
  (a) driven by the original RNG streams it must reproduce the generator's states and support rows
      bit-for-bit (verified in tests and in the diagnostic run), and
  (b) driven by a fresh RNG from an exact internal snapshot it yields Monte Carlo continuations.
No frozen module is modified. Nothing here may feed a deployable model or API.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from data.synthetic.latent_state import initial_state
from data.synthetic.rng import subject_rng
from data.synthetic.support import _sigmoid, _truncated_lognormal

QUALIFYING_VASO_EXCLUDED = {"dobutamine"}


def candidate_map(config, n_subjects: int) -> List[dict]:
    """Subject ordinal -> candidate ordinal and admission-time latents, mirroring generator.generate()."""

    out: List[dict] = []
    cand = 0
    p = config.values["population"]
    d = config.values["duration"]
    while len(out) < n_subjects:
        cand += 1
        irng = subject_rng(config.seed, cand, "identity")
        irng.integers(p["age_min"], p["age_max"] + 1)
        gi = int(irng.choice(len(p["condition_groups"]), p=p["condition_probabilities"]))
        irng.choice(p["sex_categories"], p=p["sex_probabilities"])
        z0, equilibrium, propensity = initial_state(config, cand, gi)
        erng = subject_rng(config.seed, cand, "episode")
        log_hours = float(d["intercept"] + np.dot(d["latent_coefficients"], z0) + d["support_propensity_coefficient"] * propensity + erng.normal(0, d["noise_scale"]))
        duration = round(max(d["minimum_hours"], min(d["maximum_hours"], math.exp(log_hours))), 6)
        if duration < float(config.values["minimum_accepted_episode_hours"]):
            continue
        out.append({"subject_ordinal": len(out) + 1, "candidate_ordinal": cand, "z0": z0, "equilibrium": equilibrium,
                    "propensity": propensity, "duration": duration, "group_index": gi,
                    "log_hours_mean": float(d["intercept"] + np.dot(d["latent_coefficients"], z0) + d["support_propensity_coefficient"] * propensity)})
    return out


class SupportSim:
    """Hour-stepped latent + support dynamics (verbatim port of the frozen loop body)."""

    def __init__(self, config, contract, z0, equilibrium, lrng, srng, hours_horizon: int):
        p = config.values["latent"]
        self.k = np.asarray(p["mean_reversion"], dtype=float)
        self.coupling = np.asarray(p["coupling_matrix"], dtype=float)
        self.innovation = np.asarray(p["innovation_loading"], dtype=float)
        self.shock_p = float(p["shock_probability_per_hour"])
        self.shock_scale = float(p["shock_scale"])
        self.equilibrium = np.asarray(equilibrium, dtype=float)
        self.lrng, self.srng = lrng, srng
        self.drift = lrng.normal(0.0, float(p["drift_scale"]), 5)
        self.shock_remaining = 0
        self.shock = np.zeros(5)
        self.subject_effect = {"vaso": float(srng.normal(0, .35)), "resp": float(srng.normal(0, .35))}
        self.response = {"vaso": _truncated_lognormal(srng, 1.0, .25, (.5, 1.5)), "resp": _truncated_lognormal(srng, 1.0, .25, (.5, 1.5))}
        agents = contract["vasoactive_vocabulary"]["concepts"]
        self.agent_names = [x["canonical_name"] for x in agents]
        self.agent_p = [x["selection_probability"] for x in agents]
        self.agent_spec = {x["canonical_name"]: x for x in agents}
        rv = contract["respiratory_vocabulary"]["category_selection_probabilities_when_respiratory_support_starts"]
        self.categories = list(rv)
        self.category_p = [rv[x] for x in self.categories]
        self.max_duration = int(contract["cessation"]["maximum_untruncated_duration_hours"])
        self.lag = int(contract["intervention_effects"]["lag_hours"])
        self.rate_ar = contract["rate_process"]["log_rate_ar1"]
        self.states = np.empty((hours_horizon + 1, 5), dtype=float)
        self.states[0] = np.asarray(z0, dtype=float)
        self.active_agents: Dict[str, dict] = {}
        self.active_resp: Dict[str, dict] = {}
        self.segments: List[dict] = []
        self.snapshots: Dict[int, dict] = {}
        self.onsets: List[Tuple[str, int]] = []  # (component, boundary hour) for qualifying OFF->ON

    # -- snapshot of everything known at the START of hour h (before any draw of iteration h)
    def snapshot(self, h: int) -> dict:
        return {
            "h": h, "z": self.states[h].copy(), "recent": self.states[max(0, h - 5):h + 1].copy(),
            "shock_remaining": int(self.shock_remaining), "shock": self.shock.copy(),
            "active_agents": copy.deepcopy(self.active_agents), "active_resp": copy.deepcopy(self.active_resp),
            "drift": self.drift.copy(), "equilibrium": self.equilibrium.copy(),
            "subject_effect": dict(self.subject_effect), "response": dict(self.response),
        }

    def _emit(self, domain, name, h, rate=None):
        self.segments.append({"domain": domain, "name": name, "hour": h, "rate": rate})

    def _qualifying_vaso_active(self) -> bool:
        return any(n not in QUALIFYING_VASO_EXCLUDED for n in self.active_agents)

    def step(self, h: int, emit: bool = True) -> Optional[int]:
        """Advance iteration h. Returns nothing; decisions take effect at boundary h+1."""

        states = self.states
        for name, state in sorted(self.active_agents.items()):
            spec = self.agent_spec[name]
            severity = states[h, 0]
            if state.get("log_rate") is None:
                base = _truncated_lognormal(self.srng, spec["rate_distribution"]["median"], spec["rate_distribution"]["log_sigma"], spec["valid_rate_domain"])
                state["log_rate"] = math.log(base)
            else:
                target = math.log(float(spec["rate_distribution"]["median"])) + .12 * float(severity)
                state["log_rate"] = .85 * state["log_rate"] + .15 * target + self.srng.normal(0, float(self.rate_ar["innovation_standard_deviation"]))
            rate = min(float(spec["valid_rate_domain"][1]), max(float(spec["valid_rate_domain"][0]), math.exp(state["log_rate"])))
            state["log_rate"] = math.log(rate)
            if emit:
                self._emit("VASOPRESSOR", name, h, round(rate, 9))
        if emit:
            for name in sorted(self.active_resp):
                self._emit("RESPIRATORY", name, h)

        qual_before = self._qualifying_vaso_active()
        inv_before = "INVASIVE" in self.active_resp

        effect = np.zeros(5)
        if any(h - x["start"] >= self.lag for x in self.active_agents.values()):
            effect += self.response["vaso"] * np.asarray([-.015, -.025, 0, 0, 0])
        if "INVASIVE" in self.active_resp and h - self.active_resp["INVASIVE"]["start"] >= self.lag:
            effect += self.response["resp"] * np.asarray([-.010, 0, -.025, 0, 0])
        effect = np.clip(effect, -.04, .04)
        if self.shock_remaining <= 0 and self.lrng.random() < self.shock_p:
            self.shock_remaining = int(self.lrng.integers(2, 9))
            self.shock = self.lrng.normal(0.0, self.shock_scale, 5)
        pulse = self.shock / max(self.shock_remaining, 1) if self.shock_remaining > 0 else 0.0
        states[h + 1] = (states[h] + self.k * (self.equilibrium - states[h]) + self.drift + self.coupling @ np.tanh(states[h])
                         + pulse + self.innovation * self.lrng.normal(size=5) + effect)
        self.shock_remaining -= 1

        z = states[h]
        for name in list(self.active_agents):
            on = h + 1 - self.active_agents[name]["start"]
            logit = -3.20 - .25 * z[0] - .25 * z[1] + .055 * min(on, 48) + self.srng.normal(0, .15)
            if on >= self.max_duration or (on >= 1 and self.srng.random() < _sigmoid(float(logit))):
                del self.active_agents[name]
        for name in list(self.active_resp):
            on = h + 1 - self.active_resp[name]["start"]
            logit = -3.40 - .25 * z[0] - .35 * z[2] + .050 * min(on, 48) + self.srng.normal(0, .15)
            if on >= self.max_duration or (on >= 1 and self.srng.random() < _sigmoid(float(logit))):
                del self.active_resp[name]
        recent = states[max(0, h - 5):h + 1]
        v_logit = -4.60 + .55 * z[0] + .75 * z[1] + .20 * float(recent[:, 0].mean()) + self.subject_effect["vaso"] + self.srng.normal(0, .15)
        if self.srng.random() < _sigmoid(float(v_logit)):
            name = str(self.srng.choice(self.agent_names, p=self.agent_p))
            self.active_agents.setdefault(name, {"start": h + 1, "log_rate": None})
        r_logit = -4.80 + .55 * z[0] + .90 * z[2] + .20 * float(recent[:, 2].mean()) + self.subject_effect["resp"] + self.srng.normal(0, .15)
        if self.srng.random() < _sigmoid(float(r_logit)):
            name = str(self.srng.choice(self.categories, p=self.category_p))
            self.active_resp.setdefault(name, {"start": h + 1})

        if not qual_before and self._qualifying_vaso_active():
            self.onsets.append(("vaso", h + 1))
        if not inv_before and "INVASIVE" in self.active_resp:
            self.onsets.append(("invasive", h + 1))
        return None


def replay_stay(config, contract, cand: Mapping[str, object], intime: datetime, hours: int, snapshot_hours=()):
    """Original-stream replay. Returns (states, support_rows_like, snapshots, sim)."""

    lrng = subject_rng(config.seed, cand["candidate_ordinal"], "latent", child=1)
    srng = subject_rng(config.seed, cand["candidate_ordinal"], "support", child=0)
    sim = SupportSim(config, contract, cand["z0"], cand["equilibrium"], lrng, srng, hours)
    wanted = set(int(x) for x in snapshot_hours)
    for h in range(hours):
        if h in wanted:
            sim.snapshots[h] = sim.snapshot(h)
        sim.step(h, emit=True)
    rows = []
    for ordinal, item in enumerate(sim.segments, 1):
        start = intime + timedelta(hours=item["hour"])
        end = min(intime + timedelta(hours=item["hour"] + 1), intime + timedelta(hours=hours))
        if end <= start:
            continue
        vaso = item["domain"] == "VASOPRESSOR"
        rows.append({"support_type": item["domain"], "name": item["name"], "start": start, "end": end,
                     "rate": item["rate"] if vaso else None, "ordinal": ordinal})
    return sim.states, rows, sim.snapshots, sim


def continuation_label(snapshot: Mapping[str, object], config, contract, rng, duration: float, horizon_h: int = 24) -> Tuple[int, int]:
    """One Monte Carlo continuation from `snapshot` (hour h0 = t). Returns (positive, negative) in {0,1};
    (0,0) means censored or not-at-risk. `duration` is the sampled admission duration (hours)."""

    h0 = int(snapshot["h"])
    sim = SupportSim.__new__(SupportSim)
    # build from a throw-away constructor path without consuming original streams
    p = config.values["latent"]
    sim.k = np.asarray(p["mean_reversion"], dtype=float)
    sim.coupling = np.asarray(p["coupling_matrix"], dtype=float)
    sim.innovation = np.asarray(p["innovation_loading"], dtype=float)
    sim.shock_p = float(p["shock_probability_per_hour"])
    sim.shock_scale = float(p["shock_scale"])
    sim.equilibrium = snapshot["equilibrium"]
    sim.lrng = rng
    sim.srng = rng
    sim.drift = snapshot["drift"]
    sim.shock_remaining = int(snapshot["shock_remaining"])
    sim.shock = snapshot["shock"].copy()
    sim.subject_effect = snapshot["subject_effect"]
    sim.response = snapshot["response"]
    agents = contract["vasoactive_vocabulary"]["concepts"]
    sim.agent_names = [x["canonical_name"] for x in agents]
    sim.agent_p = [x["selection_probability"] for x in agents]
    sim.agent_spec = {x["canonical_name"]: x for x in agents}
    rv = contract["respiratory_vocabulary"]["category_selection_probabilities_when_respiratory_support_starts"]
    sim.categories = list(rv)
    sim.category_p = [rv[x] for x in sim.categories]
    sim.max_duration = int(contract["cessation"]["maximum_untruncated_duration_hours"])
    sim.lag = int(contract["intervention_effects"]["lag_hours"])
    sim.rate_ar = contract["rate_process"]["log_rate_ar1"]
    recent = snapshot["recent"]
    sim.states = np.empty((horizon_h + 1 + len(recent), 5), dtype=float)
    base = len(recent) - 1  # index of hour h0 inside the buffer
    sim.states[: len(recent)] = recent
    sim.active_agents = copy.deepcopy(snapshot["active_agents"])
    sim.active_resp = copy.deepcopy(snapshot["active_resp"])
    sim.segments, sim.snapshots, sim.onsets = [], {}, []
    vaso_at_risk = not sim._qualifying_vaso_active()
    inv_at_risk = "INVASIVE" not in sim.active_resp
    if not (vaso_at_risk or inv_at_risk):
        return 0, 0
    # states[h] indexing in step() is absolute; shift by emulating absolute hours inside the buffer
    for j in range(horizon_h):
        h_abs = h0 + j
        if h_abs >= duration:  # generator loops range(ceil(duration)); no decisions at/after outtime
            break
        _step_buffer(sim, base + j, h_abs)
    pos = 0
    for comp, b in sim.onsets:
        if comp == "vaso" and not vaso_at_risk:
            continue
        if comp == "invasive" and not inv_at_risk:
            continue
        if b <= h0 + horizon_h and b < duration:
            pos = 1
            break
    if pos:
        return 1, 0
    if duration >= h0 + horizon_h:
        return 0, 1
    return 0, 0


def _step_buffer(sim: SupportSim, idx: int, h_abs: int) -> None:
    """step() with buffer-relative state indexing but absolute hour for start/lag bookkeeping."""

    # temporarily swap: run the verbatim step on a view whose index equals idx
    # (step uses `h` for both the state index and start-hour arithmetic, so shift start hours).
    shift = h_abs - idx
    for d in (sim.active_agents, sim.active_resp):
        for v in d.values():
            v["start"] -= shift
    existing = sim.onsets
    sim.onsets = []
    sim.step(idx, emit=False)
    for d in (sim.active_agents, sim.active_resp):
        for v in d.values():
            v["start"] += shift
    sim.onsets = existing + [(c, b + shift) for c, b in sim.onsets]
