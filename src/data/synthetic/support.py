"""Frozen synthetic support process integrated with hourly latent dynamics."""
from __future__ import annotations

from datetime import datetime, timedelta
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from .provenance import sha256_file
from .rng import subject_rng

STATE_NAMES = ("systemic_severity", "cardiac_stress", "respiratory_burden", "renal_burden", "neurologic_burden")


def load_support_contract(path: Path, latent_path: Path) -> Mapping[str, object]:
    value = yaml.safe_load(path.read_text())
    if value.get("status") != "FROZEN_SYNTHETIC_AUTHORIZED":
        raise ValueError("support contract is not frozen/authorized")
    if value.get("latent_process_version") != "synthetic_latent_process_v1":
        raise ValueError("support contract latent version mismatch")
    if value.get("latent_process_sha256") != sha256_file(latent_path):
        raise ValueError("support contract latent SHA-256 mismatch")
    return value


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def _truncated_lognormal(rng, median, sigma, bounds):
    for _ in range(10000):
        value = float(rng.lognormal(math.log(float(median)), float(sigma)))
        if float(bounds[0]) <= value <= float(bounds[1]):
            return value
    raise RuntimeError("truncated lognormal rejection limit exceeded")


def simulate_with_support(config, subject_ordinal: int, z0, equilibrium, hours: int, contract, intime: datetime):
    """Return latent states and canonical one-hour support rate/state segments.

    Within hour h: use state/support known at h, apply support effects whose
    two-hour lag has elapsed, advance physiology, then decide transitions at
    h+1. Thus a new start cannot affect the state that caused that start.
    """
    p = config.values["latent"]
    lrng = subject_rng(config.seed, subject_ordinal, "latent", child=1)
    srng = subject_rng(config.seed, subject_ordinal, "support", child=0)
    states = np.empty((hours + 1, 5), dtype=float); states[0] = z0
    k = np.asarray(p["mean_reversion"], dtype=float)
    coupling = np.asarray(p["coupling_matrix"], dtype=float)
    innovation = np.asarray(p["innovation_loading"], dtype=float)
    drift = lrng.normal(0.0, float(p["drift_scale"]), 5)
    shock_remaining = 0; shock = np.zeros(5)
    subject_effect = {"vaso": float(srng.normal(0, .35)), "resp": float(srng.normal(0, .35))}
    response = {"vaso": _truncated_lognormal(srng, 1.0, .25, (.5, 1.5)), "resp": _truncated_lognormal(srng, 1.0, .25, (.5, 1.5))}
    agents = contract["vasoactive_vocabulary"]["concepts"]
    agent_names = [x["canonical_name"] for x in agents]; agent_p = [x["selection_probability"] for x in agents]
    agent_spec = {x["canonical_name"]: x for x in agents}
    categories = list(contract["respiratory_vocabulary"]["category_selection_probabilities_when_respiratory_support_starts"])
    category_p = [contract["respiratory_vocabulary"]["category_selection_probabilities_when_respiratory_support_starts"][x] for x in categories]
    active_agents = {}; active_resp = {}; segments = []
    max_duration = int(contract["cessation"]["maximum_untruncated_duration_hours"])
    lag = int(contract["intervention_effects"]["lag_hours"])
    rate_ar = contract["rate_process"]["log_rate_ar1"]

    def emit(domain, name, h, rate=None):
        segments.append({"domain": domain, "name": name, "hour": h, "rate": rate})

    for h in range(hours):
        # Export the state/rate genuinely active through this hour.
        for name, state in sorted(active_agents.items()):
            spec = agent_spec[name]; severity = states[h, 0]
            if state.get("log_rate") is None:
                base = _truncated_lognormal(srng, spec["rate_distribution"]["median"], spec["rate_distribution"]["log_sigma"], spec["valid_rate_domain"])
                state["log_rate"] = math.log(base)
            else:
                target = math.log(float(spec["rate_distribution"]["median"])) + .12 * float(severity)
                state["log_rate"] = .85 * state["log_rate"] + .15 * target + srng.normal(0, float(rate_ar["innovation_standard_deviation"]))
            rate = min(float(spec["valid_rate_domain"][1]), max(float(spec["valid_rate_domain"][0]), math.exp(state["log_rate"])))
            state["log_rate"] = math.log(rate); emit("VASOPRESSOR", name, h, round(rate, 9))
        for name in sorted(active_resp): emit("RESPIRATORY", name, h)

        effect = np.zeros(5)
        if any(h - x["start"] >= lag for x in active_agents.values()):
            effect += response["vaso"] * np.asarray([-.015, -.025, 0, 0, 0])
        if "INVASIVE" in active_resp and h - active_resp["INVASIVE"]["start"] >= lag:
            effect += response["resp"] * np.asarray([-.010, 0, -.025, 0, 0])
        effect = np.clip(effect, -.04, .04)
        if shock_remaining <= 0 and lrng.random() < float(p["shock_probability_per_hour"]):
            shock_remaining = int(lrng.integers(2, 9)); shock = lrng.normal(0.0, float(p["shock_scale"]), 5)
        pulse = shock / max(shock_remaining, 1) if shock_remaining > 0 else 0.0
        states[h + 1] = states[h] + k*(equilibrium-states[h]) + drift + coupling@np.tanh(states[h]) + pulse + innovation*lrng.normal(size=5) + effect
        shock_remaining -= 1

        # Cessations and candidates become effective only at boundary h+1.
        z = states[h]
        for name in list(active_agents):
            on = h + 1 - active_agents[name]["start"]
            logit = -3.20 - .25*z[0] - .25*z[1] + .055*min(on, 48) + srng.normal(0, .15)
            if on >= max_duration or (on >= 1 and srng.random() < _sigmoid(float(logit))): del active_agents[name]
        for name in list(active_resp):
            on = h + 1 - active_resp[name]["start"]
            logit = -3.40 - .25*z[0] - .35*z[2] + .050*min(on, 48) + srng.normal(0, .15)
            if on >= max_duration or (on >= 1 and srng.random() < _sigmoid(float(logit))): del active_resp[name]
        recent = states[max(0, h-5):h+1]
        v_logit = -4.60 + .55*z[0] + .75*z[1] + .20*float(recent[:,0].mean()) + subject_effect["vaso"] + srng.normal(0,.15)
        if srng.random() < _sigmoid(float(v_logit)):
            name = str(srng.choice(agent_names, p=agent_p))
            active_agents.setdefault(name, {"start": h+1, "log_rate": None})
        r_logit = -4.80 + .55*z[0] + .90*z[2] + .20*float(recent[:,2].mean()) + subject_effect["resp"] + srng.normal(0,.15)
        if srng.random() < _sigmoid(float(r_logit)):
            name = str(srng.choice(categories, p=category_p))
            active_resp.setdefault(name, {"start": h+1})

    rows = []
    for ordinal, item in enumerate(segments, 1):
        start = intime + timedelta(hours=item["hour"]); end = min(intime + timedelta(hours=item["hour"]+1), intime + timedelta(hours=hours))
        if end <= start: continue
        vaso = item["domain"] == "VASOPRESSOR"
        rows.append({
            "support_event_id": f"SYN-E-{subject_ordinal:08d}-SUP-{ordinal:06d}",
            "subject_id": f"SYN-S-{subject_ordinal:08d}", "stay_id": f"SYN-E-{subject_ordinal:08d}",
            "support_type": item["domain"], "support_category": item["name"],
            "agent_key": item["name"] if vaso else None, "respiratory_category": None if vaso else item["name"],
            "interval_start": start.isoformat(timespec="microseconds").replace("+00:00","Z"),
            "interval_end": end.isoformat(timespec="microseconds").replace("+00:00","Z"),
            "rate_value": item["rate"] if vaso else None, "rate_unit": "ug/kg/min" if vaso else None,
            "normalization_provenance_ref": "synthetic_support_adapter_v2",
        })
    return states, rows
