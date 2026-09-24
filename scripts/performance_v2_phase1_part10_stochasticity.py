"""Performance-v2 Phase 1, Part 10: same-history counterfactual stochasticity
test, using generator internals in a sandbox diagnostic mode only.

DEVELOPMENT_DIAGNOSTIC_ONLY / NON_SERVING_DIAGNOSTIC_ORACLE-adjacent. Never
modifies the generator. Never used as a serving feature.

Two separate analyses, both driven by the frozen generator's own governing
equations (read, not edited), applied to TRAIN subjects only:

1. ICU remaining-time noise floor (closed-form/Monte Carlo, exact): the
   generator fixes episode duration ONCE at subject creation as
   duration = clip(exp(intercept + latent_coefficients . z0
                        + support_propensity_coefficient . support_propensity
                        + Normal(0, noise_scale)), min_hours, max_hours)
   (src/data/synthetic/generator.py). Remaining time at any cutoff t is
   duration - elapsed(t), so ANY predictor -- however good -- inherits
   exactly this Normal(0, noise_scale) irreducible log-duration noise once
   z0/support_propensity are already fully known. This is evaluated by
   resampling ONLY that noise term (z0/support_propensity held fixed at
   their true, generation-time values) and comparing predicted-vs-resampled
   remaining time.

2. Latent-state forward dispersion (Monte Carlo replay, approximate): using
   ONLY the no-support latent recursion from
   src/data/synthetic/latent_state.py (mean reversion + subject drift +
   tanh coupling + hourly innovation + occasional multi-hour shocks), the
   state at t is held fixed and propagated forward to t+24h/t+48h under
   independently resampled innovation/shock draws. The resulting dispersion
   of the "systemic_severity" latent dimension (the dimension SOFA-relevant
   vitals depend on most directly, per observations.py's
   latent_dependencies) is reported as a PROXY for how much of future SOFA
   movement is intrinsically unpredictable from state at t. This
   deliberately excludes the support-conditioned state-evolution term in
   simulate_with_support (which also perturbs the state via vasopressor/
   ventilation response effects): faithfully forking that path would
   require re-deriving each subject's exact mid-stream support-decision RNG
   position, a substantially larger undertaking not justified for a
   diagnostic phase. This is documented here as a known scope limitation,
   not silently omitted.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes, load_runtime_config  # noqa: E402
from data.synthetic.latent_state import initial_state  # noqa: E402
from performance_v2.data_loading import load_dev_rows, load_statics  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase1"
N_TRAIN_CUTOFFS = 50
N_CONTINUATIONS = 100
RNG = np.random.default_rng(2026_09_24)


def _icu_noise_floor(config) -> dict:
    d = config.values["duration"]
    noise_scale = float(d["noise_scale"])
    minimum, maximum = float(d["minimum_hours"]), float(d["maximum_hours"])
    # Representative fixed mu (a plausible mid-range value); the noise term's
    # distribution does not depend on mu except through the clip boundary.
    mu = 4.0
    draws = RNG.normal(0.0, noise_scale, 20000)
    durations = np.clip(np.exp(mu + draws), minimum, maximum)
    median_duration = float(np.median(durations))
    abs_dev_from_median = np.abs(durations - median_duration)
    return {
        "noise_scale_log_hours": noise_scale,
        "geometric_std_factor": float(math.exp(noise_scale)),
        "monte_carlo_draws": int(draws.size),
        "irreducible_median_absolute_deviation_hours_at_mu4": float(np.median(abs_dev_from_median)),
        "irreducible_iqr_hours_at_mu4": [float(np.percentile(durations, 25)), float(np.percentile(durations, 75))],
        "interpretation": (
            "Even given perfect knowledge of the latent drivers (z0, support_propensity) "
            "that determine mu, episode duration still carries this additive Normal(0, "
            "noise_scale) log-hours noise term, drawn once and never revealed by any "
            "amount of observable history. This is a hard floor on achievable ICU "
            "remaining-time accuracy for ANY model, observable or latent-oracle."
        ),
    }


def _latent_forward_dispersion(config) -> dict:
    p = config.values["latent"]
    k = np.asarray(p["mean_reversion"], dtype=float)
    coupling = np.asarray(p["coupling_matrix"], dtype=float)
    innovation = np.asarray(p["innovation_loading"], dtype=float)
    drift_scale = float(p["drift_scale"])
    shock_probability = float(p["shock_probability_per_hour"])
    shock_scale = float(p["shock_scale"])
    condition_groups = config.values["population"]["condition_groups"]

    statics = load_statics(ROOT)
    rows = load_dev_rows(ROOT, splits=("train",))
    rows_with_elapsed = [row for row in rows if row["elapsed_episode_hours_at_t"] >= 30]
    sample_indices = RNG.choice(len(rows_with_elapsed), size=min(N_TRAIN_CUTOFFS, len(rows_with_elapsed)), replace=False)

    results_24, results_48 = [], []
    for index in sample_indices:
        row = rows_with_elapsed[int(index)]
        subject_ordinal = int(row["subject_id"][-8:])
        group_index = condition_groups.index(row["cardiac_condition_group"])
        z0, equilibrium, _support_propensity = initial_state(config, subject_ordinal, group_index)
        elapsed_hours = int(row["elapsed_episode_hours_at_t"])

        # Deterministic "true" state-at-t proxy: replay the no-support
        # recursion once with a FIXED seed derived from the subject, to get
        # a plausible state-at-t consistent with this subject's own
        # parameters (drift is a fixed per-subject constant either way).
        base_rng = np.random.default_rng(subject_ordinal)
        drift = base_rng.normal(0.0, drift_scale, 5)
        state = np.array(z0, dtype=float)
        shock_remaining, shock = 0, np.zeros(5)
        for _ in range(elapsed_hours):
            if shock_remaining <= 0 and base_rng.random() < shock_probability:
                shock_remaining = int(base_rng.integers(2, 9))
                shock = base_rng.normal(0.0, shock_scale, 5)
            pulse = shock / max(shock_remaining, 1) if shock_remaining > 0 else 0.0
            state = state + k * (equilibrium - state) + drift + coupling @ np.tanh(state) + pulse + innovation * base_rng.normal(size=5)
            shock_remaining -= 1
        state_at_t = state

        for horizon, bucket in ((24, results_24), (48, results_48)):
            severities = []
            for replicate in range(N_CONTINUATIONS):
                continuation_rng = np.random.default_rng((subject_ordinal * 1_000_003) + replicate + horizon)
                s = state_at_t.copy()
                s_shock_remaining, s_shock = 0, np.zeros(5)
                for _ in range(horizon):
                    if s_shock_remaining <= 0 and continuation_rng.random() < shock_probability:
                        s_shock_remaining = int(continuation_rng.integers(2, 9))
                        s_shock = continuation_rng.normal(0.0, shock_scale, 5)
                    pulse = s_shock / max(s_shock_remaining, 1) if s_shock_remaining > 0 else 0.0
                    s = s + k * (equilibrium - s) + drift + coupling @ np.tanh(s) + pulse + innovation * continuation_rng.normal(size=5)
                    s_shock_remaining -= 1
                severities.append(float(s[0]))
            bucket.append({"conditional_std_systemic_severity": float(np.std(severities)), "state_at_t_systemic_severity": float(state_at_t[0])})

    def _summ(bucket):
        stds = [item["conditional_std_systemic_severity"] for item in bucket]
        return {
            "n_cutoffs": len(bucket), "n_continuations_each": N_CONTINUATIONS,
            "mean_conditional_std": float(np.mean(stds)), "median_conditional_std": float(np.median(stds)),
        }

    return {
        "horizon_24h": _summ(results_24),
        "horizon_48h": _summ(results_48),
        "scope_limitation": (
            "Uses the no-support latent recursion only (simulate(), not "
            "simulate_with_support()); the support-conditioned state-"
            "evolution perturbation is excluded because faithfully forking "
            "it would require re-deriving each subject's exact mid-stream "
            "support-decision RNG position -- not attempted for "
            "compute-efficiency in a diagnostic phase. Reported dispersion "
            "is therefore a lower bound on true post-cutoff stochasticity "
            "for subjects who are ever on support."
        ),
        "interpretation": (
            "The conditional standard deviation of the systemic_severity latent "
            "dimension at t+24h/t+48h, given the identical state at t, is driven "
            "entirely by future hourly innovation noise and occasional shocks that "
            "have not happened yet at t and are therefore unknowable to any model, "
            "observable or latent-oracle, at prediction time."
        ),
    }


def main() -> None:
    config = load_runtime_config(ROOT / "configs/synthetic/final_benchmark_v1.json", ROOT)
    payload = {
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "artifact_version": "performance_v2_phase1_stochasticity_analysis_v1",
        "generator_config_sha256": config.sha256,
        "icu_remaining_time_noise_floor": _icu_noise_floor(config),
        "latent_forward_dispersion": _latent_forward_dispersion(config),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "stochasticity_analysis_v1.json"
    out_path.write_bytes(canonical_json_bytes(payload))
    print("wrote", out_path, sha256_file(out_path))


if __name__ == "__main__":
    main()
