"""Performance-v2 Phase 1, Part 3: canonical feature audit.

DEVELOPMENT_DIAGNOSTIC_ONLY. Reads the frozen v1 feature schema and model-
input contract (never modifies them) and produces a machine-readable
inventory of what is and is not in the current canonical representation.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.config import canonical_json_bytes  # noqa: E402
from vedant_infra.hashing import sha256_file  # noqa: E402

OUT_DIR = ROOT / "artifacts/performance_v2/phase1"

# Structural/derived quantities the task explicitly asks about, and whether
# the CURRENT canonical representation (schema v2 + model-input contract)
# supplies them, verified against configs/synthetic/feature_schema_v2.json
# and src/data/xgb_canonical.py's build_flat_feature_map (the sole place
# the flat XGB/GRU model input is constructed).
STRUCTURAL_PRESENCE = [
    {"item": "episode_elapsed_time_at_t", "present_in_model_input": False,
     "note": "Not a temporal channel, not a static, not in the flat feature map. Only grid_index (cutoff ordinal) exists in the raw row, and it is not fed to the model either. Lawfully derivable as (prediction_time - intime)."},
    {"item": "prediction_grid_index", "present_in_model_input": False,
     "note": "Present in raw rows (used for slicing/governance) but never added to the XGB flat vector or the GRU structured input."},
    {"item": "time_since_icu_like_episode_start", "present_in_model_input": False,
     "note": "Same as episode_elapsed_time_at_t; absent."},
    {"item": "current_sofa", "present_in_model_input": False,
     "note": "configs/synthetic/feature_schema_v2.json sofa_feature.included=false, reason='RECOVERY_TARGET_SOURCE_IS_NOT_AUTOMATICALLY_A_MODEL_FEATURE'. baseline_sofa exists in the pre-split scientific package (lawful: computed at/before t) but is never joined into the model feature vector."},
    {"item": "recent_sofa_trajectory", "present_in_model_input": False,
     "note": "No past-SOFA channel of any kind is fed to the model."},
    {"item": "current_support_state", "present_in_model_input": True,
     "note": "vasopressor_on / invasive_ventilation_on are 2 of the 21 temporal channels (LATEST/STATE_AT_BIN_END per 6h bin), so the state at each of the 8 bins IS available."},
    {"item": "prior_support_transitions", "present_in_model_input": False,
     "note": "No derived duration-in-state / transition-count feature; only the raw per-bin binary state (from which a GRU could in principle learn transitions across its 8-step sequence, but XGBoost sees 8 independent binary flags with no explicit transition-count feature)."},
    {"item": "latest_feature_value", "present_in_model_input": True,
     "note": "Every one of the 21 temporal channels is a LATEST-aggregation value per bin (block 'value' in the flat map)."},
    {"item": "6h_12h_24h_48h_delta", "present_in_model_input": False,
     "note": "No difference/delta feature of any kind is constructed; only the raw per-bin latest value."},
    {"item": "slope_12_24_48h", "present_in_model_input": False, "note": "Not present."},
    {"item": "rolling_mean_min_max_std", "present_in_model_input": False, "note": "Not present; only the single LATEST value per bin per channel."},
    {"item": "observation_count_or_fraction_observed", "present_in_model_input": False,
     "note": "The per-bin observation_mask block exists (True/False per channel per bin), from which an aggregate count/fraction COULD be derived, but no such aggregate is itself supplied as a feature."},
    {"item": "time_since_last_observation_tslo", "present_in_model_input": True,
     "note": "The flat map's 'tslo' block supplies TSLO (hours) per channel per bin directly (sentinel 54.0 when never observed in-window). This is the one derived-style signal already present."},
    {"item": "missingness_pattern_summaries", "present_in_model_input": False,
     "note": "Per-bin per-channel observation_mask + tslo exist, but no cross-bin/cross-channel missingness-pattern summary (e.g. total observed count, longest gap) is constructed."},
    {"item": "historical_extrema", "present_in_model_input": False, "note": "Not present."},
    {"item": "recent_volatility_std", "present_in_model_input": False, "note": "Not present."},
]


def main() -> None:
    schema = json.loads((ROOT / "configs/synthetic/feature_schema_v2.json").read_text())
    manifest_path = ROOT / "artifacts/manifests/synthetic_phase10_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads((ROOT / manifest["model_input_contract_path"]).read_text())
    preprocessor = json.loads((ROOT / "artifacts/preprocessors/synthetic_feature_preprocessor_v1.json").read_text())

    temporal_channels = schema["temporal_channels"]
    inventory = []
    for item in temporal_channels:
        inventory.append({
            "feature_name": item["name"],
            "dynamic_or_static": "dynamic",
            "raw_or_transformed": "raw" if item["aggregation"] in ("LATEST", "STATE_AT_BIN_END") else "transformed",
            "clinical_or_structural": "clinical",
            "observed_at_cutoff_only": True,
            "leakage_safe": True,
            "supplied_to_xgb": True,
            "supplied_to_gru": True,
            "aggregation": item["aggregation"],
            "unit": item["unit"],
            "also_has_observation_mask_channel": True,
            "also_has_tslo_channel": True,
        })
    for name in preprocessor["encoded_static_feature_names"]:
        inventory.append({
            "feature_name": name,
            "dynamic_or_static": "static",
            "raw_or_transformed": "transformed" if ("__z" in name or "==" in name) else "raw",
            "clinical_or_structural": "clinical" if "condition_group" in name or "sex" in name else "structural",
            "observed_at_cutoff_only": True,
            "leakage_safe": True,
            "supplied_to_xgb": True,
            "supplied_to_gru": True,
        })

    payload = {
        "status": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "artifact_version": "performance_v2_phase1_feature_audit_v1",
        "source_feature_schema_version": schema["schema_version"],
        "source_feature_schema_sha256": sha256_file(ROOT / "configs/synthetic/feature_schema_v2.json"),
        "source_model_input_contract_sha256": sha256_file(ROOT / manifest["model_input_contract_path"]),
        "shape": {
            "sequence_length_T": schema["shape"]["sequence_length"],
            "temporal_feature_count_F": schema["shape"]["feature_dimension"],
            "lookback_hours": schema["shape"]["lookback_hours"],
            "bin_width_hours": schema["shape"]["bin_width_hours"],
            "transformed_static_count": len(preprocessor["encoded_static_feature_names"]),
            "xgb_flat_dimension": contract["information_parity"]["xgboost_flat_dimension"],
        },
        "canonical_feature_inventory": inventory,
        "structural_and_derived_feature_presence": STRUCTURAL_PRESENCE,
        "prohibited_fields": schema["prohibited_fields"],
        "summary": {
            "present_count": sum(1 for x in STRUCTURAL_PRESENCE if x["present_in_model_input"]),
            "absent_count": sum(1 for x in STRUCTURAL_PRESENCE if not x["present_in_model_input"]),
            "headline_finding": (
                "Current SOFA (baseline_sofa) is NOT a model feature for the recovery task, despite "
                "being lawfully known at t and already computed in the pre-split package. Elapsed "
                "episode time and cutoff index are also absent from the model input, despite being "
                "trivially lawful. No delta/slope/rolling-summary/missingness-pattern features exist; "
                "only the single LATEST value, an observation mask, and TSLO are supplied per channel "
                "per bin. Current/recent support state IS present per-bin, but no derived duration-in-"
                "state or transition-count feature is constructed."
            ),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "canonical_feature_inventory_v1.json"
    out_path.write_bytes(canonical_json_bytes(payload))
    print("wrote", out_path, "sha256", sha256_file(out_path))


if __name__ == "__main__":
    main()
