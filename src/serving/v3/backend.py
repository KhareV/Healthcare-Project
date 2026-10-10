"""V3 product backend: payload assembly, metadata and performance views for api.v2_app.build_v2_app.

Everything numeric comes from the frozen V3 runtime or from verified frozen result artifacts; nothing
here makes a scientific decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Tuple

from serving.v3.explanations import explain_task
from serving.v3.runtime import FINAL_FREEZE_REL, THRESHOLD_VALUE, V3ServingError, V3ServingRuntime, _sha256

SCHEMA_VERSION = "v3_prediction_response_v1"
RELEASE = "V3"
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
COPILOT_PROMPT_REL = "configs/product_v3/trajectory_copilot_prompt_v2.txt"
PHASE4 = "artifacts/performance_v3/phase4/"
D1_NOTE = ("Defect D1: frozen V2's offline harness exposed future horizon availability through missing current-SOFA; "
           "V2_AS_EVALUATED is historical and compromised, V2_AS_SERVED is what V2 actually produced when deployed.")


def data_quality(row: Mapping[str, object]) -> dict:
    mask = row["observation_mask"]
    padding = row.get("padding_mask", [False] * len(mask))
    total_values = sum(len(b) for b in mask)
    observed_values = sum(1 for b in mask for v in b if v)
    return {
        "total_bins": len(mask), "observed_bins": sum(1 for b, p in zip(mask, padding) if not p and any(b)),
        "padding_bins": sum(1 for p in padding if p), "total_feature_values": total_values,
        "observed_feature_values": observed_values, "missing_feature_values": total_values - observed_values,
        "observed_feature_fraction": observed_values / total_values if total_values else None,
    }


class V3Backend:
    release = RELEASE
    serving_error = V3ServingError
    copilot_prompt_path: Optional[Path] = None

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.runtime = V3ServingRuntime(self.root)  # raises V3ServingError; the app must not start without V3
        self.copilot_prompt_path = self.root / COPILOT_PROMPT_REL

    # -- metadata ---------------------------------------------------------------------------------
    @property
    def bundle_info(self) -> dict:
        return self.runtime.release_info

    def model_bundle(self) -> dict:
        return json.loads(json.dumps(self.bundle_info))

    def model_metadata(self) -> dict:
        info = self.bundle_info
        return {
            "schema_version": SCHEMA_VERSION, "release": RELEASE, "bundle_id": info["bundle_id"], "status": info["status"],
            "model_versions": {t: {"family": v["family"], "feature_variant": v["feature_variant"], "artifact_sha256": v["model_sha256"]} for t, v in info["tasks"].items()},
            "calibrator": info["calibrator"], "support_threshold": THRESHOLD_VALUE,
            "selected_models_sha256": info["selected_models_sha256"], "explanation_method": "tree_shap",
            "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY", "defect_d1_note": D1_NOTE,
        }

    def health_extras(self) -> dict:
        info = self.bundle_info
        return {"scope": "PERFORMANCE_V3_SYNTHETIC_BENCHMARK", "release": RELEASE, "bundle_id": info["bundle_id"],
                "selected_models_sha256": info["selected_models_sha256"], "model_families": {t: v["family"] for t, v in info["tasks"].items()},
                "calibrator": info["calibrator"]["method"], "support_threshold": THRESHOLD_VALUE}

    # -- prediction -------------------------------------------------------------------------------
    def build_payload(self, stay_id: str, prediction_time: str) -> Tuple[dict, dict]:
        p = self.runtime.predict(stay_id, prediction_time)
        explanations = {
            t: explain_task(self.runtime.models[t], t, p.matrices[t], p.feature_names[t], stay_id=stay_id, prediction_time=prediction_time)
            for t in TASKS
        }
        tasks = self.bundle_info["tasks"]
        payload = {
            "schema_version": SCHEMA_VERSION, "release": RELEASE,
            "mode": "RETROSPECTIVE_SEQUENTIAL_REPLAY",
            "stay_id": p.stay_id, "prediction_time": p.prediction_time, "grid_index": p.grid_index,
            "elapsed_icu_hours": p.icu_elapsed_hours, "current_sofa": p.current_sofa,
            "recovery": {
                "delta_24h": p.recovery24_delta, "delta_48h": p.recovery48_delta,
                "sofa_hat_24h": p.recovery24_sofa, "sofa_hat_48h": p.recovery48_sofa,
                "model_metadata": {
                    "family": f"{tasks['recovery24']['family']} (+24h) / {tasks['recovery48']['family']} (+48h)",
                    "family_24h": tasks["recovery24"]["family"], "family_48h": tasks["recovery48"]["family"],
                    "feature_variant": tasks["recovery24"]["feature_variant"], "feature_variant_24h": tasks["recovery24"]["feature_variant"],
                    "feature_variant_48h": tasks["recovery48"]["feature_variant"],
                    "artifact_sha256_24h": p.model_hashes["recovery24"], "artifact_sha256_48h": p.model_hashes["recovery48"],
                },
            },
            "icu_stay_time": {
                "remaining_hours": p.icu_remaining_hours, "raw_log_prediction": p.icu_raw_log_prediction,
                "model_metadata": {"family": tasks["icu_stay_time"]["family"], "feature_variant": tasks["icu_stay_time"]["feature_variant"],
                                   "artifact_sha256": p.model_hashes["icu_stay_time"]},
            },
            "organ_support": {
                "raw_probability": p.support_raw_probability, "probability_24h": p.support_calibrated_probability,
                "threshold": p.support_threshold, "alert": p.support_alert,
                "model_metadata": {"family": tasks["organ_support"]["family"], "feature_variant": tasks["organ_support"]["feature_variant"],
                                   "artifact_sha256": p.model_hashes["organ_support"], "calibrator": self.bundle_info["calibrator"]["method"]},
            },
            "explanations": explanations,
            "data_quality": data_quality(p.feature_row),
            "temporal_window": {
                "channel_names": list(p.feature_row["temporal_feature_names"]),
                "observation_mask": [list(b) for b in p.feature_row["observation_mask"]],
                "padding_mask": list(p.feature_row.get("padding_mask", [False] * len(p.feature_row["observation_mask"]))),
            },
            "versions": {"selected_models_v3_sha256": self.bundle_info["selected_models_sha256"], "v3_model_freeze_sha256": self.bundle_info["model_freeze_sha256"],
                         "serving_version": "performance_v3_serving_v1"},
            "model_bundle": self.model_bundle(),
            "data_revision": p.data_revision,
        }
        snapshot = {
            "current_sofa": p.current_sofa, "delta_sofa_24": p.recovery24_delta, "delta_sofa_48": p.recovery48_delta,
            "predicted_sofa_24h": p.recovery24_sofa, "predicted_sofa_48h": p.recovery48_sofa, "remaining_icu_hours": p.icu_remaining_hours,
            "support_raw_probability": p.support_raw_probability, "support_calibrated_probability": p.support_calibrated_probability,
            "support_threshold": p.support_threshold, "support_alert": p.support_alert,
        }
        return payload, snapshot

    # -- research-facing views ---------------------------------------------------------------------
    def performance(self) -> dict:
        def read(rel: str):
            return json.loads((self.root / rel).read_text(encoding="utf-8"))

        paired = read(PHASE4 + "paired_v2_v3_metrics.json")
        final = read(PHASE4 + "final_metrics_v3.json")
        models = final["models"]
        return {
            "release": RELEASE, "bundle_id": self.bundle_info["bundle_id"], "status": "FINAL_SEALED_TEST_ONE_TIME_EVALUATION",
            "evaluation_cohort": {"namespace": "SYN-V3-T", "subjects": 1500, "denominators": final["denominators"]},
            "final_metrics": models["V3"], "final_metrics_all_models": models,
            "bootstrap": final["bootstrap"], "thresholds": final["thresholds"], "naive_baselines": read(PHASE4 + "naive_baselines_v1.json"),
            "paired_v2_v3": {"primary_comparator": paired["primary_comparator"], "promotion": paired["promotion"], "point_differences": paired["point_differences"]},
            "comparators": {
                "V2_AS_SERVED": {"label": "Frozen V2 as it actually served (SOFA(t) supplied for every cutoff)", "role": "operational comparison", "metrics": models["V2_AS_SERVED"]},
                "CONTROL": {"label": "V2 recipe refit on corrected features and larger development data", "role": "modelling-method comparison", "metrics": models["CONTROL"]},
                "V2_AS_EVALUATED": {"label": "Original V2 offline harness (contains defect D1)", "role": "HISTORICAL_COMPROMISED", "metrics": models["V2_AS_EVALUATED"], "note": D1_NOTE},
            },
            "evaluation_freeze": {"ref": FINAL_FREEZE_REL, "sha256": _sha256(self.root / FINAL_FREEZE_REL) if (self.root / FINAL_FREEZE_REL).exists() else None},
            "predictions_sha256": final.get("predictions_sha256"),
            "defect_d1": D1_NOTE,
            "notes": ["Confidence intervals are population-level stay-cluster bootstrap intervals, not individual prediction intervals.",
                      "ICU V3 equals the repaired control (V2 recipe retained by the pre-registered rule).",
                      "Recovery48 advantage over the control is not established at the stricter paired interval."],
        }

    def demo_subjects(self, manifest: Mapping[str, object], aliases: Mapping[str, Mapping[str, object]]) -> dict:
        subjects = [{**s, "patient_alias": aliases.get(s["stay_id"], {}).get("patient_alias", s["stay_id"]),
                     "training_cohort_demo": True,
                     "note": "Drawn from the V1 development cohort that is part of V3's training pool; illustrative, not an evaluation case."}
                    for s in manifest["demo_subjects"]]
        return {"status": manifest["status"], "selection_criteria": manifest["selection_criteria"], "demo_subjects": subjects, "release": RELEASE}
