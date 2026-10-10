"""Read-only scenario-showcase endpoints (additive). Predictions are NOT served here; use POST /predict."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from serving.v3 import showcase as S

SUMMARY_FIELDS = ("scenario_id", "patient_alias", "category", "title", "subtype", "objective", "source_cohort", "stay_id", "subject_id",
                  "age_years", "sex_category", "n_cutoffs", "legal_cutoffs", "selection_rationale", "key_event", "analytical_focus", "status")


def build_showcase_router(root: Path, runtime=None) -> APIRouter:
    root = Path(root).resolve()
    router = APIRouter(prefix="/showcase", tags=["showcase"])

    def _summary(scn: dict) -> dict:
        out = {k: scn[k] for k in SUMMARY_FIELDS}
        out["measured_criteria"] = scn["measured_criteria"]
        if runtime is None:
            registered = False
        elif scn["stay_id"] in getattr(runtime, "_demo_extra", {}):
            registered = True
        else:
            registered = runtime.ephemeral_entry(scn["stay_id"]) is not None
        out["registered_for_inference"] = registered
        return out

    def _get(scenario_id: str) -> dict:
        try:
            return S.get_scenario(root, scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown scenario") from exc
        except S.ShowcaseError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.get("/scenarios")
    def scenarios():
        try:
            manifest = S.load_showcase(root)
        except S.ShowcaseError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": manifest["status"], "source_cohort": manifest["source_cohort"], "selection_policy": manifest["selection_policy"],
            "substitutions": manifest["substitutions"], "manifest_sha256": manifest["_sha256"],
            "scenarios": [_summary(s) for s in manifest["scenarios"]],
        }

    @router.get("/scenarios/{scenario_id}")
    def scenario(scenario_id: str):
        scn = _get(scenario_id)
        out = _summary(scn)
        out["cutoff_facts"] = scn["cutoff_facts"]
        return out

    @router.get("/scenarios/{scenario_id}/timeline")
    def timeline(scenario_id: str):
        _get(scenario_id)
        return S.scenario_timeline(root, scenario_id)

    @router.get("/scenarios/{scenario_id}/story")
    def story(scenario_id: str):
        _get(scenario_id)
        return S.scenario_story(root, scenario_id)

    @router.get("/compare")
    def compare(a: str, b: str):
        _get(a)
        _get(b)
        if a == b:
            raise HTTPException(status_code=422, detail="choose two different scenarios")
        return S.compare(root, a, b)

    return router
