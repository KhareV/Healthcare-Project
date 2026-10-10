"""Read-only research evidence router (frozen V3 / V3.1 artifacts). Include with
``app.include_router(build_research_router(root))``. No model inference, no sealed-cohort raw data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from serving.v3.research_data import EvidenceMissing, ResearchEvidence, to_csv, to_latex_table


def build_research_router(root: Path) -> APIRouter:
    evidence = ResearchEvidence(Path(root))
    router = APIRouter(prefix="/research", tags=["research"])

    def guarded(fn):
        try:
            return fn()
        except EvidenceMissing as exc:
            raise HTTPException(status_code=404, detail=f"evidence unavailable: {exc}") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown research dataset: {exc}") from exc

    def named(dataset_id: str) -> Dict[str, Any]:
        return guarded(lambda: evidence.dataset(dataset_id))

    @router.get("/index")
    def index():
        return evidence.index()

    @router.get("/summary")
    def summary():
        return guarded(evidence.summary)

    @router.get("/cohort")
    def cohort():
        return guarded(evidence.cohort_bundle)

    @router.get("/curves/support")
    def support_curves():
        return {"roc": named("support_roc"), "pr": named("support_pr"), "reliability": named("support_reliability"),
                "score_distribution": named("support_score_distribution"), "confusion": named("support_confusion"),
                "threshold_sensitivity": named("support_threshold_sensitivity"), "metrics": named("classification_metrics")}

    @router.get("/regression")
    def regression():
        return {"residuals": named("regression_residuals"), "predicted_vs_observed": named("regression_predicted_vs_observed"),
                "metrics": named("regression_metrics"), "error_slices": named("error_slices"), "paired": named("paired_differences")}

    @router.get("/learning-curve")
    def learning_curve():
        return named("learning_curve")

    @router.get("/ablation")
    def ablation():
        return named("ablation")

    @router.get("/leaderboard")
    def leaderboard():
        return named("leaderboard")

    @router.get("/robustness")
    def robustness():
        return named("robustness")

    @router.get("/explainability/global")
    def explainability_global():
        return named("explainability_global")

    @router.get("/defect-d1")
    def defect_d1():
        return named("defect_d1")

    @router.get("/diagnostic")
    def diagnostic():
        return {"latent_state": named("diagnostic_latent_state"), "candidate_development": named("diagnostic_candidate_development"),
                "status": "PRIVILEGED_DIAGNOSTIC_NOT_DEPLOYABLE"}

    @router.get("/provenance")
    def provenance():
        return named("provenance")

    @router.get("/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        return named(dataset_id)

    @router.get("/datasets/{dataset_id}/export")
    def export(dataset_id: str, format: str = Query("json", pattern="^(csv|json|tex)$")):
        payload = named(dataset_id)
        rows = payload["data"].get("rows", [])
        prov = payload["provenance"]
        if format == "json":
            return payload
        if format == "csv":
            header = f"# {prov['title']} | status={prov['evidence_status']} | source={prov['source_artifact']} | sha256={prov['source_sha256']}\n"
            return PlainTextResponse(header + to_csv(rows), media_type="text/csv")
        caption = f"{prov['title']} ({prov['evidence_status']}; {prov['partition']}; source {prov['source_artifact']})"
        return PlainTextResponse(to_latex_table(rows, caption, "tab:" + dataset_id), media_type="application/x-tex")

    return router
