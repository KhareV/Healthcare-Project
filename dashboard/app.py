"""FastAPI shell for the retrospective sequential replay dashboard."""

from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from dashboard.api_client import HTTPDashboardAPIClient
from dashboard.catalog import load_synthetic_dashboard_catalog
from dashboard.render import render_dashboard
from dashboard.replay import ReplayController
from dashboard.view_models import build_dashboard_view


ControllerFactory = Callable[[], ReplayController]
DEFAULT_FIXTURE = Path(__file__).with_name("fixtures") / "synthetic_dashboard_phase13_v1.json"


def create_dashboard_app(
    controller_factory: Optional[ControllerFactory] = None,
    *,
    synthetic: bool = False,
) -> FastAPI:
    application = FastAPI(title="Retrospective recovery replay dashboard")

    @application.get("/", response_class=HTMLResponse)
    def index(
        stay_id: Optional[str] = Query(default=None),
        prediction_time: Optional[str] = Query(default=None),
    ) -> HTMLResponse:
        if controller_factory is None:
            return HTMLResponse(
                render_dashboard(
                    stays=(), selected_stay=None, cutoffs=(), selected_cutoff=None,
                    view=None,
                    error="dashboard is unavailable until an approved API and catalog are configured",
                    synthetic=False,
                ),
                status_code=503,
            )
        controller = controller_factory()
        error = None
        view = None
        try:
            if stay_id is not None:
                controller.select_stay(stay_id)
            if prediction_time is not None:
                snapshot = controller.select_cutoff(prediction_time)
                view = build_dashboard_view(snapshot, synthetic=synthetic)
        except (RuntimeError, ValueError, KeyError, TypeError):
            error = controller.last_error or "replay selection is unavailable or incompatible"
        return HTMLResponse(
            render_dashboard(
                stays=controller.available_stays,
                selected_stay=controller.selected_stay_id,
                cutoffs=controller.available_cutoffs,
                selected_cutoff=controller.selected_cutoff,
                view=view,
                error=error,
                synthetic=synthetic,
            ),
            status_code=200 if error is None else 422,
        )

    return application


def create_real_dashboard_app(
    api_base_url: str,
    *,
    catalog,
    timeout_seconds: float = 5.0,
) -> FastAPI:
    """Real Stage-4 replay dashboard: calls the live API for every cutoff,
    same as the synthetic composition — no precomputed predictions."""

    def factory() -> ReplayController:
        return ReplayController(
            catalog,
            HTTPDashboardAPIClient(
                api_base_url, synthetic=False, timeout_seconds=timeout_seconds
            ),
        )

    return create_dashboard_app(factory, synthetic=False)


def create_synthetic_dashboard_app(
    api_base_url: str,
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
    timeout_seconds: float = 5.0,
) -> FastAPI:
    catalog = load_synthetic_dashboard_catalog(fixture_path)

    def factory() -> ReplayController:
        return ReplayController(
            catalog,
            HTTPDashboardAPIClient(
                api_base_url, synthetic=True, timeout_seconds=timeout_seconds
            ),
        )

    return create_dashboard_app(factory, synthetic=True)


# Import-safe and fail-closed: synthetic mode is never an implicit fallback.
app = create_dashboard_app()
