from dashboard.render import render_dashboard
from dashboard.view_models import REPLAY_BANNER


def test_mandatory_banner_and_no_claims_in_empty_render():
    text = render_dashboard(
        stays=(), selected_stay=None, cutoffs=(), selected_cutoff=None,
        view=None, error=None, synthetic=False,
    )
    assert REPLAY_BANNER in text
    lower = text.lower()
    for forbidden in (
        "real-time clinical prediction system",
        "confidence interval",
        "patient-specific uncertainty",
        "caused by",
        "will recover",
        "actual future",
    ):
        assert forbidden not in lower
