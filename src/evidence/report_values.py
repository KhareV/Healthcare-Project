"""Structured report-value generation that refuses missing scientific sources."""


def blocked_report_values(display_precision):
    return {
        "report_values_version": "report_values_v1",
        "scientific_values": [],
        "validation_results": {
            "status": "BLOCKED",
            "detail": "BLOCKED — EVIDENCE SOURCE ARTIFACT REQUIRED",
        },
        "final_test_results": {
            "status": "BLOCKED",
            "detail": "BLOCKED — ACTIVE G3 / FINAL TEST REQUIRED",
        },
        "manual_scientific_values_allowed": False,
        "display_precision": display_precision,
        "display_precision_status": "FROZEN_BEFORE_RESULT_GENERATION",
    }
