"""Descriptive Phase-6 SOFA QA; never computes labels or model metrics."""
from collections import Counter
from typing import Iterable, Mapping

from .sofa import COMPONENTS, SOFAResult


def summarize_sofa_results(results: Iterable[SOFAResult], *, status: str) -> Mapping:
    rows = tuple(results)
    for row in rows:
        row.validate()
    observed = {
        name: sum(row.components[index].observed for row in rows)
        for index, name in enumerate(COMPONENTS)
    }
    component_scores = {
        name: {str(key): value for key, value in sorted(Counter(row.components[index].score for row in rows).items())}
        for index, name in enumerate(COMPONENTS)
    }
    total_counts = {str(key): value for key, value in sorted(Counter(row.total_score for row in rows).items())}
    count = len(rows)
    return {
        "qa_version": "synthetic_sofa_phase6_qa_v1",
        "status": status,
        "score_count": count,
        "score_range": None if not rows else [min(row.total_score for row in rows), max(row.total_score for row in rows)],
        "total_score_distribution": total_counts,
        "component_observed": {
            name: {"count": value, "rate": (value / count if count else None)}
            for name, value in observed.items()
        },
        "all_six_observed": {
            "count": sum(all(row.component_observed) for row in rows),
            "rate": (sum(all(row.component_observed) for row in rows) / count if count else None),
        },
        "observed_component_count_distribution": {str(key): value for key, value in sorted(Counter(sum(row.component_observed) for row in rows).items())},
        "component_score_distribution": component_scores,
        "blocked_or_error_count": 0,
    }
