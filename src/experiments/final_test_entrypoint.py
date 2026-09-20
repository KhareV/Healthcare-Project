"""The single canonical guarded entrypoint reserved for Phase-20 evaluation."""

from pathlib import Path
from typing import Callable

from vedant_infra.g3 import guarded_test_access


def run_guarded_final_test(
    root: Path, test_loader: Callable[[], object]
):
    """Authorize once, then invoke the supplied Phase-20 loader."""

    return guarded_test_access(
        root, test_loader, expected_scope="real"
    )
