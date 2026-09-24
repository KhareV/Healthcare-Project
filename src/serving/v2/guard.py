"""Fail-closed guard: ordinary V2 dashboard/API requests may only ever reach
the small, versioned demo-manifest subject set. Any other stay_id --
including every fresh V2 test-cohort subject (namespace SYN-V2-S-*) and every
subject outside the demo manifest -- is rejected exactly like an unknown
stay, so a caller cannot distinguish "sealed" from "never existed".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Tuple


class UnknownDemoStayError(LookupError):
    """Raised for any stay_id not in the frozen V2 demo manifest."""


def load_demo_manifest(root: Path) -> Mapping[str, object]:
    path = root / "configs/performance_v2/v2_demo_manifest_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def demo_stay_index(root: Path) -> Mapping[str, Mapping[str, object]]:
    manifest = load_demo_manifest(root)
    return {item["stay_id"]: item for item in manifest["demo_subjects"]}


def guard_demo_stay(root: Path, stay_id: str) -> Mapping[str, object]:
    """Raise UnknownDemoStayError unless stay_id is an approved demo subject.

    This is the ONLY function ordinary V2 serving code may use to resolve a
    stay_id. It never inspects fresh-test membership directly -- the demo
    manifest is itself built exclusively from DEV TRAIN subjects (see
    scripts/build_v2_demo_manifest.py), so a fresh V2 test subject (or any
    other non-demo subject) simply is not present and is rejected the same
    way an unknown identifier would be.
    """

    index = demo_stay_index(root)
    entry = index.get(stay_id)
    if entry is None:
        raise UnknownDemoStayError("unknown stay_id")
    return entry


def legal_cutoffs_for_stay(root: Path, stay_id: str) -> Tuple[str, ...]:
    entry = guard_demo_stay(root, stay_id)
    return tuple(entry["legal_cutoffs"])
