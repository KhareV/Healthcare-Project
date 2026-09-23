"""Stage-4 history-provider composition: prediction-time carriage and the
fail-closed sealed-test-subject guard.

Neither class alters truncation, feature-building, or SOFA semantics. Each
wraps an already-frozen, unmodified Pulkit component.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from serving.history import HistoryProvider, StoredStayTimeline, UnknownStayError
from serving.interfaces import CanonicalInputProvider
from serving.preprocessing import CanonicalHistoryInputProvider
from serving.real.predictors import PREDICTION_TIME_KEY
from serving.recovery import CurrentSOFAState


class SealedTestSubjectError(UnknownStayError):
    """Raised when Stage-4 serving would touch a sealed final-test subject.

    Subclasses the existing ``UnknownStayError`` (already mapped to a 404 by
    ``api/main.py``, unmodified) deliberately: an ordinary caller must not be
    able to distinguish "no such stay" from "this stay exists but is
    sealed," which would itself leak which subjects are in the final-test
    partition.
    """


class Stage4TimestampError(RuntimeError):
    """Raised when a request cutoff cannot be normalized to the frozen UTC contract."""


def load_stay_split_index(*, retained_cohort_path: Path, split_path: Path) -> Mapping[object, str]:
    """Map every retained stay_id to its accepted train/validation/test split."""

    subject_by_stay = {}
    with retained_cohort_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            subject_by_stay[row["stay_id"]] = row["subject_id"]
    split_by_subject = {}
    with split_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_by_subject[row["subject_id"]] = row["split"]
    return {
        stay_id: split_by_subject[subject_id]
        for stay_id, subject_id in subject_by_stay.items()
        if subject_id in split_by_subject
    }


class NonTestHistoryProvider:
    """Fail closed on any stay whose subject is in the sealed test split.

    This is the only Stage-4 serving/demo path to stay history; it refuses
    sealed subjects before the frozen truncator or feature builder ever see
    them, so ordinary development/replay requests cannot open final-test
    data. Only the dedicated Stage-5 final-test runner is authorized to
    consume ``AUTHORIZED_NOT_RUN``.
    """

    def __init__(self, inner: HistoryProvider, *, split_by_stay: Mapping[object, str]) -> None:
        self._inner = inner
        self._split_by_stay = dict(split_by_stay)

    def get_stay(self, stay_id: object) -> StoredStayTimeline:
        split = self._split_by_stay.get(stay_id)
        if split is None:
            raise UnknownStayError("unknown stay")
        if split == "test":
            raise SealedTestSubjectError(
                "Stage-4 serving refuses sealed final-test subjects; "
                "only the Stage-5 final-test runner may authorize test access"
            )
        return self._inner.get_stay(stay_id)


def _as_z_suffixed_utc(prediction_time: str) -> str:
    """Normalize any timezone-aware ISO-8601 UTC string to literal ``...Z``.

    ``serving.history``'s truncator accepts ``+00:00`` (Python 3.9
    ``datetime.fromisoformat`` cannot parse ``Z``); the frozen Sanskruti
    ``data.synthetic.validation.parse_utc`` used by ``sofa_at`` accepts only
    literal ``Z``. Both are pre-existing, tested, unmodified contracts; this
    is the one seam where the same request cutoff must satisfy both.
    """

    instant = datetime.fromisoformat(prediction_time.replace("Z", "+00:00"))
    if instant.utcoffset() != timezone.utc.utcoffset(None):
        raise Stage4TimestampError("prediction_time must be UTC")
    return instant.isoformat().replace("+00:00", "Z")


class Stage4CurrentSOFAProvider:
    """Adapt the frozen ``SyntheticCurrentSOFAProvider`` to the request's
    cutoff format without modifying it."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def current_sofa(self, *, stay_id: object, prediction_time: str) -> CurrentSOFAState:
        state = self._inner.current_sofa(
            stay_id=stay_id, prediction_time=_as_z_suffixed_utc(prediction_time)
        )
        # Echo back the caller's exact cutoff spelling (RecoveryServingPostprocessor
        # requires an exact string match against the request), not this
        # adapter's internal Z-normalized form.
        return replace(state, prediction_time=prediction_time)


class Stage4CanonicalInputProvider:
    """Carry ``prediction_time`` alongside the shared family view.

    Wraps the frozen, unmodified ``CanonicalHistoryInputProvider`` — the same
    class Phase 14 used to prove offline/serving equivalence — without
    changing its truncation, feature-building, or validation behavior. The
    extra key is stripped again by ``serving.real.predictors.RealFrozenPreprocessor``
    before the view reaches the frozen Phase-10 transform, which only reads
    its five known keys.
    """

    def __init__(self, inner: CanonicalHistoryInputProvider) -> None:
        self._inner = inner

    def get_canonical_input(
        self,
        *,
        stay_id: object,
        prediction_time: str,
        task: str,
        family: str,
        feature_version: str,
    ) -> object:
        view = self._inner.get_canonical_input(
            stay_id=stay_id,
            prediction_time=prediction_time,
            task=task,
            family=family,
            feature_version=feature_version,
        )
        enriched = dict(view)
        enriched[PREDICTION_TIME_KEY] = prediction_time
        return enriched

    def data_quality(self, *, stay_id: object, prediction_time: str) -> Mapping[str, int]:
        return self._inner.data_quality(stay_id=stay_id, prediction_time=prediction_time)
