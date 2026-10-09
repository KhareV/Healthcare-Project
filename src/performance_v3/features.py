"""V3 feature cache: streaming, cutoff-safe, column-selectable.

Every number is produced by the unmodified V2 builders
(performance_v2.diagnostic_features.build_all_groups, context_v2,
context_normalization, sofa_trend's scoring machinery). This module only
streams rows through them, fixes a column universe, stores a float32 matrix
and selects columns per feature variant.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from evaluation.weights import compute_stay_weights
from performance_v2.context_normalization import TRANSFORM_VERSION, _subject_hash, fit_normalization
from performance_v2.context_v2 import VARIANT_KEYS
from performance_v2.diagnostic_features import build_all_groups, build_group_a, feature_group_of

BUILDER_VERSION = "performance_v3_feature_cache_v1"

META_STR = ("subject_id", "stay_id", "prediction_time", "cohort_id")
META_FLOAT = ("delta_sofa_24", "delta_sofa_48", "icu_time_log1p", "organ_support_label", "baseline_sofa")
META_BOOL = ("recovery24_eligible", "recovery48_eligible", "icu_time_eligible", "organ_support_eligible")

TASK_COLUMNS = {
    "recovery24": ("recovery24_eligible", "delta_sofa_24"),
    "recovery48": ("recovery48_eligible", "delta_sofa_48"),
    "icu_time": ("icu_time_eligible", "icu_time_log1p"),
    "organ_support": ("organ_support_eligible", "organ_support_label"),
}
TASK_ALIASES = {"icu_stay_time": "icu_time", "support": "organ_support"}
CTRL_VARIANT_BY_TASK = {"recovery24": "B_MIN", "recovery48": "B_MIN", "icu_time": "B_PLUS_F", "organ_support": "B_FULL"}
V3_VARIANTS = ("V3_CTRL", "V3_B", "V3_BF", "V3_BC", "V3_BD", "V3_BE", "V3_BFS", "V3_FULL")


class FeatureCacheError(ValueError):
    pass


@dataclass
class FeatureCache:
    names: Tuple[str, ...]
    matrix: np.ndarray
    meta: Dict[str, np.ndarray]
    builder_version: str = BUILDER_VERSION
    input_sha256: str = ""

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])

    def column_index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}


# --------------------------------------------------------------------------
# SOFA component state at t (S group) -- frozen sofa_at only
# --------------------------------------------------------------------------

def _component_names() -> Tuple[str, ...]:
    from data.synthetic.sofa import COMPONENTS

    return tuple(COMPONENTS)


def sofa_component_feature_names() -> Tuple[str, ...]:
    out = []
    for component in _component_names():
        out.append(f"S__{component}__score")
        out.append(f"S__{component}__observed")
    return tuple(out)


def sofa_component_features(result) -> Dict[str, float]:
    """Six component scores + observed flags from a SOFAResult scored at the cutoff t.

    ``result`` may be a data.synthetic.sofa.SOFAResult, a mapping
    component -> (score, observed), or None (all NaN).
    """

    features: Dict[str, float] = {}
    if result is None:
        return {name: float("nan") for name in sofa_component_feature_names()}
    if hasattr(result, "components"):
        lookup = {c.component_name: (c.score, c.observed) for c in result.components}
    else:
        lookup = dict(result)
    for component in _component_names():
        if component not in lookup:
            features[f"S__{component}__score"] = float("nan")
            features[f"S__{component}__observed"] = float("nan")
            continue
        score, observed = lookup[component]
        features[f"S__{component}__score"] = float(score)
        features[f"S__{component}__observed"] = float(bool(observed))
    return features


@contextlib.contextmanager
def cohort_scoring_context(root: Path, *, raw_manifest_ref: str, timeline_manifest_ref: str):
    """Build the frozen SOFA scoring context for a NEW cohort.

    experiments.stage5_final_test.build_sofa_scoring_context reads two
    module-level path constants hard-wired to the V1 cohort. They are
    re-pointed (and restored) for the duration of the build; no frozen file is
    edited. Not thread-safe.
    """

    from experiments import stage5_final_test as s5

    saved = (s5.TIMELINE_MANIFEST_PATH, s5.RAW_DATASET_MANIFEST_PATH)
    s5.TIMELINE_MANIFEST_PATH, s5.RAW_DATASET_MANIFEST_PATH = timeline_manifest_ref, raw_manifest_ref
    try:
        yield s5.build_sofa_scoring_context(Path(root))
    finally:
        s5.TIMELINE_MANIFEST_PATH, s5.RAW_DATASET_MANIFEST_PATH = saved


def compute_sofa_trends_and_components(rows: Iterable[Mapping[str, object]], context):
    """Same cutoffs as performance_v2.sofa_trend.compute_sofa_trends (t, t-6h, t-12h, t-24h, t-48h,
    clamped to intime, never future) but with a caller-supplied scoring context, additionally returning
    the six component scores/observed flags at t.
    """

    from datetime import datetime, timedelta, timezone

    from data.synthetic.sofa import sofa_at
    from experiments.stage5_final_test import _stay_support_provider, _z
    from performance_v2.sofa_trend import _LAGS_HOURS

    provider = context.provider
    cache: dict = {}
    trends: Dict[Tuple[str, str], Dict[str, float]] = {}
    components: Dict[Tuple[str, str], Dict[str, Tuple[int, bool]]] = {}
    for row in rows:
        stay_id = row["stay_id"]
        stay_record = provider.stays.get(stay_id)
        if stay_record is None:
            continue
        if stay_id not in cache:
            cache[stay_id] = _stay_support_provider(context, stay_id)
        history = context.history_by_stay.get(stay_id, ())
        intime = datetime.fromisoformat(str(stay_record["intime"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        t = datetime.fromisoformat(str(row["prediction_time"]).replace("Z", "+00:00")).astimezone(timezone.utc)
        trend: Dict[str, float] = {}
        key = (stay_id, row["prediction_time"])
        for label, lag in _LAGS_HOURS.items():
            cutoff = t - timedelta(hours=lag)
            if cutoff < intime:
                continue
            try:
                result = sofa_at(history, stay_record, _z(cutoff), spec=provider.spec, bindings=provider.bindings, support_provider=cache[stay_id])
            except Exception:
                continue
            trend[label] = float(result.total_score)
            if lag == 0:
                components[key] = {c.component_name: (c.score, c.observed) for c in result.components}
        trends[key] = trend
    return trends, components


# --------------------------------------------------------------------------
# Streaming cache construction
# --------------------------------------------------------------------------

RowSource = Union[Sequence[Mapping[str, object]], Callable[[], Iterable[Mapping[str, object]]]]


def _iter_rows(source: RowSource) -> Iterator[Mapping[str, object]]:
    return iter(source() if callable(source) else source)


def _row_features(row, sofa_trends, sofa_components) -> Dict[str, float]:
    key = (row["stay_id"], row["prediction_time"])
    features = build_all_groups(row, baseline_sofa=row.get("baseline_sofa"), sofa_trend=(sofa_trends or {}).get(key, {}))
    if sofa_components is not None:
        features.update(sofa_component_features(sofa_components.get(key)))
    return features


def _meta_of(row: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {name: str(row.get(name, "")) for name in META_STR}
    out["grid_index"] = int(row["grid_index"])
    for name in META_FLOAT:
        value = row.get(name)
        out[name] = float("nan") if value is None else float(value)
    for name in META_BOOL:
        out[name] = bool(row.get(name))
    return out


def _collect_names(source: RowSource, sofa_trends, sofa_components) -> Tuple[str, ...]:
    """Column universe without building every per-row dict: key sets of groups B-F/S are row-independent;
    only static one-hot names vary, so group A is re-derived once per distinct categorical-static combination."""

    names: set = set()
    first = None
    seen_static: set = set()
    temporal_names = None
    for row in _iter_rows(source):
        if first is None:
            first = row
            names.update(_row_features(row, sofa_trends, sofa_components))
            temporal_names = tuple(row["temporal_feature_names"])
        elif tuple(row["temporal_feature_names"]) != temporal_names:
            raise FeatureCacheError("temporal_feature_names differ between rows")
        combo = tuple((n, v) for n, v in zip(row["static_feature_names"], row["static_features"]) if isinstance(v, str))
        if combo not in seen_static:
            seen_static.add(combo)
            names.update(build_group_a(row))
    if first is None:
        raise FeatureCacheError("no rows supplied")
    return tuple(sorted(names))


def _compute_chunk(args):
    rows, names, sofa_trends, sofa_components = args
    index_of = {n: i for i, n in enumerate(names)}
    onehot = np.asarray(["==" in n for n in names], dtype=bool)
    matrix = np.full((len(rows), len(names)), np.nan, dtype=np.float32)
    matrix[:, onehot] = 0.0
    metas = []
    for r, row in enumerate(rows):
        for name, value in _row_features(row, sofa_trends, sofa_components).items():
            j = index_of.get(name)
            if j is None:
                raise FeatureCacheError(f"unseen feature name {name!r}; column universe is fixed")
            matrix[r, j] = value
        metas.append(_meta_of(row))
    return matrix, metas


def _chunks(source: RowSource, size: int, sofa_trends, sofa_components, names) -> Iterator[tuple]:
    buffer: List[Mapping[str, object]] = []
    for row in _iter_rows(source):
        buffer.append(row)
        if len(buffer) >= size:
            yield _chunk_args(buffer, names, sofa_trends, sofa_components)
            buffer = []
    if buffer:
        yield _chunk_args(buffer, names, sofa_trends, sofa_components)


def _chunk_args(rows, names, sofa_trends, sofa_components):
    keys = [(r["stay_id"], r["prediction_time"]) for r in rows]
    trends = None if sofa_trends is None else {k: sofa_trends[k] for k in keys if k in sofa_trends}
    comps = None if sofa_components is None else {k: sofa_components[k] for k in keys if k in sofa_components}
    return (tuple(rows), names, trends, comps)


def build_feature_cache(
    rows: RowSource,
    *,
    sofa_trends: Optional[Mapping] = None,
    sofa_components: Optional[Mapping] = None,
    n_jobs: int = 8,
    names: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
) -> FeatureCache:
    """Stream rows (a sequence or a zero-arg callable returning an iterable; re-iterated once for names
    when ``names`` is None) into a float32 matrix. Per-row dicts are never all alive at once."""

    if names is None:
        if not callable(rows) and not isinstance(rows, Sequence):
            rows = list(rows)
        names = _collect_names(rows, sofa_trends, sofa_components)
    names = tuple(names)
    if len(set(names)) != len(names):
        raise FeatureCacheError("duplicate feature names")

    matrices: List[np.ndarray] = []
    metas: List[dict] = []

    def absorb(result):
        matrices.append(result[0])
        metas.extend(result[1])

    if n_jobs <= 1:
        for args in _chunks(rows, chunk_size, sofa_trends, sofa_components, names):
            absorb(_compute_chunk(args))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as pool:
            pending: deque = deque()
            for args in _chunks(rows, chunk_size, sofa_trends, sofa_components, names):
                pending.append(pool.submit(_compute_chunk, args))
                while len(pending) >= 2 * n_jobs:
                    absorb(pending.popleft().result())
            while pending:
                absorb(pending.popleft().result())

    if not matrices:
        raise FeatureCacheError("no rows supplied")
    matrix = np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)
    meta: Dict[str, np.ndarray] = {}
    for name in META_STR:
        meta[name] = np.asarray([m[name] for m in metas], dtype=str)
    meta["grid_index"] = np.asarray([m["grid_index"] for m in metas], dtype=np.int64)
    for name in META_FLOAT:
        meta[name] = np.asarray([m[name] for m in metas], dtype=np.float64)
    for name in META_BOOL:
        meta[name] = np.asarray([m[name] for m in metas], dtype=bool)
    cache = FeatureCache(names=names, matrix=matrix, meta=meta)
    cache.input_sha256 = _identity_sha256(cache)
    return cache


def _identity_sha256(cache: FeatureCache) -> str:
    h = hashlib.sha256()
    h.update(("|".join(cache.names) + "\n").encode("utf-8"))
    for s, st, pt, c in zip(cache.meta["subject_id"], cache.meta["stay_id"], cache.meta["prediction_time"], cache.meta["cohort_id"]):
        h.update(f"{s}|{st}|{pt}|{c}\n".encode("utf-8"))
    return h.hexdigest()


def _matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes()).hexdigest()


def save_feature_cache(cache: FeatureCache, path_prefix: Union[str, Path]) -> Tuple[Path, Path]:
    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
    np.savez(npz_path, matrix=cache.matrix, **{f"meta__{k}": v for k, v in cache.meta.items()})
    sidecar = {
        "builder_version": cache.builder_version,
        "input_sha256": cache.input_sha256,
        "matrix_sha256": _matrix_sha256(cache.matrix),
        "n_rows": cache.n_rows,
        "n_features": len(cache.names),
        "names": list(cache.names),
    }
    json_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    return npz_path, json_path


def load_feature_cache(path_prefix: Union[str, Path]) -> FeatureCache:
    prefix = Path(path_prefix)
    sidecar = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if sidecar["builder_version"] != BUILDER_VERSION:
        raise FeatureCacheError("feature cache builder version mismatch")
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as data:
        matrix = data["matrix"]
        meta = {k[len("meta__"):]: data[k] for k in data.files if k.startswith("meta__")}
    if _matrix_sha256(matrix) != sidecar["matrix_sha256"]:
        raise FeatureCacheError("feature cache matrix hash mismatch")
    cache = FeatureCache(names=tuple(sidecar["names"]), matrix=matrix, meta=meta, builder_version=sidecar["builder_version"], input_sha256=sidecar["input_sha256"])
    if _identity_sha256(cache) != cache.input_sha256:
        raise FeatureCacheError("feature cache identity hash mismatch")
    return cache


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

def canonical_task(task: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    task = TASK_ALIASES.get(task, task)
    if task not in TASK_COLUMNS:
        raise ValueError(f"unknown task {task!r}")
    return task


def variant_columns(names: Sequence[str], variant: str, task: Optional[str] = None, *, include_s: bool = False) -> np.ndarray:
    """Boolean column mask over ``names``: Group A always, plus the variant's context keys and C/D/E/S groups."""

    if variant not in V3_VARIANTS:
        raise ValueError(f"unknown V3 variant {variant!r}")
    context = None
    groups: Tuple[str, ...] = ()
    if variant == "V3_CTRL":
        task = canonical_task(task)
        if task is None:
            raise ValueError("V3_CTRL is per-task; pass task=")
        context = CTRL_VARIANT_BY_TASK[task]
    elif variant == "V3_B":
        context = "B_FULL"
    elif variant == "V3_BF":
        context = "B_PLUS_F"
    elif variant == "V3_BC":
        context, groups = "B_FULL", ("C",)
    elif variant == "V3_BD":
        context, groups = "B_FULL", ("D",)
    elif variant == "V3_BE":
        context, groups = "B_FULL", ("E",)
    elif variant == "V3_BFS":
        context, groups = "B_PLUS_F", ("S",)
    elif variant == "V3_FULL":
        context, groups = "B_PLUS_F", ("C", "D", "E") + (("S",) if include_s else ())
    if "S" in groups and not any(feature_group_of(n) == "S" for n in names):
        raise NotImplementedError(
            "SOFA component (S__*) columns are absent from this cache; build it with sofa_components= "
            "from compute_sofa_trends_and_components(rows, context) for the cohort"
        )
    keys = set(VARIANT_KEYS[context])
    return np.asarray(
        [(feature_group_of(n) == "A") or (n in keys) or (feature_group_of(n) in groups) for n in names],
        dtype=bool,
    )


def forbidden_name_tokens() -> Tuple[str, ...]:
    return ("outtime", "delta_sofa", "future", "latent", "label")


# --------------------------------------------------------------------------
# Targets / weights / normalization
# --------------------------------------------------------------------------

def task_mask(meta: Mapping[str, np.ndarray], task: str) -> np.ndarray:
    task = canonical_task(task)
    elig, target = TASK_COLUMNS[task]
    return meta[elig] & ~np.isnan(meta[target])


def weights_stay_balanced(meta: Mapping[str, np.ndarray], task: str) -> np.ndarray:
    """1/n_i weights over eligible rows (0 elsewhere), via evaluation.weights.compute_stay_weights."""

    mask = task_mask(meta, task)
    result = compute_stay_weights([str(s) for s in meta["stay_id"]], [bool(m) for m in mask])
    return np.asarray(result.weights, dtype=np.float64)


def fit_context_normalization(source, variant: str, *, row_mask: Optional[np.ndarray] = None) -> Mapping[str, object]:
    """Train-only affine context normalization.

    ``source`` is a sequence of rows (delegates to performance_v2.context_normalization.fit_normalization)
    or a FeatureCache (same statistics from the cached raw context columns; ``row_mask`` selects the
    training rows). Output format and statistics are identical to V2's.
    """

    if isinstance(source, FeatureCache):
        keys = VARIANT_KEYS[variant]
        mask = np.ones(source.n_rows, dtype=bool) if row_mask is None else np.asarray(row_mask, dtype=bool)
        subjects = [{"subject_id": s} for s in sorted(set(source.meta["subject_id"][mask]))]
        fit_hash = _subject_hash(subjects)
        if not keys:
            return {"transform_version": TRANSFORM_VERSION, "variant": variant, "feature_names": [], "mean": [], "std": [], "fit_subject_hash": fit_hash}
        index = source.column_index()
        matrix = source.matrix[mask][:, [index[k] for k in keys]].astype(np.float64)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return {
            "transform_version": TRANSFORM_VERSION, "variant": variant, "feature_names": list(keys),
            "mean": mean.tolist(), "std": std.tolist(), "fit_subject_hash": fit_hash, "n_rows_fit": int(mask.sum()),
        }
    return fit_normalization(source, variant)


def apply_context_normalization(cache: FeatureCache, normalization: Mapping[str, object], row_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """float32 [n, k]; NaN -> 0 after standardization, as in performance_v2.context_normalization.apply_normalization."""

    mask = np.ones(cache.n_rows, dtype=bool) if row_mask is None else np.asarray(row_mask, dtype=bool)
    keys = list(normalization["feature_names"])
    if not keys:
        return np.zeros((int(mask.sum()), 0), dtype=np.float32)
    index = cache.column_index()
    matrix = cache.matrix[mask][:, [index[k] for k in keys]].astype(np.float64)
    normalized = (matrix - np.asarray(normalization["mean"])) / np.asarray(normalization["std"])
    return np.nan_to_num(normalized, nan=0.0).astype(np.float32)
