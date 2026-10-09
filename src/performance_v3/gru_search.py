"""Protocol stage S3: bounded, cross-fitted GRU candidate search (development evidence only).

Reuse and parity
----------------
* Architecture: ``models.gru.GRUEncoder`` through ``performance_v2.gru_training.SingleOutputGRU`` /
  ``make_encoder_config`` (unchanged). Recovery uses one independent single-horizon head per horizon
  (24h and 48h are separate models; horizons are never chained). ICU time regresses log1p hours with the
  frozen postprocess ``expm1(max(0, .))``; organ support is a logit head with BCE-with-logits, scored on RAW
  probabilities (AUPRC primary).
* Data/context: statics = the 8 frozen encoded statics + the V2 context vector of the variant, fitted
  train-fold-only (same affine rule as ``performance_v2.context_normalization``). Batches are built from
  compact numpy arrays with exactly the semantics of ``performance_v2.gru_training.build_canonical_batch``
  (None -> 0.0, row padding mask, ``encode_statics``); the loop mirrors ``train_gru`` (Adam, standardised
  targets for regression, best-state restore, patience) so full-pool search stays tractable.
* Folds and inner early-stopping subjects are the SAME as ``performance_v3.search`` (crossfit fold of the
  subject id; 10% of TRAINING subjects hashed per outer fold).

Information parity statement
----------------------------
Feature variants that add engineered C/D/E/S columns are tree-model statistics derived deterministically
from the canonical 8x21 tensor, masks and TSLO. The GRU receives that same canonical tensor plus ONLY the
variant's context keys (B/F groups); a sequence model can learn those summaries from the tensor itself, so
no family receives information the other lacks.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from data.collate import CanonicalBatch
from performance_v2.context_v2 import VARIANT_KEYS, raw_context_features
from performance_v2.gru_training import (
    CARDIAC_GROUPS, SEX_CATEGORIES, TEMPORAL_FEATURE_NAMES, SingleOutputGRU, make_encoder_config,
)
from performance_v2.search_utils import config_hash, sample_configs
from performance_v3.features import CTRL_VARIANT_BY_TASK, TASK_COLUMNS, canonical_task
from performance_v3.models import postprocess_icu
from performance_v3.search import (
    PRIMARY_NAME, TASKS, TrialLedger, _es_mask, assign_folds, fold_metrics, trial_id_for,
)
from performance_v3.splits import V3_CROSSFIT_VERSION

GRU_SEARCH_VERSION = "performance_v3_gru_search_v1"
MAX_EPOCHS = 40
PATIENCE = 6
HIDDEN_DIMS = (64, 128, 192)
NUM_LAYERS = (1, 2)
BATCH_SIZES = (64, 128)
CONTEXT_ORDER = tuple(VARIANT_KEYS["B_PLUS_F"])
GRU_SPACE = {"dropout": (0.10, 0.40, "linear"), "lr": (1e-4, 3e-3, "log"), "weight_decay": (1e-6, 1e-3, "log")}


def context_variant_for(variant: str, task: str) -> str:
    """V2 context-vector variant implied by a V3 feature variant (engineered C/D/E/S are tree-only)."""

    if variant == "V3_CTRL":
        return CTRL_VARIANT_BY_TASK[canonical_task(task)]
    if variant in ("V3_B", "V3_BC", "V3_BD", "V3_BE"):
        return "B_FULL"
    if variant in ("V3_BF", "V3_BFS", "V3_FULL"):
        return "B_PLUS_F"
    raise ValueError("unknown variant " + repr(variant))


# --------------------------------------------------------------------------- data
@dataclass
class GRUPool:
    seq: np.ndarray            # [n,8,21] float32 (None -> 0)
    obs: np.ndarray            # [n,8,21] bool
    tslo: np.ndarray           # [n,8,21] float32
    pad: np.ndarray            # [n,8] bool
    age: np.ndarray            # [n] float64
    sex_idx: np.ndarray        # [n] int (-1 = unmatched)
    card_idx: np.ndarray       # [n] int
    ctx: np.ndarray            # [n,12] float64 raw context in CONTEXT_ORDER (nan allowed)
    meta: Dict[str, np.ndarray]

    @property
    def n_rows(self) -> int:
        return int(self.seq.shape[0])


_META_FLOAT = ("delta_sofa_24", "delta_sofa_48", "icu_time_log1p", "organ_support_label")
_META_BOOL = ("recovery24_eligible", "recovery48_eligible", "icu_time_eligible", "organ_support_eligible")


def _f(value) -> float:
    return math.nan if value is None else float(value)


def pool_from_rows(rows: Iterable[Mapping[str, object]], limit: Optional[int] = None) -> GRUPool:
    seq, obs, tslo, pad, age, sex, card, ctx = [], [], [], [], [], [], [], []
    meta: Dict[str, list] = {k: [] for k in ("subject_id", "stay_id", "prediction_time", "grid_index", "cohort_id") + _META_FLOAT + _META_BOOL}
    for row in itertools.islice(rows, limit) if limit else rows:
        seq.append([[0.0 if v is None else float(v) for v in b] for b in row["history_values"]])
        obs.append([[bool(v) for v in b] for b in row["observation_mask"]])
        tslo.append(row["tslo_hours"])
        pad.append([bool(v) for v in row["padding_mask"]])
        age.append(float(row["age_years"]))
        sex.append(SEX_CATEGORIES.index(str(row["sex_category"])) if str(row["sex_category"]) in SEX_CATEGORIES else -1)
        card.append(CARDIAC_GROUPS.index(str(row["cardiac_condition_group"])) if str(row["cardiac_condition_group"]) in CARDIAC_GROUPS else -1)
        feats = raw_context_features(row)
        ctx.append([_f(feats[k]) for k in CONTEXT_ORDER])
        for key in meta:
            if key in _META_FLOAT:
                meta[key].append(_f(row.get(key)))
            elif key in _META_BOOL:
                meta[key].append(bool(row.get(key)))
            elif key == "grid_index":
                meta[key].append(int(row["grid_index"]))
            else:
                meta[key].append(str(row.get(key, "")))
    return GRUPool(
        seq=np.asarray(seq, dtype=np.float32), obs=np.asarray(obs, dtype=bool), tslo=np.asarray(tslo, dtype=np.float32),
        pad=np.asarray(pad, dtype=bool), age=np.asarray(age, dtype=np.float64), sex_idx=np.asarray(sex, dtype=np.int64),
        card_idx=np.asarray(card, dtype=np.int64), ctx=np.asarray(ctx, dtype=np.float64),
        meta={k: np.asarray(v) for k, v in meta.items()},
    )


def load_pool(root: Path, chunk_dirs, limit: Optional[int] = None) -> GRUPool:
    from performance_v3.data import iter_dev_pool

    return pool_from_rows(iter_dev_pool(root, chunk_dirs, compact=False), limit)


@dataclass
class GRUTask:
    task: str
    idx: np.ndarray        # row indices into the pool (eligible rows)
    y: np.ndarray          # training target (ICU: log1p)
    y_eval: np.ndarray     # evaluation target (ICU: hours)
    fold: np.ndarray
    subject_id: np.ndarray
    stay_id: np.ndarray
    prediction_time: np.ndarray


def prepare_gru_task(pool: GRUPool, task: str, n_folds: int = 5, version: str = V3_CROSSFIT_VERSION) -> GRUTask:
    elig_col, target_col = TASK_COLUMNS[canonical_task(task)]
    mask = pool.meta[elig_col].astype(bool) & ~np.isnan(pool.meta[target_col])
    idx = np.flatnonzero(mask)
    y = pool.meta[target_col][idx].astype(np.float64)
    subjects = pool.meta["subject_id"][idx]
    by_subject = assign_folds(subjects, n_folds=n_folds, version=version)
    fold = np.asarray([by_subject[str(s)] for s in subjects], dtype=np.int32)
    return GRUTask(task, idx, y, np.expm1(y) if task == "icu_stay_time" else y, fold, subjects,
                   pool.meta["stay_id"][idx], pool.meta["prediction_time"][idx])


# --------------------------------------------------------------------------- batches / training
def _norm_params(pool: GRUPool, rows: np.ndarray, variant: str) -> Tuple[List[int], np.ndarray, np.ndarray]:
    keys = list(VARIANT_KEYS[variant])
    cols = [CONTEXT_ORDER.index(k) for k in keys]
    if not cols:
        return cols, np.zeros(0), np.ones(0)
    matrix = pool.ctx[rows][:, cols]
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return cols, mean, np.where(std < 1e-8, 1.0, std)


def _context(pool: GRUPool, rows: np.ndarray, cols, mean, std) -> np.ndarray:
    if not cols:
        return np.zeros((len(rows), 0), dtype=np.float32)
    return np.nan_to_num((pool.ctx[rows][:, cols] - mean) / std, nan=0.0).astype(np.float32)


def make_batch(pool: GRUPool, rows: np.ndarray, targets: np.ndarray, context: np.ndarray, age_mean: float, age_std: float) -> CanonicalBatch:
    n = len(rows)
    sex = np.zeros((n, len(SEX_CATEGORIES)), dtype=np.float32)
    card = np.zeros((n, len(CARDIAC_GROUPS)), dtype=np.float32)
    s_idx, c_idx = pool.sex_idx[rows], pool.card_idx[rows]
    sex[np.arange(n)[s_idx >= 0], s_idx[s_idx >= 0]] = 1.0
    card[np.arange(n)[c_idx >= 0], c_idx[c_idx >= 0]] = 1.0
    age_z = ((pool.age[rows] - age_mean) / age_std).astype(np.float32)[:, None]
    statics = np.concatenate([age_z, sex, card, context], axis=1)
    return CanonicalBatch(
        identifiers={"stay_id": tuple(pool.meta["stay_id"][rows]), "prediction_time": tuple(pool.meta["prediction_time"][rows]),
                     "subject_id": tuple(pool.meta["subject_id"][rows]), "grid_index": tuple(int(g) for g in pool.meta["grid_index"][rows]),
                     "split": tuple("dev" for _ in rows)},
        sequence=torch.from_numpy(pool.seq[rows]), padding_mask=torch.from_numpy(pool.pad[rows]),
        observation_mask=torch.from_numpy(pool.obs[rows]), tslo=torch.from_numpy(pool.tslo[rows]),
        static_features=torch.from_numpy(statics), targets={"y": torch.tensor(targets, dtype=torch.float32)},
        eligibility={"y": torch.ones(n, dtype=torch.bool)}, versions={"performance_v3_gru": "v1"},
        feature_names=TEMPORAL_FEATURE_NAMES,
    )


def train_arrays(model, pool, tr_rows, tr_y, tr_ctx, es_rows, es_y, es_ctx, *, loss_name, lr, weight_decay, batch_size,
                 max_epochs, patience, seed, age_mean, age_std, target_mean, target_std):
    """Mirror of performance_v2.gru_training.train_gru on array batches (+ BCE for support)."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = {"huber": lambda: torch.nn.HuberLoss(delta=1.0), "l1": torch.nn.L1Loss, "bce": torch.nn.BCEWithLogitsLoss}[loss_name]()
    tr_t = (tr_y - target_mean) / target_std
    es_batch = None
    if len(es_rows):
        es_batch = make_batch(pool, es_rows, (es_y - target_mean) / target_std, es_ctx, age_mean, age_std)
    rng = np.random.default_rng(seed)
    best_loss, best_state, bad, best_epoch, epochs_run = math.inf, None, 0, 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(tr_rows))
        for start in range(0, len(order), batch_size):
            sel = order[start:start + batch_size]
            batch = make_batch(pool, tr_rows[sel], tr_t[sel], tr_ctx[sel], age_mean, age_std)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.targets["y"])
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1
        if es_batch is None:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(es_batch), es_batch.targets["y"]).item()
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, bad, best_epoch = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0, epoch + 1
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs_run
    return model, best_epoch, epochs_run


def predict_arrays(model, pool, rows, ctx, age_mean, age_std, target_mean, target_std, *, probability: bool, chunk: int = 4096):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            r = rows[start:start + chunk]
            batch = make_batch(pool, r, np.zeros(len(r), dtype=np.float32), ctx[start:start + chunk], age_mean, age_std)
            out.append(model(batch).numpy())
    raw = np.concatenate(out) if out else np.zeros(0)
    return 1.0 / (1.0 + np.exp(-raw)) if probability else raw * target_std + target_mean


# --------------------------------------------------------------------------- trials
def sample_gru_configs(n: int, seed: int, task: str) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed + 7)
    losses = ("bce",) if task == "organ_support" else ("huber", "l1")
    out = []
    for c in sample_configs(GRU_SPACE, n, seed):
        c = dict(c)
        c["hidden_dim"] = int(rng.choice(HIDDEN_DIMS))
        c["num_layers"] = int(rng.choice(NUM_LAYERS))
        c["batch_size"] = int(rng.choice(BATCH_SIZES))
        c["loss_name"] = str(rng.choice(losses))
        c.update({"family": "gru", "task": task, "max_epochs": MAX_EPOCHS, "patience": PATIENCE})
        c["config_hash"] = config_hash({k: v for k, v in c.items()})
        out.append(c)
    return out


def run_gru_cv(pool: GRUPool, gt: GRUTask, variant: str, config: Mapping[str, object], *, folds: Sequence[int], seed: int,
               max_epochs: Optional[int] = None, version: str = V3_CROSSFIT_VERSION, keep_oof: bool = True) -> Dict[str, object]:
    ctx_variant = context_variant_for(variant, gt.task)
    probability = gt.task == "organ_support"
    per_fold, best_epochs, epochs_all = [], [], []
    oof = np.full(len(gt.idx), np.nan)
    fit_seconds = 0.0
    for k in folds:
        ev = gt.fold == k
        tr = ~ev
        es = tr & _es_mask(gt.subject_id, k, version)
        fit = tr & ~es
        if not es.any() or not fit.any():
            fit, es = tr, np.zeros_like(tr)
        fit_rows, es_rows, ev_rows = gt.idx[fit], gt.idx[es], gt.idx[ev]
        cols, mean, std = _norm_params(pool, fit_rows, ctx_variant)
        fit_ctx, es_ctx, ev_ctx = (_context(pool, r, cols, mean, std) for r in (fit_rows, es_rows, ev_rows))
        age_mean = float(pool.age[fit_rows].mean())
        age_std = float(pool.age[fit_rows].std()) or 1.0
        age_std = age_std if age_std > 1e-8 else 1.0
        y_fit = gt.y[fit]
        if probability:
            t_mean, t_std = 0.0, 1.0
        else:
            t_mean, t_std = float(y_fit.mean()), float(y_fit.std())
            t_std = t_std if t_std > 1e-8 else 1.0
        torch.manual_seed(seed)
        model = SingleOutputGRU(make_encoder_config(ctx_variant, int(config["hidden_dim"]), int(config["num_layers"]), float(config["dropout"])))
        started = time.perf_counter()
        model, best_epoch, epochs_run = train_arrays(
            model, pool, fit_rows, y_fit, fit_ctx, es_rows, gt.y[es], es_ctx, loss_name=str(config["loss_name"]), lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]), batch_size=int(config["batch_size"]),
            max_epochs=int(max_epochs or config.get("max_epochs", MAX_EPOCHS)), patience=int(config.get("patience", PATIENCE)),
            seed=seed, age_mean=age_mean, age_std=age_std, target_mean=t_mean, target_std=t_std)
        fit_seconds += time.perf_counter() - started
        pred = predict_arrays(model, pool, ev_rows, ev_ctx, age_mean, age_std, t_mean, t_std, probability=probability)
        oof[ev] = pred
        per_fold.append(fold_metrics(gt.task, gt.y_eval[ev], pred, gt.stay_id[ev], gt.prediction_time[ev]))
        best_epochs.append(int(best_epoch))
        epochs_all.append(int(epochs_run))
    primary = PRIMARY_NAME[gt.task]
    vals = np.asarray([m[primary] for m in per_fold], dtype=np.float64)
    result = {
        "primary_metric": primary, "primary_mean": float(vals.mean()),
        "primary_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "per_fold_primary": [float(v) for v in vals],
        "secondary_means": {k: float(np.mean([m[k] for m in per_fold])) for k in per_fold[0]},
        "best_iterations": best_epochs, "epochs_run": epochs_all, "fit_seconds": fit_seconds,
        "seconds_per_epoch": fit_seconds / max(1, sum(epochs_all)), "folds": [int(k) for k in folds],
    }
    if keep_oof:
        result["oof"] = oof
    return result


def run_gru_trial_logged(ledger: TrialLedger, pool: GRUPool, gt: GRUTask, variant: str, config: Mapping[str, object], *,
                         folds: Sequence[int], seed: int, oof_dir: Optional[Path] = None, max_epochs: Optional[int] = None):
    tid = trial_id_for("S3", gt.task, "gru", variant, str(config["config_hash"]), seed, folds)
    existing = ledger.get(tid)
    if existing is not None and existing.get("status") == "ok":
        return existing
    started = time.time()
    record: Dict[str, object] = {
        "trial_id": tid, "stage": "S3", "task": gt.task, "family": "gru", "variant": variant, "config": dict(config),
        "config_hash": config["config_hash"], "seed": seed, "n_rows": int(len(gt.idx)), "n_stays": int(len(np.unique(gt.stay_id))),
        "n_features": None, "search_version": GRU_SEARCH_VERSION, "context_variant": context_variant_for(variant, gt.task),
    }
    try:
        result = run_gru_cv(pool, gt, variant, config, folds=folds, seed=seed, max_epochs=max_epochs)
        oof = result.pop("oof")
        if oof_dir is not None and gt.task == "organ_support":
            oof_dir.mkdir(parents=True, exist_ok=True)
            path = oof_dir / (tid.replace("|", "_") + ".npz")
            keep = ~np.isnan(oof)
            np.savez_compressed(path, stay_id=gt.stay_id[keep], prediction_time=gt.prediction_time[keep], subject_id=gt.subject_id[keep],
                                fold=gt.fold[keep], y=gt.y[keep], oof_score=oof[keep])
            record["oof_path"] = str(path)
        record.update(result)
        record["status"] = "ok"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
    record["wall_seconds"] = round(time.time() - started, 2)
    ledger.append(record)
    return record
