"""Read-only, provenance-stamped research evidence over the ALREADY FROZEN V3/V3.1 artifacts.

Nothing here retrains, re-selects, or runs model inference, and no sealed cohort raw data is
opened: curves/histograms are derived only from the saved test-prediction file
(artifacts/performance_v3/phase4/predictions/test_predictions_v1.jsonl) with the stay-balanced
weights used by the frozen evaluation, or served verbatim from frozen JSON/CSV artifacts.
A missing artifact raises ``EvidenceMissing`` (HTTP 404), never a fabricated value.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

RESEARCH_API_VERSION = "performance_v3_research_api_v1"
FROZEN_THRESHOLD = 0.3021480061849302
MODELS = ("V3", "V2_AS_SERVED", "V2_AS_EVALUATED", "CONTROL")
MODEL_NOTES = {
    "V3": "corrected selected V3 release",
    "V2_AS_SERVED": "frozen V2 fed what the V2 serving runtime feeds it (SOFA(t) for every row) - operational comparator",
    "V2_AS_EVALUATED": "frozen V2 in its original offline harness; COMPROMISED by defect D1 (current_sofa missingness leaked horizon availability)",
    "CONTROL": "V2 recipe refit on the V3 development pool with corrected features (modeling-method comparator)",
}
STATUS_FINAL = "FINAL_SEALED_TEST"
STATUS_DEV = "DEVELOPMENT"
STATUS_HIST = "HISTORICAL"
STATUS_DEFECT = "DEFECT_ANALYSIS"
STATUS_PRIV = "PRIVILEGED_DIAGNOSTIC"

DC = "artifacts/performance_v3/data_characterization"
P2 = "artifacts/performance_v3/phase2"
P3 = "artifacts/performance_v3/phase3"
P4 = "artifacts/performance_v3/phase4"
RS = "artifacts/performance_v3/research"
D31 = "artifacts/performance_v3_1/diagnostic"
PRED = P4 + "/predictions/test_predictions_v1.jsonl"


class EvidenceMissing(FileNotFoundError):
    """A required frozen artifact is absent (HTTP 404; never fabricate)."""


# ----------------------------------------------------------------------------- helpers
_SHA_CACHE: Dict[Tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def to_csv(rows: Sequence[Mapping[str, Any]], columns: Optional[Sequence[str]] = None) -> str:
    rows = list(rows)
    cols = list(columns) if columns else []
    if not cols:
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow(["" if r.get(c) is None else _fmt(r.get(c)) for c in cols])
    return buf.getvalue()


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return v


_TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "\\": r"\textbackslash{}"}


def _tex(s: Any) -> str:
    return "".join(_TEX.get(c, c) for c in str(_fmt(s) if s is not None else ""))


def to_latex_table(rows: Sequence[Mapping[str, Any]], caption: str, label: str, columns: Optional[Sequence[str]] = None) -> str:
    rows = list(rows)
    cols = list(columns) if columns else []
    if not cols:
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(_tex(c) for c in cols) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(_tex(r.get(c)) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{" + _tex(caption) + "}", r"\label{" + re_label(label) + "}", r"\end{table}", ""]
    return "\n".join(lines)


def re_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in ":-" else "-" for c in label)


def flatten_rows(obj: Any, max_depth: int = 4) -> List[Dict[str, Any]]:
    """Best-effort flat rows from a JSON document (list of dicts as-is; dicts -> key/value rows)."""

    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            return [_flat(x) for x in obj]
        return [{"index": i, "value": x} for i, x in enumerate(obj)]
    rows: List[Dict[str, Any]] = []

    def walk(node: Any, path: Tuple[str, ...], depth: int) -> None:
        if isinstance(node, dict):
            scalars = {k: v for k, v in node.items() if not isinstance(v, (dict, list))}
            nested = {k: v for k, v in node.items() if isinstance(v, (dict, list))}
            if scalars and (not nested or depth >= max_depth):
                rows.append({"path": ".".join(path), **scalars, **({"_nested": "omitted"} if nested else {})})
            elif scalars:
                rows.append({"path": ".".join(path), **scalars})
            for k, v in nested.items():
                if depth < max_depth:
                    walk(v, path + (str(k),), depth + 1)
        elif isinstance(node, list):
            if node and all(isinstance(x, dict) for x in node):
                for i, x in enumerate(node):
                    walk(x, path + (str(i),), depth + 1)
            elif node and all(not isinstance(x, (dict, list)) for x in node) and len(node) <= 64:
                rows.append({"path": ".".join(path), "values": node})
        else:
            rows.append({"path": ".".join(path), "value": node})

    walk(obj, (), 0)
    return rows


def _flat(d: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = v2 if not isinstance(v2, (dict, list)) else json.dumps(v2, sort_keys=True)
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = 500) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n = len(x)
    if n <= max_points:
        return x, y, {"applied": False, "points_in": int(n), "points_out": int(n)}
    idx = np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))
    return x[idx], y[idx], {"applied": True, "points_in": int(n), "points_out": int(len(idx)), "method": "uniform index subsampling retaining both endpoints"}


# ----------------------------------------------------------------------------- weighted metrics from saved predictions
def stay_weights(stay_ids: Sequence[str]) -> np.ndarray:
    counts: Dict[str, int] = defaultdict(int)
    for s in stay_ids:
        counts[s] += 1
    return np.asarray([1.0 / counts[s] for s in stay_ids], dtype=float)


def weighted_roc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-s, kind="mergesort")
    s_o, y_o, w_o = s[order], y[order], w[order]
    pos = float(np.sum(w_o * (y_o == 1)))
    neg = float(np.sum(w_o * (y_o == 0)))
    tp = np.cumsum(w_o * (y_o == 1))
    fp = np.cumsum(w_o * (y_o == 0))
    distinct = np.r_[np.where(np.diff(s_o) != 0)[0], len(s_o) - 1]
    fpr = np.r_[0.0, fp[distinct] / neg]
    tpr = np.r_[0.0, tp[distinct] / pos]
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def weighted_pr(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-s, kind="mergesort")
    s_o, y_o, w_o = s[order], y[order], w[order]
    pos = float(np.sum(w_o * (y_o == 1)))
    tp = np.cumsum(w_o * (y_o == 1))
    tot = np.cumsum(w_o)
    distinct = np.r_[np.where(np.diff(s_o) != 0)[0], len(s_o) - 1]
    recall = tp[distinct] / pos
    precision = tp[distinct] / tot[distinct]
    prev = np.r_[0.0, recall[:-1]]
    ap = float(np.sum((recall - prev) * precision))  # step integral (same convention as frozen evaluation)
    return np.r_[0.0, recall], np.r_[precision[0], precision], ap


def _threshold_row(y, p, w, thr):
    pred = p >= thr
    tp = float(np.sum(w[(y == 1) & pred])); fp = float(np.sum(w[(y == 0) & pred]))
    fn = float(np.sum(w[(y == 1) & ~pred])); tn = float(np.sum(w[(y == 0) & ~pred]))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    return {"threshold": float(thr), "precision": prec, "recall": rec, "specificity": spec, "f1": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


# ----------------------------------------------------------------------------- main class
class ResearchEvidence:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._cache: Dict[str, Any] = {}
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._build_registry()

    # -- file access
    def path(self, rel: str) -> Path:
        p = self.root / rel
        if not p.exists():
            raise EvidenceMissing(f"frozen artifact absent: {rel}")
        return p

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def read_json(self, rel: str) -> Any:
        key = "json:" + rel
        if key not in self._cache:
            self._cache[key] = json.loads(self.path(rel).read_text(encoding="utf-8"))
        return self._cache[key]

    def read_csv(self, rel: str) -> List[Dict[str, Any]]:
        key = "csv:" + rel
        if key not in self._cache:
            with self.path(rel).open(newline="", encoding="utf-8") as handle:
                rows = []
                for r in csv.DictReader(handle):
                    rows.append({k: _num(v) for k, v in r.items()})
            self._cache[key] = rows
        return self._cache[key]

    def read_ledger(self, rel: str) -> List[Dict[str, Any]]:
        key = "jsonl:" + rel
        if key not in self._cache:
            self._cache[key] = [json.loads(l) for l in self.path(rel).read_text(encoding="utf-8").splitlines() if l.strip()]
        return self._cache[key]

    def sha(self, rel: str) -> Optional[str]:
        p = self.root / rel
        return sha256_file(p) if p.exists() else None

    # -- saved test predictions (read-only)
    def predictions(self) -> Dict[str, Any]:
        if "pred" in self._cache:
            return self._cache["pred"]
        path = self.path(PRED)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self._cache["pred"] = {"rows": rows}
        return self._cache["pred"]

    def _support_arrays(self, model: str, which: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        key = f"sup:{model}:{which}"
        if key not in self._cache:
            ys, ps, st = [], [], []
            field = "support_calibrated_probability" if which == "calibrated" else "support_raw_probability"
            for r in self.predictions()["rows"]:
                lab = r["labels"]
                if lab.get("organ_support_eligible") and lab.get("organ_support_label") is not None:
                    ys.append(float(lab["organ_support_label"])); ps.append(float(r["models"][model][field])); st.append(r["stay_id"])
            y = np.asarray(ys); p = np.asarray(ps)
            self._cache[key] = (y, p, stay_weights(st), st)
        return self._cache[key]

    # ------------------------------------------------------------------ registry
    def _reg(self, id_: str, title: str, group: str, status: str, partition: str, metric_definition: str, sources: Sequence[str],
             bundle: str, chart_hint: str, loader: Callable[[], Dict[str, Any]], units: Optional[Mapping[str, str]] = None,
             n_rows: Optional[int] = None, n_stays: Optional[int] = None, note: str = "") -> None:
        self._registry[id_] = {"id": id_, "title": title, "group": group, "evidence_status": status, "partition": partition,
                               "metric_definition": metric_definition, "sources": list(sources), "bundle": bundle, "chart_hint": chart_hint,
                               "loader": loader, "units": dict(units or {}), "n_rows": n_rows, "n_stays": n_stays, "note": note}

    def _build_registry(self) -> None:
        R = self._reg
        # cohort analytics (development pool + learning-curve evaluation cohort)
        for id_, fn, title, hint, units in (
            ("cohort_summary", "cohort_summary", "Cohort summary", "table", {"subjects": "count", "stays": "count", "rows": "legal cutoff rows"}),
            ("cohort_subtype", "subtype_distribution", "Cardiac subtype composition", "bar", {"proportion": "fraction"}),
            ("cohort_age_sex", "age_sex_composition", "Age and sex composition", "bar", {"age": "years"}),
            ("cohort_duration", "episode_duration_summary", "ICU-like episode duration", "hist", {"duration": "hours"}),
            ("cohort_eligibility", "target_eligibility", "Task eligibility (rows and unique stays)", "table", {"rows": "count", "stays": "count"}),
            ("cohort_outcome_prevalence", "outcome_prevalence", "Organ-support outcome prevalence and censoring", "bar", {"prevalence": "fraction"}),
            ("cohort_missingness_feature", "missingness_by_feature", "Missingness by feature", "bar", {"missing_fraction": "fraction"}),
            ("cohort_missingness_bin", "missingness_by_bin", "Observation availability by 6-hour bin", "heatmap", {"observed_fraction": "fraction"}),
            ("cohort_sofa_distributions", "sofa_distributions", "SOFA distributions", "hist", {"sofa": "points (0-24)"}),
            ("cohort_cutoffs_per_stay", "legal_cutoffs_per_stay", "Legal cutoffs per stay", "hist", {"cutoffs": "count"}),
            ("cohort_correlations", "correlations", "Feature correlations (channel means)", "heatmap", {"r": "pearson"}),
            ("cohort_between_subject", "between_subject_variability", "Between-subject variability (ICC)", "bar", {"icc": "ratio"}),
            ("cohort_variable_distributions", "variable_distributions", "Variable distributions (genuine observations)", "hist", {}),
        ):
            R(id_, title, "cohort", STATUS_DEV, "development pool (V1 train+validation + D1-D3) and LCE", "descriptive, development partitions only; no final-test data",
              [f"{DC}/{fn}.json"], "n/a", hint, (lambda fn=fn: self._cohort_doc(fn)), units)
        R("cohort_sofa_by_cutoff", "SOFA by cutoff index", "cohort", STATUS_DEV, "development pool and LCE", "mean/sd/median total SOFA at each cutoff index",
          [f"{DC}/sofa_by_cutoff.csv"], "n/a", "line", lambda: self._csv_data(f"{DC}/sofa_by_cutoff.csv"), {"mean": "SOFA points", "sd": "SOFA points", "p50": "SOFA points", "n": "rows"})
        # classification from saved test predictions
        sup_src = [PRED, P4 + "/final_metrics_v3.json"]
        R("support_roc", "Organ-support ROC (calibrated probability, stay-balanced)", "classification", STATUS_FINAL, "sealed V3 test SYN-V3-T (1,500 subjects)",
          "weighted TPR vs FPR, weights 1/n_i per stay", sup_src, "V3", "line", lambda: self._roc_pr("roc"), {"fpr": "fraction", "tpr": "fraction"})
        R("support_pr", "Organ-support precision-recall (calibrated probability, stay-balanced)", "classification", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "weighted precision vs recall (step integral = AUPRC)", sup_src, "V3", "line", lambda: self._roc_pr("pr"), {"recall": "fraction", "precision": "fraction"})
        R("support_reliability", "Organ-support reliability (raw vs calibrated)", "classification", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "10 equal-mass bins, stay-balanced observed rate vs mean predicted probability", [P4 + "/calibration_evidence_v1.json"], "V3", "line", self._reliability,
          {"mean_predicted": "probability", "observed_rate": "fraction"})
        R("support_score_distribution", "Predicted-probability distributions by outcome", "classification", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "20 equal-width bins of calibrated probability, stay-balanced weight by observed outcome", sup_src, "V3", "hist", self._score_dist, {"probability": "probability"})
        R("support_confusion", "Confusion matrices at frozen thresholds", "classification", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "stay-balanced weighted counts and row counts at each model's frozen threshold", [P4 + "/calibration_evidence_v1.json", P4 + "/final_metrics_v3.json"], "V3", "table", self._confusion,
          {"TP": "weighted stays", "threshold": "probability"})
        R("support_threshold_sensitivity", "Threshold sensitivity (read-only explorer)", "classification", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "precision/recall/F1/specificity vs threshold; PRECOMPUTED, never alters the frozen deployed threshold", sup_src, "V3", "line", self._threshold_sensitivity,
          {"threshold": "probability", "precision": "fraction", "recall": "fraction", "f1": "fraction"})
        R("classification_metrics", "Organ-support metrics by model", "classification", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "calibrated and raw AUROC/AUPRC, Brier, log loss, F1 at frozen threshold; 95% stay-cluster bootstrap CI", [P4 + "/final_metrics_v3.json"], "V3", "table", self._support_metrics)
        # regression
        R("regression_residuals", "Residual distributions (prediction minus observed)", "regression", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "unweighted and stay-balanced counts in 40 equal-width bins (1st-99th percentile range, tails recorded)", [PRED], "V3", "hist", self._residuals,
          {"residual_recovery24": "SOFA points", "residual_recovery48": "SOFA points", "residual_icu_stay_time": "hours"})
        R("regression_predicted_vs_observed", "Predicted vs observed (quantile-binned means)", "regression", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "20 predicted-quantile bins; stay-balanced mean predicted and mean observed", [PRED], "V3", "scatter", self._pred_vs_obs,
          {"mean_predicted": "SOFA points or hours", "mean_observed": "SOFA points or hours"})
        R("regression_metrics", "Regression metrics by model", "regression", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "MAE, RMSE, median AE (SOFA points) and ICU weighted median AE (hours), stay-balanced", [P4 + "/final_metrics_v3.json"], "V3", "table", self._regression_metrics,
          {"mae": "SOFA points (recovery) / hours (ICU)", "weighted_median_ae_hours": "hours"})
        R("error_slices", "Prespecified error slices", "regression", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "primary metric per prespecified slice with row and stay denominators", [P4 + "/error_analysis_v1.json"], "V3", "bar", self._error_slices)
        # comparison / paired
        R("model_comparison", "Model comparison matrix (four tasks)", "comparison", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "primary metric per task for V3 / V2 as served / V2 as evaluated (compromised) / control / naive", [P4 + "/final_metrics_v3.json", P4 + "/naive_baselines_v1.json"], "V3", "table", self._comparison_matrix)
        R("paired_differences", "Paired V3 vs comparator differences", "comparison", STATUS_FINAL, "sealed V3 test SYN-V3-T",
          "metric(V3) - metric(comparator), paired stay-cluster bootstrap, 2,000 replicates, 95% and 98.75% percentile CIs",
          [P4 + "/paired_v2_v3_metrics.json"], "V3", "table", self._paired)
        R("historical_context", "Historical context (V1 / V2 as published / V3 / naive)", "history", STATUS_HIST, "historical final tests (each consumed once) and V3 sealed test",
          "values copied from frozen reports; V2-as-published ICU/support contain the defect D1 inflation; different cohorts, not directly comparable",
          [RS + "/tables/t20b_historical_context.csv"], "n/a", "table", lambda: self._csv_data(RS + "/tables/t20b_historical_context.csv"))
        # development evidence
        R("learning_curve", "Learning curves (frozen V2 recipe, corrected features)", "learning_curve", STATUS_DEV, "LCE (1,500 subjects), fixed development holdout",
          "primary metric vs training subjects; mean/sd over repeats; 95% bootstrap CI on repeat 0", [DC + "/learning_curve_table.csv", DC + "/learning_curve_metrics.json"], "n/a", "line", self._learning_curve,
          {"size_subjects": "subjects", "train_seconds": "seconds"}, note="development evidence on a reused holdout; not final validation")
        R("ablation", "Feature-variant ablation (S2)", "ablation", STATUS_DEV, "5/3-fold subject-grouped cross-fitting on the 8,000-subject pool",
          "primary cross-fitted metric per feature variant, mean over seeds", [P2 + "/trial_ledger_v1.jsonl"], "n/a", "bar", self._ablation)
        R("leaderboard", "Model-family leaderboard and all search trials", "comparison", STATUS_DEV, "cross-fitted development pool",
          "best cross-fitted primary metric per family; every recorded trial including failures", [P2 + "/trial_ledger_v1.jsonl"], "n/a", "table", self._leaderboard)
        R("robustness", "Robustness stress tests", "robustness", STATUS_DEV, "LCE stay subsample (500 stays)",
          "primary metric under perturbation; deltas vs unperturbed; no claim about real-world robustness", [P2 + "/robustness_v1.json"], "V3", "bar", self._robustness)
        R("explainability_global", "Global TreeSHAP importance", "explainability", STATUS_DEV, "LCE deterministic sample",
          "mean |SHAP| on the RAW margin (pre-calibration/postprocess); non-causal", [P2 + "/explainability_v1.json"], "V3", "bar", self._explain_global)
        R("defect_d1", "Defect D1: V2 as evaluated vs as served", "defect", STATUS_DEFECT, "LCE (development) and sealed V3 test",
          "frozen V2 models fed null vs actual current SOFA; clean-subset caveat", [DC + "/v2_modes_on_lce_v1.json", P4 + "/final_metrics_v3.json", DC + "/null_pattern_audit_v1.json"], "V2_AS_EVALUATED", "table", self._defect)
        R("diagnostic_latent_state", "V3.1 latent-state diagnostic (privileged)", "diagnostic", STATUS_PRIV, "fresh development cohort DIAG_EVAL (900 subjects)",
          "stay-balanced AUPRC/AUROC of privileged simulator-state predictors vs frozen V3; NOT deployable, NOT a clinical bound",
          [D31 + "/oracle_metrics.json", D31 + "/oracle_bootstrap.json", D31 + "/oracle_comparison.json", D31 + "/oracle_verification_v1.json"], "n/a", "table", self._diagnostic)
        R("diagnostic_candidate_development", "V3.1 candidate development and held-out decision", "diagnostic", STATUS_DEV, "V3.1 development pool + V31_HOLD (1,500 subjects)",
          "32 pre-declared trials and one held-out comparison vs frozen V3; Gate C failed, V3 retained", ["artifacts/performance_v3_1/phase2/candidate_selection_v1.json", "artifacts/performance_v3_1/phase2/hold_evaluation_v1.json"], "V3", "table", self._candidate_dev)
        R("provenance", "Bundle and scientific provenance", "history", STATUS_FINAL, "n/a", "hashes and lineage of the served release",
          [P3 + "/selected_models_v3.json", "artifacts/performance_v3/product_handoff_v1.json"], "V3", "table", self._provenance)

    # ------------------------------------------------------------------ catalogue
    def provenance_block(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        sources = []
        for rel in entry["sources"]:
            sources.append({"path": rel, "sha256": self.sha(rel), "present": self.exists(rel)})
        primary = next((s for s in sources if s["present"]), sources[0] if sources else None)
        return {"id": entry["id"], "title": entry["title"], "group": entry["group"], "evidence_status": entry["evidence_status"],
                "partition": entry["partition"], "n_rows": entry["n_rows"], "n_stays": entry["n_stays"], "metric_definition": entry["metric_definition"],
                "source_artifact": primary["path"] if primary else None, "source_sha256": primary["sha256"] if primary else None, "sources": sources,
                "bundle": entry["bundle"], "chart_hint": entry["chart_hint"], "units": entry["units"], "note": entry["note"],
                "downloadable_formats": ["json", "csv", "tex"], "api_version": RESEARCH_API_VERSION}

    def index(self) -> Dict[str, Any]:
        items = [self.provenance_block(e) for e in self._registry.values()]
        for it in items:
            it["available"] = all(s["present"] for s in it["sources"])
        return {"api_version": RESEARCH_API_VERSION, "count": len(items), "datasets": items,
                "status_legend": {STATUS_FINAL: "one-time sealed V3 test evaluation", STATUS_DEV: "development evidence (not final validation)",
                                  STATUS_HIST: "historical frozen results", STATUS_DEFECT: "analysis of defect D1",
                                  STATUS_PRIV: "privileged simulator-state diagnostic; never deployable"}}

    def dataset(self, id_: str) -> Dict[str, Any]:
        if id_ not in self._registry:
            raise KeyError(id_)
        entry = self._registry[id_]
        data = entry["loader"]()
        if "rows" not in data:
            data = {**data, "rows": flatten_rows({k: v for k, v in data.items() if k not in ("provenance",)})}
        prov = self.provenance_block(entry)
        if data.get("n_rows") is not None and prov["n_rows"] is None:
            prov["n_rows"] = data["n_rows"]
        if data.get("n_stays") is not None and prov["n_stays"] is None:
            prov["n_stays"] = data["n_stays"]
        return {"provenance": prov, "data": data}

    # ------------------------------------------------------------------ loaders
    def _cohort_doc(self, name: str) -> Dict[str, Any]:
        doc = self.read_json(f"{DC}/{name}.json")
        return {"document": doc, "artifact_status": doc.get("artifact_status")}

    def _csv_data(self, rel: str) -> Dict[str, Any]:
        rows = self.read_csv(rel)
        return {"rows": rows, "columns": list(rows[0].keys()) if rows else []}

    def _roc_pr(self, kind: str) -> Dict[str, Any]:
        series, rows, aucs = {}, [], {}
        for model in MODELS:
            y, p, w, st = self._support_arrays(model, "calibrated")
            if kind == "roc":
                x, yy, auc = weighted_roc(y, p, w)
            else:
                x, yy, auc = weighted_pr(y, p, w)
            xs, ys, ds = _downsample(x, yy)
            series[model] = {"x": [round(float(v), 6) for v in xs], "y": [round(float(v), 6) for v in ys], "auc_or_ap": auc, "downsampling": ds,
                             "note": MODEL_NOTES[model]}
            aucs[model] = auc
            xn, yn = ("fpr", "tpr") if kind == "roc" else ("recall", "precision")
            rows += [{"model": model, xn: float(a), yn: float(b)} for a, b in zip(xs, ys)]
        y0, _, w0, st0 = self._support_arrays("V3", "calibrated")
        return {"curve": kind, "series": series, "rows": rows, "metric_by_model": aucs, "n_rows": int(len(y0)), "n_stays": len(set(st0)),
                "prevalence_stay_balanced": float(np.sum(w0 * y0) / np.sum(w0)),
                "weighting": "stay-balanced 1/n_i", "probability_type": "calibrated (each model's own frozen calibrator)"}

    def _reliability(self) -> Dict[str, Any]:
        ce = self.read_json(P4 + "/calibration_evidence_v1.json")["models"]
        rows, out = [], {}
        for m in MODELS:
            if m not in ce:
                continue
            out[m] = {"reliability_calibrated": ce[m]["reliability_calibrated"], "reliability_raw": ce[m]["reliability_raw"],
                      "calibration_slope": ce[m]["calibrated"]["calibration_slope"], "calibration_intercept": ce[m]["calibrated"]["calibration_intercept"],
                      "ece_10_equal_mass": ce[m]["calibrated"]["ece_10_equal_mass"], "brier": ce[m]["calibrated"]["brier"], "log_loss": ce[m]["calibrated"]["log_loss"]}
            for kind in ("calibrated", "raw"):
                for b in ce[m][f"reliability_{kind}"]:
                    rows.append({"model": m, "probability_type": kind, **{k: b[k] for k in ("bin", "mean_predicted", "observed_rate", "n_rows", "weight")}})
        return {"models": out, "rows": rows, "diagonal": "perfect calibration: observed_rate = mean_predicted"}

    def _score_dist(self) -> Dict[str, Any]:
        edges = np.linspace(0, 1, 21)
        rows, out = [], {}
        for m in MODELS:
            y, p, w, _ = self._support_arrays(m, "calibrated")
            for outcome in (0, 1):
                sel = y == outcome
                h, _ = np.histogram(p[sel], bins=edges, weights=w[sel])
                c, _ = np.histogram(p[sel], bins=edges)
                for i in range(20):
                    rows.append({"model": m, "outcome": "positive" if outcome else "negative", "bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                                 "weighted_count": float(h[i]), "row_count": int(c[i])})
        return {"rows": rows, "frozen_threshold_V3": FROZEN_THRESHOLD, "bins": 20}

    def _confusion(self) -> Dict[str, Any]:
        ce = self.read_json(P4 + "/calibration_evidence_v1.json")["models"]
        fm = self.read_json(P4 + "/final_metrics_v3.json")
        rows = []
        for m in MODELS:
            c = ce[m]
            w, r = c["confusion_weighted"], c["confusion_rows"]
            tp, fp, fn, tn = w["TP"], w["FP"], w["FN"], w["TN"]
            prec = tp / (tp + fp); rec = tp / (tp + fn); spec = tn / (tn + fp)
            rows.append({"model": m, "threshold": c["frozen_threshold"], "TP_weighted": tp, "FP_weighted": fp, "FN_weighted": fn, "TN_weighted": tn,
                         "TP_rows": r["TP"], "FP_rows": r["FP"], "FN_rows": r["FN"], "TN_rows": r["TN"], "precision": prec, "recall_sensitivity": rec,
                         "specificity": spec, "f1": 2 * prec * rec / (prec + rec), "n_rows": c["n_rows"], "n_stays": c["n_stays"],
                         "f1_reported_in_final_metrics": fm["models"][m]["organ_support"]["f1"], "note": MODEL_NOTES[m]})
        return {"rows": rows, "frozen_threshold_V3": FROZEN_THRESHOLD, "weighting": "stay-balanced 1/n_i for *_weighted; raw row counts for *_rows"}

    def _threshold_sensitivity(self) -> Dict[str, Any]:
        fm = self.read_json(P4 + "/final_metrics_v3.json")
        grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
        rows = []
        for m in MODELS:
            y, p, w, _ = self._support_arrays(m, "calibrated")
            for t in grid:
                rows.append({"model": m, **_threshold_row(y, p, w, float(t))})
            rows.append({"model": m, **_threshold_row(y, p, w, float(fm["thresholds"][m])), "frozen": True})
        return {"rows": rows, "frozen_thresholds": fm["thresholds"], "frozen_threshold_V3": FROZEN_THRESHOLD,
                "statement": "read-only explorer over saved test predictions; it does not change, tune or select the deployed operating threshold"}

    def _support_metrics(self) -> Dict[str, Any]:
        fm = self.read_json(P4 + "/final_metrics_v3.json")
        rows = []
        for m in MODELS:
            c, r = fm["models"][m]["organ_support"], fm["models"][m]["organ_support_raw_scores"]
            ci = c.get("primary_ci", {}).get("0.95")
            rows.append({"model": m, "calibrated_auprc": c["auprc"], "calibrated_auprc_ci95_lo": ci[0] if ci else None, "calibrated_auprc_ci95_hi": ci[1] if ci else None,
                         "calibrated_auroc": c["auroc"], "brier": c["brier"], "log_loss": c["log_loss"], "f1_at_frozen_threshold": c["f1"], "precision": c["precision"],
                         "recall": c["recall"], "specificity": c["specificity"], "raw_auprc": r["auprc"], "raw_auroc": r["auroc"], "raw_brier": r["brier"],
                         "n_rows": c["n_rows"], "n_stays": c["n_stays"], "frozen_threshold": fm["thresholds"][m], "note": MODEL_NOTES[m]})
        return {"rows": rows, "ci_method": "paired-resample stay-cluster percentile bootstrap, 2,000 replicates, seed 20261012; population uncertainty, not individual prediction intervals"}

    def _task_arrays(self, task: str, model: str):
        key = f"reg:{task}:{model}"
        if key not in self._cache:
            ys, ps, st, band, sign = [], [], [], [], []
            for r in self.predictions()["rows"]:
                lab, m = r["labels"], r["models"][model]
                if task == "recovery24" and lab.get("recovery24_eligible") and lab.get("delta_sofa_24") is not None:
                    ys.append(float(lab["delta_sofa_24"])); ps.append(float(m["recovery24_delta"]))
                elif task == "recovery48" and lab.get("recovery48_eligible") and lab.get("delta_sofa_48") is not None:
                    ys.append(float(lab["delta_sofa_48"])); ps.append(float(m["recovery48_delta"]))
                elif task == "icu_stay_time" and lab.get("icu_time_eligible") and lab.get("icu_time_log1p") is not None:
                    ys.append(float(np.expm1(lab["icu_time_log1p"]))); ps.append(float(m["icu_remaining_hours"]))
                else:
                    continue
                st.append(r["stay_id"])
            self._cache[key] = (np.asarray(ys), np.asarray(ps), stay_weights(st), st)
        return self._cache[key]

    def _residuals(self) -> Dict[str, Any]:
        rows, summary = [], {}
        units = {"recovery24": "SOFA points", "recovery48": "SOFA points", "icu_stay_time": "hours"}
        for task in ("recovery24", "recovery48", "icu_stay_time"):
            # common bin range from V3 residuals, shared across models for comparability
            y, p, w, _ = self._task_arrays(task, "V3")
            res = p - y
            lo, hi = np.percentile(res, [1, 99])
            edges = np.linspace(lo, hi, 41)
            for m in MODELS:
                y, p, w, st = self._task_arrays(task, m)
                r = p - y
                inside = (r >= lo) & (r <= hi)
                h, _ = np.histogram(r[inside], bins=edges, weights=w[inside])
                c, _ = np.histogram(r[inside], bins=edges)
                for i in range(40):
                    rows.append({"task": task, "model": m, "bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]), "row_count": int(c[i]),
                                 "weighted_count": float(h[i]), "unit": units[task]})
                summary[f"{task}:{m}"] = {"n_rows": int(len(r)), "n_stays": len(set(st)), "below_range": int((r < lo).sum()), "above_range": int((r > hi).sum()),
                                           "mean_residual": float(np.sum(w * r) / np.sum(w)), "unit": units[task]}
        return {"rows": rows, "summary": summary, "units": units, "weighting": "stay-balanced 1/n_i (weighted_count); unweighted (row_count)"}

    def _pred_vs_obs(self) -> Dict[str, Any]:
        rows = []
        for task in ("recovery24", "recovery48", "icu_stay_time"):
            for m in MODELS:
                y, p, w, _ = self._task_arrays(task, m)
                qs = np.quantile(p, np.linspace(0, 1, 21))
                qs[-1] += 1e-9
                idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, 19)
                for b in range(20):
                    sel = idx == b
                    if sel.sum() == 0:
                        continue
                    ws = w[sel]
                    rows.append({"task": task, "model": m, "bin": b, "n_rows": int(sel.sum()), "mean_predicted": float(np.sum(ws * p[sel]) / np.sum(ws)),
                                 "mean_observed": float(np.sum(ws * y[sel]) / np.sum(ws)), "unit": "hours" if task == "icu_stay_time" else "SOFA points (delta)"})
        return {"rows": rows, "binning": "20 predicted-quantile bins per task and model; weighted means (stay-balanced)"}

    def _regression_metrics(self) -> Dict[str, Any]:
        fm = self.read_json(P4 + "/final_metrics_v3.json")
        rows = []
        for task in ("recovery24", "recovery48", "icu_stay_time"):
            for m in MODELS:
                d = fm["models"][m][task]
                ci = d.get("primary_ci", {}).get("0.95")
                rows.append({"task": task, "model": m, "mae": d.get("mae"), "rmse": d.get("rmse"), "median_ae": d.get("median_ae"),
                             "weighted_median_ae_hours": d.get("weighted_median_ae_hours"), "primary_ci95_lo": ci[0] if ci else None, "primary_ci95_hi": ci[1] if ci else None,
                             "n_rows": d["n_rows"], "n_stays": d["n_stays"],
                             "primary_metric": "weighted_median_ae_hours" if task == "icu_stay_time" else "mae",
                             "unit": "hours" if task == "icu_stay_time" else "SOFA points"})
        return {"rows": rows}

    def _error_slices(self) -> Dict[str, Any]:
        ea = self.read_json(P4 + "/error_analysis_v1.json")
        rows = []
        for task, dims in ea["tasks"].items():
            for dim, slices in dims.items():
                for name, d in slices.items():
                    models = d.get("models", {})
                    for m, v in models.items():
                        rows.append({"task": task, "dimension": dim, "slice": name, "model": m, "value": v, "n_rows": d.get("n_rows"), "n_stays": d.get("n_stays")})
        return {"rows": rows, "slice_cutpoints": ea.get("slice_cutpoints"), "prespecified": ea.get("prespecified"),
                "value_meaning": "task primary metric within the slice (MAE in SOFA points; ICU weighted median AE in hours; support calibrated AUPRC)"}

    def _comparison_matrix(self) -> Dict[str, Any]:
        fm = self.read_json(P4 + "/final_metrics_v3.json")
        nv = self.read_json(P4 + "/naive_baselines_v1.json")["tasks"]
        rows = []
        spec = (("recovery24", "mae", "SOFA points", "mae"), ("recovery48", "mae", "SOFA points", "mae"),
                ("icu_stay_time", "weighted_median_ae_hours", "hours", "weighted_median_ae_hours"), ("organ_support", "auprc", "probability-score AUPRC (calibrated)", "auprc"))
        for task, metric, unit, nk in spec:
            row = {"task": task, "metric": metric, "unit": unit, "lower_is_better": task != "organ_support", "naive": nv[task][nk]}
            for m in MODELS:
                d = fm["models"][m][task]
                ci = d.get("primary_ci", {}).get("0.95")
                row[m] = d[metric]
                row[m + "_ci95"] = ci
            row["denominator_rows"] = fm["denominators"][task]
            rows.append(row)
        return {"rows": rows, "model_notes": MODEL_NOTES, "reading_guide": [
            "V3 vs V2_AS_SERVED is the operational comparison; V3 vs CONTROL is the modeling-method comparison.",
            "V2_AS_EVALUATED contains defect D1 and is shown only for historical attribution; it is not reliable serving performance.",
            "ICU V3 equals the repaired control; recovery48 advantage over control is not established at the stricter paired interval."]}

    def _paired(self) -> Dict[str, Any]:
        pv = self.read_json(P4 + "/paired_v2_v3_metrics.json")
        rows = []
        for key, d in pv["point_differences"].items():
            parts = key.split("|")
            rows.append({"task": parts[0], "metric": parts[1], "comparison": parts[2], "diff_v3_minus_comparator": d["diff_point"],
                         "ci95_lo": d["ci"]["0.95"][0], "ci95_hi": d["ci"]["0.95"][1], "ci9875_lo": d["ci"]["0.9875"][0], "ci9875_hi": d["ci"]["0.9875"][1],
                         "n_rows": d["n_rows"], "n_stays": d["n_stays"], "comparator_value": d["point_a"], "v3_value": d["point_b"]})
        return {"rows": rows, "promotion": pv["promotion"], "primary_comparator": pv["primary_comparator"],
                "interval_method": "paired stay-cluster percentile bootstrap (population-level, not an individual prediction interval)"}

    def _learning_curve(self) -> Dict[str, Any]:
        table = self.read_csv(DC + "/learning_curve_table.csv")
        meta = self.read_json(DC + "/learning_curve_metrics.json")
        times: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for f in meta["fits"]:
            times[(f["task"], f["size_subjects"])].append(f["train_seconds"])
        rows = []
        for r in table:
            t = times.get((r["task"], int(r["size_subjects"])))
            rows.append({**r, "train_seconds_mean": float(np.mean(t)) if t else None})
        return {"rows": rows, "caveat": meta.get("caveat"), "evaluation_cohort": meta.get("evaluation_cohort"), "n_eval_subjects": meta.get("n_eval_subjects"),
                "sizes": meta.get("sizes")}

    def _trials(self) -> List[Dict[str, Any]]:
        return self.read_ledger(P2 + "/trial_ledger_v1.jsonl")

    def _ablation(self) -> Dict[str, Any]:
        by: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for r in self._trials():
            if r.get("stage") == "S2" and r.get("status") == "ok":
                by[(r["task"], r["variant"])].append(r["primary_mean"])
        rows = [{"task": t, "variant": v, "primary_mean": float(np.mean(x)), "n_runs": len(x), "primary_metric": "weighted_median_ae_hours" if t == "icu_stay_time" else ("auprc" if t == "organ_support" else "mae")}
                for (t, v), x in sorted(by.items())]
        s0 = [{"task": r["task"], "variant": "V2_RECIPE_CONTROL(S0)", "primary_mean": r["primary_mean"], "n_runs": 1} for r in self._trials() if r.get("stage") == "S0" and r.get("status") == "ok"]
        return {"rows": rows + s0, "variant_definitions": {"V3_CTRL": "frozen V2 context per task", "V3_B": "B_FULL", "V3_BF": "B_PLUS_F", "V3_BC": "B_FULL + C (deltas/slopes)",
                "V3_BD": "B_FULL + D (windowed stats)", "V3_BE": "B_FULL + E (observed fractions)", "V3_BFS": "B_PLUS_F + SOFA components", "V3_FULL": "B_PLUS_F + C + D + E"},
                "note": "differences are mostly within the pre-registered tie tolerance; negative results retained"}

    def _leaderboard(self) -> Dict[str, Any]:
        trials = [{"stage": r["stage"], "task": r["task"], "family": r["family"], "variant": r["variant"], "primary_metric": r.get("primary_metric"),
                   "primary_mean": r.get("primary_mean"), "primary_sd": r.get("primary_sd"), "n_features": r.get("n_features"), "status": r["status"],
                   "wall_seconds": r.get("wall_seconds"), "error": r.get("error")} for r in self._trials()]
        best: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for t in trials:
            if t["status"] != "ok" or t["primary_mean"] is None or t["stage"] == "S0":
                continue
            k = (t["task"], t["family"])
            if k not in best:
                best[k] = t
                continue
            cur = best[k]["primary_mean"]
            if (t["primary_mean"] > cur) if t["task"] == "organ_support" else (t["primary_mean"] < cur):
                best[k] = t
        return {"best_per_family": sorted(best.values(), key=lambda d: (d["task"], d["family"])), "rows": trials, "n_trials": len(trials),
                "n_failed": sum(1 for t in trials if t["status"] != "ok"),
                "note": "diverged/failed configurations are kept (strip plot should clip the axis, not drop them)"}

    def _robustness(self) -> Dict[str, Any]:
        d = self.read_json(P2 + "/robustness_v1.json")
        rows = []
        for sc, v in d["scenarios"].items():
            for m, tasks in v["models"].items():
                for task, met in tasks.items():
                    key = "weighted_median_ae_hours" if task == "icu_stay_time" else ("auprc" if task == "organ_support" else "mae")
                    rows.append({"scenario": sc, "model": m, "task": task, "metric": key, "value": met.get(key),
                                 "delta_vs_unperturbed": (v.get("delta_vs_unperturbed", {}).get(m, {}) or {}).get(task), "n_rows": met.get("n_rows")})
        return {"rows": rows, "scenario_specs": {k: v.get("spec") for k, v in d["scenarios"].items()}, "limitation": d.get("limitation"),
                "cohort": d.get("cohort"), "n_rows": d.get("n_rows"), "n_stays": d.get("n_stays"),
                "statement": "synthetic perturbations on a development subsample; not evidence of real-world prospective robustness"}

    def _explain_global(self) -> Dict[str, Any]:
        d = self.read_json(P2 + "/explainability_v1.json")
        rows = []
        for task, v in d["tasks"].items():
            for rank, it in enumerate(v.get("global_top20", []), 1):
                rows.append({"task": task, "family": v.get("family"), "rank": rank, "feature": it["feature"], "readable": it.get("readable"), "mean_abs_shap": it["mean_abs_shap"]})
        return {"rows": rows, "output_domain": d.get("output_domain"), "note": d.get("note"), "interpretation_wording": d.get("interpretation_wording"),
                "additivity_max_abs_error": d.get("additivity_max_abs_error_overall"), "sample": d.get("sample")}

    def _defect(self) -> Dict[str, Any]:
        lce = self.read_json(DC + "/v2_modes_on_lce_v1.json")
        fm = self.read_json(P4 + "/final_metrics_v3.json")
        audit = self.read_json(DC + "/null_pattern_audit_v1.json") if self.exists(DC + "/null_pattern_audit_v1.json") else None
        rows = []
        for mode, d in lce["modes"].items():
            for task, met in d["all_eligible_rows"].items():
                if met:
                    rows.append({"partition": "LCE (development)", "mode": mode, "task": task, **{k: v for k, v in met.items() if isinstance(v, (int, float))}})
        for m in ("V2_AS_EVALUATED", "V2_AS_SERVED"):
            for task in ("icu_stay_time", "organ_support"):
                d = fm["models"][m][task]
                rows.append({"partition": "sealed V3 test", "mode": m, "task": task, **{k: v for k, v in d.items() if isinstance(v, (int, float))}})
        return {"rows": rows, "fraction_rows_null_baseline_sofa_lce": lce.get("fraction_rows_null_baseline_sofa"),
                "null_pattern_audit": {k: audit[k] for k in ("verdict",) if audit and k in audit} if audit else None,
                "explanation": "baseline_sofa was null exactly when the 24h horizon was unavailable, so its missingness leaked future availability; V2 serving always supplied SOFA(t)",
                "caveat": "the clean (>=24h follow-up) subset is itself outcome-selected for ICU time and support and is not a leak-free comparison there"}

    def _diagnostic(self) -> Dict[str, Any]:
        met = self.read_json(D31 + "/oracle_metrics.json")
        boot = self.read_json(D31 + "/oracle_bootstrap.json")
        comp = self.read_json(D31 + "/oracle_comparison.json")
        ver = self.read_json(D31 + "/oracle_verification_v1.json")
        rows = []
        for k, v in met["metrics"].items():
            b = boot["comparisons"].get(k)
            rows.append({"model": k, "auprc": v["auprc"], "auroc": v["auroc"], "brier": v["brier"], "log_loss": v["log_loss"], "n_rows": v["n_rows"], "n_stays": v["n_stays"],
                         "gap_vs_v3_auprc": b["diff_point"] if b else None, "gap_ci95_lo": b["ci95"][0] if b else None, "gap_ci95_hi": b["ci95"][1] if b else None,
                         "status": STATUS_PRIV if k.startswith("ORACLE") else (STATUS_DEV if k.startswith("REFERENCE") or k.startswith("OBSERVED") else STATUS_PRIV),
                         "deployable": False})
        for k, v in ver["models"].items():
            rows.append({"model": k, "auprc": v["metrics"]["auprc"], "auroc": v["metrics"]["auroc"], "brier": v["metrics"]["brier"], "log_loss": v["metrics"]["log_loss"],
                         "n_rows": v["metrics"]["n_rows"], "n_stays": v["metrics"]["n_stays"], "gap_vs_v3_auprc": v["diff_vs_reference"], "gap_ci95_lo": v["ci95"][0],
                         "gap_ci95_hi": v["ci95"][1], "status": STATUS_PRIV, "deployable": False})
        return {"rows": rows, "gate_a": comp["gate_a"], "gate_a_after_verification": ver["gate_a_after_verification"], "audits": comp["audits"],
                "oracle_c": met["oracle_c"], "status": STATUS_PRIV,
                "statement": "privileged simulator-state predictors are research diagnostics only; hidden states are never serving features and these are not clinical performance bounds"}

    def _candidate_dev(self) -> Dict[str, Any]:
        sel = self.read_json("artifacts/performance_v3_1/phase2/candidate_selection_v1.json")
        hold = self.read_json("artifacts/performance_v3_1/phase2/hold_evaluation_v1.json")
        rows = [{"feature_family": k, "best_oof_auprc": v} for k, v in sel["by_family_best"].items()]
        hrows = [{"model": k, **{kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}} for k, v in hold["metrics"].items()]
        return {"rows": rows, "held_out": hrows, "paired_vs_v3": hold["paired_vs_v3"], "gate_c": hold["gate_c"], "n_trials": sel["n_trials"],
                "tie_group_size": sel["tie_group"], "decision": "V3 retained; no sealed V3.1 test created", "status": STATUS_DEV}

    def _provenance(self) -> Dict[str, Any]:
        sel = self.read_json(P3 + "/selected_models_v3.json")
        ho = self.read_json("artifacts/performance_v3/product_handoff_v1.json")
        files = {k: self.sha(v) for k, v in {
            "selected_models_v3": P3 + "/selected_models_v3.json", "protocol_v1": "configs/performance_v3/scientific_protocol_v1.yaml",
            "protocol_amendment_v1_1": "configs/performance_v3/protocol_amendment_v1_1.yaml", "final_metrics_v3": P4 + "/final_metrics_v3.json",
            "paired_v2_v3_metrics": P4 + "/paired_v2_v3_metrics.json", "evaluation_freeze": "artifacts/performance_v3/governance/v3_final_evaluation_freeze_v1.json",
            "research_registry": RS + "/research_registry_v1.json", "product_handoff_v3_1": "artifacts/performance_v3_1/product_handoff.json"}.items()}
        tasks = {t: {k: e.get(k) for k in ("family", "feature_variant", "model_sha256", "n_estimators", "postprocess")} for t, e in sel["tasks"].items()}
        return {"bundle_id": sel.get("bundle_id"), "status": sel.get("status"), "tasks": tasks, "calibrator": {"method": sel["calibrator"]["method"], "params_sha256": sel["calibrator"]["sha256"]},
                "threshold": sel["threshold"], "file_sha256": files, "rows": [{"task": t, **v} for t, v in tasks.items()],
                "handoff_known_limitations": ho.get("known_limitations"), "defect": "D1 (see /research/defect-d1)"}

    # ------------------------------------------------------------------ summary
    def summary(self) -> Dict[str, Any]:
        fm = self.read_json(P4 + "/final_metrics_v3.json")
        pv = self.read_json(P4 + "/paired_v2_v3_metrics.json")
        nv = self.read_json(P4 + "/naive_baselines_v1.json")
        v3 = fm["models"]["V3"]
        headline = {
            "recovery24_mae": {"value": v3["recovery24"]["mae"], "ci95": v3["recovery24"]["primary_ci"]["0.95"], "unit": "SOFA points", "n_rows": v3["recovery24"]["n_rows"], "n_stays": v3["recovery24"]["n_stays"]},
            "recovery48_mae": {"value": v3["recovery48"]["mae"], "ci95": v3["recovery48"]["primary_ci"]["0.95"], "unit": "SOFA points", "n_rows": v3["recovery48"]["n_rows"], "n_stays": v3["recovery48"]["n_stays"]},
            "icu_weighted_median_ae_hours": {"value": v3["icu_stay_time"]["weighted_median_ae_hours"], "ci95": v3["icu_stay_time"]["primary_ci"]["0.95"], "unit": "hours", "n_rows": v3["icu_stay_time"]["n_rows"], "n_stays": v3["icu_stay_time"]["n_stays"]},
            "support_calibrated_auroc": {"value": v3["organ_support"]["auroc"], "unit": "area", "n_rows": v3["organ_support"]["n_rows"], "n_stays": v3["organ_support"]["n_stays"]},
            "support_calibrated_auprc": {"value": v3["organ_support"]["auprc"], "ci95": v3["organ_support"]["primary_ci"]["0.95"], "unit": "area (prevalence-dependent)", "n_rows": v3["organ_support"]["n_rows"], "n_stays": v3["organ_support"]["n_stays"]},
            "support_calibrated_brier": {"value": v3["organ_support"]["brier"], "unit": "squared probability error"},
            "support_f1_at_frozen_threshold": {"value": v3["organ_support"]["f1"], "unit": "fraction", "threshold": fm["thresholds"]["V3"]},
        }
        return {"api_version": RESEARCH_API_VERSION, "evidence_status": STATUS_FINAL, "partition": "sealed V3 test SYN-V3-T, 1,500 independent subjects, one-time access",
                "bundle": "V3", "headline": headline, "denominators": fm["denominators"], "thresholds": fm["thresholds"], "calibrator": "Platt scaling",
                "ci_method": fm["bootstrap"], "naive": nv["tasks"], "promotion": pv["promotion"], "comparison": self._comparison_matrix(),
                "source_artifacts": [{"path": p, "sha256": self.sha(p)} for p in (P4 + "/final_metrics_v3.json", P4 + "/paired_v2_v3_metrics.json", P4 + "/naive_baselines_v1.json")],
                "interpretation_notes": [
                    "Calibrated AUPRC depends on prevalence; AUROC is prevalence-independent.",
                    "Confidence intervals are population bootstrap intervals over stays, not individual prediction intervals.",
                    "Recovery metrics are SOFA-point changes; ICU metrics are hours; they are not comparable across tasks.",
                    "V2_AS_EVALUATED is compromised by defect D1 and is not reliable serving performance."]}

    def cohort_bundle(self) -> Dict[str, Any]:
        names = ["cohort_summary", "cohort_subtype", "cohort_age_sex", "cohort_duration", "cohort_eligibility", "cohort_outcome_prevalence", "cohort_missingness_feature",
                 "cohort_missingness_bin", "cohort_sofa_distributions", "cohort_sofa_by_cutoff", "cohort_cutoffs_per_stay", "cohort_correlations", "cohort_between_subject",
                 "cohort_variable_distributions"]
        out, missing = {}, []
        for n in names:
            try:
                ds = self.dataset(n)
                out[n] = ds
            except EvidenceMissing as e:
                missing.append({"id": n, "reason": str(e)})
        return {"api_version": RESEARCH_API_VERSION, "evidence_status": STATUS_DEV, "datasets": out, "missing": missing,
                "partitions": "development pool (8,000 subjects) and LCE (1,500); no final-test data"}


def _num(v: str) -> Any:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except ValueError:
        return v
