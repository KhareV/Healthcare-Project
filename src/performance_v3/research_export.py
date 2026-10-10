"""Deterministic research exports for Performance V3 (figures, tables, plot data, registry, index).

Read-only on every existing artifact. Each exported file gets a sidecar JSON recording
the exact source artifacts (path + sha256), the model bundle id and the git commit.
Anything not backed by a real artifact is skipped and logged, never fabricated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXPORT_VERSION = "performance_v3_research_export_v1"
BUNDLE_ID = "performance_v3_v3"
OUT_REL = "artifacts/performance_v3/research"
INDEX_REL = "docs/performance_v3/FIGURE_AND_TABLE_INDEX.md"
REGISTRY_REL = OUT_REL + "/research_registry_v1.json"
MODEL_COLORS = {"V3": "#0072B2", "CONTROL": "#009E73", "V2_AS_SERVED": "#D55E00", "V2_AS_EVALUATED": "#E69F00"}
MODELS = ("V3", "CONTROL", "V2_AS_SERVED", "V2_AS_EVALUATED")
TASKS = ("recovery24", "recovery48", "icu_stay_time", "organ_support")
RQ = {"RQ1": "learning curve / data size", "RQ2": "feature families", "RQ3": "model families and objectives",
      "RQ4": "support calibration", "RQ5": "sealed-test paired V2 vs V3"}

DC = "artifacts/performance_v3/data_characterization/"
P2 = "artifacts/performance_v3/phase2/"
P3 = "artifacts/performance_v3/phase3/"
P4 = "artifacts/performance_v3/phase4/"

# Historical context values, copied verbatim from the cited documents (labelled historical).
HISTORICAL = {
    "V1_final_test": {
        "source": "docs/vedant/FINAL_TEST_EVALUATION_REVIEW.md",
        "recovery24_mae": (1.1211, 1.0265, 1.2267), "recovery48_mae": (1.5424, 1.3611, 1.7310),
        "icu_median_ae_hours": (9.4797, 8.5929, 10.6097), "support_calibrated_auprc": (0.6337, 0.5609, 0.7038),
    },
    "V2_final_test": {
        "source": "docs/performance_v2/FINAL_V2_TEST_EVALUATION.md",
        "recovery24_mae": (1.0744, 1.0303, 1.1199), "recovery48_mae": (1.3379, 1.2585, 1.4250),
        "icu_median_ae_hours": (5.2060, 5.0602, 5.3947), "support_calibrated_auprc": (0.7042, 0.6612, 0.7437),
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, default=str) + "\n").encode("utf-8")


class Exporter:
    def __init__(self, root: Path, out_rel: str = OUT_REL):
        self.root = Path(root)
        self.out = self.root / out_rel
        self.commit = git_commit(self.root)
        self.entries: List[Dict[str, object]] = []
        self.skipped: List[Dict[str, str]] = []
        for sub in ("figures", "tables", "plot_data"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)

    # ---- sources -------------------------------------------------------
    def load_json(self, rel: str):
        return json.loads((self.root / rel).read_text(encoding="utf-8"))

    def source_record(self, rels: Iterable[str]) -> List[Dict[str, str]]:
        out = []
        for rel in rels:
            path = self.root / rel
            if not path.exists():
                raise FileNotFoundError(rel)
            out.append({"path": rel, "sha256": sha256_file(path)})
        return out

    def _sidecar(self, file_path: Path, kind: str, fid: str, title: str, sources: Sequence[str], notes: str, extra: Optional[Mapping] = None) -> Dict[str, object]:
        rel = str(file_path.relative_to(self.root))
        meta = {
            "export_version": EXPORT_VERSION, "id": fid, "kind": kind, "title": title, "file": rel,
            "file_sha256": sha256_file(file_path), "sources": self.source_record(sources),
            "model_bundle_id": BUNDLE_ID, "code_commit": self.commit, "notes": notes,
        }
        if extra:
            meta.update(extra)
        side = file_path.with_name(file_path.name + ".meta.json")
        side.write_bytes(canonical(meta))
        meta["sidecar"] = str(side.relative_to(self.root))
        self.entries.append(meta)
        return meta

    # ---- writers -------------------------------------------------------
    def save_figure(self, fig, fid: str, title: str, sources: Sequence[str], notes: str = "", formats: Sequence[str] = ("png",)) -> None:
        for fmt in formats:
            path = self.out / "figures" / f"{fid}.{fmt}"
            kwargs = {"format": fmt, "bbox_inches": "tight"}
            if fmt == "png":
                kwargs.update(dpi=150, metadata={"Software": None})
            elif fmt == "svg":
                kwargs.update(metadata={"Date": None, "Creator": None})
            elif fmt == "pdf":
                kwargs.update(metadata={"CreationDate": None, "Creator": None, "Producer": None})
            matplotlib.rcParams["svg.hashsalt"] = fid
            fig.savefig(path, **kwargs)
            self._sidecar(path, "figure", fid, title, sources, notes, {"format": fmt})
        plt.close(fig)

    def save_table(self, tid: str, title: str, columns: Sequence[str], rows: Sequence[Sequence[object]], sources: Sequence[str],
                   notes: str = "", latex: bool = False, directory: str = "tables") -> None:
        path = self.out / directory / f"{tid}.csv"
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(v) for v in row])
        path.write_text(buf.getvalue(), encoding="utf-8")
        self._sidecar(path, "table" if directory == "tables" else "plot_data", tid, title, sources, notes, {"rows": len(rows)})
        if latex:
            tex = self.out / directory / f"{tid}.tex"
            tex.write_text(_latex(columns, rows, title), encoding="utf-8")
            self._sidecar(tex, "table", tid, title, sources, notes, {"rows": len(rows), "format": "tex"})

    def save_plot_data(self, pid: str, title: str, columns: Sequence[str], rows: Sequence[Sequence[object]], sources: Sequence[str], notes: str = "") -> None:
        self.save_table(pid, title, columns, rows, sources, notes, directory="plot_data")

    def skip(self, fid: str, reason: str) -> None:
        self.skipped.append({"id": fid, "reason": reason})


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if (v != v) else f"{float(v):.6g}"
    return v


def _latex(columns, rows, caption) -> str:
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\hline", " & ".join(esc(c) for c in columns) + " \\\\", "\\hline"]
    for row in rows:
        lines.append(" & ".join(esc(_fmt(v)) for v in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "% " + esc(caption)]
    return "\n".join(lines) + "\n"


# ============================================================================
# Figure builders: each takes the Exporter and writes figures/tables (may raise FileNotFoundError -> skipped)
# ============================================================================

def _grouped_bar(ax, labels, series: Mapping[str, Sequence[float]], colors=None, errs=None):
    n = len(series)
    x = np.arange(len(labels))
    w = 0.8 / max(n, 1)
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name, color=(colors or {}).get(name),
               yerr=None if errs is None else errs.get(name), capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")


def fig_population(ex: Exporter):
    src = [DC + "subtype_distribution.json"]
    d = ex.load_json(src[0])["cohorts"]
    cohorts = ["DEV_POOL", "LCE"]
    subtypes = sorted(d["DEV_POOL"])
    rows = [(c, s, d[c][s]["stays"], d[c][s]["fraction"]) for c in cohorts for s in subtypes]
    ex.save_plot_data("pd01_subtype_composition", "Cardiac subtype composition", ("cohort", "subtype", "stays", "fraction"), rows, src)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _grouped_bar(ax, [s.replace("SYNTHETIC_", "") for s in subtypes], {c: [d[c][s]["fraction"] for s in subtypes] for c in cohorts})
    ax.set_ylabel("fraction of stays"); ax.set_title("Cardiac subtype composition (development data)"); ax.legend()
    ex.save_figure(fig, "f01_subtype_composition", "Cardiac subtype composition", src)


def fig_age_duration(ex: Exporter):
    src = [DC + "age_sex_composition.json", DC + "episode_duration_summary.json"]
    age = ex.load_json(src[0])["cohorts"]
    dur = ex.load_json(src[1])["cohorts"]
    pct = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, name, data, key in ((axes[0], "Age (years)", age, "age_years"), (axes[1], "Episode duration (hours)", dur, "all")):
        for cohort in ("DEV_POOL", "LCE"):
            stats = data[cohort][key]
            ys = [stats[f"p{p}"] for p in pct]
            ax.plot(pct, ys, marker="o", label=cohort)
            rows += [(name, cohort, p, y) for p, y in zip(pct, ys)]
        ax.set_xlabel("percentile"); ax.set_ylabel(name); ax.set_title(name + " quantiles"); ax.legend()
    ex.save_plot_data("pd02_age_duration_quantiles", "Age and episode-duration quantiles", ("variable", "cohort", "percentile", "value"), rows, src)
    ex.save_figure(fig, "f02_age_duration", "Age and episode-duration quantile curves", src, "Only stored percentiles are available (no raw histogram).")


def fig_sofa(ex: Exporter):
    src = [DC + "sofa_distributions.json"]
    d = ex.load_json(src[0])["cohorts"]["DEV_POOL"]
    idx = sorted(int(k) for k in d["by_cutoff_index"])
    rows = [(i, d["by_cutoff_index"][str(i)]["n"], d["by_cutoff_index"][str(i)]["mean"], d["by_cutoff_index"][str(i)]["p25"],
             d["by_cutoff_index"][str(i)]["p50"], d["by_cutoff_index"][str(i)]["p75"]) for i in idx]
    ex.save_plot_data("pd03_sofa_by_cutoff", "Baseline SOFA by cutoff index (recovery24-eligible rows)", ("cutoff_index", "n", "mean", "p25", "p50", "p75"), rows, src)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.fill_between(idx, [r[3] for r in rows], [r[5] for r in rows], alpha=0.25, label="IQR")
    ax.plot(idx, [r[2] for r in rows], marker="o", label="mean"); ax.plot(idx, [r[4] for r in rows], ls="--", label="median")
    ax.set_xlabel("cutoff index k"); ax.set_ylabel("SOFA(t)"); ax.set_title("SOFA at the cutoff (development pool)"); ax.legend()
    ex.save_figure(fig, "f03_sofa_by_cutoff", "SOFA by cutoff index", src, "Component scores are not persisted in the pre-split package (stated in sofa_distributions.json).")
    ex.skip("f03b_sofa_components", "component-level SOFA distributions unavailable in characterization artifacts")


def fig_observation(ex: Exporter):
    src = [DC + "missingness_by_bin.json"]
    d = ex.load_json(src[0])["cohorts"]["DEV_POOL"]
    bins = sorted(d, key=int)
    chans = sorted(d[bins[0]]["missing_fraction_by_clinical_channel"])
    obs = np.array([[1 - d[b]["missing_fraction_by_clinical_channel"][c] for b in bins] for c in chans])
    rows = [(c, int(b), obs[i, j]) for i, c in enumerate(chans) for j, b in enumerate(bins)]
    ex.save_plot_data("pd04_observed_fraction_by_bin", "Observed fraction by channel and bin (non-padded rows)", ("channel", "bin", "observed_fraction"), rows, src)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [3, 1.3]})
    im = axes[0].imshow(obs, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_yticks(range(len(chans))); axes[0].set_yticklabels(chans, fontsize=7)
    axes[0].set_xticks(range(len(bins))); axes[0].set_xlabel("bin (0 = oldest, 7 = newest)"); axes[0].set_title("Observed fraction")
    fig.colorbar(im, ax=axes[0])
    axes[1].plot([d[b]["padded_fraction"] for b in bins], range(len(bins)), marker="o"); axes[1].invert_yaxis()
    axes[1].set_xlabel("padded fraction"); axes[1].set_ylabel("bin"); axes[1].set_title("Structural padding")
    ex.save_figure(fig, "f04_observation_availability", "Temporal observation availability", src)


def fig_missingness(ex: Exporter):
    src = [DC + "missingness_by_feature.json"]
    d = ex.load_json(src[0])["cohorts"]["DEV_POOL"]
    items = sorted(((k, v["missing_fraction_nonpadded_bins"], v["channel_type"]) for k, v in d.items()), key=lambda t: -t[1])
    ex.save_plot_data("pd05_feature_missingness", "Feature missingness over non-padded bins", ("feature", "missing_fraction", "channel_type"), items, src)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([i[0] for i in items], [i[1] for i in items], color=["#0072B2" if i[2] == "clinical_observation" else "#999999" for i in items])
    ax.invert_yaxis(); ax.set_xlabel("missing fraction"); ax.set_title("Feature missingness (non-padded bins)")
    ex.save_figure(fig, "f05_feature_missingness", "Feature missingness", src, "grey = support-state channels (always known)")


def fig_eligibility(ex: Exporter):
    src = [DC + "target_eligibility.json"]
    d = ex.load_json(src[0])["cohorts"]
    keys = ["recovery24", "recovery48", "icu_time", "organ_support"]
    rows = [(c, k, d[c][k]["eligible_rows"], d[c][k]["unique_stays"]) for c in ("DEV_POOL", "LCE") for k in keys]
    ex.save_table("t06_task_eligibility", "Task eligibility denominators", ("cohort", "task", "eligible_rows", "unique_stays"), rows, src, latex=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    _grouped_bar(ax, keys, {c: [d[c][k]["eligible_rows"] for k in keys] for c in ("DEV_POOL", "LCE")})
    ax.set_yscale("log"); ax.set_ylabel("eligible rows (log)"); ax.set_title("Task eligibility"); ax.legend()
    ex.save_figure(fig, "f06_task_eligibility", "Task eligibility", src)


def fig_learning_curve(ex: Exporter):
    src = [DC + "learning_curve_table.csv", DC + "learning_curve_metrics.json"]
    with (ex.root / src[0]).open() as handle:
        rows = list(csv.DictReader(handle))
    ex.save_plot_data("pd07_learning_curve", "Corrected learning curve (frozen V2 recipe, LCE)", ("task", "metric", "size_subjects", "mean", "sd", "ci_lower_repeat0", "ci_upper_repeat0"),
                      [(r["task"], r["metric"], int(r["size_subjects"]), float(r["mean"]), float(r["sd"]), float(r["ci_lower_repeat0"]), float(r["ci_upper_repeat0"])) for r in rows], src)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, task in zip(axes, TASKS):
        sel = [r for r in rows if r["task"] == task]
        x = [int(r["size_subjects"]) for r in sel]; y = [float(r["mean"]) for r in sel]
        ax.errorbar(x, y, yerr=[float(r["sd"]) for r in sel], marker="o", capsize=3, label="mean ± SD over repeats")
        ax.fill_between(x, [float(r["ci_lower_repeat0"]) for r in sel], [float(r["ci_upper_repeat0"]) for r in sel], alpha=0.2, label="95% stay-cluster CI (repeat 0)")
        ax.set_xscale("log"); ax.set_xlabel("training subjects"); ax.set_title(f"{task}\n({sel[0]['metric']})")
    axes[0].legend(fontsize=7)
    fig.suptitle("Corrected learning curves on LCE (development evidence, not final validation)", y=1.03)
    ex.save_figure(fig, "f07_learning_curve", "Learning curves", src, "Frozen V2 recipe with repaired current SOFA; development evidence only.", formats=("png", "svg"))


def _ledger(ex: Exporter):
    return [json.loads(l) for l in (ex.root / (P2 + "trial_ledger_v1.jsonl")).read_text().splitlines() if l.strip()]


def fig_leaderboard(ex: Exporter):
    src = [P2 + "trial_ledger_v1.jsonl"]
    recs = [r for r in _ledger(ex) if r["status"] == "ok" and r["stage"] in ("S1", "S3", "S4")]
    lower = {"recovery24": True, "recovery48": True, "icu_stay_time": True, "organ_support": False}
    fams = ["xgboost", "lightgbm", "catboost", "ridge", "logistic", "gru"]
    table, strip_rows, clipped = [], [], 0
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, task in zip(axes, TASKS):
        pool = [r for r in recs if r["task"] == task]
        vals = np.array([r["primary_mean"] for r in pool])
        sane = vals[vals < 5 * np.median(vals)]  # diverged trials (e.g. 1e14) must not set the display range
        lo, hi = float(sane.min()), float(sane.max())
        for fi, fam in enumerate(fams):
            sub = [r for r in pool if r["family"] == fam]
            if not sub:
                continue
            best = min(sub, key=lambda r: r["primary_mean"]) if lower[task] else max(sub, key=lambda r: r["primary_mean"])
            table.append((task, fam, len(sub), best["primary_mean"], best["primary_sd"], best["variant"], best["stage"]))
            ys = np.array([r["primary_mean"] for r in sub])
            clipped += int(np.sum((ys > hi) | (ys < lo)))
            strip_rows += [(task, fam, r["stage"], r["variant"], r["primary_mean"]) for r in sub]
            jitter = (np.arange(len(ys)) % 7 - 3) * 0.04
            ax.scatter(fi + jitter, np.clip(ys, lo - 0.02 * abs(lo), hi), s=8, alpha=0.5)
            ax.scatter([fi], [best["primary_mean"]], marker="D", color="k", s=28, zorder=5)
        ax.set_ylim(lo - 0.03 * abs(lo), hi + 0.03 * abs(hi)); ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{task} ({pool[0]['primary_metric']})")
    fig.suptitle("Model-family leaderboard (all S1/S3/S4 trials; diamonds = best per family; display range = non-diverged trials; diverged points drawn at the upper edge)", y=1.03)
    ex.save_plot_data("pd08_leaderboard_all_trials", "All S1/S3/S4 trials", ("task", "family", "stage", "variant", "primary_mean"), strip_rows, src)
    ex.save_table("t08_leaderboard_best_per_family", "Best trial per family and task", ("task", "family", "n_trials", "primary_mean", "primary_sd", "variant", "stage"), table, src, latex=True)
    ex.save_figure(fig, "f08_model_leaderboard", "Model-family leaderboard", src, f"{clipped} trial points outside the display range (e.g. diverged configs) drawn at the edge; values kept in plot data.")


def fig_ablation(ex: Exporter):
    src = [P2 + "trial_ledger_v1.jsonl"]
    recs = [r for r in _ledger(ex) if r["status"] == "ok" and r["stage"] == "S2"]
    variants = ["V3_CTRL", "V3_B", "V3_BF", "V3_BC", "V3_BD", "V3_BE", "V3_BFS", "V3_FULL"]
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, task in zip(axes, TASKS):
        means, sds = [], []
        for v in variants:
            sub = [r for r in recs if r["task"] == task and r["variant"] == v]
            m = float(np.mean([r["primary_mean"] for r in sub])); s = float(np.mean([r["primary_sd"] for r in sub]))
            means.append(m); sds.append(s); rows.append((task, v, len(sub), m, s))
        ax.errorbar(range(len(variants)), means, yerr=sds, fmt="o", capsize=3)
        ax.axhline(means[0], ls="--", color="grey", lw=0.8)
        ax.set_xticks(range(len(variants))); ax.set_xticklabels(variants, rotation=60, ha="right", fontsize=7); ax.set_title(task)
    fig.suptitle("Feature-variant ablation (S2; mean over 2 seeds, bars = mean fold SD; dashed = V3_CTRL)", y=1.03)
    ex.save_table("t09_feature_ablation", "Feature-variant ablation (S2)", ("task", "variant", "n_seeds", "primary_mean", "mean_fold_sd"), rows, src, latex=True)
    ex.save_figure(fig, "f09_feature_ablation", "Feature-variant ablation", src)


def fig_control_vs_selected(ex: Exporter):
    src = [P2 + "candidate_selection_v1.json"]
    d = ex.load_json(src[0])["tasks"]
    rows, fig = [], None
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, task in zip(axes, TASKS):
        c, w = d[task]["v2_recipe_control"], d[task]["winner"]
        rows.append((task, c["primary_mean"], c["primary_sd"], w["primary_mean"], w["primary_sd"], w["family"], w["variant"], d[task]["beats_v2_recipe_beyond_tolerance"], d[task]["tolerance"]))
        ax.bar([0, 1], [c["primary_mean"], w["primary_mean"]], yerr=[c["primary_sd"], w["primary_sd"]], capsize=3, color=["#009E73", "#0072B2"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["V2 recipe\n(repaired)", f"S4 winner\n{w['family']}/{w['variant']}"], fontsize=7)
        lo = min(c["primary_mean"], w["primary_mean"]); ax.set_ylim(lo * 0.9, max(c["primary_mean"], w["primary_mean"]) * 1.03); ax.set_title(task)
    fig.suptitle("5-fold cross-fitted primary metric: V2-recipe control vs selected V3 candidate", y=1.03)
    ex.save_table("t10_control_vs_selected", "V2-recipe control vs S4 winner", ("task", "control_mean", "control_sd", "winner_mean", "winner_sd", "family", "variant", "beats_beyond_tolerance", "tolerance"), rows, src, latex=True)
    ex.save_figure(fig, "f10_control_vs_selected", "Control vs selected", src, "Axis truncated for readability; bars are not zero-based.")


def _cal(ex: Exporter):
    return ex.load_json(P4 + "calibration_evidence_v1.json")["models"]


def fig_roc_pr(ex: Exporter):
    src = [P4 + "calibration_evidence_v1.json"]
    m = _cal(ex)
    for kind, fid, title in (("roc", "f11_support_roc", "Support ROC (calibrated scores, sealed V3 test)"), ("pr", "f12_support_pr", "Support precision-recall (calibrated scores, sealed V3 test)")):
        fig, ax = plt.subplots(figsize=(5, 4.4))
        rows = []
        for name in MODELS:
            c = m[name][f"{kind}_calibrated"]
            auc = m[name]["calibrated"]["auroc" if kind == "roc" else "auprc"]
            ax.plot(c["x"], c["y"], color=MODEL_COLORS[name], ls="--" if name == "V2_AS_EVALUATED" else "-", label=f"{name} ({'AUROC' if kind == 'roc' else 'AUPRC'} {auc:.3f})")
            rows += [(name, x, y) for x, y in zip(c["x"], c["y"])]
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        else:
            ax.axhline(m["V3"]["prevalence_stay_balanced"], color="grey", lw=0.8, label="prevalence"); ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
        ex.save_plot_data("pd" + fid[1:], title, ("model", "x", "y"), rows, src, "stay-balanced curves from frozen test predictions (TEST)")
        ex.save_figure(fig, fid, title, src, "TEST partition (one-time governed access).", formats=("png", "svg") if kind == "pr" else ("png",))


def fig_reliability(ex: Exporter):
    src = [P4 + "calibration_evidence_v1.json"]
    m = _cal(ex)
    fig, ax = plt.subplots(figsize=(5, 4.6)); rows = []
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    for name in MODELS:
        b = m[name]["reliability_calibrated"]
        ax.plot([x["mean_predicted"] for x in b], [x["observed_rate"] for x in b], marker="o", color=MODEL_COLORS[name],
                label=f"{name} (ECE {m[name]['calibrated']['ece_10_equal_mass']:.3f})")
        rows += [(name, x["bin"], x["mean_predicted"], x["observed_rate"], x["n_rows"], x["weight"]) for x in b]
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed rate (stay-balanced)"); ax.set_title("Reliability (10 equal-mass bins, TEST)", fontsize=9); ax.legend(fontsize=7)
    ex.save_plot_data("pd13_reliability", "Reliability bins", ("model", "bin", "mean_predicted", "observed_rate", "n_rows", "weight"), rows, src)
    ex.save_figure(fig, "f13_reliability", "Reliability diagram", src, "TEST partition.")


def fig_confusion(ex: Exporter):
    src = [P4 + "calibration_evidence_v1.json"]
    m = _cal(ex)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4)); rows = []
    for ax, name in zip(axes, MODELS):
        c = m[name]["confusion_weighted"]
        mat = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
        norm = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({mat[i, j]:.0f})", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred -", "pred +"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true -", "true +"])
        ax.set_title(f"{name}\nthr={m[name]['frozen_threshold']:.3f}", fontsize=8)
        rows.append((name, m[name]["frozen_threshold"], c["TN"], c["FP"], c["FN"], c["TP"], *[m[name]["confusion_rows"][k] for k in ("TN", "FP", "FN", "TP")]))
    ex.save_table("t14_confusion_at_frozen_thresholds", "Confusion matrices at frozen thresholds (TEST)",
                  ("model", "threshold", "w_TN", "w_FP", "w_FN", "w_TP", "rows_TN", "rows_FP", "rows_FN", "rows_TP"), rows, src, latex=True)
    ex.save_figure(fig, "f14_confusion", "Confusion matrices", src, "Row-normalised stay-balanced; counts in parentheses are stay-weighted.")


def fig_threshold_sensitivity(ex: Exporter):
    from performance_v3 import calibration as C
    from performance_v3 import task_metrics as tm
    src = [P4 + "predictions/test_predictions_v1.jsonl", P2 + "support_calibration_v1.json", "artifacts/performance_v3/cache/support_oof_v1.npz"]
    rep = ex.load_json(src[1])
    z = np.load(ex.root / src[2], allow_pickle=True)
    cal_dev = C.apply_calibrator(rep["frozen_calibrator"], z["oof"])
    w_dev = np.asarray(z["w"], dtype=float); y_dev = np.asarray(z["y"])
    thr = np.linspace(0.05, 0.8, 76)

    def curve(p, y, w):
        out = []
        for t in thr:
            pred = p >= t
            tp = w[(y == 1) & pred].sum(); fp = w[(y == 0) & pred].sum(); fn = w[(y == 1) & ~pred].sum()
            prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
            out.append((prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0))
        return np.array(out)

    dev = curve(cal_dev, y_dev, w_dev)
    rows_test = [json.loads(l) for l in (ex.root / src[0]).read_text().splitlines() if l.strip()]
    sel = [r for r in rows_test if r["labels"]["organ_support_eligible"] and r["labels"]["organ_support_label"] is not None]
    yt = np.array([r["labels"]["organ_support_label"] for r in sel]); pt = np.array([r["models"]["V3"]["support_calibrated_probability"] for r in sel])
    wt = tm.stay_weights([r["stay_id"] for r in sel])
    test = curve(pt, yt, wt)
    frozen = rep["threshold"]["threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, c, label in ((axes[0], dev, "DEVELOPMENT cross-fitted OOF (threshold chosen here)"), (axes[1], test, "TEST (V3, frozen threshold shown; not used for choice)")):
        for i, nm in enumerate(("precision", "recall", "F1")):
            ax.plot(thr, c[:, i], label=nm)
        ax.axvline(frozen, color="k", ls=":", label=f"frozen {frozen:.3f}"); ax.set_xlabel("threshold"); ax.set_title(label, fontsize=8); ax.legend(fontsize=7)
    rows = [("dev_oof", float(t), *dev[i]) for i, t in enumerate(thr)] + [("test", float(t), *test[i]) for i, t in enumerate(thr)]
    ex.save_plot_data("pd15_threshold_sensitivity", "Threshold sensitivity", ("partition", "threshold", "precision", "recall", "f1"), rows, src)
    ex.save_figure(fig, "f15_threshold_sensitivity", "Threshold sensitivity", src, "Left: development OOF only (used for the frozen threshold). Right: TEST, labelled.")


def fig_residuals(ex: Exporter):
    src = [P4 + "predictions/test_predictions_v1.jsonl"]
    rows_all = [json.loads(l) for l in (ex.root / src[0]).read_text().splitlines() if l.strip()]
    spec = (("recovery24", "recovery24_eligible", "delta_sofa_24", "recovery24_delta", 1.0, "SOFA units"),
            ("recovery48", "recovery48_eligible", "delta_sofa_48", "recovery48_delta", 1.0, "SOFA units"),
            ("icu_stay_time", "icu_time_eligible", "icu_time_log1p", "icu_remaining_hours", None, "hours"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5)); data_rows = []
    for ax, (task, el, lab, key, _, unit) in zip(axes, spec):
        sel = [r for r in rows_all if r["labels"][el] and r["labels"][lab] is not None]
        y = np.array([r["labels"][lab] for r in sel]); y = np.expm1(y) if task == "icu_stay_time" else y
        lim = 12 if task == "recovery24" or task == "recovery48" else 80
        lim = 5 if task != "icu_stay_time" else 60
        bins = np.linspace(-lim, lim, 61)
        for name in ("V3", "V2_AS_SERVED", "V2_AS_EVALUATED"):
            res = np.array([r["models"][name][key] for r in sel]) - y
            h, edges = np.histogram(np.clip(res, -lim, lim), bins=bins)
            ax.step(edges[:-1], h / len(res), where="post", color=MODEL_COLORS[name], label=f"{name} (median {np.median(res):+.2f})")
            data_rows += [(task, name, float(edges[i]), float(edges[i + 1]), int(h[i]), len(res)) for i in range(len(h))]
        ax.set_title(f"{task} residual (pred - true, {unit}; clipped)", fontsize=8); ax.legend(fontsize=7)
    ex.save_plot_data("pd16_residual_histograms", "Residual histograms (TEST)", ("task", "model", "bin_lo", "bin_hi", "count", "n"), data_rows, src)
    ex.save_figure(fig, "f16_residuals", "Residual distributions", src, "TEST partition; tails clipped for display.")


def fig_error_slices(ex: Exporter):
    src = [P4 + "error_analysis_v1.json"]
    tasks = ex.load_json(src[0])["tasks"]
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(17, 6))
    for ax, task in zip(axes, TASKS):
        labels = []
        for dim, levels in tasks[task].items():
            for lvl, rec in levels.items():
                v = rec["models"]
                if v.get("V3") is None:
                    continue
                labels.append((f"{dim}:{lvl}", v, rec.get("n_rows"), rec.get("n_stays")))
                rows.append((task, dim, lvl, rec.get("n_rows"), rec.get("n_stays"), *[v.get(m) for m in MODELS]))
        y = np.arange(len(labels))
        for i, name in enumerate(("V3", "V2_AS_SERVED")):
            ax.barh(y + (i - 0.5) * 0.4, [l[1][name] for l in labels], 0.4, color=MODEL_COLORS[name], label=name)
        ax.set_yticks(y); ax.set_yticklabels([f"{l[0]} (n={l[2]})" for l in labels], fontsize=6); ax.invert_yaxis(); ax.set_title(task); ax.legend(fontsize=7)
    fig.suptitle("Prespecified slices on the sealed V3 test (primary metric per task; support = AUPRC; lower is better except support)", y=1.0)
    ex.save_table("t17_error_slices", "Prespecified slice metrics (TEST)", ("task", "dimension", "level", "n_rows", "n_stays", *MODELS), rows, src, latex=False)
    ex.save_figure(fig, "f17_error_slices", "Error slices", src, "Cut-points from the development pool only; undefined slices omitted.")


def fig_robustness(ex: Exporter):
    src = [P2 + "robustness_v1.json"]
    d = ex.load_json(src[0])
    names = list(d["scenarios"])
    rows = [(s, m, t, d["scenarios"][s]["delta_vs_unperturbed"][m][t]) for s in names for m in ("V3", "CONTROL", "V2_AS_SERVED") for t in TASKS]
    ex.save_table("t18_robustness_deltas", "Robustness: change vs unperturbed (LCE subsample)", ("scenario", "model", "task", "delta_primary_metric"), rows, src, latex=True,
                  notes="Positive = worse for recovery/ICU (error); negative = worse for support (AUPRC).")
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, task in zip(axes, TASKS):
        _grouped_bar(ax, names, {m: [d["scenarios"][s]["delta_vs_unperturbed"][m][task] for s in names] for m in ("V3", "CONTROL", "V2_AS_SERVED")},
                     colors=MODEL_COLORS)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(task); ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Robustness (development only, LCE {d['n_stays']}-stay subsample): change in primary metric vs unperturbed", y=1.03)
    ex.save_figure(fig, "f18_robustness", "Robustness", src, d.get("limitation", ""))


def fig_shap(ex: Exporter):
    src = [P2 + "explainability_v1.json"]
    d = ex.load_json(src[0])
    fig, axes = plt.subplots(1, 4, figsize=(20, 6)); rows = []
    for ax, task in zip(axes, TASKS):
        top = d["tasks"][task]["global_top20"]
        ax.barh(range(len(top)), [t["mean_abs_shap"] for t in top], color="#0072B2")
        ax.set_yticks(range(len(top))); ax.set_yticklabels([t["readable"][:48] for t in top], fontsize=6); ax.invert_yaxis()
        ax.set_title(f"{task} ({d['tasks'][task]['family']})", fontsize=9)
        rows += [(task, i + 1, t["feature"], t["readable"], t["mean_abs_shap"]) for i, t in enumerate(top)]
    fig.suptitle("Global mean |SHAP| on the RAW margin (pre-calibration; features 'contributed to the prediction', not causal)", y=1.0)
    ex.save_table("t19_shap_global_top20", "Global TreeSHAP top-20 per task (raw margin)", ("task", "rank", "feature", "readable", "mean_abs_shap"), rows, src)
    ex.save_figure(fig, "f19_shap_global", "SHAP global importance", src, d.get("note", ""))


def fig_paired(ex: Exporter):
    src = [P4 + "paired_v2_v3_metrics.json", P4 + "paired_bootstrap_v1.json", P4 + "final_metrics_v3.json", P4 + "naive_baselines_v1.json",
           "docs/performance_v2/FINAL_V2_TEST_EVALUATION.md", "docs/vedant/FINAL_TEST_EVALUATION_REVIEW.md"]
    paired = ex.load_json(src[0])["point_differences"]
    prim = [("recovery24|mae|V3_vs_V2_AS_SERVED", "recovery24 MAE"), ("recovery48|mae|V3_vs_V2_AS_SERVED", "recovery48 MAE"),
            ("icu_stay_time|weighted_median_ae_hours|V3_vs_V2_AS_SERVED", "ICU wMedAE (h)"), ("organ_support|auprc|V3_vs_V2_AS_SERVED", "support cal. AUPRC"),
            ("organ_support|brier|V3_vs_V2_AS_SERVED", "support Brier"),
            ("recovery24|mae|V3_vs_CONTROL", "rec24 vs CONTROL"), ("recovery48|mae|V3_vs_CONTROL", "rec48 vs CONTROL"),
            ("organ_support|auprc|V3_vs_CONTROL", "support AUPRC vs CONTROL"),
            ("icu_stay_time|weighted_median_ae_hours|V3_vs_V2_AS_EVALUATED", "ICU vs V2_AS_EVALUATED"),
            ("organ_support|auprc|V3_vs_V2_AS_EVALUATED", "support AUPRC vs V2_AS_EVALUATED")]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label) in enumerate(prim):
        p = paired[key]
        c95, c98 = p["ci"]["0.95"], p["ci"]["0.9875"]
        ax.plot(c98, [i, i], color="#999999", lw=3); ax.plot(c95, [i, i], color="#0072B2", lw=3); ax.plot([p["diff_point"]], [i], "kD")
        rows.append((key, p["point_a"], p["point_b"], p["diff_point"], c95[0], c95[1], c98[0], c98[1], p["n_rows"], p["n_stays"]))
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(prim))); ax.set_yticklabels([p[1] for p in prim], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("paired difference (V3 - comparator); error metrics: negative favours V3, AUPRC: positive favours V3")
    ax.set_title("Paired stay-cluster bootstrap differences (blue 95%, grey 98.75%, 2000 resamples)", fontsize=9)
    ex.save_table("t20_paired_differences", "Paired V3 vs comparator differences (TEST)", ("comparison", "point_comparator", "point_v3", "diff", "ci95_lo", "ci95_hi", "ci9875_lo", "ci9875_hi", "n_rows", "n_stays"), rows, src[:2], latex=True)
    ex.save_figure(fig, "f20_paired_differences", "Paired differences", src[:2], formats=("png", "svg", "pdf"))
    # historical context table (values copied from the cited docs; labelled historical, different cohorts)
    fm = ex.load_json(src[2])["models"]; naive = ex.load_json(src[3])["tasks"]
    metric_rows = []
    for key, label, v3, naive_v in (
        ("recovery24_mae", "recovery24 MAE", fm["V3"]["recovery24"]["mae"], naive["recovery24"]["mae"]),
        ("recovery48_mae", "recovery48 MAE", fm["V3"]["recovery48"]["mae"], naive["recovery48"]["mae"]),
        ("icu_median_ae_hours", "ICU weighted median AE (h)", fm["V3"]["icu_stay_time"]["weighted_median_ae_hours"], naive["icu_stay_time"]["weighted_median_ae_hours"]),
        ("support_calibrated_auprc", "support calibrated AUPRC", fm["V3"]["organ_support"]["auprc"], naive["organ_support"]["auprc"])):
        v1, v2 = HISTORICAL["V1_final_test"][key], HISTORICAL["V2_final_test"][key]
        metric_rows.append((label, f"{v1[0]} [{v1[1]}, {v1[2]}]", f"{v2[0]} [{v2[1]}, {v2[2]}]", round(v3, 4), round(naive_v, 4)))
    ex.save_table("t20b_historical_context", "Historical V1/V2 final-test values (labelled historical; different cohorts and, for V2, defect D1 applies) vs V3 on SYN-V3-T",
                  ("metric", "V1_final_test_historical", "V2_final_test_historical_as_published", "V3_sealed_test", "naive_floor_V3_test"), metric_rows, src, latex=True,
                  notes="V1/V2 values copied verbatim from the cited documents; not comparable row-for-row (different cohorts). V2 published values use the V2 offline harness (defect D1).")


def fig_defect_d1(ex: Exporter):
    src = [DC + "v2_modes_on_lce_v1.json", P4 + "final_metrics_v3.json"]
    lce = ex.load_json(src[0])["modes"]; fm = ex.load_json(src[1])["models"]
    panels = [("ICU weighted median AE (hours, lower better)", "icu_stay_time", "weighted_median_ae_hours"),
              ("Support calibrated AUPRC (higher better)", "organ_support", "cal_auprc")]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (title, task, key) in zip(axes, panels):
        labels = ["LCE (dev)", "SYN-V3-T (sealed test)"]
        test_key = "weighted_median_ae_hours" if task == "icu_stay_time" else "auprc"
        series = {
            "V2_AS_EVALUATED": [lce["V2_AS_EVALUATED"]["all_eligible_rows"][task][key], fm["V2_AS_EVALUATED"][task][test_key]],
            "V2_AS_SERVED": [lce["V2_AS_SERVED"]["all_eligible_rows"][task][key], fm["V2_AS_SERVED"][task][test_key]],
            "V3": [None, fm["V3"][task][test_key]],
        }
        n = len(series); w = 0.8 / n
        for i, (name, vals) in enumerate(series.items()):
            for j, v in enumerate(vals):
                if v is None:
                    continue
                ax.bar(j + (i - (n - 1) / 2) * w, v, w, color=MODEL_COLORS[name], label=name if j == 0 or v is not None and name == "V3" else None)
                ax.text(j + (i - (n - 1) / 2) * w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
                rows.append((task, labels[j], name, v))
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels); ax.set_title(title, fontsize=9)
    h, l = axes[0].get_legend_handles_labels(); seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Defect D1: frozen V2 as originally evaluated (null baseline SOFA leaks horizon availability) vs as served (SOFA(t) for every row)", y=1.02, fontsize=10)
    ex.save_table("t21_defect_d1", "Frozen V2 as-evaluated vs as-served, and V3", ("task", "cohort", "model", "value"), rows, src, latex=True)
    ex.save_figure(fig, "f21_defect_d1_v2_modes", "Defect D1 figure", src, "Recovery tasks are unaffected (eligible rows always have SOFA(t)).", formats=("png", "svg", "pdf"))


BUILDERS: Sequence[Callable[[Exporter], None]] = (
    fig_population, fig_age_duration, fig_sofa, fig_observation, fig_missingness, fig_eligibility, fig_learning_curve,
    fig_leaderboard, fig_ablation, fig_control_vs_selected, fig_roc_pr, fig_reliability, fig_confusion, fig_threshold_sensitivity,
    fig_residuals, fig_error_slices, fig_robustness, fig_shap, fig_paired, fig_defect_d1,
)


# ============================================================================
# Registry and index
# ============================================================================

def registry_entries(root: Path) -> List[Dict[str, object]]:
    """Source artifacts -> experiment entries (hashes computed from the files on disk)."""

    commit = git_commit(root)
    ledger = [json.loads(l) for l in (root / (P2 + "trial_ledger_v1.jsonl")).read_text().splitlines() if l.strip()]
    stage_n = {s: sum(1 for r in ledger if r["stage"] == s) for s in ("S0", "S1", "S2", "S3", "S4")}
    fm = json.loads((root / (P4 + "final_metrics_v3.json")).read_text())
    den = fm["denominators"]
    lc = json.loads((root / (DC + "learning_curve_metrics.json")).read_text())

    def E(eid, rq, partition, ref, status, metrics, sizes):
        return {"experiment_id": eid, "research_question": rq, "partition": partition, "artifact_ref": ref, "sha256": sha256_file(root / ref),
                "status": status, "metric_definitions": metrics, "sample_sizes": sizes, "code_commit": commit}

    stay_bal = "stay-balanced (1/n_i per stay)"
    entries = [
        E("DC-cohort-summary", "RQ1", "development pool + LCE", DC + "cohort_summary.json", "DEVELOPMENT_CHARACTERIZATION", "descriptive counts", {"pool_subjects": 8000, "lce_subjects": 1500}),
        E("DC-target-eligibility", "RQ1", "development pool + LCE", DC + "target_eligibility.json", "DEVELOPMENT_CHARACTERIZATION", "eligible rows / unique stays per task", {}),
        E("DC-outcome-prevalence", "RQ1", "development pool + LCE", DC + "outcome_prevalence.json", "DEVELOPMENT_CHARACTERIZATION", "support positive / negative / censored", {}),
        E("DC-missingness", "RQ1", "development pool", DC + "missingness_by_feature.json", "DEVELOPMENT_CHARACTERIZATION", "missing fraction over non-padded bins", {}),
        E("DC-overlap-audit", "RQ1", "all cohorts", DC + "overlap_audit_v1.json", "AUDIT_PASS", "subject / stay / exact trajectory-fingerprint intersections", {}),
        E("DC-generator-equivalence", "RQ1", "V3 cohorts", DC + "generator_equivalence_audit_v1.json", "AUDIT_PASS", "scientific config and code identity", {}),
        E("DC-structural-audit", "RQ1", "V3 cohorts", DC + "structural_audit_v1.json", "AUDIT_PASS", "temporal / label structural invariants", {}),
        E("DC-null-pattern-audit", "RQ2", "development pool + LCE", DC + "null_pattern_audit_v1.json", "AUDIT_PASS", "elapsed-matched NaN-rate gap vs horizon availability (defect D1)", {}),
        E("RQ1-learning-curve", "RQ1", "train subsets of dev pool; evaluated on LCE", DC + "learning_curve_metrics.json", "DEVELOPMENT_EVIDENCE_CORRECTED",
          "task primary metric " + stay_bal + "; 500-replicate stay-cluster bootstrap (repeat 0)", {"eval_subjects": lc.get("n_eval_subjects"), "sizes": lc.get("sizes")}),
        E("RQ3-trial-ledger", "RQ3", "5/3-fold cross-fitting on 8000-subject dev pool", P2 + "trial_ledger_v1.jsonl", "DEVELOPMENT_EVIDENCE",
          "recovery MAE; ICU weighted median AE (h); support raw-score AUPRC; all stay-balanced", {"trials_by_stage": stage_n, "total": len(ledger)}),
        E("RQ3-candidate-selection", "RQ3", "5-fold cross-fitting, dev pool", P2 + "candidate_selection_v1.json", "DEVELOPMENT_EVIDENCE",
          "lowest mean primary; tie tolerance 0.5% (AUPRC 0.003); V2-recipe retention rule", {"folds": 5}),
        E("RQ4-support-calibration", "RQ4", "nested cross-fit OOF scores, dev pool", P2 + "support_calibration_v1.json", "DEVELOPMENT_EVIDENCE",
          "stay-balanced log loss, Brier, ECE(10 equal-mass), slope/intercept", {"oof_rows": 31434}),
        E("DEV-robustness", "RQ3", "LCE 500-stay subsample", P2 + "robustness_v1.json", "DEVELOPMENT_STRESS_TEST", "primary metrics under perturbation", {"stays": 500, "rows": 2859}),
        E("DEV-explainability", "RQ3", "LCE 1500-row sample", P2 + "explainability_v1.json", "DEVELOPMENT_EVIDENCE", "mean |SHAP| on raw margin", {"rows": 1500}),
        E("DEV-lce-preview", "RQ5", "LCE (development)", P2 + "lce_preview_v1.json", "DEVELOPMENT_SANITY_PREVIEW", "primary metrics, not used for selection", {}),
        E("D1-v2-modes-on-lce", "RQ5", "LCE (development)", DC + "v2_modes_on_lce_v1.json", "DEFECT_EVIDENCE", "frozen V2 as-evaluated vs as-served", {"rows": 8909}),
        E("D1-v2-discrepancy-check", "RQ5", "LCE + structural V2 fresh", DC + "v2_support_discrepancy_check_v1.json", "DEFECT_EVIDENCE", "identity + structural comparison", {}),
        E("BUNDLE-v3-manifest", "RQ5", "development pool refit", P3 + "v3/bundle_manifest.json", "FROZEN_PRETEST", "model bundle with hashes", {"subjects": 8000}),
        E("BUNDLE-control-manifest", "RQ5", "development pool refit", P3 + "control/bundle_manifest.json", "FROZEN_PRETEST", "V2 recipe refit (repaired)", {"subjects": 8000}),
        E("FREEZE-selected-models-v3", "RQ5", "pre-test freeze", P3 + "selected_models_v3.json", "FROZEN_PRETEST", "frozen selection manifest", {}),
        E("FINAL-metrics", "RQ5", "SYN-V3-T sealed test (1500 subjects)", P4 + "final_metrics_v3.json", "FINAL_TEST_EVALUATED_ONCE",
          "recovery MAE; ICU weighted median AE (h); support calibrated AUPRC/AUROC/Brier; " + stay_bal, {"denominators": den}),
        E("FINAL-paired", "RQ5", "SYN-V3-T sealed test", P4 + "paired_v2_v3_metrics.json", "FINAL_TEST_EVALUATED_ONCE", "paired stay-cluster bootstrap differences; promotion rule", {"bootstrap": fm["bootstrap"]}),
        E("FINAL-paired-bootstrap", "RQ5", "SYN-V3-T sealed test", P4 + "paired_bootstrap_v1.json", "FINAL_TEST_EVALUATED_ONCE", "percentile CIs 95% / 98.75%", {"replicates": 2000}),
        E("FINAL-calibration", "RQ4", "SYN-V3-T sealed test", P4 + "calibration_evidence_v1.json", "FINAL_TEST_EVALUATED_ONCE", "reliability, ROC/PR curves, confusion", {}),
        E("FINAL-error-analysis", "RQ5", "SYN-V3-T sealed test", P4 + "error_analysis_v1.json", "FINAL_TEST_EVALUATED_ONCE", "prespecified slices", {}),
        E("FINAL-naive-baselines", "RQ5", "SYN-V3-T sealed test", P4 + "naive_baselines_v1.json", "FINAL_TEST_DESCRIPTIVE", "Master Plan section 19 naive floors (dev constants)", {}),
        E("GOV-protocol-freeze", "RQ5", "governance", "artifacts/performance_v3/governance/v3_protocol_freeze_v1.json", "GOVERNANCE", "protocol hash binding", {}),
        E("GOV-amendment-freeze", "RQ5", "governance", "artifacts/performance_v3/governance/v3_protocol_amendment_v1_1_freeze.json", "GOVERNANCE", "defect D1 amendment", {}),
        E("GOV-model-freeze", "RQ5", "governance", "artifacts/performance_v3/governance/v3_model_freeze_v1.json", "GOVERNANCE", "pre-test model freeze", {}),
        E("GOV-final-freeze", "RQ5", "governance", "artifacts/performance_v3/governance/v3_final_evaluation_freeze_v1.json", "GOVERNANCE", "final evaluation freeze", {}),
    ]
    invalid_candidates = [
        (P2 + "INVALIDATED_trial_ledger_v1_nullSOFA_leak.jsonl", "INVALIDATED_D1_ledger", "RQ3", "pre-repair search ledger (79 trials)"),
        (DC + "invalidated_pre_D1/learning_curve_metrics.json", "INVALIDATED_D1_learning_curve", "RQ1", "pre-repair learning curve"),
        (DC + "invalidated_pre_D1/learning_curve_table.csv", "INVALIDATED_D1_learning_curve_table", "RQ1", "pre-repair learning curve table"),
        (DC + "invalidated_pre_D1/learning_curve_bootstrap.json", "INVALIDATED_D1_learning_curve_bootstrap", "RQ1", "pre-repair bootstrap"),
    ]
    for ref, eid, rq, what in invalid_candidates:
        if (root / ref).exists():
            entries.append(E(eid, rq, "development (pre-repair)", ref, "INVALIDATED_D1", what + "; built on the null-baseline-SOFA feature", {}))
    cache = root / "artifacts/performance_v3/cache"
    for f in sorted(cache.glob("INVALIDATED_preD1_*.json")) if cache.exists() else []:
        entries.append(E("INVALIDATED_D1_" + f.stem, "RQ2", "development (pre-repair cache sidecar)", str(f.relative_to(root)), "INVALIDATED_D1",
                         "feature cache sidecar built before repair (local, gitignored)", {}))
    entries.sort(key=lambda e: e["experiment_id"])
    return entries


def write_registry(ex: Exporter) -> Dict[str, object]:
    entries = registry_entries(ex.root)
    registry = {"artifact_version": "performance_v3_research_registry_v1", "export_version": EXPORT_VERSION, "model_bundle_id": BUNDLE_ID,
                "code_commit": ex.commit, "research_questions": RQ, "entries": entries,
                "exports": sorted({e["file"] for e in ex.entries}), "skipped": ex.skipped}
    (ex.root / REGISTRY_REL).write_bytes(canonical(registry))
    return registry


def write_index(ex: Exporter) -> None:
    lines = ["# Performance V3 — Figure and Table Index", "",
             "Evidence index (not a paper). Every file is generated by `scripts/performance_v3_research_export.py` from frozen or",
             "development-tagged artifacts; each has a `.meta.json` sidecar with source paths, sha256, bundle id and git commit.",
             "", f"- bundle id: `{BUNDLE_ID}`", f"- export version: `{EXPORT_VERSION}`", f"- code commit: `{ex.commit}`", "",
             "| id | kind | file | file sha256 (first 16) | sources |", "|---|---|---|---|---|"]
    for e in sorted(ex.entries, key=lambda e: (e["id"], e["file"])):
        srcs = "<br>".join(f"`{s['path']}` ({s['sha256'][:12]})" for s in e["sources"])
        lines.append(f"| {e['id']} | {e['kind']} | `{e['file']}` | {e['file_sha256'][:16]} | {srcs} |")
    if ex.skipped:
        lines += ["", "## Skipped (not supported by an existing artifact)", ""] + [f"- `{s['id']}`: {s['reason']}" for s in ex.skipped]
    (ex.root / INDEX_REL).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(root: Path, out_rel: str = OUT_REL, write_docs: bool = True) -> Exporter:
    ex = Exporter(root, out_rel)
    for build in BUILDERS:
        try:
            build(ex)
        except FileNotFoundError as error:
            ex.skip(build.__name__, f"missing source artifact: {error}")
        except KeyError as error:
            ex.skip(build.__name__, f"artifact lacks expected field: {error}")
    if write_docs:
        write_registry(ex)
        write_index(ex)
    return ex
