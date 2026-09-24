<script lang="ts">
	import { onMount } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import GaugeRing from '$lib/components/dashboard/GaugeRing.svelte';
	import { api } from '$lib/services/api';

	type Performance = {
		final_metrics: any;
		bootstrap: any;
		naive_comparison: any;
		calibration_evidence: any;
		generalization_comparison: any;
		v1_historical_test: { recovery24_mae: number; recovery48_mae: number; icu_median_ae_hours: number; support_calibrated_auprc: number };
	};

	let data = $state<Performance | null>(null);
	let error = $state('');

	onMount(async () => {
		try {
			data = (await api.performance()) as Performance;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not load the frozen Phase-4 evaluation artifacts';
		}
	});

	function ci(entry: any) {
		return `[${entry.ci_lower.toFixed(4)}, ${entry.ci_upper.toFixed(4)}]`;
	}
</script>

<svelte:head><title>Model Performance | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage eyebrow="03 / MODEL PERFORMANCE" title="The final, one-time fresh-test evaluation." description="Every number on this page is read directly from the frozen Phase-4 artifacts. No inference runs on this page.">
	{#if error}
		<div class="error">{error}</div>
	{:else if !data}
		<div class="loading">Loading frozen evaluation artifacts…</div>
	{:else}
		<div class="meta">
			<span>SOURCE / artifacts/performance_v2/phase4/*</span>
			<span>BOOTSTRAP / B={data.bootstrap.config.n_bootstrap} · SEED={data.bootstrap.config.seed}</span>
			<span>STATUS / ONE-TIME FRESH-TEST EVALUATION, FROZEN</span>
		</div>

		<div class="metrics">
			<MetricTile label="Recovery +24 MAE" value={data.final_metrics.recovery24.mae.toFixed(4)} detail={`95% CI ${ci(data.bootstrap.recovery24.mae)}`} tone="teal" />
			<MetricTile label="Recovery +48 MAE" value={data.final_metrics.recovery48.mae.toFixed(4)} detail={`95% CI ${ci(data.bootstrap.recovery48.mae)}`} tone="teal" />
			<MetricTile label="ICU median AE" value={data.final_metrics.icu_stay_time.median_absolute_error.toFixed(4)} unit="h" detail={`95% CI ${ci(data.bootstrap.icu_stay_time.median_absolute_error)}`} tone="cyan" />
			<MetricTile label="Support calibrated AUPRC" value={data.final_metrics.organ_support_calibrated.auprc.toFixed(4)} detail={`95% CI ${ci(data.bootstrap.organ_support_calibrated.auprc)}`} tone="amber" />
		</div>

		<div class="grid two">
			<Panel eyebrow="RECOVERY / VS NAIVE" title="MAE vs. zero-delta naive baseline" note="STAY-BALANCED">
				<div class="bars">
					<div class="bar-row"><span>+24h model</span><i style={`width:${(data.final_metrics.recovery24.mae / data.naive_comparison.recovery24.naive_mae) * 100}%`}></i><b>{data.final_metrics.recovery24.mae.toFixed(3)}</b></div>
					<div class="bar-row naive"><span>+24h naive</span><i style="width:100%"></i><b>{data.naive_comparison.recovery24.naive_mae.toFixed(3)}</b></div>
					<div class="bar-row"><span>+48h model</span><i style={`width:${(data.final_metrics.recovery48.mae / data.naive_comparison.recovery48.naive_mae) * 100}%`}></i><b>{data.final_metrics.recovery48.mae.toFixed(3)}</b></div>
					<div class="bar-row naive"><span>+48h naive</span><i style="width:100%"></i><b>{data.naive_comparison.recovery48.naive_mae.toFixed(3)}</b></div>
				</div>
				<p class="caption">+{(data.naive_comparison.recovery24.relative_improvement * 100).toFixed(1)}% / +{(data.naive_comparison.recovery48.relative_improvement * 100).toFixed(1)}% improvement over naive (lower MAE is better).</p>
			</Panel>
			<Panel eyebrow="ICU / VS NAIVE" title="Median AE vs. DEV-median naive baseline" note="STAY-BALANCED">
				<div class="bars">
					<div class="bar-row"><span>model</span><i style={`width:${(data.final_metrics.icu_stay_time.median_absolute_error / data.naive_comparison.icu_stay_time.naive_median_ae) * 100}%`}></i><b>{data.final_metrics.icu_stay_time.median_absolute_error.toFixed(2)}h</b></div>
					<div class="bar-row naive"><span>naive</span><i style="width:100%"></i><b>{data.naive_comparison.icu_stay_time.naive_median_ae.toFixed(2)}h</b></div>
				</div>
				<p class="caption">+{(data.naive_comparison.icu_stay_time.relative_improvement * 100).toFixed(1)}% improvement over the DEV stay-balanced median-hours baseline.</p>
			</Panel>
		</div>

		<Panel eyebrow="ORGAN SUPPORT / FULL METRICS" title="Raw vs. calibrated discrimination" note="FROZEN THRESHOLD 0.3978">
			<div class="support-grid">
				<div class="gauges">
					<GaugeRing value={Math.round(data.final_metrics.organ_support_raw.auprc * 100)} label="RAW AUPRC" color="#94a3b8" size={100} />
					<GaugeRing value={Math.round(data.final_metrics.organ_support_calibrated.auprc * 100)} label="CAL. AUPRC" color="#2bb8b0" size={100} />
					<GaugeRing value={Math.round(data.final_metrics.organ_support_calibrated.f1 * 100)} label="F1 @ THRESH." color="#0ea5e9" size={100} />
					<GaugeRing value={Math.round(data.naive_comparison.organ_support.relative_improvement * 100)} label="VS NAIVE" color="#fbbf24" size={100} />
				</div>
				<table>
					<tbody>
						<tr><td>Raw AUROC</td><td>{data.final_metrics.organ_support_raw.auroc.toFixed(4)}</td><td>Raw Brier</td><td>{data.final_metrics.organ_support_raw.brier.toFixed(4)}</td></tr>
						<tr><td>Calibrated AUROC</td><td>{data.final_metrics.organ_support_calibrated.auroc.toFixed(4)}</td><td>Calibrated Brier</td><td>{data.final_metrics.organ_support_calibrated.brier.toFixed(4)}</td></tr>
						<tr><td>Precision</td><td>{data.final_metrics.organ_support_calibrated.precision.toFixed(4)}</td><td>Recall / sensitivity</td><td>{data.final_metrics.organ_support_calibrated.sensitivity.toFixed(4)}</td></tr>
						<tr><td>Specificity</td><td>{data.final_metrics.organ_support_calibrated.specificity.toFixed(4)}</td><td>Naive AUPRC</td><td>{data.naive_comparison.organ_support.naive_auprc.toFixed(4)}</td></tr>
					</tbody>
				</table>
			</div>
			<p class="caption">Calibrated AUPRC ({data.final_metrics.organ_support_calibrated.auprc.toFixed(4)}) is below raw AUPRC ({data.final_metrics.organ_support_raw.auprc.toFixed(4)}) — isotonic regression is monotonic non-decreasing, so this reflects tie-resolution loss, not a ranking reversal.</p>
		</Panel>

		<Panel eyebrow="BENCHMARK V1 → V2" title="Historical comparison across independent cohorts" note="NOT PAIRED STATISTICAL TESTING">
			<table class="compare">
				<thead><tr><th>Task</th><th>v1 final-test (historical)</th><th>v2 fresh-test</th></tr></thead>
				<tbody>
					<tr><td>Recovery24 MAE</td><td>{data.v1_historical_test.recovery24_mae.toFixed(4)}</td><td>{data.final_metrics.recovery24.mae.toFixed(4)}</td></tr>
					<tr><td>Recovery48 MAE</td><td>{data.v1_historical_test.recovery48_mae.toFixed(4)}</td><td>{data.final_metrics.recovery48.mae.toFixed(4)}</td></tr>
					<tr><td>ICU median AE (h)</td><td>{data.v1_historical_test.icu_median_ae_hours.toFixed(4)}</td><td>{data.final_metrics.icu_stay_time.median_absolute_error.toFixed(4)}</td></tr>
					<tr><td>Support calibrated AUPRC</td><td>{data.v1_historical_test.support_calibrated_auprc.toFixed(4)}</td><td>{data.final_metrics.organ_support_calibrated.auprc.toFixed(4)}</td></tr>
				</tbody>
			</table>
			<p class="caption">{data.generalization_comparison.interpretation === 'descriptive_only_no_retraining' ? 'Descriptive only — no retraining occurred in response to either comparison.' : ''} v1 and v2 used different independent final-test cohorts (different generator seed and subject namespace).</p>
		</Panel>
	{/if}
</WorkbenchPage>

<style>
	.loading, .error { padding: 20px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.error { color: #fecdd3; }
	.meta { display: flex; flex-wrap: wrap; gap: 24px; padding: 12px 0; border-block: 1px solid rgba(148,163,184,.14); color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .1em; margin-bottom: 16px; }
	.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 8px; }
	.grid.two { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0; }
	.bars { display: grid; gap: 10px; }
	.bar-row { display: grid; grid-template-columns: 90px 1fr 60px; align-items: center; gap: 10px; }
	.bar-row span { color: #94a3b8; font: 9px 'JetBrains Mono', monospace; }
	.bar-row i { display: block; height: 8px; border-radius: 2px; background: #2bb8b0; box-shadow: 0 0 6px #2bb8b040; }
	.bar-row.naive i { background: #64748b; box-shadow: none; }
	.bar-row b { color: #dce9e8; font: 10px 'JetBrains Mono', monospace; text-align: right; }
	.caption { padding-top: 14px; margin: 14px 0 0; border-top: 1px solid rgba(148,163,184,.12); color: #71829a; font-size: 10px; line-height: 1.6; }
	.support-grid { display: grid; grid-template-columns: auto 1fr; gap: 24px; align-items: center; }
	.gauges { display: flex; flex-wrap: wrap; gap: 14px; }
	.support-grid table, .compare { width: 100%; border-collapse: collapse; font-size: 11px; }
	.support-grid td { padding: 8px; color: #cbd5e1; border-bottom: 1px solid rgba(148,163,184,.1); font: 10px 'JetBrains Mono', monospace; }
	.support-grid td:nth-child(odd) { color: #64748b; }
	.compare th { text-align: left; padding: 8px; color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; border-bottom: 1px solid rgba(148,163,184,.16); }
	.compare td { padding: 8px; color: #cbd5e1; border-bottom: 1px solid rgba(148,163,184,.08); font: 10px 'JetBrains Mono', monospace; }
	@media (max-width: 900px) {
		.metrics { grid-template-columns: repeat(2, 1fr); }
		.grid.two { grid-template-columns: 1fr; }
		.support-grid { grid-template-columns: 1fr; }
	}
</style>
