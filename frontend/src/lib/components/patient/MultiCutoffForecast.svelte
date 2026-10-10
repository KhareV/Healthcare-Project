<script lang="ts">
	// MultiCutoffForecast — how the forecasts evolved across the legal cutoffs actually computed.
	// Observed SOFA(t) and the forecasts available at or before the selected cutoff are solid;
	// later observed values (retrospective hindsight, optional) are dashed and labelled as not
	// available to the earlier forecasts.
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import LineChart from '$lib/components/research/LineChart.svelte';
	import type { PredictionResponse } from '$lib/services/api';
	import type { LineSeries } from '$lib/research/types';

	let { computed = {}, cutoffIndex = 0, showHindsight = false, onToggleHindsight }:
		{ computed?: Record<number, PredictionResponse>; cutoffIndex?: number; showHindsight?: boolean; onToggleHindsight?: () => void } = $props();

	const known = $derived(Object.keys(computed).map(Number).filter((i) => i <= cutoffIndex).sort((a, b) => a - b).map((i) => computed[i]));
	const later = $derived(Object.keys(computed).map(Number).filter((i) => i > cutoffIndex).sort((a, b) => a - b).map((i) => computed[i]));
	const x = (p: PredictionResponse) => p.elapsed_icu_hours;
	const pt = (xv: number, yv: number) => ({ x: xv, y: yv });

	const sofaSeries = $derived<LineSeries[]>([
		{ key: 'obs', name: 'Observed SOFA at cutoff (≤ selected cutoff)', color: '#2bb8b0', points: known.map((p) => pt(x(p), p.current_sofa)) },
		{ key: 'f24', name: 'Forecast +24h (made at each cutoff)', color: '#38bdf8', dashed: true, points: known.map((p) => pt(x(p) + 24, p.recovery.sofa_hat_24h)) },
		{ key: 'f48', name: 'Forecast +48h (independent)', color: '#fb7185', dashed: true, points: known.map((p) => pt(x(p) + 48, p.recovery.sofa_hat_48h)) },
		...(showHindsight && later.length
			? [{ key: 'hind', name: 'Observed later — not available to the earlier forecasts', color: '#94a3b8', dashed: true, points: [known[known.length - 1], ...later].filter(Boolean).map((p) => pt(x(p), p.current_sofa)) } as LineSeries]
			: [])
	]);
	const icuSeries = $derived<LineSeries[]>([{ key: 'icu', name: 'Predicted remaining ICU-episode hours', color: '#0ea5e9', points: known.map((p) => pt(x(p), p.icu_stay_time.remaining_hours)) }]);
	const threshold = $derived(known[known.length - 1]?.organ_support.threshold ?? 0);
	const supportSeries = $derived<LineSeries[]>([
		{ key: 'risk', name: 'Calibrated new-support risk (24h)', color: '#fbbf24', points: known.map((p) => pt(x(p), p.organ_support.probability_24h)) },
		{ key: 'thr', name: 'Frozen threshold', color: '#fb7185', dashed: true, points: known.length ? [pt(x(known[0]), threshold), pt(x(known[known.length - 1]), threshold)] : [] }
	]);
</script>

<Panel eyebrow="FORECAST / ACROSS CUTOFFS" title="How the forecasts evolved as new history became available" note={`${known.length} OF ${Object.keys(computed).length || known.length} CUTOFFS COMPUTED`}>
	<div class="grid">
		<div>
			<h4>SOFA — observed vs +24h/+48h forecasts</h4>
			<LineChart series={sofaSeries} xLabel="Elapsed ICU hours" yLabel="SOFA" xUnit="h" yDomain={[0, 24]} height={250} ariaLabel="Observed SOFA and independent forecasts across cutoffs" />
		</div>
		<div>
			<h4>Remaining ICU-episode time</h4>
			<LineChart series={icuSeries} xLabel="Elapsed ICU hours" yLabel="Remaining hours" yUnit="h" height={250} ariaLabel="Predicted remaining ICU hours across cutoffs" />
		</div>
		<div>
			<h4>New organ-support initiation risk (24h)</h4>
			<LineChart series={supportSeries} xLabel="Elapsed ICU hours" yLabel="Probability" yDomain={[0, 1]} height={250} ariaLabel="Calibrated support risk versus the frozen threshold across cutoffs" />
		</div>
	</div>
	<div class="foot">
		<p>Each forecast point was produced from history truncated at its own cutoff. The +48h value is never derived from the +24h value.</p>
		{#if onToggleHindsight}
			<button onclick={onToggleHindsight} aria-pressed={showHindsight}>{showHindsight ? 'Hide' : 'Show'} retrospective hindsight (observed SOFA at later cutoffs)</button>
		{/if}
	</div>
</Panel>

<style>
	.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
	h4 { margin: 0 0 6px; color: #2bb8b0; font: 9px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; }
	.foot { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; margin-top: 10px; }
	.foot p { margin: 0; color: #64748b; font-size: 10px; line-height: 1.6; flex: 1 1 280px; }
	.foot button { padding: 8px 12px; border: 1px solid rgba(148,163,184,.25); background: #0a1220; color: #cbd5e1; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; cursor: pointer; }
	.foot button[aria-pressed='true'] { border-color: #94a3b8; color: #e2e8f0; }
</style>
