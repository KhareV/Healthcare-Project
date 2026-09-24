<script lang="ts">
	import { onMount } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import { api } from '$lib/services/api';

	let demoCount = $state<number | null>(null);
	let apiReady = $state<boolean | null>(null);

	onMount(async () => {
		try {
			const [health, demo] = await Promise.all([api.health(), api.demoSubjects()]);
			apiReady = health.ready;
			demoCount = demo.demo_subjects.length;
		} catch {
			apiReady = false;
		}
	});
</script>

<svelte:head><title>Overview | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage eyebrow="00 / PROJECT OVERVIEW" title="Personalized Patient Recovery Trajectory." description="Time-series forecasting of recovery, remaining ICU stay time, and new organ-support initiation risk — from a fixed 48-hour history window, recomputed at every sequential replay cutoff.">
	<div class="metrics">
		<MetricTile label="Lookback window" value="48" unit="h" detail="EIGHT 6-HOUR BINS" />
		<MetricTile label="Model family" value="XGBoost" detail="ALL 4 FINAL V2 TASKS" tone="cyan" />
		<MetricTile label="Demo cohort" value={demoCount ?? '—'} detail="DEV / TRAIN SUBJECTS ONLY" tone="teal" />
		<MetricTile label="Serving API" value={apiReady === null ? 'CHECKING' : apiReady ? 'READY' : 'OFFLINE'} detail="MODEL-BACKED" tone={apiReady ? 'teal' : 'amber'} />
	</div>

	<div class="grid three">
		<Panel eyebrow="OUTPUT 01" title="Recovery trajectory" note="ΔSOFA +24h / +48h">
			<p>Two independent forecasts of SOFA change reconstructed as absolute severity: <code>clip(current + Δ, 0, 24)</code>. Never chained.</p>
			<a href="/patients">Open Patient Replay →</a>
		</Panel>
		<Panel eyebrow="OUTPUT 02" title="Remaining ICU stay time" note="HOURS, POSTPROCESSED">
			<p>A single regression forecast in raw log1p space, postprocessed via <code>expm1(clamp_min(raw,0))</code> to non-negative hours.</p>
			<a href="/trends">Open Forecast Details →</a>
		</Panel>
		<Panel eyebrow="OUTPUT 03" title="New organ-support initiation risk" note="24H, CALIBRATED">
			<p>Raw XGBoost probability → isotonic calibration → frozen decision threshold. Vasopressor or invasive ventilation, OFF→ON only.</p>
			<a href="/ai/insights">Open Explainability →</a>
		</Panel>
	</div>

	<Panel eyebrow="METHOD" title="Retrospective sequential replay, not real-time prediction">
		<p class="method">At each of the demo cohort's legal historical cutoffs, the API recomputes every forecast from raw event-time history truncated strictly at that cutoff — the same canonical timestamp generator, feature builder, and frozen Phase-3 XGBoost models used throughout Performance-V2. The fresh V2 test cohort used for the final scientific evaluation is a fully sealed, separate population and is structurally unreachable from this application.</p>
		<div class="links">
			<a href="/patients">Patient Replay</a>
			<a href="/trends">Forecast Details</a>
			<a href="/research/analytics">Model Performance</a>
			<a href="/ai/insights">AI + Explainability</a>
			<a href="/system/data">Data Quality &amp; Provenance</a>
			<a href="/demo">Guided demo</a>
		</div>
	</Panel>
</WorkbenchPage>

<style>
	.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 8px; }
	.grid.three { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 8px 0; }
	.grid.three p { margin: 0 0 14px; color: #94a3b8; font-size: 12px; line-height: 1.7; }
	.grid.three code { color: #8fc6bf; font: 10px 'JetBrains Mono', monospace; }
	.grid.three a { color: #2bb8b0; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; text-decoration: none; }
	.grid.three a:hover { text-decoration: underline; }
	.method { margin: 0; color: #94a3b8; font-size: 12px; line-height: 1.75; }
	.links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(148,163,184,.12); }
	.links a { padding: 8px 12px; border: 1px solid rgba(148,163,184,.2); color: #cbd5e1; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; text-decoration: none; }
	.links a:hover { border-color: #2bb8b0; color: #2bb8b0; }
	@media (max-width: 900px) { .metrics { grid-template-columns: 1fr 1fr; } .grid.three { grid-template-columns: 1fr; } }
</style>
