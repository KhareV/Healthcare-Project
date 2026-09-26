<script lang="ts">
	import { onMount } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import { api } from '$lib/services/api';
	import { authFlag } from '$lib/stores/auth.svelte';
	import { Database, PencilLine, PlugZap } from '@lucide/svelte';

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
	{#if authFlag.enabled}
		{#await import('$lib/components/dashboard/WorkspaceGreeting.svelte') then { default: WorkspaceGreeting }}<WorkspaceGreeting />{/await}
	{/if}

	<div class="source-grid">
		<a class="source-card active" href="/patients">
			<Database size={18} /><strong>Explore Demo Patients</strong><span>Three structurally-selected synthetic ICU episodes, ready now</span>
		</a>
		<a class="source-card active" href="/patients/custom">
			<PencilLine size={18} /><strong>Enter My Own Record</strong><span>Type in vitals & labs and forecast them through the real frozen models</span>
		</a>
		<div class="source-card soon">
			<PlugZap size={18} /><strong>Connect EHR Sandbox</strong><span>Advanced / sandbox — coming in a follow-up release</span><em>SOON</em>
		</div>
	</div>

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
	.source-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 16px; }
	.source-card { position: relative; display: flex; flex-direction: column; align-items: flex-start; gap: 6px; padding: 16px; border: 1px solid rgba(148,163,184,.16); background: #080e1d; color: #94a3b8; text-decoration: none; }
	.source-card strong { color: #eef7f6; font: 600 12px 'Space Grotesk', sans-serif; }
	.source-card span { color: #64748b; font-size: 11px; }
	.source-card.active { border-color: rgba(43,184,176,.4); }
	.source-card.active:hover { background: rgba(43,184,176,.06); }
	.source-card.soon { opacity: .55; }
	.source-card em { position: absolute; top: 12px; right: 12px; color: #fbbf24; font: 7px 'JetBrains Mono', monospace; letter-spacing: .1em; font-style: normal; }
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
	@media (max-width: 900px) { .metrics { grid-template-columns: 1fr 1fr; } .grid.three { grid-template-columns: 1fr; } .source-grid { grid-template-columns: 1fr; } }
</style>
