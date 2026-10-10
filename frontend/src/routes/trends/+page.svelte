<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { api, type PredictionResponse } from '$lib/services/api';
	import ModelBadge from '$lib/components/patient/ModelBadge.svelte';
	import { patientContext as ctx } from '$lib/stores/patient-context.svelte';
	import { authToken } from '$lib/stores/auth-token.svelte';
	import { untrack } from 'svelte';

	const subjects = $derived(ctx.subjects);
	const stayId = $derived(ctx.stayId);
	const cutoffIndex = $derived(ctx.cutoffIndex);
	const prediction = $derived<PredictionResponse | null>(ctx.prediction);
	const loading = $derived(ctx.loading || ctx.loadingSubjects);
	const error = $derived(ctx.error);
	const cutoffs = $derived(ctx.cutoffs);
	const tasks = $derived(prediction?.model_bundle?.tasks);

	async function selectStay(id: string) {
		await ctx.select(id, 0);
		void goto(ctx.urlQuery(), { replaceState: true, noScroll: true });
	}
	async function setCutoff(index: number) {
		await ctx.setCutoff(index);
		void goto(ctx.urlQuery(), { replaceState: true, noScroll: true });
	}
	async function init() {
		await ctx.ensureSubjects(ctx.subjects.length === 0);
		const { stay, cutoff } = ctx.parseUrl(page.url.searchParams);
		if (stay) await ctx.ensureCustom(stay);
		const requested = stay && ctx.subjects.some((s) => s.stay_id === stay) ? stay : null;
		if (requested && (requested !== ctx.stayId || cutoff !== ctx.cutoffIndex)) await ctx.select(requested, cutoff);
		else if (!ctx.stayId && ctx.subjects[0]) await ctx.select(ctx.subjects[0].stay_id, 0);
		else if (!ctx.prediction) await ctx.load();
	}
	let lastToken: string | null | undefined = undefined;
	$effect(() => {
		const token = authToken.value;
		untrack(() => {
			if (lastToken !== undefined && token !== lastToken && ctx.ensureOwner()) void init();
			lastToken = token;
		});
	});

	onMount(() => {
		void init();
	});
</script>

<svelte:head><title>Forecast Details | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage eyebrow="02 / FORECAST DETAILS" title="Every raw number, every reconstruction step." description="Task-specific detail panels showing raw model output alongside its exact, frozen postprocessing — nothing hidden, nothing recomputed differently than the replay view.">
	{#if error}<div class="error">{error}</div>{/if}
	{#if subjects.length}
		<div class="controls">
			<select value={stayId} onchange={(e) => selectStay((e.target as HTMLSelectElement).value)}>
				{#each subjects as s}<option value={s.stay_id}>{s.patient_alias ?? s.subject_id}</option>{/each}
			</select>
			<select value={cutoffIndex} onchange={(e) => setCutoff(Number((e.target as HTMLSelectElement).value))}>
				{#each cutoffs as c, i}<option value={i}>t{i} — {c.slice(0, 16).replace('T', ' ')}</option>{/each}
			</select>
			<ModelBadge bundle={prediction?.model_bundle} versions={prediction?.versions} />
		</div>
	{/if}

	{#if loading}
		<div class="loading">Loading…</div>
	{:else if prediction}
		<div class="grid three">
			<Panel eyebrow="RECOVERY" title="Independent ΔSOFA forecasts">
				<dl>
					<div><dt>Current SOFA</dt><dd>{prediction.current_sofa.toFixed(2)}</dd></div>
					<div><dt>Δ24 (raw)</dt><dd>{prediction.recovery.delta_24h.toFixed(4)}</dd></div>
					<div><dt>Δ48 (raw)</dt><dd>{prediction.recovery.delta_48h.toFixed(4)}</dd></div>
					<div><dt>Reconstructed +24h</dt><dd>clip({prediction.current_sofa.toFixed(1)} + {prediction.recovery.delta_24h.toFixed(2)}, 0, 24) = <b>{prediction.recovery.sofa_hat_24h.toFixed(2)}</b></dd></div>
					<div><dt>Reconstructed +48h</dt><dd>clip({prediction.current_sofa.toFixed(1)} + {prediction.recovery.delta_48h.toFixed(2)}, 0, 24) = <b>{prediction.recovery.sofa_hat_48h.toFixed(2)}</b></dd></div>
					<div><dt>Model (+24h)</dt><dd>{tasks?.recovery24?.family} · {tasks?.recovery24?.feature_variant}</dd></div>
					<div><dt>Model (+48h)</dt><dd>{tasks?.recovery48?.family} · {tasks?.recovery48?.feature_variant}</dd></div>
				</dl>
				<p class="note">The two horizons are computed independently from the same baseline — +48h is never derived from the +24h output.</p>
			</Panel>
			<Panel eyebrow="ICU" title="Remaining stay-time forecast">
				<dl>
					<div><dt>Predicted hours</dt><dd><b>{prediction.icu_stay_time.remaining_hours.toFixed(2)}</b> h</dd></div>
					<div><dt>Human-readable</dt><dd>{Math.floor(prediction.icu_stay_time.remaining_hours / 24)}d {(prediction.icu_stay_time.remaining_hours % 24).toFixed(1)}h</dd></div>
					<div><dt>Raw log1p prediction</dt><dd>{prediction.icu_stay_time.raw_log_prediction.toFixed(4)}</dd></div>
					<div><dt>Postprocess</dt><dd>expm1(clamp_min(raw, 0))</dd></div>
					<div><dt>Model</dt><dd>{tasks?.icu_stay_time?.family} · {tasks?.icu_stay_time?.feature_variant}</dd></div>
				</dl>
			</Panel>
			<Panel eyebrow="SUPPORT" title="Raw vs. calibrated probability">
				<dl>
					<div><dt>Raw model probability</dt><dd>{prediction.organ_support.raw_probability.toFixed(4)}</dd></div>
					<div><dt>Calibrated probability</dt><dd><b>{prediction.organ_support.probability_24h.toFixed(4)}</b></dd></div>
					<div><dt>Frozen threshold</dt><dd>{prediction.organ_support.threshold.toFixed(6)}</dd></div>
					<div><dt>Current alert state</dt><dd class={prediction.organ_support.alert ? 'alert-above' : 'alert-below'}>{prediction.organ_support.alert ? 'ABOVE THRESHOLD' : 'BELOW THRESHOLD'}</dd></div>
					<div><dt>Model</dt><dd>{tasks?.organ_support?.family} · {tasks?.organ_support?.feature_variant}</dd></div>
					<div><dt>Calibration</dt><dd>{prediction.model_bundle?.calibrator?.method ?? '—'} (fit on development data only)</dd></div>
				</dl>
				<p class="note">Raw model score and calibrated serving probability are always shown separately — never presented as identical.</p>
			</Panel>
		</div>
	{/if}
</WorkbenchPage>

<style>
	.loading, .error { padding: 20px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.error { color: #fecdd3; }
	.controls { display: flex; gap: 8px; padding: 12px 0; border-block: 1px solid rgba(148,163,184,.14); margin-bottom: 16px; }
	.controls select { padding: 8px 12px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #cbd5e1; font: 9px 'JetBrains Mono', monospace; }
	.grid.three { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
	dl { display: grid; gap: 10px; margin: 0; }
	dl div { display: grid; gap: 2px; }
	dt { color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
	dd { margin: 0; color: #cbd5e1; font: 11px 'JetBrains Mono', monospace; }
	dd b { color: #eef7f6; font-weight: 500; }
	.alert-above { color: #fb7185; }
	.alert-below { color: #2bb8b0; }
	.note { margin: 12px 0 0; padding-top: 12px; border-top: 1px solid rgba(148,163,184,.1); color: #71829a; font-size: 10px; line-height: 1.5; }
	@media (max-width: 900px) { .grid.three { grid-template-columns: 1fr; } }
</style>
