<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import BarContrib from '$lib/components/dashboard/BarContrib.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import RadarPlot from '$lib/components/dashboard/RadarPlot.svelte';
	import GaugeRing from '$lib/components/dashboard/GaugeRing.svelte';
	import { api, type AIRecommendation } from '$lib/services/api';
	import ModelBadge from '$lib/components/patient/ModelBadge.svelte';
	import ShapPanel from '$lib/components/patient/ShapPanel.svelte';
	import { patientContext as ctx } from '$lib/stores/patient-context.svelte';
	import { authToken } from '$lib/stores/auth-token.svelte';
	import { untrack } from 'svelte';
	import { Sparkles, BrainCircuit, Volume2, VolumeX, Loader2 } from '@lucide/svelte';

	const TASK_TITLES: Record<string, string> = {
		recovery24: 'Recovery +24h', recovery48: 'Recovery +48h', icu_stay_time: 'Remaining ICU stay time', organ_support: 'Organ support (raw margin)'
	};

	let recommendation = $state<AIRecommendation | null>(null);
	let loadingAI = $state(false);
	let shapTask = $state<'recovery24' | 'recovery48' | 'icu_stay_time' | 'organ_support'>('organ_support');
	const subjects = $derived(ctx.subjects);
	const stayId = $derived(ctx.stayId);
	const cutoffIndex = $derived(ctx.cutoffIndex);
	const prediction = $derived(ctx.prediction);
	const loadingPrediction = $derived(ctx.loading || ctx.loadingSubjects);
	const error = $derived(ctx.error);

	let audioEl: HTMLAudioElement | null = $state(null);
	let narrateLoading = $state(false);
	let narratePlaying = $state(false);
	let narrateError = $state('');
	let narratedFor: string | null = null;
	let narratedUrl: string | null = null;

	function resetNarration() {
		if (audioEl) audioEl.pause();
		narratePlaying = false;
		narrateError = '';
		if (narratedUrl) URL.revokeObjectURL(narratedUrl);
		narratedUrl = null;
		narratedFor = null;
	}

	async function toggleNarration() {
		if (!recommendation || recommendation.status !== 'OK' || !recommendation.summary) return;
		if (narratePlaying) {
			audioEl?.pause();
			narratePlaying = false;
			return;
		}
		narrateError = '';
		try {
			if (narratedFor !== recommendation.summary) {
				narrateLoading = true;
				const blob = await api.speak(recommendation.summary);
				if (narratedUrl) URL.revokeObjectURL(narratedUrl);
				narratedUrl = URL.createObjectURL(blob);
				narratedFor = recommendation.summary;
				narrateLoading = false;
			}
			if (audioEl && narratedUrl) {
				if (audioEl.src !== narratedUrl) audioEl.src = narratedUrl;
				await audioEl.play();
				narratePlaying = true;
			}
		} catch (cause) {
			narrateLoading = false;
			narrateError = cause instanceof Error ? cause.message : 'Narration unavailable right now.';
		}
	}

	const subject = $derived(ctx.subject);
	const cutoffs = $derived(ctx.cutoffs);

	function contribFeatures(items: { label: string; attribution: number }[]) {
		return items.map((i) => ({ name: i.label, value: i.attribution }));
	}

	const crossTaskRadar = $derived.by(() => {
		if (!prediction) return { labels: [], values: [] };
		const entries = Object.entries(prediction.explanations);
		const topAbs = entries.map(([, exp]) => {
			const all = [...exp.top_positive_contributors, ...exp.top_negative_contributors];
			return all.length ? Math.max(...all.map((c) => Math.abs(c.attribution))) : 0;
		});
		const maxAbs = Math.max(...topAbs, 1e-6);
		return {
			labels: entries.map(([task]) => (TASK_TITLES[task] ?? task).toUpperCase().slice(0, 11)),
			values: topAbs.map((v) => Math.round((v / maxAbs) * 100))
		};
	});

	async function askAI() {
		if (!subject) return;
		loadingAI = true;
		recommendation = null;
		resetNarration();
		try {
			recommendation = await api.aiRecommendation(ctx.stayId, cutoffs[cutoffIndex]);
		} catch (cause) {
			recommendation = { status: 'UNAVAILABLE', summary: null, model: 'unknown', disclaimer: '', error: cause instanceof Error ? cause.message : 'request failed' };
		} finally {
			loadingAI = false;
		}
	}

	async function selectStay(id: string) {
		recommendation = null;
		resetNarration();
		await ctx.select(id, 0);
		void goto(ctx.urlQuery(), { replaceState: true, noScroll: true });
	}
	async function setCutoff(index: number) {
		recommendation = null;
		resetNarration();
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

	onMount(() => {
		void init();
	});
	let lastToken: string | null | undefined = undefined;
	$effect(() => {
		const token = authToken.value;
		untrack(() => {
			if (lastToken !== undefined && token !== lastToken && ctx.ensureOwner()) void init();
			lastToken = token;
		});
	});

	onDestroy(() => {
		if (narratedUrl) URL.revokeObjectURL(narratedUrl);
	});
</script>

<svelte:head><title>AI Insight + Explainability | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage eyebrow="04 / AI INSIGHT + EXPLAINABILITY" title="Make the forecast legible, then let a model narrate it." description="TreeSHAP decomposes each frozen V3 tree-ensemble output (XGBoost; CatBoost for the +48h recovery head) feature-by-feature; a separate LLM (Groq-hosted openai/gpt-oss-120b) turns those numbers into a plain-language research note — generated only after prediction, never influencing it.">
	{#if error}<div class="error">{error}</div>{/if}
	{#if subjects.length}
		<div class="controls">
			<select value={stayId} onchange={(e) => selectStay((e.target as HTMLSelectElement).value)}>
				{#each subjects as s}<option value={s.stay_id}>{s.patient_alias ?? s.subject_id} — {s.cardiac_condition_group.replace('SYNTHETIC_', '')}</option>{/each}
			</select>
			<select value={cutoffIndex} onchange={(e) => setCutoff(Number((e.target as HTMLSelectElement).value))}>
				{#each cutoffs as c, i}<option value={i}>t{i} — {c.slice(0, 16).replace('T', ' ')}</option>{/each}
			</select>
			<ModelBadge bundle={prediction?.model_bundle} versions={prediction?.versions} />
		</div>
	{/if}

	{#if loadingPrediction}
		<div class="loading">Recomputing prediction and TreeSHAP attributions…</div>
	{:else if prediction}
		<section class="ai-panel">
			<audio
				bind:this={audioEl}
				onended={() => (narratePlaying = false)}
				onpause={() => (narratePlaying = false)}
				onplay={() => (narratePlaying = true)}
			></audio>
			<header>
				<Sparkles size={16} /><div><span>AI RESEARCH NOTE</span><h3>Generated interpretive summary</h3></div>
				<button onclick={askAI} disabled={loadingAI}><BrainCircuit size={14} /> {loadingAI ? 'Thinking…' : recommendation ? 'Regenerate' : 'Generate summary'}</button>
				{#if recommendation?.status === 'OK' && recommendation.summary}
					<button class="voice-btn" class:speaking={narratePlaying} onclick={toggleNarration} disabled={narrateLoading}>
						{#if narrateLoading}<Loader2 size={13} class="spin" /> VOICING…{:else if narratePlaying}<VolumeX size={13} /> STOP{:else}<Volume2 size={13} /> LISTEN (KOKORO){/if}
					</button>
				{/if}
			</header>
			{#if loadingAI}
				<p class="ai-loading">Calling the language model with this prediction's numbers…</p>
			{:else if recommendation}
				{#if recommendation.status === 'OK'}
					<p class="ai-text">{recommendation.summary}</p>
					{#if narratePlaying}<div class="voice-indicator"><span></span><span></span><span></span><span></span> NARRATING VIA KOKORO TTS</div>{/if}
					{#if narrateError}<p class="ai-unavailable">{narrateError}</p>{/if}
					<footer><span>MODEL / {recommendation.model}</span><span>{recommendation.disclaimer}</span></footer>
				{:else}
					<p class="ai-unavailable">AI summary unavailable right now ({recommendation.error}). The quantitative forecast and TreeSHAP attributions below are unaffected.</p>
				{/if}
			{:else}
				<p class="ai-prompt">Click "Generate summary" for an AI-written research note synthesizing the forecasts and their statistical drivers below. Non-causal, non-diagnostic — a narrated readout, not a recommendation to act.</p>
			{/if}
		</section>

		<div class="metrics">
			<MetricTile label="Current SOFA" value={prediction.current_sofa.toFixed(1)} />
			<MetricTile label="Support risk (calibrated)" value={(prediction.organ_support.probability_24h * 100).toFixed(1)} unit="%" tone={prediction.organ_support.alert ? 'rose' : 'teal'} detail={prediction.organ_support.alert ? 'ABOVE THRESHOLD' : 'BELOW THRESHOLD'} />
			<MetricTile label="Additivity checks" value={Object.values(prediction.explanations).every((e) => e.diagnostics.additivity_check_passed) ? 'PASS' : 'FAIL'} detail="ALL 4 TASKS" tone="cyan" />
		</div>

		<div class="grid two attribution-overview">
			<Panel eyebrow="CROSS-TASK / MAX |SHAP|" title="Which task leans hardest on its top factor" note="NORMALIZED 0–100">
				<div class="radar-row">
					<RadarPlot values={crossTaskRadar.values} labels={crossTaskRadar.labels} color="#2bb8b0" />
				</div>
			</Panel>
			<Panel eyebrow="TEMPORAL WINDOW / QUALITY" title="Observation completeness at this cutoff" note={`${prediction.data_quality.observed_bins}/${prediction.data_quality.total_bins} BINS`}>
				<div class="dq-gauge-row">
					<GaugeRing value={Math.round((prediction.data_quality.observed_feature_fraction ?? 0) * 100)} size={92} stroke={9} color="#38bdf8" label="OBSERVED" />
					<dl>
						<div><dt>Observed feature values</dt><dd>{prediction.data_quality.observed_feature_values} / {prediction.data_quality.total_feature_values}</dd></div>
						<div><dt>Missing</dt><dd>{prediction.data_quality.missing_feature_values}</dd></div>
						<div><dt>Pre-admission padding bins</dt><dd>{prediction.data_quality.padding_bins}</dd></div>
					</dl>
				</div>
			</Panel>
		</div>

		<ShapPanel {prediction} computed={ctx.computed} cutoffIndex={ctx.cutoffIndex} bind:task={shapTask} />

		<div class="grid two overview-grid">
			{#each Object.entries(prediction.explanations) as [task, explanation]}
				<Panel eyebrow={`TREESHAP / ${task.toUpperCase()}`} title={TASK_TITLES[task]} note={explanation.diagnostics.additivity_check_passed ? 'ADDITIVITY OK' : 'ADDITIVITY FAILED'}>
					<div class="shap-cols">
						<div><span class="col-label pos">Increased the output</span><BarContrib features={contribFeatures(explanation.top_positive_contributors)} maxAbs={Math.max(0.05, ...explanation.top_positive_contributors.map((i) => Math.abs(i.attribution)), ...explanation.top_negative_contributors.map((i) => Math.abs(i.attribution)))} height={160} /></div>
						<div><span class="col-label neg">Decreased the output</span><BarContrib features={contribFeatures(explanation.top_negative_contributors)} maxAbs={Math.max(0.05, ...explanation.top_positive_contributors.map((i) => Math.abs(i.attribution)), ...explanation.top_negative_contributors.map((i) => Math.abs(i.attribution)))} height={160} /></div>
					</div>
					{#if task === 'organ_support'}<p class="note">Explains the raw model margin (log-odds), before Platt calibration — calibration is a separate monotone transform and is not decomposed here.</p>{/if}
				</Panel>
			{/each}
		</div>
		<p class="interpretation-note">Attributions "contributed to" this prediction; they never establish causation, and are never a treatment recommendation.</p>
	{/if}
</WorkbenchPage>

<style>
	.loading, .error { padding: 20px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.error { color: #fecdd3; }
	.controls { display: flex; gap: 8px; padding: 12px 0; border-block: 1px solid rgba(148,163,184,.14); margin-bottom: 16px; }
	.controls select { padding: 8px 12px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #cbd5e1; font: 9px 'JetBrains Mono', monospace; }
	.ai-panel { margin-bottom: 16px; padding: 18px; border: 1px solid rgba(43,184,176,.28); background: linear-gradient(135deg, rgba(43,184,176,.07), #080e1d 60%); }
	.ai-panel header { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; color: #2bb8b0; }
	.ai-panel header span { display: block; color: #2bb8b0; font: 7px 'JetBrains Mono', monospace; letter-spacing: .14em; }
	.ai-panel header h3 { margin: 4px 0 0; color: #eef7f6; font: 500 16px 'Space Grotesk', sans-serif; }
	.ai-panel header button { margin-left: auto; display: flex; align-items: center; gap: 7px; padding: 9px 14px; border: 1px solid #2bb8b0; background: #2bb8b0; color: #03110f; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; cursor: pointer; }
	.ai-panel header button:disabled { opacity: .6; cursor: default; }
	.ai-panel header button.voice-btn { margin-left: 0; border-color: rgba(43,184,176,.4); background: rgba(43,184,176,.07); color: #2bb8b0; }
	.ai-panel header button.voice-btn.speaking { background: #2bb8b0; color: #03110f; }
	.ai-panel header button :global(.spin) { animation: ai-spin .9s linear infinite; }
	@keyframes ai-spin { to { transform: rotate(360deg); } }
	.voice-indicator { display: flex; align-items: center; gap: 8px; margin: 14px 0 0; padding-top: 14px; border-top: 1px solid rgba(148,163,184,.12); color: #2bb8b0; font: 8px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.voice-indicator span { display: inline-block; width: 3px; height: 10px; border-radius: 2px; background: #2bb8b0; animation: ai-voicebar .9s ease-in-out infinite; }
	.voice-indicator span:nth-child(2) { animation-delay: .15s; }
	.voice-indicator span:nth-child(3) { animation-delay: .3s; }
	.voice-indicator span:nth-child(4) { animation-delay: .45s; }
	@keyframes ai-voicebar { 0%, 100% { transform: scaleY(.4); } 50% { transform: scaleY(1); } }
	.ai-text { margin: 16px 0 0; padding-top: 14px; border-top: 1px solid rgba(148,163,184,.12); color: #dce9e8; font-size: 13px; line-height: 1.75; }
	.ai-prompt, .ai-loading, .ai-unavailable { margin: 16px 0 0; padding-top: 14px; border-top: 1px solid rgba(148,163,184,.12); color: #71829a; font-size: 11px; line-height: 1.6; }
	.ai-unavailable { color: #fbbf24; }
	.ai-panel footer { display: flex; justify-content: space-between; gap: 16px; margin-top: 12px; color: #53647b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .04em; }
	.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px; }
	.grid.two { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
	.attribution-overview { margin-bottom: 8px; }
	.overview-grid { margin-top: 8px; }
	.radar-row { display: flex; justify-content: center; padding: 6px 0 2px; }
	.dq-gauge-row { display: flex; align-items: center; gap: 22px; }
	.dq-gauge-row dl { flex: 1; display: grid; gap: 10px; margin: 0; }
	.dq-gauge-row dl div { display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid rgba(148,163,184,.1); padding-bottom: 8px; }
	.dq-gauge-row dt { color: #71829a; font: 9px 'JetBrains Mono', monospace; letter-spacing: .04em; }
	.dq-gauge-row dd { margin: 0; color: #dce9e8; font: 600 12px 'Space Grotesk', sans-serif; }
	.shap-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
	.col-label { display: block; margin-bottom: 8px; font: 8px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.col-label.pos { color: #60a5fa; }
	.col-label.neg { color: #f87171; }
	.note { margin: 10px 0 0; padding-top: 10px; border-top: 1px solid rgba(148,163,184,.1); color: #71829a; font-size: 9px; line-height: 1.5; }
	.interpretation-note { margin-top: 8px; padding: 14px 18px; border: 1px solid rgba(148,163,184,.14); color: #71829a; font-size: 10px; }
	@media (max-width: 900px) {
		.metrics { grid-template-columns: 1fr 1fr; }
		.grid.two { grid-template-columns: 1fr; }
		.shap-cols { grid-template-columns: 1fr; }
	}
</style>
