<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import BarContrib from '$lib/components/dashboard/BarContrib.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import { api, type AIRecommendation, type DemoSubject, type PredictionResponse } from '$lib/services/api';
	import { Sparkles, BrainCircuit } from '@lucide/svelte';

	const TASK_TITLES: Record<string, string> = {
		recovery24: 'Recovery +24h', recovery48: 'Recovery +48h', icu_stay_time: 'Remaining ICU stay time', organ_support: 'Organ support (raw margin)'
	};

	let subjects = $state<DemoSubject[]>([]);
	let stayId = $state('');
	let cutoffIndex = $state(0);
	let prediction = $state<PredictionResponse | null>(null);
	let recommendation = $state<AIRecommendation | null>(null);
	let loadingPrediction = $state(true);
	let loadingAI = $state(false);
	let error = $state('');

	const subject = $derived(subjects.find((s) => s.stay_id === stayId) ?? null);
	const cutoffs = $derived(subject?.legal_cutoffs ?? []);

	function contribFeatures(items: { label: string; attribution: number }[]) {
		return items.map((i) => ({ name: i.label, value: i.attribution }));
	}

	async function load() {
		if (!subject) return;
		loadingPrediction = true;
		error = '';
		try {
			prediction = await api.predict(stayId, cutoffs[cutoffIndex]);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Prediction failed';
			prediction = null;
		} finally {
			loadingPrediction = false;
		}
	}

	async function askAI() {
		if (!subject) return;
		loadingAI = true;
		recommendation = null;
		try {
			recommendation = await api.aiRecommendation(stayId, cutoffs[cutoffIndex]);
		} catch (cause) {
			recommendation = { status: 'UNAVAILABLE', summary: null, model: 'unknown', disclaimer: '', error: cause instanceof Error ? cause.message : 'request failed' };
		} finally {
			loadingAI = false;
		}
	}

	function selectStay(id: string) {
		stayId = id;
		cutoffIndex = 0;
		recommendation = null;
		goto(`?stay_id=${id}&cutoff=0`, { replaceState: true, noScroll: true });
		void load();
	}
	function setCutoff(index: number) {
		cutoffIndex = Math.max(0, Math.min(cutoffs.length - 1, index));
		recommendation = null;
		goto(`?stay_id=${stayId}&cutoff=${cutoffIndex}`, { replaceState: true, noScroll: true });
		void load();
	}

	onMount(async () => {
		try {
			const result = await api.demoSubjects();
			subjects = result.demo_subjects;
			const params = page.url.searchParams;
			const requested = params.get('stay_id');
			stayId = requested && subjects.some((s) => s.stay_id === requested) ? requested : subjects[0]?.stay_id ?? '';
			cutoffIndex = Number(params.get('cutoff') ?? 0) || 0;
			await load();
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not load demo subjects';
		}
	});
</script>

<svelte:head><title>AI Insight + Explainability | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage eyebrow="04 / AI INSIGHT + EXPLAINABILITY" title="Make the forecast legible, then let a model narrate it." description="TreeSHAP decomposes each frozen XGBoost output feature-by-feature; a separate LLM (Groq / Llama-family inference) turns those numbers into a plain-language research note — generated only after prediction, never influencing it.">
	{#if error}<div class="error">{error}</div>{/if}
	{#if subjects.length}
		<div class="controls">
			<select value={stayId} onchange={(e) => selectStay((e.target as HTMLSelectElement).value)}>
				{#each subjects as s}<option value={s.stay_id}>{s.subject_id} — {s.cardiac_condition_group.replace('SYNTHETIC_', '')}</option>{/each}
			</select>
			<select value={cutoffIndex} onchange={(e) => setCutoff(Number((e.target as HTMLSelectElement).value))}>
				{#each cutoffs as c, i}<option value={i}>t{i} — {c.slice(0, 16).replace('T', ' ')}</option>{/each}
			</select>
		</div>
	{/if}

	{#if loadingPrediction}
		<div class="loading">Recomputing prediction and TreeSHAP attributions…</div>
	{:else if prediction}
		<section class="ai-panel">
			<header><Sparkles size={16} /><div><span>AI RESEARCH NOTE</span><h3>Generated interpretive summary</h3></div><button onclick={askAI} disabled={loadingAI}><BrainCircuit size={14} /> {loadingAI ? 'Thinking…' : recommendation ? 'Regenerate' : 'Generate summary'}</button></header>
			{#if loadingAI}
				<p class="ai-loading">Calling the language model with this prediction's numbers…</p>
			{:else if recommendation}
				{#if recommendation.status === 'OK'}
					<p class="ai-text">{recommendation.summary}</p>
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

		<div class="grid two">
			{#each Object.entries(prediction.explanations) as [task, explanation]}
				<Panel eyebrow={`TREESHAP / ${task.toUpperCase()}`} title={TASK_TITLES[task]} note={explanation.diagnostics.additivity_check_passed ? 'ADDITIVITY OK' : 'ADDITIVITY FAILED'}>
					<div class="shap-cols">
						<div><span class="col-label pos">Increased the output</span><BarContrib features={contribFeatures(explanation.top_positive_contributors)} maxAbs={Math.max(0.05, ...explanation.top_positive_contributors.map((i) => Math.abs(i.attribution)), ...explanation.top_negative_contributors.map((i) => Math.abs(i.attribution)))} height={160} /></div>
						<div><span class="col-label neg">Decreased the output</span><BarContrib features={contribFeatures(explanation.top_negative_contributors)} maxAbs={Math.max(0.05, ...explanation.top_positive_contributors.map((i) => Math.abs(i.attribution)), ...explanation.top_negative_contributors.map((i) => Math.abs(i.attribution)))} height={160} /></div>
					</div>
					{#if task === 'organ_support'}<p class="note">Explains the raw XGBoost margin, before isotonic calibration — calibration is a non-additive transform and is not decomposed here.</p>{/if}
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
	.ai-panel header { display: flex; align-items: center; gap: 12px; color: #2bb8b0; }
	.ai-panel header span { display: block; color: #2bb8b0; font: 7px 'JetBrains Mono', monospace; letter-spacing: .14em; }
	.ai-panel header h3 { margin: 4px 0 0; color: #eef7f6; font: 500 16px 'Space Grotesk', sans-serif; }
	.ai-panel header button { margin-left: auto; display: flex; align-items: center; gap: 7px; padding: 9px 14px; border: 1px solid #2bb8b0; background: #2bb8b0; color: #03110f; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; cursor: pointer; }
	.ai-panel header button:disabled { opacity: .6; cursor: default; }
	.ai-text { margin: 16px 0 0; padding-top: 14px; border-top: 1px solid rgba(148,163,184,.12); color: #dce9e8; font-size: 13px; line-height: 1.75; }
	.ai-prompt, .ai-loading, .ai-unavailable { margin: 16px 0 0; padding-top: 14px; border-top: 1px solid rgba(148,163,184,.12); color: #71829a; font-size: 11px; line-height: 1.6; }
	.ai-unavailable { color: #fbbf24; }
	.ai-panel footer { display: flex; justify-content: space-between; gap: 16px; margin-top: 12px; color: #53647b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .04em; }
	.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px; }
	.grid.two { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
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
