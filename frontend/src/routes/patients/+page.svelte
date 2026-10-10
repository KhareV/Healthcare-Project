<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import MultiLine from '$lib/components/dashboard/MultiLine.svelte';
	import ModelBadge from '$lib/components/patient/ModelBadge.svelte';
	import EightBinWindow from '$lib/components/patient/EightBinWindow.svelte';
	import MultiCutoffForecast from '$lib/components/patient/MultiCutoffForecast.svelte';
	import ShapPanel from '$lib/components/patient/ShapPanel.svelte';
	import CompletenessPanel from '$lib/components/patient/CompletenessPanel.svelte';
	import SupportTimeline, { type SupportRun } from '$lib/components/patient/SupportTimeline.svelte';
	import { api, request, type PredictionRunSnapshot } from '$lib/services/api';
	import { authToken } from '$lib/stores/auth-token.svelte';
	import { openCopilot } from '$lib/stores/copilot.svelte';
	import { patientContext as ctx } from '$lib/stores/patient-context.svelte';
	import { ChevronLeft, ChevronRight, RotateCcw, Sparkles, Plus, Play, Pause } from '@lucide/svelte';

	const GROUP_OF: Record<string, string> = {
		mean_arterial_pressure: 'Cardiovascular', heart_rate: 'Cardiovascular', systolic_blood_pressure: 'Cardiovascular',
		diastolic_blood_pressure: 'Cardiovascular', vasopressor_on: 'Cardiovascular', norepinephrine_rate: 'Cardiovascular',
		epinephrine_rate: 'Cardiovascular', dopamine_rate: 'Cardiovascular', dobutamine_rate: 'Cardiovascular',
		pao2: 'Respiratory', fio2: 'Respiratory', respiratory_rate: 'Respiratory', oxygen_saturation: 'Respiratory', invasive_ventilation_on: 'Respiratory',
		creatinine: 'Renal', urine_output_volume: 'Renal',
		glasgow_coma_scale: 'Neurologic',
		platelet_count: 'Hematologic / Liver', bilirubin_total: 'Hematologic / Liver',
		temperature: 'Other', lactate: 'Other'
	};

	let task = $state<'recovery24' | 'recovery48' | 'icu_stay_time' | 'organ_support'>('organ_support');
	let supportRuns = $state<SupportRun[]>([]);
	let persisted = $state<PredictionRunSnapshot[]>([]);
	let persistedNote = $state('');

	const subject = $derived(ctx.subject);
	const prediction = $derived(ctx.prediction);
	const history = $derived(ctx.historyPredictions);
	const demoSubjects = $derived(ctx.subjects.filter((s) => s.kind === 'demo'));
	const showcaseSubjects = $derived(ctx.subjects.filter((s) => s.kind === 'showcase'));
	const customSubjects = $derived(ctx.subjects.filter((s) => s.kind === 'custom'));

	const groupedEvents = $derived.by(() => {
		const groups: Record<string, typeof ctx.events> = {};
		for (const event of ctx.events) {
			const group = GROUP_OF[event.canonical_concept] ?? 'Other';
			(groups[group] ??= []).push(event);
		}
		for (const key of Object.keys(groups)) groups[key] = groups[key].slice(-8);
		return groups;
	});
	const sofaLabels = $derived(history.map((p) => p.prediction_time.slice(11, 16)));
	const sofaValues = $derived(history.map((p) => p.current_sofa));

	function writeUrl() {
		void goto(ctx.urlQuery(), { replaceState: true, noScroll: true });
	}
	async function pick(id: string) {
		await ctx.select(id, 0);
		writeUrl();
		void loadSideData();
	}
	async function setCutoff(i: number) {
		await ctx.setCutoff(i);
		writeUrl();
		void loadSideData();
	}
	async function toggleHindsight() {
		ctx.showHindsight = !ctx.showHindsight;
		if (ctx.showHindsight) await ctx.loadAllCutoffs();
	}

	async function loadSideData() {
		supportRuns = [];
		persisted = [];
		persistedNote = '';
		const s = ctx.subject;
		if (!s || !ctx.prediction) return;
		const idx = ctx.cutoffIndex;
		try {
			if (s.kind === 'showcase' && s.scenario_id) {
				const t = await request<{ support_runs: { support_type: 'VASOPRESSOR' | 'RESPIRATORY'; category: string; qualifying_for_initiation: boolean; start_hour: number; end_hour: number; first_visible_cutoff: number; end_first_visible_cutoff: number }[] }>(`/showcase/scenarios/${s.scenario_id}/timeline`);
				supportRuns = t.support_runs
					.filter((r) => r.first_visible_cutoff <= idx)
					.map((r) => ({ label: r.category.toLowerCase(), domain: r.support_type, qualifying: r.qualifying_for_initiation, start_hour: r.start_hour, end_hour: r.end_first_visible_cutoff <= idx ? r.end_hour : null }));
			} else if (s.kind === 'custom') {
				const enc = (await api.healthRecord.listEncounters()).encounters.find((e) => e.stay_id === s.stay_id);
				supportRuns = (enc?.support_intervals ?? []).map((r) => ({
					label: r.kind === 'vasopressor' ? String((r as { agent?: string }).agent ?? 'vasopressor') : 'invasive ventilation',
					domain: r.kind === 'vasopressor' ? 'VASOPRESSOR' : 'RESPIRATORY', start_hour: r.start_hour, end_hour: r.end_hour
				}));
				const hist = await api.getCustomRecordPredictionHistory(s.stay_id);
				persisted = hist.predictions;
				persistedNote = hist.note ?? '';
			}
		} catch {
			/* side data is supplementary; the forecast itself already loaded */
		}
	}

	async function init() {
		await ctx.ensureSubjects(true);
		const { stay, cutoff } = ctx.parseUrl(page.url.searchParams);
		if (stay) await ctx.ensureCustom(stay);
		const target = stay && ctx.subjects.some((s) => s.stay_id === stay) ? stay : ctx.subjects[0]?.stay_id ?? '';
		if (target) await ctx.select(target, cutoff);
		await loadSideData();
	}

	onMount(() => {
		void init();
		return () => ctx.pause();
	});
	let lastToken: string | null | undefined = undefined;
	$effect(() => {
		const token = authToken.value;
		untrack(() => {
			if (lastToken !== undefined && token !== lastToken && ctx.ensureOwner()) void init();
			lastToken = token;
		});
	});
	const cutoffHour = $derived(prediction?.elapsed_icu_hours ?? 0);
</script>

<svelte:head><title>Patient Replay | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage eyebrow="01 / PATIENT REPLAY" title="Step through what the model knew, when it knew it." description="Every cutoff recomputes recovery, remaining ICU-stay time, and organ-support risk from raw history truncated at that moment — never a precomputed table.">
	{#if ctx.loadingSubjects}
		<div class="loading">Loading patients…</div>
	{:else if !subject}
		<div class="error">No patient available.</div>
	{:else}
		<div class="selector-row">
			<div class="stay-picker">
				<div class="picker-caption">DEMO PATIENTS</div>
				{#each demoSubjects as s}
					<button class:active={s.stay_id === ctx.stayId} onclick={() => pick(s.stay_id)}>
						<span class="sid">{s.patient_alias ?? s.subject_id}</span>
						<small>{s.cardiac_condition_group.replace('SYNTHETIC_', '')}</small>
					</button>
				{/each}
				{#if showcaseSubjects.length}
					<div class="picker-caption">SCENARIO CASES</div>
					{#each showcaseSubjects as s}
						<button class="showcase" class:active={s.stay_id === ctx.stayId} onclick={() => pick(s.stay_id)}>
							<span class="sid">{s.patient_alias} · {s.title}</span>
							<small>{s.cardiac_condition_group.replace('SYNTHETIC_', '')} · {s.n_legal_cutoffs} CUTOFFS</small>
						</button>
					{/each}
				{/if}
				{#if customSubjects.length}
					<div class="picker-caption">YOUR RECORDS</div>
					{#each customSubjects as s}
						<button class="custom" class:active={s.stay_id === ctx.stayId} onclick={() => pick(s.stay_id)}>
							<span class="sid">{s.patient_alias ?? s.subject_id}</span>
							<small>YOUR RECORD</small>
						</button>
					{/each}
				{/if}
				<a class="stay-picker-add" href="/patients/custom"><Plus size={14} /> Enter my own record</a>
			</div>
			<div class="overview-card">
				<div><span>Patient</span><b>{subject.patient_alias ?? subject.subject_id}</b></div>
				<div><span>Age / Sex</span><b>{subject.age_years}y · {subject.sex_category.replace('SYNTHETIC_', '')}</b></div>
				<div><span>ICU admission</span><b>{subject.intime ? subject.intime.slice(0, 16).replace('T', ' ') : '—'}</b></div>
				<div><span>Selected cutoff</span><b>{ctx.cutoff?.slice(0, 16).replace('T', ' ')}</b></div>
				<div><span>Elapsed ICU hours</span><b>{prediction?.elapsed_icu_hours ?? '—'} h</b></div>
				{#if subject.kind === 'demo'}<div><span>Cohort</span><b>training-cohort demo</b></div>{/if}
				{#if subject.kind === 'showcase'}<div><span>Cohort</span><b>held-out development</b></div>{/if}
			</div>
		</div>

		<div class="toolbar-row"><ModelBadge bundle={prediction?.model_bundle} versions={prediction?.versions} /></div>

		<div class="replay-controls">
			<button onclick={() => setCutoff(ctx.cutoffIndex - 1)} disabled={ctx.cutoffIndex === 0}><ChevronLeft size={14} /> Previous</button>
			<select value={ctx.cutoffIndex} onchange={(e) => setCutoff(Number((e.target as HTMLSelectElement).value))} aria-label="Legal prediction cutoff">
				{#each ctx.cutoffs as c, i}<option value={i}>t{i} — {c.slice(0, 16).replace('T', ' ')}</option>{/each}
			</select>
			<button onclick={() => setCutoff(ctx.cutoffIndex + 1)} disabled={ctx.cutoffIndex === ctx.cutoffs.length - 1}>Next <ChevronRight size={14} /></button>
			<button onclick={() => setCutoff(0)}><RotateCcw size={14} /> Reset</button>
			<button class:playing={ctx.replaying} onclick={() => (ctx.replaying ? ctx.pause() : ctx.play())} disabled={ctx.cutoffs.length < 2} aria-pressed={ctx.replaying}>
				{#if ctx.replaying}<Pause size={14} /> Pause{:else}<Play size={14} /> Play{/if}
			</button>
			<select class="speed" bind:value={ctx.replaySpeed} aria-label="Replay speed"><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option></select>
			<button
				class="ask-copilot"
				onclick={() => openCopilot({ stayId: ctx.stayId, patientAlias: subject?.patient_alias ?? ctx.stayId, predictionTime: ctx.cutoff ?? '', previousPredictionTime: ctx.cutoffIndex > 0 ? ctx.cutoffs[ctx.cutoffIndex - 1] : null })}
			><Sparkles size={14} /> Ask Copilot</button>
			<span class="cutoff-count">cutoff {ctx.cutoffIndex + 1} / {ctx.cutoffs.length}</span>
		</div>

		{#if ctx.error}<div class="error">{ctx.error}</div>{/if}

		{#if ctx.loading && !prediction}
			<div class="loading">Recomputing from truncated history…</div>
		{:else if prediction}
			<div class="metrics">
				<MetricTile label="Current SOFA" value={prediction.current_sofa.toFixed(1)} detail="Scored at cutoff" values={sofaValues} />
				<MetricTile label="Predicted SOFA +24h" value={prediction.recovery.sofa_hat_24h.toFixed(1)} detail={`Δ ${prediction.recovery.delta_24h >= 0 ? '+' : ''}${prediction.recovery.delta_24h.toFixed(2)}`} tone="cyan" values={history.map((p) => p.recovery.sofa_hat_24h)} />
				<MetricTile label="Predicted SOFA +48h" value={prediction.recovery.sofa_hat_48h.toFixed(1)} detail={`Δ ${prediction.recovery.delta_48h >= 0 ? '+' : ''}${prediction.recovery.delta_48h.toFixed(2)}`} tone="cyan" values={history.map((p) => p.recovery.sofa_hat_48h)} />
				<MetricTile label="Remaining ICU stay time" value={prediction.icu_stay_time.remaining_hours.toFixed(1)} unit="h" detail={`≈ ${Math.floor(prediction.icu_stay_time.remaining_hours / 24)}d ${(prediction.icu_stay_time.remaining_hours % 24).toFixed(0)}h`} tone="amber" values={history.map((p) => p.icu_stay_time.remaining_hours)} />
				<MetricTile
					label="New organ-support risk (24h)"
					value={(prediction.organ_support.probability_24h * 100).toFixed(1)}
					unit="%"
					detail={prediction.organ_support.alert ? `ABOVE THRESHOLD ${(prediction.organ_support.threshold * 100).toFixed(1)}%` : `BELOW THRESHOLD ${(prediction.organ_support.threshold * 100).toFixed(1)}%`}
					tone={prediction.organ_support.alert ? 'rose' : 'teal'}
					values={history.map((p) => p.organ_support.probability_24h * 100)}
				/>
			</div>

			<Panel eyebrow="RECOVERY / TRAJECTORY" title="Observed SOFA vs. independent +24h / +48h forecast" note="RECONSTRUCTED, DISPLAY CLIPPED 0–24">
				<MultiLine
					series={[
						{ name: 'Observed SOFA (≤ cutoff)', color: '#2bb8b0', values: [...sofaValues, NaN, NaN] },
						{ name: 'Forecast +24h/+48h', color: '#fb7185', values: [...sofaValues.slice(0, -1).map(() => NaN), sofaValues[sofaValues.length - 1], prediction.recovery.sofa_hat_24h, prediction.recovery.sofa_hat_48h] }
					]}
					xLabels={[...sofaLabels, '+24h', '+48h']}
					yMin={0}
					yMax={24}
					yLabel="SOFA"
					height={220}
				/>
				<p class="note">The 48h forecast is never chained through the 24h forecast — both are computed independently from the same cutoff baseline.</p>
			</Panel>

			<div class="stack">
				<MultiCutoffForecast computed={ctx.computed} cutoffIndex={ctx.cutoffIndex} showHindsight={ctx.showHindsight} onToggleHindsight={toggleHindsight} />

				<Panel eyebrow="MODEL INPUT / TEMPORAL WINDOW" title="48-hour lookback — eight 6-hour bins" note={`${prediction.data_quality.observed_bins}/${prediction.data_quality.total_bins} BINS OBSERVED`}>
					<EightBinWindow channelNames={prediction.temporal_window.channel_names} observationMask={prediction.temporal_window.observation_mask} paddingMask={prediction.temporal_window.padding_mask} events={ctx.events} cutoff={prediction.prediction_time} />
				</Panel>

				<div class="grid two">
					<Panel eyebrow="ORGAN SUPPORT / STATE" title="Support-state timeline up to the cutoff" note="NO FUTURE INTERVALS SHOWN">
						<SupportTimeline runs={supportRuns} {cutoffHour} />
					</Panel>
					<CompletenessPanel {prediction} />
				</div>

				<ShapPanel {prediction} computed={ctx.computed} cutoffIndex={ctx.cutoffIndex} bind:task />

				<Panel eyebrow="HISTORICAL TIMELINE" title="Observed variables at or before the selected cutoff" note={`${ctx.events.length} EVENTS VISIBLE`}>
					<div class="timeline-groups">
						{#each Object.entries(groupedEvents) as [group, rows]}
							<div class="tgroup">
								<h4>{group}</h4>
								<table>
									<tbody>
										{#each rows as row}
											<tr><td>{row.event_time.slice(11, 19)}</td><td>{row.canonical_concept.replace(/_/g, ' ')}</td><td>{row.value_numeric.toFixed(2)} {row.unit}</td></tr>
										{/each}
									</tbody>
								</table>
							</div>
						{/each}
					</div>
				</Panel>

				<Panel eyebrow="REPLAY HISTORY" title="How predictions evolved as replay advanced" note="EVERY ROW IS A REAL API CALL — MODEL VERSION SHOWN">
					<table class="history-table">
						<thead><tr><th>Cutoff</th><th>Δ24</th><th>Δ48</th><th>ICU remaining</th><th>Support risk</th><th>Alert</th><th>Model</th></tr></thead>
						<tbody>
							{#each history as p, i}
								<tr class:current={i === history.length - 1}>
									<td>t{i}</td>
									<td>{p.recovery.delta_24h >= 0 ? '+' : ''}{p.recovery.delta_24h.toFixed(2)}</td>
									<td>{p.recovery.delta_48h >= 0 ? '+' : ''}{p.recovery.delta_48h.toFixed(2)}</td>
									<td>{p.icu_stay_time.remaining_hours.toFixed(1)}h</td>
									<td>{(p.organ_support.probability_24h * 100).toFixed(1)}%</td>
									<td>{p.organ_support.alert ? 'ABOVE' : 'below'}</td>
									<td><span class="version-pill v3">{p.model_bundle?.release ?? 'V3'}</span></td>
								</tr>
							{/each}
						</tbody>
					</table>
					{#if subject.kind === 'custom'}
						<h4 class="persisted-title">Stored prediction history for this record</h4>
						{#if persisted.length}
							<table class="history-table">
								<thead><tr><th>Cutoff</th><th>SOFA +24h</th><th>ICU remaining</th><th>Support risk</th><th>Stored under</th><th>State</th></tr></thead>
								<tbody>
									{#each persisted as r}
										<tr>
											<td>{r.prediction_time.slice(0, 16).replace('T', ' ')}</td>
											<td>{r.predicted_sofa_24h.toFixed(1)}</td>
											<td>{r.remaining_icu_hours.toFixed(1)}h</td>
											<td>{(r.support_calibrated_probability * 100).toFixed(1)}%</td>
											<td><span class="version-pill" class:v3={r.release === 'V3'} class:legacy={r.legacy}>{r.release ?? (r.legacy ? 'V2' : '—')}{r.bundle_id ? ` · ${r.bundle_id}` : ''}</span></td>
											<td>{#if r.stale}<span class="version-pill stale">STALE · DATA CHANGED</span>{:else if r.legacy}<span class="version-pill legacy">HISTORICAL V2</span>{:else}current{/if}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						{:else}
							<p class="hist-note">{persistedNote || 'No stored forecast yet for this record.'}</p>
						{/if}
					{/if}
				</Panel>
			</div>
		{/if}
	{/if}
</WorkbenchPage>

<style>
	.loading, .error { padding: 20px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.error { color: #fecdd3; }
	.selector-row { display: grid; grid-template-columns: 1fr 280px; gap: 8px; margin-bottom: 16px; }
	.stay-picker { display: flex; flex-direction: column; gap: 4px; }
	.stay-picker button { display: flex; justify-content: space-between; align-items: center; padding: 12px 14px; border: 1px solid rgba(148,163,184,.16); background: #080e1d; color: #94a3b8; cursor: pointer; font: 11px 'Space Grotesk', sans-serif; text-align: left; }
	.stay-picker button.active { border-color: rgba(43,184,176,.45); background: rgba(43,184,176,.07); color: #eef7f6; }
	.stay-picker button.custom { border-style: dashed; }
	.stay-picker button.custom small { color: #38bdf8; }
	.stay-picker small { color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.stay-picker-add { display: flex; align-items: center; justify-content: center; gap: 7px; margin-top: 4px; padding: 12px 14px; border: 1px dashed rgba(148,163,184,.28); color: #71829a; text-decoration: none; font: 10px 'JetBrains Mono', monospace; letter-spacing: .06em; }
	.stay-picker-add:hover { border-color: #2bb8b0; color: #2bb8b0; }
	.overview-card { display: grid; gap: 8px; padding: 14px; border-left: 2px solid #2bb8b0; background: rgba(15,23,42,.5); }
	.overview-card div { display: flex; justify-content: space-between; gap: 10px; }
	.overview-card span { color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.overview-card b { color: #dce9e8; font: 10px 'Space Grotesk', sans-serif; text-align: right; }
	.replay-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 12px 0; border-block: 1px solid rgba(148,163,184,.14); margin-bottom: 16px; }
	.replay-controls button, .replay-controls select { display: flex; align-items: center; gap: 6px; padding: 8px 12px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #cbd5e1; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; cursor: pointer; }
	.replay-controls button:disabled { opacity: .35; cursor: default; }
	.replay-controls button:not(:disabled):hover { border-color: #2bb8b0; color: #2bb8b0; }
	.cutoff-count { margin-left: auto; color: #64748b; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.ask-copilot { border-color: rgba(56,189,248,.35) !important; color: #38bdf8 !important; }
	.ask-copilot:hover { border-color: #38bdf8 !important; }
	.metrics { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 8px; }
	.grid.two { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
	.note { margin: 10px 0 0; color: #64748b; font-size: 10px; line-height: 1.6; }
	.timeline-groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
	.tgroup h4 { margin: 0 0 8px; color: #2bb8b0; font: 9px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; }
	.tgroup table, .history-table { width: 100%; border-collapse: collapse; font-size: 10px; }
	.tgroup td { padding: 3px 4px; color: #94a3b8; border-bottom: 1px solid rgba(148,163,184,.08); }
	.history-table th { text-align: left; padding: 6px 8px; color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; border-bottom: 1px solid rgba(148,163,184,.16); }
	.history-table td { padding: 6px 8px; color: #cbd5e1; border-bottom: 1px solid rgba(148,163,184,.08); font: 10px 'JetBrains Mono', monospace; }
	.history-table tr.current td { color: #2bb8b0; }
	@media (max-width: 900px) {
		.selector-row { grid-template-columns: 1fr; }
		.metrics { grid-template-columns: repeat(2, 1fr); }
		.grid.two { grid-template-columns: 1fr; }
	}

	.picker-caption { margin: 8px 0 2px; color: #53647b; font: 7px 'JetBrains Mono', monospace; letter-spacing: .14em; }
	.stay-picker button.showcase small { color: #fbbf24; }
	.toolbar-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
	.speed { width: 70px; }
	.replay-controls button.playing { border-color: #2bb8b0; color: #2bb8b0; }
	.version-pill { padding: 2px 6px; border: 1px solid rgba(148,163,184,.25); color: #94a3b8; font: 7px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.version-pill.v3 { border-color: rgba(43,184,176,.5); color: #2bb8b0; }
	.version-pill.legacy { border-color: rgba(251,191,36,.5); color: #fbbf24; }
	.version-pill.stale { border-color: rgba(251,113,133,.5); color: #fb7185; }
	.stack { display: grid; gap: 8px; margin-top: 8px; }
	.hist-note { margin: 10px 0 0; color: #64748b; font-size: 10px; line-height: 1.6; }
	.persisted-title { margin: 18px 0 8px; color: #2bb8b0; font: 9px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; }
</style>
