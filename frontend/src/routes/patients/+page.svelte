<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import MultiLine from '$lib/components/dashboard/MultiLine.svelte';
	import TemporalHeatmap from '$lib/components/dashboard/TemporalHeatmap.svelte';
	import GaugeRing from '$lib/components/dashboard/GaugeRing.svelte';
	import { api, type DemoSubject, type PredictionResponse } from '$lib/services/api';
	import { setCopilotContext, openCopilot } from '$lib/stores/copilot.svelte';
	import { ChevronLeft, ChevronRight, RotateCcw, Sparkles, Plus } from '@lucide/svelte';

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

	let subjects = $state<DemoSubject[]>([]);
	let loadingSubjects = $state(true);
	let stayId = $state('');
	let cutoffIndex = $state(0);
	let prediction = $state<PredictionResponse | null>(null);
	let historyPredictions = $state<PredictionResponse[]>([]);
	let sofaHistoryLabels = $state<string[]>([]);
	let sofaHistoryValues = $state<number[]>([]);
	let events = $state<{ event_time: string; canonical_concept: string; value_numeric: number; unit: string }[]>([]);
	let loading = $state(false);
	let errorMessage = $state('');
	const cache = new Map<string, PredictionResponse>();

	const subject = $derived(subjects.find((s) => s.stay_id === stayId) ?? null);
	const cutoffs = $derived(subject?.legal_cutoffs ?? []);

	async function fetchPrediction(sid: string, t: string): Promise<PredictionResponse> {
		const key = `${sid}|${t}`;
		const cached = cache.get(key);
		if (cached) return cached;
		const result = await api.predict(sid, t);
		cache.set(key, result);
		return result;
	}

	async function loadCutoff() {
		if (!subject) return;
		loading = true;
		errorMessage = '';
		const t = cutoffs[cutoffIndex];
		try {
			prediction = await fetchPrediction(stayId, t);
			const upTo = cutoffs.slice(0, cutoffIndex + 1);
			historyPredictions = await Promise.all(upTo.map((c) => fetchPrediction(stayId, c)));
			sofaHistoryLabels = upTo.map((c) => c.slice(11, 16));
			sofaHistoryValues = historyPredictions.map((p) => p.current_sofa);
			const historyResult = await api.history(stayId, t);
			events = historyResult.events;
			setCopilotContext({
				stayId,
				patientAlias: subject?.patient_alias ?? stayId,
				predictionTime: t,
				previousPredictionTime: cutoffIndex > 0 ? cutoffs[cutoffIndex - 1] : null
			});
		} catch (cause) {
			errorMessage = cause instanceof Error ? cause.message : 'Prediction request failed';
			prediction = null;
		} finally {
			loading = false;
		}
	}

	function selectStay(id: string) {
		stayId = id;
		cutoffIndex = 0;
		goto(`?stay_id=${id}&cutoff=0`, { replaceState: true, noScroll: true });
		void loadCutoff();
	}
	function setCutoff(index: number) {
		cutoffIndex = Math.max(0, Math.min(cutoffs.length - 1, index));
		goto(`?stay_id=${stayId}&cutoff=${cutoffIndex}`, { replaceState: true, noScroll: true });
		void loadCutoff();
	}

	const groupedEvents = $derived.by(() => {
		const groups: Record<string, typeof events> = {};
		for (const event of events) {
			const group = GROUP_OF[event.canonical_concept] ?? 'Other';
			(groups[group] ??= []).push(event);
		}
		for (const key of Object.keys(groups)) groups[key] = groups[key].slice(-8);
		return groups;
	});

	onMount(async () => {
		try {
			const result = await api.demoSubjects();
			subjects = result.demo_subjects;

			// Merge in every custom ("bring your own data") record owned by
			// the signed-in caller, via the real backend index (persisted in
			// MongoDB when configured — see /health's persistence_mode — so
			// this list survives an API restart; otherwise whatever this
			// process currently has loaded in memory, the pre-persistence
			// behavior). No client-side list to keep in sync any more.
			try {
				const { records } = await api.listCustomRecords();
				const fetched = await Promise.allSettled(records.map((r) => api.getCustomRecord(r.stay_id)));
				fetched.forEach((outcome) => {
					if (outcome.status === 'fulfilled') subjects = [...subjects, outcome.value];
				});
			} catch {
				/* not signed in yet, or persistence temporarily unreachable — demo cohort still loads */
			}

			const params = page.url.searchParams;
			const requested = params.get('stay_id');
			if (requested && requested.startsWith('CUSTOM-') && !subjects.some((s) => s.stay_id === requested)) {
				try {
					const record = await api.getCustomRecord(requested);
					subjects = [...subjects, record];
				} catch {
					/* not found (e.g. never persisted and the server that held it in memory has since restarted) — falls through below */
				}
			}

			stayId = requested && subjects.some((s) => s.stay_id === requested) ? requested : subjects[0]?.stay_id ?? '';
			cutoffIndex = Number(params.get('cutoff') ?? 0) || 0;
			loadingSubjects = false;
			await loadCutoff();
		} catch (cause) {
			loadingSubjects = false;
			errorMessage = cause instanceof Error ? cause.message : 'Could not load demo subjects';
		}
	});
</script>

<svelte:head><title>Patient Replay | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage eyebrow="01 / PATIENT REPLAY" title="Step through what the model knew, when it knew it." description="Every cutoff recomputes recovery, remaining ICU-stay time, and organ-support risk from raw history truncated at that moment — never a precomputed table.">
	{#if loadingSubjects}
		<div class="loading">Loading demo cohort…</div>
	{:else if !subject}
		<div class="error">No demo subject available.</div>
	{:else}
		<div class="selector-row">
			<div class="stay-picker">
				{#each subjects as s}
					<button class:active={s.stay_id === stayId} class:custom={s.cardiac_condition_group === 'CUSTOM_RECORD'} onclick={() => selectStay(s.stay_id)}>
						<span class="sid">{s.patient_alias ?? s.subject_id}</span>
						<small>{s.cardiac_condition_group === 'CUSTOM_RECORD' ? 'YOUR RECORD' : s.cardiac_condition_group.replace('SYNTHETIC_', '')}</small>
					</button>
				{/each}
				<a class="stay-picker-add" href="/patients/custom"><Plus size={14} /> Enter my own record</a>
			</div>
			<div class="overview-card">
				<div><span>Patient</span><b>{subject.patient_alias ?? subject.subject_id}</b></div>
				<div><span>Age / Sex</span><b>{subject.age_years}y · {subject.sex_category.replace('SYNTHETIC_', '')}</b></div>
				<div><span>ICU admission</span><b>{subject.intime.slice(0, 16).replace('T', ' ')}</b></div>
				<div><span>Selected cutoff</span><b>{cutoffs[cutoffIndex]?.slice(0, 16).replace('T', ' ')}</b></div>
				<div><span>Elapsed ICU hours</span><b>{prediction?.elapsed_icu_hours ?? '—'} h</b></div>
			</div>
		</div>

		<div class="replay-controls">
			<button onclick={() => setCutoff(cutoffIndex - 1)} disabled={cutoffIndex === 0}><ChevronLeft size={14} /> Previous</button>
			<select value={cutoffIndex} onchange={(e) => setCutoff(Number((e.target as HTMLSelectElement).value))}>
				{#each cutoffs as c, i}<option value={i}>t{i} — {c.slice(0, 16).replace('T', ' ')}</option>{/each}
			</select>
			<button onclick={() => setCutoff(cutoffIndex + 1)} disabled={cutoffIndex === cutoffs.length - 1}>Next <ChevronRight size={14} /></button>
			<button onclick={() => setCutoff(0)}><RotateCcw size={14} /> Reset</button>
			<button
				class="ask-copilot"
				onclick={() =>
					openCopilot({
						stayId,
						patientAlias: subject?.patient_alias ?? stayId,
						predictionTime: cutoffs[cutoffIndex],
						previousPredictionTime: cutoffIndex > 0 ? cutoffs[cutoffIndex - 1] : null
					})}
			><Sparkles size={14} /> Ask Copilot</button>
			<span class="cutoff-count">cutoff {cutoffIndex + 1} / {cutoffs.length}</span>
		</div>

		{#if errorMessage}<div class="error">{errorMessage}</div>{/if}

		{#if loading && !prediction}
			<div class="loading">Recomputing from truncated history…</div>
		{:else if prediction}
			<div class="metrics">
				<MetricTile label="Current SOFA" value={prediction.current_sofa.toFixed(1)} detail="Observed at cutoff" values={sofaHistoryValues} />
				<MetricTile label="Predicted SOFA +24h" value={prediction.recovery.sofa_hat_24h.toFixed(1)} detail={`Δ ${prediction.recovery.delta_24h >= 0 ? '+' : ''}${prediction.recovery.delta_24h.toFixed(2)}`} tone="cyan" values={historyPredictions.map((p) => p.recovery.sofa_hat_24h)} />
				<MetricTile label="Predicted SOFA +48h" value={prediction.recovery.sofa_hat_48h.toFixed(1)} detail={`Δ ${prediction.recovery.delta_48h >= 0 ? '+' : ''}${prediction.recovery.delta_48h.toFixed(2)}`} tone="cyan" values={historyPredictions.map((p) => p.recovery.sofa_hat_48h)} />
				<MetricTile label="Remaining ICU stay time" value={prediction.icu_stay_time.remaining_hours.toFixed(1)} unit="h" detail={`≈ ${Math.floor(prediction.icu_stay_time.remaining_hours / 24)}d ${(prediction.icu_stay_time.remaining_hours % 24).toFixed(0)}h`} tone="amber" values={historyPredictions.map((p) => p.icu_stay_time.remaining_hours)} />
				<MetricTile
					label="New organ-support risk (24h)"
					value={(prediction.organ_support.probability_24h * 100).toFixed(1)}
					unit="%"
					detail={prediction.organ_support.alert ? 'ABOVE THRESHOLD' : 'BELOW THRESHOLD'}
					tone={prediction.organ_support.alert ? 'rose' : 'teal'}
					values={historyPredictions.map((p) => p.organ_support.probability_24h * 100)}
				/>
			</div>

			<Panel eyebrow="RECOVERY / TRAJECTORY" title="Observed SOFA vs. independent +24h / +48h forecast" note="RECONSTRUCTED, CLIPPED 0–24">
				<MultiLine
					series={[
						{ name: 'Observed SOFA (≤ cutoff)', color: '#2bb8b0', values: [...sofaHistoryValues, NaN, NaN] },
						{ name: 'Forecast +24h/+48h', color: '#fb7185', values: [...sofaHistoryValues.slice(0, -1).map(() => NaN), sofaHistoryValues[sofaHistoryValues.length - 1], prediction.recovery.sofa_hat_24h, prediction.recovery.sofa_hat_48h] }
					]}
					xLabels={[...sofaHistoryLabels, '+24h', '+48h']}
					yMin={0}
					yMax={24}
					yLabel="SOFA"
					height={220}
				/>
				<p class="note">The 48h forecast is never chained through the 24h forecast — both are computed independently from the same cutoff baseline.</p>
			</Panel>

			<div class="grid two">
				<Panel eyebrow="ICU / TREND" title="Remaining stay-time trend" note="ACROSS REPLAY SO FAR">
					{#if historyPredictions.length > 1}
						<MultiLine series={[{ name: 'Remaining ICU hours', color: '#0ea5e9', values: historyPredictions.map((p) => p.icu_stay_time.remaining_hours) }]} xLabels={sofaHistoryLabels} yMin={0} yMax={Math.max(...historyPredictions.map((p) => p.icu_stay_time.remaining_hours)) * 1.2} yLabel="hours" height={180} />
					{:else}
						<p class="note">Advance through at least 2 cutoffs to see a trend.</p>
					{/if}
				</Panel>
				<Panel eyebrow="SUPPORT / TREND" title="Calibrated risk vs. frozen threshold" note="ACROSS REPLAY SO FAR">
					{#if historyPredictions.length > 1}
						{@const threshold = prediction.organ_support.threshold}
						<MultiLine series={[{ name: 'Calibrated risk', color: '#fbbf24', values: historyPredictions.map((p) => p.organ_support.probability_24h) }, { name: 'Threshold', color: '#fb7185', values: historyPredictions.map(() => threshold) }]} xLabels={sofaHistoryLabels} yMin={0} yMax={1} yLabel="probability" height={180} />
					{:else}
						<p class="note">Advance through at least 2 cutoffs to see a trend.</p>
					{/if}
				</Panel>
			</div>

			<Panel eyebrow="MODEL INPUT / TEMPORAL WINDOW" title="48-hour lookback — eight 6-hour bins" note={`${prediction.data_quality.observed_bins}/${prediction.data_quality.total_bins} BINS OBSERVED`}>
				<div class="window-layout">
					<GaugeRing value={Math.round((prediction.data_quality.observed_feature_fraction ?? 0) * 100)} size={84} stroke={8} color="#2bb8b0" label="OBSERVED" />
					<div class="window-heatmap">
						<TemporalHeatmap channelNames={prediction.temporal_window.channel_names} observationMask={prediction.temporal_window.observation_mask} paddingMask={prediction.temporal_window.padding_mask} />
					</div>
				</div>
				<p class="note">{prediction.data_quality.observed_feature_values} of {prediction.data_quality.total_feature_values} feature values observed · {prediction.data_quality.missing_feature_values} missing · never a prediction confidence score.</p>
			</Panel>

			<Panel eyebrow="HISTORICAL TIMELINE" title="Observed variables at or before the selected cutoff" note={`${events.length} EVENTS VISIBLE`}>
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

			<Panel eyebrow="REPLAY HISTORY" title="How predictions evolve as replay advances" note="SEQUENTIAL API CALLS, NOT A CACHE">
				<table class="history-table">
					<thead><tr><th>Cutoff</th><th>Δ24</th><th>Δ48</th><th>ICU remaining</th><th>Support risk</th><th>Alert</th></tr></thead>
					<tbody>
						{#each historyPredictions as p, i}
							<tr class:current={i === historyPredictions.length - 1}>
								<td>t{i}</td>
								<td>{p.recovery.delta_24h >= 0 ? '+' : ''}{p.recovery.delta_24h.toFixed(2)}</td>
								<td>{p.recovery.delta_48h >= 0 ? '+' : ''}{p.recovery.delta_48h.toFixed(2)}</td>
								<td>{p.icu_stay_time.remaining_hours.toFixed(1)}h</td>
								<td>{(p.organ_support.probability_24h * 100).toFixed(1)}%</td>
								<td>{p.organ_support.alert ? 'ABOVE' : 'below'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</Panel>
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
	.window-layout { display: flex; align-items: center; gap: 24px; }
	.window-heatmap { flex: 1; min-width: 0; }
	@media (max-width: 700px) { .window-layout { flex-direction: column; align-items: flex-start; } }
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
</style>
