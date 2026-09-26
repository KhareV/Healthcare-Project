<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { api, type CanonicalConcept, type VasopressorAgent } from '$lib/services/api';
	import { Plus, Trash2, Sparkles } from '@lucide/svelte';

	type Row = { concept: string; hours_since_admission: number | null; value: number | null };
	type SupportRow = { kind: 'vasopressor' | 'ventilation'; agent: VasopressorAgent; rate: number | null; start_hour: number | null; ongoing: boolean; end_hour: number | null };

	const VASOPRESSOR_AGENTS: { value: VasopressorAgent; label: string }[] = [
		{ value: 'norepinephrine', label: 'Norepinephrine' },
		{ value: 'epinephrine', label: 'Epinephrine' },
		{ value: 'dopamine', label: 'Dopamine' },
		{ value: 'dobutamine', label: 'Dobutamine' }
	];

	let concepts = $state<CanonicalConcept[]>([]);
	let loadingConcepts = $state(true);
	let error = $state('');

	let patientAlias = $state('MY-PATIENT-1');
	let ageYears = $state(60);
	let sexCategory = $state('FEMALE');
	let rows = $state<Row[]>([]);
	let supportRows = $state<SupportRow[]>([]);

	let submitting = $state(false);
	let result: Awaited<ReturnType<typeof api.createCustomRecord>> | null = $state(null);

	const conceptByKey = $derived(new Map(concepts.map((c) => [c.concept, c])));

	function addRow() {
		rows = [...rows, { concept: concepts[0]?.concept ?? '', hours_since_admission: 6, value: null }];
	}
	function removeRow(index: number) {
		rows = rows.filter((_, i) => i !== index);
	}

	function addSupportRow() {
		supportRows = [...supportRows, { kind: 'vasopressor', agent: 'norepinephrine', rate: 0.1, start_hour: 6, ongoing: true, end_hour: null }];
	}
	function removeSupportRow(index: number) {
		supportRows = supportRows.filter((_, i) => i !== index);
	}

	function fillPreset(kind: 'stable' | 'deteriorating') {
		if (!concepts.length) return;
		const base = (hour: number, values: Record<string, number>): Row[] =>
			Object.entries(values).map(([concept, value]) => ({ concept, hours_since_admission: hour, value }));
		if (kind === 'stable') {
			rows = [
				...base(6, { heart_rate: 78, mean_arterial_pressure: 82, systolic_blood_pressure: 118, diastolic_blood_pressure: 70, respiratory_rate: 16, oxygen_saturation: 98, temperature: 36.9, glasgow_coma_scale: 15, lactate: 1.1 }),
				...base(30, { heart_rate: 76, mean_arterial_pressure: 84, oxygen_saturation: 98, respiratory_rate: 15, lactate: 1.0, glasgow_coma_scale: 15 })
			];
		} else {
			rows = [
				...base(6, { heart_rate: 92, mean_arterial_pressure: 74, systolic_blood_pressure: 104, diastolic_blood_pressure: 62, respiratory_rate: 20, oxygen_saturation: 95, temperature: 37.6, glasgow_coma_scale: 14, lactate: 2.0 }),
				...base(30, { heart_rate: 110, mean_arterial_pressure: 60, oxygen_saturation: 90, respiratory_rate: 26, lactate: 3.4, glasgow_coma_scale: 12 })
			];
		}
	}

	async function submit() {
		error = '';
		result = null;
		const observations = rows
			.filter((r) => r.concept && r.hours_since_admission !== null && r.value !== null)
			.map((r) => ({ concept: r.concept, hours_since_admission: r.hours_since_admission as number, value: r.value as number }));
		if (!observations.length) {
			error = 'Add at least one observation with a time and a value.';
			return;
		}
		const support_intervals = supportRows
			.filter((r) => r.start_hour !== null && (r.ongoing || r.end_hour !== null))
			.map((r) =>
				r.kind === 'vasopressor'
					? { kind: 'vasopressor' as const, agent: r.agent, rate: r.rate ?? 0.1, start_hour: r.start_hour as number, end_hour: r.ongoing ? null : r.end_hour }
					: { kind: 'ventilation' as const, start_hour: r.start_hour as number, end_hour: r.ongoing ? null : r.end_hour }
			);
		submitting = true;
		try {
			// Persisted server-side (MongoDB when configured, see /health's
			// persistence_mode) and bound to the authenticated caller — no
			// client-side remembering needed; /patients lists it via the
			// real GET /custom-records endpoint.
			result = await api.createCustomRecord({ patient_alias: patientAlias, age_years: ageYears, sex_category: sexCategory, observations, support_intervals });
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not create the record';
		} finally {
			submitting = false;
		}
	}

	function openReplay() {
		if (result) void goto(`/patients?stay_id=${encodeURIComponent(result.stay_id)}&cutoff=0`);
	}

	onMount(async () => {
		try {
			const schema = await api.customRecordSchema();
			concepts = schema.concepts;
			addRow();
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not load the concept schema';
		} finally {
			loadingConcepts = false;
		}
	});
</script>

<svelte:head><title>Enter My Own Record | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage
	eyebrow="WORKSPACE / ENTER MY OWN RECORD"
	title="Build a synthetic ICU episode and forecast it."
	description="Type in vitals and labs at hours since a synthetic ICU admission. This runs through the exact same frozen models as the demo cohort — real feature construction, real TreeSHAP, real forecasts. It's bound to your account and saved to your health record (see My Health Record) when persistence is configured on this server; otherwise it lives only for this server process's lifetime."
>
	{#if error}<div class="error">{error}</div>{/if}

	{#if loadingConcepts}
		<div class="loading">Loading canonical concept schema…</div>
	{:else}
		<Panel eyebrow="PATIENT" title="Basic details">
			<div class="basics-grid">
				<label>Patient alias<input type="text" bind:value={patientAlias} maxlength="64" /></label>
				<label>Age (years)<input type="number" bind:value={ageYears} min="0" max="120" /></label>
				<label>Sex<select bind:value={sexCategory}><option value="FEMALE">Female</option><option value="MALE">Male</option><option value="OTHER">Other</option></select></label>
			</div>
		</Panel>

		<Panel eyebrow="OBSERVATIONS" title="Vitals & labs" note="TIME IS HOURS SINCE ADMISSION">
			<div class="presets">
				<span>Quick fill:</span>
				<button type="button" onclick={() => fillPreset('stable')}>Stable recovery</button>
				<button type="button" onclick={() => fillPreset('deteriorating')}>Deteriorating trajectory</button>
			</div>
			<div class="rows">
				<div class="row row--head"><span>Concept</span><span>Hours since admission</span><span>Value</span><span></span></div>
				{#each rows as row, index}
					{@const meta = conceptByKey.get(row.concept)}
					<div class="row">
						<select bind:value={row.concept}>
							{#each concepts as c}<option value={c.concept}>{c.label}</option>{/each}
						</select>
						<input type="number" min="0.01" max="90" step="0.5" bind:value={row.hours_since_admission} placeholder="e.g. 6" />
						<div class="value-cell">
							<input type="number" step="0.1" bind:value={row.value} placeholder={meta?.hint ?? ''} />
							<small>{meta?.unit ?? ''}{meta ? ` · ${meta.hint}` : ''}</small>
						</div>
						<button type="button" class="icon-btn" onclick={() => removeRow(index)} aria-label="Remove row"><Trash2 size={14} /></button>
					</div>
				{/each}
			</div>
			<button type="button" class="add-row" onclick={addRow}><Plus size={14} /> Add observation</button>
		</Panel>

		<Panel eyebrow="ORGAN SUPPORT" title="Vasopressor & ventilation state (optional)" note="OFF UNLESS ENTERED">
			<p class="hint-text">Leave empty to model this record with no active organ support at every cutoff — a real, valid state, not an error. Interval boundaries are hour-since-admission, same as observations above; "currently active" means the support has no known end within the replay grid.</p>
			{#if supportRows.length}
				<div class="rows">
					<div class="row row--support row--head"><span>Type</span><span>Agent</span><span>Rate (µg/kg/min)</span><span>Start hour</span><span>End / ongoing</span><span></span></div>
					{#each supportRows as row, index}
						<div class="row row--support">
							<select bind:value={row.kind}>
								<option value="vasopressor">Vasopressor</option>
								<option value="ventilation">Invasive ventilation</option>
							</select>
							{#if row.kind === 'vasopressor'}
								<select bind:value={row.agent}>
									{#each VASOPRESSOR_AGENTS as a}<option value={a.value}>{a.label}</option>{/each}
								</select>
								<input type="number" step="0.01" min="0.01" bind:value={row.rate} placeholder="0.1" />
							{:else}
								<span class="dash-cell">—</span><span class="dash-cell">—</span>
							{/if}
							<input type="number" min="0.01" max="90" step="0.5" bind:value={row.start_hour} placeholder="e.g. 6" />
							<div class="ongoing-cell">
								{#if row.ongoing}
									<span class="ongoing-badge">CURRENTLY ACTIVE</span>
								{:else}
									<input type="number" min="0.01" max="120" step="0.5" bind:value={row.end_hour} placeholder="end hour" />
								{/if}
								<label class="ongoing-toggle"><input type="checkbox" bind:checked={row.ongoing} /> ongoing</label>
							</div>
							<button type="button" class="icon-btn" onclick={() => removeSupportRow(index)} aria-label="Remove support interval"><Trash2 size={14} /></button>
						</div>
					{/each}
				</div>
			{/if}
			<button type="button" class="add-row" onclick={addSupportRow}><Plus size={14} /> Add support interval</button>
		</Panel>

		<div class="submit-row">
			<button class="submit-btn" onclick={submit} disabled={submitting}>{submitting ? 'Building…' : 'Build my record'}</button>
			<span class="note">Observations must be timed strictly after admission (hour 0) and no later than hour 90 — the frozen replay grid never extends further.</span>
		</div>

		{#if result}
			{@const rs = result.readiness_summary}
			<Panel eyebrow="DATA READINESS" title={result.patient_alias} note={result.stay_id}>
				<div class="result-grid">
					<div><span>Legal cutoffs</span><b>{rs.legal_cutoffs}</b></div>
					<div><span>Observations entered</span><b>{rs.observations_entered}</b></div>
					<div><span>Concepts represented</span><b>{rs.concepts_represented} / {rs.concepts_total}</b></div>
					<div><span>Earliest / latest hour</span><b>{rs.earliest_observation_hour ?? '—'}h / {rs.latest_observation_hour ?? '—'}h</b></div>
					<div><span>First-cutoff observed bins</span><b>{rs.first_cutoff_observed_bins ?? '—'} / {rs.first_cutoff_total_bins ?? '—'}</b></div>
					<div><span>Support state entered</span><b>{rs.support_state_entered.vasopressor ? 'Vasopressor' : ''}{rs.support_state_entered.vasopressor && rs.support_state_entered.invasive_ventilation ? ' + ' : ''}{rs.support_state_entered.invasive_ventilation ? 'Ventilation' : ''}{!rs.support_state_entered.vasopressor && !rs.support_state_entered.invasive_ventilation ? 'None' : ''}</b></div>
				</div>
				<div class="sofa-coverage">
					<span class="sofa-coverage-label">SOFA COMPONENTS SUPPORTED BY ENTERED DATA</span>
					<div class="sofa-pills">
						{#each rs.sofa_components_observed as c}<span class="sofa-pill sofa-pill--observed">{c.replace(/_/g, ' ')}</span>{/each}
						{#each rs.sofa_components_missing as c}<span class="sofa-pill sofa-pill--missing">{c.replace(/_/g, ' ')}</span>{/each}
					</div>
				</div>
				{#if result.warnings.length}
					<div class="missing-observations">
						<span class="missing-observations-label">MISSING OBSERVATIONS</span>
						<ul class="warnings">{#each result.warnings as w}<li>{w}</li>{/each}</ul>
					</div>
				{/if}
				<p class="urine-note">Urine output is not supported by direct entry — renal SOFA requires a fully continuous 24-hour reading chain that manual entry cannot honestly provide; renal scoring here uses creatinine alone.</p>
				<button class="submit-btn" onclick={openReplay}><Sparkles size={14} /> Open in Patient Replay</button>
			</Panel>
		{/if}
	{/if}
</WorkbenchPage>

<style>
	.loading, .error { padding: 20px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.error { color: #fecdd3; }
	.basics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
	.basics-grid label { display: grid; gap: 6px; color: #71829a; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.basics-grid input, .basics-grid select { padding: 9px 10px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #dce9e8; font: 12px 'Space Grotesk', sans-serif; }
	.presets { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; color: #64748b; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; }
	.presets button { padding: 7px 11px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #94a3b8; cursor: pointer; font: inherit; }
	.presets button:hover { border-color: #2bb8b0; color: #2bb8b0; }
	.rows { display: grid; gap: 6px; margin-bottom: 12px; }
	.row { display: grid; grid-template-columns: 1.4fr 1fr 1.4fr 32px; gap: 8px; align-items: center; }
	.row--head { color: #53647b; font: 7px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; }
	.row select, .row input { padding: 8px 10px; border: 1px solid rgba(148,163,184,.18); background: #080e1d; color: #dce9e8; font: 11px 'Space Grotesk', sans-serif; width: 100%; box-sizing: border-box; }
	.value-cell { display: grid; gap: 3px; }
	.value-cell small { color: #53647b; font: 8px 'JetBrains Mono', monospace; }
	.icon-btn { display: grid; place-items: center; padding: 8px; border: 1px solid rgba(251,113,133,.25); background: transparent; color: #fb7185; cursor: pointer; }
	.icon-btn:hover { background: rgba(251,113,133,.08); }
	.add-row { display: flex; align-items: center; gap: 7px; padding: 9px 14px; border: 1px dashed rgba(148,163,184,.3); background: transparent; color: #94a3b8; cursor: pointer; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.add-row:hover { border-color: #2bb8b0; color: #2bb8b0; }
	.submit-row { display: flex; align-items: center; gap: 16px; margin: 18px 0; flex-wrap: wrap; }
	.submit-btn { display: flex; align-items: center; gap: 8px; padding: 12px 20px; border: 1px solid #2bb8b0; background: #2bb8b0; color: #03110f; font: 10px 'JetBrains Mono', monospace; letter-spacing: .08em; cursor: pointer; }
	.submit-btn:disabled { opacity: .6; cursor: wait; }
	.note { color: #53647b; font-size: 10px; line-height: 1.6; max-width: 480px; }
	.result-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 14px; }
	.result-grid span { display: block; color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.result-grid b { display: block; margin-top: 6px; color: #eef7f6; font: 500 16px 'Space Grotesk', sans-serif; }
	.warnings { margin: 0; padding-left: 18px; color: #fbbf24; font-size: 11px; line-height: 1.7; }
	.hint-text { margin: 0 0 14px; color: #64748b; font-size: 11px; line-height: 1.7; max-width: 640px; }
	.row--support { grid-template-columns: 1.1fr 1.1fr 1fr 0.9fr 1.3fr 32px; }
	.dash-cell { color: #405067; text-align: center; }
	.ongoing-cell { display: flex; align-items: center; gap: 8px; }
	.ongoing-cell input { flex: 1; }
	.ongoing-badge { padding: 8px 10px; border: 1px solid rgba(43,184,176,.3); background: rgba(43,184,176,.08); color: #2bb8b0; font: 8px 'JetBrains Mono', monospace; letter-spacing: .06em; white-space: nowrap; flex: 1; text-align: center; }
	.ongoing-toggle { display: flex; align-items: center; gap: 5px; color: #71829a; font: 8px 'JetBrains Mono', monospace; letter-spacing: .06em; white-space: nowrap; }
	.ongoing-toggle input { width: auto; accent-color: #2bb8b0; }
	.sofa-coverage { margin: 4px 0 14px; }
	.sofa-coverage-label, .missing-observations-label { display: block; margin-bottom: 8px; color: #53647b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.sofa-pills { display: flex; flex-wrap: wrap; gap: 6px; }
	.sofa-pill { padding: 5px 10px; border: 1px solid rgba(148,163,184,.2); font: 9px 'JetBrains Mono', monospace; letter-spacing: .04em; text-transform: uppercase; }
	.sofa-pill--observed { border-color: rgba(43,184,176,.35); color: #2bb8b0; background: rgba(43,184,176,.06); }
	.sofa-pill--missing { color: #64748b; }
	.missing-observations { margin-bottom: 14px; }
	.urine-note { margin: 0 0 16px; padding: 10px 12px; border: 1px solid rgba(148,163,184,.14); color: #64748b; font-size: 10px; line-height: 1.6; }
	@media (max-width: 760px) {
		.basics-grid { grid-template-columns: 1fr; }
		.row { grid-template-columns: 1fr; }
		.row--head { display: none; }
		.result-grid { grid-template-columns: 1fr 1fr; }
	}
</style>
