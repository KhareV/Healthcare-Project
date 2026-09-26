<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { api, type CanonicalConcept } from '$lib/services/api';
	import { rememberCustomRecord } from '$lib/stores/custom-records';
	import { Plus, Trash2, Sparkles } from '@lucide/svelte';

	type Row = { concept: string; hours_since_admission: number | null; value: number | null };

	let concepts = $state<CanonicalConcept[]>([]);
	let loadingConcepts = $state(true);
	let error = $state('');

	let patientAlias = $state('MY-PATIENT-1');
	let ageYears = $state(60);
	let sexCategory = $state('FEMALE');
	let rows = $state<Row[]>([]);

	let submitting = $state(false);
	let result: Awaited<ReturnType<typeof api.createCustomRecord>> | null = $state(null);

	const conceptByKey = $derived(new Map(concepts.map((c) => [c.concept, c])));

	function addRow() {
		rows = [...rows, { concept: concepts[0]?.concept ?? '', hours_since_admission: 6, value: null }];
	}
	function removeRow(index: number) {
		rows = rows.filter((_, i) => i !== index);
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
		submitting = true;
		try {
			result = await api.createCustomRecord({ patient_alias: patientAlias, age_years: ageYears, sex_category: sexCategory, observations });
			rememberCustomRecord({ stay_id: result.stay_id, patient_alias: result.patient_alias });
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
	description="Type in vitals and labs at hours since a synthetic ICU admission. This runs through the exact same frozen models as the demo cohort — real feature construction, real TreeSHAP, real forecasts — for a record that exists only in this server process and is never written to disk."
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

		<div class="submit-row">
			<button class="submit-btn" onclick={submit} disabled={submitting}>{submitting ? 'Building…' : 'Build my record'}</button>
			<span class="note">Observations must be timed strictly after admission (hour 0) and no later than hour 90 — the frozen replay grid never extends further.</span>
		</div>

		{#if result}
			<Panel eyebrow="RECORD CREATED" title={result.patient_alias} note={result.stay_id}>
				<div class="result-grid">
					<div><span>Legal cutoffs</span><b>{result.n_legal_cutoffs}</b></div>
					<div><span>Concepts covered</span><b>{result.concept_coverage.observed} / {result.concept_coverage.total}</b></div>
					<div><span>Data readiness</span><b>{result.data_readiness}</b></div>
				</div>
				{#if result.warnings.length}
					<ul class="warnings">{#each result.warnings as w}<li>{w}</li>{/each}</ul>
				{/if}
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
	.warnings { margin: 0 0 14px; padding-left: 18px; color: #fbbf24; font-size: 11px; line-height: 1.7; }
	@media (max-width: 760px) {
		.basics-grid { grid-template-columns: 1fr; }
		.row { grid-template-columns: 1fr; }
		.row--head { display: none; }
	}
</style>
