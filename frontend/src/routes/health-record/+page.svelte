<script lang="ts">
	import { onMount } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import {
		api,
		type ConditionStatus,
		type HealthRecordCondition,
		type HealthRecordEncounter,
		type HealthRecordObservationRow,
		type HealthRecordReport,
		type HealthRecordSupportRow,
		type PatientProfile,
		type PredictionRunSnapshot
	} from '$lib/services/api';
	import { Download, FileText, Plus, Sparkles, Trash2, Upload } from '@lucide/svelte';

	type Tab = 'overview' | 'conditions' | 'encounters' | 'vitals' | 'reports' | 'predictions';
	const TABS: { id: Tab; label: string }[] = [
		{ id: 'overview', label: 'Overview' },
		{ id: 'conditions', label: 'Conditions' },
		{ id: 'encounters', label: 'Encounters' },
		{ id: 'vitals', label: 'Vitals & Labs' },
		{ id: 'reports', label: 'Reports' },
		{ id: 'predictions', label: 'Prediction History' }
	];
	let activeTab = $state<Tab>('overview');

	let loading = $state(true);
	let error = $state('');
	let persistenceMode = $state<'mongodb' | 'in_memory_only' | ''>('');

	let profile = $state<PatientProfile | null>(null);
	let profileForm = $state({ display_name_or_alias: '', age_years: null as number | null, sex_category: '', blood_group: '', height_cm: null as number | null, weight_kg: null as number | null });
	let savingProfile = $state(false);

	let conditions = $state<HealthRecordCondition[]>([]);
	let newCondition = $state({ name: '', code: '', diagnosed_date: '', status: 'active' as ConditionStatus, notes: '' });
	let savingCondition = $state(false);
	let editingConditionId = $state<string | null>(null);
	let editForm = $state({ name: '', code: '', diagnosed_date: '', status: 'active' as ConditionStatus, notes: '' });

	let encounters = $state<HealthRecordEncounter[]>([]);
	let observations = $state<HealthRecordObservationRow[]>([]);
	let supportRows = $state<HealthRecordSupportRow[]>([]);
	let predictions = $state<PredictionRunSnapshot[]>([]);
	let reports = $state<HealthRecordReport[]>([]);

	let vitalsEncounterFilter = $state('all');
	let vitalsConceptFilter = $state('all');

	let uploadTitle = $state('');
	let uploadDocType = $state('lab_report');
	let uploadDate = $state('');
	let uploadFile = $state<File | null>(null);
	let uploading = $state(false);

	const encounterAlias = $derived(new Map(encounters.map((e) => [e.stay_id, e.patient_alias])));
	const conceptOptions = $derived([...new Set(observations.map((o) => o.concept))].sort());
	const filteredObservations = $derived(
		observations.filter(
			(o) => (vitalsEncounterFilter === 'all' || o.encounter_id === vitalsEncounterFilter) && (vitalsConceptFilter === 'all' || o.concept === vitalsConceptFilter)
		)
	);

	async function refreshAll(isRetry = false) {
		loading = true;
		error = '';
		try {
			const health = await api.health();
			persistenceMode = health.persistence_mode ?? '';
			const [profileResult, conditionsResult, encountersResult, observationsResult, supportResult, predictionsResult, reportsResult] = await Promise.all([
				api.healthRecord.getProfile(),
				api.healthRecord.listConditions(),
				api.healthRecord.listEncounters(),
				api.healthRecord.listObservations(),
				api.healthRecord.listSupport(),
				api.healthRecord.listPredictions(),
				api.healthRecord.listReports()
			]);
			profile = profileResult;
			profileForm = {
				display_name_or_alias: profileResult.display_name_or_alias,
				age_years: profileResult.age_years,
				sex_category: profileResult.sex_category ?? '',
				blood_group: profileResult.blood_group ?? '',
				height_cm: profileResult.height_cm,
				weight_kg: profileResult.weight_kg
			};
			conditions = conditionsResult.conditions;
			encounters = encountersResult.encounters;
			observations = observationsResult.observations;
			supportRows = supportResult.support_intervals;
			predictions = predictionsResult.predictions;
			reports = reportsResult.reports;
		} catch (cause) {
			// One automatic retry: the backend's seven-way concurrent read on
			// this page's initial load occasionally races a transient MongoDB
			// connection-pool reset (see src/serving/v2/persistence.py's
			// _retrying) -- most such blips resolve within a second, and a
			// silent retry beats surfacing a scary error for a load that
			// would have worked on the very next attempt.
			if (!isRetry) {
				await new Promise((resolve) => setTimeout(resolve, 800));
				return refreshAll(true);
			}
			error = cause instanceof Error ? cause.message : 'Could not load your health record';
		} finally {
			loading = false;
		}
	}

	async function saveProfile() {
		savingProfile = true;
		try {
			profile = await api.healthRecord.updateProfile({
				display_name_or_alias: profileForm.display_name_or_alias,
				age_years: profileForm.age_years,
				sex_category: profileForm.sex_category || undefined,
				blood_group: profileForm.blood_group || undefined,
				height_cm: profileForm.height_cm,
				weight_kg: profileForm.weight_kg
			});
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not save your profile';
		} finally {
			savingProfile = false;
		}
	}

	async function addCondition() {
		if (!newCondition.name.trim()) return;
		savingCondition = true;
		try {
			const created = await api.healthRecord.addCondition({
				name: newCondition.name.trim(),
				code: newCondition.code || null,
				diagnosed_date: newCondition.diagnosed_date || null,
				status: newCondition.status,
				notes: newCondition.notes || null
			});
			conditions = [...conditions, created];
			newCondition = { name: '', code: '', diagnosed_date: '', status: 'active', notes: '' };
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not save the condition';
		} finally {
			savingCondition = false;
		}
	}

	function startEditCondition(c: HealthRecordCondition) {
		editingConditionId = c.condition_id;
		editForm = { name: c.name, code: c.code ?? '', diagnosed_date: c.diagnosed_date ?? '', status: (c.status as ConditionStatus) || 'active', notes: c.notes ?? '' };
	}

	async function saveConditionEdit() {
		if (!editingConditionId) return;
		try {
			const updated = await api.healthRecord.editCondition(editingConditionId, {
				name: editForm.name,
				code: editForm.code || null,
				diagnosed_date: editForm.diagnosed_date || null,
				status: editForm.status,
				notes: editForm.notes || null
			});
			conditions = conditions.map((c) => (c.condition_id === updated.condition_id ? updated : c));
			editingConditionId = null;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not update the condition';
		}
	}

	async function removeCondition(id: string) {
		try {
			await api.healthRecord.deleteCondition(id);
			conditions = conditions.filter((c) => c.condition_id !== id);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not remove the condition';
		}
	}

	async function submitReport() {
		if (!uploadFile || !uploadTitle.trim()) return;
		uploading = true;
		try {
			const created = await api.healthRecord.uploadReport({
				title: uploadTitle.trim(), document_type: uploadDocType, report_date: uploadDate || null, file: uploadFile
			});
			reports = [created, ...reports];
			uploadTitle = '';
			uploadDate = '';
			uploadFile = null;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not upload the report';
		} finally {
			uploading = false;
		}
	}

	async function downloadReport(report: HealthRecordReport) {
		try {
			const blob = await api.healthRecord.downloadReport(report.report_id);
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = report.original_filename;
			a.click();
			URL.revokeObjectURL(url);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not download the report';
		}
	}

	async function removeReport(id: string) {
		try {
			await api.healthRecord.deleteReport(id);
			reports = reports.filter((r) => r.report_id !== id);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not delete the report';
		}
	}

	async function exportRecord() {
		try {
			const data = await api.healthRecord.exportRecord();
			const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `health-record-export-${new Date().toISOString().slice(0, 10)}.json`;
			a.click();
			URL.revokeObjectURL(url);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not export your record';
		}
	}

	onMount(refreshAll);
</script>

<svelte:head><title>My Health Record | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage
	eyebrow="WORKSPACE / MY HEALTH RECORD"
	title="One consolidated, longitudinal research record."
	description="Profile, conditions, encounters, vitals/labs, organ-support history, uploaded reports, and prediction history for your account — all owner-scoped, all derived from the same structured input the forecasting pipeline already uses. This is a synthetic/research record, not a production clinical EHR."
>
	{#if error}<div class="error">{error}</div>{/if}

	{#if persistenceMode === 'in_memory_only'}
		<div class="notice">Persistence is not configured on this server — profile, conditions, and reports cannot be saved, and encounters/predictions only last for this process's lifetime. Set <code>MONGODB_URI</code> to enable durable storage.</div>
	{/if}

	{#if loading}
		<div class="loading">Loading your health record…</div>
	{:else}
		<div class="tab-strip" role="tablist" aria-label="Health record sections">
			{#each TABS as tab}
				<button type="button" role="tab" aria-selected={activeTab === tab.id} class:active={activeTab === tab.id} onclick={() => (activeTab = tab.id)}>{tab.label}</button>
			{/each}
		</div>

		{#if activeTab === 'overview'}
			<Panel eyebrow="OVERVIEW" title={profile?.display_name_or_alias ?? 'My Health Record'} note={profile?.patient_id}>
				<div class="overview-grid">
					<div class="overview-card"><span>Active conditions</span><b>{conditions.filter((c) => c.status === 'active').length}</b></div>
					<div class="overview-card"><span>Encounters</span><b>{encounters.length}</b></div>
					<div class="overview-card"><span>Reports</span><b>{reports.length}</b></div>
					<div class="overview-card"><span>Latest prediction</span><b>{predictions.length ? predictions[predictions.length - 1].prediction_time.slice(0, 10) : '—'}</b></div>
				</div>
				<button type="button" class="add-row" onclick={exportRecord}><Download size={14} /> Export my record (JSON)</button>
			</Panel>

			<Panel eyebrow="PROFILE" title="Demographics" note="DISPLAY CONTEXT — SEPARATE FROM PER-ENCOUNTER SNAPSHOTS USED BY THE MODEL">
				<div class="profile-grid">
					<label>Display name<input type="text" bind:value={profileForm.display_name_or_alias} maxlength="120" /></label>
					<label>Age (years)<input type="number" bind:value={profileForm.age_years} min="0" max="120" /></label>
					<label>Sex<select bind:value={profileForm.sex_category}><option value="">—</option><option value="FEMALE">Female</option><option value="MALE">Male</option><option value="OTHER">Other</option></select></label>
					<label>Blood group<input type="text" bind:value={profileForm.blood_group} maxlength="8" placeholder="e.g. O+" /></label>
					<label>Height (cm)<input type="number" bind:value={profileForm.height_cm} min="0" max="300" /></label>
					<label>Weight (kg)<input type="number" bind:value={profileForm.weight_kg} min="0" max="500" /></label>
				</div>
				<button type="button" class="submit-btn" onclick={saveProfile} disabled={savingProfile || persistenceMode !== 'mongodb'}>{savingProfile ? 'Saving…' : 'Save profile'}</button>
			</Panel>
		{/if}

		{#if activeTab === 'conditions'}
			<Panel eyebrow="CONDITIONS" title="Diagnosed conditions" note="DISPLAY CONTEXT ONLY — NOT A MODEL INPUT">
				<div class="condition-form">
					<input type="text" placeholder="e.g. Ischemic heart disease" bind:value={newCondition.name} maxlength="200" />
					<input type="text" placeholder="Code (optional)" bind:value={newCondition.code} maxlength="64" />
					<input type="date" bind:value={newCondition.diagnosed_date} />
					<select bind:value={newCondition.status}>
						<option value="active">Active</option>
						<option value="resolved">Resolved</option>
						<option value="historical">Historical</option>
					</select>
					<button type="button" class="add-row" onclick={addCondition} disabled={savingCondition || persistenceMode !== 'mongodb'}><Plus size={14} /> Add condition</button>
				</div>
				{#if conditions.length}
					<ul class="condition-list">
						{#each conditions as c (c.condition_id)}
							<li>
								{#if editingConditionId === c.condition_id}
									<div class="condition-edit-form">
										<input type="text" bind:value={editForm.name} maxlength="200" />
										<input type="text" placeholder="Code" bind:value={editForm.code} maxlength="64" />
										<input type="date" bind:value={editForm.diagnosed_date} />
										<select bind:value={editForm.status}>
											<option value="active">Active</option>
											<option value="resolved">Resolved</option>
											<option value="historical">Historical</option>
										</select>
										<textarea placeholder="Notes" bind:value={editForm.notes}></textarea>
										<div class="condition-edit-actions">
											<button type="button" class="submit-btn" onclick={saveConditionEdit}>Save</button>
											<button type="button" class="add-row" onclick={() => (editingConditionId = null)}>Cancel</button>
										</div>
									</div>
								{:else}
									<span class="condition-label">{c.name}{#if c.code}<small> · {c.code}</small>{/if}</span>
									<span class="condition-meta">{c.diagnosed_date ?? '—'} · {c.status}</span>
									<div class="condition-actions">
										<button type="button" class="add-row" onclick={() => startEditCondition(c)}>Edit</button>
										<button type="button" class="icon-btn" onclick={() => removeCondition(c.condition_id)} aria-label="Remove condition"><Trash2 size={14} /></button>
									</div>
								{/if}
							</li>
						{/each}
					</ul>
				{:else}
					<p class="hint-text">No conditions recorded yet.</p>
				{/if}
			</Panel>
		{/if}

		{#if activeTab === 'encounters'}
			<Panel eyebrow="ENCOUNTERS" title="Your custom records" note="PERSISTS ACROSS RESTARTS WHEN MONGODB IS CONFIGURED">
				{#if encounters.length}
					<div class="rows">
						<div class="row row--encounters row--head"><span>Patient</span><span>Type</span><span>Source</span><span>Observations</span><span></span></div>
						{#each encounters as enc (enc.stay_id)}
							<div class="row row--encounters">
								<span>{enc.patient_alias} <small class="dim">{enc.stay_id}</small></span>
								<span>{enc.encounter_type.replace(/_/g, ' ')}</span>
								<span>{enc.source.replace(/_/g, ' ')}</span>
								<span>{enc.observations.length} obs · {enc.support_intervals.length} support</span>
								<a class="submit-btn" href={`/patients?stay_id=${encodeURIComponent(enc.stay_id)}&cutoff=0`}><Sparkles size={13} /> Open</a>
							</div>
						{/each}
					</div>
				{:else}
					<p class="hint-text">No encounters yet. <a href="/patients/custom">Enter my own record</a> to build one.</p>
				{/if}
			</Panel>
		{/if}

		{#if activeTab === 'vitals'}
			<Panel eyebrow="VITALS & LABS" title="Chronological observation history" note="ACROSS ALL ENCOUNTERS">
				<div class="filters">
					<select bind:value={vitalsEncounterFilter}>
						<option value="all">All encounters</option>
						{#each encounters as enc}<option value={enc.stay_id}>{enc.patient_alias}</option>{/each}
					</select>
					<select bind:value={vitalsConceptFilter}>
						<option value="all">All concepts</option>
						{#each conceptOptions as concept}<option value={concept}>{concept.replace(/_/g, ' ')}</option>{/each}
					</select>
				</div>
				{#if filteredObservations.length}
					<div class="rows">
						<div class="row row--vitals row--head"><span>Hour</span><span>Concept</span><span>Value</span><span>Encounter</span></div>
						{#each filteredObservations as o, i (i)}
							<div class="row row--vitals">
								<span>{o.hours_since_admission}h</span>
								<span>{o.concept.replace(/_/g, ' ')}</span>
								<span>{o.value}</span>
								<span class="dim">{encounterAlias.get(o.encounter_id) ?? o.encounter_id}</span>
							</div>
						{/each}
					</div>
				{:else}
					<p class="hint-text">No observations match this filter.</p>
				{/if}
			</Panel>

			<Panel eyebrow="SUPPORT / INTERVENTIONS" title="Organ-support history" note="OFF→ON DEFINITION UNCHANGED FROM THE FROZEN SCIENTIFIC PIPELINE">
				{#if supportRows.length}
					<div class="rows">
						<div class="row row--support-hist row--head"><span>Type</span><span>Agent</span><span>Start</span><span>End</span><span>Encounter</span></div>
						{#each supportRows as s, i (i)}
							<div class="row row--support-hist">
								<span>{s.kind}</span>
								<span>{s.agent ?? '—'}</span>
								<span>{s.start_hour}h</span>
								<span>{s.end_hour ?? 'ongoing'}</span>
								<span class="dim">{encounterAlias.get(s.encounter_id) ?? s.encounter_id}</span>
							</div>
						{/each}
					</div>
				{:else}
					<p class="hint-text">No organ-support intervals recorded.</p>
				{/if}
			</Panel>
		{/if}

		{#if activeTab === 'reports'}
			<Panel eyebrow="REPORTS" title="Upload a report" note="PDF, PNG, OR JPEG — 10MB LIMIT">
				<div class="report-form">
					<input type="text" placeholder="Title" bind:value={uploadTitle} maxlength="200" />
					<select bind:value={uploadDocType}>
						<option value="lab_report">Lab report</option>
						<option value="imaging">Imaging</option>
						<option value="discharge_summary">Discharge summary</option>
						<option value="other">Other</option>
					</select>
					<input type="date" bind:value={uploadDate} />
					<input type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" onchange={(e) => (uploadFile = (e.currentTarget as HTMLInputElement).files?.[0] ?? null)} />
					<button type="button" class="submit-btn" onclick={submitReport} disabled={uploading || !uploadFile || !uploadTitle.trim() || persistenceMode !== 'mongodb'}><Upload size={14} /> {uploading ? 'Uploading…' : 'Upload'}</button>
				</div>
				<p class="hint-text">Uploaded content is never used as a model input — see the Data & Provenance page for how forecasts are actually computed. Parsing is not implemented in this pass; every report stays <code>NOT_PARSED</code>.</p>
			</Panel>

			<Panel eyebrow="REPORTS" title="Your reports">
				{#if reports.length}
					<div class="report-cards">
						{#each reports as r (r.report_id)}
							<div class="report-card">
								<div class="report-icon"><FileText size={18} /></div>
								<div class="report-meta">
									<b>{r.title}</b>
									<small>{r.document_type.replace(/_/g, ' ')} · {(r.size_bytes / 1024).toFixed(0)} KB · {r.report_date ?? r.uploaded_at.slice(0, 10)}</small>
								</div>
								<div class="report-actions">
									<button type="button" class="icon-btn" onclick={() => downloadReport(r)} aria-label="Download report"><Download size={14} /></button>
									<button type="button" class="icon-btn icon-btn--danger" onclick={() => removeReport(r.report_id)} aria-label="Delete report"><Trash2 size={14} /></button>
								</div>
							</div>
						{/each}
					</div>
				{:else}
					<p class="hint-text">No reports uploaded yet.</p>
				{/if}
			</Panel>
		{/if}

		{#if activeTab === 'predictions'}
			<Panel eyebrow="PREDICTION HISTORY" title="Forecasts across every cutoff you've replayed" note="DERIVED EVIDENCE — REPLAY ALWAYS RECOMPUTES THROUGH THE FROZEN MODEL PIPELINE">
				{#if predictions.length}
					<div class="rows">
						<div class="row row--predictions row--head"><span>Encounter</span><span>Cutoff</span><span>SOFA</span><span>+24h</span><span>+48h</span><span>Remaining ICU</span><span>Support risk</span></div>
						{#each predictions as p (p.stay_id + p.prediction_time)}
							<div class="row row--predictions">
								<span class="dim">{encounterAlias.get(p.stay_id) ?? p.stay_id}</span>
								<span>{p.prediction_time.slice(11, 16)}</span>
								<span>{p.current_sofa.toFixed(1)}</span>
								<span>{p.predicted_sofa_24h.toFixed(1)}</span>
								<span>{p.predicted_sofa_48h.toFixed(1)}</span>
								<span>{p.remaining_icu_hours.toFixed(0)}h</span>
								<span class:alert={p.support_alert}>{(p.support_calibrated_probability * 100).toFixed(0)}%</span>
							</div>
						{/each}
					</div>
				{:else}
					<p class="hint-text">No cutoffs replayed yet. Open an encounter in Patient Replay and step through a few cutoffs.</p>
				{/if}
			</Panel>
		{/if}
	{/if}
</WorkbenchPage>

<style>
	.loading, .error { padding: 20px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.error { color: #fecdd3; }
	.notice { margin-bottom: 18px; padding: 12px 14px; border: 1px solid rgba(251,191,36,.3); background: rgba(251,191,36,.06); color: #fbbf24; font-size: 11px; line-height: 1.6; }
	.notice code { font-family: 'JetBrains Mono', monospace; }
	.hint-text { margin: 0; color: #64748b; font-size: 11px; line-height: 1.7; }
	.hint-text a { color: #2bb8b0; }
	.hint-text code { font-family: 'JetBrains Mono', monospace; }
	.dim { color: #53647b; }

	.tab-strip { display: flex; gap: 6px; margin-bottom: 18px; flex-wrap: wrap; border-bottom: 1px solid rgba(148,163,184,.14); padding-bottom: 10px; }
	.tab-strip button { padding: 8px 14px; border: 1px solid transparent; background: transparent; color: #71829a; cursor: pointer; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
	.tab-strip button:hover { color: #dce9e8; }
	.tab-strip button.active { color: #2bb8b0; border-color: rgba(43,184,176,.35); background: rgba(43,184,176,.08); }

	.overview-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
	.overview-card { padding: 14px; border: 1px solid rgba(148,163,184,.14); background: #080e1d; }
	.overview-card span { display: block; color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.overview-card b { display: block; margin-top: 8px; color: #eef7f6; font: 500 20px 'Space Grotesk', sans-serif; }

	.profile-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 16px; }
	.profile-grid label { display: grid; gap: 6px; color: #71829a; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.profile-grid input, .profile-grid select { padding: 9px 10px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #dce9e8; font: 12px 'Space Grotesk', sans-serif; }

	.condition-form { display: grid; grid-template-columns: 1.6fr 1fr 1fr 1fr auto; gap: 10px; margin-bottom: 14px; }
	.condition-form input, .condition-form select { padding: 9px 10px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #dce9e8; font: 12px 'Space Grotesk', sans-serif; }
	.add-row { display: flex; align-items: center; gap: 7px; padding: 9px 14px; border: 1px dashed rgba(148,163,184,.3); background: transparent; color: #94a3b8; cursor: pointer; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; white-space: nowrap; }
	.add-row:hover { border-color: #2bb8b0; color: #2bb8b0; }
	.add-row:disabled { opacity: .5; cursor: not-allowed; }
	.condition-list { display: grid; gap: 6px; margin: 0; padding: 0; list-style: none; }
	.condition-list li { display: grid; grid-template-columns: 1.4fr auto auto; align-items: center; gap: 10px; padding: 10px 12px; border: 1px solid rgba(148,163,184,.14); background: #080e1d; }
	.condition-label { color: #dce9e8; font: 12px 'Space Grotesk', sans-serif; }
	.condition-label small { color: #53647b; }
	.condition-meta { color: #64748b; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; }
	.condition-actions { display: flex; gap: 8px; }
	.condition-edit-form { grid-column: 1 / -1; display: grid; grid-template-columns: 1.6fr 1fr 1fr 1fr; gap: 8px; }
	.condition-edit-form textarea { grid-column: 1 / -1; min-height: 50px; padding: 8px 10px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #dce9e8; font: 11px 'Space Grotesk', sans-serif; resize: vertical; }
	.condition-edit-form input, .condition-edit-form select { padding: 8px 10px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #dce9e8; font: 11px 'Space Grotesk', sans-serif; }
	.condition-edit-actions { grid-column: 1 / -1; display: flex; gap: 8px; }
	.icon-btn { display: grid; place-items: center; padding: 8px; border: 1px solid rgba(148,163,184,.2); background: transparent; color: #94a3b8; cursor: pointer; }
	.icon-btn:hover { border-color: #2bb8b0; color: #2bb8b0; }
	.icon-btn--danger { border-color: rgba(251,113,133,.25); color: #fb7185; }
	.icon-btn--danger:hover { background: rgba(251,113,133,.08); border-color: #fb7185; color: #fb7185; }

	.submit-btn { display: flex; align-items: center; gap: 8px; padding: 10px 16px; border: 1px solid #2bb8b0; background: #2bb8b0; color: #03110f; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; cursor: pointer; text-decoration: none; white-space: nowrap; }
	.submit-btn:disabled { opacity: .6; cursor: wait; }

	.rows { display: grid; gap: 6px; margin-bottom: 12px; }
	.row { display: grid; gap: 8px; align-items: center; padding: 10px 12px; border: 1px solid rgba(148,163,184,.14); background: #080e1d; font: 11px 'Space Grotesk', sans-serif; color: #dce9e8; }
	.row--head { background: transparent; border: none; padding: 0 12px; color: #53647b; font: 7px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; }
	.row--encounters { grid-template-columns: 1.6fr 1fr 1fr 1.2fr auto; }
	.row--vitals { grid-template-columns: .6fr 1.2fr .8fr 1fr; }
	.row--support-hist { grid-template-columns: 1fr 1fr .8fr .8fr 1fr; }
	.row--predictions { grid-template-columns: 1.2fr .7fr .6fr .6fr .6fr 1fr .8fr; }
	.row--predictions .alert { color: #fb7185; font-weight: 600; }

	.filters { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
	.filters select { padding: 8px 10px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #dce9e8; font: 11px 'Space Grotesk', sans-serif; }

	.report-form { display: grid; grid-template-columns: 1.4fr 1fr 1fr 1.2fr auto; gap: 10px; align-items: center; margin-bottom: 12px; }
	.report-form input, .report-form select { padding: 9px 10px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #dce9e8; font: 12px 'Space Grotesk', sans-serif; }
	.report-form input[type='file'] { padding: 6px; font-size: 10px; }
	.report-cards { display: grid; gap: 8px; }
	.report-card { display: grid; grid-template-columns: 34px 1fr auto; align-items: center; gap: 12px; padding: 12px; border: 1px solid rgba(148,163,184,.14); background: #080e1d; }
	.report-icon { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid rgba(43,184,176,.25); color: #2bb8b0; }
	.report-meta b { display: block; color: #eef7f6; font: 500 12px 'Space Grotesk', sans-serif; }
	.report-meta small { color: #64748b; font: 9px 'JetBrains Mono', monospace; letter-spacing: .04em; }
	.report-actions { display: flex; gap: 8px; }

	@media (max-width: 900px) {
		.overview-grid, .profile-grid { grid-template-columns: repeat(2, 1fr); }
		.condition-form, .report-form { grid-template-columns: 1fr; }
		.row--encounters, .row--vitals, .row--support-hist, .row--predictions { grid-template-columns: 1fr; }
		.row--head { display: none; }
	}
</style>
