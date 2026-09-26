<script lang="ts">
	import { onMount } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { api, type Condition, type CustomRecordSummary, type PredictionRunSnapshot } from '$lib/services/api';
	import { Plus, Trash2, Sparkles, History } from '@lucide/svelte';

	let loading = $state(true);
	let error = $state('');
	let persistenceMode = $state<'mongodb' | 'in_memory_only' | ''>('');

	let conditions = $state<Condition[]>([]);
	let conditionsNote = $state('');
	let newLabel = $state('');
	let newYear = $state<number | null>(null);
	let newStatus = $state('active');
	let savingCondition = $state(false);

	let records = $state<CustomRecordSummary[]>([]);

	let historyByStay = $state<Record<string, PredictionRunSnapshot[]>>({});
	let expandedStay = $state<string | null>(null);
	let loadingHistory = $state(false);

	async function refresh() {
		loading = true;
		error = '';
		try {
			const [health, conditionsResult, recordsResult] = await Promise.all([api.health(), api.listConditions(), api.listCustomRecords()]);
			persistenceMode = health.persistence_mode ?? '';
			conditions = conditionsResult.conditions;
			conditionsNote = conditionsResult.note ?? '';
			records = recordsResult.records;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not load your health record';
		} finally {
			loading = false;
		}
	}

	async function addCondition() {
		if (!newLabel.trim()) return;
		savingCondition = true;
		try {
			const created = await api.addCondition({ label: newLabel.trim(), diagnosed_year: newYear, status: newStatus });
			conditions = [...conditions, created];
			newLabel = '';
			newYear = null;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not save the condition';
		} finally {
			savingCondition = false;
		}
	}

	async function removeCondition(id: string) {
		try {
			await api.deleteCondition(id);
			conditions = conditions.filter((c) => c.condition_id !== id);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not remove the condition';
		}
	}

	async function toggleHistory(stayId: string) {
		if (expandedStay === stayId) {
			expandedStay = null;
			return;
		}
		expandedStay = stayId;
		if (historyByStay[stayId]) return;
		loadingHistory = true;
		try {
			const result = await api.getCustomRecordPredictionHistory(stayId);
			historyByStay = { ...historyByStay, [stayId]: result.predictions };
		} catch {
			historyByStay = { ...historyByStay, [stayId]: [] };
		} finally {
			loadingHistory = false;
		}
	}

	onMount(refresh);
</script>

<svelte:head><title>My Health Record | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage
	eyebrow="WORKSPACE / MY HEALTH RECORD"
	title="Your conditions and custom records, in one place."
	description="Conditions are display-only context you keep about yourself — they are never fed into the forecasting models. Records are the synthetic ICU episodes you've built under Enter My Own Record, together with the prediction history from every cutoff you've replayed."
>
	{#if error}<div class="error">{error}</div>{/if}

	{#if persistenceMode === 'in_memory_only'}
		<div class="notice">Persistence is not configured on this server — conditions cannot be saved, and records/predictions only last for this process's lifetime (the original behavior). Set <code>MONGODB_URI</code> to enable durable storage.</div>
	{/if}

	{#if loading}
		<div class="loading">Loading your health record…</div>
	{:else}
		<Panel eyebrow="CONDITIONS" title="Diagnosed conditions" note={conditionsNote || 'DISPLAY CONTEXT ONLY — NOT A MODEL INPUT'}>
			<div class="condition-form">
				<input type="text" placeholder="e.g. Ischemic heart disease" bind:value={newLabel} maxlength="200" />
				<input type="number" placeholder="Year" bind:value={newYear} min="1900" max="2100" />
				<select bind:value={newStatus}>
					<option value="active">Active</option>
					<option value="resolved">Resolved</option>
					<option value="monitoring">Monitoring</option>
				</select>
				<button type="button" class="add-row" onclick={addCondition} disabled={savingCondition || persistenceMode !== 'mongodb'}><Plus size={14} /> Add condition</button>
			</div>
			{#if conditions.length}
				<ul class="condition-list">
					{#each conditions as c (c.condition_id)}
						<li>
							<span class="condition-label">{c.label}</span>
							<span class="condition-meta">{c.diagnosed_year ?? '—'} · {c.status}</span>
							<button type="button" class="icon-btn" onclick={() => removeCondition(c.condition_id)} aria-label="Remove condition"><Trash2 size={14} /></button>
						</li>
					{/each}
				</ul>
			{:else}
				<p class="hint-text">No conditions recorded yet.</p>
			{/if}
		</Panel>

		<Panel eyebrow="RECORDS" title="Your custom records" note="PERSISTS ACROSS RESTARTS WHEN MONGODB IS CONFIGURED">
			{#if records.length}
				<div class="records">
					{#each records as r (r.stay_id)}
						<div class="record-card">
							<div class="record-head">
								<div>
									<b>{r.patient_alias}</b>
									<small>{r.stay_id}</small>
								</div>
								<div class="record-actions">
									<a class="submit-btn" href={`/patients?stay_id=${encodeURIComponent(r.stay_id)}&cutoff=0`}><Sparkles size={13} /> Open in Patient Replay</a>
									<button type="button" class="add-row" onclick={() => toggleHistory(r.stay_id)}><History size={13} /> {expandedStay === r.stay_id ? 'Hide history' : 'Prediction history'}</button>
								</div>
							</div>
							<div class="record-meta">
								<span>{r.age_years}y · {r.sex_category}</span>
								<span>{r.observations_entered} observations · {r.support_intervals_entered} support intervals</span>
								<span>{r.created_at ? new Date(r.created_at).toLocaleString() : 'not yet persisted'}</span>
							</div>
							{#if expandedStay === r.stay_id}
								<div class="history-table">
									{#if loadingHistory && !historyByStay[r.stay_id]}
										<p class="hint-text">Loading…</p>
									{:else if (historyByStay[r.stay_id] ?? []).length === 0}
										<p class="hint-text">No cutoffs replayed yet for this record — open it in Patient Replay and step through a few cutoffs.</p>
									{:else}
										<div class="history-row history-row--head"><span>Cutoff</span><span>SOFA</span><span>+24h</span><span>+48h</span><span>Remaining ICU</span><span>Support risk</span></div>
										{#each historyByStay[r.stay_id] as run (run.prediction_time)}
											<div class="history-row">
												<span>{run.prediction_time.slice(11, 16)}</span>
												<span>{run.current_sofa.toFixed(1)}</span>
												<span>{run.predicted_sofa_24h.toFixed(1)}</span>
												<span>{run.predicted_sofa_48h.toFixed(1)}</span>
												<span>{run.remaining_icu_hours.toFixed(0)}h</span>
												<span class:alert={run.support_alert}>{(run.support_calibrated_probability * 100).toFixed(0)}%</span>
											</div>
										{/each}
									{/if}
								</div>
							{/if}
						</div>
					{/each}
				</div>
			{:else}
				<p class="hint-text">No custom records yet. <a href="/patients/custom">Enter my own record</a> to build one.</p>
			{/if}
		</Panel>
	{/if}
</WorkbenchPage>

<style>
	.loading, .error { padding: 20px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.error { color: #fecdd3; }
	.notice { margin-bottom: 18px; padding: 12px 14px; border: 1px solid rgba(251,191,36,.3); background: rgba(251,191,36,.06); color: #fbbf24; font-size: 11px; line-height: 1.6; }
	.notice code { font-family: 'JetBrains Mono', monospace; }
	.hint-text { margin: 0; color: #64748b; font-size: 11px; line-height: 1.7; }
	.hint-text a { color: #2bb8b0; }
	.condition-form { display: grid; grid-template-columns: 2fr 1fr 1fr auto; gap: 10px; margin-bottom: 14px; }
	.condition-form input, .condition-form select { padding: 9px 10px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #dce9e8; font: 12px 'Space Grotesk', sans-serif; }
	.add-row { display: flex; align-items: center; gap: 7px; padding: 9px 14px; border: 1px dashed rgba(148,163,184,.3); background: transparent; color: #94a3b8; cursor: pointer; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; white-space: nowrap; }
	.add-row:hover { border-color: #2bb8b0; color: #2bb8b0; }
	.add-row:disabled { opacity: .5; cursor: not-allowed; }
	.condition-list { display: grid; gap: 6px; margin: 0; padding: 0; list-style: none; }
	.condition-list li { display: grid; grid-template-columns: 1fr auto 32px; align-items: center; gap: 10px; padding: 10px 12px; border: 1px solid rgba(148,163,184,.14); background: #080e1d; }
	.condition-label { color: #dce9e8; font: 12px 'Space Grotesk', sans-serif; }
	.condition-meta { color: #64748b; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; }
	.icon-btn { display: grid; place-items: center; padding: 8px; border: 1px solid rgba(251,113,133,.25); background: transparent; color: #fb7185; cursor: pointer; }
	.icon-btn:hover { background: rgba(251,113,133,.08); }
	.records { display: grid; gap: 12px; }
	.record-card { border: 1px solid rgba(148,163,184,.14); background: #080e1d; padding: 14px; }
	.record-head { display: flex; justify-content: space-between; align-items: start; gap: 14px; flex-wrap: wrap; }
	.record-head b { display: block; color: #eef7f6; font: 500 14px 'Space Grotesk', sans-serif; }
	.record-head small { color: #53647b; font: 9px 'JetBrains Mono', monospace; }
	.record-actions { display: flex; gap: 8px; flex-wrap: wrap; }
	.submit-btn { display: flex; align-items: center; gap: 6px; padding: 8px 12px; border: 1px solid #2bb8b0; background: #2bb8b0; color: #03110f; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; cursor: pointer; text-decoration: none; }
	.record-meta { display: flex; gap: 16px; margin-top: 10px; color: #64748b; font: 9px 'JetBrains Mono', monospace; letter-spacing: .05em; flex-wrap: wrap; }
	.history-table { margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(148,163,184,.12); display: grid; gap: 4px; }
	.history-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; font: 11px 'Space Grotesk', sans-serif; color: #dce9e8; }
	.history-row--head { color: #53647b; font: 7px 'JetBrains Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
	.history-row .alert { color: #fb7185; font-weight: 600; }
	@media (max-width: 760px) {
		.condition-form { grid-template-columns: 1fr; }
		.history-row { grid-template-columns: repeat(3, 1fr); }
	}
</style>
