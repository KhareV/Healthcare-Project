<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import { api, type DemoSubject, type PredictionResponse } from '$lib/services/api';

	let subjects = $state<DemoSubject[]>([]);
	let stayId = $state('');
	let cutoffIndex = $state(0);
	let prediction = $state<PredictionResponse | null>(null);
	let loading = $state(true);
	let error = $state('');
	let showFullHashes = $state(false);

	const subject = $derived(subjects.find((s) => s.stay_id === stayId) ?? null);
	const cutoffs = $derived(subject?.legal_cutoffs ?? []);

	async function load() {
		if (!subject) return;
		loading = true;
		error = '';
		try {
			prediction = await api.predict(stayId, cutoffs[cutoffIndex]);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Prediction failed';
		} finally {
			loading = false;
		}
	}
	function selectStay(id: string) {
		stayId = id;
		cutoffIndex = 0;
		goto(`?stay_id=${id}&cutoff=0`, { replaceState: true, noScroll: true });
		void load();
	}
	function setCutoff(index: number) {
		cutoffIndex = Math.max(0, Math.min(cutoffs.length - 1, index));
		goto(`?stay_id=${stayId}&cutoff=${cutoffIndex}`, { replaceState: true, noScroll: true });
		void load();
	}
	function short(hash: string) {
		return hash.slice(0, 12) + '…';
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

<svelte:head><title>Data Quality &amp; Provenance | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage eyebrow="05 / DATA QUALITY &amp; PROVENANCE" title="Transparent counts, never a confidence score." description="Observed-bin, padding, and missingness counts describe what the model actually saw at this cutoff. No such quantity is ever labeled model confidence, certainty, or a quality score here.">
	{#if error}<div class="error">{error}</div>{/if}
	{#if subjects.length}
		<div class="controls">
			<select value={stayId} onchange={(e) => selectStay((e.target as HTMLSelectElement).value)}>
				{#each subjects as s}<option value={s.stay_id}>{s.subject_id}</option>{/each}
			</select>
			<select value={cutoffIndex} onchange={(e) => setCutoff(Number((e.target as HTMLSelectElement).value))}>
				{#each cutoffs as c, i}<option value={i}>t{i} — {c.slice(0, 16).replace('T', ' ')}</option>{/each}
			</select>
		</div>
	{/if}

	{#if loading}
		<div class="loading">Loading…</div>
	{:else if prediction}
		<div class="metrics">
			<MetricTile label="Observed bins" value={`${prediction.data_quality.observed_bins}/${prediction.data_quality.total_bins}`} detail="OF THE 8-BIN WINDOW" />
			<MetricTile label="Padding bins" value={prediction.data_quality.padding_bins} detail="PRE-ICU-ADMISSION" tone="cyan" />
			<MetricTile label="Feature values observed" value={`${((prediction.data_quality.observed_feature_fraction ?? 0) * 100).toFixed(0)}%`} detail={`${prediction.data_quality.observed_feature_values} / ${prediction.data_quality.total_feature_values}`} tone="teal" />
			<MetricTile label="Missing feature values" value={prediction.data_quality.missing_feature_values} detail="NOT A CONFIDENCE SIGNAL" tone="amber" />
		</div>

		<Panel eyebrow="SEMANTICS" title="What these counts are — and are not">
			<p class="semantics">These are transparent observation counts, never a prediction confidence, certainty, or quality score. A missing feature value means the corresponding clinical variable was not observed inside the relevant 6-hour bin — the frozen models handle it as a native missing value, not a fabricated physiologic zero.</p>
		</Panel>

		<Panel eyebrow="MODEL / PROVENANCE METADATA" title="Per-task model identity" note={showFullHashes ? 'FULL HASHES' : 'SHORT HASHES'}>
			<button class="hash-toggle" onclick={() => (showFullHashes = !showFullHashes)}>{showFullHashes ? 'Show short hashes' : 'Show full hashes'}</button>
			<table>
				<thead><tr><th>Task</th><th>Family</th><th>Feature variant</th><th>Artifact hash</th></tr></thead>
				<tbody>
					<tr><td>recovery24</td><td>xgboost</td><td>B_MIN</td><td class="hash">{showFullHashes ? prediction.recovery.model_metadata.artifact_sha256_24h : short(prediction.recovery.model_metadata.artifact_sha256_24h)}</td></tr>
					<tr><td>recovery48</td><td>xgboost</td><td>B_MIN</td><td class="hash">{showFullHashes ? prediction.recovery.model_metadata.artifact_sha256_48h : short(prediction.recovery.model_metadata.artifact_sha256_48h)}</td></tr>
					<tr><td>icu_stay_time</td><td>xgboost</td><td>B_PLUS_F</td><td class="hash">{showFullHashes ? prediction.icu_stay_time.model_metadata.artifact_sha256 : short(prediction.icu_stay_time.model_metadata.artifact_sha256)}</td></tr>
					<tr><td>organ_support</td><td>xgboost</td><td>B_FULL</td><td class="hash">{showFullHashes ? prediction.organ_support.model_metadata.artifact_sha256 : short(prediction.organ_support.model_metadata.artifact_sha256)}</td></tr>
				</tbody>
			</table>
			<div class="chain-hashes">
				<div><span>selected_models_v2.json</span><b class="hash">{showFullHashes ? prediction.versions.selected_models_v2_sha256 : short(prediction.versions.selected_models_v2_sha256)}</b></div>
				<div><span>v2_model_freeze_v1.json</span><b class="hash">{showFullHashes ? prediction.versions.v2_model_freeze_sha256 : short(prediction.versions.v2_model_freeze_sha256)}</b></div>
			</div>
		</Panel>

		<Panel eyebrow="REPLAY" title="Mode / cutoff / schema">
			<table>
				<tbody>
					<tr><td>Replay mode</td><td>{prediction.mode}</td></tr>
					<tr><td>Prediction schema</td><td>{prediction.schema_version}</td></tr>
					<tr><td>Selected cutoff</td><td>{prediction.prediction_time}</td></tr>
					<tr><td>Elapsed ICU hours</td><td>{prediction.elapsed_icu_hours}</td></tr>
				</tbody>
			</table>
		</Panel>
	{/if}
</WorkbenchPage>

<style>
	.loading, .error { padding: 20px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.error { color: #fecdd3; }
	.controls { display: flex; gap: 8px; padding: 12px 0; border-block: 1px solid rgba(148,163,184,.14); margin-bottom: 16px; }
	.controls select { padding: 8px 12px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #cbd5e1; font: 9px 'JetBrains Mono', monospace; }
	.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 8px; }
	.semantics { margin: 0; color: #94a3b8; font-size: 12px; line-height: 1.7; }
	.hash-toggle { margin-bottom: 12px; padding: 7px 12px; border: 1px solid rgba(148,163,184,.2); background: #0a1220; color: #2bb8b0; font: 9px 'JetBrains Mono', monospace; letter-spacing: .06em; cursor: pointer; }
	table { width: 100%; border-collapse: collapse; font-size: 11px; }
	th { text-align: left; padding: 7px 8px; color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; border-bottom: 1px solid rgba(148,163,184,.16); }
	td { padding: 7px 8px; color: #cbd5e1; border-bottom: 1px solid rgba(148,163,184,.08); font: 10px 'JetBrains Mono', monospace; }
	td.hash, b.hash { color: #8fc6bf; }
	.chain-hashes { display: grid; gap: 8px; margin-top: 14px; padding-top: 14px; border-top: 1px solid rgba(148,163,184,.1); }
	.chain-hashes div { display: flex; justify-content: space-between; gap: 12px; }
	.chain-hashes span { color: #64748b; font: 9px 'JetBrains Mono', monospace; }
	.chain-hashes b { font: 9px 'JetBrains Mono', monospace; font-weight: 400; }
	@media (max-width: 700px) { .metrics { grid-template-columns: 1fr 1fr; } }
</style>
