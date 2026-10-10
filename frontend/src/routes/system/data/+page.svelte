<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import { api, request, type DemoSubject, type PredictionResponse } from '$lib/services/api';

	let subjects = $state<DemoSubject[]>([]);
	let stayId = $state('');
	let cutoffIndex = $state(0);
	let prediction = $state<PredictionResponse | null>(null);
	let loading = $state(true);
	let error = $state('');
	let showFullHashes = $state(false);
	type Prov = { data: { bundle_id: string; status: string; rows: { task: string; family: string; feature_variant: string; model_sha256: string; n_estimators: number; postprocess: string }[]; calibrator: { method: string; params_sha256: string }; threshold: { value: number; comparator: string }; file_sha256: Record<string, string> } };
	type D1Row = { mode: string; task: string; weighted_median_ae_hours?: number; raw_auprc?: number; partition: string };
	let prov = $state<Prov | null>(null);
	let health = $state<Awaited<ReturnType<typeof api.health>> | null>(null);
	let d1 = $state<D1Row[]>([]);
	let v3Error = $state('');

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
			const [h, pv, dd] = await Promise.all([api.health(), request<Prov>('/research/provenance'), request<{ data: { rows: D1Row[] } }>('/research/defect-d1')]);
			health = h; prov = pv; d1 = dd.data.rows;
			if (h.release !== 'V3') v3Error = 'The serving API did not report a V3 bundle. Forecasts are disabled; no other model is substituted.';
		} catch (cause) {
			v3Error = 'V3 forecasting model unavailable — forecasts are disabled. ' + (cause instanceof Error ? cause.message : '');
		}
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
	{#if v3Error}<div class="error failsafe" role="alert" data-testid="v3-unavailable">V3 forecasting model unavailable — forecasts are disabled. {v3Error}</div>{/if}
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

		<Panel eyebrow="V3 BUNDLE / PROVENANCE" title="The release that produced this forecast" note={showFullHashes ? 'FULL HASHES' : 'SHORT HASHES'}>
			<button class="hash-toggle" onclick={() => (showFullHashes = !showFullHashes)}>{showFullHashes ? 'Show short hashes' : 'Show full hashes'}</button>
			{#if prediction.model_bundle}
				<div class="chain-hashes first">
					<div><span>release / bundle</span><b data-testid="bundle-badge">{prediction.model_bundle.release} · {prediction.model_bundle.bundle_id} · {prediction.model_bundle.status ?? ''}</b></div>
					<div><span>selected_models_v3.json</span><b class="hash">{showFullHashes ? prediction.model_bundle.selected_models_sha256 : short(prediction.model_bundle.selected_models_sha256)}</b></div>
					<div><span>support calibrator</span><b>{prediction.model_bundle.calibrator.method} · {short(prediction.model_bundle.calibrator.params_sha256)}</b></div>
					<div><span>frozen alert threshold (≥)</span><b>{prediction.model_bundle.threshold}</b></div>
				</div>
			{:else}
				<p class="semantics">This response carries no V3 bundle identity, so it is not displayed as a V3 forecast.</p>
			{/if}
			{#if prov}
				<table>
					<thead><tr><th>Task</th><th>Family</th><th>Feature variant</th><th>Trees</th><th>Model hash</th></tr></thead>
					<tbody>
						{#each prov.data.rows as r}
							<tr><td>{r.task}</td><td>{r.family}</td><td>{r.feature_variant}</td><td>{r.n_estimators}</td><td class="hash">{showFullHashes ? r.model_sha256 : short(r.model_sha256)}</td></tr>
						{/each}
					</tbody>
				</table>
				<div class="chain-hashes">
					{#each [['scientific_protocol_v1', 'protocol_v1'], ['protocol_amendment_v1_1', 'protocol_amendment_v1_1'], ['v3_final_evaluation_freeze', 'evaluation_freeze'], ['final_metrics_v3', 'final_metrics_v3'], ['research_registry', 'research_registry']] as [label, key]}
						{#if prov.data.file_sha256[key]}<div><span>{label}</span><b class="hash">{showFullHashes ? prov.data.file_sha256[key] : short(prov.data.file_sha256[key])}</b></div>{/if}
					{/each}
				</div>
			{/if}
		</Panel>

		<Panel eyebrow="DEFECT D1 / DISCLOSURE" title="Why V3 replaced the V2 serving models">
			<p class="semantics">Frozen V2's <code>current_sofa</code> feature was missing exactly on rows whose 24-hour follow-up did not exist, which leaked future availability into its offline evaluation. The V2 serving runtime always supplied SOFA, so the deployed V2 behaviour differs from its published numbers. V3 computes SOFA(t) from data at or before the cutoff for every legal cutoff. This product serves V3 only; there is no fallback to V2.</p>
		</Panel>

		<Panel eyebrow="HISTORICAL V2 / NOT SERVED" title="Frozen V2 evidence kept for comparison only" note="DEVELOPMENT COHORT LCE · DEFECT-D1 ANALYSIS">
			<table>
				<thead><tr><th>Mode</th><th>Task</th><th>ICU weighted median AE (h)</th><th>Support raw AUPRC</th></tr></thead>
				<tbody>
					{#each d1.filter((r) => r.task === 'icu_stay_time' || r.task === 'organ_support') as r}
						<tr><td>{r.mode.replaceAll('_', ' ').toLowerCase()}</td><td>{r.task}</td><td>{r.weighted_median_ae_hours?.toFixed(2) ?? '—'}</td><td>{r.raw_auprc?.toFixed(3) ?? '—'}</td></tr>
					{/each}
				</tbody>
			</table>
			<p class="semantics">“V2 as evaluated” reproduces the original offline harness containing defect D1 and must not be read as serving performance. “V2 as served” is what the V2 runtime actually produced.</p>
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
	.failsafe { border: 1px solid rgba(251,113,133,.4); background: rgba(251,113,133,.08); padding: 12px 14px; margin-bottom: 12px; }
	.chain-hashes.first { margin: 0 0 14px; padding: 0 0 14px; border-top: 0; border-bottom: 1px solid rgba(148,163,184,.1); }
	code { color: #8fc6bf; font: 10px 'JetBrains Mono', monospace; }
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
