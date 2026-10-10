<script lang="ts">
	// CompletenessPanel — factual description of the INPUT window: observed bins, padding, missing
	// feature groups, density. Technical processability is not completeness, and neither is a
	// confidence score for the forecast.
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import GaugeRing from '$lib/components/dashboard/GaugeRing.svelte';
	import type { PredictionResponse } from '$lib/services/api';
	let { prediction }: { prediction: PredictionResponse } = $props();

	const GROUP_OF: Record<string, string> = {
		mean_arterial_pressure: 'Cardiovascular', heart_rate: 'Cardiovascular', systolic_blood_pressure: 'Cardiovascular', diastolic_blood_pressure: 'Cardiovascular',
		pao2: 'Respiratory', fio2: 'Respiratory', respiratory_rate: 'Respiratory', oxygen_saturation: 'Respiratory',
		creatinine: 'Renal', urine_output_volume: 'Renal', glasgow_coma_scale: 'Neurologic',
		platelet_count: 'Hematologic / liver', bilirubin_total: 'Hematologic / liver', temperature: 'Other measurements', lactate: 'Other measurements',
		vasopressor_on: 'Organ-support state', invasive_ventilation_on: 'Organ-support state',
		norepinephrine_rate: 'Vasopressor rates', epinephrine_rate: 'Vasopressor rates', dopamine_rate: 'Vasopressor rates', dobutamine_rate: 'Vasopressor rates'
	};
	const concept = (n: string) => n.replace(/__latest$|__sum$/g, '');
	const dq = $derived(prediction.data_quality);
	const rows = $derived.by(() => {
		const { channel_names, observation_mask, padding_mask } = prediction.temporal_window;
		const live = observation_mask.map((_, i) => i).filter((i) => !padding_mask[i]);
		const groups = new Map<string, { observed: number; total: number; never: string[] }>();
		channel_names.forEach((name, ci) => {
			const g = GROUP_OF[concept(name)] ?? 'Other';
			const entry = groups.get(g) ?? { observed: 0, total: 0, never: [] };
			let seen = 0;
			for (const bi of live) { entry.total += 1; if (observation_mask[bi]?.[ci]) { entry.observed += 1; seen += 1; } }
			if (live.length && seen === 0) entry.never.push(concept(name).replace(/_/g, ' '));
			groups.set(g, entry);
		});
		return [...groups.entries()].map(([group, v]) => ({ group, ...v, fraction: v.total ? v.observed / v.total : 0 }));
	});
	const neverObserved = $derived(rows.flatMap((r) => r.never));
</script>

<Panel eyebrow="INPUT COMPLETENESS" title="What the model could actually see" note="DESCRIBES THE INPUT — NOT FORECAST CONFIDENCE">
	<div class="layout">
		<GaugeRing value={Math.round((dq.observed_feature_fraction ?? 0) * 100)} size={92} stroke={9} color="#2bb8b0" label="OBSERVED" />
		<dl>
			<div><dt>Observed bins</dt><dd>{dq.observed_bins} of {dq.total_bins}</dd></div>
			<div><dt>Structural padding bins (before ICU admission)</dt><dd>{dq.padding_bins}</dd></div>
			<div><dt>Feature values observed</dt><dd>{dq.observed_feature_values} of {dq.total_feature_values}</dd></div>
			<div><dt>Missing values (not observed in their bin)</dt><dd>{dq.missing_feature_values}</dd></div>
			<div><dt>Measurements never observed in the window</dt><dd>{neverObserved.length ? neverObserved.join(', ') : 'none'}</dd></div>
		</dl>
	</div>
	<table>
		<thead><tr><th>Feature group</th><th>Observed (bin × channel)</th><th>Share</th></tr></thead>
		<tbody>
			{#each rows as r}
				<tr><td>{r.group}</td><td>{r.observed} / {r.total}</td><td><span class="bar"><i style={`width:${r.fraction * 100}%`}></i></span>{(r.fraction * 100).toFixed(0)}%</td></tr>
			{/each}
		</tbody>
	</table>
	<p class="note">A forecast can be computed from sparse input; sparsity limits what the model saw and is reported here as a fact about the data. It is not an uncertainty estimate or a model confidence. Component-level SOFA availability is not exposed by the prediction payload.</p>
</Panel>

<style>
	.layout { display: flex; gap: 22px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
	dl { display: grid; gap: 8px; margin: 0; flex: 1; min-width: 240px; }
	dl div { display: flex; justify-content: space-between; gap: 14px; border-bottom: 1px solid rgba(148,163,184,.08); padding-bottom: 5px; }
	dt { color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
	dd { margin: 0; color: #cbd5e1; font: 10px 'JetBrains Mono', monospace; text-align: right; }
	table { width: 100%; border-collapse: collapse; font: 10px 'JetBrains Mono', monospace; }
	th { text-align: left; padding: 6px 8px; color: #64748b; font-size: 8px; letter-spacing: .08em; border-bottom: 1px solid rgba(148,163,184,.16); }
	td { padding: 5px 8px; color: #cbd5e1; border-bottom: 1px solid rgba(148,163,184,.08); }
	.bar { display: inline-block; width: 90px; height: 6px; margin-right: 8px; background: rgba(148,163,184,.12); vertical-align: middle; }
	.bar i { display: block; height: 100%; background: #2bb8b0; }
	.note { margin: 12px 0 0; color: #64748b; font-size: 10px; line-height: 1.6; }
</style>
