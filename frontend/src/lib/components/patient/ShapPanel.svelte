<script lang="ts">
	// ShapPanel — task-specific TreeSHAP view: signed contributions, feature-group and temporal-bin
	// aggregation, and how the top contributors changed across computed cutoffs. Raw-margin domain
	// is always stated. Attribution is statistical ("contributed to the prediction"), never causal.
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import BarContrib from '$lib/components/dashboard/BarContrib.svelte';
	import type { PredictionResponse, TaskExplanation } from '$lib/services/api';

	type Task = 'recovery24' | 'recovery48' | 'icu_stay_time' | 'organ_support';
	let { prediction, computed = {}, cutoffIndex = 0, task = $bindable('organ_support' as Task), showSelector = true }:
		{ prediction: PredictionResponse; computed?: Record<number, PredictionResponse>; cutoffIndex?: number; task?: Task; showSelector?: boolean } = $props();

	const TITLES: Record<Task, string> = { recovery24: 'Recovery +24h', recovery48: 'Recovery +48h', icu_stay_time: 'Remaining ICU time', organ_support: 'Organ-support risk' };
	const DOMAIN: Record<Task, string> = {
		recovery24: 'Raw ΔSOFA(+24h) model output, in SOFA units, before the display reconstruction clip.',
		recovery48: 'Raw ΔSOFA(+48h) model output, in SOFA units, before the display reconstruction clip.',
		icu_stay_time: 'Raw log1p(remaining hours) margin — before the non-negative clamp and expm1 to hours.',
		organ_support: 'Raw model margin (log-odds) before Platt calibration — calibration is not decomposed here.'
	};
	type Ext = Omit<TaskExplanation, 'diagnostics'> & { diagnostics: TaskExplanation['diagnostics'] & { model_family?: string; wording?: string; output_domain_code?: string }; by_feature_group?: { group: string; signed_sum: number; abs_sum: number }[]; by_temporal_bin?: { bin: number; hours_ago_start: number; hours_ago_end: number; signed_sum: number; abs_sum: number }[] };
	const exp = $derived(prediction.explanations[task] as Ext);
	const bars = $derived.by(() => {
		const rows = [...exp.top_positive_contributors, ...exp.top_negative_contributors].map((c) => ({ name: c.label, value: c.attribution }));
		return rows.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 12);
	});
	const maxAbs = $derived(Math.max(0.05, ...bars.map((b) => Math.abs(b.value))));
	const groups = $derived((exp.by_feature_group ?? []).filter((g) => g.abs_sum > 0).sort((a, b) => b.abs_sum - a.abs_sum));
	const bins = $derived(exp.by_temporal_bin ?? []);
	const binMax = $derived(Math.max(1e-9, ...bins.map((b) => b.abs_sum)));
	const groupMax = $derived(Math.max(1e-9, ...groups.map((g) => g.abs_sum)));

	// change across cutoffs: union of top-4 features at each computed cutoff <= selected
	const evolution = $derived.by(() => {
		const idxs = Object.keys(computed).map(Number).filter((i) => i <= cutoffIndex).sort((a, b) => a - b);
		const names = new Map<string, string>();
		const per = idxs.map((i) => {
			const e = computed[i].explanations[task];
			const all = [...e.top_positive_contributors, ...e.top_negative_contributors];
			const top = [...all].sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution)).slice(0, 4);
			top.forEach((c) => names.set(c.feature_name, c.label));
			return { i, map: new Map(all.map((c) => [c.feature_name, c.attribution])) };
		});
		return { idxs, rows: [...names.entries()].map(([k, label]) => ({ label, values: per.map((p) => p.map.get(k)) })) };
	});
	const sgn = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;
</script>

<Panel eyebrow={`TREESHAP / ${task.toUpperCase()}`} title={`Features that contributed to the ${TITLES[task]} forecast`} note={exp.diagnostics.additivity_check_passed ? 'ADDITIVITY VERIFIED' : 'ADDITIVITY FAILED'}>
	{#if showSelector}
		<div class="tabs" role="tablist" aria-label="Forecast head">
			{#each Object.keys(TITLES) as t}
				<button role="tab" aria-selected={task === t} class:active={task === t} onclick={() => (task = t as Task)}>{TITLES[t as Task]}<small>{prediction.model_bundle?.tasks?.[t]?.family ?? ''}</small></button>
			{/each}
		</div>
	{/if}
	<p class="domain"><b>Output domain:</b> {DOMAIN[task]} <b>Method:</b> {exp.diagnostics.method.split('(')[0].trim()} on the {exp.diagnostics.model_family ?? 'tree'} model · base value {exp.diagnostics.base_value.toFixed(3)} → margin {exp.diagnostics.raw_margin_prediction.toFixed(3)}.</p>
	<div class="cols">
		<div>
			<h4>Signed contributions (positive raises this output)</h4>
			<BarContrib features={bars} {maxAbs} height={Math.max(220, bars.length * 26)} />
		</div>
		<div>
			{#if groups.length}
				<h4>By feature group (|SHAP| sum)</h4>
				{#each groups as g}
					<div class="meter"><span>{g.group}</span><i style={`width:${(g.abs_sum / groupMax) * 100}%`}></i><em>{g.abs_sum.toFixed(3)}</em></div>
				{/each}
			{/if}
			{#if bins.length}
				<h4>By temporal bin (|SHAP| sum, hours before cutoff)</h4>
				{#each bins as b}
					<div class="meter"><span>{b.hours_ago_end === 0 ? 'last 6h' : `${b.hours_ago_end}–${b.hours_ago_start}h ago`}</span><i class="bin" style={`width:${(b.abs_sum / binMax) * 100}%`}></i><em>{b.abs_sum.toFixed(3)}</em></div>
				{/each}
			{/if}
		</div>
	</div>
	{#if evolution.idxs.length > 1}
		<h4 class="evo">How the leading contributors changed across computed cutoffs</h4>
		<div class="scroll">
			<table>
				<thead><tr><th>Feature</th>{#each evolution.idxs as i}<th>t{i}</th>{/each}</tr></thead>
				<tbody>{#each evolution.rows as r}<tr><td>{r.label}</td>{#each r.values as v}<td class:pos={v !== undefined && v > 0} class:neg={v !== undefined && v < 0}>{v === undefined ? '—' : sgn(v)}</td>{/each}</tr>{/each}</tbody>
			</table>
		</div>
	{/if}
	<p class="note">{exp.diagnostics.wording ?? 'Statistical attribution of this model output — not a causal explanation.'} Values are for this patient at this cutoff only.</p>
</Panel>

<style>
	.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
	.tabs button { display: grid; gap: 3px; padding: 9px 13px; border: 1px solid rgba(148,163,184,.18); background: #080e1d; color: #94a3b8; font: 10px 'Space Grotesk', sans-serif; cursor: pointer; text-align: left; }
	.tabs button small { color: #64748b; font: 7px 'JetBrains Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
	.tabs button.active { border-color: rgba(43,184,176,.5); background: rgba(43,184,176,.08); color: #eef7f6; }
	.domain { margin: 0 0 12px; color: #8494a9; font-size: 10px; line-height: 1.6; }
	.domain b { color: #cbd5e1; font-weight: 500; }
	.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 22px; }
	h4 { margin: 4px 0 10px; color: #2bb8b0; font: 9px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; }
	h4.evo { margin-top: 18px; }
	.meter { display: grid; grid-template-columns: 150px 1fr 54px; align-items: center; gap: 8px; margin-bottom: 6px; color: #94a3b8; font: 9px 'JetBrains Mono', monospace; }
	.meter i { display: block; height: 8px; background: #2bb8b0; border-radius: 2px; min-width: 1px; }
	.meter i.bin { background: #38bdf8; }
	.meter em { font-style: normal; text-align: right; color: #cbd5e1; }
	.scroll { overflow-x: auto; }
	table { width: 100%; border-collapse: collapse; font: 10px 'JetBrains Mono', monospace; }
	th, td { padding: 5px 8px; text-align: right; border-bottom: 1px solid rgba(148,163,184,.08); color: #94a3b8; white-space: nowrap; }
	th:first-child, td:first-child { text-align: left; color: #cbd5e1; }
	td.pos { color: #60a5fa; } td.neg { color: #f87171; }
	.note { margin: 12px 0 0; color: #64748b; font-size: 10px; line-height: 1.6; }
</style>
