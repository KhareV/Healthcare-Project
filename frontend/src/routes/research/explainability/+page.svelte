<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import Segmented from '$lib/components/research-pages/Segmented.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { ChartFrame, BarChart } from '$lib/components/research';
	import { fmtNum, fmtInt } from '$lib/research/format';
	import { researchApi, errorText, taskLabel, toProvenance, TASKS, type Row, type Wrapped } from '$lib/services/research';

	let ex = $state<Wrapped<Row> | null>(null);
	let error = $state('');
	let task = $state('organ_support');
	onMount(async () => {
		try { ex = await researchApi.globalExplain(); } catch (c) { error = errorText(c); }
	});

	const rows = $derived<Row[]>((ex?.data?.rows ?? []).filter((r: Row) => r.task === task).sort((a: Row, b: Row) => a.rank - b.rank));
	const family = $derived<string>(rows[0]?.family ?? '');
	function group(feature: string): string {
		if (/^A__static/.test(feature)) return 'Static (age, sex, cardiac group)';
		if (/^A__bin\d__.+__value$/.test(feature)) return 'Temporal value (latest in 6h bin)';
		if (/^A__bin\d__.+__mask$/.test(feature)) return 'Observation mask';
		if (/^A__bin\d__.+__tslo$/.test(feature)) return 'Time since last observation';
		if (feature.startsWith('B__')) return 'Context (elapsed time, current SOFA)';
		if (feature.startsWith('D__')) return 'Windowed statistics (12/24/48h)';
		if (feature.startsWith('E__')) return 'Observed fraction';
		if (feature.startsWith('F__')) return 'Support-state history';
		return 'Other';
	}
	const topGroups = $derived(rows.map((r) => ({ label: `${r.rank}. ${String(r.readable).replace(' [value]', '').slice(0, 52)}`, values: [{ series: 'mean |SHAP|', value: r.mean_abs_shap as number }] })));
	const familyGroups = $derived.by(() => {
		const acc = new Map<string, number>();
		for (const r of rows) acc.set(group(r.feature), (acc.get(group(r.feature)) ?? 0) + (r.mean_abs_shap as number));
		return [...acc.entries()].sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label, values: [{ series: 'sum of top-20 mean |SHAP|', value }] }));
	});
	const prov = $derived(toProvenance(ex?.provenance));
	const marginUnit = $derived(task === 'icu_stay_time' ? 'log1p(hours) margin' : task === 'organ_support' ? 'log-odds margin' : 'SOFA points (delta)');
	const FAMILY_NOTE: Record<string, string> = { xgboost: 'XGBoost · TreeSHAP (tree-path-dependent, raw margin)', catboost: 'CatBoost · TreeSHAP (raw margin)' };
</script>

<svelte:head><title>Explainability | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03.6 / EXPLAINABILITY" title="Which inputs contributed to the forecasts." description="Global TreeSHAP importance of the four selected V3 heads on a held-out development sample. Attributions describe model contributions, not causes." {error} loading={!ex && !error}>
	{#if ex}
		<p class="rnote"><b>Output domain.</b> {ex.data.note}. Wording: features <b>{ex.data.interpretation_wording}</b>. They do not cause deterioration or recovery and do not imply treatment. Additivity check: max |Σ SHAP + base − margin| = {ex.data.additivity_max_abs_error?.toExponential(1)}. Sample: {ex.data.sample?.cohort}, {fmtInt(ex.data.sample?.rows)} rows (development, not the sealed test).</p>
		<Segmented label="Task head" bind:value={task} options={TASKS.map((t) => ({ value: t.value, label: t.label }))} />
		<p class="rmeta"><span>MODEL FAMILY / {FAMILY_NOTE[family] ?? family}</span><span>MARGIN UNIT / {marginUnit}</span></p>
		<div class="rgrid two">
			<ChartFrame title={`Top-20 features — ${taskLabel(task)}`} subtitle={`Mean |SHAP| of the raw margin (${marginUnit}); higher = larger average contribution`} provenance={prov} data={rows} units={marginUnit} empty={!topGroups.length}>
				<BarChart groups={topGroups} orientation="horizontal" unit="" valueFormat={(v) => v.toFixed(4)} showValues height={Math.max(420, topGroups.length * 24 + 60)} ariaLabel="Global SHAP importance" />
			</ChartFrame>
			<ChartFrame title="Importance by feature family" subtitle="Sum of mean |SHAP| of the top-20 features, grouped by feature family (within the top 20 only)" provenance={prov} data={familyGroups.map((g) => ({ family: g.label, value: g.values[0].value }))} units={marginUnit} empty={!familyGroups.length}>
				<BarChart groups={familyGroups} orientation="horizontal" unit="" valueFormat={(v) => v.toFixed(4)} height={Math.max(260, familyGroups.length * 40 + 60)} ariaLabel="Importance by feature family" />
			</ChartFrame>
		</div>
		<Panel eyebrow="TABLE" title="Ranked contributors" note={family.toUpperCase()}>
			<DataTable columns={[{ key: 'rank', label: '#', align: 'right' }, { key: 'readable', label: 'Feature' }, { key: 'feature', label: 'Column' }, { key: 'mean_abs_shap', label: 'Mean |SHAP|', align: 'right', fmt: (v) => fmtNum(v, 5) }]} {rows} defaultSort="rank" />
			<p class="rnote warn" style="margin-top:12px">For organ support the explanation is of the raw XGBoost margin <i>before</i> Platt calibration; the calibrated probability is a monotone transform of that margin and is not itself a SHAP-additive quantity. ICU contributions are on the log1p(hours) margin before back-transformation.</p>
		</Panel>
	{/if}
</ResearchPage>
