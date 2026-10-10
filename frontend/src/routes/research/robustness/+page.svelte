<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import Segmented from '$lib/components/research-pages/Segmented.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { ChartFrame, BarChart, LineChart } from '$lib/components/research';
	import { bundleStyle } from '$lib/research/palette';
	import { fmtNum, fmtInt } from '$lib/research/format';
	import { researchApi, errorText, taskLabel, toProvenance, TASKS, type Row, type Wrapped } from '$lib/services/research';

	let rb = $state<Wrapped<Row> | null>(null);
	let reg = $state<Record<string, Wrapped<Row>> | null>(null);
	let error = $state('');
	let task = $state('recovery24');
	onMount(async () => {
		try { [rb, reg] = await Promise.all([researchApi.robustness(), researchApi.regression()]); } catch (c) { error = errorText(c); }
	});

	const rows = $derived<Row[]>((rb?.data?.rows ?? []).filter((r: Row) => r.task === task));
	const info = $derived(TASKS.find((t) => t.value === task)!);
	const MODELS = ['V3', 'CONTROL', 'V2_AS_SERVED'];
	const SCEN_LABEL: Record<string, string> = { extra_padding_oldest_1: '+1 oldest bin padded', extra_padding_oldest_2: '+2 oldest bins padded', extra_padding_oldest_3: '+3 oldest bins padded', noise_5pct: 'noise 5%', noise_10pct: 'noise 10%', noise_20pct: 'noise 20%', outage_labs: 'lab outage', outage_vitals: 'vital outage' };
	const scenarios = Object.keys(SCEN_LABEL);
	const deltaGroups = $derived(scenarios.map((s) => ({ label: SCEN_LABEL[s], values: MODELS.map((m) => {
		const r = rows.find((x) => x.scenario === s && x.model === m);
		return r ? { series: bundleStyle(m).label, value: r.delta_vs_unperturbed as number, n: r.n_rows as number } : null;
	}).filter(Boolean) as { series: string; value: number; n: number }[] })));
	const colors = Object.fromEntries(MODELS.map((m) => [bundleStyle(m).label, bundleStyle(m).color]));
	const noise = $derived(MODELS.map((m) => ({ name: bundleStyle(m).label, color: bundleStyle(m).color, dashed: m === 'V2_AS_SERVED',
		points: [['unperturbed', 0], ['noise_5pct', 5], ['noise_10pct', 10], ['noise_20pct', 20]].map(([s, x]) => {
			const r = rows.find((q) => q.scenario === s && q.model === m);
			return { x: x as number, y: r ? (r.value as number) : NaN };
		}) })));
	const base = $derived(rows.filter((r) => r.scenario === 'unperturbed'));

	const SLICE_DIMS = [['cardiac_subtype', 'Cardiac subtype'], ['cutoff_band', 'Cutoff band']] as const;
	let dim = $state<string>('cardiac_subtype');
	const sliceRows = $derived<Row[]>((reg?.error_slices?.data?.rows ?? []).filter((r: Row) => r.task === task && r.dimension === dim && (r.model === 'V3' || r.model === 'CONTROL' || r.model === 'V2_AS_SERVED')));
	const sliceGroups = $derived([...new Set<string>(sliceRows.map((r) => r.slice))].map((l) => ({ label: l, values: MODELS.map((m) => {
		const r = sliceRows.find((x) => x.slice === l && x.model === m);
		return r ? { series: bundleStyle(m).label, value: r.value as number, n: r.n_stays as number } : null;
	}).filter(Boolean) as { series: string; value: number; n: number }[] })));
	const p = $derived(toProvenance(rb?.provenance));
	const pSlice = $derived(toProvenance(reg?.error_slices?.provenance));
</script>

<svelte:head><title>Robustness | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03.5 / ROBUSTNESS" title="Development stress tests." description="Controlled perturbations of held-out development records. These are synthetic stress tests, not evidence of real-world prospective robustness." {error} loading={!rb && !error}>
	{#if rb}
		<p class="rnote amber"><b>Scope.</b> {rb.data.statement}. Sample: {rb.data.cohort}, {fmtInt(rb.data.n_stays)} stays / {fmtInt(rb.data.n_rows)} rows. {rb.data.limitation}.</p>
		<Segmented label="Task" bind:value={task} options={TASKS.map((t) => ({ value: t.value, label: t.label }))} />
		<div class="rgrid two">
			<ChartFrame title="Degradation by scenario" subtitle={`Change in ${info.metric} (${info.unit}) vs unperturbed · ${info.lowerIsBetter ? 'positive = worse' : 'negative = worse'}`} provenance={p} data={rows} units={info.unit} empty={!rows.length}>
				<BarChart groups={deltaGroups} colors={colors} unit={info.unit} valueFormat={(v) => (v >= 0 ? '+' : '') + v.toFixed(3)} height={340} ariaLabel="Robustness degradation" />
			</ChartFrame>
			<ChartFrame title="Measurement-noise sweep" subtitle={`${info.metric} (${info.unit}) vs multiplicative noise level on observed values`} provenance={p} data={rows.filter((r) => String(r.scenario).startsWith('noise') || r.scenario === 'unperturbed')} units={info.unit} empty={!noise[0].points.length}>
				<LineChart series={noise} xLabel="Noise sd (% of value)" yLabel={info.unit} xUnit="%" ariaLabel="Noise sweep" />
			</ChartFrame>
		</div>
		<h3 class="rh">Stratified performance on the sealed test</h3>
		<Segmented label="Slice" bind:value={dim} options={SLICE_DIMS.map(([v, l]) => ({ value: v, label: l }))} />
		<ChartFrame title={`${taskLabel(task)} by ${dim.replaceAll('_', ' ')}`} subtitle={`Primary metric (${info.unit}) per prespecified slice · final sealed-test evidence, not a stress test`} provenance={pSlice} data={sliceRows} units={info.unit} empty={!sliceGroups.length}>
			<BarChart groups={sliceGroups} colors={colors} unit={info.unit} valueFormat={(v) => v.toFixed(3)} height={300} ariaLabel="Stratified performance" />
		</ChartFrame>
		<Panel eyebrow="TABLE" title={`Unperturbed reference — ${taskLabel(task)}`} note={`${base[0]?.n_rows ?? ''} ROWS`}>
			<DataTable columns={[{ key: 'model', label: 'Bundle', fmt: (v) => bundleStyle(String(v)).label }, { key: 'metric', label: 'Metric' }, { key: 'value', label: `Value (${info.unit})`, align: 'right', fmt: (v) => fmtNum(v, 4) }, { key: 'n_rows', label: 'Rows', align: 'right' }]} rows={base} />
			<p class="rnote" style="margin-top:12px">Bars show measured differences on the development subsample, per task and bundle; they are not projections to other populations or to prospective use.</p>
		</Panel>
	{/if}
</ResearchPage>
