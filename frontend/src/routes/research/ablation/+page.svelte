<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import Segmented from '$lib/components/research-pages/Segmented.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { ChartFrame, HeatmapGrid, BarChart } from '$lib/components/research';
	import { fmtNum } from '$lib/research/format';
	import { researchApi, errorText, taskLabel, toProvenance, TASKS, type Row, type Wrapped } from '$lib/services/research';

	let ab = $state<Wrapped<Row> | null>(null);
	let error = $state('');
	let task = $state('recovery24');
	onMount(async () => {
		try { ab = await researchApi.ablation(); } catch (c) { error = errorText(c); }
	});

	const rows = $derived<Row[]>(ab?.data?.rows ?? []);
	const variants = $derived<string[]>([...new Set<string>(rows.map((r) => r.variant))]);
	const info = $derived(TASKS.find((t) => t.value === task)!);
	const control = (t: string) => rows.find((r) => r.task === t && r.variant === 'V3_CTRL')?.primary_mean as number | undefined;
	// Relative improvement vs the frozen-V2 context variant (V3_CTRL), sign-adjusted so + is always better.
	function rel(t: string, v: string): number | null {
		const r = rows.find((x) => x.task === t && x.variant === v);
		const c = control(t);
		if (!r || c === undefined || c === 0) return null;
		const lower = TASKS.find((x) => x.value === t)!.lowerIsBetter;
		return ((lower ? c - r.primary_mean : r.primary_mean - c) / Math.abs(c)) * 100;
	}
	const values = $derived(variants.map((v) => TASKS.map((t) => rel(t.value, v))));
	const bars = $derived([...rows.filter((r) => r.task === task)].map((r) => ({ label: r.variant, values: [{ series: info.metric, value: r.primary_mean as number, n: r.n_runs as number }] })));
	const defs = $derived<Record<string, string>>(ab?.data?.variant_definitions ?? {});
	const prov = $derived(toProvenance(ab?.provenance));
</script>

<svelte:head><title>Feature Ablation | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03.4 / FEATURE ABLATION" title="What the extra feature families added." description="Feature-group variants evaluated with the same cross-fitting. Most differences are within the pre-registered tie tolerance; the negative result is the finding." {error} loading={!ab && !error}>
	{#if ab}
		<p class="rnote amber"><b>Development evidence (cross-fitted, 3-fold screening, 2 seeds).</b> {ab.data.note}. Percentages are relative to the frozen-V2 context variant (V3_CTRL) and are sign-adjusted: positive always means better.</p>
		<ChartFrame title="Relative change vs V3_CTRL by feature variant and task" subtitle="Heatmap of sign-adjusted % improvement (positive = better than the frozen-V2 context)" provenance={prov} data={rows} units="% relative to V3_CTRL" empty={!variants.length}>
			<HeatmapGrid rows={variants} cols={TASKS.map((t) => t.label)} values={values as (number | null)[][]} unit="%" format={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`} cellHeight={34} colorLabel="% improvement vs V3_CTRL" ariaLabel="Ablation heatmap" />
		</ChartFrame>
		<Segmented label="Task" bind:value={task} options={TASKS.map((t) => ({ value: t.value, label: t.label }))} />
		<div class="rgrid two">
			<ChartFrame title={`${taskLabel(task)} — metric by variant`} subtitle={`${info.metric} (${info.unit}) · ${info.lowerIsBetter ? 'lower' : 'higher'} is better`} provenance={prov} data={rows.filter((r) => r.task === task)} units={info.unit} empty={!bars.length}>
				<BarChart groups={bars} orientation="horizontal" unit={info.unit} valueFormat={(v) => v.toFixed(3)} height={Math.max(260, bars.length * 34 + 60)} ariaLabel="Ablation by variant" />
			</ChartFrame>
			<Panel eyebrow="DEFINITIONS" title="Feature variants">
				<DataTable columns={[{ key: 'variant', label: 'Variant' }, { key: 'definition', label: 'Definition' }]} rows={Object.entries(defs).map(([variant, definition]) => ({ variant, definition }))} />
				<p class="rnote" style="margin-top:12px">V3_CTRL / V3_B / V3_BF keep the context features of V2 (with current SOFA corrected for defect D1). Windowed statistics (D), observed fractions (E), SOFA components and trend features (C) added at most a small, tolerance-level gain.</p>
			</Panel>
		</div>
		<Panel eyebrow="TABLE" title="All variants" note="CROSS-FITTED DEVELOPMENT">
			<DataTable
				columns={[
					{ key: 'task', label: 'Task', fmt: (v) => taskLabel(String(v)) },
					{ key: 'variant', label: 'Variant' },
					{ key: 'primary_metric', label: 'Metric' },
					{ key: 'primary_mean', label: 'Mean', align: 'right', fmt: (v) => fmtNum(v, 4) },
					{ key: 'n_runs', label: 'Runs', align: 'right' },
					{ key: 'rel', label: 'vs V3_CTRL', align: 'right', fmt: (_v, r) => { const x = rel(r.task, r.variant); return x === null ? '—' : `${x >= 0 ? '+' : ''}${x.toFixed(2)}%`; } }
				]}
				{rows}
				defaultSort="task"
			/>
		</Panel>
	{/if}
</ResearchPage>
