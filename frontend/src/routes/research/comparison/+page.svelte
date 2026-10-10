<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import Segmented from '$lib/components/research-pages/Segmented.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { ChartFrame, ForestPlot } from '$lib/components/research';
	import { fmtNum, fmtCI, fmtSigned } from '$lib/research/format';
	import { researchApi, errorText, taskLabel, toProvenance, TASKS, type Row, type Wrapped } from '$lib/services/research';

	let paired = $state<Wrapped<Row> | null>(null);
	let model = $state<Wrapped<Row> | null>(null);
	let hist = $state<Wrapped<Row> | null>(null);
	let error = $state('');
	let taskFilter = $state('all');
	let showEvaluated = $state(false);

	onMount(async () => {
		try {
			[paired, model, hist] = await Promise.all([researchApi.dataset('paired_differences'), researchApi.dataset('model_comparison'), researchApi.dataset('historical_context')]);
		} catch (cause) {
			error = errorText(cause);
		}
	});

	const pairedRows = $derived<Row[]>(paired?.data?.rows ?? []);
	const compRows = $derived<Row[]>((model?.data?.rows ?? []).filter((r: Row) => taskFilter === 'all' || r.task === taskFilter));
	const COMP_LABEL: Record<string, string> = { V3_vs_V2_AS_SERVED: 'V3 vs V2 as served (operational)', V3_vs_CONTROL: 'V3 vs V2-recipe control (method)', V3_vs_V2_AS_EVALUATED: 'V3 vs V2 as evaluated (compromised)' };
	const METRIC_LABEL: Record<string, string> = { mae: 'MAE', weighted_median_ae_hours: 'weighted median AE', auprc: 'AUPRC (cal.)', auroc: 'AUROC (cal.)', brier: 'Brier', rmse: 'RMSE', median_ae: 'median AE', log_loss: 'log loss' };

	function forest(comparison: string, picks: [string, string][]) {
		return picks.flatMap(([task, metric]) => {
			const r = pairedRows.find((x) => x.task === task && x.metric === metric && x.comparison === comparison);
			if (!r) return [];
			return [{ label: `${taskLabel(task)} · ${METRIC_LABEL[metric] ?? metric}`, estimate: r.diff_v3_minus_comparator, lo: r.ci95_lo, hi: r.ci95_hi, group: COMP_LABEL[comparison],
				note: `98.75% CI ${fmtCI(r.ci9875_lo, r.ci9875_hi, 3)} · ${r.n_rows} rows / ${r.n_stays} stays` }];
		});
	}
	const SOFA = [['recovery24', 'mae'], ['recovery48', 'mae']] as [string, string][];
	const HOURS = [['icu_stay_time', 'weighted_median_ae_hours'], ['icu_stay_time', 'mae']] as [string, string][];
	const PROB = [['organ_support', 'auprc'], ['organ_support', 'auroc'], ['organ_support', 'brier']] as [string, string][];
	const both = (picks: [string, string][]) => [...forest('V3_vs_V2_AS_SERVED', picks), ...forest('V3_vs_CONTROL', picks)];
	const evaluated = (picks: [string, string][]) => forest('V3_vs_V2_AS_EVALUATED', picks);

	const pairedProv = $derived(toProvenance(paired?.provenance, { ci_method: 'paired stay-cluster bootstrap, 2,000 replicates, 95% shown (98.75% in tooltip)' }));
	const modelProv = $derived(toProvenance(model?.provenance));
	const histProv = $derived(toProvenance(hist?.provenance));

	function cmpCell(r: Row, k: string) {
		const v = r[k];
		if (typeof v !== 'number') return '—';
		const ci = r[`${k}_ci95`];
		return ci ? `${fmtNum(v, 3)} ${fmtCI(ci[0], ci[1], 3)}` : fmtNum(v, 3);
	}
	const histCols = $derived<string[]>(hist?.data?.columns ?? Object.keys(hist?.data?.rows?.[0] ?? {}));
	const unitOf = (rows: [string, string][]) => (rows[0][0].startsWith('recovery') ? 'SOFA points' : rows[0][0] === 'icu_stay_time' ? 'hours' : 'metric units');
</script>

<svelte:head><title>Model Comparison | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03.1 / MODEL COMPARISON" title="Operational and modelling-method comparisons." description="V3 against what V2 would have served, against a repaired V2-recipe control, and against naive floors. Historical results are labelled and separated." {error} loading={!paired && !error}>
	{#if paired && model}
		<p class="rnote"><b>How to read this page.</b> <b>V3 vs V2 as served</b> is the operational comparison (what the deployed V2 runtime would have produced). <b>V3 vs control</b> isolates the modelling method (same recipe, corrected features and larger development data). Differences are V3 minus comparator: negative favours V3 for MAE/hours/Brier, positive favours V3 for AUROC/AUPRC.</p>

		<Panel eyebrow="TABLE" title="Sortable comparison on the sealed V3 test" note={`${compRows.length} METRICS`}>
			<Segmented label="Task" bind:value={taskFilter} options={[{ value: 'all', label: 'All tasks' }, ...TASKS.map((t) => ({ value: t.value, label: t.label }))]} />
			<label class="tog"><input type="checkbox" bind:checked={showEvaluated} /> show V2 as evaluated (historical, defect D1)</label>
			<DataTable
				columns={[
					{ key: 'task', label: 'Task', fmt: (v) => taskLabel(String(v)) },
					{ key: 'metric', label: 'Metric', fmt: (v, r) => `${METRIC_LABEL[v] ?? v} (${r.unit})` },
					{ key: 'naive', label: 'Naive', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'V3', label: 'V3', align: 'right', fmt: (_v, r) => cmpCell(r, 'V3') },
					{ key: 'V2_AS_SERVED', label: 'V2 as served', align: 'right', fmt: (_v, r) => cmpCell(r, 'V2_AS_SERVED') },
					{ key: 'CONTROL', label: 'Control', align: 'right', fmt: (_v, r) => cmpCell(r, 'CONTROL') },
					...(showEvaluated ? [{ key: 'V2_AS_EVALUATED', label: 'V2 as evaluated ▲', align: 'right' as const, fmt: (_v: any, r: Row) => cmpCell(r, 'V2_AS_EVALUATED') }] : []),
					{ key: 'denominator_rows', label: 'Rows', align: 'right' }
				]}
				rows={compRows}
				defaultSort="task"
			/>
		</Panel>

		<h3 class="rh">Paired differences with 95% stay-cluster CIs (V3 − comparator)</h3>
		<div class="rgrid three">
			<ChartFrame title="Recovery MAE" subtitle={`Difference in ${unitOf(SOFA)}; left of zero favours V3`} provenance={pairedProv} data={both(SOFA)} units="SOFA points" empty={!both(SOFA).length}>
				<ForestPlot rows={both(SOFA)} xLabel="V3 − comparator" xUnit="SOFA pts" favourLeft="favours V3" favourRight="favours comparator" format={(v) => fmtSigned(v, 3)} />
			</ChartFrame>
			<ChartFrame title="ICU remaining time" subtitle="Difference in hours; left of zero favours V3. V3 equals the control (zero by construction)." provenance={pairedProv} data={both(HOURS)} units="hours" empty={!both(HOURS).length}>
				<ForestPlot rows={both(HOURS)} xLabel="V3 − comparator" xUnit="h" favourLeft="favours V3" favourRight="favours comparator" format={(v) => fmtSigned(v, 2)} />
			</ChartFrame>
			<ChartFrame title="Organ-support initiation" subtitle="AUPRC/AUROC: right of zero favours V3. Brier: left of zero favours V3." provenance={pairedProv} data={both(PROB)} units="probability-scale metrics" empty={!both(PROB).length}>
				<ForestPlot rows={both(PROB)} xLabel="V3 − comparator" favourLeft="lower" favourRight="higher" format={(v) => fmtSigned(v, 3)} />
			</ChartFrame>
		</div>

		<details class="acc">
			<summary>Historical / defect analysis — V3 vs V2 as evaluated (compromised by defect D1)</summary>
			<div class="acc-body">
				<p class="rnote warn">The original V2 offline harness retained the missingness of current SOFA, which leaked whether the stay ended within 24 h. Its ICU-time and support numbers are therefore not valid estimates of served performance. They are shown for transparency only.</p>
				<div class="rgrid three">
					<ChartFrame title="Recovery MAE (V2 as evaluated)" provenance={{ ...pairedProv, status: 'DEFECT' }} data={evaluated(SOFA)} empty={!evaluated(SOFA).length}>
						<ForestPlot rows={evaluated(SOFA)} xLabel="V3 − V2 as evaluated" xUnit="SOFA pts" format={(v) => fmtSigned(v, 3)} />
					</ChartFrame>
					<ChartFrame title="ICU time (V2 as evaluated)" provenance={{ ...pairedProv, status: 'DEFECT' }} data={evaluated(HOURS)} empty={!evaluated(HOURS).length}>
						<ForestPlot rows={evaluated(HOURS)} xLabel="V3 − V2 as evaluated" xUnit="h" format={(v) => fmtSigned(v, 2)} />
					</ChartFrame>
					<ChartFrame title="Support (V2 as evaluated)" provenance={{ ...pairedProv, status: 'DEFECT' }} data={evaluated(PROB)} empty={!evaluated(PROB).length}>
						<ForestPlot rows={evaluated(PROB)} xLabel="V3 − V2 as evaluated" format={(v) => fmtSigned(v, 3)} />
					</ChartFrame>
				</div>
				<p class="rnote">V3 is <b>worse</b> than V2 as evaluated on ICU time and calibrated support AUPRC because that comparator contains the leak; this is the expected signature of defect D1, not a regression of V3.</p>
			</div>
		</details>

		{#if hist}
			<Panel eyebrow="HISTORICAL CONTEXT" title="V1 / V2 as published versus V3" note="HISTORICAL">
				<DataTable columns={histCols.map((c) => ({ key: c, label: c.replaceAll('_', ' ') }))} rows={hist.data.rows} />
				<p class="rnote warn" style="margin-top:12px">V1 and V2 columns are historical values as published on their own cohorts; they are not comparable row for row with the V3 sealed test and the V2 ICU/support values are affected by defect D1.</p>
			</Panel>
		{/if}
	{/if}
</ResearchPage>

<style>
	.tog{display:flex;gap:8px;align-items:center;margin:0 0 12px;color:#71829a;font:8px 'JetBrains Mono',monospace;letter-spacing:.08em}
	.tog input{accent-color:#2bb8b0}
</style>
