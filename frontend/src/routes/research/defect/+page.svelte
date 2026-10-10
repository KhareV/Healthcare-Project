<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import { ChartFrame, BarChart } from '$lib/components/research';
	import { BUNDLE_STYLE } from '$lib/research/palette';
	import { fmtNum, fmtPct, fmtInt } from '$lib/research/format';
	import { researchApi, errorText, toProvenance, type Row, type Wrapped } from '$lib/services/research';

	let d = $state<Wrapped<Row> | null>(null);
	let error = $state('');
	onMount(async () => {
		try { d = await researchApi.defect(); } catch (c) { error = errorText(c); }
	});

	const rows = $derived<Row[]>(d?.data?.rows ?? []);
	const pick = (part: string, mode: string, task: string) => rows.find((r) => r.partition === part && r.mode === mode && r.task === task);
	const PARTS = ['LCE (development)', 'sealed V3 test'];
	const icu = (r?: Row) => r?.weighted_median_ae_hours as number | undefined;
	const sup = (r?: Row) => (r?.raw_auprc ?? r?.auprc) as number | undefined;
	const icuGroups = $derived(PARTS.map((p) => ({ label: p === 'sealed V3 test' ? 'Sealed V3 test' : 'LCE (development)', values: (['V2_AS_EVALUATED', 'V2_AS_SERVED'] as const).map((m) => ({ series: BUNDLE_STYLE[m].label, value: icu(pick(p, m, 'icu_stay_time')) as number })).filter((v) => Number.isFinite(v.value)) })));
	const supGroups = $derived(PARTS.map((p) => ({ label: p === 'sealed V3 test' ? 'Sealed V3 test (calibrated AUPRC)' : 'LCE (development, raw AUPRC)', values: (['V2_AS_EVALUATED', 'V2_AS_SERVED'] as const).map((m) => ({ series: BUNDLE_STYLE[m].label, value: sup(pick(p, m, 'organ_support')) as number })).filter((v) => Number.isFinite(v.value)) })));
	const colors = { [BUNDLE_STYLE.V2_AS_EVALUATED.label]: BUNDLE_STYLE.V2_AS_EVALUATED.color, [BUNDLE_STYLE.V2_AS_SERVED.label]: BUNDLE_STYLE.V2_AS_SERVED.color };
	const prov = $derived({ ...toProvenance(d?.provenance), status: 'DEFECT' as const });
	const v3Icu = $derived<number | undefined>(undefined);
</script>

<svelte:head><title>Defect D1 | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03.7 / DEFECT D1" title="The V2 current-SOFA leak, with evidence." description="Frozen V2 learned from the missingness of current SOFA, which encoded whether a stay ended within 24 hours. Its published ICU-time and support numbers are not valid serving performance." {error} loading={!d && !error}>
	{#if d}
		<p class="rnote warn"><b>Mechanism.</b> {d.data.explanation}. Serving always supplied SOFA(t), so deployed V2 behaves like “V2 as served”, not like the offline harness (“V2 as evaluated”).</p>
		<div class="rtiles">
			<MetricTile label="Development rows with null baseline SOFA (LCE)" value={fmtPct((d.data.fraction_rows_null_baseline_sofa_lce ?? 0) * 100, 1)} detail="null exactly when the 24h horizon was unavailable" tone="rose" />
			<MetricTile label="Null-pattern audit (V3 caches)" value={String(d.data.null_pattern_audit?.verdict ?? '—')} detail="no feature's missingness tracks horizon availability" tone="teal" />
		</div>
		<div class="rgrid two">
			<ChartFrame title="ICU remaining time: as evaluated vs as served" subtitle="Weighted median absolute error (hours) of the frozen V2 models; lower is better" provenance={prov} data={rows.filter((r) => r.task === 'icu_stay_time')} units="hours" empty={!icuGroups.some((g) => g.values.length)}>
				<BarChart groups={icuGroups} {colors} unit="h" valueFormat={(v) => `${v.toFixed(2)} h`} height={300} ariaLabel="ICU error as evaluated versus as served" />
			</ChartFrame>
			<ChartFrame title="Organ support: as evaluated vs as served" subtitle="AUPRC of the frozen V2 models; higher is better (note: each partition's prevalence differs)" provenance={prov} data={rows.filter((r) => r.task === 'organ_support')} units="AUPRC" empty={!supGroups.some((g) => g.values.length)}>
				<BarChart groups={supGroups} {colors} unit="" valueFormat={(v) => v.toFixed(3)} height={300} ariaLabel="Support AUPRC as evaluated versus as served" />
			</ChartFrame>
		</div>
		<Panel eyebrow="EVIDENCE TABLE" title="Frozen V2 under both input paths" note="V2 ARTIFACTS UNMODIFIED">
			<DataTable
				columns={[
					{ key: 'partition', label: 'Partition' },
					{ key: 'mode', label: 'Input path', fmt: (v) => (v === 'V2_AS_EVALUATED' ? 'As evaluated (D1, nulls kept)' : 'As served (SOFA(t) supplied)') },
					{ key: 'task', label: 'Task' },
					{ key: 'headline', label: 'Primary', align: 'right', fmt: (_v, r) => (r.task === 'icu_stay_time' ? `${fmtNum(r.weighted_median_ae_hours, 2)} h wMedAE` : r.task === 'organ_support' ? `AUPRC ${fmtNum(r.cal_auprc ?? r.auprc, 3)} · AUROC ${fmtNum(r.cal_auroc ?? r.auroc, 3)}` : `MAE ${fmtNum(r.mae, 3)}`) },
					{ key: 'n_rows', label: 'Rows', align: 'right', fmt: (v) => fmtInt(v) }
				]}
				{rows}
				rowClass={(r) => (r.mode === 'V2_AS_EVALUATED' && r.task !== 'recovery24' && r.task !== 'recovery48' ? 'flag' : '')}
			/>
		</Panel>
		<div class="rgrid two">
			<p class="rnote warn"><b>Why V2's published ICU/support numbers are not valid.</b> They were produced by an offline harness that preserved the null pattern. A model fed the real SOFA(t) (as the runtime does) loses that shortcut and performs much worse on ICU time and support. Recovery tasks are unaffected because eligible rows always have a value.</p>
			<p class="rnote"><b>Corrected behaviour in V3.</b> Current SOFA is the frozen scorer's value at the cutoff for every legal cutoff; a missing value is a hard error. A null-pattern audit shows no feature's missingness tracks horizon availability, and V3's own results are measured through this corrected path.</p>
		</div>
		<p class="rnote amber"><b>Caveat.</b> {d.data.caveat}. The V1 benchmark is unaffected (its flat feature map excludes current SOFA). Historical V2 artifacts remain byte-for-byte unchanged and are only annotated here.</p>
	{/if}
</ResearchPage>
