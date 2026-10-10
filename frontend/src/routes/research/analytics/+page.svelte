<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { ChartFrame, BarChart } from '$lib/components/research';
	import { BUNDLE_STYLE } from '$lib/research/palette';
	import { fmtNum, fmtCI, fmtInt, shortHash } from '$lib/research/format';
	import { researchApi, errorText, taskLabel, type Row } from '$lib/services/research';

	let summary = $state<Row | null>(null);
	let prov = $state<Row | null>(null);
	let paired = $state<Row | null>(null);
	let error = $state('');

	onMount(async () => {
		try {
			[summary, prov, paired] = await Promise.all([researchApi.summary(), researchApi.provenance(), researchApi.dataset('paired_differences')]);
		} catch (cause) {
			error = errorText(cause);
		}
	});

	const H = $derived(summary?.headline ?? {});
	const rows = $derived<Row[]>(summary?.comparison?.rows ?? []);
	const bundleKeys = ['V3', 'V2_AS_SERVED', 'V2_AS_EVALUATED', 'CONTROL'] as const;

	function cell(row: Row, key: string) {
		const v = row[key];
		if (typeof v !== 'number') return '—';
		const ci = row[`${key}_ci95`];
		return ci ? `${fmtNum(v, 3)} ${fmtCI(ci[0], ci[1], 3)}` : fmtNum(v, 3);
	}
	function improvement(row: Row, key: string): number | null {
		const v = row[key], n = row.naive;
		if (typeof v !== 'number' || typeof n !== 'number' || n === 0) return null;
		return (row.lower_is_better ? (n - v) / n : (v - n) / n) * 100;
	}
	const improvementGroups = $derived(
		rows.map((r) => ({
			label: `${taskLabel(r.task)} (${r.metric})`,
			values: (['V3', 'V2_AS_SERVED', 'CONTROL'] as const)
				.map((k) => ({ series: BUNDLE_STYLE[k].label, value: improvement(r, k) as number }))
				.filter((x) => x.value !== null)
		}))
	);
	const barColors = { [BUNDLE_STYLE.V3.label]: BUNDLE_STYLE.V3.color, [BUNDLE_STYLE.V2_AS_SERVED.label]: BUNDLE_STYLE.V2_AS_SERVED.color, [BUNDLE_STYLE.CONTROL.label]: BUNDLE_STYLE.CONTROL.color };

	const icuRow = $derived(rows.find((r) => r.task === 'icu_stay_time'));
	const icuEqualsControl = $derived(!!icuRow && Math.abs(icuRow.V3 - icuRow.CONTROL) < 1e-9);
	const r48ctrl = $derived<Row | undefined>(paired?.data?.rows?.find((r: Row) => r.task === 'recovery48' && r.metric === 'mae' && r.comparison === 'V3_vs_CONTROL'));
	const r48notEstablished = $derived(!!r48ctrl && r48ctrl.ci9875_lo <= 0 && r48ctrl.ci9875_hi >= 0);
	const tasks = $derived<Row>(prov?.data?.tasks ?? {});
	const finalProv = $derived({ status: 'FINAL' as const, bundle: 'V3', partition: summary?.partition, n_rows: H.icu_weighted_median_ae_hours?.n_rows ?? null, n_stays: H.icu_weighted_median_ae_hours?.n_stays ?? null, metric: 'stay-balanced primary metrics, 95% paired stay-cluster bootstrap', source: summary?.source_artifacts?.[0]?.path, sha256: summary?.source_artifacts?.[0]?.sha256, ci_method: `${summary?.ci_method?.n ?? 'unreported'} replicates, seed ${summary?.ci_method?.seed ?? 'unreported'}` });
</script>

<svelte:head><title>Model Research | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03 / MODEL RESEARCH" title="The corrected V3 release, with its evidence." description="Every number is read from frozen artifacts and carries its evidence status. V3 is the final scientific evaluation; V2 results are shown only with their defect-D1 limitation." legend loading={!summary && !error} {error}>
	{#if summary}
		<div class="rmeta"><span>PARTITION / {summary.partition}</span><span>BUNDLE / V3 · {prov?.data?.bundle_id}</span><span>CI / {summary.ci_method?.n} STAY-CLUSTER BOOTSTRAP REPLICATES</span><span>STATUS / FINAL SEALED TEST (ONE ACCESS)</span></div>

		<div class="rtiles">
			<MetricTile label="Recovery +24h MAE" value={fmtNum(H.recovery24_mae?.value, 4)} unit="SOFA pts" detail={`95% CI ${fmtCI(H.recovery24_mae?.ci95?.[0], H.recovery24_mae?.ci95?.[1], 4)} · ${fmtInt(H.recovery24_mae?.n_rows)} rows / ${fmtInt(H.recovery24_mae?.n_stays)} stays`} tone="teal" />
			<MetricTile label="Recovery +48h MAE" value={fmtNum(H.recovery48_mae?.value, 4)} unit="SOFA pts" detail={`95% CI ${fmtCI(H.recovery48_mae?.ci95?.[0], H.recovery48_mae?.ci95?.[1], 4)} · ${fmtInt(H.recovery48_mae?.n_rows)} rows / ${fmtInt(H.recovery48_mae?.n_stays)} stays`} tone="teal" />
			<MetricTile label="ICU weighted median AE" value={fmtNum(H.icu_weighted_median_ae_hours?.value, 3)} unit="h" detail={`95% CI ${fmtCI(H.icu_weighted_median_ae_hours?.ci95?.[0], H.icu_weighted_median_ae_hours?.ci95?.[1], 3)} · ${fmtInt(H.icu_weighted_median_ae_hours?.n_stays)} stays`} tone="cyan" />
			<MetricTile label="Support AUROC (calibrated)" value={fmtNum(H.support_calibrated_auroc?.value, 4)} detail={`${fmtInt(H.support_calibrated_auroc?.n_rows)} rows / ${fmtInt(H.support_calibrated_auroc?.n_stays)} stays`} tone="amber" />
			<MetricTile label="Support AUPRC (calibrated)" value={fmtNum(H.support_calibrated_auprc?.value, 4)} detail={`95% CI ${fmtCI(H.support_calibrated_auprc?.ci95?.[0], H.support_calibrated_auprc?.ci95?.[1], 4)} · prevalence-dependent`} tone="amber" />
			<MetricTile label="Support Brier (calibrated)" value={fmtNum(H.support_calibrated_brier?.value, 4)} detail="squared probability error" tone="amber" />
			<MetricTile label="Support F1 at frozen threshold" value={fmtNum(H.support_f1_at_frozen_threshold?.value, 4)} detail={`threshold ${H.support_f1_at_frozen_threshold?.threshold}`} tone="amber" />
			<MetricTile label="Frozen alert threshold" value={fmtNum(H.support_f1_at_frozen_threshold?.threshold, 4)} detail={`${summary.calibrator} · fixed before test`} tone="rose" />
		</div>

		<Panel eyebrow="BUNDLE MATRIX" title="Five-way comparison on the sealed V3 test" note="STAY-BALANCED">
			<DataTable
				caption="Lower is better for MAE / hours; higher is better for AUPRC. Values with 95% CI. 'Naive' is the Master-Plan floor."
				columns={[
					{ key: 'label', label: 'Task / metric' },
					{ key: 'naive', label: 'Naive', align: 'right', fmt: (_v, r) => fmtNum(r.naive, 3) },
					{ key: 'V3', label: 'V3 (selected)', align: 'right', fmt: (_v, r) => cell(r, 'V3') },
					{ key: 'V2_AS_SERVED', label: 'V2 as served · operational', align: 'right', fmt: (_v, r) => cell(r, 'V2_AS_SERVED') },
					{ key: 'CONTROL', label: 'V2-recipe control · method', align: 'right', fmt: (_v, r) => cell(r, 'CONTROL') },
					{ key: 'V2_AS_EVALUATED', label: 'V2 as evaluated ▲ D1', align: 'right', fmt: (_v, r) => cell(r, 'V2_AS_EVALUATED') }
				]}
				rows={rows.map((r) => ({ ...r, label: `${taskLabel(r.task)} — ${r.metric} (${r.unit})` }))}
			/>
			<p class="rnote warn" style="margin-top:12px"><b>▲ V2 as evaluated</b> is the original offline harness containing defect D1 (the missingness of current SOFA leaked whether a stay ended within 24 h). It is historical and is not a valid estimate of served V2 behaviour. See the Defect D1 tab.</p>
		</Panel>

		<div class="rgrid two">
			<ChartFrame title="Improvement over the naive floor" subtitle="Relative improvement (%) of each bundle versus the naive baseline, per task" provenance={finalProv} data={improvementGroups.flatMap((g) => g.values.map((v) => ({ task_metric: g.label, bundle: v.series, improvement_pct: v.value })))} units="% improvement vs naive" empty={!improvementGroups.length}>
				<BarChart groups={improvementGroups} colors={barColors} orientation="horizontal" unit="%" valueFormat={(v) => `${v.toFixed(1)}%`} height={300} ariaLabel="Relative improvement over naive by bundle" />
			</ChartFrame>
			<Panel eyebrow="HONEST READING" title="What the evidence does and does not show">
				<div class="rgrid">
					<p class="rnote"><b>Operational comparison.</b> V3 versus V2 as served is a comparison against what the deployed V2 runtime would have produced on the same rows; part of V3's advantage is the correction of a training/evaluation mismatch (defect D1), not only a better model.</p>
					{#if icuEqualsControl}<p class="rnote amber"><b>ICU time:</b> V3 equals the repaired V2-recipe control — the pre-registered rule retained the V2 recipe because no candidate beat it beyond tolerance.</p>{/if}
					{#if r48notEstablished}<p class="rnote amber"><b>Recovery +48h:</b> the advantage over the repaired control is not established at the stricter 98.75% paired interval ({fmtCI(r48ctrl?.ci9875_lo, r48ctrl?.ci9875_hi, 3)}).</p>{/if}
					<p class="rnote"><b>Population uncertainty.</b> Intervals describe aggregate performance across stays, not individual-patient prediction intervals. AUPRC is prevalence dependent; calibrated and raw scores are reported separately.</p>
				</div>
			</Panel>
		</div>

		<Panel eyebrow="PROVENANCE" title="Served V3 bundle" note={String(prov?.data?.status ?? '')}>
			<div class="rgrid">
				<DataTable
					columns={[
						{ key: 'task', label: 'Task', fmt: (v) => taskLabel(String(v)) },
						{ key: 'family', label: 'Family' },
						{ key: 'feature_variant', label: 'Feature variant' },
						{ key: 'n_estimators', label: 'Trees', align: 'right' },
						{ key: 'model_sha256', label: 'Model SHA-256', fmt: (v) => shortHash(String(v), 16) }
					]}
					rows={Object.entries(tasks).map(([task, t]: [string, any]) => ({ task, ...t }))}
				/>
				<dl class="kv">
					<dt>bundle</dt><dd>{prov?.data?.bundle_id} (selected_models_v3 {shortHash(prov?.data?.file_sha256?.selected_models_v3, 16)})</dd>
					<dt>calibrator</dt><dd>{prov?.data?.calibrator?.method} · params {shortHash(prov?.data?.calibrator?.params_sha256, 16)}</dd>
					<dt>threshold</dt><dd>{prov?.data?.threshold?.comparator} {prov?.data?.threshold?.value}</dd>
					<dt>protocol / amendment</dt><dd>{shortHash(prov?.data?.file_sha256?.protocol_v1, 12)} / {shortHash(prov?.data?.file_sha256?.protocol_amendment_v1_1, 12)}</dd>
					<dt>evaluation freeze</dt><dd>{shortHash(prov?.data?.file_sha256?.evaluation_freeze, 16)}</dd>
					<dt>research registry</dt><dd>{shortHash(prov?.data?.file_sha256?.research_registry, 16)}</dd>
				</dl>
			</div>
		</Panel>
	{/if}
</ResearchPage>
