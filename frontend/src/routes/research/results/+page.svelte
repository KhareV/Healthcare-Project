<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import Segmented from '$lib/components/research-pages/Segmented.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { ChartFrame, LineChart, Histogram, ConfusionMatrix, BarChart } from '$lib/components/research';
	import { bundleStyle } from '$lib/research/palette';
	import { fmtNum, fmtCI, fmtInt } from '$lib/research/format';
	import { researchApi, errorText, taskLabel, toProvenance, TASKS, type Row, type Wrapped } from '$lib/services/research';

	let curves = $state<Record<string, Wrapped<Row>> | null>(null);
	let reg = $state<Record<string, Wrapped<Row>> | null>(null);
	let error = $state('');
	let model = $state('V3');
	let confusionMode = $state<'rows' | 'weighted'>('rows');
	let regTask = $state('recovery24');
	let sliceTask = $state('recovery24');
	let sliceDim = $state('cardiac_subtype');
	let thrIndex = $state(0);

	onMount(async () => {
		try {
			[curves, reg] = await Promise.all([researchApi.curves(), researchApi.regression()]);
		} catch (cause) {
			error = errorText(cause);
		}
	});

	const MODELS = ['V3', 'V2_AS_SERVED', 'CONTROL', 'V2_AS_EVALUATED'];
	const mLabel = (m: string) => bundleStyle(m).label;
	const modelOptions = MODELS.map((m) => ({ value: m, label: m === 'V2_AS_EVALUATED' ? 'V2 as evaluated ▲' : m === 'V2_AS_SERVED' ? 'V2 as served' : m === 'CONTROL' ? 'Control' : 'V3' }));

	function curveSeries(key: 'roc' | 'pr', xk: string, yk: string) {
		const s = curves?.[key]?.data?.series ?? {};
		return MODELS.filter((m) => s[m]).map((m) => {
			const st = bundleStyle(m);
			return { name: st.label, color: st.color, dashed: m === 'V2_AS_EVALUATED', points: (s[m][xk] as number[]).map((x, i) => ({ x, y: (s[m][yk] as number[])[i] })) };
		});
	}
	const rocSeries = $derived(curveSeries('roc', 'x', 'y'));
	const prSeries = $derived.by(() => {
		const prev = curves?.pr?.data?.prevalence_stay_balanced;
		const base = typeof prev === 'number' ? [{ name: `Prevalence baseline (${fmtNum(prev, 3)})`, color: '#94a3b8', dashed: true, points: [{ x: 0, y: prev }, { x: 1, y: prev }] }] : [];
		return [...curveSeries('pr', 'x', 'y'), ...base];
	});
	const rocAuc = $derived<Row>(curves?.roc?.data?.metric_by_model ?? {});
	const prAuc = $derived<Row>(curves?.pr?.data?.metric_by_model ?? {});

	const relModel = $derived<Row>(curves?.reliability?.data?.models?.[model] ?? {});
	const relSeries = $derived([
		{ name: 'Calibrated', color: '#2bb8b0', points: (relModel.reliability_calibrated ?? []).map((b: Row) => ({ x: b.mean_predicted, y: b.observed_rate })) },
		{ name: 'Raw score', color: '#fbbf24', dashed: true, points: (relModel.reliability_raw ?? []).map((b: Row) => ({ x: b.mean_predicted, y: b.observed_rate })) }
	].filter((s) => s.points.length));

	const dist = $derived<Row[]>((curves?.score_distribution?.data?.rows ?? []).filter((r: Row) => r.model === model));
	const binsOf = (outcome: string) => dist.filter((r) => r.outcome === outcome).map((r) => ({ x0: r.bin_lo, x1: r.bin_hi, count: r.row_count }));
	const conf = $derived<Row | undefined>((curves?.confusion?.data?.rows ?? []).find((r: Row) => r.model === model));

	const thrRows = $derived<Row[]>((curves?.threshold_sensitivity?.data?.rows ?? []).filter((r: Row) => r.model === model));
	const frozenThr = $derived<number | undefined>(curves?.threshold_sensitivity?.data?.frozen_thresholds?.[model]);
	$effect(() => {
		if (thrRows.length) thrIndex = Math.min(thrRows.length - 1, Math.max(0, thrRows.reduce((best, r, i) => (Math.abs(r.threshold - (frozenThr ?? 0.5)) < Math.abs(thrRows[best].threshold - (frozenThr ?? 0.5)) ? i : best), 0)));
	});
	const thrNow = $derived<Row | undefined>(thrRows[thrIndex]);
	const thrSeries = $derived(['precision', 'recall', 'specificity', 'f1'].map((k, i) => ({ name: k === 'f1' ? 'F1' : k, color: ['#38bdf8', '#2bb8b0', '#a78bfa', '#fbbf24'][i], points: thrRows.map((r) => ({ x: r.threshold, y: r[k] })) }))
		.concat(frozenThr !== undefined ? [{ name: `Frozen threshold ${fmtNum(frozenThr, 4)} (deployed)`, color: '#fb7185', points: [{ x: frozenThr, y: 0 }, { x: frozenThr, y: 1 }], dashed: true } as any] : []));

	const clsRows = $derived<Row[]>(curves?.metrics?.data?.rows ?? []);

	// Regression
	const residRows = $derived<Row[]>((reg?.residuals?.data?.rows ?? []).filter((r: Row) => r.task === regTask));
	const residBins = (m: string) => residRows.filter((r) => r.model === m).map((r) => ({ x0: r.bin_lo, x1: r.bin_hi, count: r.row_count }));
	const residSummary = $derived<Row>(reg?.residuals?.data?.summary ?? {});
	const pvo = $derived(['V3', 'V2_AS_SERVED', 'CONTROL'].map((m) => ({
		name: bundleStyle(m).label, color: bundleStyle(m).color,
		points: (reg?.predicted_vs_observed?.data?.rows ?? []).filter((r: Row) => r.task === regTask && r.model === m).map((r: Row) => ({ x: r.mean_predicted, y: r.mean_observed }))
	})).filter((s) => s.points.length));
	const unit = $derived(TASKS.find((t) => t.value === regTask)?.unit ?? '');
	const sliceRows = $derived<Row[]>((reg?.error_slices?.data?.rows ?? []).filter((r: Row) => r.task === sliceTask && r.dimension === sliceDim));
	const sliceDims = $derived<string[]>([...new Set<string>((reg?.error_slices?.data?.rows ?? []).filter((r: Row) => r.task === sliceTask).map((r: Row) => r.dimension))]);
	$effect(() => { if (sliceDims.length && !sliceDims.includes(sliceDim)) sliceDim = sliceDims[0]; });
	const sliceGroups = $derived.by(() => {
		const labels = [...new Set<string>(sliceRows.map((r) => r.slice))];
		return labels.map((l) => ({ label: `${l}`, values: ['V3', 'V2_AS_SERVED', 'CONTROL'].map((m) => {
			const r = sliceRows.find((x) => x.slice === l && x.model === m);
			return r ? { series: bundleStyle(m).label, value: r.value as number, n: r.n_stays as number } : null;
		}).filter(Boolean) as { series: string; value: number; n: number }[] }));
	});
	const sliceColors = Object.fromEntries(['V3', 'V2_AS_SERVED', 'CONTROL'].map((m) => [bundleStyle(m).label, bundleStyle(m).color]));
	const sliceUnit = $derived(TASKS.find((t) => t.value === sliceTask)?.unit ?? '');
	const regMetrics = $derived<Row[]>(reg?.metrics?.data?.rows ?? []);

	const p = (k: string, c: Record<string, Wrapped<Row>> | null) => toProvenance(c?.[k]?.provenance);
</script>

<svelte:head><title>Classification &amp; Regression | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03.2 / RESULTS" title="Classification and regression evidence." description="Curves and error breakdowns computed from the saved sealed-test predictions (stay-balanced). Nothing here re-runs a model or changes a deployed setting." {error} loading={!curves && !error}>
	{#if curves && reg}
		<h3 class="rh">Organ-support initiation (classification)</h3>
		<p class="rnote">Calibrated probabilities from each bundle's own frozen calibrator. AUROC is prevalence-independent; AUPRC depends on prevalence ({fmtNum(curves.roc.data.prevalence_stay_balanced, 3)} stay-balanced). Raw and calibrated values are kept separate below. <b>V2 as evaluated ▲</b> contains defect D1.</p>
		<div class="rgrid two">
			<ChartFrame title="ROC curve" subtitle="True-positive rate vs false-positive rate · stay-balanced" provenance={p('roc', curves)} data={curves.roc.data.rows} units="fraction" empty={!rocSeries.length}>
				<LineChart series={rocSeries} xLabel="False-positive rate" yLabel="True-positive rate" xDomain={[0, 1]} yDomain={[0, 1]} diagonal ariaLabel="ROC curves" />
				<p class="rmeta" style="margin-top:8px">{#each Object.entries(rocAuc) as [m, v]}<span>{mLabel(m)} AUROC {fmtNum(v as number, 3)}</span>{/each}</p>
			</ChartFrame>
			<ChartFrame title="Precision–recall curve" subtitle="Dashed grey line = prevalence baseline" provenance={p('pr', curves)} data={curves.pr.data.rows} units="fraction" empty={!prSeries.length}>
				<LineChart series={prSeries} xLabel="Recall" yLabel="Precision" xDomain={[0, 1]} yDomain={[0, 1]} ariaLabel="Precision-recall curves" />
				<p class="rmeta" style="margin-top:8px">{#each Object.entries(prAuc) as [m, v]}<span>{mLabel(m)} AUPRC {fmtNum(v as number, 3)}</span>{/each}</p>
			</ChartFrame>
		</div>

		<Segmented label="Model for the panels below" bind:value={model} options={modelOptions} />
		<div class="rgrid two">
			<ChartFrame title="Reliability diagram" subtitle={`${mLabel(model)} · 10 equal-mass bins`} provenance={p('reliability', curves)} data={[...(relModel.reliability_calibrated ?? []).map((b: Row) => ({ type: 'calibrated', ...b })), ...(relModel.reliability_raw ?? []).map((b: Row) => ({ type: 'raw', ...b }))]} units="probability" empty={!relSeries.length}>
				<LineChart series={relSeries} xLabel="Mean predicted probability" yLabel="Observed event rate" xDomain={[0, 1]} yDomain={[0, 1]} diagonal ariaLabel="Reliability diagram" />
				<p class="rmeta" style="margin-top:8px"><span>slope {fmtNum(relModel.calibration_slope, 3)}</span><span>intercept {fmtNum(relModel.calibration_intercept, 3)}</span><span>ECE {fmtNum(relModel.ece_10_equal_mass, 3)}</span><span>Brier {fmtNum(relModel.brier, 3)}</span></p>
			</ChartFrame>
			<ChartFrame title="Score distribution by observed outcome" subtitle={`${mLabel(model)} · calibrated probability · rows`} provenance={p('score_distribution', curves)} data={dist} units="rows per bin" empty={!dist.length}>
				<Histogram bins={binsOf('positive')} overlay={binsOf('negative')} name="Initiation (positive)" overlayName="No initiation (negative)" color="#2bb8b0" overlayColor="#fbbf24" xLabel="Predicted probability" xUnit="" yLabel="rows" ariaLabel="Score distribution" />
			</ChartFrame>
		</div>

		<div class="rgrid two">
			<ChartFrame title="Confusion matrix at the frozen threshold" subtitle={`${mLabel(model)} · threshold ${fmtNum(conf?.threshold, 4)} (read-only)`} provenance={p('confusion', curves)} data={curves.confusion.data.rows} empty={!conf}>
				<Segmented label="Counts" bind:value={confusionMode} options={[{ value: 'rows', label: 'Rows' }, { value: 'weighted', label: 'Stay-balanced' }]} />
				{#if conf}
					<ConfusionMatrix tp={confusionMode === 'rows' ? conf.TP_rows : conf.TP_weighted} fp={confusionMode === 'rows' ? conf.FP_rows : conf.FP_weighted} tn={confusionMode === 'rows' ? conf.TN_rows : conf.TN_weighted} fn={confusionMode === 'rows' ? conf.FN_rows : conf.FN_weighted} threshold={`≥ ${fmtNum(conf.threshold, 4)}`} digits={confusionMode === 'rows' ? 0 : 1} />
					<p class="rmeta" style="margin-top:8px"><span>precision {fmtNum(conf.precision, 3)}</span><span>recall {fmtNum(conf.recall_sensitivity, 3)}</span><span>specificity {fmtNum(conf.specificity, 3)}</span><span>F1 {fmtNum(conf.f1, 3)}</span></p>
				{/if}
			</ChartFrame>
			<ChartFrame title="Threshold-sensitivity explorer" subtitle={`${mLabel(model)} · precomputed curves on saved test predictions`} provenance={p('threshold_sensitivity', curves)} data={thrRows} units="fraction" empty={!thrRows.length}>
				<p class="rnote warn"><b>Read-only.</b> This explorer moves a marker over curves that were already computed from saved predictions. It never changes, tunes or selects the deployed operating point (V3: {curves.threshold_sensitivity.data.frozen_thresholds?.V3}).</p>
				{#if thrRows.length}
					<label class="slider">Threshold {fmtNum(thrNow?.threshold, 3)}<input type="range" min="0" max={thrRows.length - 1} step="1" bind:value={thrIndex} aria-label="Explore threshold (read-only)" /></label>
					<LineChart series={thrSeries} xLabel="Decision threshold" yLabel="Fraction" xDomain={[0, 1]} yDomain={[0, 1]} highlight={thrNow ? { x: thrNow.threshold, y: thrNow.f1, label: `F1 ${fmtNum(thrNow.f1, 3)} @ ${fmtNum(thrNow.threshold, 2)}`, color: '#fbbf24' } : undefined} ariaLabel="Threshold sensitivity" />
					<p class="rmeta" style="margin-top:8px"><span>precision {fmtNum(thrNow?.precision, 3)}</span><span>recall {fmtNum(thrNow?.recall, 3)}</span><span>specificity {fmtNum(thrNow?.specificity, 3)}</span><span>F1 {fmtNum(thrNow?.f1, 3)}</span></p>
				{/if}
			</ChartFrame>
		</div>

		<Panel eyebrow="TABLE" title="Support metrics by bundle" note="95% CI = STAY-CLUSTER BOOTSTRAP">
			<DataTable
				columns={[
					{ key: 'model', label: 'Bundle', fmt: (v) => mLabel(String(v)) },
					{ key: 'calibrated_auprc', label: 'AUPRC cal.', align: 'right', fmt: (v, r) => `${fmtNum(v, 3)} ${fmtCI(r.calibrated_auprc_ci95_lo, r.calibrated_auprc_ci95_hi, 3)}` },
					{ key: 'raw_auprc', label: 'AUPRC raw', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'calibrated_auroc', label: 'AUROC cal.', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'raw_auroc', label: 'AUROC raw', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'brier', label: 'Brier cal.', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'raw_brier', label: 'Brier raw', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'log_loss', label: 'Log loss', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'precision', label: 'Precision', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'recall', label: 'Recall', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'specificity', label: 'Specificity', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'f1_at_frozen_threshold', label: 'F1', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'frozen_threshold', label: 'Threshold', align: 'right', fmt: (v) => fmtNum(v, 4) },
					{ key: 'n_rows', label: 'Rows', align: 'right', fmt: (v) => fmtInt(v) }
				]}
				rows={clsRows}
				rowClass={(r) => (r.model === 'V2_AS_EVALUATED' ? 'flag' : r.model === 'V3' ? 'best' : '')}
			/>
		</Panel>

		<h3 class="rh">Recovery and ICU remaining time (regression)</h3>
		<Segmented label="Task" bind:value={regTask} options={TASKS.filter((t) => t.value !== 'organ_support').map((t) => ({ value: t.value, label: `${t.label} (${t.unit})` }))} />
		<div class="rgrid two">
			<ChartFrame title="Residual distribution (prediction − observed)" subtitle={`${taskLabel(regTask)} · V3 vs V2 as served · ${unit} · negative = under-prediction`} provenance={p('residuals', reg)} data={residRows} units={unit} empty={!residBins('V3').length}>
				<Histogram bins={residBins('V3')} overlay={residBins('V2_AS_SERVED')} name="V3" overlayName="V2 as served" color="#2bb8b0" overlayColor="#fbbf24" xLabel={`Residual (${unit})`} yLabel="rows" ariaLabel="Residual distribution" />
				<p class="rmeta" style="margin-top:8px"><span>V3 mean residual {fmtNum(residSummary[`${regTask}:V3`]?.mean_residual, 3)} {unit}</span><span>outside plotted range (V3) {fmtInt((residSummary[`${regTask}:V3`]?.below_range ?? 0) + (residSummary[`${regTask}:V3`]?.above_range ?? 0))}</span><span>outside plotted range (V2 served) {fmtInt((residSummary[`${regTask}:V2_AS_SERVED`]?.below_range ?? 0) + (residSummary[`${regTask}:V2_AS_SERVED`]?.above_range ?? 0))}</span></p>
			</ChartFrame>
			<ChartFrame title="Predicted vs observed (quantile-binned means)" subtitle={`${taskLabel(regTask)} · 20 predicted-quantile bins · stay-balanced · diagonal = perfect`} provenance={p('predicted_vs_observed', reg)} data={(reg.predicted_vs_observed.data.rows as Row[]).filter((r) => r.task === regTask)} units={unit} empty={!pvo.length}>
				<LineChart series={pvo} xLabel={`Mean predicted (${unit})`} yLabel={`Mean observed (${unit})`} diagonal ariaLabel="Predicted versus observed" />
			</ChartFrame>
		</div>

		<h3 class="rh">Error breakdown by prespecified slices</h3>
		<Segmented label="Task" bind:value={sliceTask} options={TASKS.map((t) => ({ value: t.value, label: t.label }))} />
		<Segmented label="Slice" bind:value={sliceDim} options={sliceDims.map((d) => ({ value: d, label: d.replaceAll('_', ' ') }))} />
		<ChartFrame title={`${taskLabel(sliceTask)} by ${sliceDim.replaceAll('_', ' ')}`} subtitle={`Primary metric within slice (${sliceUnit}${sliceTask === 'organ_support' ? ', higher is better' : ', lower is better'}) · cut-points from the development pool only`} provenance={p('error_slices', reg)} data={sliceRows} units={sliceUnit} empty={!sliceGroups.length}>
			<BarChart groups={sliceGroups} colors={sliceColors} unit={sliceUnit} valueFormat={(v) => v.toFixed(3)} height={300} ariaLabel="Error by slice" />
			<p class="rmeta" style="margin-top:8px">{#each sliceGroups as g}<span>{g.label}: {fmtInt(g.values[0]?.n)} stays</span>{/each}</p>
		</ChartFrame>

		<Panel eyebrow="TABLE" title="Regression metrics by bundle" note="STAY-BALANCED · 95% CI">
			<DataTable
				columns={[
					{ key: 'task', label: 'Task', fmt: (v) => taskLabel(String(v)) },
					{ key: 'model', label: 'Bundle', fmt: (v) => mLabel(String(v)) },
					{ key: 'primary', label: 'Primary (unit)', align: 'right', fmt: (_v, r) => `${fmtNum(r.task === 'icu_stay_time' ? r.weighted_median_ae_hours : r.mae, 3)} ${r.task === 'icu_stay_time' ? 'h wMedAE' : 'SOFA MAE'}` },
					{ key: 'primary_ci95_lo', label: '95% CI', align: 'right', fmt: (_v, r) => fmtCI(r.primary_ci95_lo, r.primary_ci95_hi, 3) },
					{ key: 'mae', label: 'MAE', align: 'right', fmt: (v, r) => `${fmtNum(v, 3)} ${r.task === 'icu_stay_time' ? 'h' : ''}` },
					{ key: 'rmse', label: 'RMSE', align: 'right', fmt: (v, r) => `${fmtNum(v, 3)} ${r.task === 'icu_stay_time' ? 'h' : ''}` },
					{ key: 'median_ae', label: 'Median AE', align: 'right', fmt: (v) => fmtNum(v, 3) },
					{ key: 'n_rows', label: 'Rows', align: 'right', fmt: (v) => fmtInt(v) },
					{ key: 'n_stays', label: 'Stays', align: 'right', fmt: (v) => fmtInt(v) }
				]}
				rows={regMetrics}
				defaultSort="task"
				rowClass={(r) => (r.model === 'V2_AS_EVALUATED' ? 'flag' : r.model === 'V3' ? 'best' : '')}
			/>
			<p class="rnote" style="margin-top:12px">SOFA-change errors are in SOFA points and are not absolute SOFA; ICU errors are in hours. The 'primary' column uses MAE for recovery and weighted median absolute error for ICU time.</p>
		</Panel>
	{/if}
</ResearchPage>
