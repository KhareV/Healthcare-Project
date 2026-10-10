<script lang="ts">
	import { ChartFrame, LineChart, BarChart, Histogram, ForestPlot } from '$lib/components/research';
	import { toKitProvenance, cohortApi, type ResearchDataset } from '$lib/services/cohort';
	import { toTable, caveats, cell } from './report';
	import { bundleStyle } from '$lib/research/palette';
	import { downloadBlob, safeFilename } from '$lib/research/chart-export';

	let { dataset, expanded = false }: { dataset: ResearchDataset; expanded?: boolean } = $props();
	const p = $derived(dataset.provenance);
	const table = $derived(toTable(dataset));
	const notes = $derived(caveats(dataset));
	const rows = $derived<any[]>(Array.isArray(dataset.data?.rows) ? dataset.data.rows : []);
	let showAll = $state(false);
	let busy = $state('');
	let taskSel = $state('');
	let message = $state('');

	const tasks = $derived(Array.from(new Set(rows.map((r) => r.task).filter(Boolean))) as string[]);
	$effect(() => {
		if (tasks.length && !tasks.includes(taskSel)) taskSel = tasks[0];
	});
	const models = ['V3', 'V2_AS_SERVED', 'CONTROL', 'V2_AS_EVALUATED'];
	const color = (m: string) => bundleStyle(m).color;
	const bySeries = (xk: string, yk: string, filter: (r: any) => boolean = () => true) =>
		models
			.map((m) => ({ name: bundleStyle(m).label, color: color(m), dashed: !!bundleStyle(m).dash, points: rows.filter((r) => r.model === m && filter(r)).map((r) => ({ x: r[xk], y: r[yk] })) }))
			.filter((s) => s.points.length);

	type Chart = { kind: string } & Record<string, any>;
	const chart = $derived.by<Chart | null>(() => {
		const id = p.id;
		if (id === 'support_roc') return { kind: 'line', series: bySeries('fpr', 'tpr'), xLabel: 'False-positive rate', yLabel: 'True-positive rate', diagonal: true, xDomain: [0, 1], yDomain: [0, 1] };
		if (id === 'support_pr') return { kind: 'line', series: bySeries('recall', 'precision'), xLabel: 'Recall', yLabel: 'Precision', xDomain: [0, 1], yDomain: [0, 1] };
		if (id === 'support_reliability') return { kind: 'line', series: bySeries('mean_predicted', 'observed_rate', (r) => r.probability_type === 'calibrated'), xLabel: 'Mean predicted probability', yLabel: 'Observed rate', diagonal: true, xDomain: [0, 1], yDomain: [0, 1] };
		if (id === 'support_threshold_sensitivity') {
			const v3 = rows.filter((r) => r.model === 'V3');
			return { kind: 'line', xLabel: 'Decision threshold (read-only curve)', yLabel: 'Stay-balanced metric', xDomain: [0, 1], yDomain: [0, 1],
				series: ['precision', 'recall', 'f1', 'specificity'].map((k, i) => ({ name: k, color: ['#38bdf8', '#2bb8b0', '#fbbf24', '#a78bfa'][i], points: v3.map((r) => ({ x: r.threshold, y: r[k] })) })),
				highlight: dataset.data.frozen_threshold_V3 != null ? { x: dataset.data.frozen_threshold_V3, y: 0, label: `frozen ${Number(dataset.data.frozen_threshold_V3).toFixed(4)}` } : undefined };
		}
		if (id === 'learning_curve') {
			const t = rows.filter((r) => r.task === taskSel);
			return { kind: 'line', xLabel: 'Training subjects', yLabel: t[0]?.metric ?? 'metric', series: [{ name: taskSel, color: '#2bb8b0', points: t.map((r) => ({ x: r.size_subjects, y: r.mean, lo: r.mean - r.sd, hi: r.mean + r.sd })) }] };
		}
		if (id === 'regression_residuals') {
			const mk = (m: string) => rows.filter((r) => r.task === taskSel && r.model === m).map((r) => ({ x0: r.bin_lo, x1: r.bin_hi, count: r.weighted_count }));
			return { kind: 'hist', bins: mk('V3'), overlay: mk('V2_AS_SERVED'), name: 'V3', overlayName: 'V2 as served', xLabel: 'Residual (predicted − observed)', yLabel: 'stay-weighted rows' };
		}
		if (id === 'paired_differences') {
			const pr = rows.filter((r) => r.comparison === 'V3_vs_V2_AS_SERVED');
			return { kind: 'forest', rows: pr.map((r) => ({ label: `${r.task} · ${r.metric}`, estimate: r.diff_v3_minus_comparator, lo: r.ci9875_lo, hi: r.ci9875_hi })), xLabel: 'V3 − V2 as served (98.75% paired CI; units differ by row)', favourLeft: 'lower = V3 better for error metrics' };
		}
		if (id === 'ablation') {
			const t = rows.filter((r) => r.task === taskSel);
			return { kind: 'bar', groups: t.map((r) => ({ label: r.variant, values: [{ series: r.primary_metric, value: r.primary_mean }] })), orientation: 'horizontal', colors: { [t[0]?.primary_metric ?? 'm']: '#2bb8b0' } };
		}
		if (id === 'robustness') {
			const t = rows.filter((r) => r.task === taskSel && r.model === 'V3');
			return { kind: 'bar', groups: t.map((r) => ({ label: r.scenario, values: [{ series: r.metric, value: r.value }] })), orientation: 'horizontal', colors: { [t[0]?.metric ?? 'm']: '#38bdf8' } };
		}
		if (id === 'explainability_global') {
			const t = rows.filter((r) => r.task === taskSel).sort((a, b) => a.rank - b.rank).slice(0, 10);
			return { kind: 'bar', groups: t.map((r) => ({ label: String(r.readable ?? r.feature).slice(0, 48), values: [{ series: 'mean |SHAP|', value: r.mean_abs_shap }] })), orientation: 'horizontal', colors: { 'mean |SHAP|': '#a78bfa' } };
		}
		return null;
	});

	const shownRows = $derived(showAll ? table.rows : table.rows.slice(0, 12));
	const needsTask = $derived(['learning_curve', 'regression_residuals', 'ablation', 'robustness', 'explainability_global'].includes(p.id));

	async function dl(format: 'csv' | 'json' | 'tex') {
		busy = format;
		message = '';
		try {
			const text = await cohortApi.exportText(p.id, format);
			const mime = format === 'json' ? 'application/json' : format === 'csv' ? 'text/csv' : 'application/x-tex';
			downloadBlob(new Blob([text], { type: `${mime};charset=utf-8` }), safeFilename(p.id, format));
		} catch (cause) {
			message = cause instanceof Error ? cause.message : 'export failed';
		} finally {
			busy = '';
		}
	}
</script>

<article class="preview" data-dataset-id={p.id}>
	<header>
		<div>
			<h4>{p.title}</h4>
			<code>{p.id}</code>
		</div>
		<div class="actions" role="group" aria-label={`Export ${p.title}`}>
			{#each p.downloadable_formats.filter((f) => ['csv', 'json', 'tex'].includes(f)) as f}
				<button type="button" onclick={() => dl(f as 'csv' | 'json' | 'tex')} disabled={busy !== ''} aria-label={`Download ${p.title} as ${f.toUpperCase()}`}>{busy === f ? '…' : f.toUpperCase()}</button>
			{/each}
		</div>
	</header>
	{#if message}<p class="err" role="alert">{message}</p>{/if}
	{#if notes.length}
		<ul class="notes">{#each notes.slice(0, 3) as n}<li>{n}</li>{/each}</ul>
	{/if}
	{#if chart}
		<ChartFrame title={p.title} subtitle={p.partition} provenance={toKitProvenance(p)} data={rows} units={Object.values(p.units ?? {})[0] ?? ''}>
			{#if needsTask}
				<label class="task"><span>TASK</span><select bind:value={taskSel}>{#each tasks as t}<option value={t}>{t}</option>{/each}</select></label>
			{/if}
			{#if chart.kind === 'line'}
				<LineChart series={chart.series} xLabel={chart.xLabel} yLabel={chart.yLabel} diagonal={chart.diagonal} xDomain={chart.xDomain} yDomain={chart.yDomain} highlight={chart.highlight} height={260} />
			{:else if chart.kind === 'hist'}
				<Histogram bins={chart.bins} overlay={chart.overlay} name={chart.name} overlayName={chart.overlayName} xLabel={chart.xLabel} yLabel={chart.yLabel} height={240} />
			{:else if chart.kind === 'forest'}
				<ForestPlot rows={chart.rows} xLabel={chart.xLabel} favourLeft={chart.favourLeft} />
			{:else if chart.kind === 'bar'}
				<BarChart groups={chart.groups} orientation={chart.orientation} colors={chart.colors} valueFormat={(v) => v.toFixed(3)} ariaLabel={p.title} />
			{/if}
		</ChartFrame>
	{/if}
	<div class="table-wrap">
		<div class="meta">
			<span>{p.evidence_status}</span><span>{p.bundle}</span><span>{p.partition}</span>
			<span title={p.source_sha256}>sha256 {p.source_sha256.slice(0, 12)}</span>
			<span>{table.total} rows</span>
		</div>
		{#if table.rows.length}
			<div class="scroll">
				<table>
					<thead><tr>{#each table.columns as c}<th>{c}</th>{/each}</tr></thead>
					<tbody>{#each shownRows as r}<tr>{#each table.columns as c}<td>{cell(r[c])}</td>{/each}</tr>{/each}</tbody>
				</table>
			</div>
			{#if table.rows.length > 12}
				<button type="button" class="more" onclick={() => (showAll = !showAll)}>{showAll ? 'Show fewer rows' : `Show all ${table.rows.length} rows`}</button>
			{/if}
		{:else}
			<p class="empty">This dataset is document-shaped; use the JSON export.</p>
		{/if}
		{#if table.reason}<p class="reason">{table.reason}</p>{/if}
		{#if Object.keys(p.units ?? {}).length}<p class="units">Units: {Object.entries(p.units).map(([k, v]) => `${k} = ${v}`).join(' · ')}</p>{/if}
	</div>
</article>

<style>
	.preview{display:grid;gap:12px;padding:16px;border:1px solid rgba(148,163,184,.14);background:rgba(7,13,25,.55);min-width:0}
	header{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:flex-start}
	h4{margin:0 0 4px;color:#e5f1f0;font:500 15px 'Space Grotesk',sans-serif}code{color:#53647b;font:9px 'JetBrains Mono',monospace}
	.actions{display:flex;gap:6px}.actions button,.more{padding:6px 10px;color:#2bb8b0;background:rgba(43,184,176,.06);border:1px solid rgba(43,184,176,.3);font:8px 'JetBrains Mono',monospace;letter-spacing:.1em;cursor:pointer}
	.actions button:hover:not(:disabled),.more:hover{background:rgba(43,184,176,.14)}.actions button:disabled{opacity:.5;cursor:wait}
	button:focus-visible,select:focus-visible{outline:2px solid #2bb8b0;outline-offset:2px}
	.notes{margin:0;padding-left:18px;color:#8494a9;font-size:11px;line-height:1.6}
	.table-wrap{display:grid;gap:8px;min-width:0}.meta{display:flex;flex-wrap:wrap;gap:6px 12px;color:#53647b;font:8px 'JetBrains Mono',monospace;letter-spacing:.06em}
	.scroll{overflow:auto;max-height:340px;border:1px solid rgba(148,163,184,.1)}table{border-collapse:collapse;min-width:100%;font:10px 'JetBrains Mono',monospace}
	th,td{padding:5px 9px;border-bottom:1px solid rgba(148,163,184,.08);text-align:left;white-space:nowrap;color:#b7c6d4}th{position:sticky;top:0;background:#0a1222;color:#2bb8b0;font-weight:500}
	.empty,.reason,.units{margin:0;color:#71829a;font-size:11px}.err{color:#fb7185;margin:0;font-size:11px}
	.task{display:flex;gap:10px;align-items:center;margin-bottom:8px;color:#53647b;font:8px 'JetBrains Mono',monospace}
	select{padding:6px 8px;color:#dbe7e7;background:#060c18;border:1px solid rgba(148,163,184,.25);font:10px 'JetBrains Mono',monospace}
</style>
