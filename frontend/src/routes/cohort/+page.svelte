<script lang="ts">
	import { onMount } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import { ChartFrame, LineChart, BarChart, Histogram, HeatmapGrid } from '$lib/components/research';
	import { cohortApi, toKitProvenance, type CohortBundle, type DatasetProvenance } from '$lib/services/cohort';
	import {
		ladder, SUBTYPE_LABEL, SUBTYPE_COLOR, PARTITION_COLOR, TASK_LABEL, DISPOSITION_LABEL, channelLabel,
		partitionLabel, doc, meanObserved, correlationMatrix
	} from '$lib/components/cohort/adapters';

	let bundle = $state<CohortBundle | null>(null);
	let error = $state('');
	let loading = $state(true);
	let partition = $state('DEV_POOL');
	let compare = $state(true);
	let subtype = $state('ALL');
	let cutoff = $state(0);
	let variable = $state('heart_rate__latest');

	onMount(async () => {
		try {
			bundle = await cohortApi.cohort();
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not load the cohort characterization evidence.';
		} finally {
			loading = false;
		}
	});

	const D = (id: string) => doc(bundle, id);
	const P = (id: string): DatasetProvenance | null => bundle?.datasets?.[id]?.provenance ?? null;
	const summary = $derived(D('cohort_summary'));
	const partitions = $derived<string[]>(summary ? Object.keys(summary.cohorts) : []);
	const comparePartition = $derived(partition === 'DEV_POOL' ? 'LCE' : partition === 'LCE' ? 'DEV_POOL' : 'DEV_POOL');
	const shown = $derived(compare ? Array.from(new Set([partition, comparePartition])) : [partition]);
	const pLabel = (p: string) => partitionLabel(p, summary);
	const SUBTYPES = ['SYNTHETIC_HF', 'SYNTHETIC_ISCHEMIC', 'SYNTHETIC_ARRHYTHMIA'];
	const visibleSubtypes = $derived(subtype === 'ALL' ? SUBTYPES : [subtype]);

	function prov(id: string, partitions: string[] = [partition], extra: Record<string, unknown> = {}) {
		const p = P(id);
		if (!p) return { status: 'DEVELOPMENT' as const };
		const s = summary?.cohorts?.[partitions[0]];
		return toKitProvenance(p, {
			partition: partitions.map(pLabel).join(' vs '),
			n_rows: s?.rows ?? null,
			n_stays: s?.retained_stays ?? null,
			...(extra as object)
		});
	}

	// ---- overview tiles -------------------------------------------------
	const sum = $derived(summary?.cohorts?.[partition]);
	const elig = $derived(D('cohort_eligibility')?.cohorts?.[partition]);
	const prev = $derived(D('cohort_outcome_prevalence')?.cohorts?.[partition]);
	const obsDensity = $derived(meanObserved(D('cohort_missingness_feature'), partition));
	const fmtN = (v: unknown) => (typeof v === 'number' ? v.toLocaleString() : '—');
	const fmtP = (v: unknown, d = 1) => (typeof v === 'number' ? `${(v * 100).toFixed(d)}` : '—');

	// ---- chart data -----------------------------------------------------
	const subtypeGroups = $derived.by(() => {
		const d = D('cohort_subtype')?.cohorts;
		if (!d) return [];
		return SUBTYPES.map((s) => ({
			label: SUBTYPE_LABEL[s],
			values: shown.filter((p) => d[p]?.[s]).map((p) => ({ series: p, value: d[p][s].fraction * 100, n: d[p][s].stays }))
		}));
	});
	const subtypeRows = $derived(subtypeGroups.flatMap((g) => g.values.map((v) => ({ subtype: g.label, partition: v.series, percent_of_stays: v.value, stays: v.n }))));

	const ageSeries = $derived.by(() => {
		const d = D('cohort_age_sex')?.cohorts;
		if (!d) return [];
		return shown.filter((p) => d[p]).map((p) => ({ name: p, color: PARTITION_COLOR[p] ?? '#94a3b8', points: ladder(d[p].age_years) }));
	});
	const ageSubtypeGroups = $derived.by(() => {
		const d = D('cohort_age_sex')?.cohorts;
		if (!d) return [];
		return visibleSubtypes.map((s) => ({
			label: SUBTYPE_LABEL[s],
			values: shown.filter((p) => d[p]?.age_by_subtype_mean?.[s] != null).map((p) => ({ series: p, value: d[p].age_by_subtype_mean[s] }))
		}));
	});
	const sexGroups = $derived.by(() => {
		const d = D('cohort_age_sex')?.cohorts;
		if (!d) return [];
		return shown.filter((p) => d[p]).map((p) => ({
			label: p,
			values: ['FEMALE', 'MALE'].map((s) => ({ series: s === 'FEMALE' ? 'Female' : 'Male', value: d[p].sex[s].fraction * 100, n: d[p].sex[s].stays }))
		}));
	});
	const durationSeries = $derived.by(() => {
		const d = D('cohort_duration')?.cohorts?.[partition];
		if (!d) return [];
		const out: { name: string; color: string; points: { x: number; y: number }[] }[] = [];
		if (subtype === 'ALL') out.push({ name: 'All stays', color: '#e5f1f0', points: ladder(d.all) });
		for (const s of visibleSubtypes) if (d.by_subtype?.[s]) out.push({ name: SUBTYPE_LABEL[s], color: SUBTYPE_COLOR[s], points: ladder(d.by_subtype[s]) });
		return out;
	});
	const sofaOverall = $derived.by(() => {
		const d = D('cohort_sofa_distributions')?.cohorts;
		if (!d) return [];
		return shown.filter((p) => d[p]?.overall).map((p) => ({ name: p, color: PARTITION_COLOR[p] ?? '#94a3b8', points: ladder(d[p].overall) }));
	});
	const sofaByCutoff = $derived(D('cohort_sofa_distributions')?.cohorts?.[partition]?.by_cutoff_index ?? {});
	const cutoffKeys = $derived(Object.keys(sofaByCutoff).map(Number).sort((a, b) => a - b));
	$effect(() => {
		if (cutoffKeys.length && !cutoffKeys.includes(cutoff)) cutoff = cutoffKeys[0];
	});
	const sofaCutoffSeries = $derived.by(() => {
		const mean = cutoffKeys.map((k) => ({ x: 24 + 6 * k, y: sofaByCutoff[k].mean, lo: Math.max(0, sofaByCutoff[k].mean - sofaByCutoff[k].sd), hi: sofaByCutoff[k].mean + sofaByCutoff[k].sd, label: `n=${sofaByCutoff[k].n}` }));
		const median = cutoffKeys.map((k) => ({ x: 24 + 6 * k, y: sofaByCutoff[k].p50 }));
		return [
			{ name: 'Mean ± SD', color: '#2bb8b0', points: mean },
			{ name: 'Median', color: '#fbbf24', points: median, dashed: true }
		];
	});
	const sofaSel = $derived(sofaByCutoff[cutoff]);
	const sofaCutoffRows = $derived(cutoffKeys.map((k) => ({ cutoff_index: k, hours_since_admission: 24 + 6 * k, ...sofaByCutoff[k] })));

	const eligRows = ['legal_prediction_rows', 'icu_time', 'recovery24', 'recovery48', 'organ_support'];
	const eligMatrix = $derived.by(() => {
		const d = D('cohort_eligibility')?.cohorts;
		if (!d) return null;
		const cols = partitions.filter((p) => d[p]);
		return {
			cols,
			rows: eligRows.map((r) => TASK_LABEL[r]),
			values: eligRows.map((r) => cols.map((c) => d[c][r].eligible_rows / d[c].legal_prediction_rows.eligible_rows)),
			counts: eligRows.map((r) => cols.map((c) => d[c][r].eligible_rows)),
			flat: eligRows.flatMap((r) => cols.map((c) => ({ task: TASK_LABEL[r], partition: c, eligible_rows: d[c][r].eligible_rows, unique_stays: d[c][r].unique_stays, fraction_of_legal_rows: d[c][r].eligible_rows / d[c].legal_prediction_rows.eligible_rows })))
		};
	});
	const dispositionGroups = $derived.by(() => {
		const d = prev;
		if (!d) return [];
		return Object.entries<number>(d.disposition_rows).map(([k, v]) => ({ label: DISPOSITION_LABEL[k] ?? k, values: [{ series: 'rows', value: v }] }));
	});
	const cutoffHist = $derived.by(() => {
		const h = D('cohort_cutoffs_per_stay')?.cohorts?.[partition]?.histogram;
		if (!h) return [];
		return Object.entries<number>(h).map(([k, v]) => ({ x0: Number(k) - 0.5, x1: Number(k) + 0.5, count: v })).sort((a, b) => a.x0 - b.x0);
	});
	const missRows = $derived.by(() => {
		const m = D('cohort_missingness_feature')?.cohorts?.[partition];
		if (!m) return { clinical: [] as any[], support: [] as string[] };
		const entries = Object.entries<any>(m);
		return {
			clinical: entries.filter(([, v]) => v.channel_type === 'clinical_observation').map(([k, v]) => ({ feature: channelLabel(k), missing_percent: v.missing_fraction_nonpadded_bins * 100, observed_percent: v.observed_fraction_nonpadded_bins * 100 })).sort((a, b) => b.missing_percent - a.missing_percent),
			support: entries.filter(([, v]) => v.channel_type !== 'clinical_observation').map(([k]) => channelLabel(k))
		};
	});
	const missGroups = $derived(missRows.clinical.map((r) => ({ label: r.feature, values: [{ series: 'missing', value: r.missing_percent }] })));
	const binMatrix = $derived.by(() => {
		const b = D('cohort_missingness_bin')?.cohorts?.[partition];
		if (!b) return null;
		const keys = Object.keys(b).sort((x, y) => Number(x) - Number(y));
		const channels = Object.keys(b[keys[0]].missing_fraction_by_clinical_channel).sort();
		return {
			cols: keys.map((k) => `${-48 + 6 * Number(k)} to ${-42 + 6 * Number(k)}h`),
			rows: channels.map(channelLabel),
			values: channels.map((c) => keys.map((k) => 1 - b[k].missing_fraction_by_clinical_channel[c])),
			counts: channels.map(() => keys.map((k) => b[k].nonpadded_rows)),
			padded: keys.map((k) => ({ bin: `${-48 + 6 * Number(k)} to ${-42 + 6 * Number(k)}h`, padded_percent: b[k].padded_fraction * 100, non_padded_rows: b[k].nonpadded_rows }))
		};
	});
	const paddedGroups = $derived((binMatrix?.padded ?? []).map((r) => ({ label: r.bin, values: [{ series: 'padded', value: r.padded_percent, n: r.non_padded_rows }] })));
	const corr = $derived(correlationMatrix(D('cohort_correlations'), partition === 'LCE' ? 'LCE' : 'DEV_POOL'));
	const icc = $derived.by(() => {
		const c = D('cohort_between_subject')?.cohorts?.[partition === 'LCE' ? 'LCE' : 'DEV_POOL']?.channels;
		if (!c) return [];
		return Object.entries<any>(c).map(([k, v]) => ({ channel: channelLabel(k), icc1: v.icc1, between_var: v.between_var, within_var: v.within_var, n: v.n, subjects: v.groups })).sort((a, b) => b.icc1 - a.icc1);
	});
	const varDoc = $derived(D('cohort_variable_distributions'));
	const varKeys = $derived<string[]>(Object.keys(varDoc?.cohorts?.[partition === 'LCE' ? 'LCE' : 'DEV_POOL'] ?? {}));
	const varSeries = $derived.by(() => {
		const c = varDoc?.cohorts;
		if (!c) return [];
		return (['DEV_POOL', 'LCE'] as const).filter((p) => c[p]?.[variable]).map((p) => ({ name: p, color: PARTITION_COLOR[p], points: ladder(c[p][variable]) }));
	});
	const varN = $derived(varDoc?.cohorts?.[partition === 'LCE' ? 'LCE' : 'DEV_POOL']?.[variable]?.n);
	const corrPartition = $derived(partition === 'LCE' ? 'LCE' : 'DEV_POOL');
	const componentNote = $derived(D('cohort_sofa_distributions')?.cohorts?.[partition]?.components as string | undefined);
	const rowsFlat = (m: any) => (m ? m.rows.flatMap((r: string, i: number) => m.cols.map((c: string, j: number) => ({ row: r, col: c, value: m.values[i][j], n: m.counts?.[i]?.[j] ?? null }))) : []);
</script>

<svelte:head><title>Cohort Analytics | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage
	eyebrow="COHORT ANALYTICS"
	title="Who is in the data, and what can the models see."
	description="Population structure, eligibility, outcomes and observation availability for the development pool and the held-out LCE cohort. Descriptive evidence only — sealed final-test cohorts are never read here."
>
	{#if loading}
		<div class="state">Loading frozen cohort characterization…</div>
	{:else if error}
		<div class="state error" role="alert">{error}</div>
	{:else if bundle && summary}
		<div class="controls" role="group" aria-label="Cohort controls">
			<label>
				<span>PARTITION</span>
				<select bind:value={partition}>
					{#each partitions as p}<option value={p}>{pLabel(p)}</option>{/each}
				</select>
			</label>
			<label class="check"><input type="checkbox" bind:checked={compare} /><span>OVERLAY {comparePartition}</span></label>
			<label>
				<span>SUBTYPE FILTER</span>
				<select bind:value={subtype}>
					<option value="ALL">All cardiac subtypes</option>
					{#each SUBTYPES as s}<option value={s}>{SUBTYPE_LABEL[s]}</option>{/each}
				</select>
			</label>
			<p class="scope">
				Evidence status <b>DEVELOPMENT</b> · excluded and never read: V1 test, V2 fresh test, V3 sealed test ·
				<code>{(bundle.datasets.cohort_summary?.provenance.source_sha256 ?? '').slice(0, 12)}</code>
			</p>
		</div>

		<div class="tiles">
			<MetricTile label="Subjects" value={fmtN(sum?.subjects_generated)} detail={pLabel(partition)} tone="teal" />
			<MetricTile label="Episodes (stays)" value={fmtN(sum?.retained_stays)} detail="one ICU-like episode per subject" tone="teal" />
			<MetricTile label="Legal cutoff rows" value={fmtN(sum?.rows)} detail={`mean ${sum?.legal_cutoffs_per_stay_mean?.toFixed?.(2) ?? '—'} cutoffs / stay`} tone="cyan" />
			<MetricTile label="Support event rate" value={fmtP(prev?.row_prevalence)} unit="%" detail={`stay-balanced ${fmtP(prev?.stay_balanced_prevalence)}% · ${fmtN(prev?.eligible_rows)} eligible rows`} tone="amber" />
			<MetricTile label="Observation density" value={fmtP(obsDensity)} unit="%" detail="clinical channels observed per non-padded bin" tone="teal" />
			<MetricTile label="Structural padding" value={fmtP(sum?.padding?.padded_bin_fraction)} unit="%" detail="pre-ICU bins (padding, not missingness)" tone="cyan" />
			{#each ['recovery24', 'recovery48', 'icu_time', 'organ_support'] as t}
				<MetricTile label={TASK_LABEL[t]} value={fmtN(elig?.[t]?.eligible_rows)} unit="rows" detail={`${fmtN(elig?.[t]?.unique_stays)} unique stays`} tone={t === 'organ_support' ? 'amber' : 'teal'} />
			{/each}
		</div>

		<div class="grid">
			<ChartFrame title="Cardiac subtype composition" subtitle="Share of stays (%)" provenance={prov('cohort_subtype', shown)} data={subtypeRows} units="% of stays">
				<BarChart groups={subtypeGroups} colors={Object.fromEntries(shown.map((p) => [p, PARTITION_COLOR[p] ?? '#94a3b8']))} unit="%" valueFormat={(v) => v.toFixed(1)} yLabel="% of stays" ariaLabel="Cardiac subtype composition" />
			</ChartFrame>

			<ChartFrame title="Age distribution (quantile profile)" subtitle="Age in years at percentile 0…100; frozen export holds percentiles, not raw ages" provenance={prov('cohort_age_sex', shown)} data={ageSeries.flatMap((s) => s.points.map((p) => ({ partition: s.name, percentile: p.x, age_years: p.y })))} units="years">
				<LineChart series={ageSeries} xLabel="Percentile" yLabel="Age" xUnit="%" yUnit="y" xDomain={[0, 100]} ariaLabel="Age quantile profile" />
			</ChartFrame>

			<ChartFrame title="Mean age by cardiac subtype" provenance={prov('cohort_age_sex', shown)} data={ageSubtypeGroups.flatMap((g) => g.values.map((v) => ({ subtype: g.label, partition: v.series, mean_age_years: v.value })))} units="years">
				<BarChart groups={ageSubtypeGroups} colors={Object.fromEntries(shown.map((p) => [p, PARTITION_COLOR[p] ?? '#94a3b8']))} unit="y" valueFormat={(v) => v.toFixed(1)} yLabel="Mean age (years)" ariaLabel="Mean age by subtype" />
			</ChartFrame>

			<ChartFrame title="Sex composition" subtitle="Share of stays (%)" provenance={prov('cohort_age_sex', shown)} data={sexGroups.flatMap((g) => g.values.map((v) => ({ partition: g.label, sex: v.series, percent_of_stays: v.value, stays: v.n })))} units="% of stays">
				<BarChart groups={sexGroups} stacked colors={{ Female: '#fb7185', Male: '#38bdf8' }} unit="%" valueFormat={(v) => v.toFixed(1)} yLabel="% of stays" ariaLabel="Sex composition" />
			</ChartFrame>

			<ChartFrame title="ICU episode duration (quantile profile)" subtitle={`Hours at percentile 0…100 — ${pLabel(partition)}; legal-cutoff episodes only`} provenance={prov('cohort_duration', [partition])} data={durationSeries.flatMap((s) => s.points.map((p) => ({ series: s.name, percentile: p.x, duration_hours: p.y })))} units="hours">
				<LineChart series={durationSeries} xLabel="Percentile" yLabel="Episode duration" xUnit="%" yUnit="h" xDomain={[0, 100]} ariaLabel="Episode duration quantile profile" />
			</ChartFrame>

			<ChartFrame title="Legal cutoffs per stay" subtitle="Number of stays by count of canonical cutoffs (max 12)" provenance={prov('cohort_cutoffs_per_stay', [partition])} data={cutoffHist.map((b) => ({ cutoffs_per_stay: Math.round((b.x0 + b.x1) / 2), stays: b.count }))} units="stays">
				<Histogram bins={cutoffHist} name="stays" xLabel="Legal cutoffs per stay" xUnit="count" yLabel="stays" ariaLabel="Legal cutoffs per stay" />
			</ChartFrame>

			<ChartFrame title="SOFA distribution (quantile profile)" subtitle="Baseline SOFA(t) total score; rows with a computable 24h horizon" provenance={prov('cohort_sofa_distributions', shown)} data={sofaOverall.flatMap((s) => s.points.map((p) => ({ partition: s.name, percentile: p.x, sofa_total: p.y })))} units="SOFA points">
				<LineChart series={sofaOverall} xLabel="Percentile" yLabel="Total SOFA" xUnit="%" yUnit="pts" xDomain={[0, 100]} ariaLabel="SOFA quantile profile" />
				{#if componentNote}<p class="note">Component-level SOFA: <b>not in the frozen export</b>. {componentNote}</p>{/if}
			</ChartFrame>

			<ChartFrame title="SOFA evolution by cutoff hour" subtitle="Mean ± SD (band) and median of total SOFA by canonical cutoff; later cutoffs contain fewer stays" provenance={prov('cohort_sofa_distributions', [partition])} data={sofaCutoffRows} units="SOFA points">
				<LineChart series={sofaCutoffSeries} xLabel="Hours since ICU admission at cutoff" yLabel="Total SOFA" xUnit="h" yUnit="pts" highlight={sofaSel ? { x: 24 + 6 * cutoff, y: sofaSel.mean, label: `cutoff ${cutoff}: mean ${sofaSel.mean.toFixed(2)}`, color: '#fb7185' } : undefined} ariaLabel="SOFA by cutoff hour" />
				{#if cutoffKeys.length}
					<div class="slider">
						<label for="cutoff-range">CUTOFF {cutoff} · +{24 + 6 * cutoff}h</label>
						<input id="cutoff-range" type="range" min={cutoffKeys[0]} max={cutoffKeys[cutoffKeys.length - 1]} step="1" bind:value={cutoff} />
						{#if sofaSel}
							<p>n={sofaSel.n.toLocaleString()} rows · mean {sofaSel.mean.toFixed(2)} · SD {sofaSel.sd.toFixed(2)} · p5 {sofaSel.p5} · p25 {sofaSel.p25} · median {sofaSel.p50} · p75 {sofaSel.p75} · p95 {sofaSel.p95}</p>
						{/if}
					</div>
				{/if}
			</ChartFrame>

			<ChartFrame title="Task eligibility matrix" subtitle="Fraction of legal cutoff rows eligible per task; hover for row counts" provenance={prov('cohort_eligibility', partitions)} data={eligMatrix?.flat ?? []} units="fraction of rows">
				{#if eligMatrix}
					<HeatmapGrid rows={eligMatrix.rows} cols={eligMatrix.cols} values={eligMatrix.values} counts={eligMatrix.counts} format={(v) => (v * 100).toFixed(1) + '%'} domain={[0, 1]} colorLabel="eligible / legal rows" ariaLabel="Task eligibility matrix" />
				{/if}
			</ChartFrame>

			<ChartFrame title="Organ-support outcome dispositions" subtitle={`Cutoff rows by label disposition — ${pLabel(partition)}`} provenance={prov('cohort_outcome_prevalence', [partition])} data={dispositionGroups.map((g) => ({ disposition: g.label, rows: g.values[0].value }))} units="rows">
				<BarChart groups={dispositionGroups} orientation="horizontal" colors={{ rows: '#fbbf24' }} valueFormat={(v) => v.toLocaleString()} unit="rows" ariaLabel="Support dispositions" />
				<p class="note">Positive = new qualifying vasopressor or invasive-ventilation initiation in (t, t+24h]. Censored rows are excluded from the task, never counted negative.</p>
			</ChartFrame>

			<ChartFrame title="Per-feature missingness" subtitle="Missing share of non-padded bins (%); support-state channels are always known" provenance={prov('cohort_missingness_feature', [partition])} data={missRows.clinical} units="% of non-padded bins">
				<BarChart groups={missGroups} orientation="horizontal" colors={{ missing: '#fb7185' }} unit="%" valueFormat={(v) => v.toFixed(1)} yDomain={[0, 100]} ariaLabel="Per-feature missingness" />
				{#if missRows.support.length}<p class="note">Always-known state channels (not shown): {missRows.support.join(', ')}.</p>{/if}
			</ChartFrame>

			<ChartFrame title="Eight-bin observation availability" subtitle="Observed share per channel and 6h bin of the (t−48h, t] window; padding excluded" provenance={prov('cohort_missingness_bin', [partition])} data={rowsFlat(binMatrix).map((r: any) => ({ channel: r.row, bin: r.col, observed_fraction: r.value, non_padded_rows: r.n }))} units="fraction observed">
				{#if binMatrix}
					<HeatmapGrid rows={binMatrix.rows} cols={binMatrix.cols} values={binMatrix.values} counts={binMatrix.counts} domain={[0, 1]} format={(v) => (v * 100).toFixed(0) + '%'} cellHeight={24} ariaLabel="Observation availability by bin" />
				{/if}
			</ChartFrame>

			<ChartFrame title="Structural padding by bin" subtitle="Share of cutoff rows whose bin precedes ICU admission (padding ≠ missingness)" provenance={prov('cohort_missingness_bin', [partition])} data={binMatrix?.padded ?? []} units="% of rows">
				<BarChart groups={paddedGroups} colors={{ padded: '#38bdf8' }} unit="%" valueFormat={(v) => v.toFixed(1)} yLabel="% padded" yDomain={[0, 100]} ariaLabel="Padding by bin" />
			</ChartFrame>

			<ChartFrame title="Selected channel correlations" subtitle={corr ? `Pearson r, newest 6h bin, pairwise-complete (${corr.evaluated} pairs evaluated; blank = not in the selected pairs)` : ''} provenance={prov('cohort_correlations', [corrPartition])} data={corr ? corr.vars.flatMap((a, i) => corr.vars.map((b, j) => ({ a, b, pearson_r: corr.values[i][j], n_rows: corr.counts[i][j] }))).filter((r) => r.pearson_r != null && r.a < r.b) : []} units="Pearson r" empty={!corr} emptyMessage="No correlation evidence for this partition.">
				{#if corr}
					<HeatmapGrid rows={corr.vars.map(channelLabel)} cols={corr.vars.map(channelLabel)} values={corr.values} counts={corr.counts} domain={[-1, 1]} format={(v) => v.toFixed(2)} cellHeight={22} ariaLabel="Channel correlations" />
				{/if}
			</ChartFrame>

			<ChartFrame title="Between-subject variability (ICC)" subtitle="Share of variance that is between subjects (ICC(1)); high = stable patient-level traits" provenance={prov('cohort_between_subject', [corrPartition])} data={icc} units="ICC(1)" empty={!icc.length}>
				<BarChart groups={icc.map((r) => ({ label: r.channel, values: [{ series: 'ICC(1)', value: r.icc1, n: r.n }] }))} orientation="horizontal" colors={{ 'ICC(1)': '#a78bfa' }} valueFormat={(v) => v.toFixed(2)} yDomain={[0, 1]} ariaLabel="Between-subject ICC" />
			</ChartFrame>

			<ChartFrame title="Observed channel distribution" subtitle={`Newest 6h bin, genuine observations${varN ? ` (n=${varN.toLocaleString()})` : ''}; units follow the frozen schema`} provenance={prov('cohort_variable_distributions', ['DEV_POOL'])} data={varSeries.flatMap((s) => s.points.map((p) => ({ partition: s.name, channel: variable, percentile: p.x, value: p.y })))} units="channel units">
				<label class="inline"><span>CHANNEL</span>
					<select bind:value={variable}>{#each varKeys as v}<option value={v}>{channelLabel(v)}</option>{/each}</select>
				</label>
				<LineChart series={varSeries} xLabel="Percentile" yLabel={channelLabel(variable)} xUnit="%" xDomain={[0, 100]} ariaLabel="Channel quantile profile" />
			</ChartFrame>
		</div>

		<p class="foot">
			Per-patient observation-density distributions and component-level SOFA are <b>not</b> part of the frozen characterization export, so they are not drawn.
			Percentile ladders are the exported summary statistics; no histogram was reconstructed from them.
		</p>
	{/if}
</WorkbenchPage>

<style>
	.state{padding:28px;border:1px solid rgba(148,163,184,.16);color:#8494a9;font:11px 'JetBrains Mono',monospace}.state.error{color:#fb7185;border-color:rgba(251,113,133,.4)}
	.controls{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:flex-end;margin-bottom:18px;padding:14px 16px;border:1px solid rgba(148,163,184,.14);background:rgba(7,13,25,.7)}
	.controls label{display:grid;gap:6px}.controls label>span,.inline>span{color:#53647b;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em}
	.controls .check{display:flex;align-items:center;gap:8px;padding-bottom:6px}
	select{min-width:220px;max-width:100%;padding:8px 10px;color:#dbe7e7;background:#060c18;border:1px solid rgba(148,163,184,.25);font:11px 'JetBrains Mono',monospace}
	select:focus-visible,input[type=range]:focus-visible{outline:2px solid #2bb8b0;outline-offset:2px}
	.scope{flex:1 1 280px;margin:0;color:#64748b;font:8px/1.7 'JetBrains Mono',monospace;letter-spacing:.06em;text-align:right}.scope b{color:#38bdf8}.scope code{color:#2bb8b0}
	.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:18px}
	.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px}
	.note{margin:10px 0 0;color:#71829a;font-size:11px;line-height:1.6}.note b{color:#fbbf24}
	.slider{display:grid;gap:6px;margin-top:12px}.slider label,.inline{color:#53647b;font:8px 'JetBrains Mono',monospace;letter-spacing:.1em}.slider input{width:100%;accent-color:#2bb8b0}.slider p{margin:0;color:#8494a9;font:10px/1.6 'JetBrains Mono',monospace}
	.inline{display:flex;align-items:center;gap:10px;margin-bottom:10px}
	.foot{margin:20px 0 0;color:#64748b;font-size:11px;line-height:1.7}.foot b{color:#fbbf24}
	@media(max-width:600px){.scope{text-align:left}select{min-width:0;width:100%}.controls label{width:100%}}
</style>
