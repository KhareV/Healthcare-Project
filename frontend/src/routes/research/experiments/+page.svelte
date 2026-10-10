<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import Segmented from '$lib/components/research-pages/Segmented.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import { ChartFrame, LineChart, BarChart, StripLeaderboard } from '$lib/components/research';
	import { fmtNum, fmtInt } from '$lib/research/format';
	import { researchApi, errorText, taskLabel, toProvenance, TASKS, type Row, type Wrapped } from '$lib/services/research';

	let lc = $state<Wrapped<Row> | null>(null);
	let lb = $state<Wrapped<Row> | null>(null);
	let ab = $state<Wrapped<Row> | null>(null);
	let error = $state('');
	let task = $state('recovery24');

	onMount(async () => {
		try {
			[lc, lb, ab] = await Promise.all([researchApi.learningCurve(), researchApi.leaderboard(), researchApi.ablation()]);
		} catch (cause) {
			error = errorText(cause);
		}
	});

	const info = $derived(TASKS.find((t) => t.value === task)!);
	const lcRows = $derived<Row[]>(lc?.data?.rows ?? []);
	const lcSeries = (t: string) => [{ name: 'Frozen V2 recipe, repeat-0 95% CI band', color: '#2bb8b0', points: lcRows.filter((r) => r.task === t).sort((a, b) => a.size_subjects - b.size_subjects).map((r) => ({ x: r.size_subjects, y: r.mean, lo: r.ci_lower_repeat0, hi: r.ci_upper_repeat0 })) }];
	const timeSeries = $derived(TASKS.map((t, i) => ({ name: t.label, color: ['#2bb8b0', '#38bdf8', '#fbbf24', '#a78bfa'][i], points: lcRows.filter((r) => r.task === t.value).sort((a, b) => a.size_subjects - b.size_subjects).map((r) => ({ x: r.size_subjects, y: r.train_seconds_mean })) })));

	const trials = $derived<Row[]>((lb?.data?.rows ?? []).filter((r: Row) => r.task === task && r.stage !== 'S0'));
	const controlRow = $derived<Row | undefined>((lb?.data?.rows ?? []).find((r: Row) => r.task === task && r.stage === 'S0'));
	const leaderGroups = $derived.by(() => {
		const vals = trials.filter((t) => t.status === 'ok' && Number.isFinite(t.primary_mean)).map((t) => t.primary_mean as number).sort((a, b) => a - b);
		const q = (p: number) => vals[Math.min(vals.length - 1, Math.floor(p * vals.length))];
		const hi = vals.length ? q(0.9) + (q(0.9) - vals[0]) * 0.5 : 0;
		const families = [...new Set<string>(trials.map((t) => t.family))];
		return families.map((f) => ({ label: f, trials: trials.filter((t) => t.family === f).map((t) => {
			const bad = t.status !== 'ok' || !Number.isFinite(t.primary_mean) || t.primary_mean > hi * (info.lowerIsBetter ? 1 : 1e9) && info.lowerIsBetter;
			return { value: t.primary_mean as number, id: `${t.stage} · ${t.variant}`, failed: bad, note: bad ? 'failed / diverged (outside plotted range)' : `${t.stage} · ${t.variant} · fold SD ${fmtNum(t.primary_sd, 3)}` };
		}) }));
	});
	const leaderDomain = $derived.by<[number, number] | undefined>(() => {
		const v = trials.filter((t) => t.status === 'ok').map((t) => t.primary_mean as number).filter(Number.isFinite).sort((a, b) => a - b);
		if (!v.length) return undefined;
		const p90 = v[Math.floor(0.9 * (v.length - 1))];
		return info.lowerIsBetter ? [v[0] - (p90 - v[0]) * 0.1, p90 + (p90 - v[0]) * 0.2] : [v[0], v[v.length - 1]];
	});
	const best = $derived<Row[]>((lb?.data?.best_per_family ?? []).filter((r: Row) => r.task === task));
	const bestGroups = $derived(best.map((r) => ({ label: `${r.family} · ${r.variant} (${r.stage})`, values: [{ series: 'best configuration', value: r.primary_mean, lo: r.primary_mean - (r.primary_sd ?? 0), hi: r.primary_mean + (r.primary_sd ?? 0), n: r.n_features }] })));

	const abRows = $derived<Row[]>((ab?.data?.rows ?? []).filter((r: Row) => r.task === task));
	const s2 = (v: string) => (lb?.data?.rows ?? []).filter((r: Row) => r.task === task && r.stage === 'S2' && r.variant === v && r.status === 'ok').map((r: Row) => r.primary_mean as number);
	const abGroups = $derived(abRows.map((r) => { const s = s2(r.variant); return { label: r.variant, values: [{ series: info.metric, value: r.primary_mean, lo: s.length ? Math.min(...s) : undefined, hi: s.length ? Math.max(...s) : undefined, n: r.n_runs }] }; }));

	const s4best = $derived.by<Row | undefined>(() => {
		const r = (lb?.data?.rows ?? []).filter((x: Row) => x.task === task && x.stage === 'S4' && x.status === 'ok');
		if (!r.length) return undefined;
		return r.reduce((a: Row, b: Row) => ((info.lowerIsBetter ? b.primary_mean < a.primary_mean : b.primary_mean > a.primary_mean) ? b : a));
	});
	const ctrlGroups = $derived([{ label: taskLabel(task), values: [
		controlRow ? { series: 'V2 recipe, corrected features (control)', value: controlRow.primary_mean, lo: controlRow.primary_mean - controlRow.primary_sd, hi: controlRow.primary_mean + controlRow.primary_sd } : null,
		s4best ? { series: `Best focused-search candidate (${s4best.family} · ${s4best.variant})`, value: s4best.primary_mean, lo: s4best.primary_mean - s4best.primary_sd, hi: s4best.primary_mean + s4best.primary_sd } : null
	].filter(Boolean) as { series: string; value: number; lo: number; hi: number }[] }]);
	const p = (w: Wrapped<Row> | null) => toProvenance(w?.provenance);
</script>

<svelte:head><title>Selection &amp; Learning | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03.3 / SELECTION &amp; LEARNING" title="Learning curves, trials and selection." description="How much data helps, which model families were tried, and what the feature ablations showed. Negative and tied results are retained." {error} loading={!lc && !error}>
	{#if lc && lb && ab}
		<p class="rnote amber"><b>Development evidence.</b> {lc.data.caveat} Fold SDs and ranges below come from cross-fitted development runs; they are not sealed-test results.</p>

		<h3 class="rh">Learning curves — metric vs training subjects (evaluation cohort {lc.data.evaluation_cohort}, {fmtInt(lc.data.n_eval_subjects)} subjects)</h3>
		<div class="rgrid two">
			{#each TASKS as t}
				<ChartFrame title={`${t.label}`} subtitle={`${t.metric === 'mae' ? 'MAE' : t.metric === 'auprc' ? 'raw AUPRC' : 'weighted median AE'} (${t.unit}) · ${t.lowerIsBetter ? 'lower' : 'higher'} is better`} provenance={p(lc)} data={lcRows.filter((r) => r.task === t.value)} units={t.unit} empty={!lcSeries(t.value)[0].points.length}>
					<LineChart series={lcSeries(t.value)} xLabel="Training subjects" yLabel={t.unit} xUnit="" ariaLabel={`Learning curve ${t.label}`} />
				</ChartFrame>
			{/each}
		</div>
		<ChartFrame title="Training time vs size" subtitle="Mean fit seconds of the frozen recipe per training size" provenance={p(lc)} data={lcRows} units="seconds" empty={!timeSeries[0].points.length}>
			<LineChart series={timeSeries} xLabel="Training subjects" yLabel="Mean train seconds" yUnit="s" ariaLabel="Training time" />
		</ChartFrame>

		<h3 class="rh">Model selection — {taskLabel(task)}</h3>
		<Segmented label="Task" bind:value={task} options={TASKS.map((t) => ({ value: t.value, label: t.label }))} />
		<div class="rgrid two">
			<ChartFrame title="Trial leaderboard" subtitle={`Every search trial (S1–S4) by family · ${info.metric} (${info.unit}) · failed/diverged shown as ✕`} provenance={p(lb)} data={trials} units={info.unit} empty={!leaderGroups.length}>
				<StripLeaderboard groups={leaderGroups} metric={info.metric} unit={info.unit} lowerIsBetter={info.lowerIsBetter} domain={leaderDomain} />
			</ChartFrame>
			<ChartFrame title="Best configuration per family" subtitle="Error bars = ± fold SD across cross-fitting folds" provenance={p(lb)} data={best} units={info.unit} empty={!bestGroups.length}>
				<BarChart groups={bestGroups} orientation="horizontal" unit={info.unit} valueFormat={(v) => v.toFixed(3)} height={300} ariaLabel="Best per family" />
			</ChartFrame>
		</div>
		<div class="rgrid two">
			<ChartFrame title="Feature-variant ablation (stage S2)" subtitle="Bars = mean across seeds; whiskers = min–max of S2 trials. Differences are mostly within the pre-registered tie tolerance." provenance={p(ab)} data={abRows} units={info.unit} empty={!abGroups.length}>
				<BarChart groups={abGroups} orientation="horizontal" unit={info.unit} valueFormat={(v) => v.toFixed(3)} height={Math.max(260, abGroups.length * 34 + 60)} ariaLabel="Ablation" />
			</ChartFrame>
			<ChartFrame title="Control versus best candidate" subtitle="V2 recipe on corrected features vs best focused-search trial (5-fold cross-fit, ± fold SD)" provenance={p(lb)} data={[controlRow, s4best].filter(Boolean) as Row[]} units={info.unit} empty={!ctrlGroups[0].values.length}>
				<BarChart groups={ctrlGroups} unit={info.unit} valueFormat={(v) => v.toFixed(3)} height={300} ariaLabel="Control vs best candidate" />
				<p class="rnote" style="margin-top:10px">Candidate and control differ by less than the pre-registered tolerance for several tasks; for ICU time the V2 recipe was retained.</p>
			</ChartFrame>
		</div>

		<Panel eyebrow="TABLE" title={`Best per family — ${taskLabel(task)}`} note={`${fmtInt(lb.data.n_trials)} TRIALS · ${fmtInt(lb.data.n_failed)} FAILED`}>
			<DataTable
				columns={[
					{ key: 'family', label: 'Family' },
					{ key: 'variant', label: 'Feature variant' },
					{ key: 'stage', label: 'Stage' },
					{ key: 'primary_mean', label: `${info.metric} (${info.unit})`, align: 'right', fmt: (v) => fmtNum(v, 4) },
					{ key: 'primary_sd', label: 'Fold SD', align: 'right', fmt: (v) => fmtNum(v, 4) },
					{ key: 'n_features', label: 'Features', align: 'right' },
					{ key: 'status', label: 'Status' }
				]}
				rows={best}
				defaultSort="primary_mean"
			/>
		</Panel>
	{/if}
</ResearchPage>
