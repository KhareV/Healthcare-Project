<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchPage from '$lib/components/research-pages/ResearchPage.svelte';
	import DataTable from '$lib/components/research-pages/DataTable.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import { ChartFrame, BarChart, ForestPlot } from '$lib/components/research';
	import { fmtCI, fmtInt, fmtNum, fmtSigned } from '$lib/research/format';
	import { researchApi, errorText, toProvenance, type Row, type Wrapped } from '$lib/services/research';

	type DiagnosticPayload = { latent_state: Wrapped<Row>; candidate_development: Wrapped<Row>; status: string };
	let evidence = $state<DiagnosticPayload | null>(null);
	let error = $state('');
	onMount(async () => {
		try { evidence = await researchApi.diagnostic() as DiagnosticPayload; }
		catch (cause) { error = errorText(cause); }
	});

	const latent = $derived(evidence?.latent_state);
	const dev = $derived(evidence?.candidate_development);
	const oracleRows = $derived<Row[]>(latent?.data?.rows ?? []);
	const familyRows = $derived<Row[]>(dev?.data?.rows ?? []);
	const holdRows = $derived<Row[]>(dev?.data?.held_out ?? []);
	const reference = $derived(oracleRows.find((r) => r.model === 'REFERENCE_V3'));
	const largeB = $derived(oracleRows.find((r) => r.model === 'ORACLE_B_LARGE_XGBOOST'));
	const mc = $derived(oracleRows.find((r) => r.model === 'ORACLE_C_MC'));
	const paired = $derived<Row>(dev?.data?.paired_vs_v3 ?? {});
	const gateC = $derived<Row>(dev?.data?.gate_c ?? {});

	const LABEL: Record<string, string> = {
		REFERENCE_V3: 'Frozen V3 (development reference)',
		OBSERVED_REFIT_CONTROL_XGB: 'Observed-data refit',
		ORACLE_A_LOGISTIC: 'Oracle A · five-state',
		ORACLE_A_XGBOOST: 'Oracle A · five-state XGB',
		ORACLE_B_LOGISTIC: 'Oracle B · augmented state',
		ORACLE_B_XGBOOST: 'Oracle B · augmented XGB',
		ORACLE_B_LARGE_LOGISTIC: 'Oracle B · larger fit',
		ORACLE_B_LARGE_XGBOOST: 'Oracle B · larger fit XGB',
		ORACLE_C_MC: 'Oracle C · generator Monte Carlo',
		V3_FROZEN: 'Frozen V3',
		CANDIDATE: 'V3.1 candidate ensemble',
		NEW_MODEL_ONLY_CALIBRATED: 'New model only, calibrated'
	};
	const label = (key: string) => LABEL[key] ?? key;
	const oracleOrder = ['REFERENCE_V3', 'ORACLE_A_LOGISTIC', 'ORACLE_B_LOGISTIC', 'ORACLE_B_LARGE_XGBOOST', 'ORACLE_C_MC'];
	const oracleSubset = $derived(oracleOrder.map((name) => oracleRows.find((r) => r.model === name)).filter(Boolean) as Row[]);
	const mainGroups = $derived(oracleSubset.map((r) => ({ label: label(r.model), values: [
		{ series: 'AUPRC', value: r.auprc as number, n: r.n_stays as number },
		{ series: 'AUROC', value: r.auroc as number, n: r.n_stays as number }
	] })));
	const forestRows = $derived(oracleSubset.filter((r) => r.model !== 'REFERENCE_V3' && Number.isFinite(r.gap_ci95_lo)).map((r) => ({
		label: label(r.model), estimate: r.gap_vs_v3_auprc as number, lo: r.gap_ci95_lo as number, hi: r.gap_ci95_hi as number,
		color: '#a78bfa', note: 'Paired 95% stay-cluster bootstrap interval'
	})));
	const familyGroups = $derived(familyRows.map((r) => ({ label: r.feature_family as string, values: [{ series: 'Best cross-fitted AUPRC', value: r.best_oof_auprc as number }] })));
	const holdGroups = $derived(holdRows.map((r) => ({ label: label(r.model), values: [
		{ series: 'AUPRC', value: r.auprc as number, n: r.n_stays as number },
		{ series: 'AUROC', value: r.auroc as number, n: r.n_stays as number }
	] })));
	const pLatent = $derived({ ...toProvenance(latent?.provenance), status: 'PRIVILEGED' as const });
	const pDev = $derived(toProvenance(dev?.provenance));
</script>

<svelte:head><title>Latent-State Diagnostic | Personalized Patient Recovery Trajectory</title></svelte:head>

<ResearchPage eyebrow="03.8 / PRIVILEGED DIAGNOSTIC" title="What the hidden simulator state reveals." description="A V3.1 research diagnostic asked whether information absent from the observed record could improve organ-support prediction. Its oracle scores are never serving scores or clinical bounds." {error} loading={!evidence && !error}>
	{#if latent && dev}
		<p class="rnote priv"><b>Research-only boundary.</b> Oracle A/B use hidden simulator state; Oracle C uses the conditional generator. Neither is available to the product forecast. The frozen V3 reference below is scored on the same fresh development rows; its original sealed-test result remains the current final evaluation.</p>
		<div class="rtiles">
			<MetricTile label="Frozen V3 · diagnostic AUPRC" value={fmtNum(reference?.auprc, 3)} detail="DIAG_EVAL development cohort" tone="teal" />
			<MetricTile label="Five-state oracle · AUPRC" value={fmtNum(oracleRows.find((r) => r.model === 'ORACLE_A_LOGISTIC')?.auprc, 3)} detail="exact latent coordinates at cutoff" tone="sky" />
			<MetricTile label="Augmented oracle · AUPRC" value={fmtNum(largeB?.auprc, 3)} detail="bounded verification, larger fit" tone="violet" />
			<MetricTile label="Monte Carlo reference · AUPRC" value={fmtNum(mc?.auprc, 3)} detail="200 conditional continuations · privileged" tone="amber" />
		</div>
		<div class="rgrid two">
			<ChartFrame title="Same-cohort diagnostic scores" subtitle="AUPRC and AUROC on DIAG_EVAL; Oracle A/B/C have privileged information" provenance={pLatent} data={oracleSubset} units="score" empty={!oracleSubset.length}>
				<BarChart groups={mainGroups} colors={{ AUPRC: '#a78bfa', AUROC: '#38bdf8' }} orientation="horizontal" showValues={true} valueFormat={(v) => v.toFixed(3)} yDomain={[0, 1]} height={330} ariaLabel="Development diagnostic AUPRC and AUROC" />
			</ChartFrame>
			<ChartFrame title="Privileged information gap" subtitle="AUPRC difference from frozen V3 on identical rows; paired 95% intervals" provenance={pLatent} data={oracleSubset.filter((r) => r.model !== 'REFERENCE_V3')} units="AUPRC difference" empty={!forestRows.length}>
				<ForestPlot rows={forestRows} xLabel="AUPRC difference vs V3" format={(v) => fmtSigned(v, 3)} ariaLabel="Paired oracle AUPRC gaps versus V3" />
			</ChartFrame>
		</div>
		<Panel eyebrow="DIAGNOSTIC EVIDENCE" title="All tested information sets" note={`${fmtInt(reference?.n_rows)} ROWS / ${fmtInt(reference?.n_stays)} ELIGIBLE STAYS`}>
			<DataTable rows={oracleRows} columns={[
				{ key: 'model', label: 'Information set / estimator', fmt: (v) => label(String(v)) },
				{ key: 'auprc', label: 'AUPRC', align: 'right', fmt: (v) => fmtNum(v, 4) },
				{ key: 'auroc', label: 'AUROC', align: 'right', fmt: (v) => fmtNum(v, 4) },
				{ key: 'brier', label: 'Brier', align: 'right', fmt: (v) => fmtNum(v, 4) },
				{ key: 'log_loss', label: 'Log loss', align: 'right', fmt: (v) => fmtNum(v, 4) },
				{ key: 'gap_vs_v3_auprc', label: 'AUPRC gap vs V3 · 95% CI', align: 'right', fmt: (v, r) => v == null ? 'reference' : `${fmtSigned(v, 4)} ${fmtCI(r.gap_ci95_lo, r.gap_ci95_hi, 4)}` }
			]} />
		</Panel>
		<p class="rnote"><b>Information sets.</b> A sees the exact five-dimensional latent state at the cutoff. B also sees hidden subject effects, drift, equilibrium, support effects and shocks. C estimates risk with future Monte Carlo continuations of the generator. A/B are finite-sample learned models, not ceilings. On the primary 1,600-subject fit, B's gap did not clear Gate A; the one declared 7,900-subject verification did. Generator replay and truncation audits passed. The gap demonstrates information in privileged state, not recoverability from observed vitals and labs.</p>
		<h3 class="rh">Observed-data development</h3>
		<p class="rnote amber"><b>Candidate scope.</b> Forward-only EWMA, Kalman and momentum features were compared with the plain F0 baseline in 32 pre-declared grouped cross-validation trials. These use observed history only. All families tied within the frozen 0.003 AUPRC tolerance, so the simpler F0 set won.</p>
		<div class="rgrid two">
			<ChartFrame title="Causal filter feature families" subtitle="Best out-of-fold stay-balanced AUPRC per family; 5-fold development CV" provenance={pDev} data={familyRows} units="AUPRC" empty={!familyRows.length}>
				<BarChart groups={familyGroups} colors={{ 'Best cross-fitted AUPRC': '#38bdf8' }} yDomain={[0, 1]} valueFormat={(v) => v.toFixed(4)} height={300} ariaLabel="Causal filter feature-family comparison" />
			</ChartFrame>
			<ChartFrame title="Single held-out development decision" subtitle="V31_HOLD: frozen V3 vs selected V3.1 candidate and new model alone" provenance={pDev} data={holdRows} units="score" empty={!holdRows.length}>
				<BarChart groups={holdGroups} colors={{ AUPRC: '#2bb8b0', AUROC: '#38bdf8' }} yDomain={[0, 1]} valueFormat={(v) => v.toFixed(4)} height={300} ariaLabel="Held-out development candidate comparison" />
			</ChartFrame>
		</div>
		<Panel eyebrow="HELD-OUT DEVELOPMENT" title="Why V3 was retained" note={`${fmtInt(holdRows[0]?.n_rows)} ROWS / ${fmtInt(holdRows[0]?.n_stays)} STAYS`}>
			<DataTable rows={holdRows} columns={[
				{ key: 'model', label: 'Model', fmt: (v) => label(String(v)) },
				{ key: 'auprc', label: 'AUPRC', align: 'right', fmt: (v) => fmtNum(v, 4) },
				{ key: 'auroc', label: 'AUROC', align: 'right', fmt: (v) => fmtNum(v, 4) },
				{ key: 'brier', label: 'Brier', align: 'right', fmt: (v) => fmtNum(v, 4) },
				{ key: 'log_loss', label: 'Log loss', align: 'right', fmt: (v) => fmtNum(v, 4) }
			]} />
			<p class="rnote" style="margin-top:14px"><b>Gate C: {gateC.passes_gate_c ? 'passed' : 'failed'}.</b> Candidate − frozen V3 AUPRC was {fmtSigned(paired.diff_auprc, 4)} {fmtCI(paired.ci95?.[0], paired.ci95?.[1], 4)} (paired 95% CI). The prespecified gate required a gain of at least +0.015 with a lower bound above zero and stable gains across thirds. The candidate did not meet those criteria. Frozen V3 remains selected for all four heads; no V3.1 sealed test was created.</p>
		</Panel>
		<p class="rnote priv"><b>Scientific release.</b> This page reports development evidence only. The privileged oracles and the non-promoted V3.1 candidate are absent from serving. The final model evaluation is the original frozen V3 sealed-test result, available in the Classification &amp; Regression section.</p>
	{/if}
</ResearchPage>
