<script lang="ts">
	import MultiLine from '$lib/components/dashboard/MultiLine.svelte';
	import type { PredictionResponse } from '$lib/services/api';
	let { history, labels, observedSofa = [] }: { history: (PredictionResponse | null)[]; labels: string[]; observedSofa?: number[] } = $props();
	const num = (f: (p: PredictionResponse) => number) => history.map((p) => (p ? f(p) : NaN));
	const sofaNow = $derived(num((p) => p.current_sofa));
	const rec24 = $derived(num((p) => p.recovery.sofa_hat_24h));
	const rec48 = $derived(num((p) => p.recovery.sofa_hat_48h));
	const icu = $derived(num((p) => p.icu_stay_time.remaining_hours));
	const risk = $derived(num((p) => p.organ_support.probability_24h * 100));
	const thr = $derived(history.map((p) => (p ? p.organ_support.threshold * 100 : NaN)));
</script>

<div class="strip">
	<div class="card"><header><span>SOFA · OBSERVED VS FORECAST</span><b>OBSERVED NOW · +24h · +48h</b></header>
		<MultiLine series={[{ name: 'Observed SOFA(t)', color: '#2bb8b0', values: sofaNow }, { name: 'Forecast +24h', color: '#38bdf8', values: rec24 }, { name: 'Forecast +48h', color: '#a78bfa', values: rec48 }]} xLabels={labels} yMin={0} yMax={24} height={150} /></div>
	<div class="card"><header><span>REMAINING ICU TIME</span><b>HOURS · FORECAST</b></header>
		<MultiLine series={[{ name: 'Remaining hours', color: '#fbbf24', values: icu }]} xLabels={labels} yMin={0} yMax={Math.max(24, ...icu.filter(Number.isFinite), 1)} height={150} /></div>
	<div class="card"><header><span>SUPPORT INITIATION RISK (24h)</span><b>CALIBRATED % VS FROZEN THRESHOLD</b></header>
		<MultiLine series={[{ name: 'Calibrated probability', color: '#fb7185', values: risk }, { name: 'Frozen threshold', color: 'rgba(251,191,36,.6)', values: thr }]} xLabels={labels} yMin={0} yMax={100} height={150} /></div>
</div>

<style>
	.strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.card{padding:14px;border:1px solid rgba(148,163,184,.14);background:#070d18;min-width:0}
	header{display:flex;justify-content:space-between;gap:8px;margin-bottom:8px;color:#64748b;font:7px 'JetBrains Mono',monospace;letter-spacing:.1em}header span{color:#2bb8b0}header b{font-weight:400}
	@media(max-width:1100px){.strip{grid-template-columns:1fr}}
</style>
