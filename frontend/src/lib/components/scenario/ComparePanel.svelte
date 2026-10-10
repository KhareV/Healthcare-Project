<script lang="ts">
	import MultiLine from '$lib/components/dashboard/MultiLine.svelte';
	import GaugeRing from '$lib/components/dashboard/GaugeRing.svelte';
	import ObservedTimeline from './ObservedTimeline.svelte';
	import ContributorPanel from './ContributorPanel.svelte';
	import { prettySubtype } from '$lib/services/showcase';
	import type { ReplaySession } from './session.svelte';

	let { a, b, index }: { a: ReplaySession; b: ReplaySession; index: number } = $props();
	let conceptA = $state('');
	let conceptB = $state('');
	let taskA = $state<'recovery24' | 'recovery48' | 'icu_stay_time' | 'organ_support'>('organ_support');
	let taskB = $state<'recovery24' | 'recovery48' | 'icu_stay_time' | 'organ_support'>('organ_support');

	const n = $derived(Math.min(a.cutoffs.length, b.cutoffs.length));
	const labels = $derived(Array.from({ length: Math.min(index + 1, n) }, (_, i) => `t${i}`));
	const series = (s: ReplaySession, f: (p: NonNullable<ReplaySession['current']>) => number) =>
		Array.from({ length: Math.min(index + 1, n) }, (_, i) => (s.predictions[i] ? f(s.predictions[i]) : NaN));
	const charts = $derived([
		{ title: 'OBSERVED SOFA(t)', unit: 'SOFA', max: 24, f: (p: any) => p.current_sofa },
		{ title: 'FORECAST SOFA +24h', unit: 'SOFA', max: 24, f: (p: any) => p.recovery.sofa_hat_24h },
		{ title: 'REMAINING ICU TIME', unit: 'h', max: 0, f: (p: any) => p.icu_stay_time.remaining_hours },
		{ title: 'SUPPORT RISK 24h', unit: '%', max: 100, f: (p: any) => p.organ_support.probability_24h * 100 }
	]);
	const ca = $derived(a.predictions[index] ?? null);
	const cb = $derived(b.predictions[index] ?? null);
	const rows = [
		{ k: 'SOFA now', f: (p: any) => p.current_sofa.toFixed(1) },
		{ k: 'Forecast +24h', f: (p: any) => p.recovery.sofa_hat_24h.toFixed(1) },
		{ k: 'Forecast +48h', f: (p: any) => p.recovery.sofa_hat_48h.toFixed(1) },
		{ k: 'Remaining ICU (h)', f: (p: any) => p.icu_stay_time.remaining_hours.toFixed(1) },
		{ k: 'Support risk (24h)', f: (p: any) => `${(p.organ_support.probability_24h * 100).toFixed(1)}%${p.organ_support.alert ? ' · above threshold' : ''}` },
		{ k: 'Observed bins', f: (p: any) => `${p.data_quality.observed_bins}/${p.data_quality.total_bins}` },
		{ k: 'Observed values', f: (p: any) => `${Math.round((p.data_quality.observed_feature_fraction ?? 0) * 100)}%` }
	];
</script>

<section class="cmp" data-testid="compare-view">
	<p class="caveat">Two hand-picked patients illustrate different trajectories; they are <b>not</b> a cohort sample and say nothing about average model performance. Cutoffs are aligned by index (t0 = ICU hour 24, then every 6 h).</p>
	<div class="charts">
		{#each charts as c}
			<div class="card"><header><span>{c.title}</span><b>{c.unit}</b></header>
				<MultiLine series={[{ name: a.participant?.alias ?? 'A', color: '#2bb8b0', values: series(a, c.f) }, { name: b.participant?.alias ?? 'B', color: '#fbbf24', values: series(b, c.f) }]} xLabels={labels} yMin={0} yMax={c.max || Math.max(24, ...series(a, c.f).filter(Number.isFinite), ...series(b, c.f).filter(Number.isFinite))} height={140} />
				<footer><i class="a"></i>{a.participant?.alias} <i class="b"></i>{b.participant?.alias}</footer></div>
		{/each}
	</div>
	<div class="cols">
		{#each [{ s: a, cur: ca, which: 'A' }, { s: b, cur: cb, which: 'B' }] as col}
			<article class="col" data-testid={`compare-col-${col.which}`}>
				<header><span>PATIENT {col.which}</span><h3>{col.s.participant?.title}</h3><small>{col.s.participant ? prettySubtype(col.s.participant.subtype) : ''} · {col.s.participant?.alias}</small></header>
				{#if col.cur}
					<table><tbody>{#each rows as r}<tr><td>{r.k}</td><td>{r.f(col.cur)}</td></tr>{/each}</tbody></table>
					<div class="dq"><GaugeRing value={Math.round((col.cur.data_quality.observed_feature_fraction ?? 0) * 100)} size={50} stroke={6} color="#2bb8b0" label="DATA" /><p>{col.cur.data_quality.padding_bins} pre-admission padding bins · completeness is not model confidence</p></div>
				{:else}<p class="muted">Computing forecast…</p>{/if}
				{#if col.which === 'A'}
					<ObservedTimeline timeline={col.s.timeline} cutoffIndex={index} bind:concept={conceptA} height={200} title="Observed history" />
					<ContributorPanel prediction={col.cur} bind:task={taskA} compact />
				{:else}
					<ObservedTimeline timeline={col.s.timeline} cutoffIndex={index} bind:concept={conceptB} height={200} title="Observed history" />
					<ContributorPanel prediction={col.cur} bind:task={taskB} compact />
				{/if}
			</article>
		{/each}
	</div>
</section>

<style>
	.cmp{display:grid;gap:8px}.caveat{margin:0;padding:10px 12px;border:1px solid rgba(251,191,36,.25);color:#8fa3b8;font:9px 'JetBrains Mono',monospace;line-height:1.6}.caveat b{color:#fbbf24}
	.charts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}.card{padding:12px;border:1px solid rgba(148,163,184,.14);background:#070d18;min-width:0}
	.card header{display:flex;justify-content:space-between;margin-bottom:6px;font:7px 'JetBrains Mono',monospace;letter-spacing:.1em;color:#2bb8b0}.card header b{color:#64748b;font-weight:400}
	.card footer{display:flex;gap:10px;align-items:center;margin-top:4px;color:#8494a9;font:8px 'JetBrains Mono',monospace}.card footer i{display:inline-block;width:12px;height:3px}.card footer i.a{background:#2bb8b0}.card footer i.b{background:#fbbf24}
	.cols{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.col{display:grid;gap:8px;align-content:start;min-width:0;padding:14px;border:1px solid rgba(148,163,184,.16);background:#080f1b}
	.col header span{color:#2bb8b0;font:7px 'JetBrains Mono',monospace;letter-spacing:.14em}.col h3{margin:5px 0 2px;color:#e5f1f0;font:500 16px 'Space Grotesk',sans-serif}.col small{color:#64748b;font:8px 'JetBrains Mono',monospace}
	table{width:100%;border-collapse:collapse;font-size:12px;color:#cfe3e1}td{padding:6px 4px;border-bottom:1px solid rgba(148,163,184,.08)}td:first-child{color:#64748b;font:8px 'JetBrains Mono',monospace;letter-spacing:.08em}td:last-child{text-align:right}
	.dq{display:flex;gap:12px;align-items:center}.dq p{margin:0;color:#64748b;font:8px 'JetBrains Mono',monospace;line-height:1.6}.muted{color:#64748b;font-size:11px}
	@media(max-width:1100px){.charts{grid-template-columns:repeat(2,minmax(0,1fr))}}
	@media(max-width:760px){.cols,.charts{grid-template-columns:1fr}}
</style>
