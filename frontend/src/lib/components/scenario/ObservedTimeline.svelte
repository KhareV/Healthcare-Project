<script lang="ts">
	import type { Timeline } from '$lib/services/showcase';
	let {
		timeline,
		cutoffIndex,
		concept = $bindable(''),
		height = 230,
		title = 'Observed history'
	}: { timeline: Timeline | null; cutoffIndex: number; concept?: string; height?: number; title?: string } = $props();

	const W = 720, PAD = { top: 16, right: 18, bottom: 46, left: 46 };
	const label = (c: string) => c.replace(/_/g, ' ');

	const cutoffHours = $derived(timeline?.cutoff_hours ?? []);
	const nowH = $derived(cutoffHours[cutoffIndex] ?? 0);
	const xMax = $derived(Math.max(cutoffHours[cutoffHours.length - 1] ?? 1, nowH) + 2);

	// Only observations the model could see at the selected cutoff (first_visible_cutoff <= cutoffIndex).
	function visible(p: { first_visible_cutoff: number | null }) {
		return p.first_visible_cutoff !== null && p.first_visible_cutoff <= cutoffIndex;
	}
	const concepts = $derived(
		(timeline?.series ?? []).filter((s) => s.points.some((p) => visible(p))).map((s) => ({ concept: s.concept, unit: s.unit }))
	);
	$effect(() => {
		if (concepts.length && !concepts.some((c) => c.concept === concept)) {
			concept = (concepts.find((c) => c.concept === 'heart_rate') ?? concepts[0]).concept;
		}
	});
	const active = $derived((timeline?.series ?? []).find((s) => s.concept === concept) ?? null);
	const pts = $derived((active?.points ?? []).filter(visible).sort((a, b) => a.hours_since_admission - b.hours_since_admission));
	const yLo = $derived(pts.length ? Math.min(...pts.map((p) => p.value)) : 0);
	const yHi = $derived(pts.length ? Math.max(...pts.map((p) => p.value)) : 1);
	const pad = $derived((yHi - yLo || Math.abs(yHi) * 0.1 || 1) * 0.12);
	const y0 = $derived(yLo - pad), y1 = $derived(yHi + pad);
	const ph = $derived(height - PAD.top - PAD.bottom);
	const pw = W - PAD.left - PAD.right;
	const X = (h: number) => PAD.left + (h / xMax) * pw;
	const Y = (v: number) => PAD.top + ph - ((v - y0) / (y1 - y0 || 1)) * ph;
	const yTicks = $derived(Array.from({ length: 5 }, (_, i) => y0 + (i / 4) * (y1 - y0)));
	const xTicks = $derived(Array.from({ length: Math.floor(xMax / 12) + 1 }, (_, i) => i * 12));

	const runs = $derived(
		(timeline?.support_runs ?? [])
			.filter((r) => r.first_visible_cutoff !== null && r.first_visible_cutoff <= cutoffIndex)
			.map((r) => {
				const endKnown = r.end_first_visible_cutoff !== null && r.end_first_visible_cutoff <= cutoffIndex;
				return { ...r, drawEnd: endKnown ? r.end_hour : nowH, ongoing: !endKnown };
			})
	);
	const bandY = $derived(height - PAD.bottom + 8);
	const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
	let hover = $state<number | null>(null);
</script>

<figure class="tl" aria-label={`${title}: ${active ? label(active.concept) : 'no observations'}`}>
	<header>
		<div><span>{title.toUpperCase()}</span><b>Only data available at the selected cutoff is drawn</b></div>
		<select bind:value={concept} aria-label="Measurement shown">
			{#each concepts as c}<option value={c.concept}>{label(c.concept)} ({c.unit})</option>{/each}
		</select>
	</header>
	{#if !timeline || !active}
		<div class="empty">No observations are visible yet at this cutoff.</div>
	{:else}
		<svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} role="img" aria-label={`${label(active.concept)} observations up to the cutoff`}>
			{#each yTicks as t}
				<line x1={PAD.left} x2={W - PAD.right} y1={Y(t)} y2={Y(t)} stroke="rgba(148,163,184,.1)" />
				<text x={PAD.left - 5} y={Y(t) + 3} text-anchor="end" fill="#64748b" font-size="9" font-family="JetBrains Mono, monospace">{fmt(t)}</text>
			{/each}
			{#each xTicks as h}
				<text x={X(h)} y={height - PAD.bottom + 34} text-anchor="middle" fill="#64748b" font-size="9" font-family="JetBrains Mono, monospace">{h}h</text>
			{/each}
			<!-- Not-yet-visible future region -->
			<rect x={X(nowH)} y={PAD.top} width={Math.max(0, W - PAD.right - X(nowH))} height={ph} fill="rgba(2,6,16,.55)" />
			<text x={X(nowH) + 6} y={PAD.top + 11} fill="#53647b" font-size="8" font-family="JetBrains Mono, monospace">NOT VISIBLE TO THE MODEL AT THIS CUTOFF</text>
			{#each cutoffHours as h, i}
				<line x1={X(h)} x2={X(h)} y1={PAD.top} y2={PAD.top + ph} stroke={i === cutoffIndex ? '#2bb8b0' : 'rgba(148,163,184,.25)'} stroke-width={i === cutoffIndex ? 1.6 : 1} stroke-dasharray={i === cutoffIndex ? '' : '2 4'} opacity={i <= cutoffIndex ? 1 : 0.35} />
			{/each}
			<text x={X(nowH)} y={PAD.top + ph + 12} text-anchor="middle" fill="#2bb8b0" font-size="8" font-family="JetBrains Mono, monospace">t{cutoffIndex}</text>
			{#if pts.length > 1}
				<polyline fill="none" stroke="#38bdf8" stroke-width="1.5" points={pts.map((p) => `${X(p.hours_since_admission).toFixed(1)},${Y(p.value).toFixed(1)}`).join(' ')} />
			{/if}
			{#each pts as p, i}
				{@const fresh = cutoffIndex > 0 && p.first_visible_cutoff === cutoffIndex}
				<circle cx={X(p.hours_since_admission)} cy={Y(p.value)} r={fresh ? 5 : 3.2} fill={fresh ? '#fbbf24' : '#38bdf8'} stroke={fresh ? '#fde68a' : 'none'} onpointerenter={() => (hover = i)} onpointerleave={() => (hover = null)} role="presentation" />
			{/each}
			{#if hover !== null && pts[hover]}
				{@const p = pts[hover]}
				<g transform={`translate(${Math.min(X(p.hours_since_admission) + 8, W - 190)},${Math.max(Y(p.value) - 34, 4)})`}>
					<rect width="180" height="30" fill="#050a14" stroke="rgba(148,163,184,.3)" />
					<text x="8" y="13" fill="#dce9e8" font-size="10" font-family="JetBrains Mono, monospace">{fmt(p.value)} {active.unit}</text>
					<text x="8" y="25" fill="#64748b" font-size="9" font-family="JetBrains Mono, monospace">hour {p.hours_since_admission.toFixed(1)}{p.first_visible_cutoff === cutoffIndex && cutoffIndex > 0 ? ' · revealed after previous cutoff' : ''}</text>
				</g>
			{/if}
			<!-- Organ-support bands -->
			{#each runs as r, i}
				<g>
					<rect x={X(r.start_hour)} y={bandY + (i % 3) * 7} width={Math.max(2, X(r.drawEnd) - X(r.start_hour))} height="5" fill={r.support_type === 'VASOPRESSOR' ? '#fbbf24' : '#2bb8b0'} opacity={r.qualifying_for_initiation ? 0.95 : 0.4} />
					{#if r.ongoing}<text x={X(r.drawEnd) + 2} y={bandY + (i % 3) * 7 + 5} fill="#94a3b8" font-size="7">▸</text>{/if}
				</g>
			{/each}
		</svg>
		<footer>
			<span><i class="dot"></i> observed</span><span><i class="dot fresh"></i> revealed since previous cutoff</span>
			<span><i class="bar vaso"></i> vasopressor</span><span><i class="bar resp"></i> respiratory</span><span>faded = non-qualifying for initiation</span>
		</footer>
	{/if}
</figure>

<style>
	.tl{margin:0;padding:14px;border:1px solid rgba(148,163,184,.14);background:#070d18;min-width:0}
	header{display:flex;justify-content:space-between;gap:10px;align-items:start;margin-bottom:8px;flex-wrap:wrap}
	header span{display:block;color:#2bb8b0;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em}header b{display:block;margin-top:5px;color:#8494a9;font:400 11px Inter,sans-serif}
	select{padding:6px 8px;border:1px solid rgba(148,163,184,.2);background:#0a1220;color:#dce9e8;font:9px 'JetBrains Mono',monospace}
	.empty{padding:36px 0;text-align:center;color:#64748b;font:10px 'JetBrains Mono',monospace}
	footer{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;color:#64748b;font:8px 'JetBrains Mono',monospace;letter-spacing:.06em}
	.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#38bdf8;margin-right:5px}.dot.fresh{background:#fbbf24}
	.bar{display:inline-block;width:14px;height:5px;margin-right:5px}.bar.vaso{background:#fbbf24}.bar.resp{background:#2bb8b0}
</style>
