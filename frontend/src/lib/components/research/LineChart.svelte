<script lang="ts">
	/**
	 * LineChart — multi-series numeric line chart (pure SVG).
	 * Props: series[{name,color,points[{x,y,lo?,hi?}],dashed?,step?,width?}], xLabel,yLabel, xUnit,yUnit,
	 * xDomain?,yDomain?, diagonal (draw y=x reference), highlight{x,y,label,color}, height, xFormat,yFormat,
	 * ariaLabel. Non-finite y values break the line (gaps); lo/hi draw a CI band. Legend click toggles series;
	 * pointer/arrow keys move a crosshair with tooltip.
	 */
	import type { LineSeries } from '$lib/research/types';
	import { extent, linear, niceTicks, pad, layoutLegend } from '$lib/research/scale';
	import { AXIS_COLOR, GRID_COLOR, MONO } from '$lib/research/palette';
	import SvgLegend from './SvgLegend.svelte';
	import SvgTooltip from './SvgTooltip.svelte';

	let {
		series = [], xLabel = '', yLabel = '', xUnit = '', yUnit = '', xDomain, yDomain, diagonal = false,
		highlight, height = 300, xFormat, yFormat, ariaLabel = 'line chart'
	}: {
		series?: LineSeries[]; xLabel?: string; yLabel?: string; xUnit?: string; yUnit?: string;
		xDomain?: [number, number]; yDomain?: [number, number]; diagonal?: boolean;
		highlight?: { x: number; y: number; label: string; color?: string };
		height?: number; xFormat?: (v: number) => string; yFormat?: (v: number) => string; ariaLabel?: string;
	} = $props();

	let w = $state(640);
	const W = $derived(Math.max(280, w));
	let hidden = $state<Record<string, boolean>>({});
	let hoverX = $state<number | null>(null);
	const keyOf = (s: LineSeries, i: number) => s.key ?? `${s.name}#${i}`;
	const visible = $derived(series.map((s, i) => ({ ...s, _k: keyOf(s, i) })).filter((s) => !hidden[s._k]));
	const hasData = $derived(series.some((s) => s.points.some((p) => Number.isFinite(p.x) && Number.isFinite(p.y))));

	const legend = $derived(layoutLegend(series.map((s, i) => ({ key: keyOf(s, i), label: s.name, color: s.color, dash: s.dashed ? '5 3' : undefined, hidden: !!hidden[keyOf(s, i)] })), W - 24));
	const M = $derived({ l: 58, r: 16, t: 10 + (legend.height ? legend.height + 8 : 0), b: 46 });
	const plotW = $derived(W - M.l - M.r);
	const plotH = $derived(height - M.t - M.b);

	const xs = $derived(visible.flatMap((s) => s.points.map((p) => p.x)));
	const ys = $derived(visible.flatMap((s) => s.points.flatMap((p) => [p.y, p.lo ?? NaN, p.hi ?? NaN])));
	const xd = $derived<[number, number]>(xDomain ?? extent(xs) ?? [0, 1]);
	const yd = $derived<[number, number]>(yDomain ?? pad(extent(ys) ?? [0, 1], 0.06));
	const sx = $derived(linear(xd[0], xd[1], M.l, M.l + plotW));
	const sy = $derived(linear(yd[0], yd[1], M.t + plotH, M.t));
	const xt = $derived(niceTicks(xd[0], xd[1], Math.max(3, Math.floor(plotW / 90))));
	const yt = $derived(niceTicks(yd[0], yd[1], 5));
	const fx = (v: number) => (xFormat ? xFormat(v) : Number.isInteger(v) ? String(v) : v.toFixed(2));
	const fy = (v: number) => (yFormat ? yFormat(v) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));

	function segments(points: LineSeries['points']) {
		const out: LineSeries['points'][] = [];
		let cur: LineSeries['points'] = [];
		for (const p of points) {
			if (Number.isFinite(p.x) && Number.isFinite(p.y)) cur.push(p);
			else if (cur.length) { out.push(cur); cur = []; }
		}
		if (cur.length) out.push(cur);
		return out;
	}
	function path(seg: LineSeries['points'], step: boolean) {
		let d = '';
		seg.forEach((p, i) => {
			const x = sx(p.x), y = sy(p.y);
			if (i === 0) d += `M${x.toFixed(1)},${y.toFixed(1)}`;
			else if (step) d += `H${x.toFixed(1)}V${y.toFixed(1)}`;
			else d += `L${x.toFixed(1)},${y.toFixed(1)}`;
		});
		return d;
	}
	function band(seg: LineSeries['points']) {
		const ok = seg.filter((p) => Number.isFinite(p.lo) && Number.isFinite(p.hi));
		if (ok.length < 2) return '';
		const up = ok.map((p) => `${sx(p.x).toFixed(1)},${sy(p.hi as number).toFixed(1)}`);
		const dn = ok.slice().reverse().map((p) => `${sx(p.x).toFixed(1)},${sy(p.lo as number).toFixed(1)}`);
		return `M${up.join('L')}L${dn.join('L')}Z`;
	}
	const allX = $derived(Array.from(new Set(xs.filter(Number.isFinite))).sort((a, b) => a - b));
	function nearest(s: LineSeries, x: number) {
		let best: LineSeries['points'][number] | null = null;
		let bd = Infinity;
		for (const p of s.points) {
			if (!Number.isFinite(p.y)) continue;
			const d = Math.abs(p.x - x);
			if (d < bd) { bd = d; best = p; }
		}
		return best;
	}
	function onMove(e: PointerEvent) {
		const r = (e.currentTarget as SVGElement).getBoundingClientRect();
		const px = e.clientX - r.left;
		if (px < M.l || px > M.l + plotW || !allX.length) { hoverX = null; return; }
		const x = xd[0] + ((px - M.l) / plotW) * (xd[1] - xd[0]);
		hoverX = allX.reduce((a, b) => (Math.abs(b - x) < Math.abs(a - x) ? b : a), allX[0]);
	}
	function onKey(e: KeyboardEvent) {
		if (!allX.length) return;
		const i = hoverX == null ? -1 : allX.indexOf(hoverX);
		if (e.key === 'ArrowRight') { e.preventDefault(); hoverX = allX[Math.min(allX.length - 1, i + 1)]; }
		else if (e.key === 'ArrowLeft') { e.preventDefault(); hoverX = allX[Math.max(0, i < 0 ? 0 : i - 1)]; }
		else if (e.key === 'Escape') hoverX = null;
	}
	const tip = $derived.by(() => {
		if (hoverX == null) return null;
		const lines = visible.map((s) => ({ s, p: nearest(s, hoverX as number) })).filter((o) => o.p).map((o) => ({
			text: `${o.s.name}: ${fy((o.p as { y: number }).y)}${yUnit ? ' ' + yUnit : ''}` +
				(Number.isFinite((o.p as { lo?: number }).lo) && Number.isFinite((o.p as { hi?: number }).hi) ? ` [${fy((o.p as { lo: number }).lo)}, ${fy((o.p as { hi: number }).hi)}]` : ''),
			color: o.s.color
		}));
		return { title: `${xLabel || 'x'} = ${fx(hoverX)}${xUnit ? ' ' + xUnit : ''}`, lines };
	});
</script>

<div class="host" bind:clientWidth={w}>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
	<svg data-chart-root viewBox={`0 0 ${W} ${height}`} width={W} {height} role="img" aria-label={ariaLabel}
		tabindex="0" onpointermove={onMove} onpointerleave={() => (hoverX = null)} onkeydown={onKey} onblur={() => (hoverX = null)}>
		<title>{ariaLabel}</title>
		{#if !hasData}
			<text x={W / 2} y={height / 2} text-anchor="middle" fill="#71829a" font-size="11" font-family={MONO}>No data supplied</text>
		{:else}
			{#if legend.height}<SvgLegend layout={legend} x={12} y={6} onToggle={(k) => (hidden = { ...hidden, [k]: !hidden[k] })} />{/if}
			{#each yt as t}
				<line x1={M.l} x2={M.l + plotW} y1={sy(t)} y2={sy(t)} stroke={GRID_COLOR} />
				<text x={M.l - 6} y={sy(t) + 3} text-anchor="end" fill={AXIS_COLOR} font-size="10" font-family={MONO}>{fy(t)}</text>
			{/each}
			{#each xt as t}
				<line x1={sx(t)} x2={sx(t)} y1={M.t} y2={M.t + plotH} stroke={GRID_COLOR} />
				<text x={sx(t)} y={M.t + plotH + 14} text-anchor="middle" fill={AXIS_COLOR} font-size="10" font-family={MONO}>{fx(t)}</text>
			{/each}
			<rect x={M.l} y={M.t} width={plotW} height={plotH} fill="none" stroke="rgba(148,163,184,.22)" />
			{#if diagonal}<line x1={sx(Math.max(xd[0], yd[0]))} y1={sy(Math.max(xd[0], yd[0]))} x2={sx(Math.min(xd[1], yd[1]))} y2={sy(Math.min(xd[1], yd[1]))} stroke="#53647b" stroke-dasharray="4 4" />{/if}
			{#each visible as s (s._k)}
				{#each segments(s.points) as seg}
					{#if band(seg)}<path d={band(seg)} fill={s.color} fill-opacity="0.14" stroke="none" />{/if}
					{#if seg.length > 1}
						<path d={path(seg, !!s.step)} fill="none" stroke={s.color} stroke-width={s.width ?? 2} stroke-dasharray={s.dashed ? '6 4' : ''} stroke-linejoin="round" />
					{:else}
						<circle cx={sx(seg[0].x)} cy={sy(seg[0].y)} r="3" fill={s.color} />
					{/if}
				{/each}
			{/each}
			{#if highlight && Number.isFinite(highlight.x) && Number.isFinite(highlight.y)}
				<circle cx={sx(highlight.x)} cy={sy(highlight.y)} r="5" fill="none" stroke={highlight.color ?? '#fbbf24'} stroke-width="2" />
				<text x={sx(highlight.x) + 8} y={sy(highlight.y) - 8} fill={highlight.color ?? '#fbbf24'} font-size="10" font-family={MONO}>{highlight.label}</text>
			{/if}
			{#if hoverX != null}
				<line x1={sx(hoverX)} x2={sx(hoverX)} y1={M.t} y2={M.t + plotH} stroke="#94a3b8" stroke-dasharray="3 3" data-export-hide />
				{#each visible as s (s._k)}
					{@const p = nearest(s, hoverX)}
					{#if p}<circle cx={sx(p.x)} cy={sy(p.y)} r="4" fill={s.color} stroke="#050a15" data-export-hide />{/if}
				{/each}
			{/if}
			<text x={M.l + plotW / 2} y={height - 6} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>{xLabel}{xUnit ? ` (${xUnit})` : ''}</text>
			<text transform={`translate(12,${M.t + plotH / 2}) rotate(-90)`} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>{yLabel}{yUnit ? ` (${yUnit})` : ''}</text>
			{#if tip && hoverX != null}
				<SvgTooltip x={sx(hoverX)} y={M.t + plotH / 2} lines={tip.lines} title={tip.title} bounds={{ w: W, h: height }} />
			{/if}
		{/if}
	</svg>
</div>

<style>
	.host{width:100%;min-width:0}
	svg{display:block;max-width:100%;outline:none}
	svg:focus-visible{outline:2px solid #38bdf8;outline-offset:2px}
</style>
