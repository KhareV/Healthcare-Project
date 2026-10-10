<script lang="ts">
	/**
	 * Histogram — pre-binned bars {x0,x1,count}. Props: bins, overlay? (second pre-binned series on the same
	 * bin edges), name, overlayName, color, overlayColor, xLabel, xUnit, yLabel, height, ariaLabel.
	 * Toggle between counts and density (count / (total * width)); legend click hides a series.
	 */
	import { extent, linear, niceTicks, layoutLegend } from '$lib/research/scale';
	import { AXIS_COLOR, GRID_COLOR, MONO } from '$lib/research/palette';
	import SvgLegend from './SvgLegend.svelte';
	import SvgTooltip from './SvgTooltip.svelte';

	type Bin = { x0: number; x1: number; count: number };
	let {
		bins = [], overlay = [], name = 'count', overlayName = 'overlay', color = '#2bb8b0', overlayColor = '#fbbf24',
		xLabel = '', xUnit = '', yLabel = 'rows', height = 260, ariaLabel = 'histogram'
	}: {
		bins?: Bin[]; overlay?: Bin[]; name?: string; overlayName?: string; color?: string; overlayColor?: string;
		xLabel?: string; xUnit?: string; yLabel?: string; height?: number; ariaLabel?: string;
	} = $props();

	let w = $state(640);
	const W = $derived(Math.max(280, w));
	let density = $state(false);
	let hidden = $state<Record<string, boolean>>({});
	let hov = $state<{ i: number; k: 'a' | 'b' } | null>(null);
	const hasData = $derived(bins.some((b) => Number.isFinite(b.count)));
	const items = $derived([
		{ key: 'a', label: name, color, hidden: !!hidden.a },
		...(overlay.length ? [{ key: 'b', label: overlayName, color: overlayColor, hidden: !!hidden.b }] : []),
		{ key: 'dens', label: density ? 'density (click: counts)' : 'counts (click: density)', color: '#94a3b8' }
	]);
	const legend = $derived(layoutLegend(items, W - 24));
	const M = $derived({ l: 54, r: 14, t: 10 + legend.height + 8, b: 44 });
	const plotW = $derived(W - M.l - M.r);
	const plotH = $derived(height - M.t - M.b);
	const total = (b: Bin[]) => b.reduce((s, x) => s + (Number.isFinite(x.count) ? x.count : 0), 0);
	const val = (b: Bin, tot: number) => (density ? b.count / (tot * Math.max(1e-12, b.x1 - b.x0)) : b.count);
	const ta = $derived(total(bins));
	const tb = $derived(total(overlay));
	const xd = $derived<[number, number]>(extent([...bins, ...overlay].flatMap((b) => [b.x0, b.x1])) ?? [0, 1]);
	const ymax = $derived(Math.max(1e-9, ...(hidden.a ? [] : bins.map((b) => val(b, ta))), ...(hidden.b ? [] : overlay.map((b) => val(b, tb)))));
	const sx = $derived(linear(xd[0], xd[1], M.l, M.l + plotW));
	const sy = $derived(linear(0, ymax * 1.08, M.t + plotH, M.t));
	const xt = $derived(niceTicks(xd[0], xd[1], Math.max(3, Math.floor(plotW / 80))));
	const yt = $derived(niceTicks(0, ymax * 1.08, 4));
	const fy = (v: number) => (density ? v.toPrecision(2) : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v)));
	const tip = $derived.by(() => {
		if (!hov) return null;
		const b = (hov.k === 'a' ? bins : overlay)[hov.i];
		if (!b) return null;
		const tot = hov.k === 'a' ? ta : tb;
		return { x: sx((b.x0 + b.x1) / 2), y: sy(val(b, tot)), title: `${b.x0.toFixed(2)} – ${b.x1.toFixed(2)}${xUnit ? ' ' + xUnit : ''}`,
			lines: [{ text: `${hov.k === 'a' ? name : overlayName}: ${b.count}`, color: hov.k === 'a' ? color : overlayColor }, { text: `share: ${(100 * b.count / tot).toFixed(1)}%`, color: undefined as unknown as string }] };
	});
	function toggle(k: string) {
		if (k === 'dens') density = !density;
		else hidden = { ...hidden, [k]: !hidden[k] };
	}
</script>

<div class="host" bind:clientWidth={w}>
	<svg data-chart-root viewBox={`0 0 ${W} ${height}`} width={W} {height} role="img" aria-label={ariaLabel}>
		<title>{ariaLabel}</title>
		{#if !hasData}
			<text x={W / 2} y={height / 2} text-anchor="middle" fill="#71829a" font-size="11" font-family={MONO}>No data supplied</text>
		{:else}
			<SvgLegend layout={legend} x={12} y={6} onToggle={toggle} />
			{#each yt as t}
				<line x1={M.l} x2={M.l + plotW} y1={sy(t)} y2={sy(t)} stroke={GRID_COLOR} />
				<text x={M.l - 6} y={sy(t) + 3} text-anchor="end" fill={AXIS_COLOR} font-size="10" font-family={MONO}>{fy(t)}</text>
			{/each}
			{#each xt as t}
				<text x={sx(t)} y={M.t + plotH + 14} text-anchor="middle" fill={AXIS_COLOR} font-size="10" font-family={MONO}>{Number.isInteger(t) ? t : t.toFixed(1)}</text>
			{/each}
			{#if !hidden.a}
				{#each bins as b, i}
					<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
					<rect role="img" tabindex="0" aria-label={`${b.x0} to ${b.x1}: ${b.count}`} x={sx(b.x0) + 0.5} y={sy(val(b, ta))} width={Math.max(1, sx(b.x1) - sx(b.x0) - 1)} height={Math.max(0, M.t + plotH - sy(val(b, ta)))}
						fill={color} fill-opacity={hov?.k === 'a' && hov.i === i ? 1 : 0.8} onpointerenter={() => (hov = { i, k: 'a' })} onpointerleave={() => (hov = null)} onfocus={() => (hov = { i, k: 'a' })} onblur={() => (hov = null)} style="outline:none" />
				{/each}
			{/if}
			{#if overlay.length && !hidden.b}
				{#each overlay as b, i}
					<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
					<rect role="img" tabindex="0" aria-label={`${overlayName} ${b.x0} to ${b.x1}: ${b.count}`} x={sx(b.x0) + 0.5} y={sy(val(b, tb))} width={Math.max(1, sx(b.x1) - sx(b.x0) - 1)} height={Math.max(0, M.t + plotH - sy(val(b, tb)))}
						fill="none" stroke={overlayColor} stroke-width="1.6" onpointerenter={() => (hov = { i, k: 'b' })} onpointerleave={() => (hov = null)} onfocus={() => (hov = { i, k: 'b' })} onblur={() => (hov = null)} style="outline:none" />
				{/each}
			{/if}
			<line x1={M.l} x2={M.l + plotW} y1={M.t + plotH} y2={M.t + plotH} stroke="rgba(148,163,184,.4)" />
			<text x={M.l + plotW / 2} y={height - 6} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>{xLabel}{xUnit ? ` (${xUnit})` : ''}</text>
			<text transform={`translate(12,${M.t + plotH / 2}) rotate(-90)`} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>{density ? 'density' : yLabel}</text>
			{#if tip}<SvgTooltip x={tip.x} y={tip.y} lines={tip.lines} title={tip.title} bounds={{ w: W, h: height }} />{/if}
		{/if}
	</svg>
</div>
<style>.host{width:100%;min-width:0}svg{display:block;max-width:100%}</style>
