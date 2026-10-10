<script lang="ts">
	/**
	 * BarChart — grouped / stacked, vertical or horizontal bars with optional error bars (lo/hi).
	 * Props: groups[{label, values[{series,value,lo?,hi?,n?}]}], colors{series:color}, orientation
	 * 'vertical'|'horizontal', stacked, showValues, valueFormat, unit, yLabel, height, yDomain, ariaLabel.
	 * Legend click toggles a series.
	 */
	import { extent, linear, niceTicks, layoutLegend, approxTextWidth } from '$lib/research/scale';
	import { AXIS_COLOR, GRID_COLOR, MONO, SERIES_COLORS } from '$lib/research/palette';
	import SvgLegend from './SvgLegend.svelte';
	import SvgTooltip from './SvgTooltip.svelte';

	type V = { series: string; value: number; lo?: number; hi?: number; n?: number };
	let {
		groups = [], colors = {}, orientation = 'vertical', stacked = false, showValues = true, valueFormat, unit = '',
		yLabel = '', height = 300, yDomain, ariaLabel = 'bar chart'
	}: {
		groups?: { label: string; values: V[] }[]; colors?: Record<string, string>; orientation?: 'vertical' | 'horizontal';
		stacked?: boolean; showValues?: boolean; valueFormat?: (v: number) => string; unit?: string; yLabel?: string;
		height?: number; yDomain?: [number, number]; ariaLabel?: string;
	} = $props();

	let w = $state(640);
	const W = $derived(Math.max(280, w));
	let hidden = $state<Record<string, boolean>>({});
	let hover = $state<{ g: number; s: string } | null>(null);
	const seriesNames = $derived(Array.from(new Set(groups.flatMap((g) => g.values.map((v) => v.series)))));
	const colorOf = (s: string) => colors[s] ?? SERIES_COLORS[seriesNames.indexOf(s) % SERIES_COLORS.length];
	const active = $derived(seriesNames.filter((s) => !hidden[s]));
	const hasData = $derived(groups.some((g) => g.values.some((v) => Number.isFinite(v.value))));
	const legend = $derived(layoutLegend(seriesNames.length > 1 ? seriesNames.map((s) => ({ key: s, label: s, color: colorOf(s), hidden: !!hidden[s] })) : [], W - 24));
	const horiz = $derived(orientation === 'horizontal');
	const labelW = $derived(horiz ? Math.min(180, Math.max(60, ...groups.map((g) => approxTextWidth(g.label, 10) + 8))) : 0);
	const M = $derived({ l: horiz ? labelW + 8 : 54, r: 16, t: 10 + (legend.height ? legend.height + 8 : 0), b: horiz ? 34 : 56 });
	const plotW = $derived(W - M.l - M.r);
	const H = $derived(horiz ? Math.max(height, M.t + M.b + groups.length * (stacked ? 26 : 22 * Math.max(1, active.length) + 10)) : height);
	const plotH = $derived(H - M.t - M.b);
	const vals = (g: { values: V[] }) => g.values.filter((v) => !hidden[v.series]);
	const domain = $derived.by<[number, number]>(() => {
		if (yDomain) return yDomain;
		const all: number[] = [0];
		for (const g of groups) {
			const vs = vals(g);
			if (stacked) all.push(vs.reduce((a, v) => a + (Number.isFinite(v.value) ? v.value : 0), 0));
			else vs.forEach((v) => all.push(v.value, v.hi ?? NaN, v.lo ?? NaN));
		}
		const e = extent(all) ?? [0, 1];
		return [Math.min(0, e[0]), e[1] === e[0] ? e[1] + 1 : e[1] * 1.06];
	});
	const ticks = $derived(niceTicks(domain[0], domain[1], 5));
	const sv = $derived(horiz ? linear(domain[0], domain[1], M.l, M.l + plotW) : linear(domain[0], domain[1], M.t + plotH, M.t));
	const fmt = (v: number) => (valueFormat ? valueFormat(v) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
	const band = $derived((horiz ? plotH : plotW) / Math.max(1, groups.length));

	function bars(g: { values: V[] }, gi: number) {
		const vs = vals(g).filter((v) => Number.isFinite(v.value));
		const inner = band * 0.72;
		const start = gi * band + (band - inner) / 2;
		if (stacked) {
			let acc = 0;
			return vs.map((v) => { const a = acc; acc += v.value; return { v, a, b: acc, off: start, thick: inner }; });
		}
		const thick = inner / Math.max(1, vs.length);
		return vs.map((v, i) => ({ v, a: 0, b: v.value, off: start + i * thick, thick: thick * 0.9 }));
	}
	const tip = $derived.by(() => {
		if (!hover) return null;
		const g = groups[hover.g];
		const v = g?.values.find((x) => x.series === hover!.s);
		if (!g || !v) return null;
		const lines = [{ text: `${v.series}: ${fmt(v.value)}${unit ? ' ' + unit : ''}`, color: colorOf(v.series) }];
		if (Number.isFinite(v.lo) && Number.isFinite(v.hi)) lines.push({ text: `interval: [${fmt(v.lo as number)}, ${fmt(v.hi as number)}]`, color: undefined as unknown as string });
		if (v.n != null) lines.push({ text: `n = ${v.n}`, color: undefined as unknown as string });
		return { title: g.label, lines };
	});
	function tipPos() {
		if (!hover) return { x: 0, y: 0 };
		const g = groups[hover.g];
		const b = bars(g, hover.g).find((o) => o.v.series === hover!.s);
		if (!b) return { x: 0, y: 0 };
		return horiz ? { x: sv(b.b), y: M.t + b.off + b.thick / 2 } : { x: M.l + b.off + b.thick / 2, y: sv(b.b) };
	}
</script>

<div class="host" bind:clientWidth={w}>
	<svg data-chart-root viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={ariaLabel}>
		<title>{ariaLabel}</title>
		{#if !hasData}
			<text x={W / 2} y={H / 2} text-anchor="middle" fill="#71829a" font-size="11" font-family={MONO}>No data supplied</text>
		{:else}
			{#if legend.height}<SvgLegend layout={legend} x={12} y={6} onToggle={(k) => (hidden = { ...hidden, [k]: !hidden[k] })} />{/if}
			{#each ticks as t}
				{#if horiz}
					<line x1={sv(t)} x2={sv(t)} y1={M.t} y2={M.t + plotH} stroke={GRID_COLOR} />
					<text x={sv(t)} y={M.t + plotH + 14} text-anchor="middle" fill={AXIS_COLOR} font-size="10" font-family={MONO}>{fmt(t)}</text>
				{:else}
					<line x1={M.l} x2={M.l + plotW} y1={sv(t)} y2={sv(t)} stroke={GRID_COLOR} />
					<text x={M.l - 6} y={sv(t) + 3} text-anchor="end" fill={AXIS_COLOR} font-size="10" font-family={MONO}>{fmt(t)}</text>
				{/if}
			{/each}
			{#each groups as g, gi}
				{#if horiz}
					<text x={M.l - 6} y={M.t + gi * band + band / 2 + 3} text-anchor="end" fill="#a7b8c9" font-size="10" font-family={MONO}>{g.label.length > 26 ? g.label.slice(0, 25) + '…' : g.label}</text>
				{:else}
					<text x={M.l + gi * band + band / 2} y={M.t + plotH + 14} text-anchor={groups.length > 6 ? 'end' : 'middle'}
						transform={groups.length > 6 ? `rotate(-30 ${M.l + gi * band + band / 2} ${M.t + plotH + 14})` : ''} fill="#a7b8c9" font-size="10" font-family={MONO}>{g.label.length > 18 ? g.label.slice(0, 17) + '…' : g.label}</text>
				{/if}
				{#each bars(g, gi) as b}
					{@const lo = Math.min(b.a, b.b)}
					{@const hi = Math.max(b.a, b.b)}
					<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
					<g role="img" tabindex="0" aria-label={`${g.label}, ${b.v.series}: ${fmt(b.v.value)} ${unit}`}
						onpointerenter={() => (hover = { g: gi, s: b.v.series })} onpointerleave={() => (hover = null)}
						onfocus={() => (hover = { g: gi, s: b.v.series })} onblur={() => (hover = null)} style="outline:none">
						{#if horiz}
							<rect x={sv(lo)} y={M.t + b.off} width={Math.max(1, sv(hi) - sv(lo))} height={b.thick} fill={colorOf(b.v.series)} fill-opacity={hover && hover.g === gi && hover.s === b.v.series ? 1 : 0.85} />
							{#if Number.isFinite(b.v.lo) && Number.isFinite(b.v.hi)}
								<line x1={sv(b.v.lo as number)} x2={sv(b.v.hi as number)} y1={M.t + b.off + b.thick / 2} y2={M.t + b.off + b.thick / 2} stroke="#e5f1f0" />
								<line x1={sv(b.v.lo as number)} x2={sv(b.v.lo as number)} y1={M.t + b.off + b.thick / 2 - 4} y2={M.t + b.off + b.thick / 2 + 4} stroke="#e5f1f0" />
								<line x1={sv(b.v.hi as number)} x2={sv(b.v.hi as number)} y1={M.t + b.off + b.thick / 2 - 4} y2={M.t + b.off + b.thick / 2 + 4} stroke="#e5f1f0" />
							{/if}
							{#if showValues && !stacked}<text x={sv(hi) + 5} y={M.t + b.off + b.thick / 2 + 3} fill="#cbd5e1" font-size="10" font-family={MONO}>{fmt(b.v.value)}</text>{/if}
						{:else}
							<rect x={M.l + b.off} y={sv(hi)} width={b.thick} height={Math.max(1, sv(lo) - sv(hi))} fill={colorOf(b.v.series)} fill-opacity={hover && hover.g === gi && hover.s === b.v.series ? 1 : 0.85} />
							{#if Number.isFinite(b.v.lo) && Number.isFinite(b.v.hi)}
								<line y1={sv(b.v.lo as number)} y2={sv(b.v.hi as number)} x1={M.l + b.off + b.thick / 2} x2={M.l + b.off + b.thick / 2} stroke="#e5f1f0" />
								<line y1={sv(b.v.lo as number)} y2={sv(b.v.lo as number)} x1={M.l + b.off + b.thick / 2 - 4} x2={M.l + b.off + b.thick / 2 + 4} stroke="#e5f1f0" />
								<line y1={sv(b.v.hi as number)} y2={sv(b.v.hi as number)} x1={M.l + b.off + b.thick / 2 - 4} x2={M.l + b.off + b.thick / 2 + 4} stroke="#e5f1f0" />
							{/if}
							{#if showValues && !stacked && b.thick >= 26}<text x={M.l + b.off + b.thick / 2} y={sv(hi) - 5} text-anchor="middle" fill="#cbd5e1" font-size="10" font-family={MONO}>{fmt(b.v.value)}</text>{/if}
						{/if}
					</g>
				{/each}
			{/each}
			<line x1={horiz ? sv(0) : M.l} x2={horiz ? sv(0) : M.l + plotW} y1={horiz ? M.t : sv(0)} y2={horiz ? M.t + plotH : sv(0)} stroke="rgba(148,163,184,.4)" />
			{#if yLabel}<text transform={horiz ? '' : `translate(12,${M.t + plotH / 2}) rotate(-90)`} x={horiz ? M.l + plotW / 2 : 0} y={horiz ? H - 6 : 0} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>{yLabel}{unit ? ` (${unit})` : ''}</text>{/if}
			{#if tip}{@const p = tipPos()}<SvgTooltip x={p.x} y={p.y} lines={tip.lines} title={tip.title} bounds={{ w: W, h: H }} />{/if}
		{/if}
	</svg>
</div>

<style>.host{width:100%;min-width:0}svg{display:block;max-width:100%}</style>
