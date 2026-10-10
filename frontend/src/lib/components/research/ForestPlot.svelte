<script lang="ts">
	/**
	 * ForestPlot — point estimate + interval per row against a zero line (paired differences).
	 * Props: rows[{label, estimate, lo, hi, group?, note?, color?}], xLabel, xUnit, favourLeft, favourRight,
	 * format, zero (default 0), ariaLabel, rowHeight.
	 */
	import { extent, linear, niceTicks, pad, approxTextWidth } from '$lib/research/scale';
	import { AXIS_COLOR, GRID_COLOR, MONO } from '$lib/research/palette';
	import SvgTooltip from './SvgTooltip.svelte';

	type R = { label: string; estimate: number; lo: number; hi: number; group?: string; note?: string; color?: string };
	let {
		rows = [], xLabel = 'difference', xUnit = '', favourLeft = '', favourRight = '', format, zero = 0, rowHeight = 28, ariaLabel = 'forest plot'
	}: {
		rows?: R[]; xLabel?: string; xUnit?: string; favourLeft?: string; favourRight?: string; format?: (v: number) => string;
		zero?: number; rowHeight?: number; ariaLabel?: string;
	} = $props();

	let w = $state(640);
	const W = $derived(Math.max(280, w));
	let hov = $state<number | null>(null);
	const ok = $derived(rows.filter((r) => Number.isFinite(r.estimate)));
	const labelW = $derived(Math.min(W * 0.4, Math.max(80, ...ok.map((r) => approxTextWidth(r.label, 10) + 12))));
	const M = $derived({ l: labelW + 8, r: 16, t: 12, b: 58 });
	const plotW = $derived(W - M.l - M.r);
	const H = $derived(M.t + ok.length * rowHeight + M.b);
	const dom = $derived(pad(extent([zero, ...ok.flatMap((r) => [r.estimate, r.lo, r.hi])]) ?? [-1, 1], 0.1));
	const sx = $derived(linear(dom[0], dom[1], M.l, M.l + plotW));
	const ticks = $derived(niceTicks(dom[0], dom[1], Math.max(3, Math.floor(plotW / 90))));
	const fmt = (v: number) => (format ? format(v) : v.toFixed(3));
	const tip = $derived.by(() => {
		if (hov == null) return null;
		const r = ok[hov];
		if (!r) return null;
		return { x: sx(r.estimate), y: M.t + hov * rowHeight + rowHeight / 2, title: r.label,
			lines: [{ text: `estimate ${fmt(r.estimate)}${xUnit ? ' ' + xUnit : ''}`, color: undefined as unknown as string },
				{ text: `interval [${fmt(r.lo)}, ${fmt(r.hi)}]`, color: undefined as unknown as string },
				...(r.note ? [{ text: r.note, color: undefined as unknown as string }] : [])] };
	});
</script>

<div class="host" bind:clientWidth={w}>
	<svg data-chart-root viewBox={`0 0 ${W} ${ok.length ? H : 140}`} width={W} height={ok.length ? H : 140} role="img" aria-label={ariaLabel}>
		<title>{ariaLabel}</title>
		{#if !ok.length}
			<text x={W / 2} y={70} text-anchor="middle" fill="#71829a" font-size="11" font-family={MONO}>No data supplied</text>
		{:else}
			{#each ticks as t}
				<line x1={sx(t)} x2={sx(t)} y1={M.t} y2={M.t + ok.length * rowHeight} stroke={GRID_COLOR} />
				<text x={sx(t)} y={M.t + ok.length * rowHeight + 14} text-anchor="middle" fill={AXIS_COLOR} font-size="10" font-family={MONO}>{fmt(t)}</text>
			{/each}
			<line x1={sx(zero)} x2={sx(zero)} y1={M.t} y2={M.t + ok.length * rowHeight} stroke="#94a3b8" stroke-dasharray="4 3" />
			{#each ok as r, i}
				{@const cy = M.t + i * rowHeight + rowHeight / 2}
				{@const c = r.color ?? (r.lo > zero || r.hi < zero ? '#2bb8b0' : '#94a3b8')}
				<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
				<g role="img" tabindex="0" aria-label={`${r.label}: ${fmt(r.estimate)}, interval ${fmt(r.lo)} to ${fmt(r.hi)}`} onpointerenter={() => (hov = i)} onpointerleave={() => (hov = null)}
					onfocus={() => (hov = i)} onblur={() => (hov = null)} style="outline:none">
					<rect x={0} y={M.t + i * rowHeight} width={W} height={rowHeight} fill={hov === i ? 'rgba(148,163,184,.07)' : 'transparent'} />
					<text x={M.l - 8} y={cy + 3} text-anchor="end" fill="#a7b8c9" font-size="10" font-family={MONO}>{r.label.length > 44 ? r.label.slice(0, 43) + '…' : r.label}</text>
					{#if Number.isFinite(r.lo) && Number.isFinite(r.hi)}
						<line x1={sx(r.lo)} x2={sx(r.hi)} y1={cy} y2={cy} stroke={c} stroke-width="2" />
						<line x1={sx(r.lo)} x2={sx(r.lo)} y1={cy - 5} y2={cy + 5} stroke={c} />
						<line x1={sx(r.hi)} x2={sx(r.hi)} y1={cy - 5} y2={cy + 5} stroke={c} />
					{/if}
					<rect x={sx(r.estimate) - 4} y={cy - 4} width="8" height="8" fill={c} />
				</g>
			{/each}
			<text x={M.l} y={H - 28} fill="#53647b" font-size="10" font-family={MONO}>{favourLeft ? '◀ ' + favourLeft : ''}</text>
			<text x={M.l + plotW} y={H - 28} text-anchor="end" fill="#53647b" font-size="10" font-family={MONO}>{favourRight ? favourRight + ' ▶' : ''}</text>
			<text x={M.l + plotW / 2} y={H - 8} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>{xLabel}{xUnit ? ` (${xUnit})` : ''}</text>
			{#if tip}<SvgTooltip x={tip.x} y={tip.y} lines={tip.lines} title={tip.title} bounds={{ w: W, h: H }} />{/if}
		{/if}
	</svg>
</div>
<style>.host{width:100%;min-width:0}svg{display:block;max-width:100%}</style>
