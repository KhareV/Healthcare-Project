<script lang="ts">
	/**
	 * HeatmapGrid — matrix heatmap. Props: rows[], cols[], values[row][col] (NaN/null = not observed, drawn
	 * hatched), counts?[row][col] (shown in tooltip), unit, format, domain?, cellHeight, ariaLabel, colorLabel.
	 */
	import { approxTextWidth } from '$lib/research/scale';
	import { MONO, sequentialColor } from '$lib/research/palette';
	import SvgTooltip from './SvgTooltip.svelte';

	let {
		rows = [], cols = [], values = [], counts, unit = '', format, domain, cellHeight = 28, colorLabel = '', ariaLabel = 'heatmap'
	}: {
		rows?: string[]; cols?: string[]; values?: (number | null)[][]; counts?: (number | null)[][]; unit?: string;
		format?: (v: number) => string; domain?: [number, number]; cellHeight?: number; colorLabel?: string; ariaLabel?: string;
	} = $props();

	let w = $state(640);
	const W = $derived(Math.max(280, w));
	let hov = $state<{ r: number; c: number } | null>(null);
	const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
	const flat = $derived(values.flat().filter(finite));
	const hasData = $derived(flat.length > 0 && rows.length > 0 && cols.length > 0);
	const dom = $derived<[number, number]>(domain ?? (flat.length ? [Math.min(...flat), Math.max(...flat)] : [0, 1]));
	const rowLabelW = $derived(Math.min(170, Math.max(40, ...rows.map((r) => approxTextWidth(r, 10) + 10))));
	const rotate = $derived(cols.length > 8 || cols.some((c) => c.length > 6));
	const M = $derived({ l: rowLabelW + 8, r: 14, t: rotate ? 70 : 28, b: 54 });
	const plotW = $derived(W - M.l - M.r);
	const cw = $derived(plotW / Math.max(1, cols.length));
	const H = $derived(M.t + rows.length * cellHeight + M.b);
	const fmt = (v: number) => (format ? format(v) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
	const colorOf = (v: number) => sequentialColor((v - dom[0]) / (dom[1] - dom[0] || 1));
	const tip = $derived.by(() => {
		if (!hov) return null;
		const v = values[hov.r]?.[hov.c];
		const n = counts?.[hov.r]?.[hov.c];
		return {
			x: M.l + (hov.c + 0.5) * cw, y: M.t + (hov.r + 0.5) * cellHeight, title: `${rows[hov.r]} × ${cols[hov.c]}`,
			lines: [{ text: finite(v) ? `${fmt(v)}${unit ? ' ' + unit : ''}` : 'not observed / not available', color: undefined as unknown as string },
				...(n != null ? [{ text: `n = ${n}`, color: undefined as unknown as string }] : [])]
		};
	});
</script>

<div class="host" bind:clientWidth={w}>
	<svg data-chart-root viewBox={`0 0 ${W} ${hasData ? H : 140}`} width={W} height={hasData ? H : 140} role="img" aria-label={ariaLabel}>
		<title>{ariaLabel}</title>
		{#if !hasData}
			<text x={W / 2} y={70} text-anchor="middle" fill="#71829a" font-size="11" font-family={MONO}>No data supplied</text>
		{:else}
			<defs>
				<pattern id="rk-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#0b1323" /><line x1="0" y1="0" x2="0" y2="6" stroke="#334155" stroke-width="2" /></pattern>
				<linearGradient id="rk-grad" x1="0" x2="1"><stop offset="0" stop-color={sequentialColor(0)} /><stop offset="1" stop-color={sequentialColor(1)} /></linearGradient>
			</defs>
			{#each cols as c, ci}
				<text x={M.l + (ci + 0.5) * cw} y={M.t - 8} text-anchor={rotate ? 'start' : 'middle'} transform={rotate ? `rotate(-40 ${M.l + (ci + 0.5) * cw} ${M.t - 8})` : ''} fill="#a7b8c9" font-size="10" font-family={MONO}>{c}</text>
			{/each}
			{#each rows as r, ri}
				<text x={M.l - 6} y={M.t + ri * cellHeight + cellHeight / 2 + 3} text-anchor="end" fill="#a7b8c9" font-size="10" font-family={MONO}>{r}</text>
				{#each cols as _c, ci}
					{@const v = values[ri]?.[ci]}
					<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
					<g role="img" tabindex="0" aria-label={`${r}, ${cols[ci]}: ${finite(v) ? fmt(v) : 'not available'}`} onpointerenter={() => (hov = { r: ri, c: ci })}
						onpointerleave={() => (hov = null)} onfocus={() => (hov = { r: ri, c: ci })} onblur={() => (hov = null)} style="outline:none">
						<rect x={M.l + ci * cw + 0.5} y={M.t + ri * cellHeight + 0.5} width={Math.max(1, cw - 1)} height={cellHeight - 1} fill={finite(v) ? colorOf(v) : 'url(#rk-hatch)'}
							stroke={hov && hov.r === ri && hov.c === ci ? '#e5f1f0' : 'none'} />
						{#if finite(v) && cw >= 38}<text x={M.l + (ci + 0.5) * cw} y={M.t + ri * cellHeight + cellHeight / 2 + 3} text-anchor="middle" fill={((v - dom[0]) / (dom[1] - dom[0] || 1)) > 0.55 ? '#031210' : '#cbd5e1'} font-size="10" font-family={MONO}>{fmt(v)}</text>{/if}
					</g>
				{/each}
			{/each}
			<g transform={`translate(${M.l},${M.t + rows.length * cellHeight + 18})`}>
				<rect width={Math.min(220, plotW)} height="10" fill="url(#rk-grad)" />
				<text y="24" fill="#71829a" font-size="10" font-family={MONO}>{fmt(dom[0])}</text>
				<text x={Math.min(220, plotW)} y="24" text-anchor="end" fill="#71829a" font-size="10" font-family={MONO}>{fmt(dom[1])}{unit ? ' ' + unit : ''}</text>
				<rect x={Math.min(220, plotW) + 16} y="0" width="10" height="10" fill="url(#rk-hatch)" /><text x={Math.min(220, plotW) + 30} y="9" fill="#71829a" font-size="10" font-family={MONO}>not observed</text>
				{#if colorLabel}<text x="0" y="40" fill="#8fa2b8" font-size="10" font-family={MONO}>{colorLabel}</text>{/if}
			</g>
			{#if tip}<SvgTooltip x={tip.x} y={tip.y} lines={tip.lines} title={tip.title} bounds={{ w: W, h: H }} />{/if}
		{/if}
	</svg>
</div>
<style>.host{width:100%;min-width:0}svg{display:block;max-width:100%}</style>
