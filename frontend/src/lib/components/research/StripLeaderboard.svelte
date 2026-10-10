<script lang="ts">
	/**
	 * StripLeaderboard — every trial as a dot per group (family / feature variant) with the best marked.
	 * Props: groups[{label, trials[{value, id?, note?, failed?}]}], metric label, unit, lowerIsBetter,
	 * color, format, ariaLabel, domain?. Failed/diverged trials are drawn at the right edge as ✕, never hidden.
	 */
	import { extent, jitter, linear, niceTicks, pad, approxTextWidth } from '$lib/research/scale';
	import { AXIS_COLOR, GRID_COLOR, MONO } from '$lib/research/palette';
	import SvgTooltip from './SvgTooltip.svelte';

	type T = { value: number; id?: string; note?: string; failed?: boolean };
	let {
		groups = [], metric = 'metric', unit = '', lowerIsBetter = true, color = '#2bb8b0', format, domain, rowHeight = 40, ariaLabel = 'trial leaderboard'
	}: { groups?: { label: string; trials: T[] }[]; metric?: string; unit?: string; lowerIsBetter?: boolean; color?: string; format?: (v: number) => string;
		domain?: [number, number]; rowHeight?: number; ariaLabel?: string } = $props();

	let w = $state(640);
	const W = $derived(Math.max(280, w));
	let hov = $state<{ g: number; t: number } | null>(null);
	const finite = (t: T) => !t.failed && Number.isFinite(t.value);
	const all = $derived(groups.flatMap((g) => g.trials.filter(finite).map((t) => t.value)));
	const hasData = $derived(all.length > 0);
	const labelW = $derived(Math.min(W * 0.35, Math.max(70, ...groups.map((g) => approxTextWidth(g.label, 10) + 10))));
	const M = $derived({ l: labelW + 8, r: 28, t: 10, b: 48 });
	const plotW = $derived(W - M.l - M.r);
	const H = $derived(M.t + groups.length * rowHeight + M.b);
	const dom = $derived<[number, number]>(domain ?? pad(extent(all) ?? [0, 1], 0.06));
	const sx = $derived(linear(dom[0], dom[1], M.l, M.l + plotW));
	const ticks = $derived(niceTicks(dom[0], dom[1], Math.max(3, Math.floor(plotW / 90))));
	const fmt = (v: number) => (format ? format(v) : v.toFixed(3));
	const bestOf = (g: { trials: T[] }) => {
		const v = g.trials.filter(finite).map((t) => t.value);
		return v.length ? (lowerIsBetter ? Math.min(...v) : Math.max(...v)) : NaN;
	};
	const tip = $derived.by(() => {
		if (!hov) return null;
		const t = groups[hov.g]?.trials[hov.t];
		if (!t) return null;
		return { x: finite(t) ? sx(t.value) : M.l + plotW, y: M.t + hov.g * rowHeight + rowHeight / 2, title: groups[hov.g].label,
			lines: [{ text: t.failed ? 'trial failed / diverged' : `${metric}: ${fmt(t.value)}${unit ? ' ' + unit : ''}`, color: undefined as unknown as string },
				...(t.id ? [{ text: t.id.length > 40 ? t.id.slice(0, 39) + '…' : t.id, color: undefined as unknown as string }] : []),
				...(t.note ? [{ text: t.note, color: undefined as unknown as string }] : [])] };
	});
</script>

<div class="host" bind:clientWidth={w}>
	<svg data-chart-root viewBox={`0 0 ${W} ${hasData ? H : 140}`} width={W} height={hasData ? H : 140} role="img" aria-label={ariaLabel}>
		<title>{ariaLabel}</title>
		{#if !hasData}
			<text x={W / 2} y={70} text-anchor="middle" fill="#71829a" font-size="11" font-family={MONO}>No data supplied</text>
		{:else}
			{#each ticks as t}
				<line x1={sx(t)} x2={sx(t)} y1={M.t} y2={M.t + groups.length * rowHeight} stroke={GRID_COLOR} />
				<text x={sx(t)} y={M.t + groups.length * rowHeight + 14} text-anchor="middle" fill={AXIS_COLOR} font-size="10" font-family={MONO}>{fmt(t)}</text>
			{/each}
			{#each groups as g, gi}
				{@const cy = M.t + gi * rowHeight + rowHeight / 2}
				{@const best = bestOf(g)}
				<text x={M.l - 8} y={cy + 3} text-anchor="end" fill="#a7b8c9" font-size="10" font-family={MONO}>{g.label.length > 24 ? g.label.slice(0, 23) + '…' : g.label}</text>
				<line x1={M.l} x2={M.l + plotW} y1={cy} y2={cy} stroke="rgba(148,163,184,.08)" />
				{#each g.trials as t, ti}
					<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
					<g role="img" tabindex="0" aria-label={`${g.label} trial ${ti + 1}: ${t.failed ? 'failed' : fmt(t.value)}`} onpointerenter={() => (hov = { g: gi, t: ti })} onpointerleave={() => (hov = null)}
						onfocus={() => (hov = { g: gi, t: ti })} onblur={() => (hov = null)} style="outline:none">
						{#if t.failed || !Number.isFinite(t.value)}
							<text x={M.l + plotW + 10} y={cy + jitter(gi * 997 + ti) * (rowHeight / 2 - 6) + 3} text-anchor="middle" fill="#fb7185" font-size="11" font-family={MONO}>✕</text>
						{:else}
							<circle cx={sx(t.value)} cy={cy + jitter(gi * 997 + ti) * (rowHeight / 2 - 8)} r={t.value === best ? 5 : 3.2} fill={t.value === best ? '#fbbf24' : color} fill-opacity={t.value === best ? 1 : 0.6}
								stroke={t.value === best ? '#fff7d6' : 'none'} />
						{/if}
					</g>
				{/each}
			{/each}
			<text x={M.l + plotW / 2} y={H - 22} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>{metric}{unit ? ` (${unit})` : ''} · {lowerIsBetter ? 'lower is better' : 'higher is better'}</text>
			<circle cx={M.l + 4} cy={H - 8} r="4" fill="#fbbf24" /><text x={M.l + 14} y={H - 4} fill="#71829a" font-size="10" font-family={MONO}>best in group</text>
			<text x={M.l + 120} y={H - 4} fill="#fb7185" font-size="10" font-family={MONO}>✕ failed / diverged (not plotted on axis)</text>
			{#if tip}<SvgTooltip x={tip.x} y={tip.y} lines={tip.lines} title={tip.title} bounds={{ w: W, h: H }} />{/if}
		{/if}
	</svg>
</div>
<style>.host{width:100%;min-width:0}svg{display:block;max-width:100%}</style>
