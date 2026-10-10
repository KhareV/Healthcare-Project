<svelte:options namespace="svg" />
<script lang="ts">
	// Pointer/focus tooltip drawn inside the chart SVG; flips to stay within bounds.
	import { approxTextWidth } from '$lib/research/scale';
	import { MONO } from '$lib/research/palette';
	let {
		x, y, lines, bounds, title = ''
	}: { x: number; y: number; lines: { text: string; color?: string }[]; bounds: { w: number; h: number }; title?: string } = $props();
	const w = $derived(Math.max(approxTextWidth(title, 10), ...lines.map((l) => approxTextWidth(l.text, 10) + (l.color ? 14 : 0))) + 16);
	const h = $derived((title ? 18 : 0) + lines.length * 15 + 10);
	const tx = $derived(x + 12 + w > bounds.w ? Math.max(2, x - 12 - w) : x + 12);
	const ty = $derived(Math.max(2, Math.min(bounds.h - h - 2, y - h / 2)));
</script>

<g transform={`translate(${tx},${ty})`} pointer-events="none" data-export-hide>
	<rect width={w} height={h} rx="3" fill="#0a1220" stroke="rgba(148,163,184,.35)" />
	{#if title}<text x="8" y="14" fill="#e5f1f0" font-size="10" font-weight="600" font-family={MONO}>{title}</text>{/if}
	{#each lines as l, i}
		{@const yy = (title ? 18 : 0) + 14 + i * 15}
		{#if l.color}<rect x="8" y={yy - 8} width="8" height="8" fill={l.color} />{/if}
		<text x={l.color ? 20 : 8} y={yy} fill="#cbd5e1" font-size="10" font-family={MONO}>{l.text}</text>
	{/each}
</g>
