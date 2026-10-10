<svelte:options namespace="svg" />
<script lang="ts">
	// Clickable SVG legend (toggles series). Rendered inside chart <svg> so it is included in exports.
	import type { LegendLayout } from '$lib/research/scale';
	import { MONO } from '$lib/research/palette';
	let { layout, x = 0, y = 0, onToggle }: { layout: LegendLayout; x?: number; y?: number; onToggle?: (key: string) => void } = $props();
</script>

<g transform={`translate(${x},${y})`}>
	{#each layout.items as it (it.key)}
		<g
			transform={`translate(${it.x},${it.y})`}
			role="button"
			tabindex="0"
			aria-pressed={!it.hidden}
			aria-label={`Toggle series ${it.label}`}
			style="cursor:pointer;outline:none"
			opacity={it.hidden ? 0.35 : 1}
			onclick={() => onToggle?.(it.key)}
			onkeydown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					onToggle?.(it.key);
				}
			}}
		>
			<line x1="0" y1="8" x2="20" y2="8" stroke={it.color} stroke-width="3" stroke-dasharray={it.dash ?? ''} stroke-linecap="round" />
			<text x="26" y="12" fill="#8fa2b8" font-size="10" font-family={MONO}>{it.label}</text>
		</g>
	{/each}
</g>
