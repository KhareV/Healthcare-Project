<script lang="ts">
	// TemporalHeatmap — 8-bin x N-channel observation-availability grid for
	// the frozen 48h/6h-bin lookback window. Pure SVG, no deps.
	let { channelNames = [], observationMask = [], paddingMask = [] }:
		{ channelNames?: string[]; observationMask?: boolean[][]; paddingMask?: boolean[] } = $props();

	const nBins = 8;
	const cell = 26;
	const labelWidth = 168;
	const height = $derived(channelNames.length * cell + 20);
	const width = labelWidth + nBins * cell;

	function color(binIndex: number, channelIndex: number): string {
		if (paddingMask[binIndex]) return '#1c2536';
		return observationMask[binIndex]?.[channelIndex] ? '#2bb8b0' : '#fb7185';
	}
	function opacity(binIndex: number, channelIndex: number): number {
		if (paddingMask[binIndex]) return 0.35;
		return observationMask[binIndex]?.[channelIndex] ? 0.85 : 0.28;
	}
</script>

<div class="heatmap-wrap">
	<svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img" aria-label="Temporal lookback observation heatmap">
		{#each channelNames as name, ci}
			<text x={labelWidth - 8} y={ci * cell + cell / 2 + 4} text-anchor="end" fill="#8494a9" font-size="9" font-family="JetBrains Mono, monospace">{name.replace(/__latest|__sum/g, '').replace(/_/g, ' ')}</text>
			{#each Array(nBins) as _, bi}
				<rect x={labelWidth + bi * cell + 1} y={ci * cell + 1} width={cell - 2} height={cell - 2} rx="2" fill={color(bi, ci)} opacity={opacity(bi, ci)} />
			{/each}
		{/each}
		{#each Array(nBins) as _, bi}
			<text x={labelWidth + bi * cell + cell / 2} y={height - 4} text-anchor="middle" fill="#64748b" font-size="8" font-family="JetBrains Mono, monospace">{`t-${48 - 6 * bi}`}</text>
		{/each}
	</svg>
	<div class="legend">
		<span><i style="background:#2bb8b0"></i> observed</span>
		<span><i style="background:#fb7185"></i> not observed</span>
		<span><i style="background:#1c2536"></i> padding (pre-admission)</span>
	</div>
</div>

<style>
	.heatmap-wrap { overflow-x: auto; }
	svg { display: block; min-width: 460px; }
	.legend { display: flex; gap: 18px; margin-top: 10px; color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.legend i { display: inline-block; width: 8px; height: 8px; margin-right: 6px; border-radius: 2px; vertical-align: -1px; }
</style>
