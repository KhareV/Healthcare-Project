<script lang="ts">
	// MultiLine — annotated multi-series line chart (pure SVG, no deps)
	let {
		series = [],
		xLabels = [],
		height = 180,
		yMin = 0,
		yMax = 1,
		yLabel = ''
	}: {
		series?: { name: string; color: string; values: number[] }[];
		xLabels?: string[];
		height?: number;
		yMin?: number;
		yMax?: number;
		yLabel?: string;
	} = $props();

	const W = 600, H = height;
	const PAD = { top: 14, right: 20, bottom: 28, left: 36 };
	const pw = W - PAD.left - PAD.right;
	const ph = H - PAD.top - PAD.bottom;

	function toX(i: number, len: number) {
		return PAD.left + (i / Math.max(1, len - 1)) * pw;
	}
	function toY(v: number) {
		return PAD.top + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;
	}
	function pts(vals: number[]) {
		return vals.map((v, i) => `${toX(i, vals.length).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
	}
	// Split a series on NaN/undefined gaps so a partially-known series (e.g.
	// "forecast" values that only start partway through the x-axis) renders
	// as disjoint segments instead of corrupting the whole polyline.
	function segments(vals: number[]): { start: number; values: number[] }[] {
		const out: { start: number; values: number[] }[] = [];
		let current: number[] = [];
		let start = 0;
		vals.forEach((v, i) => {
			if (v === null || v === undefined || Number.isNaN(v)) {
				if (current.length) out.push({ start, values: current });
				current = [];
				start = i + 1;
			} else {
				current.push(v);
			}
		});
		if (current.length) out.push({ start, values: current });
		return out;
	}
	function segPts(seg: { start: number; values: number[] }, total: number) {
		return seg.values.map((v, i) => `${toX(seg.start + i, total).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
	}

	const yTicks = $derived(() => {
		const steps = 4;
		return Array.from({length: steps + 1}, (_, i) => yMin + (i / steps) * (yMax - yMin));
	});
</script>

<svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="line chart">
	<!-- Grid -->
	{#each yTicks() as tick}
		<line
			x1={PAD.left} y1={toY(tick)}
			x2={W - PAD.right} y2={toY(tick)}
			stroke="rgba(148,163,184,.1)" stroke-width="1"
		/>
		<text x={PAD.left - 4} y={toY(tick) + 4} text-anchor="end" fill="#64748b" font-size="9" font-family="JetBrains Mono, monospace">
			{tick.toFixed(2)}
		</text>
	{/each}

	<!-- X labels -->
	{#each xLabels as lbl, i}
		<text
			x={toX(i, xLabels.length)} y={H - 6}
			text-anchor="middle" fill="#64748b" font-size="9" font-family="JetBrains Mono, monospace"
		>{lbl}</text>
	{/each}

	<!-- Lines -->
	{#each series as s}
		{#each segments(s.values) as seg, segIndex}
			{#if seg.values.length > 1}
				<polyline
					points={segPts(seg, s.values.length)}
					fill="none" stroke={s.color} stroke-width="2"
					style={`filter:drop-shadow(0 0 4px ${s.color}60);vector-effect:non-scaling-stroke`}
				/>
			{/if}
			{#if seg.start + seg.values.length === s.values.length}
				<!-- End dot on the last real segment -->
				<circle
					cx={toX(s.values.length-1, s.values.length)}
					cy={toY(seg.values[seg.values.length-1])}
					r="3" fill={s.color}
					style={`filter:drop-shadow(0 0 5px ${s.color})`}
				/>
			{/if}
		{/each}
	{/each}
</svg>

<!-- Legend -->
{#if series.length > 1}
	<div class="legend">
		{#each series as s}
			<span class="leg-item">
				<i style={`background:${s.color};box-shadow:0 0 5px ${s.color}`}></i>
				{s.name}
			</span>
		{/each}
	</div>
{/if}

<style>
	svg { display: block; }
	.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; }
	.leg-item { display: flex; align-items: center; gap: 5px; color: #71829a; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; }
	.leg-item i { display: inline-block; width: 20px; height: 2px; border-radius: 1px; }
</style>
