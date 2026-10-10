<script lang="ts">
	// EightBinWindow — the exact 48h lookback the forecast saw: eight 6h bins x channels.
	// Observation/padding masks come from the prediction's temporal_window; values and recency
	// are derived from /history events (all at or before the cutoff) by the same bin edges.
	import type { HistoryEvent } from '$lib/stores/patient-context.svelte';
	let { channelNames = [], observationMask = [], paddingMask = [], events = [], cutoff = '' }:
		{ channelNames?: string[]; observationMask?: boolean[][]; paddingMask?: boolean[]; events?: HistoryEvent[]; cutoff?: string } = $props();

	const cutoffMs = $derived(cutoff ? Date.parse(cutoff) : NaN);
	const concept = (name: string) => name.replace(/__latest$|__sum$/g, '');
	const label = (name: string) => concept(name).replace(/_/g, ' ');
	const cellW = 58, cellH = 24, labelW = 150, tslW = 66, futureW = 86, top = 30;
	const width = labelW + 8 * cellW + futureW + tslW;
	const height = $derived(top + channelNames.length * cellH + 12);

	// latest value per (concept, bin) and last observation time per concept (inside the lookback)
	const lookup = $derived.by(() => {
		const latest = new Map<string, { t: number; v: number; unit: string }>();
		const last = new Map<string, number>();
		if (!Number.isFinite(cutoffMs)) return { latest, last };
		const start = cutoffMs - 48 * 3_600_000;
		for (const e of events) {
			const t = Date.parse(e.event_time);
			if (!(t > start && t <= cutoffMs)) continue; // never anything after the cutoff
			const bin = Math.min(7, Math.floor((t - start) / (6 * 3_600_000)));
			const key = `${e.canonical_concept}|${bin}`;
			const prev = latest.get(key);
			if (!prev || t >= prev.t) latest.set(key, { t, v: e.value_numeric, unit: e.unit });
			last.set(e.canonical_concept, Math.max(last.get(e.canonical_concept) ?? 0, t));
		}
		return { latest, last };
	});
	const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
	function cell(bin: number, ci: number) {
		if (paddingMask[bin]) return { kind: 'pad', text: '' };
		const observed = observationMask[bin]?.[ci];
		const got = lookup.latest.get(`${concept(channelNames[ci])}|${bin}`);
		if (observed) return { kind: 'obs', text: got ? fmt(got.v) : '✓' };
		return { kind: 'miss', text: '' };
	}
	function recency(ci: number): string {
		const t = lookup.last.get(concept(channelNames[ci]));
		if (!t || !Number.isFinite(cutoffMs)) return '—';
		return `${((cutoffMs - t) / 3_600_000).toFixed(1)}h`;
	}
</script>

<div class="wrap">
	<svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label="Eight six-hour bins of observed values before the prediction cutoff; nothing after the cutoff is visible">
		{#each Array(8) as _, b}
			<text x={labelW + b * cellW + cellW / 2} y={18} text-anchor="middle" fill="#64748b" font-size="8" font-family="JetBrains Mono, monospace">{`t−${48 - 6 * b}h`}</text>
		{/each}
		<text x={labelW + 8 * cellW + futureW / 2} y={18} text-anchor="middle" fill="#fb7185" font-size="8" font-family="JetBrains Mono, monospace">FUTURE</text>
		<text x={width - tslW / 2} y={18} text-anchor="middle" fill="#64748b" font-size="8" font-family="JetBrains Mono, monospace">LAST OBS</text>
		{#each channelNames as name, ci}
			<text x={labelW - 8} y={top + ci * cellH + cellH / 2 + 3} text-anchor="end" fill="#8494a9" font-size="9" font-family="JetBrains Mono, monospace">{label(name)}</text>
			{#each Array(8) as _, b}
				{@const c = cell(b, ci)}
				<rect x={labelW + b * cellW + 1} y={top + ci * cellH + 1} width={cellW - 2} height={cellH - 2} rx="2"
					fill={c.kind === 'obs' ? 'rgba(43,184,176,.55)' : c.kind === 'pad' ? '#1c2536' : 'rgba(251,113,133,.14)'}
					stroke={c.kind === 'miss' ? 'rgba(251,113,133,.35)' : 'none'} stroke-dasharray={c.kind === 'miss' ? '2 2' : undefined} />
				{#if c.text}<text x={labelW + b * cellW + cellW / 2} y={top + ci * cellH + cellH / 2 + 3} text-anchor="middle" fill="#eef7f6" font-size="9" font-family="JetBrains Mono, monospace">{c.text}</text>{/if}
			{/each}
			<rect x={labelW + 8 * cellW + 4} y={top + ci * cellH + 1} width={futureW - 8} height={cellH - 2} fill="url(#hatch)" opacity=".5" />
			<text x={width - tslW / 2} y={top + ci * cellH + cellH / 2 + 3} text-anchor="middle" fill="#94a3b8" font-size="9" font-family="JetBrains Mono, monospace">{recency(ci)}</text>
		{/each}
		<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#fb7185" stroke-width="1" opacity=".5" /></pattern></defs>
		<line x1={labelW + 8 * cellW} y1={top - 4} x2={labelW + 8 * cellW} y2={height - 8} stroke="#fbbf24" stroke-width="2" />
		<text x={labelW + 8 * cellW} y={height - 1} text-anchor="middle" fill="#fbbf24" font-size="8" font-family="JetBrains Mono, monospace">PREDICTION CUTOFF t</text>
	</svg>
	<div class="legend">
		<span><i style="background:rgba(43,184,176,.55)"></i> observed (latest value in bin)</span>
		<span><i style="background:rgba(251,113,133,.14);border:1px dashed rgba(251,113,133,.5)"></i> not observed in bin</span>
		<span><i style="background:#1c2536"></i> structural padding (before ICU admission)</span>
		<span class="future"><i style="background:repeating-linear-gradient(45deg,#fb7185 0 1px,transparent 1px 5px)"></i> no future data is visible to this forecast</span>
	</div>
</div>

<style>
	.wrap { overflow-x: auto; }
	svg { display: block; min-width: 640px; }
	.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 10px; color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .06em; }
	.legend i { display: inline-block; width: 10px; height: 10px; margin-right: 6px; border-radius: 2px; vertical-align: -1px; }
	.future { color: #fda4af; }
</style>
