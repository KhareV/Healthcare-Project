<script module lang="ts">
	export type SupportRun = { label: string; domain: 'VASOPRESSOR' | 'RESPIRATORY'; qualifying?: boolean; start_hour: number; end_hour: number | null };
</script>

<script lang="ts">
	// SupportTimeline — organ-support intervals up to the selected cutoff. Intervals that start
	// after the cutoff are not drawn; an interval still running at the cutoff is clipped there.
	let { runs = [], cutoffHour = 0, admissionToFirstCutoff = 24 }: { runs?: SupportRun[]; cutoffHour?: number; admissionToFirstCutoff?: number } = $props();
	const W = 640, H = 22, left = 150;
	const visible = $derived(runs.filter((r) => r.start_hour < cutoffHour));
	const span = $derived(Math.max(cutoffHour, 1));
	const x = (h: number) => left + (Math.min(h, cutoffHour) / span) * (W - left - 10);
</script>

{#if runs.length === 0}
	<p class="empty">No organ-support interval is recorded for this encounter source before the selected cutoff.</p>
{:else}
	<svg viewBox={`0 0 ${W} ${visible.length * H + 34}`} width="100%" role="img" aria-label="Organ support timeline up to the prediction cutoff">
		{#each visible as r, i}
			<text x={left - 8} y={i * H + 15} text-anchor="end" fill="#8494a9" font-size="9" font-family="JetBrains Mono, monospace">{r.label}</text>
			<rect x={x(r.start_hour)} y={i * H + 4} width={Math.max(3, x(r.end_hour ?? cutoffHour) - x(r.start_hour))} height={H - 8} rx="2"
				fill={r.domain === 'VASOPRESSOR' ? 'rgba(251,113,133,.65)' : 'rgba(56,189,248,.6)'} stroke={r.qualifying === false ? '#94a3b8' : 'none'} stroke-dasharray={r.qualifying === false ? '3 2' : undefined} />
		{/each}
		<line x1={x(cutoffHour)} x2={x(cutoffHour)} y1="0" y2={visible.length * H + 6} stroke="#fbbf24" stroke-width="2" />
		<text x={x(cutoffHour)} y={visible.length * H + 22} text-anchor="end" fill="#fbbf24" font-size="8" font-family="JetBrains Mono, monospace">cutoff · hour {cutoffHour.toFixed(0)} of admission</text>
		<text x={left} y={visible.length * H + 22} fill="#64748b" font-size="8" font-family="JetBrains Mono, monospace">admission · hour 0</text>
	</svg>
	{#if visible.length === 0}<p class="empty">No organ support had started by the selected cutoff.</p>{/if}
{/if}

<style>
	.empty { margin: 4px 0; color: #64748b; font: 10px 'JetBrains Mono', monospace; letter-spacing: .04em; }
	svg { display: block; }
</style>
