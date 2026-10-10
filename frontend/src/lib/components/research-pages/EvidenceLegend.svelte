<script lang="ts">
	import { STATUS_STYLE } from '$lib/research/palette';
	import type { EvidenceStatus } from '$lib/research/types';
	const DETAIL: Record<EvidenceStatus, string> = {
		FINAL: 'One-time sealed V3 test, frozen before access.',
		DEVELOPMENT: 'Cross-fitted / held-out development evidence; informs but is not the final result.',
		HISTORICAL: 'Earlier releases (V1/V2) kept for context only.',
		DEFECT: 'Analysis of defect D1 (current-SOFA missingness leakage in V2).',
		PRIVILEGED: 'Uses hidden simulator state. Research-only; never a serving input or deployable performance.'
	};
	const order: EvidenceStatus[] = ['FINAL', 'DEVELOPMENT', 'HISTORICAL', 'DEFECT', 'PRIVILEGED'];
</script>

<ul class="legend" aria-label="Evidence status legend">
	{#each order as key}
		<li style={`--c:${STATUS_STYLE[key].color}`}><b><i aria-hidden="true">{STATUS_STYLE[key].glyph}</i>{STATUS_STYLE[key].label}</b><span>{DETAIL[key]}</span></li>
	{/each}
</ul>

<style>
	.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr));gap:10px;margin:0 0 22px;padding:0;list-style:none}
	li{padding:10px 12px;border:1px solid color-mix(in srgb,var(--c) 38%,transparent);border-left:3px solid var(--c);background:rgba(7,13,25,.8)}
	b{display:flex;gap:7px;align-items:center;color:var(--c);font:500 8px 'JetBrains Mono',monospace;letter-spacing:.09em}
	i{font-style:normal}
	span{display:block;margin-top:6px;color:#71829a;font:11px/1.45 Inter,sans-serif}
</style>
