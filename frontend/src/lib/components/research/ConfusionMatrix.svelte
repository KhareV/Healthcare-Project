<script lang="ts">
	/**
	 * ConfusionMatrix — 2×2 counts with share-of-total and row-normalised rates.
	 * Props: tp, fp, tn, fn (may be weighted/decimal), positiveLabel, negativeLabel, threshold (text),
	 * digits, ariaLabel. Rows = observed, columns = predicted.
	 */
	import { MONO } from '$lib/research/palette';
	let {
		tp = NaN, fp = NaN, tn = NaN, fn = NaN, positiveLabel = 'Initiation', negativeLabel = 'No initiation', threshold = '', digits = 0, ariaLabel = 'confusion matrix'
	}: { tp?: number; fp?: number; tn?: number; fn?: number; positiveLabel?: string; negativeLabel?: string; threshold?: string; digits?: number; ariaLabel?: string } = $props();

	let w = $state(480);
	const W = $derived(Math.max(280, Math.min(w, 560)));
	const ok = $derived([tp, fp, tn, fn].every((v) => Number.isFinite(v)));
	const total = $derived(tp + fp + tn + fn);
	const L = 96, T = 64, cellH = 76;
	const cw = $derived((W - L - 12) / 2);
	const H = T + cellH * 2 + 56;
	const cells = $derived([
		{ r: 0, c: 0, v: tp, name: 'True positive', good: true, rowSum: tp + fn },
		{ r: 0, c: 1, v: fn, name: 'False negative', good: false, rowSum: tp + fn },
		{ r: 1, c: 0, v: fp, name: 'False positive', good: false, rowSum: fp + tn },
		{ r: 1, c: 1, v: tn, name: 'True negative', good: true, rowSum: fp + tn }
	]);
	const f = (v: number) => v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
</script>

<div class="host" bind:clientWidth={w}>
	<svg data-chart-root viewBox={`0 0 ${W} ${ok ? H : 140}`} width={W} height={ok ? H : 140} role="img" aria-label={ariaLabel}>
		<title>{ariaLabel}</title>
		{#if !ok || total <= 0}
			<text x={W / 2} y={70} text-anchor="middle" fill="#71829a" font-size="11" font-family={MONO}>No data supplied</text>
		{:else}
			<text x={L + cw} y={18} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>PREDICTED{threshold ? ` · ${threshold}` : ''}</text>
			<text x={L + cw / 2} y={T - 10} text-anchor="middle" fill="#a7b8c9" font-size="10" font-family={MONO}>{positiveLabel}</text>
			<text x={L + cw * 1.5} y={T - 10} text-anchor="middle" fill="#a7b8c9" font-size="10" font-family={MONO}>{negativeLabel}</text>
			<text transform={`translate(14,${T + cellH}) rotate(-90)`} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>OBSERVED</text>
			<text x={L - 8} y={T + cellH / 2 + 3} text-anchor="end" fill="#a7b8c9" font-size="10" font-family={MONO}>{positiveLabel.slice(0, 12)}</text>
			<text x={L - 8} y={T + cellH * 1.5 + 3} text-anchor="end" fill="#a7b8c9" font-size="10" font-family={MONO}>{negativeLabel.slice(0, 12)}</text>
			{#each cells as c}
				{@const share = c.v / total}
				<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
				<g role="img" tabindex="0" aria-label={`${c.name}: ${f(c.v)}, ${(100 * share).toFixed(1)} percent of total, ${(100 * c.v / (c.rowSum || 1)).toFixed(1)} percent of observed row`} style="outline:none">
					<rect x={L + c.c * cw + 1} y={T + c.r * cellH + 1} width={cw - 2} height={cellH - 2} fill={c.good ? '#2bb8b0' : '#fb7185'} fill-opacity={0.12 + 0.6 * Math.min(1, share * 2.2)} stroke={c.good ? 'rgba(43,184,176,.5)' : 'rgba(251,113,133,.5)'} />
					<text x={L + (c.c + 0.5) * cw} y={T + c.r * cellH + 28} text-anchor="middle" fill="#eef7f6" font-size="16" font-weight="600" font-family={MONO}>{f(c.v)}</text>
					<text x={L + (c.c + 0.5) * cw} y={T + c.r * cellH + 46} text-anchor="middle" fill="#cbd5e1" font-size="10" font-family={MONO}>{c.name}</text>
					<text x={L + (c.c + 0.5) * cw} y={T + c.r * cellH + 61} text-anchor="middle" fill="#8fa2b8" font-size="10" font-family={MONO}>{(100 * share).toFixed(1)}% total · {(100 * c.v / (c.rowSum || 1)).toFixed(1)}% row</text>
				</g>
			{/each}
			<text x={L} y={T + cellH * 2 + 22} fill="#8fa2b8" font-size="10" font-family={MONO}>sensitivity {(100 * tp / (tp + fn || 1)).toFixed(1)}% · specificity {(100 * tn / (tn + fp || 1)).toFixed(1)}% · precision {(100 * tp / (tp + fp || 1)).toFixed(1)}%</text>
			<text x={L} y={T + cellH * 2 + 40} fill="#53647b" font-size="10" font-family={MONO}>total {f(total)}</text>
		{/if}
	</svg>
</div>
<style>.host{width:100%;min-width:0}svg{display:block;max-width:100%;margin:auto}</style>
