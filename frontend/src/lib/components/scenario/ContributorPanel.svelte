<script lang="ts">
	import type { PredictionResponse } from '$lib/services/api';
	type Task = 'recovery24' | 'recovery48' | 'icu_stay_time' | 'organ_support';
	let { prediction, task = $bindable('organ_support'), compact = false }: { prediction: PredictionResponse | null; task?: Task; compact?: boolean } = $props();
	const TASKS: { id: Task; label: string }[] = [
		{ id: 'recovery24', label: 'SOFA +24h' },
		{ id: 'recovery48', label: 'SOFA +48h' },
		{ id: 'icu_stay_time', label: 'ICU time' },
		{ id: 'organ_support', label: 'Support' }
	];
	const exp = $derived(prediction?.explanations?.[task] ?? null);
	const items = $derived(
		exp
			? [...exp.top_positive_contributors.slice(0, compact ? 3 : 5), ...exp.top_negative_contributors.slice(0, compact ? 3 : 5)].sort((a, b) => b.attribution - a.attribution)
			: []
	);
	const maxAbs = $derived(Math.max(1e-9, ...items.map((i) => Math.abs(i.attribution))));
	const family = $derived((exp?.diagnostics as Record<string, unknown> | undefined)?.model_family as string | undefined);
	const domain = $derived((exp?.diagnostics?.explains as string | undefined) ?? '');
</script>

<section class="contrib" aria-label="Model explanation">
	<header>
		<span>WHAT CONTRIBUTED TO THIS FORECAST</span>
		<div class="tabs" role="tablist">
			{#each TASKS as t}<button role="tab" aria-selected={task === t.id} class:on={task === t.id} onclick={() => (task = t.id)}>{t.label}</button>{/each}
		</div>
	</header>
	{#if !exp}
		<p class="muted">No explanation available.</p>
	{:else}
		<ul>
			{#each items as item}
				<li title={item.feature_name}>
					<span class="name">{item.label}</span>
					<span class="track"><i class:pos={item.attribution > 0} class:neg={item.attribution < 0} style={`width:${(Math.abs(item.attribution) / maxAbs) * 100}%`}></i></span>
					<b class:pos={item.attribution > 0} class:neg={item.attribution < 0}>{item.attribution > 0 ? '+' : ''}{item.attribution.toFixed(3)}</b>
				</li>
			{/each}
		</ul>
		<p class="note">TreeSHAP{family ? ` (${family})` : ''}. {domain || 'Raw model margin.'} Contribution to the prediction — not a causal effect.</p>
	{/if}
</section>

<style>
	.contrib{min-width:0;padding:14px;border:1px solid rgba(148,163,184,.14);background:#080f1b}
	header{display:flex;flex-wrap:wrap;gap:10px;justify-content:space-between;align-items:center;margin-bottom:10px}
	header>span{color:#64748b;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em}
	.tabs{display:flex;gap:4px;flex-wrap:wrap}.tabs button{padding:5px 8px;border:1px solid rgba(148,163,184,.18);background:#0a1220;color:#8494a9;font:8px 'JetBrains Mono',monospace;cursor:pointer}.tabs button.on{border-color:rgba(43,184,176,.5);color:#82e9e1;background:rgba(43,184,176,.1)}
	ul{list-style:none;margin:0;padding:0;display:grid;gap:7px}li{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(60px,1fr) 56px;gap:8px;align-items:center;font-size:11px;color:#b7c6d4}
	.name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.track{height:7px;background:rgba(148,163,184,.1)}.track i{display:block;height:100%}.track i.pos{background:#fb7185}.track i.neg{background:#38bdf8}
	b{font:9px 'JetBrains Mono',monospace;text-align:right}b.pos{color:#fb7185}b.neg{color:#38bdf8}
	.note{margin:10px 0 0;color:#64748b;font:8px 'JetBrains Mono',monospace;line-height:1.6}.muted{color:#64748b;font-size:11px}
</style>
