<script lang="ts">
	import type { PredictionResponse } from '$lib/services/api';
	import type { Story } from '$lib/services/showcase';
	let { story, cutoffIndex, current, previous, nCutoffs }: { story: Story | null; cutoffIndex: number; current: PredictionResponse | null; previous: PredictionResponse | null; nCutoffs: number } = $props();

	const step = $derived(story?.steps[cutoffIndex] ?? null);
	const earlier = $derived(cutoffIndex > 0 ? (story?.steps[cutoffIndex - 1] ?? null) : null);
	const dir = (v: number, eps: number) => (Math.abs(v) < eps ? 'essentially unchanged' : v > 0 ? 'higher' : 'lower');
	const rows = $derived.by(() => {
		if (!current) return [];
		const base = [
			{ label: 'SOFA now', now: current.current_sofa, prev: previous?.current_sofa, unit: '', digits: 1, eps: 0.05 },
			{ label: 'SOFA forecast +24h', now: current.recovery.sofa_hat_24h, prev: previous?.recovery.sofa_hat_24h, unit: '', digits: 1, eps: 0.05 },
			{ label: 'SOFA forecast +48h', now: current.recovery.sofa_hat_48h, prev: previous?.recovery.sofa_hat_48h, unit: '', digits: 1, eps: 0.05 },
			{ label: 'Remaining ICU time', now: current.icu_stay_time.remaining_hours, prev: previous?.icu_stay_time.remaining_hours, unit: ' h', digits: 1, eps: 0.1 },
			{ label: 'Support initiation risk', now: current.organ_support.probability_24h * 100, prev: previous ? previous.organ_support.probability_24h * 100 : undefined, unit: ' %', digits: 1, eps: 0.1 }
		];
		return base.map((r) => ({ ...r, delta: r.prev === undefined ? null : r.now - r.prev }));
	});
	const topNow = $derived.by(() => {
		if (!current) return [] as { task: string; label: string; v: number }[];
		const pick = (task: 'recovery24' | 'organ_support', name: string) => {
			const e = current!.explanations[task];
			const top = [...e.top_positive_contributors, ...e.top_negative_contributors].sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution))[0];
			return top ? { task: name, label: top.label, v: top.attribution } : null;
		};
		return [pick('recovery24', 'SOFA +24h'), pick('organ_support', 'Support risk')].filter(Boolean) as { task: string; label: string; v: number }[];
	});
</script>

<section class="story" aria-label="Guided story">
	{#if !story || !step}
		<p class="muted">The guided story is available for the manifest scenario patients (it is built from their structured observation history). The original demo patients support replay and comparison.</p>
	{:else}
		<header><span>GUIDED STORY · CUTOFF {cutoffIndex + 1} OF {nCutoffs}</span><h3>{story.title}</h3><p>{story.objective}</p></header>
		<div class="block">
			<h4>1 · What the model can see now</h4>
			<p class="sub">Only information at or before this cutoff enters the forecast.</p>
			<ul>{#each step.facts as f}<li class={f.kind}>{f.text}</li>{/each}</ul>
		</div>
		<div class="block">
			<h4>2 · The forecast</h4>
			{#if !current}<p class="muted">Computing the V3 forecast…</p>{:else}
				<table><thead><tr><th>Quantity</th><th>This cutoff</th><th>Change vs previous cutoff</th></tr></thead><tbody>
					{#each rows as r}
						<tr><td>{r.label}</td><td>{r.now.toFixed(r.digits)}{r.unit}</td>
							<td>{#if r.delta === null}first cutoff{:else}{r.delta >= 0 ? '+' : ''}{r.delta.toFixed(r.digits)}{r.unit} ({dir(r.delta, r.eps)}){/if}</td></tr>
					{/each}
				</tbody></table>
				<p class="sub">Support risk {current.organ_support.alert ? 'is at or above' : 'is below'} the frozen operating threshold ({(current.organ_support.threshold * 100).toFixed(1)}%). The SOFA +48h forecast is independent of the +24h forecast.</p>
			{/if}
		</div>
		<div class="block">
			<h4>3 · Largest single contributors</h4>
			{#if topNow.length}<ul>{#each topNow as t}<li><b>{t.task}:</b> {t.label} ({t.v >= 0 ? '+' : ''}{t.v.toFixed(3)} on the raw model margin)</li>{/each}</ul>{:else}<p class="muted">—</p>{/if}
		</div>
		<div class="block hind">
			<h4>4 · Revealed since the previous cutoff <em>(hindsight for the previous forecast)</em></h4>
			{#if earlier && earlier.revealed_next.length}
				<ul>{#each earlier.revealed_next as f}<li>{f.text}</li>{/each}</ul>
				<p class="sub">These events happened after the previous cutoff, so they could not have influenced the previous forecast.</p>
			{:else}<p class="muted">{cutoffIndex === 0 ? 'Nothing precedes the first cutoff.' : 'No support starts/stops or large SOFA changes between the previous and this cutoff.'}</p>{/if}
		</div>
		<p class="note">{story.note}</p>
	{/if}
</section>

<style>
	.story{display:grid;gap:10px;min-width:0;padding:16px;border:1px solid rgba(43,184,176,.2);background:linear-gradient(135deg,rgba(43,184,176,.045),#070d18 42%)}
	header span{color:#2bb8b0;font:7px 'JetBrains Mono',monospace;letter-spacing:.13em}header h3{margin:6px 0 4px;color:#e5f1f0;font:500 18px 'Space Grotesk',sans-serif}header p{margin:0;color:#8494a9;font-size:12px}
	.block{padding:12px;border:1px solid rgba(148,163,184,.12);background:#060c17}.block h4{margin:0 0 6px;color:#cfe3e1;font:500 12px 'Space Grotesk',sans-serif}.block h4 em{color:#fbbf24;font-style:normal;font:8px 'JetBrains Mono',monospace}
	.hind{border-color:rgba(251,191,36,.25)}.sub{margin:4px 0 6px;color:#64748b;font:9px 'JetBrains Mono',monospace;line-height:1.55}
	ul{margin:0;padding-left:16px;display:grid;gap:4px;color:#b7c6d4;font-size:12px;line-height:1.5}
	table{width:100%;border-collapse:collapse;font-size:11px;color:#b7c6d4}th{padding:5px 6px;text-align:left;color:#64748b;font:7px 'JetBrains Mono',monospace;letter-spacing:.1em;border-bottom:1px solid rgba(148,163,184,.14)}td{padding:6px;border-bottom:1px solid rgba(148,163,184,.07)}
	.note,.muted{margin:0;color:#64748b;font:9px 'JetBrains Mono',monospace;line-height:1.6}
</style>
