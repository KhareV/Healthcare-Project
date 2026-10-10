<script lang="ts">
	import { Play, Waypoints } from '@lucide/svelte';
	import { prettySubtype, type Participant } from '$lib/services/showcase';
	let { participant, onopen, oncompare }: { participant: Participant; onopen: (p: Participant) => void; oncompare?: (p: Participant) => void } = $props();
</script>

<article class="card" data-testid="scenario-card" data-scenario={participant.key}>
	<header>
		<span class="tag">{participant.kind === 'scenario' ? participant.key : participant.alias}</span>
		<b class="subtype">{prettySubtype(participant.subtype)}</b>
	</header>
	<h3>{participant.title}</h3>
	{#if participant.objective}<p class="objective">{participant.objective}</p>{/if}
	<dl>
		<div><dt>REPLAY LENGTH</dt><dd>{participant.legalCutoffs.length} legal cutoffs</dd></div>
		{#if participant.keyEvent}<div><dt>KEY OBSERVED EVENT</dt><dd>{participant.keyEvent}</dd></div>{/if}
		{#if participant.focus}<div><dt>ANALYTICAL FOCUS</dt><dd>{participant.focus}</dd></div>{/if}
		<div><dt>SOURCE</dt><dd>{participant.cohortLabel}</dd></div>
	</dl>
	<footer>
		<button class="open" onclick={() => onopen(participant)}><Play size={13} /> OPEN / REPLAY</button>
		{#if oncompare}<button onclick={() => oncompare?.(participant)}><Waypoints size={13} /> COMPARE</button>{/if}
	</footer>
</article>

<style>
	.card{display:flex;flex-direction:column;gap:10px;min-width:0;padding:16px;border:1px solid rgba(148,163,184,.16);background:linear-gradient(145deg,rgba(10,18,34,.95),rgba(5,10,21,.95));transition:border-color .2s}
	.card:hover{border-color:rgba(43,184,176,.4)}
	header{display:flex;justify-content:space-between;align-items:center;gap:8px}
	.tag{color:#2bb8b0;font:8px 'JetBrains Mono',monospace;letter-spacing:.14em}
	.subtype{padding:3px 7px;border:1px solid rgba(56,189,248,.3);color:#38bdf8;font:7px 'JetBrains Mono',monospace;letter-spacing:.1em}
	h3{margin:0;color:#e5f1f0;font:500 17px 'Space Grotesk',sans-serif}
	.objective{margin:0;color:#8494a9;font-size:12px;line-height:1.55}
	dl{display:grid;gap:8px;margin:2px 0 0}dt{color:#53647b;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em}dd{margin:3px 0 0;color:#b7c6d4;font-size:11px;line-height:1.45}
	footer{display:flex;gap:7px;margin-top:auto;padding-top:6px}
	button{display:flex;align-items:center;gap:7px;padding:9px 11px;border:1px solid rgba(148,163,184,.2);background:#0a1220;color:#b7c6d4;font:8px 'JetBrains Mono',monospace;letter-spacing:.08em;cursor:pointer}
	button.open{border-color:rgba(43,184,176,.45);background:rgba(43,184,176,.1);color:#82e9e1}button:hover{border-color:rgba(43,184,176,.6)}
</style>
