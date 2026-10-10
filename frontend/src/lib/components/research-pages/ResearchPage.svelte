<script lang="ts">
	import type { Snippet } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import ResearchTabs from './ResearchTabs.svelte';
	import EvidenceLegend from './EvidenceLegend.svelte';
	let {
		eyebrow,
		title,
		description,
		legend = false,
		loading = false,
		error = '',
		children
	}: { eyebrow: string; title: string; description: string; legend?: boolean; loading?: boolean; error?: string; children?: Snippet } = $props();
</script>

<WorkbenchPage {eyebrow} {title} {description}>
	<ResearchTabs />
	{#if legend}<EvidenceLegend />{/if}
	{#if error}
		<div class="state error" role="alert">{error}</div>
	{:else if loading}
		<div class="state" aria-live="polite">Loading frozen evidence artifacts…</div>
	{:else}
		<div class="rbody">{@render children?.()}</div>
	{/if}
</WorkbenchPage>

<style>
	.state{padding:28px;border:1px dashed rgba(148,163,184,.25);color:#8494a9;font:10px 'JetBrains Mono',monospace;letter-spacing:.08em}
	.state.error{border-color:rgba(251,113,133,.5);color:#fda4af}
	.rbody{display:grid;gap:20px;min-width:0}
	.rbody :global(.rgrid){display:grid;gap:18px;min-width:0}
	.rbody :global(.rgrid.two){grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr))}
	.rbody :global(.rgrid.three){grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}
	.rbody :global(.rtiles){display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:14px}
	.rbody :global(.rmeta){display:flex;flex-wrap:wrap;gap:8px 20px;color:#53647b;font:7px 'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase}
	.rbody :global(.rnote){margin:0;padding:12px 14px;border-left:3px solid #2bb8b0;background:rgba(43,184,176,.06);color:#9fb0c3;font:12px/1.6 Inter,sans-serif}
	.rbody :global(.rnote.warn){border-left-color:#fb7185;background:rgba(251,113,133,.07);color:#fbcfd8}
	.rbody :global(.rnote.priv){border-left-color:#a78bfa;background:rgba(167,139,250,.08);color:#d9d0ff}
	.rbody :global(.rnote.amber){border-left-color:#fbbf24;background:rgba(251,191,36,.07);color:#f5e3ad}
	.rbody :global(.rnote b){color:#eef7f6;font-weight:600}
	.rbody :global(h3.rh){margin:6px 0 -6px;color:#dce9e8;font:500 14px 'Space Grotesk',sans-serif;letter-spacing:.01em}
	.rbody :global(details.acc){border:1px solid rgba(251,113,133,.35);background:rgba(251,113,133,.04)}
	.rbody :global(details.acc>summary){cursor:pointer;padding:12px 14px;color:#fda4af;font:8px 'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase}
	.rbody :global(details.acc>.acc-body){display:grid;gap:16px;padding:14px}
	.rbody :global(.chips){display:flex;flex-wrap:wrap;gap:8px}
	.rbody :global(.chip){padding:6px 9px;border:1px solid rgba(148,163,184,.2);color:#b7c6d4;font:8px 'JetBrains Mono',monospace;letter-spacing:.06em}
	.rbody :global(.kv){display:grid;grid-template-columns:auto 1fr;gap:6px 14px;margin:0;font:10px 'JetBrains Mono',monospace;color:#8494a9}
	.rbody :global(.kv dt){color:#53647b}
	.rbody :global(.kv dd){margin:0;color:#cbd5e1;word-break:break-all}
	.rbody :global(.slider){display:grid;gap:8px;margin:0 0 12px;font:9px 'JetBrains Mono',monospace;color:#8494a9}
	.rbody :global(.slider input){width:100%;accent-color:#2bb8b0}
</style>
