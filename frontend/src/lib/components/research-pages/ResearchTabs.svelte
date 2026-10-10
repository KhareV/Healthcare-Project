<script lang="ts">
	import { page } from '$app/state';
	const TABS = [
		{ href: '/research/analytics', label: 'Overview' },
		{ href: '/research/comparison', label: 'Comparison' },
		{ href: '/research/results', label: 'Classification & regression' },
		{ href: '/research/experiments', label: 'Selection & learning' },
		{ href: '/research/ablation', label: 'Ablation' },
		{ href: '/research/robustness', label: 'Robustness' },
		{ href: '/research/explainability', label: 'Explainability' },
		{ href: '/research/defect', label: 'Defect D1' },
		{ href: '/research/diagnostic', label: 'Latent-state diagnostic' }
	];
	const current = $derived(page.url.pathname.replace(/\/$/, ''));
</script>

<nav class="tabs" aria-label="Model research sections">
	{#each TABS as tab}
		<a href={tab.href} class:active={current === tab.href} class:priv={tab.href.endsWith('diagnostic')} aria-current={current === tab.href ? 'page' : undefined}>{tab.label}</a>
	{/each}
</nav>

<style>
	.tabs{display:flex;gap:6px;margin:0 0 20px;padding-bottom:2px;overflow-x:auto;scrollbar-width:thin;max-width:100%}
	a{flex:0 0 auto;padding:9px 13px;border:1px solid rgba(148,163,184,.18);background:rgba(10,18,34,.8);color:#8494a9;text-decoration:none;font:8px 'JetBrains Mono',monospace;letter-spacing:.09em;text-transform:uppercase}
	a:hover{color:#dce9e8;border-color:rgba(43,184,176,.4)}
	a.active{color:#eef7f6;border-color:#2bb8b0;background:rgba(43,184,176,.12)}
	a.priv{border-style:dashed;border-color:rgba(167,139,250,.5);color:#c4b5fd}
	a.priv.active{background:rgba(167,139,250,.14);border-color:#a78bfa}
	a:focus-visible{outline:2px solid #38bdf8;outline-offset:1px}
</style>
