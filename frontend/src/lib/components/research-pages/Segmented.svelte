<script lang="ts">
	// Small segmented selector (task / model / dimension pickers). Keyboard + aria friendly.
	let {
		options,
		value = $bindable(),
		label = ''
	}: { options: { value: string; label: string }[]; value: string; label?: string } = $props();
</script>

<div class="seg" role="radiogroup" aria-label={label || 'selector'}>
	{#if label}<span class="seg-label">{label}</span>{/if}
	<div class="seg-row">
		{#each options as option}
			<button type="button" role="radio" aria-checked={value === option.value} class:on={value === option.value} onclick={() => (value = option.value)}>{option.label}</button>
		{/each}
	</div>
</div>

<style>
	.seg{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:0 0 14px}
	.seg-label{color:#53647b;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em;text-transform:uppercase}
	.seg-row{display:flex;flex-wrap:wrap;gap:6px}
	button{padding:7px 11px;border:1px solid rgba(148,163,184,.2);background:rgba(10,18,34,.8);color:#8494a9;font:8px 'JetBrains Mono',monospace;letter-spacing:.08em;cursor:pointer}
	button:hover{color:#dce9e8;border-color:rgba(43,184,176,.4)}
	button.on{color:#eef7f6;border-color:#2bb8b0;background:rgba(43,184,176,.12)}
	button:focus-visible{outline:2px solid #38bdf8;outline-offset:1px}
</style>
