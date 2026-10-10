<script lang="ts">
	// ModelBadge — "Forecast model: V3" chip with a details popover (bundle id, family per task,
	// artifact hashes, calibrator, frozen threshold). Reads only the response's own model_bundle.
	import type { ModelBundleInfo } from '$lib/services/api';
	let { bundle, versions = {} }: { bundle?: ModelBundleInfo; versions?: Record<string, string> } = $props();
	let open = $state(false);
	const TASKS: [string, string][] = [['recovery24', 'Recovery +24h'], ['recovery48', 'Recovery +48h'], ['icu_stay_time', 'Remaining ICU time'], ['organ_support', 'Organ-support risk']];
	const short = (h?: string) => (h ? `${h.slice(0, 10)}…${h.slice(-6)}` : '—');
</script>

<div class="badge-wrap">
	<button class="badge" class:unknown={!bundle} aria-expanded={open} aria-haspopup="dialog" onclick={() => (open = !open)} data-testid="model-badge">
		<span class="dot"></span>
		<span>FORECAST MODEL</span>
		<b>{bundle ? `${bundle.release} · ${bundle.bundle_id}` : 'UNVERIFIED'}</b>
	</button>
	{#if open && bundle}
		<div class="popover" role="dialog" aria-label="Forecast model details">
			<header><b>Scientific bundle {bundle.release}</b><button aria-label="Close" onclick={() => (open = false)}>×</button></header>
			<dl>
				{#each TASKS as [task, label]}
					{@const t = bundle.tasks?.[task]}
					{#if t}
						<div><dt>{label}</dt><dd>{t.family} · {t.feature_variant}<small>sha256 {short(t.model_sha256)}</small></dd></div>
					{/if}
				{/each}
				<div><dt>Support calibration</dt><dd>{bundle.calibrator?.method} (Platt)<small>params {short(bundle.calibrator?.params_sha256)}</small></dd></div>
				<div><dt>Frozen alert threshold</dt><dd>{Number(bundle.threshold).toFixed(6)}</dd></div>
				<div><dt>Selected-models manifest</dt><dd><small>sha256 {short(bundle.selected_models_sha256)}</small></dd></div>
				{#if versions.serving_version}<div><dt>Serving version</dt><dd>{versions.serving_version}</dd></div>{/if}
			</dl>
			<p>Frozen V3 release, selected on development evidence and evaluated once on a sealed cohort. Retrospective replay — not real-time clinical prediction.</p>
		</div>
	{/if}
</div>

<style>
	.badge-wrap { position: relative; display: inline-block; }
	.badge { display: inline-flex; align-items: center; gap: 8px; padding: 7px 12px; border: 1px solid rgba(43,184,176,.4); background: rgba(43,184,176,.07); color: #9be7e1; font: 8px 'JetBrains Mono', monospace; letter-spacing: .1em; cursor: pointer; }
	.badge b { color: #eef7f6; font-weight: 500; }
	.badge.unknown { border-color: rgba(251,113,133,.45); background: rgba(251,113,133,.08); color: #fecdd3; }
	.dot { width: 6px; height: 6px; border-radius: 50%; background: #2bb8b0; box-shadow: 0 0 8px #2bb8b0; }
	.popover { position: absolute; z-index: 40; top: calc(100% + 6px); left: 0; width: min(430px, 88vw); padding: 14px 16px; border: 1px solid rgba(148,163,184,.25); background: #070d19; box-shadow: 0 18px 40px rgba(0,0,0,.5); }
	.popover header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; color: #e5f1f0; font: 500 13px 'Space Grotesk', sans-serif; }
	.popover header button { border: 0; background: none; color: #94a3b8; font-size: 18px; cursor: pointer; }
	dl { display: grid; gap: 9px; margin: 0; }
	dl div { display: grid; grid-template-columns: 150px 1fr; gap: 10px; }
	dt { color: #64748b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
	dd { margin: 0; color: #cbd5e1; font: 10px 'JetBrains Mono', monospace; }
	dd small { display: block; margin-top: 2px; color: #64748b; font-size: 8px; }
	p { margin: 12px 0 0; color: #71829a; font-size: 10px; line-height: 1.5; }
</style>
