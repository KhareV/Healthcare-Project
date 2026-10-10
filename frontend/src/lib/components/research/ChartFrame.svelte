<script lang="ts">
	// ChartFrame — provenance-aware card around any chart: status chips, details popover,
	// PNG/SVG/CSV export (title + source + units embedded), expand/fullscreen with focus trap.
	import type { Snippet } from 'svelte';
	import { onDestroy } from 'svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import type { Provenance } from '$lib/research/types';
	import { STATUS_STYLE, MONO } from '$lib/research/palette';
	import { downloadBlob, rowsToCsv, safeFilename, svgToPng, svgToSvgString, type CsvColumn } from '$lib/research/chart-export';
	import { fmtInt, shortHash } from '$lib/research/format';

	let {
		title,
		subtitle = '',
		provenance,
		data = [],
		columns,
		units = '',
		loading = false,
		error = '',
		empty = false,
		emptyMessage = 'No evidence is available for this view.',
		children
	}: {
		title: string;
		subtitle?: string;
		provenance: Provenance;
		data?: Record<string, unknown>[];
		columns?: CsvColumn[];
		units?: string;
		loading?: boolean;
		error?: string;
		empty?: boolean;
		emptyMessage?: string;
		children?: Snippet;
	} = $props();

	let host: HTMLDivElement | undefined = $state();
	let frame: HTMLDivElement | undefined = $state();
	let expanded = $state(false);
	let showDetails = $state(false);
	let busy = $state('');
	let notice = $state('');
	let lastFocus: HTMLElement | null = null;
	let previousOverflow: string | null = null;

	const st = $derived(STATUS_STYLE[provenance.status]);
	const sourceLine = $derived(
		[provenance.status, provenance.bundle, provenance.partition, provenance.metric, provenance.source, provenance.sha256 ? `sha256 ${shortHash(provenance.sha256, 16)}` : '']
			.filter(Boolean)
			.join(' · ')
	);
	const viewState = $derived(loading ? 'loading' : error ? 'error' : empty ? 'empty' : 'ready');

	function chartSvg(): SVGSVGElement | null {
		return (host?.querySelector('svg[data-chart-root]') as SVGSVGElement | null) ?? (host?.querySelector('svg') as SVGSVGElement | null);
	}
	function meta() {
		return { title, subtitle, source: sourceLine, units };
	}
	async function exportAs(kind: 'png' | 'svg' | 'csv') {
		notice = '';
		try {
			busy = kind;
			if (kind === 'csv') {
				if (!data.length) throw new Error('No tabular data is attached to this chart');
				downloadBlob(new Blob([rowsToCsv(data, columns)], { type: 'text/csv;charset=utf-8' }), safeFilename(title, 'csv'));
			} else {
				const svg = chartSvg();
				if (!svg) throw new Error('Chart is not rendered yet');
				const s = svgToSvgString(svg, meta());
				if (kind === 'svg') downloadBlob(new Blob([s], { type: 'image/svg+xml;charset=utf-8' }), safeFilename(title, 'svg'));
				else downloadBlob(await svgToPng(s, 2), safeFilename(title, 'png'));
			}
		} catch (e) {
			notice = e instanceof Error ? e.message : 'Export failed';
		} finally {
			busy = '';
		}
	}

	function focusables(): HTMLElement[] {
		if (!frame) return [];
		return Array.from(frame.querySelectorAll<HTMLElement>('button,[href],input,select,[tabindex]:not([tabindex="-1"])')).filter((el) => !el.hasAttribute('disabled'));
	}
	function setExpanded(v: boolean) {
		if (expanded === v) return;
		if (v && typeof document !== 'undefined') {
			lastFocus = document.activeElement as HTMLElement | null;
		}
		expanded = v;
		if (typeof document !== 'undefined') {
			if (v) {
				previousOverflow = document.body.style.overflow;
				document.body.style.overflow = 'hidden';
			} else {
				document.body.style.overflow = previousOverflow ?? '';
				previousOverflow = null;
			}
		}
		if (v) {
			queueMicrotask(() => frame?.querySelector<HTMLElement>('[data-close]')?.focus());
		} else {
			// The expand button is conditionally rendered, so restore focus only
			// after Svelte has replaced the close button in the DOM.
			queueMicrotask(() => {
				const target = lastFocus?.isConnected
					? lastFocus
					: frame?.querySelector<HTMLElement>('[data-expand]');
				target?.focus();
			});
		}
	}
	onDestroy(() => {
		if (expanded && typeof document !== 'undefined') document.body.style.overflow = previousOverflow ?? '';
	});
	function onKeydown(e: KeyboardEvent) {
		if (!expanded) return;
		if (e.key === 'Escape') {
			e.preventDefault();
			setExpanded(false);
		} else if (e.key === 'Tab') {
			const f = focusables();
			if (!f.length) return;
			const first = f[0];
			const last = f[f.length - 1];
			if (e.shiftKey && document.activeElement === first) {
				e.preventDefault();
				last.focus();
			} else if (!e.shiftKey && document.activeElement === last) {
				e.preventDefault();
				first.focus();
			}
		}
	}
</script>

<svelte:window onkeydown={onKeydown} />

<div class="frame" class:expanded bind:this={frame} role={expanded ? 'dialog' : undefined} aria-modal={expanded ? 'true' : undefined} aria-label={expanded ? `${title} (expanded)` : undefined}>
	{#if expanded}<button class="scrim" tabindex="-1" aria-hidden="true" onclick={() => setExpanded(false)}></button>{/if}
	<div class="card">
		<Panel eyebrow={st.label} title={title} note={subtitle}>
			<div class="chips" aria-label="Evidence provenance">
				<span class="chip status" class:dashed={st.dashed} style={`--c:${st.color}`} title={st.label}><b aria-hidden="true">{st.glyph}</b>{provenance.status}</span>
				{#if provenance.bundle}<span class="chip">{provenance.bundle}</span>{/if}
				{#if provenance.partition}<span class="chip">{provenance.partition}</span>{/if}
				{#if provenance.n_rows != null}<span class="chip">{fmtInt(provenance.n_rows)} rows</span>{/if}
				{#if provenance.n_stays != null}<span class="chip">{fmtInt(provenance.n_stays)} stays</span>{/if}
				{#if provenance.metric}<span class="chip">{provenance.metric}</span>{/if}
				<button class="chip link" aria-expanded={showDetails} onclick={() => (showDetails = !showDetails)}>{showDetails ? 'Hide details' : 'Source details'}</button>
				<span class="spacer"></span>
				<div class="toolbar" role="toolbar" aria-label="Chart actions">
					<button onclick={() => exportAs('png')} disabled={busy !== '' || viewState !== 'ready'} aria-label={`Download ${title} as PNG`}>PNG</button>
					<button onclick={() => exportAs('svg')} disabled={busy !== '' || viewState !== 'ready'} aria-label={`Download ${title} as SVG`}>SVG</button>
					<button onclick={() => exportAs('csv')} disabled={busy !== '' || !data.length} aria-label={`Download ${title} data as CSV`}>CSV</button>
					{#if expanded}
						<button data-close onclick={() => setExpanded(false)} aria-label="Close expanded chart">Close ✕</button>
					{:else}
						<button data-expand onclick={() => setExpanded(true)} aria-label={`Expand ${title}`}>Expand ⤢</button>
					{/if}
				</div>
			</div>
			{#if showDetails}
				<dl class="details" aria-label="Source details">
					<dt>Status</dt><dd>{st.label}</dd>
					{#if provenance.bundle}<dt>Model bundle</dt><dd>{provenance.bundle}</dd>{/if}
					{#if provenance.partition}<dt>Evaluation partition</dt><dd>{provenance.partition}</dd>{/if}
					{#if provenance.n_rows != null}<dt>Eligible rows</dt><dd>{fmtInt(provenance.n_rows)}</dd>{/if}
					{#if provenance.n_stays != null}<dt>Unique stays</dt><dd>{fmtInt(provenance.n_stays)}</dd>{/if}
					{#if provenance.metric}<dt>Metric definition</dt><dd>{provenance.metric}</dd>{/if}
					{#if provenance.ci_method}<dt>Interval method</dt><dd>{provenance.ci_method}</dd>{/if}
					{#if provenance.source}<dt>Source artifact</dt><dd class="mono">{provenance.source}</dd>{/if}
					{#if provenance.sha256}<dt>SHA-256</dt><dd class="mono">{provenance.sha256}</dd>{/if}
					{#if units}<dt>Units</dt><dd>{units}</dd>{/if}
				</dl>
			{/if}
			{#if notice}<p class="notice" role="alert">{notice}</p>{/if}
			<div class="chart-host" bind:this={host} aria-busy={viewState === 'loading'}>
				{#if viewState === 'loading'}
					<div class="state" role="status"><i class="spin"></i>Loading evidence…</div>
				{:else if viewState === 'error'}
					<div class="state err" role="alert">Unable to load this view: {error}</div>
				{:else if viewState === 'empty'}
					<div class="state" role="status">{emptyMessage}</div>
				{:else}
					{@render children?.()}
				{/if}
			</div>
		</Panel>
	</div>
</div>

<style>
	.frame{min-width:0}
	.frame.expanded{position:fixed;inset:0;z-index:100;display:grid;place-items:center;padding:clamp(8px,3vw,40px)}
	.scrim{position:absolute;inset:0;border:0;background:rgba(2,6,14,.86);cursor:default}
	.card{position:relative;min-width:0;width:100%}
	.frame.expanded .card{max-width:1400px;max-height:100%;overflow:auto;background:#050a15}
	.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:12px}
	.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 7px;border:1px solid rgba(148,163,184,.2);color:#8fa2b8;background:rgba(15,23,42,.6);font:9px 'JetBrains Mono',monospace;letter-spacing:.06em}
	.chip.status{color:var(--c);border-color:var(--c);background:color-mix(in srgb,var(--c) 10%,transparent)}
	.chip.status.dashed{border-style:dashed}
	.chip.link{cursor:pointer;color:#38bdf8;border-color:rgba(56,189,248,.3)}
	.chip.link:focus-visible,.toolbar button:focus-visible{outline:2px solid #38bdf8;outline-offset:2px}
	.spacer{flex:1}
	.toolbar{display:flex;gap:4px;flex-wrap:wrap}
	.toolbar button{padding:4px 8px;border:1px solid rgba(148,163,184,.25);background:rgba(15,23,42,.7);color:#a7b8c9;font:9px 'JetBrains Mono',monospace;letter-spacing:.08em;cursor:pointer}
	.toolbar button:hover:not(:disabled){border-color:#2bb8b0;color:#dffaf6}
	.toolbar button:disabled{opacity:.4;cursor:not-allowed}
	.details{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0 0 12px;padding:10px 12px;border:1px solid rgba(148,163,184,.15);background:rgba(8,14,26,.8);font:10px 'JetBrains Mono',monospace;color:#8fa2b8}
	.details dt{color:#53647b}.details dd{margin:0;overflow-wrap:anywhere;color:#cbd5e1}.details .mono{font-size:9px}
	.notice{margin:0 0 10px;color:#fbbf24;font:10px 'JetBrains Mono',monospace}
	.chart-host{min-width:0;overflow:hidden}
	.state{display:flex;align-items:center;justify-content:center;gap:10px;min-height:160px;color:#71829a;font:11px 'JetBrains Mono',monospace;text-align:center;padding:16px}
	.state.err{color:#fb7185}
	.spin{width:12px;height:12px;border:2px solid rgba(43,184,176,.25);border-top-color:#2bb8b0;border-radius:50%;animation:rk-spin .9s linear infinite}
	@keyframes rk-spin{to{transform:rotate(360deg)}}
	@media (prefers-reduced-motion:reduce){.spin{animation:none}}
	@media (max-width:520px){.toolbar{width:100%}}
</style>
