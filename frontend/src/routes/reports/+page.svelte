<script lang="ts">
	import { onMount } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import Panel from '$lib/components/dashboard/Panel.svelte';
	import DatasetPreview from '$lib/components/cohort/DatasetPreview.svelte';
	import { cohortApi, toKitStatus, type DatasetProvenance, type ResearchDataset } from '$lib/services/cohort';
	import { SECTIONS, ALWAYS_IDS, MAX_TABLE_ROWS, buildMarkdown, buildHtml, buildTex, indexCsv, toTable, type SectionDef, type ReportInput } from '$lib/components/cohort/report';
	import { STATUS_STYLE } from '$lib/research/palette';
	import { downloadBlob } from '$lib/research/chart-export';

	let index = $state<DatasetProvenance[]>([]);
	let apiVersion = $state('');
	let error = $state('');
	let loading = $state(true);
	let selected = $state<Record<string, boolean>>(Object.fromEntries(SECTIONS.map((s) => [s.key, s.defaultOn])));
	const cache = new Map<string, ResearchDataset>();
	let loaded = $state<Record<string, ResearchDataset>>({});
	let busy = $state('');
	let notice = $state('');
	let view = $state<'builder' | 'index'>('builder');
	let filter = $state('');

	onMount(async () => {
		try {
			const idx = await cohortApi.index();
			index = idx.datasets;
			apiVersion = idx.api_version;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Could not load the research index.';
		} finally {
			loading = false;
		}
	});

	const available = $derived(new Set(index.map((d) => d.id)));
	const sections = $derived(SECTIONS.map((s) => ({ ...s, ids: s.ids.filter((i) => available.has(i)) })).filter((s) => s.ids.length));
	const activeSections = $derived(sections.filter((s) => selected[s.key]));
	const wantedIds = $derived(Array.from(new Set([...activeSections.flatMap((s) => s.ids), ...ALWAYS_IDS.filter((i) => available.has(i)), ...(available.has('defect_d1') ? ['defect_d1'] : [])])));

	async function ensure(ids: string[]) {
		const missing = ids.filter((i) => !cache.has(i));
		await Promise.all(missing.map(async (i) => cache.set(i, await cohortApi.dataset(i))));
		const next: Record<string, ResearchDataset> = {};
		ids.forEach((i) => cache.has(i) && (next[i] = cache.get(i)!));
		loaded = { ...loaded, ...next };
	}

	$effect(() => {
		const ids = wantedIds;
		if (!ids.length) return;
		ensure(ids).catch((cause) => (error = cause instanceof Error ? cause.message : 'dataset load failed'));
	});

	function stamp(): string {
		return new Date().toISOString().replace(/\.\d+Z$/, 'Z');
	}

	async function assemble(withTex: boolean): Promise<{ input: ReportInput; tex: Record<string, string> }> {
		await ensure(wantedIds);
		const prov = cache.get('provenance')?.data;
		const input: ReportInput = {
			generatedAt: stamp(),
			apiVersion,
			bundle: prov ? { bundle_id: prov.bundle_id, status: prov.status, threshold: prov.threshold, calibrator: prov.calibrator } : null,
			sections: activeSections.map((s) => ({ def: s as SectionDef, datasets: s.ids.map((i) => cache.get(i)!).filter(Boolean) })),
			always: ALWAYS_IDS.map((i) => cache.get(i)!).filter(Boolean),
			defect: cache.get('defect_d1') ?? null
		};
		const tex: Record<string, string> = {};
		if (withTex) {
			const all = [...input.sections.flatMap((s) => s.datasets), ...input.always];
			await Promise.all(
				all.map(async (ds) => {
					const t = toTable(ds);
					if (!t.omitted && t.total > 0 && t.total <= MAX_TABLE_ROWS && ds.provenance.downloadable_formats.includes('tex')) {
						try {
							tex[ds.provenance.id] = await cohortApi.exportText(ds.provenance.id, 'tex');
						} catch {
							/* table omitted from the combined document */
						}
					}
				})
			);
		}
		return { input, tex };
	}

	async function run(kind: 'md' | 'html' | 'tex' | 'bundle') {
		busy = kind;
		notice = '';
		try {
			const { input, tex } = await assemble(kind === 'tex' || kind === 'bundle');
			const day = input.generatedAt.slice(0, 10);
			const emit = (text: string, name: string, mime: string) => downloadBlob(new Blob([text], { type: `${mime};charset=utf-8` }), name);
			if (kind === 'md' || kind === 'bundle') emit(buildMarkdown(input), `prt-research-report-${day}.md`, 'text/markdown');
			if (kind === 'html' || kind === 'bundle') emit(buildHtml(input), `prt-research-report-${day}.html`, 'text/html');
			if (kind === 'tex' || kind === 'bundle') emit(buildTex(input, tex), `prt-research-report-${day}.tex`, 'application/x-tex');
			notice = `Report generated from ${wantedIds.length} datasets at ${input.generatedAt}.`;
		} catch (cause) {
			notice = cause instanceof Error ? cause.message : 'report generation failed';
		} finally {
			busy = '';
		}
	}

	const filtered = $derived(index.filter((d) => !filter || `${d.id} ${d.title} ${d.group} ${d.evidence_status} ${d.bundle}`.toLowerCase().includes(filter.toLowerCase())));
	const statusStyle = (s: string) => STATUS_STYLE[toKitStatus(s)];
	function downloadIndex() {
		downloadBlob(new Blob([indexCsv(index)], { type: 'text/csv;charset=utf-8' }), 'prt-figure-table-index.csv');
	}
</script>

<svelte:head><title>Research Report Export | Personalized Patient Recovery Trajectory</title></svelte:head>

<WorkbenchPage
	eyebrow="RESEARCH REPORT EXPORT"
	title="Assemble publication-ready evidence from the frozen results."
	description="Select sections, preview exact tables and figures, and export CSV, JSON, LaTeX, PNG/SVG, or one combined report. Every number is read from the evidence API; nothing is re-estimated or reworded."
>
	{#if loading}
		<div class="state">Loading the research index…</div>
	{:else if error}
		<div class="state error" role="alert">{error}</div>
	{:else}
		<div class="tabs" role="tablist" aria-label="Report workspace views">
			<button role="tab" aria-selected={view === 'builder'} class:on={view === 'builder'} onclick={() => (view = 'builder')}>REPORT BUILDER</button>
			<button role="tab" aria-selected={view === 'index'} class:on={view === 'index'} onclick={() => (view = 'index')}>FIGURE &amp; TABLE INDEX · {index.length}</button>
		</div>

		{#if view === 'builder'}
			<div class="layout">
				<aside>
					<Panel eyebrow="SECTIONS" title="Include in report" note={`${activeSections.length} selected`}>
						<ul class="checks">
							{#each sections as s}
								<li class:priv={s.privileged}>
									<label>
										<input type="checkbox" bind:checked={selected[s.key]} />
										<span>{s.title}</span>
										<small>{s.ids.length} dataset{s.ids.length === 1 ? '' : 's'}</small>
									</label>
								</li>
							{/each}
						</ul>
						<p class="hint">Provenance and the defect-D1 disclosure are always embedded. The latent-state diagnostic is privileged simulator evidence and is off by default.</p>
					</Panel>
					<Panel eyebrow="EXPORT" title="Combined report">
						<div class="export">
							<button type="button" onclick={() => run('md')} disabled={busy !== ''}>{busy === 'md' ? 'BUILDING…' : 'MARKDOWN (.md)'}</button>
							<button type="button" onclick={() => run('html')} disabled={busy !== ''}>{busy === 'html' ? 'BUILDING…' : 'HTML (self-contained)'}</button>
							<button type="button" onclick={() => run('tex')} disabled={busy !== ''}>{busy === 'tex' ? 'BUILDING…' : 'LATEX (.tex)'}</button>
							<button type="button" class="primary" onclick={() => run('bundle')} disabled={busy !== ''}>{busy === 'bundle' ? 'BUILDING…' : 'ALL THREE'}</button>
						</div>
						{#if notice}<p class="notice" role="status">{notice}</p>{/if}
						<p class="hint">Per-dataset CSV / JSON / LaTeX and per-figure PNG / SVG downloads are on each preview below.</p>
					</Panel>
				</aside>
				<section class="previews" aria-label="Section previews">
					{#each activeSections as s}
						<div class="section" class:priv={s.privileged}>
							<h3>{s.title}</h3>
							{#if s.privileged}<p class="banner">PRIVILEGED DIAGNOSTIC — simulator-state predictors; not deployable, not a clinical bound, never a serving input.</p>{/if}
							{#each s.ids as id}
								{#if loaded[id]}<DatasetPreview dataset={loaded[id]} />{:else}<div class="state small">Loading {id}…</div>{/if}
							{/each}
						</div>
					{/each}
					{#if !activeSections.length}<div class="state">Select at least one section to preview and export.</div>{/if}
				</section>
			</div>
		{:else}
			<div class="index-tools">
				<input type="search" placeholder="Filter by id, group, status or bundle" bind:value={filter} aria-label="Filter index" />
				<button type="button" onclick={downloadIndex}>DOWNLOAD INDEX (.csv)</button>
			</div>
			<div class="scroll">
				<table>
					<thead><tr><th>Id</th><th>Title</th><th>Group</th><th>Evidence status</th><th>Bundle</th><th>Formats</th><th>Source artifact</th><th>SHA-256</th></tr></thead>
					<tbody>
						{#each filtered as d}
							<tr>
								<td><code>{d.id}</code></td><td>{d.title}</td><td>{d.group}</td>
								<td><span class="chip" style={`color:${statusStyle(d.evidence_status).color};border-color:${statusStyle(d.evidence_status).color}55`}>{statusStyle(d.evidence_status).label}</span></td>
								<td>{d.bundle}</td><td>{d.downloadable_formats.join(', ')}</td><td class="path">{d.source_artifact}</td><td><code title={d.source_sha256}>{d.source_sha256.slice(0, 16)}…</code></td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	{/if}
</WorkbenchPage>

<style>
	.state{padding:26px;border:1px solid rgba(148,163,184,.16);color:#8494a9;font:11px 'JetBrains Mono',monospace}.state.error{color:#fb7185;border-color:rgba(251,113,133,.4)}.state.small{padding:12px}
	.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}.tabs button{padding:9px 14px;color:#71829a;background:transparent;border:1px solid rgba(148,163,184,.2);font:8px 'JetBrains Mono',monospace;letter-spacing:.12em;cursor:pointer}.tabs button.on{color:#eef7f6;border-color:rgba(43,184,176,.5);background:rgba(43,184,176,.08)}
	button:focus-visible,input:focus-visible{outline:2px solid #2bb8b0;outline-offset:2px}
	.layout{display:grid;grid-template-columns:minmax(260px,320px) minmax(0,1fr);gap:18px;align-items:start}aside{display:grid;gap:14px;position:sticky;top:76px}
	.checks{list-style:none;margin:0;padding:0;display:grid;gap:2px}.checks label{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;padding:7px 4px;color:#cbd5e1;font-size:12px;cursor:pointer}.checks small{color:#53647b;font:8px 'JetBrains Mono',monospace}.checks li.priv label{color:#c4b5fd}
	.checks input{accent-color:#2bb8b0;width:15px;height:15px}
	.export{display:grid;gap:8px}.export button,.index-tools button{padding:11px 12px;color:#2bb8b0;background:rgba(43,184,176,.06);border:1px solid rgba(43,184,176,.3);font:8px 'JetBrains Mono',monospace;letter-spacing:.12em;cursor:pointer;text-align:left}
	.export button.primary{color:#04121a;background:#2bb8b0;border-color:#2bb8b0}.export button:hover:not(:disabled){filter:brightness(1.15)}.export button:disabled{opacity:.55;cursor:wait}
	.hint{margin:12px 0 0;color:#64748b;font-size:11px;line-height:1.6}.notice{margin:10px 0 0;color:#2bb8b0;font:9px/1.6 'JetBrains Mono',monospace}
	.previews{display:grid;gap:22px;min-width:0}.section h3{margin:0 0 10px;color:#e5f1f0;font:500 19px 'Space Grotesk',sans-serif}.section>:global(*+*){margin-top:12px}
	.section.priv{padding:14px;border:1px dashed rgba(167,139,250,.5)}.banner{margin:0;padding:8px 10px;color:#c4b5fd;background:rgba(167,139,250,.08);font:8px/1.7 'JetBrains Mono',monospace;letter-spacing:.06em}
	.index-tools{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}.index-tools input{flex:1 1 260px;padding:9px 12px;color:#dbe7e7;background:#060c18;border:1px solid rgba(148,163,184,.25);font:11px 'JetBrains Mono',monospace}
	.scroll{overflow:auto;border:1px solid rgba(148,163,184,.12)}table{border-collapse:collapse;width:100%;font:10px 'JetBrains Mono',monospace}th,td{padding:7px 10px;border-bottom:1px solid rgba(148,163,184,.08);text-align:left;color:#b7c6d4;vertical-align:top}th{color:#2bb8b0;font-weight:500;background:#0a1222;position:sticky;top:0}td code{color:#8494a9}.path{color:#64748b;word-break:break-all;max-width:320px}
	.chip{display:inline-block;padding:2px 7px;border:1px solid;font-size:8px;letter-spacing:.06em;white-space:nowrap}
	@media(max-width:900px){.layout{grid-template-columns:1fr}aside{position:static}}
</style>
