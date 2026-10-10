// Client-side research report assembly (Markdown / HTML / LaTeX). Pure functions over API payloads:
// every number comes from /research/* responses; no scientific text is synthesised here beyond
// section headings and verbatim caveat strings delivered by the API.
import type { DatasetProvenance, ResearchDataset } from '$lib/services/cohort';

export type SectionDef = { key: string; title: string; ids: string[]; privileged?: boolean; defaultOn: boolean };

export const SECTIONS: SectionDef[] = [
	{ key: 'cohort', title: 'Cohort characterization', defaultOn: true, ids: ['cohort_summary', 'cohort_subtype', 'cohort_age_sex', 'cohort_duration', 'cohort_eligibility', 'cohort_outcome_prevalence', 'cohort_missingness_feature', 'cohort_missingness_bin', 'cohort_sofa_distributions', 'cohort_sofa_by_cutoff', 'cohort_cutoffs_per_stay', 'cohort_correlations', 'cohort_between_subject'] },
	{ key: 'comparison', title: 'Model comparisons', defaultOn: true, ids: ['model_comparison', 'leaderboard', 'historical_context'] },
	{ key: 'forecast', title: 'Forecast metrics', defaultOn: true, ids: ['regression_metrics', 'classification_metrics'] },
	{ key: 'classification', title: 'Classification curves', defaultOn: true, ids: ['support_roc', 'support_pr', 'support_score_distribution', 'support_confusion', 'support_threshold_sensitivity'] },
	{ key: 'calibration', title: 'Calibration', defaultOn: true, ids: ['support_reliability'] },
	{ key: 'regression', title: 'Regression analysis', defaultOn: true, ids: ['regression_residuals', 'regression_predicted_vs_observed'] },
	{ key: 'learning', title: 'Learning curves', defaultOn: true, ids: ['learning_curve'] },
	{ key: 'ablation', title: 'Feature ablations', defaultOn: true, ids: ['ablation'] },
	{ key: 'robustness', title: 'Robustness', defaultOn: true, ids: ['robustness'] },
	{ key: 'error', title: 'Error analysis', defaultOn: true, ids: ['error_slices'] },
	{ key: 'explain', title: 'Explainability', defaultOn: true, ids: ['explainability_global'] },
	{ key: 'paired', title: 'Paired V2/V3 evaluation', defaultOn: true, ids: ['paired_differences'] },
	{ key: 'defect', title: 'Defect D1 disclosure', defaultOn: true, ids: ['defect_d1'] },
	{ key: 'diagnostic', title: 'Latent-state diagnostic (PRIVILEGED — not deployable)', defaultOn: false, privileged: true, ids: ['diagnostic_latent_state', 'diagnostic_candidate_development'] }
];

export const ALWAYS_IDS = ['provenance'];
const CAVEAT_KEYS = ['caveat', 'note', 'statement', 'explanation', 'limitation', 'interpretation_wording', 'reading_guide', 'value_meaning', 'weighting', 'interval_method', 'ci_method', 'output_domain', 'primary_comparator'];
export const MAX_TABLE_ROWS = 80;

export type FlatTable = { columns: string[]; rows: Record<string, unknown>[]; total: number; omitted: boolean; reason: string };

export function cell(v: unknown): string {
	if (v === null || v === undefined) return '';
	if (typeof v === 'number') return Number.isInteger(v) ? String(v) : Math.abs(v) >= 1000 ? v.toFixed(1) : String(Number(v.toPrecision(5)));
	if (typeof v === 'boolean') return v ? 'true' : 'false';
	if (Array.isArray(v)) return '[' + v.map(cell).join(', ') + ']';
	if (typeof v === 'object') return JSON.stringify(v);
	return String(v);
}

export function toTable(ds: ResearchDataset): FlatTable {
	const data = ds.data ?? {};
	let rows: Record<string, unknown>[] = Array.isArray(data.rows) ? data.rows : [];
	if (!rows.length && data.metric_by_model && typeof data.metric_by_model === 'object') {
		rows = Object.entries<any>(data.metric_by_model).map(([model, v]) => ({ model, ...(typeof v === 'object' ? v : { value: v }) }));
	}
	if (!rows.length && data.document && typeof data.document === 'object') {
		// structured characterization documents are summarised one scalar leaf per row (path, value)
		const out: Record<string, unknown>[] = [];
		const walk = (o: any, path: string[]) => {
			if (out.length > 600) return;
			if (o && typeof o === 'object') Object.entries(o).forEach(([k, v]) => walk(v, [...path, k]));
			else out.push({ path: path.join(' / '), value: o });
		};
		walk(data.document, []);
		rows = out;
	}
	const total = rows.length;
	const cols = Array.from(new Set(rows.flatMap((r) => Object.keys(r))));
	const isCurve = ds.provenance.chart_hint === 'line' && total > MAX_TABLE_ROWS;
	if (total > MAX_TABLE_ROWS && (isCurve || ds.provenance.group === 'cohort')) {
		const summary = data.metric_by_model && typeof data.metric_by_model === 'object' ? toTable({ ...ds, data: { metric_by_model: data.metric_by_model } }) : null;
		if (summary && summary.rows.length) return { ...summary, total, omitted: true, reason: `Full curve (${total} points) is available through the CSV/JSON export; summary metrics shown.` };
		return { columns: cols, rows: rows.slice(0, 40), total, omitted: true, reason: `First 40 of ${total} rows shown; use the CSV/JSON export for all rows.` };
	}
	return { columns: cols, rows: rows.slice(0, 400), total, omitted: total > 400, reason: total > 400 ? `First 400 of ${total} rows; full table via CSV export.` : '' };
}

export function caveats(ds: ResearchDataset): string[] {
	const out: string[] = [];
	for (const k of CAVEAT_KEYS) {
		const v = ds.data?.[k];
		if (typeof v === 'string' && v.trim()) out.push(v.trim());
	}
	if (ds.provenance.note) out.push(ds.provenance.note);
	return Array.from(new Set(out));
}

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const mdEsc = (s: string) => s.replace(/\|/g, '\\|').replace(/\n/g, ' ');

export type ReportInput = {
	generatedAt: string;
	apiVersion: string;
	bundle: { bundle_id?: string; status?: string; threshold?: unknown; calibrator?: unknown; model_sha256?: Record<string, string> } | null;
	sections: { def: SectionDef; datasets: ResearchDataset[] }[];
	always: ResearchDataset[];
	defect: ResearchDataset | null;
};

function provLines(p: DatasetProvenance): [string, string][] {
	return [
		['Dataset id', p.id], ['Evidence status', p.evidence_status], ['Partition', p.partition], ['Bundle', p.bundle],
		['Rows / stays', `${p.n_rows ?? 'n/a'} / ${p.n_stays ?? 'n/a'}`], ['Metric definition', p.metric_definition],
		['Units', Object.entries(p.units ?? {}).map(([k, v]) => `${k}: ${v}`).join('; ') || 'n/a'],
		['Source artifact', p.source_artifact], ['Source SHA-256', p.source_sha256]
	];
}

function header(input: ReportInput): [string, string][] {
	const b = input.bundle;
	return [
		['Generated at (UTC)', input.generatedAt], ['Research API version', input.apiVersion],
		['Serving bundle', b?.bundle_id ?? 'see provenance dataset'], ['Bundle status', b?.status ?? 'n/a'],
		['Frozen support threshold', cell(b?.threshold)], ['Calibrator', cell((b?.calibrator as any)?.method ?? b?.calibrator)]
	];
}

const STANDING_CAVEATS = [
	'Defect D1: frozen V2 models used the missingness of current SOFA, which encodes whether 24 h of follow-up exists. Results labelled V2_AS_EVALUATED carry that leakage and are historical; V2_AS_SERVED is the deployed V2 behaviour; the V2-recipe refit is the leakage-repaired control.',
	'Confidence intervals are stay-cluster bootstrap intervals for aggregate performance, not individual prediction intervals.',
	'Synthetic cohort; descriptive and model-evaluation evidence only — not clinical validation, treatment advice or real-time performance.'
];

export function buildMarkdown(input: ReportInput): string {
	const L: string[] = ['# Personalized Patient Recovery Trajectory — research evidence report', ''];
	L.push('| Field | Value |', '|---|---|', ...header(input).map(([k, v]) => `| ${k} | ${mdEsc(v)} |`), '');
	L.push('## Standing caveats', '', ...STANDING_CAVEATS.map((c) => `- ${c}`));
	if (input.defect) caveats(input.defect).forEach((c) => L.push(`- (from API, defect_d1) ${c}`));
	L.push('');
	const blocks = [...input.sections.map((s) => ({ title: s.def.title, privileged: s.def.privileged, datasets: s.datasets })), { title: 'Provenance', privileged: false, datasets: input.always }];
	for (const sec of blocks) {
		L.push(`## ${sec.title}`, '');
		if (sec.privileged) L.push('> **PRIVILEGED DIAGNOSTIC — simulator-state predictors; not deployable, not a clinical bound, never a serving input.**', '');
		for (const ds of sec.datasets) {
			const t = toTable(ds);
			L.push(`### ${ds.provenance.title}`, '', ...provLines(ds.provenance).map(([k, v]) => `- **${k}:** ${mdEsc(String(v))}`), '');
			caveats(ds).forEach((c) => L.push(`> ${mdEsc(c)}`));
			if (caveats(ds).length) L.push('');
			if (t.rows.length) {
				L.push(`| ${t.columns.map(mdEsc).join(' | ')} |`, `|${t.columns.map(() => '---').join('|')}|`);
				t.rows.forEach((r) => L.push(`| ${t.columns.map((c) => mdEsc(cell(r[c]))).join(' | ')} |`));
				L.push('');
			}
			if (t.reason) L.push(`_${t.reason}_`, '');
		}
	}
	return L.join('\n');
}

export function buildHtml(input: ReportInput): string {
	const css = 'body{font:14px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 18px;color:#0f172a}table{border-collapse:collapse;margin:8px 0 18px;font-size:12px;width:100%}th,td{border:1px solid #cbd5e1;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f1f5f9}h1,h2,h3{font-family:Georgia,serif}dl{display:grid;grid-template-columns:200px 1fr;gap:2px 12px;font-size:12px}dt{color:#475569}blockquote{margin:8px 0;padding:6px 12px;border-left:3px solid #f59e0b;background:#fffbeb}.priv{border-left-color:#7c3aed;background:#f5f3ff}.small{font-size:11px;color:#64748b}';
	const H: string[] = [`<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Research evidence report</title><style>${css}</style></head><body>`];
	H.push('<h1>Personalized Patient Recovery Trajectory — research evidence report</h1><dl>');
	header(input).forEach(([k, v]) => H.push(`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`));
	H.push('</dl><h2>Standing caveats</h2><ul>');
	STANDING_CAVEATS.forEach((c) => H.push(`<li>${esc(c)}</li>`));
	if (input.defect) caveats(input.defect).forEach((c) => H.push(`<li>${esc(c)} <span class="small">(from API: defect_d1)</span></li>`));
	H.push('</ul>');
	const blocks = [...input.sections.map((s) => ({ title: s.def.title, privileged: s.def.privileged, datasets: s.datasets })), { title: 'Provenance', privileged: false, datasets: input.always }];
	for (const sec of blocks) {
		H.push(`<h2>${esc(sec.title)}</h2>`);
		if (sec.privileged) H.push('<blockquote class="priv"><b>PRIVILEGED DIAGNOSTIC</b> — simulator-state predictors; not deployable, not a clinical bound, never a serving input.</blockquote>');
		for (const ds of sec.datasets) {
			const t = toTable(ds);
			H.push(`<h3>${esc(ds.provenance.title)}</h3><dl>`);
			provLines(ds.provenance).forEach(([k, v]) => H.push(`<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`));
			H.push('</dl>');
			caveats(ds).forEach((c) => H.push(`<blockquote>${esc(c)}</blockquote>`));
			if (t.rows.length) {
				H.push('<table><thead><tr>' + t.columns.map((c) => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>');
				t.rows.forEach((r) => H.push('<tr>' + t.columns.map((c) => `<td>${esc(cell(r[c]))}</td>`).join('') + '</tr>'));
				H.push('</tbody></table>');
			}
			if (t.reason) H.push(`<p class="small"><i>${esc(t.reason)}</i></p>`);
		}
	}
	H.push('</body></html>');
	return H.join('\n');
}

/** Combined LaTeX document: server-formatted booktabs tables (export endpoint) + provenance comments. */
export function buildTex(input: ReportInput, texById: Record<string, string>): string {
	const T = (s: string) => s.replace(/[\\{}$&#_%^~]/g, (m) => ({ '\\': '\\textbackslash{}', '{': '\\{', '}': '\\}', $: '\\$', '&': '\\&', '#': '\\#', _: '\\_', '%': '\\%', '^': '\\^{}', '~': '\\~{}' } as Record<string, string>)[m]);
	const L: string[] = ['\\documentclass[10pt]{article}', '\\usepackage[landscape,margin=1.5cm]{geometry}', '\\usepackage{booktabs,array,graphicx}', '\\usepackage[T1]{fontenc}', '\\begin{document}',
		'\\title{Personalized Patient Recovery Trajectory --- research evidence report}', `\\date{${T(input.generatedAt)}}`, '\\maketitle'];
	L.push('\\section*{Standing caveats}', '\\begin{itemize}', ...STANDING_CAVEATS.map((c) => `\\item ${T(c)}`), '\\end{itemize}');
	const blocks = [...input.sections.map((s) => ({ title: s.def.title, datasets: s.datasets })), { title: 'Provenance', datasets: input.always }];
	for (const sec of blocks) {
		L.push(`\\section{${T(sec.title)}}`);
		for (const ds of sec.datasets) {
			const p = ds.provenance;
			L.push(`\\subsection{${T(p.title)}}`, `\\noindent\\small Status: ${T(p.evidence_status)}; partition: ${T(p.partition)}; source: \\texttt{${T(p.source_artifact)}}; SHA-256: \\texttt{${T(p.source_sha256.slice(0, 16))}\\ldots}\\normalsize\\par\\smallskip`);
			const tex = texById[p.id];
			if (tex) L.push('\\begin{center}\\resizebox{\\linewidth}{!}{%', tex.replace(/\\begin\{table\}\[[^\]]*\]|\\end\{table\}|\\centering/g, ''), '}\\end{center}');
			else L.push('\\noindent\\emph{Table omitted from the combined document (long curve or document-shaped evidence); see the CSV export.}');
		}
	}
	L.push('\\end{document}');
	return L.join('\n');
}

export function indexCsv(items: DatasetProvenance[]): string {
	const cols = ['id', 'title', 'group', 'evidence_status', 'bundle', 'partition', 'metric_definition', 'source_artifact', 'source_sha256', 'downloadable_formats'];
	const q = (v: unknown) => `"${cell(v).replace(/"/g, '""')}"`;
	return [cols.join(','), ...items.map((i) => cols.map((c) => q((i as any)[c])).join(','))].join('\n');
}
