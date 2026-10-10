// Typed wrappers over the read-only, provenance-stamped /research/* evidence API (V3 product backend).
// Nothing here computes or invents scientific values: every number shown in the research workbench is
// bound to a frozen artifact served by the backend, together with its hash and evidence status.
import { request } from '$lib/services/api';
import type { EvidenceStatus, Provenance } from '$lib/research/types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

export type RawProvenance = {
	id?: string;
	title?: string;
	evidence_status?: string;
	partition?: string;
	n_rows?: number | null;
	n_stays?: number | null;
	metric_definition?: string;
	source_artifact?: string;
	source_sha256?: string;
	bundle?: string;
	units?: Record<string, string>;
	note?: string;
};
export type Wrapped<T> = { provenance: RawProvenance; data: T };
export type Row = Record<string, any>;

const STATUS: Record<string, EvidenceStatus> = {
	FINAL_SEALED_TEST: 'FINAL',
	DEVELOPMENT: 'DEVELOPMENT',
	HISTORICAL: 'HISTORICAL',
	DEFECT_ANALYSIS: 'DEFECT',
	PRIVILEGED_DIAGNOSTIC: 'PRIVILEGED'
};

/** Map a backend provenance block to the chart kit's Provenance (the kit renders chips + source popover). */
export function toProvenance(p: RawProvenance | undefined, extra: Partial<Provenance> = {}): Provenance {
	const status = STATUS[p?.evidence_status ?? ''] ?? 'DEVELOPMENT';
	return {
		status,
		bundle: p?.bundle,
		partition: p?.partition,
		n_rows: p?.n_rows ?? null,
		n_stays: p?.n_stays ?? null,
		metric: p?.metric_definition,
		source: p?.source_artifact,
		sha256: p?.source_sha256,
		...extra
	};
}

export function statusOf(raw?: string): EvidenceStatus {
	return STATUS[raw ?? ''] ?? 'DEVELOPMENT';
}

export const researchApi = {
	summary: () => request<Row>('/research/summary'),
	cohort: () => request<Row>('/research/cohort'),
	curves: () => request<Record<string, Wrapped<Row>>>('/research/curves/support'),
	regression: () => request<Record<string, Wrapped<Row>>>('/research/regression'),
	learningCurve: () => request<Wrapped<Row>>('/research/learning-curve'),
	ablation: () => request<Wrapped<Row>>('/research/ablation'),
	leaderboard: () => request<Wrapped<Row>>('/research/leaderboard'),
	robustness: () => request<Wrapped<Row>>('/research/robustness'),
	globalExplain: () => request<Wrapped<Row>>('/research/explainability/global'),
	defect: () => request<Wrapped<Row>>('/research/defect-d1'),
	diagnostic: () => request<Record<string, any>>('/research/diagnostic'),
	provenance: () => request<Wrapped<Row>>('/research/provenance'),
	dataset: (id: string) => request<Wrapped<Row>>(`/research/datasets/${encodeURIComponent(id)}`),
	index: () => request<Row>('/research/index'),
	exportHref: (id: string, format: 'csv' | 'json' | 'tex') => `${API_BASE}/research/datasets/${encodeURIComponent(id)}/export?format=${format}`
};

export const TASKS = [
	{ value: 'recovery24', label: 'Recovery +24h', unit: 'SOFA points', metric: 'mae', lowerIsBetter: true },
	{ value: 'recovery48', label: 'Recovery +48h', unit: 'SOFA points', metric: 'mae', lowerIsBetter: true },
	{ value: 'icu_stay_time', label: 'ICU remaining time', unit: 'hours', metric: 'weighted_median_ae_hours', lowerIsBetter: true },
	{ value: 'organ_support', label: 'Organ-support initiation', unit: 'AUPRC', metric: 'auprc', lowerIsBetter: false }
] as const;

export type TaskKey = (typeof TASKS)[number]['value'];
export const taskLabel = (t: string) => TASKS.find((x) => x.value === t)?.label ?? t;
export const taskInfo = (t: string) => TASKS.find((x) => x.value === t);

export function errorText(cause: unknown, fallback = 'Could not load the research evidence'): string {
	return cause instanceof Error ? cause.message : fallback;
}
