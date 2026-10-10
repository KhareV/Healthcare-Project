// Typed wrappers over the read-only, provenance-stamped research API (api/research_router.py).
// Used by the Cohort Analytics and Research Report pages. Every value shown comes from these
// endpoints; nothing is computed or invented client-side beyond unit-preserving reshaping.
import { request } from '$lib/services/api';
import type { EvidenceStatus, Provenance } from '$lib/research/types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

export type DatasetProvenance = {
	id: string;
	title: string;
	group: string;
	evidence_status: string;
	partition: string;
	n_rows: number | null;
	n_stays: number | null;
	metric_definition: string;
	source_artifact: string;
	source_sha256: string;
	bundle: string;
	chart_hint: string;
	units: Record<string, string>;
	note?: string;
	downloadable_formats: string[];
	api_version?: string;
};

export type ResearchDataset<T = any> = { provenance: DatasetProvenance; data: T };

export type ResearchIndex = { api_version: string; count: number; datasets: DatasetProvenance[]; status_legend?: Record<string, string> };

export type CohortBundle = {
	api_version: string;
	evidence_status: string;
	datasets: Record<string, ResearchDataset>;
	missing: string[];
	partitions: string;
};

export const cohortApi = {
	index: () => request<ResearchIndex>('/research/index'),
	cohort: () => request<CohortBundle>('/research/cohort'),
	dataset: <T = any>(id: string) => request<ResearchDataset<T>>(`/research/datasets/${encodeURIComponent(id)}`),
	summary: () => request<any>('/research/summary'),
	/** Server-formatted export (csv | json | tex) fetched as text and handed to a client-side download. */
	exportText: async (id: string, format: 'csv' | 'json' | 'tex'): Promise<string> => {
		const response = await fetch(`${API_BASE}/research/datasets/${encodeURIComponent(id)}/export?format=${format}`);
		if (!response.ok) throw new Error(`export failed (${response.status}) for ${id}`);
		return await response.text();
	}
};

export function toKitStatus(status: string): EvidenceStatus {
	if (status.startsWith('FINAL')) return 'FINAL';
	if (status.startsWith('HISTORICAL')) return 'HISTORICAL';
	if (status.startsWith('DEFECT')) return 'DEFECT';
	if (status.startsWith('PRIVILEGED')) return 'PRIVILEGED';
	return 'DEVELOPMENT';
}

export function toKitProvenance(p: DatasetProvenance, extra: Partial<Provenance> = {}): Provenance {
	return {
		status: toKitStatus(p.evidence_status),
		bundle: p.bundle && p.bundle !== 'n/a' ? p.bundle : undefined,
		partition: p.partition,
		n_rows: p.n_rows,
		n_stays: p.n_stays,
		metric: p.metric_definition,
		source: p.source_artifact,
		sha256: p.source_sha256,
		...extra
	};
}
