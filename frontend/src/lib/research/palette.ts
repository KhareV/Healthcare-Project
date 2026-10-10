// Consistent colours for model bundles and evidence statuses (matches the existing dark dashboard theme).
import type { EvidenceStatus } from './types';

export type BundleKey = 'V3' | 'V2_AS_SERVED' | 'V2_AS_EVALUATED' | 'CONTROL' | 'NAIVE';

export const BUNDLE_STYLE: Record<BundleKey, { color: string; dash?: string; label: string }> = {
	V3: { color: '#2bb8b0', label: 'V3 (selected release)' },
	V2_AS_SERVED: { color: '#fbbf24', label: 'V2 as served' },
	V2_AS_EVALUATED: { color: '#fb7185', dash: '6 4', label: 'V2 as evaluated (compromised, defect D1)' },
	CONTROL: { color: '#38bdf8', label: 'V2-recipe refit (control)' },
	NAIVE: { color: '#94a3b8', dash: '2 3', label: 'Naive baseline' }
};

export function bundleStyle(key: string) {
	const k = key.toUpperCase().replace(/[^A-Z0-9]+/g, '_') as BundleKey;
	return BUNDLE_STYLE[k] ?? { color: '#a3b3c7', label: key };
}

export const STATUS_STYLE: Record<EvidenceStatus, { color: string; label: string; glyph: string; dashed?: boolean }> = {
	FINAL: { color: '#2bb8b0', label: 'FINAL SEALED TEST', glyph: '●' },
	DEVELOPMENT: { color: '#38bdf8', label: 'DEVELOPMENT', glyph: '◐' },
	HISTORICAL: { color: '#94a3b8', label: 'HISTORICAL', glyph: '◌' },
	DEFECT: { color: '#fb7185', label: 'DEFECT D1 ANALYSIS', glyph: '▲' },
	PRIVILEGED: { color: '#a78bfa', label: 'PRIVILEGED DIAGNOSTIC · NOT DEPLOYABLE', glyph: '◆', dashed: true }
};

// Categorical fallback palette for generic series.
export const SERIES_COLORS = ['#2bb8b0', '#38bdf8', '#fbbf24', '#fb7185', '#a78bfa', '#34d399', '#f97316', '#94a3b8'];
export const AXIS_COLOR = '#64748b';
export const GRID_COLOR = 'rgba(148,163,184,.12)';
export const TEXT_COLOR = '#cbd5e1';
export const MONO = "'JetBrains Mono', ui-monospace, monospace";

export function sequentialColor(t: number): string {
	// dark navy -> teal; t in [0,1]
	const c = Math.max(0, Math.min(1, t));
	const a = [11, 19, 35];
	const b = [43, 184, 176];
	const m = a.map((v, i) => Math.round(v + (b[i] - v) * c));
	return `rgb(${m[0]},${m[1]},${m[2]})`;
}
