// Pure reshaping of frozen cohort-characterization documents into chart props.
// No statistics are computed here: percentile ladders, fractions and counts are passed through as delivered.
export const PCT_KEYS = ['p0', 'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99', 'p100'] as const;
export const PCT_X: Record<string, number> = { p0: 0, p1: 1, p5: 5, p25: 25, p50: 50, p75: 75, p95: 95, p99: 99, p100: 100 };

export const SUBTYPE_LABEL: Record<string, string> = {
	SYNTHETIC_HF: 'Heart failure',
	SYNTHETIC_ISCHEMIC: 'Ischemic',
	SYNTHETIC_ARRHYTHMIA: 'Arrhythmia'
};
export const SUBTYPE_COLOR: Record<string, string> = {
	SYNTHETIC_HF: '#2bb8b0',
	SYNTHETIC_ISCHEMIC: '#fbbf24',
	SYNTHETIC_ARRHYTHMIA: '#a78bfa'
};
export const PARTITION_COLOR: Record<string, string> = {
	DEV_POOL: '#2bb8b0',
	LCE: '#38bdf8',
	V1_DEV: '#fbbf24',
	D1: '#a78bfa',
	D2: '#fb7185',
	D3: '#34d399'
};
export const TASK_LABEL: Record<string, string> = {
	legal_prediction_rows: 'Legal prediction rows',
	icu_time: 'ICU remaining time',
	recovery24: 'Recovery +24h',
	recovery48: 'Recovery +48h',
	organ_support: 'Organ-support initiation'
};
export const DISPOSITION_LABEL: Record<string, string> = {
	INTERNAL_POSITIVE_INITIATION: 'New initiation (positive)',
	INTERNAL_NEGATIVE_FULL_FOLLOWUP: 'No initiation, full 24h follow-up (negative)',
	INTERNAL_CENSORED_EARLY_EXIT_NO_INITIATION: 'Censored: ICU exit before 24h, no initiation',
	INTERNAL_NOT_AT_RISK_BOTH_COMPONENTS_ON: 'Not at risk: both supports already on'
};

export function channelLabel(name: string): string {
	return name.replace(/__latest$/, '').replace(/__sum$/, '').replace(/_/g, ' ');
}

export type Pct = Record<string, number>;

export function ladder(stats: Pct | undefined): { x: number; y: number }[] {
	if (!stats) return [];
	return PCT_KEYS.filter((k) => Number.isFinite(stats[k])).map((k) => ({ x: PCT_X[k], y: stats[k] }));
}

export function partitionLabel(key: string, summary: any): string {
	const n = summary?.cohorts?.[key]?.subjects_generated ?? summary?.cohorts?.[key]?.retained_stays;
	const base: Record<string, string> = {
		DEV_POOL: 'Development pool',
		LCE: 'LCE held-out evaluation cohort',
		V1_DEV: 'V1 train+validation',
		D1: 'Development chunk D1',
		D2: 'Development chunk D2',
		D3: 'Development chunk D3'
	};
	return `${base[key] ?? key}${n ? ` · ${n.toLocaleString()} subjects` : ''}`;
}

export function doc(bundle: any, id: string): any {
	return bundle?.datasets?.[id]?.data?.document ?? bundle?.datasets?.[id]?.data ?? null;
}

export function meanObserved(missing: any, partition: string): number | null {
	const m = missing?.cohorts?.[partition];
	if (!m) return null;
	const vals = Object.values<any>(m)
		.filter((v) => v.channel_type === 'clinical_observation' && Number.isFinite(v.observed_fraction_nonpadded_bins))
		.map((v) => v.observed_fraction_nonpadded_bins as number);
	if (!vals.length) return null;
	return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export function correlationMatrix(corrDoc: any, partition: string) {
	const c = corrDoc?.cohorts?.[partition];
	if (!c) return null;
	const pairs: { a: string; b: string; n_rows: number; pearson_r: number }[] = [...(c.clinically_motivated_pairs ?? []), ...(c.top_abs_pairs ?? [])];
	const seen = new Set<string>();
	const unique = pairs.filter((p) => {
		const k = [p.a, p.b].sort().join('|');
		if (seen.has(k)) return false;
		seen.add(k);
		return true;
	});
	const vars = Array.from(new Set(unique.flatMap((p) => [p.a, p.b]))).sort();
	const idx = new Map(vars.map((v, i) => [v, i]));
	const values: (number | null)[][] = vars.map(() => vars.map(() => null));
	const counts: (number | null)[][] = vars.map(() => vars.map(() => null));
	unique.forEach((p) => {
		const i = idx.get(p.a)!;
		const j = idx.get(p.b)!;
		values[i][j] = values[j][i] = p.pearson_r;
		counts[i][j] = counts[j][i] = p.n_rows;
	});
	vars.forEach((_, i) => (values[i][i] = 1));
	return { vars, values, counts, basis: c.basis as string, evaluated: c.pair_count_evaluated as number };
}
