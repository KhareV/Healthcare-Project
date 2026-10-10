// Shared types for the research chart kit.
export type EvidenceStatus = 'FINAL' | 'DEVELOPMENT' | 'HISTORICAL' | 'DEFECT' | 'PRIVILEGED';

export type Provenance = {
	status: EvidenceStatus;
	bundle?: string;
	partition?: string;
	n_rows?: number | null;
	n_stays?: number | null;
	metric?: string;
	source?: string;
	sha256?: string;
	ci_method?: string;
};

export type LinePoint = { x: number; y: number; lo?: number; hi?: number; label?: string };
export type LineSeries = {
	key?: string;
	name: string;
	color: string;
	points: LinePoint[];
	dashed?: boolean;
	step?: boolean;
	width?: number;
};
