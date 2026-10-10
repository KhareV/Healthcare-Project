// Scenario Studio API wrappers. Every number shown in the studio comes from real
// endpoints: /showcase/* (structured, observed facts) and POST /predict (V3 models).
import { api, request, type DemoSubject, type PredictionResponse } from '$lib/services/api';

export type KeyEvent = { type: string; hours_since_admission: number; description: string };

export type Scenario = {
	scenario_id: string;
	patient_alias: string;
	category: string;
	title: string;
	subtype: string;
	objective: string;
	source_cohort: string;
	stay_id: string;
	subject_id: string;
	age_years: number;
	sex_category: string;
	n_cutoffs: number;
	legal_cutoffs: string[];
	selection_rationale: string;
	key_event: KeyEvent;
	analytical_focus: string;
	status: string;
	registered_for_inference: boolean;
	measured_criteria?: Record<string, unknown>;
};

export type ScenarioList = {
	status: string;
	source_cohort: string;
	selection_policy: Record<string, unknown>;
	substitutions: unknown[];
	manifest_sha256: string;
	scenarios: Scenario[];
};

export type TimelinePoint = { hours_since_admission: number; value: number; first_visible_cutoff: number | null };
export type TimelineSeries = { concept: string; unit: string; points: TimelinePoint[] };
export type SupportRun = {
	support_type: string;
	category: string;
	qualifying_for_initiation: boolean;
	start_hour: number;
	end_hour: number;
	mean_rate: number | null;
	rate_unit: string | null;
	first_visible_cutoff: number | null;
	end_first_visible_cutoff: number | null;
};
export type Timeline = {
	scenario_id: string;
	stay_id: string;
	intime: string;
	outtime_hours: number;
	cutoff_hours: number[];
	legal_cutoffs: string[];
	visibility_rule: string;
	series: TimelineSeries[];
	support_runs: SupportRun[];
	observed_sofa_by_cutoff: number[];
};

export type StoryFact = { kind: string; text: string; concept?: string; hindsight?: boolean; hours_since_admission?: number };
export type StoryStep = {
	cutoff_index: number;
	prediction_time: string;
	facts: StoryFact[];
	revealed_next: StoryFact[];
	revealed_next_label: string;
};
export type Story = { scenario_id: string; title: string; objective: string; steps: StoryStep[]; note: string };

export type CompareRow = {
	cutoff_index: number;
	a: { prediction_time: string; icu_elapsed_hours: number; sofa_t: number; observed_bins: number; observation_density: number };
	b: { prediction_time: string; icu_elapsed_hours: number; sofa_t: number; observed_bins: number; observation_density: number };
};
export type Compare = { a: Record<string, unknown>; b: Record<string, unknown>; aligned_cutoffs: number; rows: CompareRow[] };

/** Unified replay participant: a manifest scenario or one of the three original demo patients. */
export type Participant = {
	key: string;
	kind: 'scenario' | 'demo';
	title: string;
	alias: string;
	subtype: string;
	stayId: string;
	legalCutoffs: string[];
	intime?: string;
	objective?: string;
	focus?: string;
	keyEvent?: string;
	cohortLabel: string;
	scenario?: Scenario;
};

export const showcase = {
	scenarios: () => request<ScenarioList>('/showcase/scenarios'),
	timeline: (id: string) => request<Timeline>(`/showcase/scenarios/${encodeURIComponent(id)}/timeline`),
	story: (id: string) => request<Story>(`/showcase/scenarios/${encodeURIComponent(id)}/story`),
	compare: (a: string, b: string) => request<Compare>(`/showcase/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`)
};

export function scenarioParticipant(s: Scenario): Participant {
	return {
		key: s.scenario_id,
		kind: 'scenario',
		title: s.title,
		alias: s.patient_alias,
		subtype: s.subtype,
		stayId: s.stay_id,
		legalCutoffs: s.legal_cutoffs,
		objective: s.objective,
		focus: s.analytical_focus,
		keyEvent: s.key_event?.description,
		cohortLabel: 'Held-out development cohort',
		scenario: s
	};
}

export function demoParticipant(d: DemoSubject, index: number): Participant {
	return {
		key: d.stay_id,
		kind: 'demo',
		title: d.patient_alias ?? `Training-cohort demo ${index + 1}`,
		alias: d.patient_alias ?? d.subject_id,
		subtype: d.cardiac_condition_group,
		stayId: d.stay_id,
		legalCutoffs: d.legal_cutoffs,
		intime: d.intime,
		cohortLabel: 'Training-cohort demonstration',
		objective: 'Original three-patient demonstration; drawn from the model-development cohort.'
	};
}

export function prettySubtype(value: string): string {
	return value.replace('SYNTHETIC_', '').replace(/_/g, ' ');
}

export function hhmm(iso: string): string {
	return iso.slice(11, 16);
}

export async function bundleKey(): Promise<string> {
	try {
		const h = await request<{ bundle_id?: string; release?: string }>('/health');
		return `${h.release ?? 'NA'}:${h.bundle_id ?? 'unknown'}`;
	} catch {
		return 'unknown';
	}
}

/** Build a visible-only timeline for original demo patients from /history (events <= t only). */
export async function timelineFromHistory(p: Participant, cutoffIndex: number): Promise<Timeline | null> {
	if (!p.intime) return null;
	const t = p.legalCutoffs[cutoffIndex];
	const res = await api.history(p.stayId, t);
	const base = new Date(p.intime).getTime();
	const by = new Map<string, TimelineSeries>();
	for (const e of res.events) {
		const h = (new Date(e.event_time).getTime() - base) / 3_600_000;
		const s = by.get(e.canonical_concept) ?? { concept: e.canonical_concept, unit: e.unit, points: [] };
		s.points.push({ hours_since_admission: h, value: e.value_numeric, first_visible_cutoff: cutoffIndex });
		by.set(e.canonical_concept, s);
	}
	return {
		scenario_id: p.key,
		stay_id: p.stayId,
		intime: p.intime,
		outtime_hours: NaN,
		cutoff_hours: p.legalCutoffs.map((c) => (new Date(c).getTime() - base) / 3_600_000),
		legal_cutoffs: p.legalCutoffs,
		visibility_rule: 'events up to the selected cutoff only (GET /history)',
		series: [...by.values()],
		support_runs: [],
		observed_sofa_by_cutoff: []
	};
}

export type { DemoSubject, PredictionResponse };
