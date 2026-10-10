// Performance-V2 API client — Personalized Patient Recovery Trajectory.
// Every call is a genuine HTTP request to the real, model-backed V2 FastAPI
// service (src/serving/v2 + api/v2_app.py). There is no local prediction
// cache or lookup table: each replay step recomputes from raw history
// truncated at the requested cutoff.
import { authToken } from '$lib/stores/auth-token.svelte';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';
const TTS_BASE = import.meta.env.VITE_TTS_BASE_URL || '/tts';

export async function request<T>(path: string, options: RequestInit = {}) {
	const headers = new Headers(options.headers);
	if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
	// Attached on every call, not just custom-record ones: harmless for demo
	// predictions (the backend only checks it for CUSTOM- stay ids — see
	// src/serving/v2/auth.py), required for the backend to verify ownership
	// of a custom record.
	if (authToken.value && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${authToken.value}`);

	const response = await fetch(`${API_BASE}${path}`, { ...options, headers });

	if (!response.ok) {
		let detail = `Request failed with status ${response.status}`;
		try {
			const body = await response.json();
			detail = body?.detail || detail;
		} catch {
			/* keep default detail */
		}
		throw new Error(detail);
	}
	return (await response.json()) as T;
}

export type DemoSubject = {
	subject_id: string;
	stay_id: string;
	cardiac_condition_group: string;
	age_years: number;
	sex_category: string;
	intime: string;
	outtime: string;
	legal_cutoffs: string[];
	n_legal_cutoffs: number;
	patient_alias?: string;
};

export type ContributorItem = { feature_name: string; label: string; attribution: number };
export type TaskExplanation = {
	top_positive_contributors: ContributorItem[];
	top_negative_contributors: ContributorItem[];
	method: string;
	diagnostics: {
		additivity_check_passed: boolean;
		base_value: number;
		raw_margin_prediction: number;
		explains: string;
		method: string;
	};
};

export type PredictionResponse = {
	schema_version: string;
	mode: string;
	stay_id: string;
	prediction_time: string;
	grid_index: number;
	elapsed_icu_hours: number;
	current_sofa: number;
	recovery: {
		delta_24h: number;
		delta_48h: number;
		sofa_hat_24h: number;
		sofa_hat_48h: number;
		model_metadata: { family: string; feature_variant: string; artifact_sha256_24h: string; artifact_sha256_48h: string };
	};
	icu_stay_time: {
		remaining_hours: number;
		raw_log_prediction: number;
		model_metadata: { family: string; feature_variant: string; artifact_sha256: string };
	};
	organ_support: {
		raw_probability: number;
		probability_24h: number;
		threshold: number;
		alert: boolean;
		model_metadata: { family: string; feature_variant: string; artifact_sha256: string };
	};
	explanations: Record<'recovery24' | 'recovery48' | 'icu_stay_time' | 'organ_support', TaskExplanation>;
	data_quality: {
		total_bins: number;
		observed_bins: number;
		padding_bins: number;
		total_feature_values: number;
		observed_feature_values: number;
		missing_feature_values: number;
		observed_feature_fraction: number | null;
	};
	temporal_window: { channel_names: string[]; observation_mask: boolean[][]; padding_mask: boolean[] };
	versions: Record<string, string>;
	/** Present on V3 responses: the exact scientific bundle that produced this forecast. */
	model_bundle?: ModelBundleInfo;
	data_revision?: string;
};

export type ModelBundleInfo = {
	bundle_id: string;
	release: 'V3' | 'V2' | string;
	status?: string;
	selected_models_sha256: string;
	tasks: Record<string, { family: string; feature_variant: string; model_sha256: string }>;
	calibrator: { method: string; params_sha256: string };
	threshold: number;
	[key: string]: unknown;
};

export type AIRecommendation = {
	status: 'OK' | 'UNAVAILABLE';
	summary: string | null;
	model: string;
	disclaimer: string;
	error: string | null;
};

export type AssistantTurn = { role: 'user' | 'assistant'; text: string };

export type CanonicalConcept = { concept: string; label: string; hint: string; unit: string; provenance_id: string };

export type CustomObservationInput = { concept: string; hours_since_admission: number; value: number };

export type VasopressorAgent = 'norepinephrine' | 'epinephrine' | 'dopamine' | 'dobutamine';

export type CustomSupportIntervalInput =
	| { kind: 'vasopressor'; agent: VasopressorAgent; rate: number; start_hour: number; end_hour: number | null }
	| { kind: 'ventilation'; start_hour: number; end_hour: number | null };

export type CustomRecordReadinessSummary = {
	observations_entered: number;
	concepts_represented: number;
	concepts_total: number;
	earliest_observation_hour: number | null;
	latest_observation_hour: number | null;
	legal_cutoffs: number;
	first_cutoff_observed_bins: number | null;
	first_cutoff_total_bins: number | null;
	sofa_components_observed: string[];
	sofa_components_missing: string[];
	support_state_entered: { vasopressor: boolean; invasive_ventilation: boolean };
	missing_concepts: string[];
	urine_output_supported: false;
};

export type CustomRecordSummary = {
	stay_id: string;
	patient_alias: string;
	age_years: number;
	sex_category: string;
	created_at: string | null;
	observations_entered: number;
	support_intervals_entered: number;
};

export type PredictionRunSnapshot = {
	stay_id: string;
	prediction_time: string;
	current_sofa: number;
	delta_sofa_24: number;
	delta_sofa_48: number;
	predicted_sofa_24h: number;
	predicted_sofa_48h: number;
	remaining_icu_hours: number;
	support_raw_probability: number;
	support_calibrated_probability: number;
	support_threshold: number;
	support_alert: boolean;
	created_at: string;
	updated_at: string;
	/** Model-version identity (V3 product). Legacy V2 rows carry bundle_id 'performance_v2_frozen', legacy: true. */
	bundle_id?: string;
	release?: 'V3' | 'V2' | string;
	legacy?: boolean;
	/** True when confirmed observations changed the encounter after this forecast was stored. */
	stale?: boolean;
	data_revision?: string;
};

export type PatientProfile = {
	patient_id: string;
	owner_user_id: string;
	display_name_or_alias: string;
	age_years: number | null;
	sex_category: string | null;
	blood_group: string | null;
	height_cm: number | null;
	weight_kg: number | null;
	created_at: string;
	updated_at: string;
	schema_version: string;
};

export type ConditionStatus = 'active' | 'resolved' | 'historical';

export type HealthRecordCondition = {
	condition_id: string;
	patient_id: string | null;
	owner_user_id: string;
	name: string;
	code: string | null;
	diagnosed_date: string | null;
	status: string;
	notes: string | null;
	created_at: string;
	updated_at: string;
	schema_version: string;
};

export type HealthRecordEncounter = {
	encounter_id: string;
	stay_id: string;
	patient_id: string | null;
	patient_alias: string;
	age_years: number;
	sex_category: string;
	intime: string;
	outtime: string;
	encounter_type: string;
	source: string;
	observations: CustomObservationInput[];
	support_intervals: CustomSupportIntervalInput[];
	created_at: string;
	updated_at: string;
};

export type HealthRecordObservationRow = {
	encounter_id: string;
	patient_alias: string;
	concept: string;
	value: number;
	hours_since_admission: number;
};

export type HealthRecordSupportRow = {
	encounter_id: string;
	patient_alias: string;
	kind: string;
	agent: string | null;
	rate: number | null;
	start_hour: number;
	end_hour: number | null;
};

export type ReportProcessingStatus = 'UPLOADED' | 'PARSED' | 'FAILED' | 'NOT_PARSED';

export type ReportCandidateMeasurement = {
	candidate_id: string;
	raw_label: string;
	concept: string | null;
	value: number;
	unit: string | null;
	observed_at: string | null;
	confirmed: boolean;
};

export type HealthRecordReport = {
	report_id: string;
	patient_id: string;
	title: string;
	document_type: string;
	original_filename: string;
	mime_type: string;
	size_bytes: number;
	report_date: string | null;
	uploaded_at: string;
	source: string;
	processing_status: ReportProcessingStatus;
	extracted_summary: string | null;
	candidate_measurements?: ReportCandidateMeasurement[];
};

export type HealthRecordAuditEvent = {
	event_id: string;
	patient_id: string | null;
	action: string;
	target_type: string;
	target_id: string;
	timestamp: string;
};

export type HealthRecordExport = {
	schema_version: string;
	exported_at: string;
	profile: PatientProfile;
	conditions: HealthRecordCondition[];
	encounters: HealthRecordEncounter[];
	reports: HealthRecordReport[];
	prediction_history: PredictionRunSnapshot[];
};

export type CustomRecordResult = {
	stay_id: string;
	subject_id: string;
	patient_alias: string;
	cardiac_condition_group: string;
	age_years: number;
	sex_category: string;
	intime: string;
	outtime: string;
	legal_cutoffs: string[];
	n_legal_cutoffs: number;
	data_readiness: string;
	warnings: string[];
	concept_coverage: { observed: number; total: number; concepts: string[] };
	readiness_summary: CustomRecordReadinessSummary;
	source: string;
};

export const api = {
	request,
	health: () =>
		request<{ status: string; ready: boolean; mode: string; scope: string; tasks: string[]; auth_mode: string; persistence_mode: 'mongodb' | 'in_memory_only'; release?: string; bundle_id?: string; selected_models_sha256?: string; model_families?: Record<string, string>; calibrator?: string; support_threshold?: number }>('/health'),
	modelMetadata: () => request('/model-metadata'),
	demoSubjects: () =>
		request<{ status: string; selection_criteria: Record<string, unknown>; demo_subjects: DemoSubject[] }>('/demo-subjects'),
	predict: (stay_id: string, prediction_time: string) =>
		request<PredictionResponse>('/predict', { method: 'POST', body: JSON.stringify({ stay_id, prediction_time }) }),
	performance: () => request('/performance'),
	history: (stay_id: string, prediction_time: string) =>
		request<{ stay_id: string; prediction_time: string; events: { event_time: string; canonical_concept: string; value_numeric: number; unit: string }[]; count: number }>(
			`/history?stay_id=${encodeURIComponent(stay_id)}&prediction_time=${encodeURIComponent(prediction_time)}`
		),
	aiRecommendation: (stay_id: string, prediction_time: string) =>
		request<AIRecommendation>('/ai/recommendation', { method: 'POST', body: JSON.stringify({ stay_id, prediction_time }) }),

	// Trajectory Copilot — grounded interpretation layer over an
	// already-computed prediction. See src/serving/v2/ai_recommendation.py.
	assistant: (stay_id: string, prediction_time: string, question?: string, previous_prediction_time?: string) =>
		request<AIRecommendation>('/assistant', {
			method: 'POST',
			body: JSON.stringify({ stay_id, prediction_time, question: question ?? null, previous_prediction_time: previous_prediction_time ?? null })
		}),

	// "Bring your own data" direct entry — a real ephemeral, session-only
	// stay served through the exact same /predict, /history, /ai/*
	// endpoints as any demo subject. See src/serving/v2/custom_record.py.
	customRecordSchema: () => request<{ concepts: CanonicalConcept[] }>('/custom-records/schema'),
	createCustomRecord: (payload: { patient_alias: string; age_years: number; sex_category: string; observations: CustomObservationInput[]; support_intervals?: CustomSupportIntervalInput[] }) =>
		request<CustomRecordResult>('/custom-records', { method: 'POST', body: JSON.stringify(payload) }),
	getCustomRecord: (stay_id: string) => request<CustomRecordResult>(`/custom-records/${encodeURIComponent(stay_id)}`),

	// Durable across API restarts when MongoDB is configured — see
	// /health's persistence_mode; degrades to "whatever is currently
	// loaded in memory" otherwise, never a fake empty/success response.
	listCustomRecords: () => request<{ records: CustomRecordSummary[] }>('/custom-records'),
	getCustomRecordPredictionHistory: (stay_id: string) =>
		request<{ stay_id: string; predictions: PredictionRunSnapshot[]; note?: string }>(`/custom-records/${encodeURIComponent(stay_id)}/predictions`),

	// My Health Record (src/serving/v2/persistence.py + report_storage.py):
	// one consolidated, owner-scoped longitudinal record -- profile,
	// conditions, encounters/observations/support (read views over the same
	// encounters custom-record entry creates), reports, prediction history.
	healthRecord: {
		getProfile: () => request<PatientProfile>('/health-record/profile'),
		updateProfile: (
			fields: Partial<Pick<PatientProfile, 'display_name_or_alias' | 'age_years' | 'sex_category' | 'blood_group' | 'height_cm' | 'weight_kg'>>
		) => request<PatientProfile>('/health-record/profile', { method: 'PUT', body: JSON.stringify(fields) }),

		listConditions: () => request<{ conditions: HealthRecordCondition[]; note?: string }>('/health-record/conditions'),
		addCondition: (payload: { name: string; code?: string | null; diagnosed_date?: string | null; status: ConditionStatus; notes?: string | null }) =>
			request<HealthRecordCondition>('/health-record/conditions', { method: 'POST', body: JSON.stringify(payload) }),
		editCondition: (
			condition_id: string,
			payload: Partial<{ name: string; code: string | null; diagnosed_date: string | null; status: ConditionStatus; notes: string | null }>
		) => request<HealthRecordCondition>(`/health-record/conditions/${encodeURIComponent(condition_id)}`, { method: 'PATCH', body: JSON.stringify(payload) }),
		deleteCondition: (condition_id: string) =>
			request<{ deleted: boolean; condition_id: string }>(`/health-record/conditions/${encodeURIComponent(condition_id)}`, { method: 'DELETE' }),

		listEncounters: () => request<{ encounters: HealthRecordEncounter[]; note?: string }>('/health-record/encounters'),
		getEncounter: (encounter_id: string) => request<HealthRecordEncounter>(`/health-record/encounters/${encodeURIComponent(encounter_id)}`),

		listObservations: () => request<{ observations: HealthRecordObservationRow[]; note?: string }>('/health-record/observations'),
		listSupport: () => request<{ support_intervals: HealthRecordSupportRow[]; note?: string }>('/health-record/support'),
		listPredictions: () => request<{ predictions: PredictionRunSnapshot[]; note?: string }>('/health-record/predictions'),

		listReports: () => request<{ reports: HealthRecordReport[]; note?: string }>('/health-record/reports'),
		getReport: (report_id: string) => request<HealthRecordReport>(`/health-record/reports/${encodeURIComponent(report_id)}`),
		uploadReport: async (payload: { title: string; document_type: string; report_date?: string | null; file: File }): Promise<HealthRecordReport> => {
			const form = new FormData();
			form.set('title', payload.title);
			form.set('document_type', payload.document_type);
			if (payload.report_date) form.set('report_date', payload.report_date);
			form.set('file', payload.file);
			const headers = new Headers();
			if (authToken.value) headers.set('Authorization', `Bearer ${authToken.value}`);
			// Deliberately not the shared request() helper: it defaults an
			// unset Content-Type to application/json, which would break the
			// browser's own multipart/form-data boundary for a FormData body.
			const response = await fetch(`${API_BASE}/health-record/reports`, { method: 'POST', body: form, headers });
			if (!response.ok) {
				let detail = `Upload failed with status ${response.status}`;
				try {
					const body = await response.json();
					detail = body?.detail || detail;
				} catch {
					/* keep default detail */
				}
				throw new Error(detail);
			}
			return response.json();
		},
		downloadReport: async (report_id: string): Promise<Blob> => {
			const headers = new Headers();
			if (authToken.value) headers.set('Authorization', `Bearer ${authToken.value}`);
			// A plain <a href> can't attach an Authorization header, so the
			// download is fetched here and handed to the caller as a Blob to
			// save via a temporary object URL -- see the Reports panel.
			const response = await fetch(`${API_BASE}/health-record/reports/${encodeURIComponent(report_id)}/download`, { headers });
			if (!response.ok) throw new Error(`Download failed with status ${response.status}`);
			return response.blob();
		},
		deleteReport: (report_id: string) =>
			request<{ deleted: boolean; report_id: string }>(`/health-record/reports/${encodeURIComponent(report_id)}`, { method: 'DELETE' }),

		// Report Intelligence: parse proposes candidates (never inserted
		// automatically); confirm is the human-in-the-loop step that turns
		// selected candidates into real observations on a named encounter.
		parseReport: (report_id: string) => request<HealthRecordReport>(`/health-record/reports/${encodeURIComponent(report_id)}/parse`, { method: 'POST' }),
		confirmReportMeasurements: (report_id: string, payload: { encounter_id: string; confirmations: { candidate_id: string; hours_since_admission: number }[] }) =>
			request<{ added: number; encounter: CustomRecordResult }>(`/health-record/reports/${encodeURIComponent(report_id)}/confirm`, { method: 'POST', body: JSON.stringify(payload) }),

		exportRecord: () => request<HealthRecordExport>('/health-record/export'),
		listAuditEvents: () => request<{ events: HealthRecordAuditEvent[]; note?: string }>('/health-record/audit-events')
	},

	// Kokoro narration microservice — a separate isolated process (see
	// tts/server.py); returns a playable audio/wav Blob or throws.
	speak: async (text: string, voice = 'af_heart'): Promise<Blob> => {
		const response = await fetch(`${TTS_BASE}/speak`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text, voice })
		});
		if (!response.ok) {
			let detail = `Narration failed with status ${response.status}`;
			try {
				const body = await response.json();
				detail = body?.detail || detail;
			} catch {
				/* keep default detail */
			}
			throw new Error(detail);
		}
		return response.blob();
	},

	// Legacy ECG/federated/device clients were removed in the V3 product release.
	reports: {
		generate: (data: unknown) => json('/reports/generate', 'POST', data),
		getReports: () => request('/reports'),
		download: (id: string) => fetch(`${API_BASE}/reports/${id}/download`)
	}
};

function toQuery(params?: Record<string, string>) {
	if (!params) return '';
	const query = new URLSearchParams(params).toString();
	return query ? `?${query}` : '';
}

function json<T>(path: string, method: string, body?: unknown) {
	return request<T>(path, { method, body: body === undefined ? undefined : JSON.stringify(body) });
}

export { API_BASE };
