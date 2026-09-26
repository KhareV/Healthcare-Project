// Performance-V2 API client — Personalized Patient Recovery Trajectory.
// Every call is a genuine HTTP request to the real, model-backed V2 FastAPI
// service (src/serving/v2 + api/v2_app.py). There is no local prediction
// cache or lookup table: each replay step recomputes from raw history
// truncated at the requested cutoff.
import { authToken } from '$lib/stores/auth-token.svelte';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';
const TTS_BASE = import.meta.env.VITE_TTS_BASE_URL || '/tts';

async function request<T>(path: string, options: RequestInit = {}) {
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
	versions: { selected_models_v2_sha256: string; v2_model_freeze_sha256: string };
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
};

export type Condition = {
	condition_id: string;
	label: string;
	diagnosed_year: number | null;
	status: string;
	created_at: string;
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
		request<{ status: string; ready: boolean; mode: string; scope: string; tasks: string[]; auth_mode: string; persistence_mode: 'mongodb' | 'in_memory_only' }>('/health'),
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

	// Persistent longitudinal record layer (src/serving/v2/persistence.py).
	// Durable across API restarts when MongoDB is configured — see
	// /health's persistence_mode; degrades to "whatever is currently
	// loaded in memory" otherwise, never a fake empty/success response.
	listCustomRecords: () => request<{ records: CustomRecordSummary[] }>('/custom-records'),
	getCustomRecordPredictionHistory: (stay_id: string) =>
		request<{ stay_id: string; predictions: PredictionRunSnapshot[]; note?: string }>(`/custom-records/${encodeURIComponent(stay_id)}/predictions`),
	listConditions: () => request<{ conditions: Condition[]; note?: string }>('/health-record/conditions'),
	addCondition: (payload: { label: string; diagnosed_year: number | null; status: string }) =>
		request<Condition>('/health-record/conditions', { method: 'POST', body: JSON.stringify(payload) }),
	deleteCondition: (condition_id: string) =>
		request<{ deleted: boolean; condition_id: string }>(`/health-record/conditions/${encodeURIComponent(condition_id)}`, { method: 'DELETE' }),

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

	// Legacy namespaces kept only so the inherited placeholder pages (outside
	// the active V2 navigation) keep compiling/degrading gracefully; the V2
	// backend does not implement these routes, so calls resolve to a normal
	// "service unavailable" state through the existing SectionPage handling.
	monitoring: {
		getSessions: (params?: Record<string, string>) => request('/monitoring/sessions' + toQuery(params)),
		startSession: (data: unknown) => json('/monitoring/sessions/start', 'POST', data),
		stopSession: (id: string) => json(`/monitoring/sessions/${id}/stop`, 'PUT'),
		getSession: (id: string) => request(`/monitoring/sessions/${id}`),
		startSimulator: (clientId: string) => json('/monitoring/simulate/start', 'POST', { client_id: clientId }),
		stopSimulator: (clientId: string) => json(`/monitoring/simulate/stop/${clientId}`, 'POST'),
		getLiveData: (clientId: string) => request(`/monitoring/live/${clientId}`)
	},
	fl: {
		getStatus: () => request('/fl/status'),
		getRounds: (params?: Record<string, string>) => request('/fl/rounds' + toQuery(params)),
		getRound: (id: string) => request(`/fl/rounds/${id}`),
		startTraining: (data: unknown) => json('/fl/training/start', 'POST', data),
		stopTraining: () => json('/fl/training/stop', 'POST'),
		getClients: () => request('/fl/clients'),
		getClient: (id: string) => request(`/fl/clients/${id}`),
		getAggregation: (roundId: string) => request(`/fl/aggregation/${roundId}`),
		getGlobalModel: () => request('/fl/global-model'),
		getModelHistory: () => request('/fl/global-model/history'),
		getPersonalModels: () => request('/fl/personal-models'),
		getPersonalModel: (userId: string) => request(`/fl/personal-models/${userId}`)
	},
	model: {
		getStatus: () => request('/model/status'),
		inferEcg: (data: unknown) => json('/model/ecg/infer', 'POST', data),
		inferVitals: (data: unknown) => json('/model/vitals/infer', 'POST', data),
		inferSystem: (data: unknown) => json('/model/system/infer', 'POST', data),
		infer: (data: unknown) => json('/model/infer', 'POST', data),
		stream: (sessionId: string, windows: unknown[]) => json(`/model/stream/${sessionId}`, 'POST', { windows }),
		resetStream: (sessionId: string) => json(`/model/stream/${sessionId}/reset`, 'POST', {})
	},
	experiments: {
		getExperiments: () => request('/experiments'),
		createExperiment: (data: unknown) => json('/experiments', 'POST', data),
		getExperiment: (id: string) => request(`/experiments/${id}`),
		runExperiment: (id: string) => json(`/experiments/${id}/run`, 'POST'),
		getResults: (id: string) => request(`/experiments/${id}/results`),
		runRobustness: (data: unknown) => json('/experiments/robustness/run', 'POST', data),
		runAblation: (data: unknown) => json('/experiments/ablation/run', 'POST', data)
	},
	devices: {
		getDevices: () => request('/devices'),
		createDevice: (data: unknown) => json('/devices', 'POST', data),
		getDevice: (id: string) => request(`/devices/${id}`),
		updateDevice: (id: string, data: unknown) => json(`/devices/${id}`, 'PUT', data),
		deleteDevice: (id: string) => request(`/devices/${id}`, { method: 'DELETE' }),
		connectDevice: (id: string) => json(`/devices/${id}/connect`, 'POST'),
		disconnectDevice: (id: string) => json(`/devices/${id}/disconnect`, 'POST')
	},
	alerts: {
		getRules: () => request('/alerts/rules'),
		createRule: (data: unknown) => json('/alerts/rules', 'POST', data),
		updateRule: (id: string, data: unknown) => json(`/alerts/rules/${id}`, 'PUT', data),
		deleteRule: (id: string) => request(`/alerts/rules/${id}`, { method: 'DELETE' }),
		getNotifications: (params?: Record<string, string>) => request('/alerts/notifications' + toQuery(params)),
		markRead: (id: string) => json(`/alerts/notifications/${id}/read`, 'PUT'),
		markAllRead: () => json('/alerts/notifications/read-all', 'PUT'),
		dismiss: (id: string) => request(`/alerts/notifications/${id}`, { method: 'DELETE' })
	},
	reports: {
		generate: (data: unknown) => json('/reports/generate', 'POST', data),
		getReports: () => request('/reports'),
		download: (id: string) => fetch(`${API_BASE}/reports/${id}/download`)
	},
	system: {
		getHealth: () => request('/system/health'),
		getLogs: (params?: Record<string, string>) => request('/system/logs' + toQuery(params)),
		getStats: () => request('/system/stats'),
		getSettings: () => request('/system/settings'),
		updateSettings: (data: unknown) => json('/system/settings', 'PUT', data)
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
