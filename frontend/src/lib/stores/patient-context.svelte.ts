// Shared patient / cutoff / model-bundle context for the patient-focused pages
// (Patient Replay, Forecast Details, AI + SHAP). One rune-based singleton so the
// three pages stay synchronised and so a forecast already computed is never
// recomputed or re-labelled by another page.
//
// Integrity rules:
//  * Predictions are cached per stay, and every cached entry carries the bundle id
//    and data revision of the response that produced it. A response whose bundle or
//    data revision differs from the cached one invalidates that stay's entries, so a
//    forecast made before a report confirmation (or under another model bundle) is
//    never presented as current.
//  * The cache is cleared whenever the signed-in identity changes (no cross-user
//    leakage); custom records are always re-fetched under the new identity.
//  * Every value comes from the real /predict and /history endpoints.
import { api, request, type DemoSubject, type PredictionResponse, type ModelBundleInfo } from '$lib/services/api';
import { authToken } from '$lib/stores/auth-token.svelte';
import { setCopilotContext } from '$lib/stores/copilot.svelte';

export type HistoryEvent = { event_time: string; canonical_concept: string; value_numeric: number; unit: string };

export type WorkspacePatient = DemoSubject & {
	kind: 'demo' | 'showcase' | 'custom';
	title?: string;
	scenario_id?: string;
	objective?: string;
};

type ShowcaseCard = {
	scenario_id: string; patient_alias: string; title: string; subtype: string; objective?: string;
	stay_id: string; subject_id: string; age_years: number; sex_category: string; legal_cutoffs: string[]; n_cutoffs: number;
};

type StayCache = { bundleId: string; revision: string; byCutoff: Map<string, PredictionResponse> };

function tokenIdentity(token: string | null): string {
	if (!token) return 'anonymous';
	try {
		const payload = token.split('.')[1];
		const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
		return String(JSON.parse(json).sub ?? 'authenticated');
	} catch {
		return 'authenticated';
	}
}

const HOUR_MS = 3_600_000;

class PatientContext {
	subjects = $state<WorkspacePatient[]>([]);
	stayId = $state('');
	cutoffIndex = $state(0);
	prediction = $state<PredictionResponse | null>(null);
	/** Predictions for every computed cutoff of the selected stay, keyed by cutoff index. */
	computed = $state<Record<number, PredictionResponse>>({});
	events = $state<HistoryEvent[]>([]);
	loadingSubjects = $state(true);
	loading = $state(false);
	error = $state('');
	replaying = $state(false);
	replaySpeed = $state(1);
	showHindsight = $state(false);

	private owner = '';
	private caches = new Map<string, StayCache>();
	private inflight = new Map<string, Promise<PredictionResponse>>();
	private timer: ReturnType<typeof setInterval> | null = null;
	private token = 0;

	get subject(): WorkspacePatient | null {
		return this.subjects.find((s) => s.stay_id === this.stayId) ?? null;
	}
	get cutoffs(): string[] {
		return this.subject?.legal_cutoffs ?? [];
	}
	get cutoff(): string | null {
		return this.cutoffs[this.cutoffIndex] ?? null;
	}
	get bundle(): ModelBundleInfo | undefined {
		return this.prediction?.model_bundle;
	}
	/** Predictions for cutoffs 0..cutoffIndex in order (only those actually computed). */
	get historyPredictions(): PredictionResponse[] {
		const out: PredictionResponse[] = [];
		for (let i = 0; i <= this.cutoffIndex; i++) if (this.computed[i]) out.push(this.computed[i]);
		return out;
	}

	/** Clear everything when the signed-in identity changes. Safe to call from a page $effect. */
	ensureOwner(): boolean {
		const identity = tokenIdentity(authToken.value);
		if (identity === this.owner) return false;
		const first = this.owner === '' && this.subjects.length === 0;
		this.owner = identity;
		if (!first) this.reset();
		return !first;
	}

	reset() {
		this.pause();
		this.caches.clear();
		this.inflight.clear();
		this.subjects = [];
		this.stayId = '';
		this.cutoffIndex = 0;
		this.prediction = null;
		this.computed = {};
		this.events = [];
		this.error = '';
		this.loadingSubjects = true;
	}

	async ensureSubjects(force = false) {
		this.ensureOwner();
		if (this.subjects.length && !force) return;
		this.loadingSubjects = true;
		const list: WorkspacePatient[] = [];
		try {
			const demo = await api.demoSubjects();
			list.push(...demo.demo_subjects.map((s) => ({ ...s, kind: 'demo' as const })));
		} catch (cause) {
			this.error = cause instanceof Error ? cause.message : 'Could not load demo subjects';
		}
		try {
			const sc = await request<{ scenarios: ShowcaseCard[] }>('/showcase/scenarios');
			for (const s of sc.scenarios ?? []) {
				const first = new Date(s.legal_cutoffs[0]);
				const intime = new Date(first.getTime() - 24 * HOUR_MS).toISOString().replace('.000Z', 'Z');
				list.push({
					kind: 'showcase', stay_id: s.stay_id, subject_id: s.subject_id, cardiac_condition_group: s.subtype,
					age_years: s.age_years, sex_category: s.sex_category, intime, outtime: '', legal_cutoffs: s.legal_cutoffs,
					n_legal_cutoffs: s.n_cutoffs, patient_alias: s.patient_alias, title: s.title, scenario_id: s.scenario_id, objective: s.objective
				});
			}
		} catch {
			/* showcase unavailable: demo cohort still works */
		}
		try {
			const { records } = await api.listCustomRecords();
			const fetched = await Promise.allSettled(records.map((r) => api.getCustomRecord(r.stay_id)));
			for (const outcome of fetched) {
				if (outcome.status === 'fulfilled') list.push({ ...(outcome.value as unknown as DemoSubject), kind: 'custom' });
			}
		} catch {
			/* not signed in or persistence unavailable */
		}
		this.subjects = list;
		this.loadingSubjects = false;
	}

	/** Make sure an unlisted custom record requested via URL is available. */
	async ensureCustom(stayId: string) {
		if (this.subjects.some((s) => s.stay_id === stayId) || !stayId.startsWith('CUSTOM-')) return;
		try {
			const record = await api.getCustomRecord(stayId);
			this.subjects = [...this.subjects, { ...(record as unknown as DemoSubject), kind: 'custom' }];
		} catch {
			/* falls through to default selection */
		}
	}

	parseUrl(params: URLSearchParams) {
		const stay = params.get('stay') ?? params.get('stay_id') ?? '';
		const cutoff = Number(params.get('cutoff') ?? 0) || 0;
		return { stay, cutoff };
	}

	urlQuery(): string {
		return `?stay=${encodeURIComponent(this.stayId)}&cutoff=${this.cutoffIndex}`;
	}

	private cacheFor(stayId: string): StayCache | undefined {
		return this.caches.get(stayId);
	}

	private store(stayId: string, t: string, response: PredictionResponse) {
		const bundleId = response.model_bundle?.bundle_id ?? response.schema_version;
		const revision = response.data_revision ?? '';
		let cache = this.caches.get(stayId);
		if (!cache || cache.bundleId !== bundleId || cache.revision !== revision) {
			cache = { bundleId, revision, byCutoff: new Map() };
			this.caches.set(stayId, cache);
			if (stayId === this.stayId) this.computed = {};
		}
		cache.byCutoff.set(t, response);
	}

	async fetchPrediction(stayId: string, t: string): Promise<PredictionResponse> {
		const cached = this.cacheFor(stayId)?.byCutoff.get(t);
		if (cached) return cached;
		const key = `${stayId}|${t}`;
		let promise = this.inflight.get(key);
		if (!promise) {
			promise = api.predict(stayId, t).then((r) => {
				this.store(stayId, t, r);
				return r;
			}).finally(() => this.inflight.delete(key));
			this.inflight.set(key, promise);
		}
		return promise;
	}

	async select(stayId: string, cutoff = 0) {
		this.pause();
		this.stayId = stayId;
		this.cutoffIndex = Math.max(0, Math.min(cutoff, Math.max(0, this.cutoffs.length - 1)));
		this.computed = {};
		this.showHindsight = false;
		await this.load();
	}

	async setCutoff(index: number) {
		this.cutoffIndex = Math.max(0, Math.min(this.cutoffs.length - 1, index));
		await this.load();
	}

	async load() {
		const subject = this.subject;
		if (!subject || !this.cutoff) return;
		const stay = this.stayId;
		const ticket = ++this.token;
		this.loading = true;
		this.error = '';
		try {
			const idx = this.cutoffIndex;
			const upTo = this.cutoffs.slice(0, idx + 1);
			const current = await this.fetchPrediction(stay, this.cutoffs[idx]);
			const all = await Promise.all(upTo.map((c) => this.fetchPrediction(stay, c)));
			const historyResult = await api.history(stay, this.cutoffs[idx]);
			if (ticket !== this.token) return;
			const next: Record<number, PredictionResponse> = {};
			all.forEach((p, i) => (next[i] = p));
			for (const [k, v] of Object.entries(this.computed)) if (Number(k) > idx) next[Number(k)] = v;
			this.computed = next;
			this.prediction = current;
			this.events = historyResult.events;
			setCopilotContext({
				stayId: stay,
				patientAlias: subject.patient_alias ?? stay,
				predictionTime: this.cutoffs[idx],
				previousPredictionTime: idx > 0 ? this.cutoffs[idx - 1] : null
			});
		} catch (cause) {
			if (ticket !== this.token) return;
			this.error = cause instanceof Error ? cause.message : 'Prediction request failed';
			this.prediction = null;
		} finally {
			if (ticket === this.token) this.loading = false;
		}
	}

	/** Compute every legal cutoff (used only for explicitly requested retrospective comparison). */
	async loadAllCutoffs() {
		const stay = this.stayId;
		const all = await Promise.all(this.cutoffs.map((c) => this.fetchPrediction(stay, c)));
		if (stay !== this.stayId) return;
		const next: Record<number, PredictionResponse> = {};
		all.forEach((p, i) => (next[i] = p));
		this.computed = next;
	}

	play() {
		if (this.timer || this.cutoffs.length < 2) return;
		this.replaying = true;
		const tick = async () => {
			if (this.cutoffIndex >= this.cutoffs.length - 1) return this.pause();
			await this.setCutoff(this.cutoffIndex + 1);
		};
		this.timer = setInterval(() => void tick(), Math.max(400, 2600 / this.replaySpeed));
	}

	pause() {
		if (this.timer) clearInterval(this.timer);
		this.timer = null;
		this.replaying = false;
	}
}

export const patientContext = new PatientContext();
