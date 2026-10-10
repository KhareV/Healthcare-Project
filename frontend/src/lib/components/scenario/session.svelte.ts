import { api, type PredictionResponse } from '$lib/services/api';
import { showcase, timelineFromHistory, bundleKey, type Participant, type Story, type Timeline } from '$lib/services/showcase';

// Module-level prediction cache keyed by "<release:bundle>|<stay>|<cutoff>": a cutoff is only ever
// recomputed by the real POST /predict once per bundle per page session.
const cache = new Map<string, PredictionResponse>();
let bundle = 'unknown';
let bundleLoaded = false;

export async function ensureBundle() {
	if (!bundleLoaded) {
		bundle = await bundleKey();
		bundleLoaded = true;
	}
	return bundle;
}

export async function fetchPrediction(stayId: string, cutoff: string): Promise<PredictionResponse> {
	await ensureBundle();
	const key = `${bundle}|${stayId}|${cutoff}`;
	const hit = cache.get(key);
	if (hit) return hit;
	const result = await api.predict(stayId, cutoff);
	cache.set(key, result);
	return result;
}

export class ReplaySession {
	participant = $state<Participant | null>(null);
	cutoffIndex = $state(0);
	predictions = $state<Record<number, PredictionResponse>>({});
	timeline = $state<Timeline | null>(null);
	story = $state<Story | null>(null);
	loading = $state(false);
	error = $state('');
	private token = 0;

	current = $derived(this.predictions[this.cutoffIndex] ?? null);
	cutoffs = $derived(this.participant?.legalCutoffs ?? []);
	/** Predictions for cutoffs 0..cutoffIndex that are already computed, in order. */
	history = $derived.by(() => {
		const out: (PredictionResponse | null)[] = [];
		for (let i = 0; i <= this.cutoffIndex; i += 1) out.push(this.predictions[i] ?? null);
		return out;
	});

	async open(p: Participant, index = 0) {
		const token = ++this.token;
		this.participant = p;
		this.predictions = {};
		this.timeline = null;
		this.story = null;
		this.error = '';
		this.cutoffIndex = Math.max(0, Math.min(index, p.legalCutoffs.length - 1));
		this.loading = true;
		try {
			if (p.kind === 'scenario') {
				const [tl, st] = await Promise.all([showcase.timeline(p.key), showcase.story(p.key)]);
				if (token !== this.token) return;
				this.timeline = tl;
				this.story = st;
			}
			await this.load(token);
		} catch (cause) {
			if (token === this.token) this.error = cause instanceof Error ? cause.message : 'Could not load this patient';
		} finally {
			if (token === this.token) this.loading = false;
		}
	}

	async goTo(index: number) {
		if (!this.participant) return;
		const bounded = Math.max(0, Math.min(index, this.participant.legalCutoffs.length - 1));
		this.cutoffIndex = bounded;
		const token = ++this.token;
		this.loading = true;
		try {
			await this.load(token);
		} catch (cause) {
			if (token === this.token) this.error = cause instanceof Error ? cause.message : 'Prediction failed';
		} finally {
			if (token === this.token) this.loading = false;
		}
	}

	private async load(token: number) {
		const p = this.participant;
		if (!p) return;
		const i = this.cutoffIndex;
		const here = await fetchPrediction(p.stayId, p.legalCutoffs[i]);
		if (token !== this.token) return;
		this.predictions = { ...this.predictions, [i]: here };
		if (p.kind === 'demo') {
			const timeline = await timelineFromHistory(p, i);
			if (token !== this.token) return;
			this.timeline = timeline;
		}
		// Fill the earlier cutoffs progressively (cached after the first computation).
		for (let j = 0; j < i; j += 1) {
			if (this.predictions[j]) continue;
			const r = await fetchPrediction(p.stayId, p.legalCutoffs[j]);
			if (token !== this.token) return;
			this.predictions = { ...this.predictions, [j]: r };
		}
		this.error = '';
	}
}
