<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { Play, Pause, RotateCcw, SkipForward, SkipBack, Maximize2, Minimize2, Sparkles, RadioTower, Volume2, VolumeX, Loader2, ArrowLeft, Waypoints, BookOpen } from '@lucide/svelte';
	import { api, request } from '$lib/services/api';
	import { setCopilotContext } from '$lib/stores/copilot.svelte';
	import type { AIRecommendation } from '$lib/services/api';
	import GaugeRing from '$lib/components/dashboard/GaugeRing.svelte';
	import MiniSpark from '$lib/components/dashboard/MiniSpark.svelte';
	import { ReplaySession } from '$lib/components/scenario/session.svelte';
	import ScenarioCard from '$lib/components/scenario/ScenarioCard.svelte';
	import ObservedTimeline from '$lib/components/scenario/ObservedTimeline.svelte';
	import ContributorPanel from '$lib/components/scenario/ContributorPanel.svelte';
	import StoryPanel from '$lib/components/scenario/StoryPanel.svelte';
	import ForecastStrip from '$lib/components/scenario/ForecastStrip.svelte';
	import ComparePanel from '$lib/components/scenario/ComparePanel.svelte';
	import { scenarioParticipant, demoParticipant, showcase, prettySubtype, hhmm, type Participant, type ScenarioList } from '$lib/services/showcase';

	type Mode = 'select' | 'replay' | 'compare';
	const SPEEDS = [0.5, 1, 2, 4];

	const session = new ReplaySession();
	const sessionA = new ReplaySession();
	const sessionB = new ReplaySession();

	let participants = $state<Participant[]>([]);
	let meta = $state<ScenarioList | null>(null);
	let mode = $state<Mode>('select');
	let booting = $state(true);
	let bootError = $state('');
	let running = $state(false);
	let speed = $state(1);
	let loopAll = $state(false);
	let fullscreen = $state(false);
	let storyOn = $state(false);
	let concept = $state('');
	let task = $state<'recovery24' | 'recovery48' | 'icu_stay_time' | 'organ_support'>('organ_support');
	let cmpIndex = $state(0);
	let cmpA = $state('');
	let cmpB = $state('');
	let evidence = $state<Record<string, { value: number; ci95?: number[]; unit?: string; threshold?: number }> | null>(null);

	// AI research note + narration (unchanged behaviours; fully optional)
	let aiNote = $state<AIRecommendation | null>(null);
	let aiLoading = $state(false);
	let aiAuto = $state(false);
	let aiError = $state('');
	let audioEl: HTMLAudioElement | null = $state(null);
	let narrateLoading = $state(false);
	let narratePlaying = $state(false);
	let narrateBlocked = $state(false);
	let narrateError = $state('');
	let narratedFor: string | null = null;
	let narratedUrl: string | null = null;

	const scenarios = $derived(participants.filter((p) => p.kind === 'scenario'));
	const demos = $derived(participants.filter((p) => p.kind === 'demo'));
	const byKey = (k: string) => participants.find((p) => p.key === k) ?? null;
	const current = $derived(session.current);
	const previous = $derived(session.cutoffIndex > 0 ? (session.predictions[session.cutoffIndex - 1] ?? null) : null);
	const nCut = $derived(session.cutoffs.length);
	const xLabels = $derived(session.cutoffs.slice(0, session.cutoffIndex + 1).map((c) => `t${session.cutoffs.indexOf(c)}`));
	const observedSofa = $derived(session.timeline?.observed_sofa_by_cutoff ?? []);
	const cmpMax = $derived(Math.max(1, Math.min(sessionA.cutoffs.length, sessionB.cutoffs.length)));
	const tone = $derived(current?.organ_support.alert ? 'danger' : 'ok');

	// Support state at the selected cutoff: derived from observed intervals (start <= t < end).
	const activeSupport = $derived.by(() => {
		const tl = session.timeline;
		if (!tl) return [] as { type: string; category: string; qualifying: boolean; since: number }[];
		const t = tl.cutoff_hours[session.cutoffIndex];
		return tl.support_runs
			.filter((r) => r.first_visible_cutoff !== null && r.first_visible_cutoff <= session.cutoffIndex && r.start_hour <= t && t < r.end_hour)
			.map((r) => ({ type: r.support_type === 'VASOPRESSOR' ? 'Vasopressor' : 'Respiratory', category: r.category, qualifying: r.qualifying_for_initiation, since: r.start_hour }));
	});

	function syncUrl() {
		const u = new URL(page.url);
		['s', 'c', 'story', 'a', 'b', 'mode'].forEach((k) => u.searchParams.delete(k));
		if (mode === 'replay' && session.participant) {
			u.searchParams.set('mode', 'replay');
			u.searchParams.set('s', session.participant.key);
			u.searchParams.set('c', String(session.cutoffIndex));
			if (storyOn) u.searchParams.set('story', '1');
		} else if (mode === 'compare') {
			u.searchParams.set('mode', 'compare');
			u.searchParams.set('a', cmpA);
			u.searchParams.set('b', cmpB);
			u.searchParams.set('c', String(cmpIndex));
		}
		goto(u, { replaceState: true, noScroll: true, keepFocus: true });
	}

	function resetNarration() {
		if (audioEl) audioEl.pause();
		narratePlaying = false;
		narrateBlocked = false;
		narrateError = '';
		if (narratedUrl) URL.revokeObjectURL(narratedUrl);
		narratedUrl = null;
		narratedFor = null;
	}

	async function playNarration(auto = false) {
		if (!aiNote || aiNote.status !== 'OK' || !aiNote.summary) return;
		narrateError = '';
		narrateBlocked = false;
		try {
			if (narratedFor !== aiNote.summary) {
				narrateLoading = true;
				const blob = await api.speak(aiNote.summary);
				if (narratedUrl) URL.revokeObjectURL(narratedUrl);
				narratedUrl = URL.createObjectURL(blob);
				narratedFor = aiNote.summary;
				narrateLoading = false;
			}
			if (audioEl && narratedUrl) {
				if (audioEl.src !== narratedUrl) audioEl.src = narratedUrl;
				await audioEl.play();
				narratePlaying = true;
			}
		} catch (cause) {
			narrateLoading = false;
			if (cause instanceof DOMException && cause.name === 'NotAllowedError') narrateBlocked = true;
			else if (!auto) narrateError = cause instanceof Error ? cause.message : 'Narration unavailable right now.';
		}
	}

	function toggleNarration() {
		if (narratePlaying) {
			audioEl?.pause();
			narratePlaying = false;
		} else void playNarration(false);
	}

	async function generateAINote(silent = false) {
		const p = session.participant;
		if (!p || !current) return;
		aiLoading = true;
		if (!silent) aiError = '';
		resetNarration();
		try {
			aiNote = await api.aiRecommendation(p.stayId, session.cutoffs[session.cutoffIndex]);
			if (aiNote.status !== 'OK') aiError = aiNote.error ?? 'AI synthesis unavailable right now.';
			else if (aiAuto) void playNarration(true);
		} catch (cause) {
			if (!silent) aiError = cause instanceof Error ? cause.message : 'AI synthesis unavailable right now.';
		} finally {
			aiLoading = false;
		}
	}

	function updateCopilot() {
		const p = session.participant;
		if (!p || !p.legalCutoffs[session.cutoffIndex]) return;
		setCopilotContext({
			stayId: p.stayId,
			patientAlias: p.alias,
			predictionTime: p.legalCutoffs[session.cutoffIndex],
			previousPredictionTime: session.cutoffIndex > 0 ? p.legalCutoffs[session.cutoffIndex - 1] : null
		});
	}

	async function openParticipant(p: Participant, index = 0) {
		mode = 'replay';
		running = false;
		aiNote = null;
		aiError = '';
		resetNarration();
		await session.open(p, index);
		updateCopilot();
		syncUrl();
		if (aiAuto) void generateAINote(true);
	}

	async function openCompare(a: string, b: string, index = 0) {
		const pa = byKey(a), pb = byKey(b);
		if (!pa || !pb) return;
		mode = 'compare';
		running = false;
		cmpA = a;
		cmpB = b;
		cmpIndex = Math.min(index, Math.min(pa.legalCutoffs.length, pb.legalCutoffs.length) - 1);
		await Promise.all([sessionA.open(pa, cmpIndex), sessionB.open(pb, cmpIndex)]);
		syncUrl();
	}

	async function setIndex(i: number) {
		if (mode === 'compare') {
			cmpIndex = Math.max(0, Math.min(i, cmpMax - 1));
			await Promise.all([sessionA.goTo(cmpIndex), sessionB.goTo(cmpIndex)]);
		} else if (session.participant) {
			await session.goTo(i);
			updateCopilot();
		}
		syncUrl();
	}

	function stepIndex() {
		return mode === 'compare' ? cmpIndex : session.cutoffIndex;
	}
	function lastIndex() {
		return mode === 'compare' ? cmpMax - 1 : nCut - 1;
	}

	async function next() {
		if (stepIndex() < lastIndex()) return setIndex(stepIndex() + 1);
		if (mode === 'replay' && loopAll && participants.length) {
			const list = session.participant?.kind === 'demo' ? demos : scenarios;
			const i = list.findIndex((p) => p.key === session.participant?.key);
			return openParticipant(list[(i + 1) % list.length]);
		}
		running = false;
	}
	function previousStep() {
		if (stepIndex() > 0) void setIndex(stepIndex() - 1);
	}
	function reset() {
		running = false;
		void setIndex(0);
		aiNote = null;
	}
	function backToSelect() {
		running = false;
		mode = 'select';
		syncUrl();
	}
	async function toggleFullscreen() {
		if (document.fullscreenElement) await document.exitFullscreen();
		else await document.documentElement.requestFullscreen();
	}
	function toggleStory() {
		storyOn = !storyOn;
		syncUrl();
	}
	function compareFrom(p: Participant) {
		const other = (scenarios.find((s) => s.key !== p.key) ?? demos.find((s) => s.key !== p.key))?.key ?? p.key;
		void openCompare(p.key, other);
	}

	onMount(() => {
		(async () => {
			try {
				const [list, demo] = await Promise.all([showcase.scenarios(), api.demoSubjects()]);
				meta = list;
				participants = [...list.scenarios.map(scenarioParticipant), ...demo.demo_subjects.map(demoParticipant)];
				request<{ headline: Record<string, { value: number; ci95?: number[]; unit?: string; threshold?: number }> }>('/research/summary').then((s) => (evidence = s.headline)).catch(() => (evidence = null));
				const q = page.url.searchParams;
				const m = q.get('mode');
				storyOn = q.get('story') === '1';
				const c = Number(q.get('c') ?? 0) || 0;
				if (m === 'replay' && q.get('s') && byKey(q.get('s')!)) await openParticipant(byKey(q.get('s')!)!, c);
				else if (m === 'compare' && q.get('a') && q.get('b')) await openCompare(q.get('a')!, q.get('b')!, c);
			} catch (cause) {
				bootError = cause instanceof Error ? cause.message : 'Could not load the scenario library';
			} finally {
				booting = false;
			}
		})();

		const onFullscreenChange = () => (fullscreen = Boolean(document.fullscreenElement));
		const onKeyDown = (event: KeyboardEvent) => {
			const target = event.target as HTMLElement | null;
			if (target?.matches('input:not([type=range]), select, textarea')) return;
			if (mode === 'select') return;
			if (event.code === 'Space') { event.preventDefault(); running = !running; }
			if (event.key === 'ArrowRight') { event.preventDefault(); void next(); }
			if (event.key === 'ArrowLeft') { event.preventDefault(); previousStep(); }
			if (event.key.toLowerCase() === 'r') reset();
			if (event.key.toLowerCase() === 'f') void toggleFullscreen();
		};
		document.addEventListener('fullscreenchange', onFullscreenChange);
		window.addEventListener('keydown', onKeyDown);
		return () => {
			document.removeEventListener('fullscreenchange', onFullscreenChange);
			window.removeEventListener('keydown', onKeyDown);
		};
	});

	// Autoplay: only advance after the previous cutoff has finished computing.
	$effect(() => {
		if (!running) return;
		const handle = window.setInterval(() => {
			const busy = mode === 'compare' ? sessionA.loading || sessionB.loading : session.loading;
			if (!busy) void next();
		}, 2600 / speed);
		return () => window.clearInterval(handle);
	});

	onDestroy(() => {
		if (narratedUrl) URL.revokeObjectURL(narratedUrl);
	});
</script>

<svelte:head>
	<title>Scenario Studio | Personalized Patient Recovery Trajectory</title>
	<meta name="description" content="Eight evidence-grounded retrospective replay scenarios running the frozen V3 forecasting models." />
</svelte:head>

<div class="demo-shell" class:danger={tone === 'danger' && mode === 'replay'} class:present={fullscreen}>
	<header class="hero">
		<div>
			<span class="kicker"><i></i> RETROSPECTIVE SEQUENTIAL REPLAY · SYNTHETIC RESEARCH BENCHMARK · V3 MODELS</span>
			<h1>Scenario Studio</h1>
			<p>Step through legal ICU cutoffs. Every value is a genuine <code>POST /predict</code> from the frozen V3 forecasting pipeline computed on history truncated at that cutoff — never scripted or precomputed. Not real-time clinical prediction.</p>
		</div>
		{#if mode !== 'select'}<button class="back" onclick={backToSelect}><ArrowLeft size={14} /> ALL SCENARIOS</button>{/if}
	</header>

	{#if booting}
		<div class="loading">Loading the scenario library…</div>
	{:else if bootError}
		<div class="notice">{bootError}</div>
	{:else if mode === 'select'}
		<div class="banner" role="note"><b>DEVELOPMENT-COHORT PATIENTS</b><span>Not training data, not a final-test cohort. Illustrative trajectories — not an unbiased evaluation sample. Final scientific performance is reported separately on the sealed V3 test.</span></div>
		<section class="group"><h2>Evidence-grounded scenarios <small>{scenarios.length} · held-out development cohort · selected by structure/observed trajectory, never by model error</small></h2>
			<div class="grid">{#each scenarios as p}<ScenarioCard participant={p} onopen={(x) => openParticipant(x)} oncompare={compareFrom} />{/each}</div>
		</section>
		<section class="group"><h2>Training-cohort demonstrations <small>{demos.length} · original demo patients · included in model development</small></h2>
			<div class="grid">{#each demos as p}<ScenarioCard participant={p} onopen={(x) => openParticipant(x)} oncompare={compareFrom} />{/each}</div>
		</section>
		{#if meta}<p class="lineage">Manifest SHA-256 {meta.manifest_sha256.slice(0, 16)}… · source cohort {meta.source_cohort} · status {meta.status}</p>{/if}
	{:else if mode === 'compare'}
		<section class="controls" aria-label="Comparison controls">
			<label>PATIENT A <select value={cmpA} onchange={(e) => openCompare((e.target as HTMLSelectElement).value, cmpB, cmpIndex)}>{#each participants as p}<option value={p.key}>{p.kind === 'scenario' ? `${p.key} · ` : ''}{p.title}</option>{/each}</select></label>
			<label>PATIENT B <select value={cmpB} onchange={(e) => openCompare(cmpA, (e.target as HTMLSelectElement).value, cmpIndex)}>{#each participants as p}<option value={p.key}>{p.kind === 'scenario' ? `${p.key} · ` : ''}{p.title}</option>{/each}</select></label>
			<button class="primary" onclick={() => (running = !running)}>{#if running}<Pause size={15} /> PAUSE{:else}<Play size={15} /> PLAY{/if}</button>
			<button onclick={previousStep}><SkipBack size={15} /> PREV</button>
			<button onclick={() => void next()}><SkipForward size={15} /> NEXT</button>
			<label>SPEED <select bind:value={speed}>{#each SPEEDS as s}<option value={s}>{s}×</option>{/each}</select></label>
			<label class="slider">CUTOFF t{cmpIndex} / t{cmpMax - 1}<input type="range" min="0" max={cmpMax - 1} step="1" value={cmpIndex} aria-label="Aligned cutoff index" oninput={(e) => setIndex(Number((e.target as HTMLInputElement).value))} /></label>
		</section>
		{#if sessionA.error || sessionB.error}<div class="notice">{sessionA.error || sessionB.error}</div>{/if}
		<ComparePanel a={sessionA} b={sessionB} index={cmpIndex} />
	{:else if session.error && !current}
		<div class="notice">{session.error}</div>
	{:else if session.participant}
		{@const p = session.participant}
		<section class="controls" aria-label="Replay controls">
			<label>SCENARIO <select value={p.key} onchange={(e) => { const x = byKey((e.target as HTMLSelectElement).value); if (x) void openParticipant(x); }}>
				<optgroup label="Held-out development cohort">{#each scenarios as s}<option value={s.key}>{s.key} · {s.title}</option>{/each}</optgroup>
				<optgroup label="Training-cohort demonstrations">{#each demos as s}<option value={s.key}>{s.title}</option>{/each}</optgroup>
			</select></label>
			<button class="primary" onclick={() => (running = !running)} aria-label={running ? 'Pause replay' : 'Play replay'}>{#if running}<Pause size={15} /> PAUSE{:else}<Play size={15} /> PLAY{/if}</button>
			<button onclick={previousStep} aria-label="Previous cutoff"><SkipBack size={15} /> PREV</button>
			<button onclick={() => void next()} aria-label="Next cutoff"><SkipForward size={15} /> NEXT</button>
			<button onclick={reset}><RotateCcw size={15} /> RESET</button>
			<label>SPEED <select bind:value={speed}>{#each SPEEDS as s}<option value={s}>{s}×</option>{/each}</select></label>
			<label class="chk"><input type="checkbox" bind:checked={loopAll} /> LOOP SCENARIOS</label>
			<button class:on={storyOn} onclick={toggleStory}><BookOpen size={15} /> STORY</button>
			<button onclick={() => compareFrom(p)}><Waypoints size={15} /> COMPARE</button>
			<button onclick={toggleFullscreen}>{#if fullscreen}<Minimize2 size={15} /> EXIT{:else}<Maximize2 size={15} /> PRESENT{/if}</button>
		</section>
		<div class="scrub">
			<label>CUTOFF <b data-testid="cutoff-label">t{session.cutoffIndex} / t{nCut - 1}</b> · {hhmm(session.cutoffs[session.cutoffIndex] ?? '')} UTC · {p.alias} · {prettySubtype(p.subtype)}
				<input type="range" min="0" max={nCut - 1} step="1" value={session.cutoffIndex} aria-label="Legal cutoff index" data-testid="cutoff-slider" oninput={(e) => setIndex(Number((e.target as HTMLInputElement).value))} /></label>
			<div class="shortcuts"><span><kbd>SPACE</kbd> PLAY/PAUSE</span><span><kbd>←</kbd><kbd>→</kbd> STEP</span><span><kbd>F</kbd> PRESENT</span><span><kbd>R</kbd> RESET</span></div>
		</div>
		{#if session.error}<div class="notice">{session.error}</div>{/if}

		{#if current}
			<section class="metric-grid" data-testid="metrics" aria-busy={session.loading}>
				<article><span>CURRENT SOFA (OBSERVED)</span><strong data-testid="m-sofa">{current.current_sofa.toFixed(1)}<small>/24</small></strong><MiniSpark values={session.history.map((x) => (x ? x.current_sofa : 0))} color="#2bb8b0" height={28} width={140} /><footer>COMPUTED FROM DATA ≤ CUTOFF</footer></article>
				<article><span>FORECAST +24H → +48H</span><strong data-testid="m-rec">{current.recovery.sofa_hat_24h.toFixed(1)}<small>→ {current.recovery.sofa_hat_48h.toFixed(1)}</small></strong><MiniSpark values={session.history.map((x) => (x ? x.recovery.sofa_hat_24h : 0))} color="#38bdf8" height={28} width={140} /><footer>INDEPENDENT HORIZONS · Δ {current.recovery.delta_24h.toFixed(2)} / {current.recovery.delta_48h.toFixed(2)}</footer></article>
				<article><span>REMAINING ICU TIME</span><strong data-testid="m-icu">{current.icu_stay_time.remaining_hours.toFixed(1)}<small>h</small></strong><MiniSpark values={session.history.map((x) => (x ? x.icu_stay_time.remaining_hours : 0))} color="#fbbf24" height={28} width={140} /><footer>CURRENT ICU STAY ONLY</footer></article>
				<article class:alert={current.organ_support.alert}><span>SUPPORT INITIATION RISK (24H)</span><strong data-testid="m-risk" class="state">{(current.organ_support.probability_24h * 100).toFixed(1)}<small>%</small></strong><MiniSpark values={session.history.map((x) => (x ? x.organ_support.probability_24h * 100 : 0))} color="#fb7185" height={28} width={140} /><footer>{current.organ_support.alert ? 'ABOVE' : 'BELOW'} FROZEN THRESHOLD {(current.organ_support.threshold * 100).toFixed(1)}%</footer></article>
			</section>

			<ForecastStrip history={session.history} labels={xLabels} {observedSofa} />

			<section class="two">
				<ObservedTimeline timeline={session.timeline} cutoffIndex={session.cutoffIndex} bind:concept />
				<div class="stack">
					<div class="state-card">
						<span>ORGAN-SUPPORT STATE AT CUTOFF</span>
						{#if p.kind === 'demo'}<p class="muted">Support state shown via the observed history for demonstration patients.</p>
						{:else if activeSupport.length === 0}<strong>No support active</strong>
						{:else}<ul>{#each activeSupport as r}<li>{r.type}: {r.category.toLowerCase().replace(/_/g, ' ')} · since hour {r.since.toFixed(0)}{r.qualifying ? ' · qualifying' : ' · non-qualifying'}</li>{/each}</ul>{/if}
						<div class="dq">
							<GaugeRing value={Math.round((current.data_quality.observed_feature_fraction ?? 0) * 100)} size={54} stroke={6} color="#2bb8b0" label="DATA" />
							<p>{current.data_quality.observed_bins}/{current.data_quality.total_bins} bins observed · {current.data_quality.padding_bins} pre-admission padding bins · completeness is not model confidence</p>
						</div>
					</div>
					<ContributorPanel prediction={current} bind:task />
				</div>
			</section>

			{#if storyOn}
				<StoryPanel story={session.story} cutoffIndex={session.cutoffIndex} {current} {previous} nCutoffs={nCut} />
			{/if}

			<section class="ai-note" class:loading={aiLoading}>
				<audio bind:this={audioEl} onended={() => (narratePlaying = false)} onpause={() => (narratePlaying = false)} onplay={() => (narratePlaying = true)}></audio>
				<header>
					<div class="ai-note-title"><Sparkles size={15} /><span>AI RESEARCH NOTE</span><small>optional · separate from the numerical models · {aiNote?.model ?? 'Groq'}</small></div>
					<label class="ai-toggle"><input type="checkbox" bind:checked={aiAuto} /> AUTO-NARRATE EACH SCENARIO</label>
					<button class="ai-btn" onclick={() => generateAINote(false)} disabled={aiLoading}><RadioTower size={13} /> {aiLoading ? 'SYNTHESIZING…' : 'WRITE THIS MOMENT'}</button>
					{#if aiNote?.status === 'OK' && aiNote.summary}
						<button class="ai-btn" class:speaking={narratePlaying} onclick={toggleNarration} disabled={narrateLoading}>
							{#if narrateLoading}<Loader2 size={13} /> VOICING…{:else if narratePlaying}<VolumeX size={13} /> STOP{:else}<Volume2 size={13} /> LISTEN{/if}
						</button>
					{/if}
				</header>
				{#if aiNote?.status === 'OK' && aiNote.summary}
					<p class="ai-body">{aiNote.summary}</p>
					{#if narrateBlocked}<p class="ai-hint">Browser blocked autoplay audio — click “Listen”.</p>{/if}
					{#if narrateError}<p class="ai-hint err">{narrateError}</p>{/if}
					<footer class="ai-disclaimer">{aiNote.disclaimer}</footer>
				{:else}
					<p class="ai-body muted">{aiError || 'No AI note yet. The studio and every number above work without it (Groq / TTS are optional).'}</p>
				{/if}
			</section>
		{:else}
			<div class="loading" aria-live="polite">Computing the V3 forecast for this cutoff…</div>
		{/if}

		<section class="evidence">
			<header><div><span>FINAL SEALED-TEST EVIDENCE (V3) — SEPARATE FROM THIS REPLAY</span><b>One-time evaluation on 1,500 independent subjects; none of these replay patients come from it</b></div><a href="/research/analytics">OPEN MODEL RESEARCH ↗</a></header>
			{#if evidence}
				<div class="evidence-grid">
					<article><span>RECOVERY +24 MAE</span><strong>{evidence.recovery24_mae.value.toFixed(3)}</strong><small>SOFA points · 95% CI [{evidence.recovery24_mae.ci95?.map((v) => v.toFixed(2)).join(', ')}]</small></article>
					<article><span>ICU WEIGHTED MEDIAN AE</span><strong>{evidence.icu_weighted_median_ae_hours.value.toFixed(2)} h</strong><small>95% CI [{evidence.icu_weighted_median_ae_hours.ci95?.map((v) => v.toFixed(2)).join(', ')}]</small></article>
					<article><span>SUPPORT CALIBRATED AUPRC</span><strong>{evidence.support_calibrated_auprc.value.toFixed(3)}</strong><small>95% CI [{evidence.support_calibrated_auprc.ci95?.map((v) => v.toFixed(2)).join(', ')}]</small></article>
						<article><span>SUPPORT CALIBRATED AUROC</span><strong>{evidence.support_calibrated_auroc.value.toFixed(3)}</strong><small>{#if evidence.support_f1_at_frozen_threshold?.threshold != null}frozen threshold {(evidence.support_f1_at_frozen_threshold.threshold * 100).toFixed(1)}%{:else}frozen threshold unavailable{/if}</small></article>
				</div>
			{:else}<p class="muted pad">Final-evaluation summary unavailable.</p>{/if}
		</section>
		<footer class="disclaimer"><b>SYNTHETIC RESEARCH BENCHMARK</b><span>Model-backed forecasts on synthetic ICU-like episodes. Not real patient data, not real-time clinical prediction, not a diagnosis or treatment recommendation. Contributions are statistical, not causal.</span></footer>
	{/if}
</div>

<style>
	.demo-shell{--accent:#2bb8b0;position:relative;max-width:1450px;margin:0 auto;padding:0 24px 28px;color:#dce9e8}.demo-shell.danger{--accent:#fb7185}
	.hero{display:flex;justify-content:space-between;align-items:end;gap:20px;margin:6px 0 20px;padding-bottom:20px;border-bottom:1px solid rgba(148,163,184,.15)}
	.present .hero{display:none}
	.kicker{display:flex;align-items:center;gap:9px;color:var(--accent);font:8px 'JetBrains Mono',monospace;letter-spacing:.14em}.kicker i{width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 9px var(--accent)}
	h1{margin:10px 0 8px;font:500 clamp(30px,4vw,52px)/1 'Space Grotesk',sans-serif;letter-spacing:-.03em}.hero p{max-width:720px;margin:0;color:#8494a9;font-size:13px;line-height:1.65}code{color:#82e9e1}
	.back{display:flex;align-items:center;gap:7px;padding:9px 12px;border:1px solid rgba(148,163,184,.2);background:#0a1220;color:#b7c6d4;font:8px 'JetBrains Mono',monospace;letter-spacing:.08em;cursor:pointer}
	.loading,.notice{padding:20px 0;color:#64748b;font:10px 'JetBrains Mono',monospace}.notice{color:#fecdd3}
	.banner{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:22px;padding:12px 14px;border:1px solid rgba(251,191,36,.28);background:rgba(251,191,36,.04);color:#a8b6c7;font-size:12px;line-height:1.5}.banner b{color:#fbbf24;font:8px 'JetBrains Mono',monospace;letter-spacing:.12em}
	.group{margin-bottom:26px}.group h2{margin:0 0 12px;color:#e5f1f0;font:500 18px 'Space Grotesk',sans-serif}.group h2 small{margin-left:10px;color:#53647b;font:8px 'JetBrains Mono',monospace;letter-spacing:.08em}
	.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:10px}.lineage{color:#53647b;font:8px 'JetBrains Mono',monospace}
	.controls{display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding:11px;border:1px solid rgba(148,163,184,.14);background:#070d18;font-size:8px;letter-spacing:.08em}
	.controls button{display:flex;align-items:center;gap:7px;padding:9px 12px;border:1px solid rgba(148,163,184,.2);background:#0a1220;color:#b7c6d4;font:8px 'JetBrains Mono',monospace;letter-spacing:.08em;cursor:pointer}
	.controls button.primary,.controls button.on{border-color:rgba(43,184,176,.5);background:rgba(43,184,176,.1);color:#82e9e1}
	.controls label{display:flex;align-items:center;gap:7px;color:#64748b;font:8px 'JetBrains Mono',monospace}.controls select{padding:7px;border:1px solid rgba(148,163,184,.2);background:#0a1220;color:#dce9e8;font:9px 'JetBrains Mono',monospace;max-width:240px}
	.controls .slider{flex:1;min-width:200px}.controls .slider input{flex:1}
	.scrub{margin:10px 0}.scrub label{display:grid;gap:7px;color:#8494a9;font:8px 'JetBrains Mono',monospace;letter-spacing:.06em}.scrub b{color:var(--accent)}.scrub input{width:100%;accent-color:#2bb8b0}
	.shortcuts{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;color:#53647b;font:7px 'JetBrains Mono',monospace}kbd{margin-right:4px;padding:2px 5px;border:1px solid rgba(148,163,184,.25);color:#94a3b8}
	.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:12px 0 7px}.metric-grid article{display:flex;flex-direction:column;min-height:118px;padding:16px;border:1px solid rgba(148,163,184,.14);background:#080f1b}
	.metric-grid article.alert{border-color:rgba(251,113,133,.45)}.metric-grid span{color:#64748b;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em}
	.metric-grid strong{margin:8px 0;color:#eef7f6;font:500 30px/1 'Space Grotesk',sans-serif}.metric-grid strong small{margin-left:5px;color:#64748b;font:9px 'JetBrains Mono',monospace}
	.metric-grid article :global(svg){display:block;margin:2px 0 8px}.metric-grid footer{color:#53647b;font:7px 'JetBrains Mono',monospace;letter-spacing:.08em}
	.two{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:7px;margin:7px 0}.stack{display:grid;gap:7px;align-content:start;min-width:0}
	.state-card{padding:14px;border:1px solid rgba(148,163,184,.14);background:#080f1b}.state-card>span{display:block;margin-bottom:8px;color:#64748b;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em}.state-card strong{color:#cfe3e1;font-size:13px}
	.state-card ul{margin:0;padding-left:16px;color:#b7c6d4;font-size:12px;line-height:1.6}.dq{display:flex;align-items:center;gap:12px;margin-top:12px}.dq p{margin:0;color:#64748b;font:8px 'JetBrains Mono',monospace;line-height:1.6}
	.muted{color:#64748b;font:10px 'JetBrains Mono',monospace}.pad{padding:14px}
	.ai-note{margin-top:7px;padding:16px;border:1px solid rgba(56,189,248,.22);background:linear-gradient(135deg,rgba(56,189,248,.05),#070d18 46%)}.ai-note>header{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
	.ai-note-title{display:flex;align-items:center;gap:8px;color:#38bdf8;font:8px 'JetBrains Mono',monospace;letter-spacing:.1em}.ai-note-title small{color:#53647b}
	.ai-toggle{color:#64748b;font:7px 'JetBrains Mono',monospace;margin-left:auto}.ai-btn{display:flex;align-items:center;gap:6px;padding:8px 10px;border:1px solid rgba(56,189,248,.3);background:rgba(56,189,248,.06);color:#38bdf8;font:8px 'JetBrains Mono',monospace;cursor:pointer}
	.ai-body{margin:12px 0 0;color:#b7c6d4;font-size:13px;line-height:1.7}.ai-hint{margin:8px 0 0;color:#fbbf24;font:9px 'JetBrains Mono',monospace}.ai-hint.err{color:#fb7185}.ai-disclaimer{margin-top:10px;color:#53647b;font:8px 'JetBrains Mono',monospace}
	.evidence{margin-top:7px;border:1px solid rgba(43,184,176,.2);background:linear-gradient(135deg,rgba(43,184,176,.045),#070d18 42%)}.evidence>header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;padding:13px 15px;border-bottom:1px solid rgba(148,163,184,.1)}
	.evidence>header span{display:block;color:#2bb8b0;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em}.evidence>header b{display:block;margin-top:5px;color:#8fa3b8;font:400 11px Inter,sans-serif}.evidence>header a{color:#2bb8b0;font:8px 'JetBrains Mono',monospace;text-decoration:none}
	.evidence-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:rgba(148,163,184,.08)}.evidence-grid article{padding:13px 15px;background:#070d18}.evidence-grid span{color:#64748b;font:7px 'JetBrains Mono',monospace;letter-spacing:.1em}.evidence-grid strong{display:block;margin:7px 0 4px;color:#eef7f6;font:500 22px 'Space Grotesk',sans-serif}.evidence-grid small{color:#53647b;font:7px 'JetBrains Mono',monospace}
	.disclaimer{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:18px;padding:14px;border:1px solid rgba(251,191,36,.22);color:#71829a;font-size:9px;letter-spacing:.04em}.disclaimer b{color:#fbbf24;font:8px 'JetBrains Mono',monospace}
	@media(max-width:1100px){.two{grid-template-columns:1fr}.evidence-grid,.metric-grid{grid-template-columns:1fr 1fr}}
	@media(max-width:680px){.demo-shell{padding:0 12px 20px}.hero{display:block}.back{margin-top:14px}.metric-grid,.evidence-grid{grid-template-columns:1fr 1fr}.controls select{max-width:100%}}
</style>
