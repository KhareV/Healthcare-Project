<script lang="ts">
	import { onMount } from 'svelte';
	import { Play, Pause, RotateCcw, SkipForward, SkipBack, Maximize2, Minimize2, Activity, BrainCircuit, BellRing, Waypoints, Sparkles, RadioTower } from '@lucide/svelte';
	import { api, type DemoSubject, type PredictionResponse, type AIRecommendation } from '$lib/services/api';
	import MultiLine from '$lib/components/dashboard/MultiLine.svelte';
	import RadarPlot from '$lib/components/dashboard/RadarPlot.svelte';
	import GaugeRing from '$lib/components/dashboard/GaugeRing.svelte';
	import MiniSpark from '$lib/components/dashboard/MiniSpark.svelte';

	let subjects = $state<DemoSubject[]>([]);
	let subjectIndex = $state(0);
	let cutoffIndex = $state(0);
	let running = $state(false);
	let speed = $state(1);
	let loops = $state(0);
	let fullscreen = $state(false);
	let loading = $state(true);
	let error = $state('');
	const cache = new Map<string, PredictionResponse>();
	let current = $state<PredictionResponse | null>(null);
	let historySofa = $state<number[]>([]);
	let historyIcu = $state<number[]>([]);
	let historySupport = $state<number[]>([]);
	let historyRec24 = $state<number[]>([]);
	let eventLog = $state<Array<{ time: string; label: string; tone: string }>>([]);

	let aiNote = $state<AIRecommendation | null>(null);
	let aiLoading = $state(false);
	let aiAuto = $state(true);
	let aiError = $state('');

	const radarData = $derived.by(() => {
		if (!current) return { labels: ['—', '—', '—', '—', '—'], values: [0, 0, 0, 0, 0] };
		const exp = current.explanations.recovery24;
		const items = [...exp.top_positive_contributors.slice(0, 3), ...exp.top_negative_contributors.slice(0, 2)];
		if (!items.length) return { labels: ['—', '—', '—', '—', '—'], values: [0, 0, 0, 0, 0] };
		const maxAbs = Math.max(...items.map((c) => Math.abs(c.attribution)), 1e-6);
		return {
			labels: items.map((c) => c.label.slice(0, 10).toUpperCase()),
			values: items.map((c) => Math.round((Math.abs(c.attribution) / maxAbs) * 100))
		};
	});

	const subject = $derived(subjects[subjectIndex] ?? null);
	const cutoffs = $derived(subject?.legal_cutoffs ?? []);
	const tone = $derived(current?.organ_support.alert ? 'danger' : 'ok');

	async function fetchPrediction(sid: string, t: string): Promise<PredictionResponse> {
		const key = `${sid}|${t}`;
		const cached = cache.get(key);
		if (cached) return cached;
		const result = await api.predict(sid, t);
		cache.set(key, result);
		return result;
	}

	async function loadStep() {
		if (!subject) return;
		try {
			const t = cutoffs[cutoffIndex];
			current = await fetchPrediction(subject.stay_id, t);
			const upTo = cutoffs.slice(0, cutoffIndex + 1);
			const preds = await Promise.all(upTo.map((c) => fetchPrediction(subject!.stay_id, c)));
			historySofa = preds.map((p) => p.current_sofa);
			historyIcu = preds.map((p) => p.icu_stay_time.remaining_hours);
			historySupport = preds.map((p) => p.organ_support.probability_24h * 100);
			historyRec24 = preds.map((p) => p.recovery.sofa_hat_24h);
			eventLog = [{ time: t.slice(11, 16), label: `t${cutoffIndex} · SOFA ${current.current_sofa.toFixed(1)}`, tone: current.organ_support.alert ? 'danger' : 'ok' }, ...eventLog].slice(0, 6);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Prediction failed';
		}
	}

	async function generateAINote(silent = false) {
		if (!subject || !current) return;
		aiLoading = true;
		if (!silent) aiError = '';
		try {
			aiNote = await api.aiRecommendation(subject.stay_id, cutoffs[cutoffIndex]);
			if (aiNote.status !== 'OK') aiError = aiNote.error ?? 'AI synthesis unavailable right now.';
		} catch (cause) {
			if (!silent) aiError = cause instanceof Error ? cause.message : 'AI synthesis unavailable right now.';
		} finally {
			aiLoading = false;
		}
	}

	function jumpToSubject(index: number) {
		subjectIndex = (index + subjects.length) % subjects.length;
		cutoffIndex = 0;
		eventLog = [];
		aiNote = null;
		aiError = '';
		void loadStep().then(() => {
			if (aiAuto) void generateAINote(true);
		});
	}
	function next() {
		if (cutoffIndex < cutoffs.length - 1) {
			cutoffIndex += 1;
		} else {
			jumpToSubject(subjectIndex + 1);
			if (subjectIndex === 0) loops += 1;
			return;
		}
		void loadStep();
	}
	function previousStep() {
		if (cutoffIndex > 0) cutoffIndex -= 1;
		else jumpToSubject(subjectIndex - 1);
		void loadStep();
	}
	function reset() {
		running = false;
		subjectIndex = 0;
		cutoffIndex = 0;
		loops = 0;
		eventLog = [];
		void loadStep();
	}
	async function toggleFullscreen() {
		if (document.fullscreenElement) await document.exitFullscreen();
		else await document.documentElement.requestFullscreen();
	}

	onMount(() => {
		let timer: number;
		(async () => {
			try {
				const result = await api.demoSubjects();
				subjects = result.demo_subjects;
				loading = false;
				await loadStep();
				if (aiAuto) void generateAINote(true);
			} catch (cause) {
				loading = false;
				error = cause instanceof Error ? cause.message : 'Could not load the demo cohort';
			}
		})();

		const onFullscreenChange = () => (fullscreen = Boolean(document.fullscreenElement));
		const onKeyDown = (event: KeyboardEvent) => {
			const target = event.target as HTMLElement | null;
			if (target?.matches('input, select, textarea')) return;
			if (event.code === 'Space') { event.preventDefault(); running = !running; }
			if (event.key === 'ArrowRight') { event.preventDefault(); next(); }
			if (event.key === 'ArrowLeft') { event.preventDefault(); previousStep(); }
			if (event.key.toLowerCase() === 'r') reset();
			if (event.key.toLowerCase() === 'f') void toggleFullscreen();
		};
		document.addEventListener('fullscreenchange', onFullscreenChange);
		window.addEventListener('keydown', onKeyDown);
		timer = window.setInterval(() => {
			if (!running) return;
			void next();
		}, 2600 / speed);
		return () => {
			window.clearInterval(timer);
			document.removeEventListener('fullscreenchange', onFullscreenChange);
			window.removeEventListener('keydown', onKeyDown);
		};
	});
</script>

<svelte:head><title>Guided Demo | Personalized Patient Recovery Trajectory</title><meta name="description" content="Real, model-backed retrospective replay walkthrough — Personalized Patient Recovery Trajectory V2." /></svelte:head>

<div class="demo-shell" class:danger={tone === 'danger'}>
	<div class="ambient" aria-hidden="true"></div>
	<header class="hero">
		<div>
			<span class="kicker"><i></i> RETROSPECTIVE SEQUENTIAL REPLAY · SYNTHETIC RESEARCH BENCHMARK</span>
			<h1>Watch the forecast<br />update, cutoff by cutoff.</h1>
			<p>A real, model-backed walkthrough across the demo cohort's legal replay cutoffs — every value below is a genuine <code>POST /predict</code> response from the frozen V2 pipeline, never scripted or precomputed. Not real-time clinical prediction.</p>
		</div>
		<div class="run-state"><span>LOOPS COMPLETE</span><strong>{loops}</strong><small>{running ? 'RUNNING' : 'PAUSED'} · {speed}× SPEED</small></div>
	</header>

	{#if loading}
		<div class="loading">Loading demo cohort…</div>
	{:else if error}
		<div class="notice">{error}</div>
	{:else if subject && current}
		<section class="controls" aria-label="Simulation controls">
			<button class="primary" onclick={() => (running = !running)}>{#if running}<Pause size={15} /> PAUSE{:else}<Play size={15} /> PLAY{/if}</button>
			<button onclick={previousStep}><SkipBack size={15} /> PREVIOUS</button>
			<button onclick={next}><SkipForward size={15} /> NEXT</button>
			<button onclick={reset}><RotateCcw size={15} /> RESET</button>
			<button onclick={toggleFullscreen}>{#if fullscreen}<Minimize2 size={15} /> EXIT FULLSCREEN{:else}<Maximize2 size={15} /> PRESENT{/if}</button>
			<label>SPEED <select bind:value={speed}><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select></label>
			<span class="clock">{subject.subject_id} · t{cutoffIndex}/{cutoffs.length - 1}</span>
		</section>
		<div class="shortcuts"><span><kbd>SPACE</kbd> PLAY/PAUSE</span><span><kbd>←</kbd><kbd>→</kbd> STEP</span><span><kbd>F</kbd> PRESENT</span><span><kbd>R</kbd> RESET</span></div>

		<section class="cohort-row">
			{#each subjects as s, i}
				<button class:active={i === subjectIndex} onclick={() => jumpToSubject(i)}>
					<span>0{i + 1}</span><b>{s.subject_id}</b><small>{s.cardiac_condition_group.replace('SYNTHETIC_', '')}</small>
				</button>
			{/each}
		</section>

		<section class="metric-grid">
			<article><span>CURRENT SOFA</span><strong>{current.current_sofa.toFixed(1)}<small>/24</small></strong><MiniSpark values={historySofa} color="#2bb8b0" height={28} width={140} /><footer>OBSERVED AT CUTOFF {cutoffs[cutoffIndex].slice(11, 16)}</footer></article>
			<article><span>PREDICTED +24H / +48H</span><strong>{current.recovery.sofa_hat_24h.toFixed(1)}<small>→ {current.recovery.sofa_hat_48h.toFixed(1)}</small></strong><MiniSpark values={historyRec24} color="#38bdf8" height={28} width={140} /><footer>INDEPENDENT RECONSTRUCTIONS</footer></article>
			<article><span>REMAINING ICU STAY</span><strong>{current.icu_stay_time.remaining_hours.toFixed(1)}<small>h</small></strong><MiniSpark values={historyIcu} color="#fbbf24" height={28} width={140} /><footer>POSTPROCESSED FORECAST</footer></article>
			<article class:alert={current.organ_support.alert}><span>SUPPORT RISK (24H)</span><strong class="state">{(current.organ_support.probability_24h * 100).toFixed(1)}<small>%</small></strong><MiniSpark values={historySupport} color="#fb7185" height={28} width={140} /><footer>{current.organ_support.alert ? 'ABOVE THRESHOLD' : 'BELOW THRESHOLD'}</footer></article>
		</section>

		<section class="pipeline" aria-label="System pipeline">
			<div class="active"><Activity size={17} /><span>01 · TRUNCATE</span><b>RAW HISTORY ≤ t</b></div><i>→</i>
			<div class="active"><Waypoints size={17} /><span>02 · FEATURES</span><b>V2 CANONICAL</b></div><i>→</i>
			<div class="active"><BrainCircuit size={17} /><span>03 · PREDICT</span><b>FROZEN XGBOOST</b></div><i>→</i>
			<div class="active"><Sparkles size={17} /><span>04 · EXPLAIN</span><b>TREESHAP + AI NOTE</b></div><i>→</i>
			<div class:active={current.organ_support.alert}><BellRing size={17} /><span>05 · THRESHOLD</span><b>SUPPORT ALERT</b></div>
		</section>

		<section class="main-grid">
			<div class="trend">
				<div class="signal-grid">
					<div class="trend-card">
						<header><span>SOFA / REPLAY SO FAR</span><b>OBSERVED</b></header>
						<MultiLine series={[{ name: 'Observed SOFA', color: '#2bb8b0', values: historySofa }]} xLabels={cutoffs.slice(0, cutoffIndex + 1).map((c) => c.slice(11, 16))} yMin={0} yMax={24} height={140} />
					</div>
					<div class="trend-card">
						<header><span>REMAINING ICU HOURS</span><b>FORECAST</b></header>
						<MultiLine series={[{ name: 'Remaining hours', color: '#fbbf24', values: historyIcu }]} xLabels={cutoffs.slice(0, cutoffIndex + 1).map((c) => c.slice(11, 16))} yMin={0} yMax={Math.max(24, ...historyIcu, 1)} height={140} />
					</div>
					<div class="trend-card">
						<header><span>SUPPORT PROBABILITY (24H)</span><b>VS THRESHOLD</b></header>
						<MultiLine
							series={[
								{ name: 'Calibrated probability', color: '#fb7185', values: historySupport },
								{ name: 'Frozen threshold', color: 'rgba(251,191,36,.55)', values: historySupport.map(() => current!.organ_support.threshold * 100) }
							]}
							xLabels={cutoffs.slice(0, cutoffIndex + 1).map((c) => c.slice(11, 16))} yMin={0} yMax={100} height={140}
						/>
					</div>
					<div class="trend-card">
						<header><span>RECOVERY +24H RECONSTRUCTION</span><b>SOFA_HAT</b></header>
						<MultiLine series={[{ name: 'SOFA at +24h', color: '#38bdf8', values: historyRec24 }]} xLabels={cutoffs.slice(0, cutoffIndex + 1).map((c) => c.slice(11, 16))} yMin={0} yMax={24} height={140} />
					</div>
				</div>
			</div>
			<aside class="reasoning">
				<div class="decision" class:blocked={current.organ_support.alert}>
					<span>SUPPORT THRESHOLD STATE</span>
					<strong>{current.organ_support.alert ? 'ABOVE THRESHOLD' : 'BELOW THRESHOLD'}</strong>
					<p>Calibrated probability {(current.organ_support.probability_24h * 100).toFixed(1)}% vs. frozen threshold {(current.organ_support.threshold * 100).toFixed(1)}%. Raw model score {(current.organ_support.raw_probability * 100).toFixed(1)}%.</p>
				</div>
				<div class="shap-radar-card">
					<span>RECOVERY+24 SHAP FACTORS</span>
					<RadarPlot values={radarData.values} labels={radarData.labels} color="#38bdf8" />
				</div>
				<div class="dq-row">
					<GaugeRing value={Math.round((current.data_quality.observed_feature_fraction ?? 0) * 100)} size={64} stroke={6} color="#2bb8b0" label="DATA" />
					<div class="dq-copy"><span>WINDOW COMPLETENESS</span><p>{current.data_quality.observed_bins}/{current.data_quality.total_bins} bins observed · {current.data_quality.padding_bins} pre-admission padding</p></div>
				</div>
				<div class="events"><span>REPLAY LOG</span>{#each eventLog as event}<div><i class={event.tone}></i><b>{event.time}</b><small>{event.label}</small></div>{/each}</div>
			</aside>
		</section>

		<section class="ai-note" class:loading={aiLoading}>
			<header>
				<div class="ai-note-title"><Sparkles size={15} /><span>AI RESEARCH NOTE</span><small>{aiNote?.model ?? 'openai/gpt-oss-120b'} · via Groq</small></div>
				<label class="ai-toggle"><input type="checkbox" bind:checked={aiAuto} /> AUTO-NARRATE EACH PATIENT</label>
				<button class="ai-btn" onclick={() => generateAINote(false)} disabled={aiLoading}><RadioTower size={13} /> {aiLoading ? 'SYNTHESIZING…' : 'NARRATE THIS MOMENT'}</button>
			</header>
			{#if aiLoading && !aiNote}
				<p class="ai-body ai-pending">Synthesizing the current forecast, SHAP drivers, and data completeness into a short research note…</p>
			{:else if aiNote?.status === 'OK' && aiNote.summary}
				<p class="ai-body">{aiNote.summary}</p>
				<footer class="ai-disclaimer">{aiNote.disclaimer}</footer>
			{:else}
				<p class="ai-body ai-muted">{aiError || 'No AI note generated yet for this cutoff — click "Narrate this moment."'}</p>
			{/if}
		</section>

		<section class="evidence">
			<header><div><span>REAL FINAL-TEST EVIDENCE</span><b>Frozen, one-time fresh-test evaluation — separate from this replay</b></div><a href="/research/analytics">OPEN MODEL PERFORMANCE ↗</a></header>
			<div class="evidence-grid">
				<article><span>RECOVERY +24 MAE</span><strong>1.074</strong><small>STAY-BALANCED, 95% CI</small></article>
				<article><span>ICU MEDIAN AE</span><strong>5.206h</strong><small>95% CI [5.06, 5.39]</small></article>
				<article><span>SUPPORT CALIBRATED AUPRC</span><strong>0.704</strong><small>95% CI [0.66, 0.74]</small></article>
				<article><span>FRESH-TEST SUBJECTS USED HERE</span><strong>0</strong><small>SEALED, STRUCTURALLY UNREACHABLE</small></article>
			</div>
		</section>

		<footer class="disclaimer"><b>SYNTHETIC RESEARCH BENCHMARK</b><span>Real model-backed predictions on demo-safe synthetic subjects. Not real patient data, not real-time clinical prediction, not a diagnosis or treatment recommendation.</span><a href="/research/analytics">VIEW FINAL EVALUATION →</a></footer>
	{/if}
</div>

<style>
	.demo-shell{--accent:#2bb8b0;--accent-rgb:43,184,176;position:relative;isolation:isolate;max-width:1450px;margin:0 auto;padding:0 24px 28px;color:#dce9e8}.demo-shell.danger{--accent:#fb7185;--accent-rgb:251,113,133}.ambient{position:fixed;z-index:-1;inset:-20%;pointer-events:none;background:radial-gradient(circle at 84% 20%,rgba(var(--accent-rgb),.10),transparent 28%),radial-gradient(circle at 15% 75%,rgba(14,165,233,.065),transparent 25%);transition:background .8s ease}.hero{display:flex;justify-content:space-between;gap:40px;padding:28px 0 42px}.kicker,.hero p,.run-state span,.run-state small,.controls,.shortcuts,.metric-grid span,.metric-grid footer,.reasoning span,.pipeline,.evidence,.disclaimer{font-family:'JetBrains Mono',monospace}.kicker{color:var(--accent);font-size:8px;letter-spacing:.18em}.kicker i{display:inline-block;width:6px;height:6px;margin-right:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent);animation:pulse 1.8s infinite}.demo-shell h1{margin:17px 0 15px;font:500 clamp(32px,4.5vw,60px)/.98 'Space Grotesk',sans-serif;letter-spacing:-.04em}.hero p{max-width:640px;margin:0;color:#71829a;font-size:11px;line-height:1.8}.hero code{color:#8fc6bf}.run-state{align-self:flex-end;min-width:180px;padding:18px;border-left:2px solid var(--accent);background:rgba(var(--accent-rgb),.06)}.run-state span,.run-state small{display:block;color:#64748b;font-size:8px;letter-spacing:.12em}.run-state strong{display:block;margin:9px 0;font:500 22px 'Space Grotesk',sans-serif}
	.loading,.notice{padding:20px 0;color:#64748b;font:10px 'JetBrains Mono',monospace}.notice{color:#fecdd3}
	.controls{display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding:11px;border:1px solid rgba(148,163,184,.14);background:#070d18;font-size:8px;letter-spacing:.08em}.controls button{display:flex;align-items:center;gap:7px;padding:9px 12px;border:1px solid #263546;color:#91a8b8;background:#0a1220;font:inherit;cursor:pointer}.controls button:hover,.controls .primary{border-color:var(--accent);color:#03110f;background:var(--accent)}.controls label{display:flex;align-items:center;gap:8px;margin-left:auto;color:#64748b}.controls select{padding:7px;border:1px solid #263546;color:#dce9e8;background:#0a1220;font:inherit}.clock{min-width:100px;color:var(--accent);text-align:right;font-size:9px}.shortcuts{display:flex;justify-content:flex-end;gap:15px;padding:8px 2px;color:#405168;font-size:6px;letter-spacing:.08em}.shortcuts kbd{padding:2px 4px;border:1px solid #263546;color:#71829a;background:#080f1b;font:inherit}
	.cohort-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}.cohort-row button{display:flex;flex-direction:column;gap:4px;padding:12px;border:1px solid rgba(148,163,184,.16);background:#080e1d;color:#94a3b8;text-align:left;cursor:pointer}.cohort-row button.active{border-color:rgba(43,184,176,.45);background:rgba(43,184,176,.07);color:#eef7f6}.cohort-row span{color:#53647b;font:7px 'JetBrains Mono',monospace}.cohort-row b{font:11px 'Space Grotesk',sans-serif}.cohort-row small{color:#64748b;font:8px 'JetBrains Mono',monospace}
	.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:18px 0 7px}.metric-grid article{min-height:118px;padding:16px;border:1px solid rgba(148,163,184,.14);background:#080f1b;transition:border-color .5s,background .5s}.metric-grid article.alert{border-color:rgba(251,113,133,.45);background:rgba(127,29,29,.14)}.metric-grid span,.metric-grid footer{display:block;color:#53647b;font-size:7px;letter-spacing:.12em}.metric-grid strong{display:block;margin:16px 0 14px;font:500 30px 'Space Grotesk',sans-serif}.metric-grid strong small{margin-left:6px;color:#64748b;font:8px 'JetBrains Mono',monospace}.metric-grid .state{color:var(--accent)}
	.pipeline{display:flex;align-items:stretch;gap:8px;margin:0 0 7px;padding:10px;border:1px solid rgba(148,163,184,.12);background:#050b14}.pipeline>div{display:grid;grid-template-columns:24px 1fr;grid-template-rows:auto auto;align-items:center;flex:1;padding:8px;color:#405168;border:1px solid rgba(148,163,184,.08)}.pipeline>div.active{color:var(--accent);border-color:rgba(var(--accent-rgb),.22);background:rgba(var(--accent-rgb),.04)}.pipeline svg{grid-row:span 2}.pipeline span{font-size:6px;letter-spacing:.12em}.pipeline b{color:#91a8b8;font-size:7px}.pipeline>i{align-self:center;color:#263546;font-style:normal}
	.main-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,.55fr);gap:7px}.trend-card{padding:16px;border:1px solid rgba(148,163,184,.14);background:#070d18}.trend-card header{display:flex;justify-content:space-between;margin-bottom:10px;color:#53647b;font:7px 'JetBrains Mono',monospace;letter-spacing:.1em}.trend-card header b{color:#dce9e8}.signal-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.reasoning{display:grid;gap:7px;align-content:start}.reasoning>div{padding:16px;border:1px solid rgba(148,163,184,.14);background:#080f1b}.decision{color:var(--accent)}.decision.blocked{color:#fb7185}.decision>span{display:inline-block;margin-bottom:8px;color:#64748b;font-size:7px;letter-spacing:.13em}.decision strong{display:block;margin:6px 0 9px;font:500 18px/1.1 'Space Grotesk',sans-serif}.decision p,.events p{margin:0;color:#71829a;font-size:10px;line-height:1.6}.events>span{display:block;margin-bottom:15px;color:#64748b;font-size:7px;letter-spacing:.13em}.events div{display:grid;grid-template-columns:8px 38px 1fr;align-items:center;gap:6px;margin:10px 0;font:7px 'JetBrains Mono',monospace}.events div i{width:5px;height:5px;border-radius:50%;background:#2bb8b0}.events div i.danger{background:#fb7185}.events div b{color:#53647b}.events div small{color:#91a8b8}
	.shap-radar-card{display:flex;flex-direction:column;align-items:center;gap:10px}.shap-radar-card>span{align-self:flex-start;color:#64748b;font:7px 'JetBrains Mono',monospace;letter-spacing:.13em}
	.dq-row{display:flex;align-items:center;gap:14px}.dq-copy span{display:block;margin-bottom:6px;color:#64748b;font:7px 'JetBrains Mono',monospace;letter-spacing:.12em}.dq-copy p{margin:0;color:#8fa3b8;font-size:9px;line-height:1.5}
	.metric-grid article{display:flex;flex-direction:column}.metric-grid article strong{margin-bottom:8px}.metric-grid article :global(svg){display:block;margin:2px 0 10px}
	.ai-note{margin-top:7px;padding:18px;border:1px solid rgba(56,189,248,.22);background:linear-gradient(135deg,rgba(56,189,248,.05),#070d18 46%);transition:border-color .4s}.ai-note.loading{border-color:rgba(56,189,248,.5)}.ai-note>header{display:flex;align-items:center;flex-wrap:wrap;gap:14px;margin-bottom:12px}.ai-note-title{display:flex;align-items:center;gap:8px;color:#38bdf8}.ai-note-title span{font:8px 'JetBrains Mono',monospace;letter-spacing:.14em}.ai-note-title small{color:#53647b;font:7px 'JetBrains Mono',monospace}.ai-toggle{display:flex;align-items:center;gap:6px;margin-left:auto;color:#71829a;font:7px 'JetBrains Mono',monospace;letter-spacing:.08em;cursor:pointer}.ai-toggle input{accent-color:#38bdf8}.ai-btn{display:flex;align-items:center;gap:7px;padding:8px 12px;border:1px solid rgba(56,189,248,.35);color:#38bdf8;background:rgba(56,189,248,.06);font:7px 'JetBrains Mono',monospace;letter-spacing:.1em;cursor:pointer}.ai-btn:disabled{opacity:.6;cursor:wait}.ai-body{margin:0;color:#c3d4e0;font-size:12px;line-height:1.75;max-width:980px}.ai-body.ai-pending{color:#64748b;font-style:italic}.ai-body.ai-muted{color:#64748b}.ai-disclaimer{margin-top:10px;color:#53647b;font:7px 'JetBrains Mono',monospace;letter-spacing:.08em}
	.evidence{margin-top:7px;border:1px solid rgba(43,184,176,.2);background:linear-gradient(135deg,rgba(43,184,176,.045),#070d18 42%)}.evidence>header{display:flex;justify-content:space-between;align-items:center;padding:13px 15px;border-bottom:1px solid rgba(148,163,184,.1)}.evidence>header div{display:grid;gap:4px}.evidence>header span{color:#2bb8b0;font-size:7px;letter-spacing:.14em}.evidence>header b{color:#64748b;font-size:7px;font-weight:400}.evidence>header a{color:#2bb8b0;font-size:7px;text-decoration:none}.evidence-grid{display:grid;grid-template-columns:repeat(4,1fr)}.evidence-grid article{padding:16px;border-right:1px solid rgba(148,163,184,.1)}.evidence-grid span,.evidence-grid small{display:block;color:#53647b;font-size:6px;letter-spacing:.11em}.evidence-grid strong{display:block;margin:10px 0 6px;color:#dce9e8;font:500 25px 'Space Grotesk',sans-serif}
	.disclaimer{display:flex;gap:14px;align-items:center;margin-top:18px;padding:14px;border:1px solid rgba(251,191,36,.22);color:#71829a;font-size:7px;letter-spacing:.08em}.disclaimer b{color:#fbbf24}.disclaimer a{margin-left:auto;color:#2bb8b0;text-decoration:none}
	@keyframes pulse{50%{opacity:.45;box-shadow:0 0 22px var(--accent)}}
	@media(max-width:950px){.main-grid{grid-template-columns:1fr}.metric-grid,.evidence-grid,.cohort-row{grid-template-columns:1fr 1fr}.pipeline>i{display:none}.pipeline{display:grid;grid-template-columns:1fr 1fr}}
	@media(max-width:680px){.demo-shell{padding:0 12px 20px}.hero{display:block}.run-state{margin-top:24px}.metric-grid,.evidence-grid,.cohort-row{grid-template-columns:1fr 1fr}.disclaimer,.evidence>header{align-items:flex-start;flex-direction:column}.disclaimer a{margin-left:0}.controls label{margin-left:0}.shortcuts{display:none}.pipeline{grid-template-columns:1fr}.evidence-grid article{border-bottom:1px solid rgba(148,163,184,.1)}.signal-grid{grid-template-columns:1fr}.ai-note>header{flex-direction:column;align-items:flex-start}.ai-toggle{margin-left:0}}
</style>
