<script lang="ts">
	import { Globe } from '$lib/components/magic/globe';

	interface ArchitectureLayer {
		id: string;
		index: string;
		title: string;
		role: string;
		protocol: string;
		latency: string;
		dataThroughput: string;
		security: string;
		description: string;
		details: string[];
	}

	const layers: ArchitectureLayer[] = [
		{
			id: 'ingest',
			index: '01',
			title: 'Historical Replay Ingest',
			role: 'Cutoff-Bounded History Loader',
			protocol: 'REST /predict · JSON',
			latency: '< 5 ms',
			dataThroughput: 'Full Stay History',
			security: 'Guard-Enforced Demo Manifest',
			description: 'Loads the observed history for one synthetic ICU stay, truncated strictly at the selected replay cutoff — never future data.',
			details: ['Only 3 whitelisted demo stays are ever servable', 'Illegal cutoffs rejected before any computation', 'Fresh-test cohort walled off from live query']
		},
		{
			id: 'windowing',
			index: '02',
			title: 'Temporal Windowing & Masking',
			role: '48h Lookback / 8×6h Bins',
			protocol: 'Deterministic Binning',
			latency: '< 2 ms',
			dataThroughput: '8×N Observation Grid',
			security: 'Deterministic, No Randomness',
			description: 'Splits the 48-hour lookback into eight 6-hour bins per channel, building an observation/padding mask for exactly what was actually recorded.',
			details: ['Vitals, labs, and organ-support flags binned identically', 'Padding distinguished from true missingness', 'Same windowing logic used at train and test time']
		},
		{
			id: 'forecast',
			index: '03',
			title: 'Gradient-Boosted Forecast',
			role: 'Four Frozen XGBoost Heads',
			protocol: 'Frozen Model Registry',
			latency: '~12 ms / Cutoff',
			dataThroughput: '4 Independent Predictions',
			security: '100% Frozen — No Retuning',
			description: 'Four independently trained XGBoost models forecast ΔSOFA+24h, ΔSOFA+48h, remaining ICU hours, and new organ-support risk.',
			details: ['Every model trained once and frozen before the fresh test', 'No horizon is ever chained from another', 'Same frozen weights used across the whole replay']
		},
		{
			id: 'shap',
			index: '04',
			title: 'TreeSHAP Attribution',
			role: 'Per-Prediction Explanation',
			protocol: 'TreeExplainer (Exact)',
			latency: '~8 ms / Prediction',
			dataThroughput: 'Full Feature Contribution Vector',
			security: 'Additivity-Verified',
			description: 'Decomposes each prediction into signed per-feature contributions, verified to sum exactly to the model output.',
			details: ['Positive and negative contributors ranked and labeled', 'Additivity check runs on every single prediction', 'Never presented as causal — "contributed to", never "caused"']
		},
		{
			id: 'calibration',
			index: '05',
			title: 'Calibration & Bootstrap CI',
			role: 'Frozen One-Time Evaluation',
			protocol: 'Grouped Bootstrap (Stay-Level)',
			latency: 'Offline / One-Time',
			dataThroughput: '1,000 Synthetic ICU Stays',
			security: 'Access Consumed Exactly Once',
			description: 'The one-time fresh-test evaluation: metrics, calibration, and 95% bootstrap confidence intervals computed once from saved predictions and then frozen.',
			details: ['Support-risk probabilities calibrated before threshold comparison', 'Grouped bootstrap resamples by ICU stay, not by row', 'Access guard permits exactly one final run, then locks']
		},
		{
			id: 'ai_note',
			index: '06',
			title: 'AI Research-Note Synthesis',
			role: 'Groq-Hosted LLM Narration',
			protocol: 'openai/gpt-oss-120b · HTTPS',
			latency: '~1 – 3 s / Note',
			dataThroughput: 'Prediction + SHAP → Short Note',
			security: 'Server-Side Key, Never Browser-Exposed',
			description: 'Turns the frozen numbers and SHAP drivers into a short, hedged natural-language research note — generated strictly after prediction, never feeding back into it.',
			details: ['Non-diagnostic, non-causal system prompt enforced', 'Graceful degradation if the model is unavailable', 'Explicit "not clinical advice" disclaimer on every note']
		}
	];

	let activeLayer = $state(layers[2]);

	const federationMarkers = [
		{ lat: 28.61, lng: 77.21, size: 0.70 },
		{ lat: 19.07, lng: 72.88, size: 0.70 },
		{ lat: 13.08, lng: 80.27, size: 0.70 },
		{ lat: 12.97, lng: 77.59, size: 0.70 },
		{ lat: 1.35, lng: 103.82, size: 0.55 },
		{ lat: 35.68, lng: 139.65, size: 0.55 },
		{ lat: 51.51, lng: -0.13, size: 0.55 },
		{ lat: 40.71, lng: -74.01, size: 0.55 },
		{ lat: 37.77, lng: -122.42, size: 0.55 }
	];
</script>

<div class="arch-master-wrapper">
	<!-- 6-Tier Architecture Step Cards -->
	<div class="arch-tiers-strip">
		{#each layers as layer}
			<button
				class="tier-box"
				class:tier-box--active={activeLayer.id === layer.id}
				onclick={() => activeLayer = layer}
				type="button"
			>
				<span class="tier-idx">{layer.index}</span>
				<strong class="tier-title">{layer.title}</strong>
				<span class="tier-role">{layer.role}</span>
				<div class="tier-indicator"></div>
			</button>
		{/each}
	</div>

	<!-- Interactive Architectural Explorer Matrix -->
	<div class="arch-content-grid">
		<!-- Left: Detailed Tier Specs & Security Breakdown -->
		<div class="tier-detail-card">
			<div class="tier-header">
				<div class="tier-tag-group">
					<span class="tier-tag">TIER {activeLayer.index} // ARCHITECTURE</span>
					<span class="tier-latency">LATENCY: {activeLayer.latency}</span>
				</div>
				<h3>{activeLayer.title}</h3>
				<p class="tier-desc">{activeLayer.description}</p>
			</div>

			<div class="tier-metrics-grid">
				<div class="t-metric">
					<span>PROTOCOL / INTERFACE</span>
					<strong>{activeLayer.protocol}</strong>
				</div>
				<div class="t-metric">
					<span>THROUGHPUT / MEMORY</span>
					<strong>{activeLayer.dataThroughput}</strong>
				</div>
				<div class="t-metric">
					<span>PRIVACY & SECURITY</span>
					<strong class="text-teal">{activeLayer.security}</strong>
				</div>
			</div>

			<div class="tier-subsystems-list">
				<span class="subsys-label">SUBSYSTEM IMPLEMENTATION</span>
				<ul>
					{#each activeLayer.details as item}
						<li><span class="bullet">▹</span> {item}</li>
					{/each}
				</ul>
			</div>
		</div>

		<!-- Right: 3D Federated Global Intelligence Network -->
		<div class="federated-globe-card">
			<div class="globe-top-bar">
				<div>
					<span class="globe-tag">SYNTHETIC MULTI-SITE COHORT</span>
					<h4>One Frozen Model · Many Synthetic Sites</h4>
				</div>
				<span class="globe-status-pill"><span class="globe-dot"></span> REPLAY LIVE</span>
			</div>

			<!-- 3D Globe Visualizer -->
			<div class="globe-viewport">
				<Globe config={{ width: 520, height: 520, dark: 1, diffuse: 1.8, mapSamples: 12000, mapBrightness: 4.5, baseColor: [0.04, 0.12, 0.18], markerColor: [0.17, 0.83, 0.75], glowColor: [0.05, 0.25, 0.28], markers: federationMarkers.map(m => ({ location: [m.lat, m.lng], size: m.size * 0.05 })) }} />
			</div>

			<div class="globe-stats-dock">
				<div class="g-stat">
					<span>SYNTHETIC SITES</span>
					<strong>9 SITES</strong>
				</div>
				<div class="g-stat">
					<span>FRESH-TEST N</span>
					<strong>1,000 STAYS</strong>
				</div>
				<div class="g-stat">
					<span>SUPPORT AUPRC</span>
					<strong class="text-teal">0.7042</strong>
				</div>
				<div class="g-stat">
					<span>RECOVERY MAE</span>
					<strong class="text-teal">1.0744</strong>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	.arch-master-wrapper {
		width: 100%;
		display: flex;
		flex-direction: column;
		gap: 32px;
	}

	/* 6-Tier Strip */
	.arch-tiers-strip {
		display: grid;
		grid-template-columns: repeat(6, 1fr);
		gap: 12px;
	}

	.tier-box {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 6px;
		padding: 20px 16px;
		background: rgba(255, 255, 255, 0.02);
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 18px;
		color: #94a3b8;
		cursor: pointer;
		position: relative;
		overflow: hidden;
		transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
		text-align: left;
	}

	.tier-box:hover {
		background: rgba(255, 255, 255, 0.05);
		color: #ffffff;
		border-color: rgba(255, 255, 255, 0.2);
	}

	.tier-box--active {
		background: rgba(45, 212, 191, 0.1) !important;
		border-color: rgba(45, 212, 191, 0.4) !important;
		color: #ffffff !important;
	}

	.tier-idx {
		font-family: var(--font-mono, monospace);
		font-size: 11px;
		color: #2dd4bf;
		font-weight: 600;
	}

	.tier-title {
		font-size: 14px;
		font-weight: 600;
		color: #ffffff;
		line-height: 1.2;
	}

	.tier-role {
		font-size: 11px;
		color: #64748b;
		line-height: 1.3;
	}

	.tier-indicator {
		position: absolute;
		bottom: 0;
		left: 0;
		right: 0;
		height: 3px;
		background: transparent;
		transition: background 0.25s;
	}

	.tier-box--active .tier-indicator {
		background: #2dd4bf;
		box-shadow: 0 0 10px #2dd4bf;
	}

	/* Content Grid */
	.arch-content-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 32px;
		align-items: stretch;
	}

	.tier-detail-card, .federated-globe-card {
		background: #090d16;
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 28px;
		padding: 32px;
		box-shadow: 0 30px 80px rgba(0, 0, 0, 0.6);
		display: flex;
		flex-direction: column;
		justify-content: space-between;
	}

	.tier-tag-group {
		display: flex;
		justify-content: space-between;
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		margin-bottom: 8px;
	}
	.tier-tag { color: #2dd4bf; font-weight: 600; letter-spacing: 0.1em; }
	.tier-latency { color: #64748b; }

	.tier-header h3 {
		font-size: 26px;
		font-weight: 600;
		color: #ffffff;
		margin: 0 0 12px;
	}

	.tier-desc {
		font-size: 14px;
		color: #94a3b8;
		line-height: 1.6;
		margin: 0 0 24px;
	}

	.tier-metrics-grid {
		display: grid;
		grid-template-columns: 1fr;
		gap: 12px;
		margin-bottom: 24px;
	}

	.t-metric {
		padding: 12px 16px;
		background: rgba(255, 255, 255, 0.02);
		border: 1px solid rgba(255, 255, 255, 0.06);
		border-radius: 12px;
	}

	.t-metric span {
		display: block;
		font-family: var(--font-mono, monospace);
		font-size: 9px;
		color: #64748b;
		margin-bottom: 4px;
	}

	.t-metric strong {
		font-family: var(--font-mono, monospace);
		font-size: 13px;
		color: #ffffff;
	}

	.text-teal {
		color: #2dd4bf !important;
	}

	.tier-subsystems-list {
		border-top: 1px solid rgba(255, 255, 255, 0.06);
		padding-top: 18px;
	}

	.subsys-label {
		display: block;
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		color: #64748b;
		margin-bottom: 10px;
	}

	.tier-subsystems-list ul {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.tier-subsystems-list li {
		font-size: 13px;
		color: #cbd5e1;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.bullet {
		color: #2dd4bf;
		font-size: 12px;
	}

	/* Globe Card */
	.globe-top-bar {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-bottom: 16px;
	}

	.globe-tag {
		display: block;
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		color: #2dd4bf;
		letter-spacing: 0.1em;
		margin-bottom: 4px;
	}

	.globe-top-bar h4 {
		margin: 0;
		font-size: 18px;
		font-weight: 600;
		color: #ffffff;
	}

	.globe-status-pill {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		background: rgba(45, 212, 191, 0.1);
		border: 1px solid rgba(45, 212, 191, 0.3);
		border-radius: 999px;
		color: #2dd4bf;
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		font-weight: 600;
	}

	.globe-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: #2dd4bf;
		box-shadow: 0 0 8px #2dd4bf;
	}

	.globe-viewport {
		position: relative;
		height: 380px;
		display: grid;
		place-items: center;
		overflow: hidden;
		margin: 10px 0;
	}

	.globe-stats-dock {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 8px;
		padding-top: 16px;
		border-top: 1px solid rgba(255, 255, 255, 0.06);
	}

	.g-stat span {
		display: block;
		font-family: var(--font-mono, monospace);
		font-size: 9px;
		color: #64748b;
		margin-bottom: 2px;
	}

	.g-stat strong {
		font-family: var(--font-mono, monospace);
		font-size: 13px;
		color: #ffffff;
	}

	@media (max-width: 1050px) {
		.arch-tiers-strip {
			grid-template-columns: repeat(3, 1fr);
		}
		.arch-content-grid {
			grid-template-columns: 1fr;
		}
	}

	@media (max-width: 640px) {
		.arch-tiers-strip {
			grid-template-columns: repeat(2, 1fr);
		}
		.globe-stats-dock {
			grid-template-columns: repeat(2, 1fr);
			gap: 12px;
		}
	}
</style>
