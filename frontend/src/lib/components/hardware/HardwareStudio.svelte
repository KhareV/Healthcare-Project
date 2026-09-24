<script lang="ts">
	import { onMount } from 'svelte';
	import WatchScene from '$lib/components/landing/WatchScene.svelte';
	import PhysiologicalWaveform from '$lib/components/landing/PhysiologicalWaveform.svelte';

	interface Subsystem {
		id: string;
		index: string;
		name: string;
		tag: string;
		summary: string;
		waveformMode: 'ecg' | 'ppg' | 'spo2';
		specs: { key: string; val: string; note: string }[];
		highlights: string[];
	}

	const subsystems: Subsystem[] = [
		{
			id: 'recovery24',
			index: '01',
			name: 'Recovery-24h Forecast Head',
			tag: 'ΔSOFA AT +24 HOURS',
			summary: 'A gradient-boosted regression head, trained once and frozen, forecasting the +24h change in Sequential Organ Failure Assessment score from the fully observed history at each replay cutoff.',
			waveformMode: 'ecg',
			specs: [
				{ key: 'FRESH-TEST MAE', val: '1.0744', note: 'One-time frozen fresh-test evaluation' },
				{ key: '95% BOOTSTRAP CI', val: '[1.030, 1.120]', note: 'Grouped bootstrap by ICU stay' },
				{ key: 'TRAINING ROWS', val: '3,515 examples', note: '730 unique synthetic ICU stays' },
				{ key: 'MODEL FAMILY', val: 'XGBoost', note: 'Gradient-boosted decision trees' }
			],
			highlights: [
				'Never chained from the +48h head — each horizon is its own independent forecast',
				'Frozen immediately after the one-time fresh-test run; no post-test retuning',
				'TreeSHAP explains every prediction with a verified additivity check'
			]
		},
		{
			id: 'recovery48',
			index: '02',
			name: 'Recovery-48h Forecast Head',
			tag: 'ΔSOFA AT +48 HOURS',
			summary: 'The longer-horizon sibling of the +24h head — a separately trained gradient-boosted model forecasting +48h organ-failure trajectory, never derived by extrapolating the +24h estimate.',
			waveformMode: 'ppg',
			specs: [
				{ key: 'FRESH-TEST MAE', val: '1.3379', note: 'One-time frozen fresh-test evaluation' },
				{ key: 'WEIGHTED MEDIAN AE', val: '1.192', note: 'Per-stay weighted absolute error' },
				{ key: 'TRAINING ROWS', val: '1,331 examples', note: '319 unique synthetic ICU stays' },
				{ key: 'MODEL FAMILY', val: 'XGBoost', note: 'Gradient-boosted decision trees' }
			],
			highlights: [
				'Trained on its own +48h-eligible cohort, independent of the +24h sample',
				'Reported alongside its 95% grouped-bootstrap confidence interval, never a point estimate alone',
				'Same TreeSHAP attribution pipeline as every other forecast head'
			]
		},
		{
			id: 'icu_stay',
			index: '03',
			name: 'ICU-Stay Forecast Head',
			tag: 'REMAINING HOURS IN ICU',
			summary: 'Forecasts the hours remaining in the ICU stay from the current cutoff — a right-skewed duration target, evaluated primarily on median absolute error rather than MAE alone.',
			waveformMode: 'ecg',
			specs: [
				{ key: 'MEDIAN ABS. ERROR', val: '5.206h', note: 'One-time frozen fresh-test evaluation' },
				{ key: 'MAE', val: '7.809h', note: 'Right-skewed duration distribution' },
				{ key: 'TRAINING ROWS', val: '6,083 examples', note: '1,000 unique synthetic ICU stays' },
				{ key: 'MODEL FAMILY', val: 'XGBoost', note: 'Gradient-boosted regression' }
			],
			highlights: [
				'Recomputed fresh at every replay cutoff — never a single admission-day estimate',
				'Median absolute error reported alongside MAE to surface tail-case behavior',
				'Shares the same frozen-model governance as every other forecast head'
			]
		},
		{
			id: 'support_risk',
			index: '04',
			name: 'Support-Risk Forecast Head',
			tag: 'NEW SUPPORT INITIATION, 24H',
			summary: 'A calibrated classifier estimating the 24h probability of a NEW qualifying vasopressor or invasive mechanical-ventilation start — continuation of already-active support is never counted.',
			waveformMode: 'spo2',
			specs: [
				{ key: 'CALIBRATED AUPRC', val: '0.7042', note: 'One-time frozen fresh-test evaluation' },
				{ key: 'CALIBRATED AUROC', val: '0.7772', note: 'Post-hoc probability calibration' },
				{ key: 'TRAINING ROWS', val: '3,953 examples', note: '778 unique synthetic ICU stays' },
				{ key: 'MODEL FAMILY', val: 'XGBoost', note: 'Gradient-boosted classifier' }
			],
			highlights: [
				'Only a genuinely new support initiation counts as positive, never continuation',
				'Calibrated before the frozen threshold comparison — raw and calibrated AUPRC both reported',
				'TreeSHAP explains every prediction with a verified additivity check'
			]
		}
	];

	let active = $state(subsystems[0]);
	let liveBpm = $state(72);
	let liveSpo2 = $state(98);

	onMount(() => {
		const interval = setInterval(() => {
			liveBpm = 70 + Math.floor(Math.sin(Date.now() * 0.002) * 3);
			liveSpo2 = 98 + (Math.random() > 0.8 ? -1 : 0);
		}, 1800);
		return () => clearInterval(interval);
	});
</script>

<div class="hardware-pro-showcase">
	<!-- Left: 3D Model Explorer with Tactile Reticle Pins -->
	<div class="viewport-card">
		<div class="viewport-badge-row">
			<span class="pro-tag">PRT · V2 FROZEN MODEL SPECIFICATION</span>
			<span class="pro-meta">4 INDEPENDENT FORECAST HEADS · XGBOOST</span>
		</div>

		<!-- 3D Interactive Stage -->
		<div class="model-canvas-stage">
			<WatchScene />

			<!-- Clean Reticle Interactive Pins -->
			<div class="pins-overlay">
				{#each subsystems as sub, i}
					<button
						class="pin-button pin-button--{i + 1}"
						class:pin-button--active={active.id === sub.id}
						onclick={() => active = sub}
						type="button"
					>
						<span class="pin-dot"></span>
						<span class="pin-label">{sub.index} · {sub.name.split(' ')[0]}</span>
					</button>
				{/each}
			</div>
		</div>

		<!-- Horizontal Subsystem Tab Bar -->
		<div class="component-tab-bar">
			{#each subsystems as sub}
				<button
					class="component-tab"
					class:component-tab--active={active.id === sub.id}
					onclick={() => active = sub}
					type="button"
				>
					<span class="tab-number">{sub.index}</span>
					<span class="tab-title">{sub.name.split(' ')[0]}</span>
				</button>
			{/each}
		</div>
	</div>

	<!-- Right: Professional Engineering Blueprint & Spec Matrix -->
	<div class="blueprint-card">
		<div class="blueprint-header">
			<span class="blueprint-category">{active.tag}</span>
			<h3>{active.name}</h3>
			<p class="blueprint-summary">{active.summary}</p>
		</div>

		<!-- Live Signal Stream for Active Subsystem -->
		<div class="signal-preview-box">
			<div class="preview-topline">
				<span>LIVE STREAM // {active.name.split(' ')[0]}</span>
				<span class="live-pill"><span class="pulse-dot"></span> LIVE ACQUISITION</span>
			</div>
			<div class="preview-waveform">
				<PhysiologicalWaveform mode={active.waveformMode} speed={0.55} amplitude={0.65} />
			</div>
		</div>

		<!-- Engineering Specifications Matrix -->
		<div class="specs-grid">
			{#each active.specs as spec}
				<div class="spec-tile">
					<span class="spec-k">{spec.key}</span>
					<strong class="spec-v">{spec.val}</strong>
					<small class="spec-n">{spec.note}</small>
				</div>
			{/each}
		</div>

		<!-- Key Architectural Highlights -->
		<div class="highlights-box">
			<span class="highlights-label">ENGINEERING ARCHITECTURE</span>
			<ul>
				{#each active.highlights as item}
					<li>
						<span class="check-icon">✓</span>
						<span>{item}</span>
					</li>
				{/each}
			</ul>
		</div>
	</div>
</div>

<style>
	.hardware-pro-showcase {
		display: grid;
		grid-template-columns: 1.15fr 0.85fr;
		gap: 32px;
		align-items: stretch;
		width: 100%;
	}

	.viewport-card, .blueprint-card {
		background: #090d16;
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 28px;
		padding: 32px;
		box-shadow: 0 30px 80px rgba(0, 0, 0, 0.6);
		position: relative;
	}

	/* Left Viewport */
	.viewport-card {
		display: flex;
		flex-direction: column;
		min-height: 600px;
	}

	.viewport-badge-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		padding-bottom: 20px;
		border-bottom: 1px solid rgba(255, 255, 255, 0.06);
	}

	.pro-tag {
		color: #2dd4bf;
		font-weight: 600;
		letter-spacing: 0.1em;
	}
	.pro-meta {
		color: #64748b;
		letter-spacing: 0.08em;
	}

	.model-canvas-stage {
		flex: 1;
		position: relative;
		min-height: 420px;
		border-radius: 20px;
		background: radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.6) 0%, rgba(3, 7, 18, 0.9) 100%);
		border: 1px solid rgba(255, 255, 255, 0.04);
		overflow: hidden;
		margin: 20px 0;
	}

	/* Pins */
	.pins-overlay {
		position: absolute;
		inset: 0;
		pointer-events: none;
		z-index: 10;
	}

	.pin-button {
		position: absolute;
		pointer-events: auto;
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 6px 14px;
		background: rgba(15, 23, 42, 0.85);
		border: 1px solid rgba(255, 255, 255, 0.15);
		border-radius: 999px;
		color: #e2e8f0;
		font-family: var(--font-mono, monospace);
		font-size: 11px;
		font-weight: 500;
		cursor: pointer;
		backdrop-filter: blur(12px);
		transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
	}

	.pin-button:hover, .pin-button--active {
		background: rgba(45, 212, 191, 0.2);
		border-color: #2dd4bf;
		color: #ffffff;
		transform: scale(1.05);
		box-shadow: 0 0 20px rgba(45, 212, 191, 0.3);
	}

	.pin-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: #2dd4bf;
		box-shadow: 0 0 8px #2dd4bf;
	}

	.pin-button--1 { top: 18%; left: 8%; }
	.pin-button--2 { top: 62%; left: 10%; }
	.pin-button--3 { top: 22%; right: 8%; }
	.pin-button--4 { bottom: 16%; right: 12%; }

	/* Tab Bar */
	.component-tab-bar {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 10px;
	}

	.component-tab {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		padding: 12px 8px;
		background: rgba(255, 255, 255, 0.02);
		border: 1px solid rgba(255, 255, 255, 0.06);
		border-radius: 12px;
		color: #94a3b8;
		cursor: pointer;
		transition: all 0.2s;
	}

	.component-tab:hover {
		background: rgba(255, 255, 255, 0.05);
		color: #ffffff;
	}

	.component-tab--active {
		background: rgba(45, 212, 191, 0.1) !important;
		border-color: rgba(45, 212, 191, 0.4) !important;
		color: #2dd4bf !important;
	}

	.tab-number {
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		color: #2dd4bf;
	}
	.tab-title {
		font-size: 12px;
		font-weight: 600;
	}

	/* Right Blueprint */
	.blueprint-card {
		display: flex;
		flex-direction: column;
		justify-content: space-between;
	}

	.blueprint-category {
		display: inline-block;
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		color: #2dd4bf;
		letter-spacing: 0.15em;
		margin-bottom: 8px;
	}

	.blueprint-header h3 {
		font-size: 26px;
		font-weight: 600;
		color: #ffffff;
		margin: 0 0 12px;
		letter-spacing: -0.02em;
	}

	.blueprint-summary {
		font-size: 14px;
		color: #94a3b8;
		line-height: 1.6;
		margin: 0 0 24px;
	}

	.signal-preview-box {
		background: rgba(0, 0, 0, 0.5);
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 16px;
		padding: 16px;
		margin-bottom: 24px;
	}

	.preview-topline {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		color: #64748b;
		margin-bottom: 8px;
	}

	.live-pill {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		color: #2dd4bf;
		font-weight: 600;
	}

	.pulse-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: #2dd4bf;
		box-shadow: 0 0 8px #2dd4bf;
	}

	.preview-waveform {
		height: 64px;
		overflow: hidden;
	}

	/* Specs Grid */
	.specs-grid {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 12px;
		margin-bottom: 24px;
	}

	.spec-tile {
		padding: 12px 14px;
		background: rgba(255, 255, 255, 0.02);
		border: 1px solid rgba(255, 255, 255, 0.06);
		border-radius: 10px;
	}

	.spec-k {
		display: block;
		font-family: var(--font-mono, monospace);
		font-size: 9px;
		color: #64748b;
		margin-bottom: 4px;
		letter-spacing: 0.05em;
	}

	.spec-v {
		display: block;
		font-family: var(--font-mono, monospace);
		font-size: 15px;
		color: #ffffff;
		font-weight: 600;
		margin-bottom: 2px;
	}

	.spec-n {
		display: block;
		font-size: 11px;
		color: #94a3b8;
	}

	/* Highlights */
	.highlights-box {
		border-top: 1px solid rgba(255, 255, 255, 0.06);
		padding-top: 18px;
	}

	.highlights-label {
		display: block;
		font-family: var(--font-mono, monospace);
		font-size: 10px;
		color: #64748b;
		letter-spacing: 0.1em;
		margin-bottom: 10px;
	}

	.highlights-box ul {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.highlights-box li {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		font-size: 13px;
		color: #cbd5e1;
		line-height: 1.5;
	}

	.check-icon {
		color: #2dd4bf;
		font-weight: 700;
		font-size: 12px;
		margin-top: 2px;
	}

	@media (max-width: 1050px) {
		.hardware-pro-showcase {
			grid-template-columns: 1fr;
		}
	}
</style>
