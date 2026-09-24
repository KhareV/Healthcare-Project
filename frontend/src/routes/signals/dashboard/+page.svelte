<script lang="ts">
	import { onMount } from 'svelte';
	import WorkbenchPage from '$lib/components/dashboard/WorkbenchPage.svelte';
	import MetricTile from '$lib/components/dashboard/MetricTile.svelte';
	import TelemetryPlot from '$lib/components/dashboard/TelemetryPlot.svelte';
	import { api } from '$lib/services/api';
	import { connectLive } from '$lib/services/websocket';

	let monitoring = $state(false);
	let channel = $state('DISCONNECTED');
	let liveData = $state<Record<string, number>>({ hr: 0, spo2: 0, ecg_sqi: 0, ppg_sqi: 0 });
	let actionError = $state('');

	onMount(() => {
		const connection = connectLive('client_01', {
			onopen: () => (channel = 'LIVE'),
			onclose: () => (channel = 'CLOSED'),
			onerror: () => (channel = 'UNAVAILABLE'),
			onmessage: (event) => {
				try {
					const payload = JSON.parse(event.data);
					liveData = { ...liveData, ...payload };
				} catch {
					/* Ignore malformed frames */
				}
			}
		});
		return () => connection.close();
	});

	async function toggleMonitoring() {
		actionError = '';
		try {
			if (monitoring) await api.monitoring.stopSimulator('client_01');
			else await api.monitoring.startSimulator('client_01');
			monitoring = !monitoring;
		} catch (cause) {
			actionError = cause instanceof Error ? cause.message : 'Simulator action failed';
		}
	}
</script>

<svelte:head>
	<title>Live Signals Dashboard | NHM</title>
</svelte:head>

<WorkbenchPage eyebrow="02 / LIVE SIGNALS" title="Stay with the signal." description="Watch the wearable stream at the edge, inspect quality in real time, and keep the raw physiological context close to every inference.">
	<div class="control-line">
		<span><i class:active={channel === 'LIVE'}></i> CHANNEL / {channel}</span>
		<span>DEVICE / ESP32_001</span>
		<button class:stop={monitoring} on:click={toggleMonitoring}>
			{monitoring ? 'STOP SIMULATOR' : 'START SIMULATOR'} <b>{monitoring ? '■' : '↗'}</b>
		</button>
	</div>
	{#if actionError}
		<div class="error">{actionError}</div>
	{/if}
	<div class="live-grid">
		<div class="signal-stack">
			<TelemetryPlot label="ECG / live electrical signal" value={liveData.hr || (monitoring ? 74 : '--')} unit="BPM" detail="AD8232 / 250HZ" />
			<TelemetryPlot label="PPG / live optical pulse" mode="ppg" color="#0ea5e9" value={liveData.spo2 || (monitoring ? 98 : '--')} unit="%" detail="MAX30102 / 100HZ" />
		</div>
		<aside class="vitals">
			<MetricTile label="Heart rate" value={liveData.hr || (monitoring ? 74 : '--')} unit="BPM" tone="rose" />
			<MetricTile label="SpO₂" value={liveData.spo2 || (monitoring ? 98 : '--')} unit="%" tone="cyan" />
			<MetricTile label="ECG SQI" value={liveData.ecg_sqi || (monitoring ? 84 : '--')} unit="%" />
			<MetricTile label="PPG SQI" value={liveData.ppg_sqi || (monitoring ? 76 : '--')} unit="%" tone="cyan" />
		</aside>
	</div>
</WorkbenchPage>

<style>
	.control-line { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; padding: 14px 0; border-top: 1px solid rgba(148,163,184,.18); border-bottom: 1px solid rgba(148,163,184,.18); color: #64748b; font: 9px 'JetBrains Mono', monospace; letter-spacing: .1em; }
	.control-line i { display: inline-block; width: 7px; height: 7px; margin-right: 7px; border-radius: 5px; background: #64748b; }
	.control-line i.active { background: #2bb8b0; box-shadow: 0 0 12px #2bb8b0; }
	.control-line button { margin-left: auto; border: 1px solid #2bb8b0; padding: 11px 14px; color: #030712; background: #2bb8b0; font: 9px 'JetBrains Mono', monospace; letter-spacing: .08em; cursor: pointer; }
	.control-line button.stop { border-color: #fb7185; background: #fb7185; }
	.control-line button b { margin-left: 8px; }
	.error { margin-top: 18px; padding: 12px; border-left: 2px solid #fb7185; color: #fecdd3; background: rgba(127,29,29,.2); font: 11px 'JetBrains Mono', monospace; }
	.live-grid { display: grid; grid-template-columns: minmax(0,1fr) 280px; gap: 8px; margin-top: 28px; }
	.signal-stack { display: grid; gap: 8px; }
	.vitals { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; align-content: start; }
	@media (max-width: 850px) { .live-grid { grid-template-columns: 1fr; } .vitals { grid-template-columns: repeat(4,1fr); } }
	@media (max-width: 560px) { .vitals { grid-template-columns: 1fr 1fr; } }
</style>
