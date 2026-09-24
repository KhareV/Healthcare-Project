<script lang="ts">
	import { onMount } from 'svelte';
	import SectionPage from '$lib/components/dashboard/SectionPage.svelte';
	import { api } from '$lib/services/api';
	let modelStatus: { ready?: boolean; model_version?: string; ecg?: { ready?: boolean; deployment_eligible?: boolean; input_contract?: string }; vitals?: { ready?: boolean; software_release_eligible?: boolean; clinical_use_eligible?: boolean; input_contract?: string; status_note?: string }; multimodal?: { ready?: boolean; deployment_eligible?: boolean; status_note?: string } } = {};
	let error = '';
	onMount(async () => { try { modelStatus = await api.model.getStatus() as typeof modelStatus; } catch (cause) { error = cause instanceof Error ? cause.message : 'Model status unavailable'; } });
</script>

<SectionPage eyebrow="MODEL / CENTRALIZED V2" title="Validated component serving." description="Locked ECG classification, non-circular PPG pulse estimation, measured-SpO₂ validation, and quality-gated system fusion. Research use only pending prospective hardware and clinical validation." endpoint="/model/status" />
<section class="status"><b>{modelStatus.ready ? 'READY' : 'CHECKING'}</b><span>{modelStatus.model_version ?? 'Loading model status…'}</span>{#if modelStatus.ecg}<small>ECG: {modelStatus.ecg.deployment_eligible ? 'SOFTWARE RELEASE' : 'RESEARCH ONLY'} · {modelStatus.ecg.input_contract}</small>{/if}{#if modelStatus.vitals}<small>PPG/SpO₂: {modelStatus.vitals.software_release_eligible ? 'SOFTWARE RELEASE' : 'RESEARCH ONLY'} · Clinical use: {modelStatus.vitals.clinical_use_eligible ? 'ELIGIBLE' : 'NOT YET ELIGIBLE'} · {modelStatus.vitals.status_note}</small>{/if}{#if modelStatus.multimodal}<small>Legacy learned fusion: {modelStatus.multimodal.deployment_eligible ? 'SOFTWARE RELEASE' : 'REJECTED / RESEARCH HISTORY'} · {modelStatus.multimodal.status_note}</small>{/if}{#if error}<small class="error">{error}</small>{/if}</section>

<style>.status{margin:20px auto;max-width:1100px;padding:18px;border:1px solid #27465b;color:#dce9ef;display:grid;gap:8px;background:#07121e}.status b{color:#38d8bd}.status small{color:#91a8b8}.error{color:#fb7185}</style>
