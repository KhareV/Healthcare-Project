<script lang="ts">
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { useClerkContext } from 'svelte-clerk';
  import { Microscope, Stethoscope, GraduationCap, Check, Database, PencilLine, PlugZap } from '@lucide/svelte';

  const ctx = useClerkContext();

  const STEPS = ['WELCOME', 'ROLE', 'DISCLAIMER', 'DATA SOURCE', 'DONE'] as const;
  let step = $state(0);

  type Role = 'RESEARCHER' | 'CLINICIAN_DEMO' | 'EVALUATOR';
  let role = $state<Role | null>(null);

  let disclaimerAccepted = $state(false);

  type DataSourceMode = 'DEMO' | 'CUSTOM_ENTRY' | 'EHR_SANDBOX';
  let dataSourceMode = $state<DataSourceMode>('DEMO');

  let saving = $state(false);

  const redirectTarget = $derived(page.url.searchParams.get('redirect_url') || '/overview');

  function next() {
    step = Math.min(STEPS.length - 1, step + 1);
  }
  function back() {
    step = Math.max(0, step - 1);
  }

  async function finish() {
    saving = true;
    try {
      await ctx.user?.update({
        unsafeMetadata: {
          onboardingComplete: true,
          role,
          disclaimerAccepted: true,
          disclaimerAcceptedAt: new Date().toISOString(),
          dataSourceMode
        }
      });
    } finally {
      saving = false;
    }
    await goto(redirectTarget);
  }
</script>

<svelte:head><title>Onboarding | Personalized Patient Recovery Trajectory</title></svelte:head>

<div class="onboarding-shell">
  <div class="onboarding-card">
    <div class="onboarding-progress">
      {#each STEPS as label, i}
        <div class="progress-step" class:active={i === step} class:done={i < step}>
          <span class="progress-dot">{#if i < step}<Check size={11} />{:else}{i + 1}{/if}</span>
          <span class="progress-label">{label}</span>
        </div>
      {/each}
    </div>

    {#if step === 0}
      <section class="step">
        <span class="step-eyebrow">WELCOME</span>
        <h1>Personalized Patient Recovery Trajectory</h1>
        <p>
          This is a retrospective research prototype: it replays a synthetic ICU-like episode
          cutoff by cutoff, forecasting recovery trajectory, remaining ICU-stay time, and new
          organ-support initiation risk from four frozen gradient-boosted models — never a
          real-time clinical tool, never a lookup table.
        </p>
        <p class="step-note">SYNTHETIC RESEARCH BENCHMARK · NOT FOR CLINICAL DECISION-MAKING</p>
        <div class="step-actions"><button class="btn btn-primary" onclick={next}>Continue</button></div>
      </section>
    {:else if step === 1}
      <section class="step">
        <span class="step-eyebrow">ROLE</span>
        <h1>How will you use this workspace?</h1>
        <div class="role-grid">
          <button class="role-card" class:active={role === 'RESEARCHER'} onclick={() => (role = 'RESEARCHER')}>
            <Microscope size={20} /><strong>Clinical Researcher</strong><span>Evaluating forecasting methodology and evidence</span>
          </button>
          <button class="role-card" class:active={role === 'CLINICIAN_DEMO'} onclick={() => (role = 'CLINICIAN_DEMO')}>
            <Stethoscope size={20} /><strong>Clinician Demo User</strong><span>Exploring the concept in a demo/review capacity</span>
          </button>
          <button class="role-card" class:active={role === 'EVALUATOR'} onclick={() => (role = 'EVALUATOR')}>
            <GraduationCap size={20} /><strong>Academic Evaluator</strong><span>Reviewing the project for assessment</span>
          </button>
        </div>
        <div class="step-actions">
          <button class="btn" onclick={back}>Back</button>
          <button class="btn btn-primary" disabled={!role} onclick={next}>Continue</button>
        </div>
      </section>
    {:else if step === 2}
      <section class="step">
        <span class="step-eyebrow">RESEARCH ACKNOWLEDGEMENT</span>
        <h1>Before you continue</h1>
        <ul class="disclaimer-list">
          <li>This application is a retrospective, synthetic research prototype.</li>
          <li>Forecasts are not medical advice and do not constitute a diagnosis.</li>
          <li>The project has not undergone clinical or prospective validation.</li>
          <li>No treatment recommendation should ever be derived from this application.</li>
        </ul>
        <label class="ack-row">
          <input type="checkbox" bind:checked={disclaimerAccepted} />
          I understand and acknowledge the above.
        </label>
        <div class="step-actions">
          <button class="btn" onclick={back}>Back</button>
          <button class="btn btn-primary" disabled={!disclaimerAccepted} onclick={next}>Continue</button>
        </div>
      </section>
    {:else if step === 3}
      <section class="step">
        <span class="step-eyebrow">DATA SOURCE</span>
        <h1>How do you want to start?</h1>
        <div class="source-grid">
          <button class="source-card" class:active={dataSourceMode === 'DEMO'} onclick={() => (dataSourceMode = 'DEMO')}>
            <Database size={20} /><strong>Explore Demo Patients</strong><span>Recommended — three structurally-selected synthetic ICU episodes, ready now</span>
          </button>
          <button class="source-card" onclick={() => (dataSourceMode = 'CUSTOM_ENTRY')} class:active={dataSourceMode === 'CUSTOM_ENTRY'}>
            <PencilLine size={20} /><strong>Enter My Own Record</strong><span>Type in vitals & labs and forecast them through the real frozen models</span>
          </button>
          <button class="source-card" onclick={() => (dataSourceMode = 'EHR_SANDBOX')} class:active={dataSourceMode === 'EHR_SANDBOX'}>
            <PlugZap size={20} /><strong>Connect EHR Sandbox</strong><span>Advanced / sandbox — coming in a follow-up release</span>
            <em>SOON</em>
          </button>
        </div>
        <div class="step-actions">
          <button class="btn" onclick={back}>Back</button>
          <button class="btn btn-primary" onclick={next}>Continue</button>
        </div>
      </section>
    {:else}
      <section class="step">
        <span class="step-eyebrow">READY</span>
        <h1>You're set up.</h1>
        <p>Your research workspace is ready. You can change your data-source mode at any time from the workspace home.</p>
        <div class="step-actions">
          <button class="btn btn-primary" disabled={saving} onclick={finish}>{saving ? 'Saving…' : 'Enter workspace'}</button>
        </div>
      </section>
    {/if}
  </div>
</div>

<style>
  .onboarding-shell { min-height: 100vh; display: grid; place-items: center; padding: 40px 20px; background: var(--nhm-bg); color: var(--nhm-text); font-family: Inter, sans-serif; }
  .onboarding-card { width: min(640px, 100%); border: 1px solid var(--nhm-border); background: var(--nhm-surface); padding: clamp(24px, 4vw, 44px); }
  .onboarding-progress { display: flex; justify-content: space-between; margin-bottom: 36px; }
  .progress-step { display: flex; flex-direction: column; align-items: center; gap: 8px; flex: 1; color: #53647b; }
  .progress-dot { display: grid; place-items: center; width: 24px; height: 24px; border-radius: 50%; border: 1px solid rgba(148,163,184,.25); font: 9px 'JetBrains Mono', monospace; }
  .progress-step.active .progress-dot { border-color: var(--nhm-accent); color: var(--nhm-accent); box-shadow: 0 0 10px rgba(43,184,176,.4); }
  .progress-step.done .progress-dot { background: var(--nhm-accent); color: #03110f; border-color: var(--nhm-accent); }
  .progress-step.active .progress-label { color: var(--nhm-text); }
  .progress-label { font: 7px 'JetBrains Mono', monospace; letter-spacing: .1em; }
  .step-eyebrow { color: var(--nhm-accent); font: 8px 'JetBrains Mono', monospace; letter-spacing: .18em; }
  .step h1 { margin: 10px 0 16px; font: 500 clamp(24px,3vw,32px)/1.15 'Space Grotesk', sans-serif; letter-spacing: -.02em; }
  .step p { color: var(--nhm-muted); font-size: 14px; line-height: 1.7; margin: 0 0 12px; }
  .step-note { color: #fbbf24; font: 9px 'JetBrains Mono', monospace; letter-spacing: .1em; }
  .step-actions { display: flex; justify-content: flex-end; gap: 12px; margin-top: 28px; }
  .btn { padding: 12px 22px; border: 1px solid var(--nhm-border); background: transparent; color: var(--nhm-text); font: 10px 'JetBrains Mono', monospace; letter-spacing: .08em; cursor: pointer; }
  .btn-primary { border-color: var(--nhm-accent); background: var(--nhm-accent); color: #03110f; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .role-grid, .source-grid { display: grid; gap: 10px; margin-top: 4px; }
  .role-card, .source-card { position: relative; display: flex; flex-direction: column; align-items: flex-start; gap: 6px; padding: 16px; border: 1px solid var(--nhm-border); background: var(--nhm-surface-raised); color: var(--nhm-text); text-align: left; cursor: pointer; }
  .role-card:hover, .source-card:hover { border-color: rgba(43,184,176,.4); }
  .role-card.active, .source-card.active { border-color: var(--nhm-accent); background: rgba(43,184,176,.07); }
  .role-card strong, .source-card strong { font: 600 13px 'Space Grotesk', sans-serif; }
  .role-card span, .source-card span { color: var(--nhm-subtle); font-size: 11px; }
  .source-card em { position: absolute; top: 12px; right: 12px; color: #fbbf24; font: 7px 'JetBrains Mono', monospace; letter-spacing: .1em; font-style: normal; }
  .disclaimer-list { margin: 0 0 18px; padding-left: 18px; color: var(--nhm-muted); font-size: 13px; line-height: 1.9; }
  .ack-row { display: flex; align-items: center; gap: 10px; color: var(--nhm-text); font-size: 13px; }
  .ack-row input { accent-color: var(--nhm-accent); width: 16px; height: 16px; }
</style>
