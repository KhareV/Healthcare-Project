<script lang="ts">
  // Trajectory Copilot — a right-side drawer, always attached to the
  // patient/cutoff currently open elsewhere in the app. It never predicts
  // independently: every answer is the LLM interpreting an already-computed
  // /predict response (see api/v2_app.py's /assistant route). Secondary to
  // the core dashboard by design — the dashboard remains fully usable if
  // this is closed, unavailable, or the AI key is unset.
  import { copilotState, closeCopilot } from '$lib/stores/copilot.svelte';
  import { api } from '$lib/services/api';
  import { Sparkles, X, Send, Loader2 } from '@lucide/svelte';

  type Turn = { role: 'user' | 'assistant' | 'notice'; text: string };
  let turnsByContext = $state<Record<string, Turn[]>>({});
  let question = $state('');
  let loading = $state(false);

  const SUGGESTED_PROMPTS = [
    'What changed since the previous cutoff?',
    'Summarize the current trajectory.',
    'Why did the support-risk estimate increase?',
    'Explain the recovery forecast in simple language.',
    'Which factors were most influential?',
    'How complete is the available data?',
    'What does the remaining ICU-stay estimate mean?',
    'What are the limitations of this prediction?'
  ];

  const contextKey = $derived(`${copilotState.stayId ?? ''}|${copilotState.predictionTime ?? ''}`);
  const turns = $derived(turnsByContext[contextKey] ?? []);
  const hasContext = $derived(Boolean(copilotState.stayId && copilotState.predictionTime));

  function pushTurn(key: string, turn: Turn) {
    turnsByContext = { ...turnsByContext, [key]: [...(turnsByContext[key] ?? []), turn] };
  }

  async function ask(text: string) {
    if (!hasContext || loading) return;
    const key = contextKey;
    const { stayId, predictionTime, previousPredictionTime } = copilotState;
    pushTurn(key, { role: 'user', text });
    loading = true;
    try {
      const result = await api.assistant(stayId!, predictionTime!, text, previousPredictionTime ?? undefined);
      if (result.status === 'OK' && result.summary) {
        pushTurn(key, { role: 'assistant', text: result.summary });
      } else {
        pushTurn(key, { role: 'notice', text: result.error ?? 'Trajectory Copilot is unavailable right now — the forecasting dashboard is unaffected.' });
      }
    } catch (cause) {
      pushTurn(key, { role: 'notice', text: cause instanceof Error ? cause.message : 'Trajectory Copilot is unavailable right now.' });
    } finally {
      loading = false;
    }
  }

  function send() {
    const text = question.trim();
    if (!text) return;
    question = '';
    void ask(text);
  }

  $effect(() => {
    if (copilotState.open && hasContext && turns.length === 0) {
      void ask('Summarize the current trajectory, including anything that changed since the previous cutoff.');
    }
  });
</script>

{#if copilotState.open}
  <div class="copilot-scrim" role="presentation" onclick={closeCopilot}></div>
  <aside class="copilot-drawer" aria-label="Trajectory Copilot">
    <header>
      <div class="copilot-title"><Sparkles size={16} /><span>TRAJECTORY COPILOT</span></div>
      <button class="copilot-close" onclick={closeCopilot} aria-label="Close"><X size={16} /></button>
    </header>

    {#if hasContext}
      <div class="copilot-context">{copilotState.patientAlias} · {copilotState.predictionTime?.slice(0, 16).replace('T', ' ')}</div>
    {:else}
      <div class="copilot-context copilot-context--empty">No patient selected — open a patient on Patient Replay or AI + SHAP first.</div>
    {/if}

    <div class="copilot-body">
      {#each turns as turn}
        <div class="turn turn--{turn.role}">{turn.text}</div>
      {/each}
      {#if loading}<div class="turn turn--assistant turn--pending"><Loader2 size={13} class="spin" /> Thinking…</div>{/if}
      {#if !loading && turns.length === 0 && hasContext}<div class="turn turn--notice">Ask a question, or pick a suggested prompt below.</div>{/if}
    </div>

    {#if hasContext}
      <div class="copilot-suggestions">
        {#each SUGGESTED_PROMPTS as prompt}
          <button onclick={() => ask(prompt)} disabled={loading}>{prompt}</button>
        {/each}
      </div>
      <form class="copilot-input" onsubmit={(e) => { e.preventDefault(); send(); }}>
        <input type="text" placeholder="Ask about this patient's trajectory…" bind:value={question} disabled={loading} />
        <button type="submit" disabled={loading || !question.trim()} aria-label="Send"><Send size={15} /></button>
      </form>
    {/if}

    <footer class="copilot-disclaimer">
      AI-generated interpretation of already-computed forecasts. Not clinical advice, not a diagnosis, not a treatment recommendation. Synthetic research benchmark.
    </footer>
  </aside>
{/if}

<style>
  .copilot-scrim { position: fixed; inset: 0; z-index: 900; background: rgba(0,0,0,.5); }
  .copilot-drawer { position: fixed; top: 0; right: 0; bottom: 0; z-index: 901; width: min(420px, 100vw); display: flex; flex-direction: column; background: var(--nhm-surface); border-left: 1px solid var(--nhm-border); box-shadow: -30px 0 80px rgba(0,0,0,.5); }
  .copilot-drawer header { display: flex; align-items: center; justify-content: space-between; padding: 16px 18px; border-bottom: 1px solid var(--nhm-border); }
  .copilot-title { display: flex; align-items: center; gap: 8px; color: #38bdf8; font: 9px 'JetBrains Mono', monospace; letter-spacing: .14em; }
  .copilot-close { display: grid; place-items: center; padding: 4px; border: 0; background: transparent; color: var(--nhm-subtle); cursor: pointer; }
  .copilot-close:hover { color: var(--nhm-text); }
  .copilot-context { padding: 10px 18px; color: var(--nhm-text); font: 500 12px 'Space Grotesk', sans-serif; border-bottom: 1px solid var(--nhm-border); }
  .copilot-context--empty { color: var(--nhm-subtle); font: 11px Inter, sans-serif; }
  .copilot-body { flex: 1; overflow-y: auto; padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; }
  .turn { padding: 10px 12px; font-size: 12px; line-height: 1.65; }
  .turn--user { align-self: flex-end; max-width: 88%; background: rgba(43,184,176,.12); border: 1px solid rgba(43,184,176,.25); color: var(--nhm-text); }
  .turn--assistant { background: #0c1426; border: 1px solid var(--nhm-border); color: #dce9e8; }
  .turn--notice { color: #fbbf24; font-family: 'JetBrains Mono', monospace; font-size: 10px; }
  .turn--pending { display: flex; align-items: center; gap: 8px; color: var(--nhm-subtle); font-family: 'JetBrains Mono', monospace; font-size: 10px; }
  .turn--pending :global(.spin) { animation: copilot-spin .9s linear infinite; }
  @keyframes copilot-spin { to { transform: rotate(360deg); } }
  .copilot-suggestions { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 18px 12px; }
  .copilot-suggestions button { padding: 6px 10px; border: 1px solid var(--nhm-border); background: transparent; color: var(--nhm-subtle); font-size: 10px; cursor: pointer; }
  .copilot-suggestions button:hover:not(:disabled) { border-color: rgba(43,184,176,.4); color: var(--nhm-text); }
  .copilot-suggestions button:disabled { opacity: .5; cursor: default; }
  .copilot-input { display: flex; gap: 8px; padding: 12px 18px; border-top: 1px solid var(--nhm-border); }
  .copilot-input input { flex: 1; padding: 10px 12px; border: 1px solid var(--nhm-border); background: #0c1426; color: var(--nhm-text); font-size: 12px; }
  .copilot-input button { display: grid; place-items: center; padding: 0 14px; border: 1px solid var(--nhm-accent); background: var(--nhm-accent); color: #03110f; cursor: pointer; }
  .copilot-input button:disabled { opacity: .5; cursor: default; }
  .copilot-disclaimer { padding: 10px 18px; border-top: 1px solid var(--nhm-border); color: #53647b; font-size: 8px; line-height: 1.6; }
  @media (max-width: 480px) { .copilot-drawer { width: 100vw; } }
</style>
