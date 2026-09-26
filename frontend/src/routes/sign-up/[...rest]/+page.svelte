<script lang="ts">
  import { SignUp } from 'svelte-clerk';
  import { page } from '$app/state';

  const redirectUrl = page.url.searchParams.get('redirect_url') || '/onboarding';

  const appearance = {
    variables: {
      colorPrimary: '#2bb8b0',
      colorBackground: '#080e1d',
      colorText: '#eef7f6',
      colorTextSecondary: '#94a3b8',
      colorInputBackground: '#0c1426',
      colorInputText: '#eef7f6',
      borderRadius: '4px',
      fontFamily: 'Inter, sans-serif'
    },
    elements: {
      card: 'auth-card',
      headerTitle: 'auth-title',
      headerSubtitle: 'auth-subtitle'
    }
  };
</script>

<svelte:head><title>Sign Up | Personalized Patient Recovery Trajectory</title></svelte:head>

<div class="auth-shell">
  <div class="auth-brand">
    <a href="/"><strong>PRT · V2</strong></a>
    <p>Create a research-workspace account</p>
    <span class="auth-disclaimer">SYNTHETIC RESEARCH BENCHMARK · NOT A CLINICAL TOOL</span>
  </div>
  <div class="auth-widget">
    <SignUp {appearance} fallbackRedirectUrl={redirectUrl} />
  </div>
</div>

<style>
  .auth-shell {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 32px;
    padding: 40px 20px;
    background: var(--nhm-bg);
    color: var(--nhm-text);
    font-family: Inter, sans-serif;
  }
  .auth-brand { text-align: center; display: grid; gap: 10px; }
  .auth-brand a { color: var(--nhm-accent); text-decoration: none; font: 700 20px 'Space Grotesk', sans-serif; letter-spacing: .04em; }
  .auth-brand p { margin: 0; color: var(--nhm-subtle); font-size: 13px; }
  .auth-disclaimer { color: #53647b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .14em; }
  .auth-widget { display: flex; justify-content: center; }
  .auth-widget :global(.auth-card) { border: 1px solid var(--nhm-border); box-shadow: 0 40px 100px rgba(0,0,0,.5); }
</style>
