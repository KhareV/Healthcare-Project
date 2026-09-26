<script lang="ts">
  import '../app.css';
  import { SmoothCursor } from '$lib/components/magic/smooth-cursor';
  import { afterNavigate } from '$app/navigation';
  import { page } from '$app/state';
  import { MonitorPlay } from '@lucide/svelte';
  import { env } from '$env/dynamic/public';
  import { authFlag } from '$lib/stores/auth.svelte';
  import type { Snippet } from 'svelte';

  let { children }: { children: Snippet } = $props();

  // This app is a fully client-rendered SPA (ssr = false), so there is no
  // server-rendered HTML to hydrate auth state into — Clerk's own JS SDK
  // establishes session state client-side regardless. The real security
  // boundary (redirecting an unauthenticated request away from protected
  // routes before any client JS runs) lives in hooks.server.ts, which
  // separately checks both PUBLIC_CLERK_PUBLISHABLE_KEY and CLERK_SECRET_KEY.
  const publishableKey = env.PUBLIC_CLERK_PUBLISHABLE_KEY ?? '';
  const authConfigured = Boolean(publishableKey);
  authFlag.enabled = authConfigured;

  afterNavigate(({ to }) => {
    // Preserve native anchor navigation while making page-to-page transitions
    // settle into the same calm, editorial scroll used by the landing page.
    if (!to?.url.hash) window.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
  });
</script>

<SmoothCursor />

{#if authConfigured}
  {#await Promise.all([import('$lib/components/dashboard/ClerkRoot.svelte'), import('$lib/components/dashboard/AuthGate.svelte')]) then [{ default: ClerkRoot }, { default: AuthGate }]}
    <ClerkRoot {publishableKey}>
      <AuthGate>
        {@render children()}
      </AuthGate>
    </ClerkRoot>
  {/await}
{:else}
  {@render children()}
{/if}

{#if page.url.pathname !== '/demo'}
  <a class="reviewer-demo-launcher" href="/demo" aria-label="Launch the interactive reviewer demonstration">
    <span class="reviewer-demo-launcher__icon"><MonitorPlay size={17} /></span>
    <span class="reviewer-demo-launcher__copy"><small>INTERACTIVE</small><strong>Reviewer demo</strong></span>
    <span class="reviewer-demo-launcher__action">PLAY</span>
  </a>
{/if}

<style>
  .reviewer-demo-launcher {
    position: fixed; right: 22px; bottom: 22px; z-index: 1200; display: grid;
    grid-template-columns: 36px auto auto; align-items: center; gap: 11px; min-width: 222px;
    padding: 8px 10px 8px 8px; border: 1px solid rgba(83, 220, 207, .48);
    color: #e9fffc; background: rgba(3, 10, 20, .9); text-decoration: none;
    box-shadow: 0 18px 50px rgba(0, 0, 0, .45), 0 0 30px rgba(43, 184, 176, .11);
    backdrop-filter: blur(18px); font-family: 'JetBrains Mono', monospace;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
  }
  .reviewer-demo-launcher:hover { transform: translateY(-3px); border-color: #56d8cf; box-shadow: 0 22px 60px rgba(0,0,0,.5), 0 0 38px rgba(43,184,176,.2); }
  .reviewer-demo-launcher__icon { display: grid; width: 36px; height: 36px; place-items: center; color: #03110f; background: #48c9c0; }
  .reviewer-demo-launcher__copy { display: grid; gap: 3px; }
  .reviewer-demo-launcher__copy small { color: #688196; font-size: 6px; letter-spacing: .18em; }
  .reviewer-demo-launcher__copy strong { font-size: 9px; letter-spacing: .04em; text-transform: uppercase; }
  .reviewer-demo-launcher__action { margin-left: 6px; color: #48c9c0; font-size: 7px; letter-spacing: .12em; }
  @media (max-width: 620px) {
    .reviewer-demo-launcher { right: 12px; bottom: 12px; min-width: 0; grid-template-columns: 34px auto; }
    .reviewer-demo-launcher__action { display: none; }
  }
</style>
