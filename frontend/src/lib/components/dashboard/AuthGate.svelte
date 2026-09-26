<script lang="ts">
  // The client-side half of route protection. hooks.server.ts covers direct/
  // hard-reload requests, but this app is ssr = false (fully client-
  // rendered), so clicking an <a> inside the app is a soft navigation that
  // never touches the server hook — without this, an already-open tab could
  // click through to a protected page with no server round-trip at all.
  //
  // Public routes render immediately, with zero wait on Clerk loading.
  // Protected routes hold rendering until Clerk has resolved the session,
  // then either redirect (not signed in -> /sign-in; signed in but
  // onboarding incomplete -> /onboarding) or render normally.
  import { useClerkContext } from 'svelte-clerk';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { isProtectedPath } from '$lib/auth/protected-routes';
  import type { Snippet } from 'svelte';

  let { children }: { children: Snippet } = $props();
  const ctx = useClerkContext();

  let ready = $state(false);

  $effect(() => {
    const path = page.url.pathname;

    if (!isProtectedPath(path)) {
      ready = true;
      return;
    }

    if (!ctx.isLoaded) {
      ready = false;
      return;
    }

    if (!ctx.auth.userId) {
      ready = false;
      void goto(`/sign-in?redirect_url=${encodeURIComponent(path + page.url.search)}`);
      return;
    }

    if (path !== '/onboarding') {
      const complete = (ctx.user?.unsafeMetadata as Record<string, unknown> | undefined)?.onboardingComplete === true;
      if (!complete) {
        ready = false;
        void goto(`/onboarding?redirect_url=${encodeURIComponent(path)}`);
        return;
      }
    }

    ready = true;
  });
</script>

{#if ready}
  {@render children()}
{:else}
  <div class="auth-gate-loading" aria-hidden="true"><span></span></div>
{/if}

<style>
  .auth-gate-loading { display: grid; place-items: center; min-height: 100vh; background: var(--nhm-bg, #030712); }
  .auth-gate-loading span { width: 28px; height: 28px; border-radius: 50%; border: 2px solid rgba(43,184,176,.2); border-top-color: #2bb8b0; animation: auth-gate-spin .8s linear infinite; }
  @keyframes auth-gate-spin { to { transform: rotate(360deg); } }
</style>
