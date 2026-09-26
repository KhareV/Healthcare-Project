<script lang="ts">
  // Client-side onboarding nudge: the hard security boundary (unauthenticated
  // users can't reach protected routes) is enforced server-side in
  // hooks.server.ts. This is a UX layer on top of that — redirect a signed-in
  // user who hasn't finished onboarding yet, tracked in Clerk's unsafeMetadata
  // (readable/writable client-side, no extra backend endpoint required).
  import { useClerkContext } from 'svelte-clerk';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';

  const ctx = useClerkContext();
  const EXEMPT_EXACT = ['/', '/onboarding'];

  $effect(() => {
    if (!ctx.isLoaded || !ctx.user) return;
    const path = page.url.pathname;
    if (EXEMPT_EXACT.includes(path) || path.startsWith('/sign-in') || path.startsWith('/sign-up')) return;

    const complete = (ctx.user.unsafeMetadata as Record<string, unknown> | undefined)?.onboardingComplete === true;
    if (!complete) {
      void goto(`/onboarding?redirect_url=${encodeURIComponent(path)}`);
    }
  });
</script>
