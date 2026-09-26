<script lang="ts">
  // Only ever imported/rendered when auth is configured (see DashboardShell),
  // so it is always mounted under <ClerkProvider> and useClerkContext() is
  // safe to call here.
  import { useClerkContext, UserButton } from 'svelte-clerk';

  const ctx = useClerkContext();
  const role = $derived((ctx.user?.unsafeMetadata as Record<string, unknown> | undefined)?.role as string | undefined);
</script>

{#if ctx.user}
  <div class="user-menu">
    <UserButton />
    <div class="user-copy">
      <strong>{ctx.user.primaryEmailAddress?.emailAddress ?? ctx.user.username ?? 'Signed in'}</strong>
      {#if role}<small>{role.replace('_', ' ')}</small>{/if}
    </div>
  </div>
{/if}

<style>
  .user-menu { display: flex; align-items: center; gap: 10px; padding: 12px 18px; border-top: 1px solid var(--nhm-border); }
  .user-copy { display: grid; gap: 2px; min-width: 0; }
  .user-copy strong { color: var(--nhm-text); font: 500 10px 'Space Grotesk', sans-serif; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .user-copy small { color: #53647b; font: 7px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; }
</style>
