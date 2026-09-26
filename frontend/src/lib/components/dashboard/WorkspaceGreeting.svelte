<script lang="ts">
  // Lazy-imported by /overview only when auth is configured (see that page),
  // so it is always mounted under <ClerkProvider>.
  import { useClerkContext } from 'svelte-clerk';

  const ctx = useClerkContext();
  const name = $derived(ctx.user?.firstName || ctx.user?.username || ctx.user?.primaryEmailAddress?.emailAddress || 'researcher');
  const role = $derived((ctx.user?.unsafeMetadata as Record<string, unknown> | undefined)?.role as string | undefined);
</script>

{#if ctx.user}
  <div class="greeting">
    <span>Welcome, {name}</span>
    {#if role}<small>{role.replace('_', ' ')} · SYNTHETIC RESEARCH WORKSPACE</small>{:else}<small>SYNTHETIC RESEARCH WORKSPACE</small>{/if}
  </div>
{/if}

<style>
  .greeting { margin-bottom: 16px; }
  .greeting span { display: block; color: #eef7f6; font: 500 16px 'Space Grotesk', sans-serif; }
  .greeting small { color: #53647b; font: 8px 'JetBrains Mono', monospace; letter-spacing: .12em; }
</style>
