<script lang="ts">
  // Thin wrapper so the root layout can `{#await import(...)}` this
  // component and keep the entire svelte-clerk client bundle out of the
  // page when authentication is not configured (see +layout.svelte).
  import { ClerkProvider } from 'svelte-clerk';
  import type { Snippet } from 'svelte';

  let { publishableKey, children }: { publishableKey: string; children: Snippet } = $props();
</script>

<ClerkProvider {publishableKey}>
  {@render children()}
</ClerkProvider>
