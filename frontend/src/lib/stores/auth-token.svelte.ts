// Reactive bearer-token holder, kept in sync by AuthGate.svelte (which has
// access to the live Clerk session via useClerkContext()). api.ts reads
// authToken.value synchronously on every request -- harmless to attach on
// calls the backend doesn't check (demo predictions), required for the
// backend to verify ownership on custom-record calls (see
// src/serving/v2/auth.py + the /custom-records* and /predict, /history,
// /assistant endpoints in api/v2_app.py).
export const authToken = $state({ value: null as string | null });
