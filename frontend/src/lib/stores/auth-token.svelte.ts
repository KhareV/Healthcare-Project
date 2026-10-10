// Reactive bearer-token holder, kept in sync by AuthGate.svelte (which has
// access to the live Clerk session via useClerkContext()). api.ts reads
// authToken.value synchronously on every request -- harmless to attach on
// calls the backend doesn't check (demo predictions), required for the
// backend to verify ownership on custom-record calls (see
// src/serving/v2/auth.py + the /custom-records* and /predict, /history,
// /assistant endpoints in api/v2_app.py).
export const authToken = $state({
	value: null as string | null,
	// Registered by AuthGate: resolves a currently-valid session token (Clerk
	// caches it and only re-fetches near expiry). api.ts awaits this before each
	// call so a throttled background tab can't send an expired bearer token.
	refresh: null as null | (() => Promise<string | null>)
});

export async function freshAuthToken(): Promise<string | null> {
	try {
		if (authToken.refresh) {
			const token = await authToken.refresh();
			if (token) authToken.value = token;
		}
	} catch {
		/* fall back to the last known token */
	}
	return authToken.value;
}
