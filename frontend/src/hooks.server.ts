import { redirect, type Handle } from '@sveltejs/kit';
import { sequence } from '@sveltejs/kit/hooks';
import { env as publicEnv } from '$env/dynamic/public';
import { env as privateEnv } from '$env/dynamic/private';
import { withClerkHandler } from 'svelte-clerk/server';
import { isProtectedPath } from '$lib/auth/protected-routes';

// Authentication is optional at the infrastructure level: if no Clerk keys
// are configured, the app runs fully open with a visible "auth disabled"
// notice (see +layout.svelte) rather than crashing — the same graceful-
// degradation philosophy already used for the Groq/Kokoro integrations.
export const clerkConfigured = Boolean(publicEnv.PUBLIC_CLERK_PUBLISHABLE_KEY && privateEnv.CLERK_SECRET_KEY);

// This only protects direct/hard-reload requests (this app is ssr=false, so
// in-app link clicks are client-side soft navigations that never reach this
// hook) — see AuthGate.svelte for the client-side guard that covers those.
const routeProtection: Handle = async ({ event, resolve }) => {
	event.locals.authConfigured = clerkConfigured;

	if (clerkConfigured && isProtectedPath(event.url.pathname)) {
		const auth = event.locals.auth?.();
		if (!auth?.userId) {
			const redirectTarget = encodeURIComponent(event.url.pathname + event.url.search);
			throw redirect(303, `/sign-in?redirect_url=${redirectTarget}`);
		}
	}

	return resolve(event);
};

export const handle: Handle = clerkConfigured ? sequence(withClerkHandler(), routeProtection) : routeProtection;
