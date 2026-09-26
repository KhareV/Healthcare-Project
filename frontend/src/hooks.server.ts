import { redirect, type Handle } from '@sveltejs/kit';
import { sequence } from '@sveltejs/kit/hooks';
import { env as publicEnv } from '$env/dynamic/public';
import { env as privateEnv } from '$env/dynamic/private';
import { withClerkHandler } from 'svelte-clerk/server';

// Authentication is optional at the infrastructure level: if no Clerk keys
// are configured, the app runs fully open with a visible "auth disabled"
// notice (see +layout.svelte) rather than crashing — the same graceful-
// degradation philosophy already used for the Groq/Kokoro integrations.
export const clerkConfigured = Boolean(publicEnv.PUBLIC_CLERK_PUBLISHABLE_KEY && privateEnv.CLERK_SECRET_KEY);

// Routes that require a signed-in session. The public landing page, sign-in/
// sign-up, and the onboarding flow itself stay reachable without a session
// (onboarding still requires auth, handled below) so an unauthenticated
// visitor always lands somewhere coherent rather than a raw redirect loop.
const PROTECTED_PREFIXES = [
	'/overview',
	'/patients',
	'/trends',
	'/research',
	'/ai',
	'/system',
	'/demo',
	'/signals',
	'/fl',
	'/monitor',
	'/monitoring',
	'/alerts',
	'/reports',
	'/onboarding'
];

function isProtectedPath(pathname: string): boolean {
	return PROTECTED_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

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
