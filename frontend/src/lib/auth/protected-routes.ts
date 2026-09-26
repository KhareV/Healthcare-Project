// Shared between hooks.server.ts (the real security boundary, for direct/
// hard-reload requests) and AuthGate.svelte (the client-side guard needed
// because this app is a pure client-rendered SPA — ssr = false — so
// in-app link clicks are soft navigations that never hit the server hook).
// Keeping one list avoids the two guards drifting apart.
export const PROTECTED_PREFIXES = [
	'/overview',
	'/patients',
	'/health-record',
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

export function isProtectedPath(pathname: string): boolean {
	return PROTECTED_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}
