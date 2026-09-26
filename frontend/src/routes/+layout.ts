export const ssr = false;
// Prerendering every route at build time doesn't make sense once routes are
// per-session auth-gated (a "prerendered" protected page is a contradiction
// — there's no session to bake in). This app already runs as a client-only
// SPA (ssr = false) served via adapter-static's `fallback: 'index.html'`,
// so turning prerendering off just makes the build match how the app is
// actually used (npm run dev; see RUNBOOK.md) instead of static-exporting
// pages that would otherwise 500 during the build's prerender crawl.
export const prerender = false;