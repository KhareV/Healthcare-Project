// Tiny reactive flag mirroring +layout.svelte's `authConfigured` so leaf
// components (DashboardShell's user menu) can decide whether it is safe to
// call Clerk hooks (which require being mounted under <ClerkProvider>)
// without threading a prop through every page.
export const authFlag = $state({ enabled: false });
