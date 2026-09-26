# Current UI Integration Audit (Product V2)

Written before/alongside the Product V2 "core loop" work, per the governing
UI-preservation directive: the existing SvelteKit dashboard (built earlier in
this same project's Performance-V2 frontend work) is the visual/product
source of truth. This audit records what already existed, what was reused,
and what was genuinely new, so the smallest coherent set of changes could be
made.

## 1. Current frontend framework

SvelteKit 5 (runes) + Vite 6 + Tailwind 4, `@sveltejs/adapter-static`
(`fallback: 'index.html'`, `prerender: true` at the root, `ssr = false` in
the root `+layout.ts`) — a fully client-rendered SPA. Dev workflow is
`npm run dev` (per the existing `RUNBOOK.md`); there is no production static
export in active use. **Decision (user-confirmed):** stay on SvelteKit, do
not migrate to Next.js as an earlier planning document suggested — the
current UI is the preserved source of truth.

## 2. Current route structure (before this work)

`/`(landing), `/overview`, `/patients`, `/trends`, `/research/analytics`,
`/ai/insights`, `/system/data`, `/demo`, plus ~15 inherited non-core
placeholder pages (`fl/*`, `signals/*`, `research/{ablation,comparison,...}`,
`system/{devices,health,notifications,settings}`, `monitor`, `monitoring`,
`alerts`, `reports`, `ai/{anomaly,baseline,confidence,model}`) that are not
in the active nav and were left untouched.

## 3. Current design system / component libraries

- Global CSS custom properties in `app.css`: `--nhm-bg`, `--nhm-surface`,
  `--nhm-surface-raised`, `--nhm-text`, `--nhm-muted`, `--nhm-subtle`,
  `--nhm-accent` (teal `#2bb8b0`), `--nhm-cyan`, `--nhm-border`.
- Fonts: Space Grotesk (headings), JetBrains Mono (labels/mono readouts),
  Inter (body).
- A "magic" component library (`animated-grid-pattern`, `blur-fade`,
  `dotted-map`, `globe`, `line-shadow-text`, `marquee`, `meteors`,
  `smooth-cursor`, `typing-animation`, etc.) used on the landing page.
- A dashboard component library in `src/lib/components/dashboard/`:
  `DashboardShell`, `WorkbenchPage`, `Panel`, `MetricTile`, `MultiLine`,
  `GaugeRing`, `BarContrib`, `RadarPlot`, `MiniSpark`, `TemporalHeatmap`,
  `SectionPage`, plus a few unused-until-now components (`ScatterField`,
  `InsightGraph`, `TelemetryPlot`, `SignalRibbon`).

**Decision:** reused these tokens and components exactly as-is everywhere a
new screen was needed (auth pages, onboarding, workspace cards, Copilot
drawer) rather than introducing shadcn/ui or any new component system.

## 4. Current navigation

`DashboardShell.svelte`'s left sidebar (`OVERVIEW`, `REPLAY`, `FORECAST`,
`PERFORMANCE`, `AI + SHAP`, `DATA QUALITY`, `DEMO`) plus a top header bar
with live API-health status. This was extended (not replaced) with a
"COPILOT" toggle button in the header and a user-identity block in the
sidebar footer.

## 5. Current dashboard structure

Each nav-linked page wraps its content in `WorkbenchPage` (which itself wraps
`DashboardShell`), except `/demo`, which is a standalone full-bleed
"guided demo" experience with its own chrome (not wrapped in
`DashboardShell`). This mattered for auth/onboarding wiring: a route-group
layout would have missed `/demo`, so the onboarding gate was placed in the
root layout instead (see §10) so it applies uniformly.

## 6. Reusable existing components (used as-is for this work)

`Panel`, `MetricTile`, `GaugeRing`, `RadarPlot`, `MiniSpark`, `WorkbenchPage`,
`DashboardShell` — all extended in place rather than duplicated.

## 7. Existing patient/time-series functionality (before this work)

Already fully real and model-backed (built in the prior Performance-V2
frontend phase, not re-done here): `/patients` (sequential replay: cutoff
stepper, recovery/ICU/support trend charts, temporal-window heatmap,
historical timeline, replay-history table), `/ai/insights` (TreeSHAP bars +
one-shot AI research note), `/research/analytics` (frozen Phase-4 metrics,
CIs, naive comparison, v1→v2 history), `/system/data` (data quality +
model/provenance metadata with a hash toggle), `/demo` (auto-replaying guided
walkthrough across the 3 demo subjects with live graphs and auto-narration).

## 8. What was genuinely missing (this work's scope)

- Authentication and session-gated routes (none existed).
- Onboarding (role, research disclaimer, data-source mode).
- A named authenticated workspace entry point with explicit data-source
  choices (demo data vs. future FHIR upload vs. future EHR sandbox).
- Human-readable patient aliases instead of raw synthetic subject/stay IDs.
- A grounded, multi-turn "Trajectory Copilot" (the existing AI feature was a
  single one-shot "Generate summary" note, not a Q&A drawer with suggested
  prompts and deterministic "what changed" grounding).

## 9. Components extended (not replaced)

- `DashboardShell.svelte` — added a Copilot header toggle, a lazy-loaded
  `UserMenu`, and the `TrajectoryCopilot` drawer mount point.
- `/overview/+page.svelte` — added a greeting row and three data-source
  cards above the existing project-summary content; everything below is
  unchanged.
- `/patients/+page.svelte` — subject list now shows `patient_alias`, and an
  "Ask Copilot" button + live copilot-context wiring were added to the
  existing replay controls; no existing chart/table was touched.
- `src/serving/v2/ai_recommendation.py` and `api/v2_app.py` — new functions
  and one new route (`/assistant`) added alongside the existing, already-
  tested `/ai/recommendation` path, which is untouched.

## 10. Genuinely new components/screens required

- `hooks.server.ts`, `+layout.svelte` Clerk wiring, `ClerkRoot.svelte`,
  `OnboardingGate.svelte`, `UserMenu.svelte`, `WorkspaceGreeting.svelte`
  (auth/session layer — no prior equivalent).
- `/sign-in/[...rest]`, `/sign-up/[...rest]`, `/onboarding` (new routes —
  no prior equivalent existed to extend).
- `TrajectoryCopilot.svelte` drawer + `stores/copilot.svelte.ts` (new — the
  prior AI feature had no drawer/chat surface to extend).
- `artifacts/performance_v2/product/demo_patients_v1.json` +
  `scripts/product_v2_build_demo_patients.py` (new product-layer artifact;
  does not touch any frozen scientific artifact).
- `configs/product_v2/trajectory_copilot_prompt_v1.txt` (new, versioned,
  separate from the existing one-shot research-note prompt).

## 11. Components explicitly NOT modified

`MultiLine`, `TemporalHeatmap`, `BarContrib`, `WorkbenchPage`'s internal
markup/CSS, the landing page's hero/marketing sections, the ~15 inherited
non-core placeholder pages, and every existing backend route
(`/predict`, `/history`, `/performance`, `/demo-subjects`'s existing shape,
`/ai/recommendation`) — all byte-for-byte unchanged in behavior (only
`/demo-subjects` gained one additive `patient_alias` field per subject).

## 12. Smallest-surface-area integration plan (what was actually built)

1. Server-side route protection (`hooks.server.ts`) gates the existing
   dashboard routes; auth is optional at the infrastructure level (no keys
   configured → app runs fully open, matching the graceful-degradation
   pattern already used for Groq/Kokoro).
2. Client-side `ClerkProvider` + a client-side onboarding nudge
   (`unsafeMetadata.onboardingComplete`), since this app is `ssr = false`
   end-to-end and a hard client-only UX gate is proportionate here.
3. One new authenticated workspace treatment layered onto the *existing*
   `/overview` page rather than a new `/workspace` route tree.
4. One new grounded AI surface (`Trajectory Copilot`) as a drawer reachable
   from the existing header and from the existing Patient Replay controls,
   not a new full-page chat screen.
5. Patient aliasing as a thin, additive product-layer artifact and one
   merged API field — the frozen demo manifest and its selection criteria
   are untouched.

## Explicitly deferred (user-agreed scope for this pass)

Synthetic FHIR upload/export + roundtrip contract tests, an EHR sandbox
connector abstraction, the full Playwright end-to-end suite, screenshot
evidence capture, and the remaining product docs (`PRODUCT_ARCHITECTURE.md`,
`SYNTHETIC_FHIR_PROFILE.md`, `TRAJECTORY_COPILOT.md`, `FINAL_DEMO_SCRIPT.md`,
`PRODUCT_VIVA_NOTES.md`) and the release manifest. These remain a coherent,
scoped follow-up; nothing above depends on them.
