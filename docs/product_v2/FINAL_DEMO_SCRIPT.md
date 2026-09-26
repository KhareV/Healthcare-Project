# Final Demo Script

Prerequisites: both servers running (`RUNBOOK.md`), Clerk configured (real
or your own test account), browser at `http://localhost:5173/`.

## 4-minute version

1. **(15s) Landing.** Point at the disclaimer band: "RETROSPECTIVE
   SEQUENTIAL REPLAY · SYNTHETIC RESEARCH BENCHMARK · NOT REAL-TIME CLINICAL
   PREDICTION." Click "Open Research Workspace."
2. **(20s) Sign in.** Real Clerk widget — sign in (or sign up) with your
   account. State plainly this is genuine backend-verified authentication,
   not a UI mock.
3. **(15s) Onboarding — skip narration if already completed.** Otherwise:
   role → research disclaimer → "Explore Demo Patients" → done.
4. **(20s) Demo patient.** Land on Patient Replay. Point at the three
   `DEMO-CARDIAC-NNN` aliases and the current-cutoff status cards (SOFA,
   +24h/+48h forecast, remaining ICU time, support risk).
5. **(30s) Replay.** Click "Next" 2-3 times. Emphasize: every click is a
   fresh `POST /predict`, recomputed from history truncated at that exact
   cutoff — never a cache. Point at the recovery-trajectory chart's
   observed→forecast transition.
6. **(30s) TreeSHAP.** Switch to AI + SHAP. Point at one task's top
   positive/negative contributors and the additivity-check pass.
7. **(30s) Trajectory Copilot.** Open the drawer, let the auto-generated
   summary render, ask "What changed since the previous cutoff?" Point out
   the disclaimer and the refusal behavior is real (mention it, don't need
   to demo the refusal itself in the short version).
8. **(45s) Enter My Own Record.** From the workspace, open the custom-entry
   form. Use "Quick fill: deteriorating trajectory," submit, and land
   directly on a real prediction for that hand-entered patient — the same
   pipeline, same charts, same SHAP.
9. **(15s) Model Performance.** Point at the four frozen final-evaluation
   metric cards with their 95% CIs; note this page performs zero inference.
10. **(10s) Close.** Restate: synthetic benchmark, not clinically validated,
    not a real-time or diagnostic tool.

## 7-minute version

Everything above, plus:

- **(+30s) Organ-support entry.** On the custom-entry form, add a vasopressor
  interval (norepinephrine, some rate, "currently active") before
  submitting. On the resulting prediction, point out the temporal-window
  heatmap showing `vasopressor_on` transitioning ON at the entered hour —
  a real OFF→ON transition through the frozen support-state logic, not a
  cosmetic toggle.
- **(+30s) Data readiness, honestly.** On the same record's creation
  screen, point at the readiness summary: observations entered, SOFA
  components observed vs. missing, and the explicit note that urine output
  is unsupported (name the reason: SOFA's renal component needs a fully
  gapless 24-hour reading chain that manual entry can't honestly provide).
  Emphasize the wording is "data completeness," never "confidence."
- **(+30s) Cross-user isolation (technical audience).** Open dev tools or
  simply state it: every custom-record endpoint verifies a real Clerk
  session token server-side and checks it against the record's stored
  owner — a second browser signed in as a different user gets a 404 on
  someone else's record, indistinguishable from "never existed."
- **(+45s) Data Quality & Provenance.** Show the model/hash metadata panel
  and the temporal-window observation heatmap on a sparse record —
  reinforce that missingness is reported as a transparent fact, never a
  fake confidence score.
- **(+45s) Fresh-test protection.** Explain the sealed one-time evaluation:
  the fresh-test cohort structurally cannot be reached by this dashboard or
  the Copilot — same guard, same 404 behavior as an unauthorized custom
  record.
- **(+30s) What's deliberately not built.** Say plainly: FHIR/EHR upload is
  a documented future adapter boundary (`ExternalRecordAdapter`), not a
  working connector; the "Connect EHR Sandbox" card says "SOON" because it
  is not implemented.
