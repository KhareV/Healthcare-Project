# Product Viva Notes

Short, direct answers. Cite the file when asked "how do you know."

**What makes the system personalized?**
The frozen models consume the selected patient's own longitudinal history —
the forecast, SHAP attribution, and Copilot narrative are all specific to
that patient's trajectory. Personalization here means "conditioned on this
patient's own data," not "a separately trained model per patient."

**Why is it time-series, not a static classifier?**
Because recovery is not a single fact — it evolves. The product's whole
point is replaying a stay cutoff by cutoff and recomputing four independent
forecasts from only the history available at that moment
(`data.timestamps.generate_prediction_rows_for_stay`).

**Why 48 hours / eight 6-hour bins?**
Frozen feature-contract decision (`configs/synthetic/feature_schema_v2.json`),
made in the scientific phase; not revisited here.

**Why repeated cutoffs instead of one prediction?**
To demonstrate — and let a user directly observe — that the model
recomputes from truncated history each time, never chains a later forecast
off an earlier one, and never uses information from beyond the cutoff.

**What does the AI assistant do — does it predict anything?**
No. Trajectory Copilot interprets an already-computed prediction in plain
language. It cannot alter, retry, or influence any model output. See
[`TRAJECTORY_COPILOT.md`](TRAJECTORY_COPILOT.md).

**Why is Clerk separate from custom-record ownership?**
Clerk answers "who is this user" (authentication). `owner_user_id` answers
"whose data is this" (authorization/ownership) — a fact derived from a
verified Clerk token, never supplied by the client. See
[`AUTH_AND_DATA_BOUNDARIES.md`](AUTH_AND_DATA_BOUNDARIES.md).

**How do you know cross-user isolation actually works, not just in
theory?**
`tests/test_v2_custom_record.py` creates two independent simulated users
against a real (locally-mocked) signature-verification path and asserts
User B gets 404 on User A's record on every touchpoint
(read/predict/history/assistant) — 30 backend tests total, plus a live
Playwright run signed in as a real Clerk test account through the actual
UI.

**Why 404 instead of 403 for an unauthorized custom-record request?**
So a caller can never distinguish "exists but isn't yours" from "never
existed" — the same fail-closed philosophy already used for the fresh-test
guard (`serving/v2/guard.py`).

**Why can a custom record express organ support but not urine output?**
Vasopressor/ventilation intervals fit the exact same interval shape the
frozen SOFA/feature-builder code already consumes for the demo corpus — no
scientific code changed, only more callers reach it. Urine output's renal
SOFA component requires a fully gapless 24-hour chain of interval readings;
episodic manual entry can't honestly provide that without fabricating
readings for hours nobody reported, so it's deliberately unsupported. See
[`CUSTOM_RECORD_FLOW.md`](CUSTOM_RECORD_FLOW.md).

**"Already active at the cutoff" vs. "newly initiated" — how is that
handled for a custom record?**
It isn't computed in serving at all — that distinction is training-label
semantics from the scientific phase. The custom-record feature only
provides a correct raw ON/OFF state per bin (verified against the frozen
`query_vasopressor_state`/`query_invasive_ventilation_state` functions in
`tests/test_v2_custom_record.py`); the frozen model already learned to read
"new" from the temporal ON/OFF pattern across bins.

**Why is support called "initiation" not "escalation"?**
Frozen label definition from the scientific phase — continuation of
already-active support is never counted as a new positive event.

**Why not hospital discharge ETA?**
The forecast is "remaining time in the current ICU stay," not a discharge
disposition or clinical decision — deliberately hedged wording throughout.

**How is SHAP used, and what does it not claim?**
TreeSHAP (`shap.TreeExplainer`) on each frozen XGBoost model's raw margin.
UI language is always "contributed to"/"influential," never "caused." For
the support task, SHAP explains the pre-calibration margin, not the
calibrated probability — stated explicitly in the Copilot's system prompt
and the AI + SHAP page's note.

**What does "data quality"/"readiness" mean here?**
Transparent structural facts (observed bins, missing concepts, SOFA
components represented) — never a confidence or certainty score. See
[`CUSTOM_RECORD_FLOW.md`](CUSTOM_RECORD_FLOW.md)'s readiness section.

**How was the final scientific test protected, and can it be rerun?**
It can't. `artifacts/performance_v2/governance/v2_fresh_test_access_state.json`
records a one-time-consumed access event; every serving path (demo, custom
records, Copilot) is structurally unable to reach that cohort — verified
by an explicit test in every test file that touches prediction
(`test_no_test_run_touched_fresh_test_access_state`) and by re-hashing the
frozen artifacts after every round of product work.

**Why is this not a medical device / not real-time?**
Retrospective sequential replay over a synthetic benchmark, stated on the
landing page, in every disclaimer banner, and in the Copilot's system
prompt. No real-time ingestion path exists anywhere in this product.
