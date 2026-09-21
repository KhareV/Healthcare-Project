# Sanskruti Rebuild Phase 12 — Governed XGBoost Validation Search Review

## Scope and outcome

The first scientific XGBoost search on the final synthetic adult cardiac benchmark completed under `synthetic_xgb_phase12_validation_search_v1`. It used training rows for fitting and validation rows for early stopping and within-family ranking. The sealed test partition was not loaded, predicted, summarized, or evaluated. These results characterize a designed synthetic benchmark and are not evidence of bedside accuracy, clinical effectiveness, treatment effects, or real cardiac-population generalization.

## Frozen prerequisite and execution identity

- Phase-11 manifest: `artifacts/manifests/synthetic_phase11_xgb_input_manifest_v1.json` (`94e1f0a4772db362ad8e742a788b5de8bee3f9c96f47c3114e06a7cc7b2a80a8`)
- implementation commit: `9f089c89a1665b1eb278d4008f44a6ac9acb10ae`
- scientific execution commit: `d72e2c8220ee65afe777fddf09c017f1d4ab8a4b`
- search-space canonical SHA-256: `2c66be21b25afbc851f6c19a631c9eabb672963dad7f9825d77cd1df9b8dce10`
- scientific environment SHA-256: `81e0478f5f71429426dedb1785a56ec03bff34947c37309a3b7b503f2efdeaf5`
- split SHA-256: `3d6af5298f219f4608e5318ca09856642a1918a1154563ed0e8aec1785563cf6`
- feature schema SHA-256: `f5de442cf4d0bf96356ef4ce4a15156541ee17a075b487cbe0c72926cee7e156`
- flat feature-map SHA-256: `8c46f66dd32029ac5c02b6e99deafd8838de9bf7e499e81b8deb098de6970d5f`
- preprocessor SHA-256: `c28bfdfbd83f08de96c30da94557a4da158ce28fedda254c1823be7b27fd6eb0`

## Sampler, budget, retry, and early stopping

The sampler was `numpy_pcg64_latin_hypercube_v1` with master seed `20260921` and task seeds `{"icu_time": 3342649955, "organ_support": 2374832336, "recovery": 3352049619}`. All three independent candidate lists were materialized and hashed before candidate 1. Each contains exactly 30 unique configurations; recovery therefore produced 30 scientific candidate bundles and 60 horizon estimators, not a 60-configuration search. Maximum attempts per candidate were `3`. Early stopping used 50 rounds and ordinary row-weighted internal monitors; authoritative ranking used the existing stay-balanced external evaluators.

Candidate-list hashes:

{
  "icu_time": "735834d88d9e94442b22d7a7b4bca22801ddbe98cb3e278b59a13609d1b9f727",
  "organ_support": "52f563e9aadbd5064346f773532325b835666ebd33f2f6ca9f6b00e09ba810b0",
  "recovery": "812ef083b8827b1f3b5345bf0f095307ebbc97001b26b60df20dae607f28c2d0"
}

Failed candidates: `{"icu_time": [], "organ_support": [], "recovery": []}`. Retry details remain in the registry; retries do not consume new candidate slots.

## Frozen ranking and winners

- recovery: minimize validation stay-balanced MAE24 in original delta-SOFA units; tie-break MAE48, RMSE24, candidate ID. Winner: `xgb-recovery-014`.
- ICU time: minimize validation stay-balanced weighted median absolute error in remaining-current-episode hours; tie-break MAE, RMSE, candidate ID. Winner: `xgb-icu-time-028`.
- organ support: maximize validation stay-balanced AUPRC from raw uncalibrated probabilities; tie-break Brier, AUROC, candidate ID. Winner: `xgb-support-024`.

This is within-XGBoost-family validation selection only. It is not XGBoost-versus-GRU selection, serving selection, calibration, threshold selection, or final-test evaluation. `selected_models_v1.json` was not created or changed.

## Recovery candidates

| candidate_id | status | mae24 | mae48 | rmse24 | rmse48 | median_absolute_error24 | median_absolute_error48 |
|---|---|---|---|---|---|---|---|
| xgb-recovery-001 | COMPLETE | 1.162034386 | 1.564390382 | 1.466377253 | 1.883453682 | 0.9751627445 | 1.473293066 |
| xgb-recovery-002 | COMPLETE | 1.169439275 | 1.538037624 | 1.47587636 | 1.866909337 | 0.9630262852 | 1.417499512 |
| xgb-recovery-003 | COMPLETE | 1.177230789 | 1.526178321 | 1.487829188 | 1.871114629 | 0.9640623927 | 1.325347424 |
| xgb-recovery-004 | COMPLETE | 1.180269823 | 1.535589286 | 1.496693803 | 1.874941252 | 0.9793538302 | 1.414754987 |
| xgb-recovery-005 | COMPLETE | 1.172061485 | 1.577080328 | 1.473118389 | 1.912807599 | 0.983627677 | 1.494042754 |
| xgb-recovery-006 | COMPLETE | 1.170136389 | 1.575546759 | 1.483120929 | 1.912212523 | 0.9370274544 | 1.400388598 |
| xgb-recovery-007 | COMPLETE | 1.159621847 | 1.552756338 | 1.470486056 | 1.8676503 | 0.9276138544 | 1.309578419 |
| xgb-recovery-008 | COMPLETE | 1.165353448 | 1.539601027 | 1.480149434 | 1.870542466 | 0.9578961432 | 1.433703065 |
| xgb-recovery-009 | COMPLETE | 1.168267947 | 1.547022494 | 1.4762313 | 1.878166402 | 0.962236762 | 1.543034315 |
| xgb-recovery-010 | COMPLETE | 1.1610656 | 1.565924132 | 1.46052788 | 1.883896423 | 0.9567346573 | 1.589817643 |
| xgb-recovery-011 | COMPLETE | 1.160435034 | 1.56231663 | 1.46599325 | 1.885511318 | 0.9593465328 | 1.517363906 |
| xgb-recovery-012 | COMPLETE | 1.162862162 | 1.534659421 | 1.464225408 | 1.867827016 | 0.9628880024 | 1.317785621 |
| xgb-recovery-013 | COMPLETE | 1.162638677 | 1.557444273 | 1.462880062 | 1.879111488 | 0.9582879543 | 1.487469435 |
| xgb-recovery-014 | COMPLETE | 1.156266005 | 1.570273639 | 1.456975336 | 1.893129463 | 0.976353839 | 1.476561785 |
| xgb-recovery-015 | COMPLETE | 1.169407437 | 1.556813539 | 1.473961773 | 1.883337191 | 0.9575103521 | 1.485793352 |
| xgb-recovery-016 | COMPLETE | 1.158847706 | 1.574428773 | 1.46509887 | 1.900071748 | 0.9357850403 | 1.419810086 |
| xgb-recovery-017 | COMPLETE | 1.167885705 | 1.546382603 | 1.473158508 | 1.871704212 | 0.9551546574 | 1.48154676 |
| xgb-recovery-018 | COMPLETE | 1.16868796 | 1.568942286 | 1.476546465 | 1.890957399 | 0.9565380812 | 1.448121548 |
| xgb-recovery-019 | COMPLETE | 1.165696823 | 1.537265965 | 1.475224151 | 1.876308739 | 0.9541350007 | 1.446191698 |
| xgb-recovery-020 | COMPLETE | 1.161375009 | 1.587123997 | 1.464545688 | 1.897491019 | 0.9729758501 | 1.444830537 |
| xgb-recovery-021 | COMPLETE | 1.157925406 | 1.577120184 | 1.456827814 | 1.904164591 | 0.9283531904 | 1.424497336 |
| xgb-recovery-022 | COMPLETE | 1.171684132 | 1.543842713 | 1.474187263 | 1.873895971 | 0.9893882871 | 1.429120779 |
| xgb-recovery-023 | COMPLETE | 1.161111263 | 1.591726974 | 1.456798661 | 1.90903388 | 0.952758193 | 1.378922701 |
| xgb-recovery-024 | COMPLETE | 1.17055539 | 1.572250771 | 1.473777078 | 1.911640057 | 0.9789723605 | 1.450083137 |
| xgb-recovery-025 | COMPLETE | 1.174219527 | 1.567263969 | 1.471853721 | 1.881835988 | 0.9472543001 | 1.472133964 |
| xgb-recovery-026 | COMPLETE | 1.165772948 | 1.562254688 | 1.462039877 | 1.884459603 | 0.9667621851 | 1.447815418 |
| xgb-recovery-027 | COMPLETE | 1.161263656 | 1.585096445 | 1.458385521 | 1.902301124 | 0.9726595432 | 1.5739398 |
| xgb-recovery-028 | COMPLETE | 1.176713962 | 1.56114235 | 1.482508229 | 1.888830204 | 0.9953835011 | 1.4534024 |
| xgb-recovery-029 | COMPLETE | 1.159150345 | 1.56624905 | 1.459306041 | 1.894941735 | 0.9878827333 | 1.425743371 |
| xgb-recovery-030 | COMPLETE | 1.168495998 | 1.552080123 | 1.472255482 | 1.872336291 | 0.9712378979 | 1.469639659 |

Directional agreement remains unavailable when exact-zero target/prediction semantics trigger the existing frozen metric blocker; no new direction rule was invented.

## ICU-time candidates

| candidate_id | status | median_absolute_error_hours | mae_hours | rmse_hours |
|---|---|---|---|---|
| xgb-icu-time-001 | COMPLETE | 12.22753991 | 13.87365275 | 17.6123797 |
| xgb-icu-time-002 | COMPLETE | 12.1296346 | 13.98271537 | 17.74952259 |
| xgb-icu-time-003 | COMPLETE | 12.27178473 | 13.77392803 | 17.55738773 |
| xgb-icu-time-004 | COMPLETE | 12.30660135 | 13.84291018 | 17.5896322 |
| xgb-icu-time-005 | COMPLETE | 12.14070204 | 13.86474888 | 17.65643446 |
| xgb-icu-time-006 | COMPLETE | 12.19580557 | 13.85182661 | 17.72970994 |
| xgb-icu-time-007 | COMPLETE | 12.24057744 | 13.8416605 | 17.61170992 |
| xgb-icu-time-008 | COMPLETE | 12.45181172 | 13.8913061 | 17.63303707 |
| xgb-icu-time-009 | COMPLETE | 12.30449213 | 13.89928436 | 17.67263692 |
| xgb-icu-time-010 | COMPLETE | 12.32084029 | 13.97395395 | 17.74081308 |
| xgb-icu-time-011 | COMPLETE | 12.18287867 | 13.90059784 | 17.65889255 |
| xgb-icu-time-012 | COMPLETE | 12.36145288 | 13.85939444 | 17.63459896 |
| xgb-icu-time-013 | COMPLETE | 12.56227166 | 13.94490355 | 17.7278407 |
| xgb-icu-time-014 | COMPLETE | 12.36280513 | 13.97673456 | 17.77009572 |
| xgb-icu-time-015 | COMPLETE | 12.02744245 | 13.90632399 | 17.66948862 |
| xgb-icu-time-016 | COMPLETE | 12.21136262 | 13.93957045 | 17.72887355 |
| xgb-icu-time-017 | COMPLETE | 12.28443362 | 13.93293195 | 17.76075634 |
| xgb-icu-time-018 | COMPLETE | 12.1088097 | 13.87199051 | 17.67065612 |
| xgb-icu-time-019 | COMPLETE | 12.39396939 | 13.87621654 | 17.67315013 |
| xgb-icu-time-020 | COMPLETE | 12.22246835 | 13.82827511 | 17.60511088 |
| xgb-icu-time-021 | COMPLETE | 12.21251053 | 14.17160353 | 18.06562826 |
| xgb-icu-time-022 | COMPLETE | 12.26314126 | 13.9199299 | 17.71275438 |
| xgb-icu-time-023 | COMPLETE | 12.4928939 | 14.1882251 | 17.95676592 |
| xgb-icu-time-024 | COMPLETE | 12.16150659 | 13.86685937 | 17.63448243 |
| xgb-icu-time-025 | COMPLETE | 12.33742667 | 13.87254578 | 17.65470827 |
| xgb-icu-time-026 | COMPLETE | 12.39576223 | 13.87180003 | 17.62896597 |
| xgb-icu-time-027 | COMPLETE | 12.3226913 | 13.86046103 | 17.64060279 |
| xgb-icu-time-028 | COMPLETE | 11.81479956 | 13.87155754 | 17.6614794 |
| xgb-icu-time-029 | COMPLETE | 12.1944308 | 13.83713554 | 17.62540956 |
| xgb-icu-time-030 | COMPLETE | 12.04019129 | 13.95609601 | 17.78038601 |

## Organ-support candidates

| candidate_id | status | auprc | auroc | brier |
|---|---|---|---|---|
| xgb-support-001 | COMPLETE | 0.6419071638 | 0.7108555626 | 0.2140719514 |
| xgb-support-002 | COMPLETE | 0.6360868663 | 0.7007332961 | 0.2171171288 |
| xgb-support-003 | COMPLETE | 0.6430851296 | 0.7105661515 | 0.2144654497 |
| xgb-support-004 | COMPLETE | 0.6396838495 | 0.7089195595 | 0.2148970826 |
| xgb-support-005 | COMPLETE | 0.6145246861 | 0.6819691198 | 0.2220341543 |
| xgb-support-006 | COMPLETE | 0.63550706 | 0.7060351409 | 0.215624699 |
| xgb-support-007 | COMPLETE | 0.6420032411 | 0.7109356124 | 0.2145204708 |
| xgb-support-008 | COMPLETE | 0.6431227505 | 0.702757906 | 0.2166451233 |
| xgb-support-009 | COMPLETE | 0.6406510012 | 0.7101822426 | 0.2145890741 |
| xgb-support-010 | COMPLETE | 0.6404381868 | 0.7091981992 | 0.2153201525 |
| xgb-support-011 | COMPLETE | 0.6412614851 | 0.7043930568 | 0.2162943768 |
| xgb-support-012 | COMPLETE | 0.6291605184 | 0.7012998264 | 0.2179885063 |
| xgb-support-013 | COMPLETE | 0.6348187669 | 0.7043961598 | 0.2161566776 |
| xgb-support-014 | COMPLETE | 0.6327793223 | 0.7057783962 | 0.2159800481 |
| xgb-support-015 | COMPLETE | 0.6298073159 | 0.7037655993 | 0.2160158867 |
| xgb-support-016 | COMPLETE | 0.6193551838 | 0.6946952913 | 0.2198399626 |
| xgb-support-017 | COMPLETE | 0.615285306 | 0.6865526431 | 0.2220067064 |
| xgb-support-018 | COMPLETE | 0.6400515284 | 0.7080622916 | 0.2151177032 |
| xgb-support-019 | COMPLETE | 0.6389134801 | 0.700159122 | 0.2174997798 |
| xgb-support-020 | COMPLETE | 0.6344439841 | 0.6990299361 | 0.2182041043 |
| xgb-support-021 | COMPLETE | 0.6167155502 | 0.6932365262 | 0.2193139243 |
| xgb-support-022 | COMPLETE | 0.6412488741 | 0.7106818757 | 0.2149321647 |
| xgb-support-023 | COMPLETE | 0.6256735705 | 0.695925664 | 0.2197888769 |
| xgb-support-024 | COMPLETE | 0.6441122591 | 0.7116533279 | 0.2142527083 |
| xgb-support-025 | COMPLETE | 0.6359855271 | 0.7085704434 | 0.2160311099 |
| xgb-support-026 | COMPLETE | 0.6432734747 | 0.7039017365 | 0.2168022476 |
| xgb-support-027 | COMPLETE | 0.632383205 | 0.7022168736 | 0.2169604903 |
| xgb-support-028 | COMPLETE | 0.6350011381 | 0.7041249744 | 0.2170279888 |
| xgb-support-029 | COMPLETE | 0.6317158132 | 0.7001111818 | 0.217237767 |
| xgb-support-030 | COMPLETE | 0.6390774443 | 0.7019111646 | 0.2166674974 |

All support values are raw and uncalibrated. No threshold-dependent metric participated in ranking.

## Reproducibility, lineage, and isolation

Winner refits reused the same candidate identities without consuming scientific slots. Reproducibility status: `PASS`. Prediction and metric differences are recorded in `artifacts/search/xgb/phase12/winner_reproducibility_audit_v1.json`. Every completed candidate has registry, native-model, validation-prediction, metric, split, feature, preprocessing, environment, and candidate-list lineage. The Phase-12 audit verifies exact budget, artifact hashes, known producers, no unauthorized candidates, no scientific orphans, and `test_accessed=false`.

## Phase-13 boundary

Phase 12 does not claim G1 acceptance. Phase 13 owns consolidated data QA, leakage attacks, reproducibility acceptance, and G1/data-freeze readiness. No Phase-13 acceptance decision is implied here.
