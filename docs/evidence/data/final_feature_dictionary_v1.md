# Final Feature Dictionary

All fields below are model-eligible dynamic channels; identities, labels, eligibility, split, and provenance metadata are non-predictive.

| Name | Unit | Source | Aggregation | Mask/TSLO | Leakage role |
|---|---|---|---|---|---|
| pao2__latest | mmHg | pao2 | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| fio2__latest | fraction | fio2 | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| platelet_count__latest | 10^9/L | platelet_count | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| bilirubin_total__latest | mg/dL | bilirubin_total | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| mean_arterial_pressure__latest | mmHg | mean_arterial_pressure | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| glasgow_coma_scale__latest | score | glasgow_coma_scale | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| creatinine__latest | mg/dL | creatinine | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| urine_output_volume__sum | mL | urine_output_volume | SUM_AVAILABLE_INTERVAL_VOLUMES | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| heart_rate__latest | beats/min | heart_rate | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| systolic_blood_pressure__latest | mmHg | systolic_blood_pressure | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| diastolic_blood_pressure__latest | mmHg | diastolic_blood_pressure | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| respiratory_rate__latest | breaths/min | respiratory_rate | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| oxygen_saturation__latest | percent | oxygen_saturation | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| temperature__latest | degree_Celsius | temperature | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| lactate__latest | mmol/L | lactate | LATEST | genuine evidence; frozen TSLO | MODEL_ELIGIBLE_DYNAMIC |
| vasopressor_on | binary | support_state | STATE_AT_BIN_END | known state; mask=true/TSLO=0 in active bins | MODEL_ELIGIBLE_DYNAMIC |
| invasive_ventilation_on | binary | support_state | STATE_AT_BIN_END | known state; mask=true/TSLO=0 in active bins | MODEL_ELIGIBLE_DYNAMIC |
| norepinephrine_rate | ug/kg/min | support_rate | STATE_AT_BIN_END | known state; mask=true/TSLO=0 in active bins | MODEL_ELIGIBLE_DYNAMIC |
| epinephrine_rate | ug/kg/min | support_rate | STATE_AT_BIN_END | known state; mask=true/TSLO=0 in active bins | MODEL_ELIGIBLE_DYNAMIC |
| dopamine_rate | ug/kg/min | support_rate | STATE_AT_BIN_END | known state; mask=true/TSLO=0 in active bins | MODEL_ELIGIBLE_DYNAMIC |
| dobutamine_rate | ug/kg/min | support_rate | STATE_AT_BIN_END | known state; mask=true/TSLO=0 in active bins | MODEL_ELIGIBLE_DYNAMIC |

Statics: age, sex category, cardiac condition group; train-fitted encoding only. Expected hard ranges and provenance are governed by the frozen raw/processed schemas and the listed provenance IDs.
