# Product limitations

- The V3 sealed evaluation is on independently generated synthetic subjects from one frozen generator. There is no prospective or external clinical validation.
- The V3.1 latent-state oracles use inaccessible generator internals. Their development scores describe an information gap; they are not serving candidates or clinical upper bounds.
- V3's ICU-time advantage over a naive constant is modest. V3 equals the repaired V2-recipe control for ICU time; recovery48's advantage over that control is not established at the stricter paired interval.
- V2 historical offline ICU/support evaluation contains defect D1. Only explicitly labeled V2-as-served results describe the authentic old deployment path.
- TreeSHAP explains model output, not causation. Support raw-margin attribution does not sum to the Platt-calibrated probability.
- A custom record can be technically processable while observationally sparse. Missingness is not an individual confidence measure.
- PDF report extraction proposes candidates only. Human confirmation assigns the encounter and event hour. Image reports lack OCR in the current report workflow.
- Groq commentary and local speech are optional. No LLM may modify a numerical forecast.
