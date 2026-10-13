# Error analysis: crisis false negatives (ALL DATA IS SYNTHETIC)

Scope: the 7 false negatives of the frozen model (M2, raw threshold 0.0927):
2 on the test set and 5 on dev out-of-fold predictions. Hand-tagged by the author.

## Headline
All 7 misses are label-noise cases: recorded label = crisis, clean label = moderate.
The model's class call is moderate-consistent on each. This agrees with the
generator's oracle ceiling (crisis recall 235/248 = 0.948 on synthetic data).
None has intact M1 crisis keywords, post-launch slang, or an ASR-driven garble.

## Taxonomy (n=7, descriptive only; counts are not rates)
| Tag | Count |
|---|---|
| Negation | 5 |
| Hedging | 1 |
| Other (indirect phrasing) | 1 |
| Slang | 0 |
| ASR error | 0 |

Secondary: language mix on 2 cases; 1 case within 0.0001 of the threshold.
Tags describe surface features, not causes. Negation may be over-represented
because negated-intent wording is where noisy crisis labels are easiest to confuse.

## Proposed mitigations (not yet evaluated)
1. Label QA: double-review crisis labels that contain negated intent or "no plan".
2. Review band: route passive-ideation text with P(crisis) just below the frozen
   threshold to human review. Cost (extra false alarms) must be measured on dev
   out-of-fold data before adoption.

## Limits
Seven cases cannot support rates or significance claims. Synthetic data, author
assumptions for noise; no clinical validity is implied.
