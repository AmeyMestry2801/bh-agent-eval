# Model card: M2 crisis-flagging model (ALL DATA IS SYNTHETIC)

## Model
TF-IDF features (word 1-2 grams plus character 2-5 grams, `char_wb`) feeding a class-weighted logistic regression
(C = 10, chosen by GroupKFold on template families, train only). Platt scaling fitted on validation only. The crisis
flag is raw P(crisis) >= 0.0927 (calibrated 0.0349), the threshold minimising expected cost on validation with a missed
crisis costing 20x a false alarm. The configuration was frozen before the test set was touched.

## Intended use
A portfolio demonstration of an evaluation and monitoring harness for a behavioral-health intake triage task. **Not
intended for any clinical or operational use.** No real patient data was used.

## Data
Synthetic conversations generated from template families, with author-assumed class mix, hedging, typos,
speech-recognition noise, English/Spanish mixing, 2.5% adjacent-class label noise, and a drift regime from week 17.
Training: weeks 1-12 (2,202 rows, 96 crisis). Test: held-out families (758 rows, 39 crisis).

## Performance (test, synthetic)
Crisis recall 0.949 [0.831, 0.986] (37/39), precision 0.841, 2 missed crises and 7 false alarms, crisis PR-AUC 0.952
[0.821, 1.000], test Brier 0.0031 and ECE 0.0039 after Platt scaling. Pre-registered criterion 1 (recall >= 0.95) was
missed by one case. See `results/RESULTS.md`.

## Slices
Test: worst channel/language slice text 0.920 (-2.9 points, 25 crisis cases); Spanish-mixed has only 2 crisis cases.
Dev out-of-fold: Spanish-mixed 0.833 (-12.5 points, 12 crisis cases), violating criterion 2 by point estimate.

## Known failure modes
All 7 observed false negatives were recorded-crisis, clean-moderate cases (negated or passive ideation, hedging,
indirect phrasing). Under simulated drift, recall fell to 0.841 overall and 0.711 on new slang families.

## Caveats and recommendations
Results come from templated synthetic text and a 39-crisis test set; they do not transfer to real conversations. Any
real use would need real data, clinical review of labels and thresholds, subgroup evaluation with adequate samples, and
human oversight of every escalation decision.
