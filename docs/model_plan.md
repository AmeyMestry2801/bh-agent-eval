# Model Plan and Promotion Rule (pre-registered before M2/M3 are trained)

> All numbers are on synthetic data. Written after rungs M0 and M1 were run on
> the validation split, and before any learned model (M2, M3) was trained.

## Ladder
| Rung | Model | Role |
|---|---|---|
| M0 | Majority class / class prior | floor |
| M1 | Hand-written keyword + regex rules (English only, negation window, no post-launch slang) | the "naive agent" |
| M2 | TF-IDF (word 1-2 grams + char_wb 2-5 grams) + multinomial Logistic Regression, class_weight="balanced" | default candidate |
| M3 | XGBoost on TF-IDF-SVD (100 comps) + engineered features (negation count, hedge words, first-person pronouns, text length, channel, OOV rate) | challenger |

The crisis score for every model is P(crisis) from a 4-class model.

## Tuning (train split only, never validation or test)
- M2: C in {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}, chosen by 5-fold GroupKFold
  (groups = template_family) on train, maximising mean out-of-fold crisis PR-AUC.
- M3: RandomizedSearchCV, fixed budget of 25 candidates, same GroupKFold(5) and
  metric, class-balanced sample weights, seed 42.
- All preprocessing (TF-IDF, SVD, scalers) is fit inside the CV folds.

## Promotion rule (M3 over M2)
Promote M3 over M2 **only if both hold**:
1. the 95% CI of the paired difference in crisis PR-AUC (M3 minus M2) **excludes 0**, and
2. the point estimate of the gain is **>= 0.02 absolute**.

Otherwise keep M2, and say why. Details fixed now:
- **Data for the decision:** out-of-fold predictions on train+val ("dev set",
  GroupKFold(5) by family, refitting each model per fold with its tuned
  hyperparameters). Reason: validation alone has 25 crisis cases, far too few
  for a paired CI, and out-of-fold predictions are family-disjoint, so they
  avoid the train/val family overlap. The validation-only comparison is
  reported as secondary and cannot overturn the decision.
- **CI method:** paired cluster bootstrap, 1,000 resamples, resampling whole
  template families with replacement (rows within a family are not independent,
  so a row-level bootstrap would give intervals that are too narrow).
- **Known bias:** the hyperparameters were tuned on rows that also appear in
  the dev out-of-fold set, which slightly favours the model with more tuning
  freedom (M3). That makes promotion easier than it should be, so a "keep M2"
  outcome is conservative and a "promote M3" outcome is read with caution.
- **M1 reference:** M2 is also compared with M1 using the same paired
  bootstrap, purely to report whether learning beat the naive rules.

## Calibration and threshold (winner only)
- Calibration: Platt scaling (sigmoid) on P(crisis), fitted on the validation
  split only. Isotonic regression is not used because validation has only 25
  crisis cases and isotonic overfits small samples.
- Report Brier score, ECE (10 equal-frequency bins) and the reliability curve.
  Calibration quality is judged on the test split (once) and on out-of-fold
  dev predictions, not on the data it was fitted on.
- Decision threshold (Phase 5): minimise expected cost (missed crisis = 20,
  false alarm = 1) on validation, apply once to test.
