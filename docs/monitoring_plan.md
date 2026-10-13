# Monitoring and retraining plan (pre-registered before looking at the monitor window)

ALL DATA IS SYNTHETIC. Thresholds below are the author's assumptions, fixed before
running src/drift.py, and not tuned on monitor outcomes.

## What is monitored (weeks 17-26, frozen M2 at its frozen raw threshold)
Label-free (available immediately):
- PSI of the crisis score logit vs the validation window (10 reference-quantile bins).
- OOV rate: share of tokens not seen in the training vocabulary.
- Voice share, negation-token rate, flagged-crisis rate.
Label-dependent (available only after label_delay_weeks):
- Crisis recall and precision over a trailing 4-week window of label ARRIVALS, with Wilson CIs.

## Rules
- R1 WARNING (investigate): in 2 consecutive weeks, PSI > 0.25 OR OOV rate > 2x the validation OOV rate.
- R2 RETRAIN: in 2 consecutive weeks where the trailing window holds >= 10 labelled crisis
  cases, windowed crisis recall < 0.90 (the success target is 0.95; 0.90 leaves room for noise).
- Success of the monitor is judged against the generator's true drift start (week 17),
  which is known only because the data is synthetic.

## Retraining simulation
At the R2 week T (fallback: week 21, labelled "scheduled fallback" if R2 never fires):
refit M2 (same C) on train + val + monitor rows whose label has arrived by T; choose the
cost-minimising threshold on family-grouped out-of-fold predictions (FN=20, FP=1);
compare against the stale frozen model on monitor weeks > T, with a paired family-cluster
bootstrap of crisis PR-AUC.

## Known limitation
Drift families that appear before T also appear after T, so the retrained model's gain
measures recovery from a persisting shift, not generalisation to never-seen slang.
Per-window crisis counts are small, so recall estimates are wide.
