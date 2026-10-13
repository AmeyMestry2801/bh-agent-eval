# Split Strategy and Leakage Risks

> All numbers are on synthetic data.

## The split (src/split.py, seed 42)
| Split | Weeks | Rule | Rows | Crisis (recorded) |
|---|---|---|---|---|
| train | 1-12 | families not in test | 2,202 | 96 |
| val | 13-16 | families not in test | 736 | 25 |
| test | 1-16 | ~20% of families, held out whole | 758 | 39 |
| monitor | 17-26 | everything after week 16 | 2,304 | 88 |

Test families are chosen per class until ~20% of that class's weeks 1-16 rows
are held out, so the rare crisis class gets a test share close to target.
The test set is evaluated **once**, after every model, threshold and
calibration decision is frozen.

## Leakage risks and what we do about each
1. **Near-duplicate (family) leakage.** Conversations from one template family
   share a skeleton. If they land on both sides of a split, the model memorises
   skeletons and metrics inflate. *Control:* whole families go to test.
   *Measured:* the share of held-out rows with a training twin (TF-IDF cosine
   >= 0.8) is 1.7% for our family split versus 5.1% for a naive random split.
   *Residual risk:* families share core phrasing, so the leak is reduced, not
   removed. Mean max cosine only falls from 0.568 to 0.508.
2. **Validation shares families with train (by design).** The spec splits
   train/val by time, not by family: 224 families appear in both. Measured:
   val rows look as close to train as a naive random split does (0.053 vs
   0.051 twin share). *Consequence:* the validation set is optimistic. Scores
   calibrated and thresholds chosen on it may be over-confident on new
   families. The test set is the honest estimate, and we expect test
   performance to be lower than validation performance. If it is not, we say so.
   *Control:* hyperparameters are tuned with GroupKFold by family on train only.
3. **Small validation set.** Val holds only 25 recorded crisis cases, so the
   cost-minimising threshold will be noisy. We report threshold-versus-cost
   curves and a bootstrap range, not a single magic number.
4. **Temporal leakage.** Train uses weeks 1-12 only, val 13-16, so no model
   learns from the future of its own evaluation data. Weeks 17-26 are never
   touched until monitoring.
5. **Label-delay leakage.** Labels arrive 1-4 weeks late. In the retraining
   simulation we may only train on rows whose label has already arrived by the
   retraining date (week + label_delay_weeks <= retrain week).
6. **Preprocessing leakage.** TF-IDF, SVD, scalers and the calibrator are fit
   on the training portion only, inside pipelines, never on all data.
7. **Forbidden columns.** `week`, `template_family`, `risk_level_clean`,
   `gen_asr_wer`, `label_delay_weeks`, ids and labels are never features
   (list in `NON_FEATURE_COLUMNS` in src/split.py; models.py enforces it).
8. **Test touched once.** No re-tuning after seeing test results. Anything
   learned from the test set afterward is reported as exploratory.
9. **Exact duplicates.** 0 test texts also appear in train (4 val texts do).
   Checked automatically by `check_split_integrity`.
