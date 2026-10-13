# Results

> ALL NUMBERS BELOW ARE ON SYNTHETIC DATA. They are not clinical validation and say nothing about real patients or real deployments.

Generated 2026-10-13T21:36:51+00:00 by `src/make_results.py` from the files in `results/`. Every number is read from the file named in its row.

## 1. Data and splits (synthetic)

| Quantity | Value | Script | File |
|---|---|---|---|
| Conversations | 6000 | generate_data.py / split.py | data/raw/conversations.csv |
| Template families | 399 | generate_data.py / split.py | data/raw/conversations.csv |
| Recorded class mix | no_risk 0.685, low 0.184, moderate 0.089, crisis 0.041 | generate_data.py / split.py | data/raw/conversations.csv |
| Label-noise rate (recorded != clean) | 0.0250 | generate_data.py / split.py | data/raw/conversations.csv |
| Oracle ceiling, crisis recall (clean label vs recorded) | 235/248 = 0.948 | generate_data.py / split.py | data/raw/conversations.csv |
| Split train | n=2202, crisis=96, weeks 1-12 | split.py | data/processed/conversations_split.csv |
| Split val | n=736, crisis=25, weeks 13-16 | split.py | data/processed/conversations_split.csv |
| Split test | n=758, crisis=39, weeks 1-16 | split.py | data/processed/conversations_split.csv |
| Split monitor | n=2304, crisis=88, weeks 17-26 | split.py | data/processed/conversations_split.csv |

## 2. Model selection (dev out-of-fold, pre-registered rule)

| Quantity | Value | Script | File |
|---|---|---|---|
| Winner | M2_tfidf_lr | promote.py | results/promotion.json |
| M3 minus M2 crisis PR-AUC | -0.327, 95% CI [-0.450, -0.220] | promote.py | results/promotion.json |
| M2 minus M1 crisis PR-AUC | +0.714, 95% CI [+0.608, +0.803] | promote.py | results/promotion.json |
| M2 selected C (GroupKFold by family, train only) | 10.000 | models.py | results/models/M2.joblib |
| M3 best params (25-candidate random search, train only) | {"xgb__subsample": 0.6, "xgb__reg_lambda": 0.5, "xgb__n_estimators": 200, "xgb__min_child_weight": 1, "xgb__max_depth": 5, "xgb__learning_rate": 0.05, "xgb__gamma": 0, "xgb__colsample_bytree": 0.7} | models.py | results/models/M3.joblib |

## 3. Frozen decision rule (set on validation before the test run)

| Quantity | Value | Script | File |
|---|---|---|---|
| Model | M2_tfidf_lr | calibrate.py | results/decision_config.json |
| Raw threshold (min expected cost, FN=20, FP=1) | 0.0927 | calibrate.py | results/decision_config.json |
| Platt a, b (validation) | 2.1122, 1.4978 | calibrate.py | results/decision_config.json |

## 4. Test set, evaluated once at frozen thresholds

**Headline metrics (PR-AUC CI: family-cluster bootstrap; recall/precision CI: Wilson)**  
_script:_ `final_test.py` | _file:_ `results/metrics_by_model_test.csv`

| model | n_crisis | pr_auc | pr_auc_lo | pr_auc_hi | recall | recall_lo | recall_hi | precision | fn | fp | cost_per_1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0_prior | 39 | 0.051 | 0.021 | 0.096 | 0.000 | 0.000 | 0.090 | n/a | 39 | 0 | 1029.024 |
| M1_rules | 39 | 0.266 | 0.045 | 0.475 | 0.308 | 0.186 | 0.464 | 0.750 | 27 | 4 | 717.678 |
| M2_tfidf_lr | 39 | 0.952 | 0.821 | 1.000 | 0.949 | 0.831 | 0.986 | 0.841 | 2 | 7 | 62.005 |
| M3_xgb | 39 | 0.833 | 0.572 | 0.944 | 0.821 | 0.673 | 0.910 | 0.582 | 7 | 23 | 215.040 |

**Exact McNemar tests on paired crisis detection (M2 vs the others)**  
_script:_ `final_test.py` | _file:_ `results/mcnemar_test.csv`

| a | b | a_only | b_only | discordant | p_value |
|---|---|---|---|---|---|
| M2_tfidf_lr | M1_rules | 25 | 0 | 25 | 5.96e-08 |
| M2_tfidf_lr | M3_xgb | 5 | 0 | 5 | 0.062 |

**Sensitivity to the assumed cost ratio (thresholds from dev out-of-fold)**  
_script:_ `final_test.py` | _file:_ `results/cost_ratio_sensitivity_test.csv`

| fn_to_fp_ratio | raw_threshold_from_dev_oof | test_recall | test_precision | flagged | fn | fp |
|---|---|---|---|---|---|---|
| 5 | 0.154 | 0.949 | 0.925 | 40 | 2 | 3 |
| 10 | 0.092 | 0.949 | 0.841 | 44 | 2 | 7 |
| 20 | 0.092 | 0.949 | 0.841 | 44 | 2 | 7 |
| 50 | 0.092 | 0.949 | 0.841 | 44 | 2 | 7 |
| 100 | 0.030 | 0.949 | 0.552 | 67 | 2 | 30 |

**M2 calibration on test (Platt fitted on validation only)**  
_script:_ `final_test.py` | _file:_ `results/test_summary.json`

| variant | brier | ece | mean_pred |
|---|---|---|---|
| raw_M2 | 0.004 | 0.009 | 0.052 |
| platt_M2 | 0.003 | 0.004 | 0.050 |

**M2 validation vs test (validation shares families with train, so it is optimistic)**  
_script:_ `final_test.py` | _file:_ `results/test_summary.json`

| val_pr_auc | test_pr_auc | val_recall | test_recall | val_precision | test_precision |
|---|---|---|---|---|---|
| 0.992 | 0.952 | 1.000 | 0.949 | 0.833 | 0.841 |

## 5. Pre-registered success criteria

| Quantity | Value | Script | File |
|---|---|---|---|
| Criterion 1: crisis recall >= 0.95 at precision >= 0.30 (M2, test) | NOT MET (recall 0.949, precision 0.841) | final_test.py | results/metrics_by_model_test.csv |
| Criterion 2 on dev out-of-fold (supplementary) | VIOLATED (point estimate); worst language=en_es, -12.5 pts, recall 0.833 on 12 crisis cases | slices.py | results/slice_metrics.csv |
| Criterion 2 on test (pre-registered) | MET (point estimate); worst channel=text, -2.9 pts, recall 0.92 on 25 crisis cases | slices.py | results/slice_metrics.csv |

**Slice metrics (recall with Wilson CI; gap vs overall in points)**  
_script:_ `slices.py` | _file:_ `results/slice_metrics.csv`

| dataset | variable | level | n | n_crisis | tp | fn | recall | recall_lo | recall_hi | gap_vs_overall_pts |
|---|---|---|---|---|---|---|---|---|---|---|
| test (pre-registered) | channel | text | 505 | 25 | 23 | 2 | 0.920 | 0.750 | 0.978 | -2.872 |
| test (pre-registered) | channel | voice | 253 | 14 | 14 | 0 | 1.000 | 0.785 | 1.000 | 5.128 |
| test (pre-registered) | language | en | 678 | 37 | 35 | 2 | 0.946 | 0.823 | 0.985 | -0.277 |
| test (pre-registered) | language | en_es | 80 | 2 | 2 | 0 | 1.000 | 0.342 | 1.000 | 5.128 |
| test (pre-registered) | age_band | 18-24 | 128 | 7 | 7 | 0 | 1.000 | 0.646 | 1.000 | 5.128 |
| test (pre-registered) | age_band | 25-34 | 177 | 9 | 8 | 1 | 0.889 | 0.565 | 0.980 | -5.983 |
| test (pre-registered) | age_band | 35-49 | 227 | 12 | 11 | 1 | 0.917 | 0.646 | 0.985 | -3.205 |
| test (pre-registered) | age_band | 50-64 | 143 | 7 | 7 | 0 | 1.000 | 0.646 | 1.000 | 5.128 |
| test (pre-registered) | age_band | 65+ | 83 | 4 | 4 | 0 | 1.000 | 0.510 | 1.000 | 5.128 |
| dev out-of-fold (supplementary) | channel | text | 1899 | 79 | 75 | 4 | 0.949 | 0.877 | 0.980 | -0.931 |
| dev out-of-fold (supplementary) | channel | voice | 1039 | 42 | 41 | 1 | 0.976 | 0.877 | 0.996 | 1.751 |
| dev out-of-fold (supplementary) | language | en | 2651 | 109 | 106 | 3 | 0.972 | 0.922 | 0.991 | 1.380 |
| dev out-of-fold (supplementary) | language | en_es | 287 | 12 | 10 | 2 | 0.833 | 0.552 | 0.953 | -12.534 |
| dev out-of-fold (supplementary) | age_band | 18-24 | 489 | 17 | 17 | 0 | 1.000 | 0.816 | 1.000 | 4.132 |
| dev out-of-fold (supplementary) | age_band | 25-34 | 799 | 35 | 33 | 2 | 0.943 | 0.814 | 0.984 | -1.582 |
| dev out-of-fold (supplementary) | age_band | 35-49 | 802 | 35 | 34 | 1 | 0.971 | 0.855 | 0.995 | 1.275 |
| dev out-of-fold (supplementary) | age_band | 50-64 | 530 | 26 | 25 | 1 | 0.962 | 0.811 | 0.993 | 0.286 |
| dev out-of-fold (supplementary) | age_band | 65+ | 318 | 8 | 7 | 1 | 0.875 | 0.529 | 0.978 | -8.368 |

## 6. Drift monitoring and retraining simulation (weeks 17-26)

| Quantity | Value | Script | File |
|---|---|---|---|
| Alert rules | {"psi_limit": 0.25, "oov_mult": 2.0, "recall_floor": 0.9, "min_window_crisis": 10, "window_weeks": 4} | drift.py | results/drift_summary.json |
| R1 warning week / R2 retrain week (drift starts week 17) | 18 / 20 | drift.py | results/drift_summary.json |
| Frozen M2 crisis recall: validation vs monitor | 1.000 vs 0.841 | drift.py | results/drift_summary.json |
| Retrain at week 20: recall stale -> retrained | 0.863 -> 0.961 | drift.py | results/drift_summary.json |
| Retrain: cost per 1000 stale -> retrained | 146.4 -> 65.2 | drift.py | results/drift_summary.json |
| Retrain: PR-AUC difference (retrained - stale), 95% family-cluster CI | +0.075 [-0.050, +0.182] | drift.py | results/drift_summary.json |

**Frozen M2 on the monitor window by family type**  
_script:_ `drift.py` | _file:_ `results/drift_by_family_type.csv`

| group | n | n_crisis | recall | recall_lo | recall_hi | precision |
|---|---|---|---|---|---|---|
| new D-* families | 826 | 38 | 0.711 | 0.552 | 0.830 | 0.243 |
| old families | 1478 | 50 | 0.940 | 0.838 | 0.979 | 0.734 |

**Weekly monitoring table**  
_script:_ `drift.py` | _file:_ `results/drift_weekly.csv`

| week | n | voice_share | oov_rate | flag_rate | psi_score | win_n_crisis | win_recall | retro_n_crisis | retro_recall |
|---|---|---|---|---|---|---|---|---|---|
| 17 | 231 | 0.528 | 0.063 | 0.091 | 0.250 | 0 | n/a | 10 | 0.800 |
| 18 | 231 | 0.550 | 0.047 | 0.052 | 0.181 | 4 | 0.500 | 9 | 0.556 |
| 19 | 231 | 0.528 | 0.051 | 0.056 | 0.186 | 14 | 0.714 | 6 | 1.000 |
| 20 | 231 | 0.545 | 0.051 | 0.100 | 0.250 | 20 | 0.700 | 12 | 0.917 |
| 21 | 230 | 0.570 | 0.052 | 0.087 | 0.236 | 28 | 0.786 | 8 | 1.000 |
| 22 | 230 | 0.565 | 0.054 | 0.070 | 0.150 | 34 | 0.853 | 5 | 0.800 |
| 23 | 230 | 0.530 | 0.054 | 0.052 | 0.216 | 32 | 0.875 | 9 | 0.667 |
| 24 | 230 | 0.526 | 0.049 | 0.096 | 0.464 | 35 | 0.914 | 13 | 0.923 |
| 25 | 230 | 0.543 | 0.055 | 0.104 | 0.278 | 34 | 0.853 | 10 | 1.000 |
| 26 | 230 | 0.578 | 0.051 | 0.052 | 0.200 | 37 | 0.865 | 6 | 0.667 |

## 7. Optional LLM variant (zero-shot, golden subset of the test split)

**Crisis decisions on the 300-case golden subset (recall exact; precision and cost re-weighted)**  
_script:_ `llm_agent.py` | _file:_ `results/llm_metrics.csv`

| method | tp | fn | recall | recall_lo | recall_hi | precision_reweighted | cost_per_1000_reweighted |
|---|---|---|---|---|---|---|---|
| M1_rules | 12 | 27 | 0.308 | 0.186 | 0.464 | 0.685 | 719.670 |
| M2_tfidf_lr | 37 | 2 | 0.949 | 0.831 | 0.986 | 0.870 | 60.039 |
| M3_xgb | 32 | 7 | 0.821 | 0.673 | 0.910 | 0.563 | 217.405 |
| LLM_openai/gpt-oss-120b | 31 | 8 | 0.795 | 0.645 | 0.892 | 0.738 | 225.619 |

| Quantity | Value | Script | File |
|---|---|---|---|
| Model / prompt sha / fallbacks | openai/gpt-oss-120b / a60937f6c21e / 0 of 300 | llm_agent.py | results/llm_summary.json |
| McNemar LLM vs M2 on the true crises | LLM-only 0, M2-only 6, p=0.031 | llm_agent.py | results/llm_summary.json |

## 8. Governance audit trail

| Quantity | Value | Script | File |
|---|---|---|---|
| Decisions logged / escalations | 2304 / 175 | audit.py | results/audit_demo.json |
| UPDATE attempt | blocked -> audit_log is append-only: UPDATE blocked | audit.py | results/audit_demo.json |
| DELETE attempt | blocked -> audit_log is append-only: DELETE blocked | audit.py | results/audit_demo.json |
| Intact chain verifies | True | audit.py | results/audit_demo.json |

## 9. Tests and tools

| Quantity | Value | Script | File |
|---|---|---|---|
| pytest | 45 passed in 1.97s | pytest | tests/ |
| Tools imported by the code that produced these results | pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, xgboost 3.4.1, matplotlib 3.11.2, seaborn 0.13.2, pydantic 2.14.1, joblib 1.6.0, pytest 9.1.1, openpyxl 3.1.5, nbformat 5.11.1, certifi 2026.7.22, sqlite3 (Python standard library) stdlib, jupyter nbconvert 7.17.2, Python 3.13.3 | make_results.py | src/, tests/, notebooks/ |
| Dashboard built (screenshots found in reports/dashboard) | False | make_results.py | reports/dashboard/ |
