# Behavioral-Health Intake and Risk Agent: Evaluation, Backtest and Monitoring Harness

> **All data in this project is synthetic.** No real patient data was used or imitated. Every number below is
> "on synthetic data", is read from `results/RESULTS.md`, and is **not** clinical validation. The noise rates,
> class mix and effect sizes are the author's assumptions (`docs/assumptions.md`).

## The question
How do you measure whether an AI intake agent can be trusted to flag crisis conversations, and how do you know
when it is degrading and needs retraining? This repo is the measurement harness: a synthetic data generator, a
leakage-safe split, a model ladder with a promotion rule written before the test run, cost-based decision
thresholds, confidence intervals, slice analysis, drift monitoring with alert rules, an audit trail, and tests.

## Framing (written before the corresponding runs)
- **Cost matrix:** a missed crisis costs 20x a false alarm (assumption; sensitivity to 5x-100x reported below).
- **Success criteria:** (1) crisis recall >= 0.95 at precision >= 0.30 on the test set; (2) no channel or
  language slice more than 5 points below overall recall.
- **Documents:** `docs/problem_framing.md`, `docs/model_plan.md`, `docs/monitoring_plan.md`.

## Data (synthetic)
6,000 conversations over 26 weeks, 399 template families, recorded mix no_risk 68.5% / low 18.4% / moderate 9.0% /
crisis 4.1%, 2.5% adjacent-class label noise, and a drift regime in weeks 17-26 (new slang families, more voice
transcripts with higher word error). Because of the label noise, an oracle that knows the clean labels would
reach only 235/248 = 0.948 crisis recall against the recorded labels.

| Split | Rows | Crisis | Rule |
|---|---|---|---|
| train | 2,202 | 96 | weeks 1-12 |
| validation | 736 | 25 | weeks 13-16 (shares families with train, so optimistic) |
| test | 758 | 39 | held-out template families from weeks 1-16, evaluated once |
| monitor | 2,304 | 88 | weeks 17-26 |

## Results (on synthetic data)
Test set, frozen thresholds chosen on validation. CIs: Wilson for recall, family-cluster bootstrap for PR-AUC.

| Model | Crisis recall [95% CI] | Precision | Missed / false alarms | Cost per 1000 | Crisis PR-AUC [95% CI] |
|---|---|---|---|---|---|
| M1 hand-written rules | 0.308 [0.186, 0.464] | 0.750 | 27 / 4 | 717.7 | 0.266 [0.045, 0.475] |
| **M2 TF-IDF + logistic regression** | **0.949 [0.831, 0.986]** | 0.841 | 2 / 7 | 62.0 | 0.952 [0.821, 1.000] |
| M3 XGBoost | 0.821 [0.673, 0.910] | 0.582 | 7 / 23 | 215.0 | 0.833 [0.572, 0.944] |
| LLM zero-shot (gpt-oss-120b)* | 0.795 [0.645, 0.892] | 0.738* | 8 / ~11* | 225.6* | not computed |

*LLM evaluated on a 300-case golden subset of the test split (all 39 crises plus 261 random non-crises);
recall is exact, precision and cost are re-weighted estimates. One prompt, one run.

**Reading the results**
- **M2 won under the pre-registered rule.** M3 had to beat M2 by a family-cluster-bootstrap CI excluding 0 and a gain
  of at least 0.02 crisis PR-AUC on dev out-of-fold predictions. It did the opposite: M3 minus M2 = -0.327
  [-0.450, -0.220]. M2 minus M1 = +0.714 [+0.608, +0.803]. The complex model was not forced to win.
- **Criterion 1 was missed by one case:** recall 0.949 vs the 0.95 target (37 of 39); precision 0.841 is well above 0.30.
- **On test, M2's edge over M3 is not statistically established:** exact McNemar p = 0.062 (5 discordant crises).
  Against M1, p < 0.001. Against the LLM, M2 detected 6 crises the LLM missed and the LLM detected none M2 missed
  (p = 0.031).
- **Calibration:** Platt scaling fitted on validation improved test Brier 0.0044 -> 0.0031 and ECE 0.0093 -> 0.0039.
- **Cost-ratio sensitivity:** with thresholds from dev out-of-fold data, test recall stayed 0.949 for FN:FP ratios 5 to
  100, while precision ranged 0.925 (5x) / 0.841 (10x-50x) / 0.552 (100x).
- **Validation was optimistic as designed:** M2 recall 1.000 and PR-AUC 0.992 on validation vs 0.949 and 0.952 on test.
- **Every false negative is a label-noise case.** All 7 misses (2 test, 5 dev out-of-fold) are conversations recorded
  as crisis but generated as moderate; none has intact crisis keywords, slang, or an ASR garble (`docs/error_analysis.md`,
  hand-tagged, n=7).

### Slices (criterion 2)
- **Test (pre-registered):** met by point estimate. Worst channel/language slice was text at 0.920 (-2.9 points, 25
  crisis cases). The Spanish-mixed slice has only 2 crisis cases, so this test is badly underpowered.
- **Dev out-of-fold (supplementary):** violated by point estimate. Spanish-mixed recall 0.833 [0.552, 0.953]
  (-12.5 points, 12 crisis cases); the Holm-adjusted association test is not significant.
- Age band 25-34 sits 6.0 points below overall on test (9 crisis cases); age is not part of the criterion.

### Drift and retraining (simulated shift starting week 17)
- The frozen M2's crisis recall fell from 1.000 on validation to 0.841 [0.750, 0.903] on the monitor window.
  New "D-*" slang families: recall 0.711 [0.552, 0.830], precision 0.243. Old families: recall 0.940, precision 0.734.
- Alert rules were fixed beforehand (`docs/monitoring_plan.md`). The label-free rule (score PSI > 0.25 or OOV rate > 2x
  reference, two consecutive weeks) fired in **week 18**. The label-based rule (windowed recall < 0.90 with at least 10
  labelled crises, two consecutive weeks) fired in **week 20**, after labels arrived.
- Retraining at week 20 and evaluating on weeks 21-26 (51 crises): recall 0.863 -> 0.961, cost per 1000 146.4 -> 65.2.
  **But the crisis PR-AUC difference was +0.075 with a 95% CI of [-0.050, +0.182], which includes 0, so recovery is not
  established.** Drift families also persist after the retrain week, so this is not evidence of generalising to unseen slang.
- The OOV signal is confounded with transcription noise (voice share also rose), and weekly recall is based on 5-13 crises
  a week.

### Optional LLM variant
A zero-shot gpt-oss-120b classifier (Groq free tier; `llama-3.3-70b-versatile` was no longer available on the key) with
Pydantic output validation, retries, a rule-based fallback, response caching and a latency/token log: 0 of 300 fallbacks,
median latency 0.53 s, $0.00 on the free tier. Six of its 8 missed crises were indirect or farewell wording the frozen
prompt did not name as crisis signals; a revised prompt would need a fresh held-out set. It is not part of `make all`.

### Governance audit trail
Every monitor-window decision (2,304, of which 175 escalations) is written to SQLite with model version, threshold, raw
and calibrated score and a text hash (no raw text). Triggers block UPDATE and DELETE, and a hash chain makes edits and
mid-log deletions detectable even if triggers are dropped (demonstrated on a copy). Limit: this is tamper-evident, not
tamper-proof; deleting the last row or replacing the file is not detectable without an external anchor.

## Limitations
1. Synthetic, templated text with author-assumed noise and mix; real conversations are far messier. Nothing here is clinical validation.
2. The test set has 39 crisis cases: recall CIs are wide (0.83-0.99) and slice tests are underpowered.
3. Label noise caps achievable crisis recall near 0.948 on these recorded labels.
4. The error taxonomy is hand-tagged on 7 cases; the LLM comparison is one prompt, one run, one model.
5. The drift simulation is generated, so detection speed here says little about real-world drift.
6. The Tableau/Excel dashboard is not included; `bi/` holds the dashboard-ready CSVs (and an `.xlsx`).

## Reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
make all          # generate -> validate -> EDA -> split -> models -> promote -> calibrate -> test (once)
                  # -> slices -> errors -> drift -> audit -> BI exports -> pytest -> results docs
```
Run `make all` in a fresh clone: the test-set evaluation is guarded by `results/TEST_RUN.lock` and refuses to run
twice. A clean-clone rebuild reproduced identical test metrics, slice metrics, drift tables and the promotion
decision on the author's machine; other platforms or library versions may differ slightly. `make llm` (optional)
needs `GROQ_API_KEY` in `.env`.

## Layout
`src/` pipeline modules | `tests/` 45 pytest tests | `docs/` framing, assumptions, plans, findings, model card |
`bi/` dashboard-ready exports | `reports/figures/` plots | `results/RESULTS.md` every number with its source script |
`results/resume_facts.json` machine-readable facts.
