# LLM variant findings (ALL DATA IS SYNTHETIC; zero-shot, one prompt, one run)

Model: openai/gpt-oss-120b via Groq's free API (llama-3.3-70b-versatile was not available
on this key). Prompt sha a60937f6c21e, frozen before any test-set LLM output existed.
Golden subset: 300 test cases = all 39 crises + 261 random non-crises (seed 42).
Decision rule: flag crisis iff the model returns risk_level == "crisis".

| Method | Crisis recall (39 crises) | Precision* | Cost per 1000* |
|---|---|---|---|
| M1 rules | 12/39 = 0.308 | 0.685 | 719.7 |
| M2 TF-IDF LR | 37/39 = 0.949 [0.831, 0.986] | 0.870 | 60.0 |
| M3 XGBoost | 32/39 = 0.821 | 0.563 | 217.4 |
| LLM zero-shot | 31/39 = 0.795 [0.645, 0.892] | 0.738 | 225.6 |
*Re-weighted from the 261 sampled non-crises to all 719 in the test set; based on 2-9 false
positives per method, so imprecise.

- LLM vs M2 on the 39 crises: M2-only detections 6, LLM-only 0, exact McNemar p = 0.031.
- LLM upper recall bound 0.892 < 0.95 target: criterion not met by the zero-shot LLM.
- Operations: 0/300 parse fallbacks, 0 retries, median latency 0.53 s (p95 0.94 s),
  101,582 input / 16,777 output tokens, $0.00 on the free tier.

Limits: one prompt and one model; M2 learned this generator's label convention (including
its label noise) while the LLM judges text semantically; no PR-AUC for the LLM because verbal
confidence is not a calibrated score. Not a statement about LLMs in general or real patients.

## Error pattern in the 8 missed crises (hand-read; n=8, descriptive only)
- 2 are passive ideation with "no plan" / "I'd never do it", also missed by M2 (same label-noise
  pattern as docs/error_analysis.md).
- 6 are indirect or farewell wording (e.g. giving things away, "you won't hear from me"), all caught
  by M2. The frozen prompt defined crisis as stated intent, plan or imminent danger and did not name
  farewell behaviour, so this is largely a prompt-definition gap rather than a model limit.
- A revised prompt would need a fresh held-out set; the test set was used once for the LLM and is not
  re-used for tuning.
