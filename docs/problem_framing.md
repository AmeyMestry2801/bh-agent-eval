# Problem Framing: Behavioral-Health Intake and Risk Agent

> All data in this project is synthetic. No real patient data is used.
> Nothing here is clinically validated and no result should be read as
> evidence of clinical performance.

## Question
How do you measure whether an AI intake agent can be trusted to run
unattended, and how do you know when it is degrading and needs retraining?

## Users
- **Clinical operations lead**: decides staffing for the human-nurse
  escalation queue and needs to know the expected alert volume.
- **Clinician-founder**: decides whether the agent may run unattended
  and what safeguards must exist.

## Decision
For each intake conversation: **auto-continue** the automated intake, or
**escalate to a human nurse**. The model produces a crisis probability;
a threshold converts it into this decision.

## Cost matrix (assumption)
| Outcome | Relative cost |
|---|---|
| Missed crisis (false negative) | 20 |
| False alarm (false positive) | 1 |
| Correct decisions | 0 |

The 20:1 ratio is an assumption, not an empirical estimate. If scores
were perfectly calibrated, the cost-minimizing threshold would be
1 / (1 + 20) = ~0.048 on crisis probability. We will check whether the
validation-chosen threshold lands near this, and report it either way.
Sensitivity to the ratio (e.g. 5:1, 10:1, 50:1) will be reported.

## Success criteria (pre-registered, fixed before any modeling)
On the held-out test set, which is touched once:
1. Crisis-class **recall >= 0.95** at **precision >= 0.30**.
2. **No channel or language slice** has recall more than **5 percentage
   points below overall recall**.

These thresholds are set now. If they are not met, the README will say
so and will not move the goalposts. Slice criteria use the observed
point estimate, and we will also report CIs because small slices will
be noisy.

## Primary metric and unit of analysis
- Unit: one intake conversation.
- Positive class: `risk_level == "crisis"`.
- Primary metric: crisis-class recall at a precision floor, with crisis
  PR-AUC as the threshold-free companion.
- How `escalate_needed` relates to `risk_level` is defined in
  docs/assumptions.md during Phase 1, before data is generated.

## Risks
| Risk | Why it matters |
|---|---|
| Negation ("I'm not thinking of hurting myself") | Keyword cues appear in non-crisis text |
| Hedging ("maybe", "sort of", "I guess") | Real crisis expressed indirectly |
| Voice transcription (ASR) errors | Key words garbled or dropped |
| Language mix (English with Spanish phrases) | Vocabulary outside the training signal |
| Distribution drift (new slang, more voice) | Performance silently degrades after launch |
| Label noise | Caps measurable performance, inflates apparent errors |

## Out of scope
Clinical validity, real-world prevalence, regulatory compliance, and
any claim about real patients.
