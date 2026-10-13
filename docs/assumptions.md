# Assumptions and Data-Generating Process

> ALL DATA IS SYNTHETIC. Every number below is an assumption chosen by the
> project author, not an estimate from real patients or real clinical data.
> Nothing in this project is clinically validated.

## Parameters (all in src/generate_data.py)
| Parameter | Value | Note |
|---|---|---|
| Conversations | 6,000 over 26 weeks | ~231 per week |
| Class mix (before label noise) | 70% no_risk, 17% low, 9% moderate, 4% crisis | exact counts: 4200 / 1020 / 540 / 240 |
| Reference regime | weeks 1-16 | one distribution |
| Drift regime | weeks 17-26 | new slang, more voice, more ASR noise |
| Voice share | 35% (weeks 1-16), 55% (weeks 17-26) | assumption |
| ASR word error rate | U(5%, 15%) weeks 1-16; U(10%, 22%) weeks 17-26 | per voice conversation |
| Typo rate | 3% of words > 3 chars | text channel only |
| Spanish-mixed conversations | 10% | see interpretation below |
| Hedging probability | crisis 25%, moderate 25%, low 20%, no_risk 10% | |
| Label noise | 2.5% of conversations get an adjacent class | |
| Age bands | 18-24: 18%, 25-34: 27%, 35-49: 27%, 50-64: 18%, 65+: 10% | style effect only, no label effect |
| New-slang families in drift weeks | 35% of weeks 17-26 conversations | families "D-*" |
| Label delay | crisis: 1-4 weeks (50/30/15/5%); other: 1-4 weeks (20/30/30/20%) | |
| Random seed | 42 | src/config.py |

## How a conversation is generated
1. Draw the true class (exact counts), a week, a channel, a language flag, an age band.
2. Pick a **template family** for that class. A family is a fixed skeleton:
   opener + 3 core paraphrase templates + context sentence + closer.
   Conversations in one family are near-duplicates (same skeleton, different
   slot fills and noise). Weeks 1-26 use 'B-' families; 35% of weeks 17-26
   conversations use new 'D-' families containing new slang.
3. Fill the core template's {a|b|c} slots randomly.
4. English core may be replaced by a Spanish sentence (60% of Spanish-mixed
   conversations) or get Spanish interjections / closers (40%).
5. Add hedging (prefix like "maybe", "I guess", optional "or something").
6. Channel noise: voice = lowercase, no punctuation, word-level ASR errors
   (55% substitution incl. homophones, 25% deletion, 20% filler insertion);
   text = character typos (swap / drop / double).
7. Label noise: with probability 2.5% the *recorded* label is replaced by an
   adjacent class (no_risk<->low, low<->moderate, moderate<->crisis).
8. `escalate_needed` is derived from the *recorded* label.

## Deliberately hard cases
- Explicit crisis, indirect crisis (no keywords: giving things away, goodbyes),
  and hedged crisis.
- Moderate cases that include negation ("I would never act on it").
- Hard negatives that share crisis vocabulary: negated statements
  ("I'm not thinking of hurting myself"), figurative speech ("my job is
  killing me"), sarcasm ("kill me now"), fiction ("a character died by
  suicide"), historical statements ("I used to have dark thoughts"), and
  bereavement paperwork.
- Text length is nearly identical across classes, so length is not a shortcut.
- ASR substitutions can drop or flip negation ("not" -> "knot"/"now",
  "can't" -> "can").

## Interpretations and decisions
- **"English with 10% Spanish phrases"** is interpreted as: 10% of
  conversations contain Spanish (language = `en_es`), the rest are English
  (`en`). It does not mean 10% of words are Spanish.
- **`escalate_needed`** = 1 only when the recorded `risk_level` is `crisis`.
  It is therefore redundant with the crisis label. This is a simplification:
  in a real service, some moderate cases would also route to a clinician.
- **Extra columns** beyond the spec: `template_family` (needed for the group
  split), `risk_level_clean` (the pre-noise label, for the label-noise audit
  only, never a feature), `gen_asr_wer` (generator parameter, audit only,
  never a feature).
- **Drift** changes inputs (slang, voice share, ASR noise) but not the class
  mix, so input drift and prevalence shift are not confounded.

## Known limitations of the generator
- Template families share the same pool of core paraphrases, so grouping by
  family prevents *skeleton* near-duplicates from straddling the split, but
  not shared core phrasing. Held-out-family performance is therefore an
  optimistic proxy for performance on truly new phrasing.
- Real language is far more varied than any template pool. Metrics measure
  the harness, not clinical safety.
- With 4% crisis, weeks 1-16 contain only ~160 crisis conversations, so the
  held-out test set will hold few crisis cases and confidence intervals on
  crisis recall will be wide. This is reported honestly, not hidden.
