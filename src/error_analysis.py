"""
Print and save every false-negative crisis case of the frozen M2 model:
test (2 expected) + dev out-of-fold (5 expected), for hand-tagging.

Hints are automatic suggestions; the hand tag (negation / hedging / slang / ASR error / other)
is the final tag. risk_level_clean and gen_asr_wer are AUDIT-ONLY aids, never features.
The test split is not re-evaluated: this reads saved predictions.

ALL DATA IS SYNTHETIC. Run once:
    python -m src.error_analysis
"""
from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd

import src.models  # noqa: F401  (needed so joblib can unpickle the saved models)
from src.config import DATA_PROCESSED, RESULTS, ensure_dirs, set_seed
from src.evaluate import CLASSES
from src.features import HEDGE_PHRASES, HEDGE_WORDS, _is_negation, tokenize
from src.models import M1Rules

SLANG = re.compile(r"\b(ngl|lowkey|tbh|fr|no cap|kms|unalive\w*|cooked|mid|af|bruh|lmao|lol)\b", re.I)


def collect(thr: float) -> pd.DataFrame:
    meta = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    cols = ["conversation_id", "risk_level", "risk_level_clean", "week", "channel", "language",
            "age_band", "text", "template_family", "gen_asr_wer"]
    test = pd.read_csv(RESULTS / "test_predictions.csv")[["conversation_id", "p_raw_M2", "flag_M2"]]
    test = test.rename(columns={"p_raw_M2": "p_raw", "flag_M2": "flag"}).assign(dataset="test")
    oof = pd.read_csv(RESULTS / "oof_dev_predictions.csv")[["conversation_id", "p_crisis_M2_tfidf_lr"]]
    oof = oof.rename(columns={"p_crisis_M2_tfidf_lr": "p_raw"}).assign(dataset="dev_oof")
    oof["flag"] = (oof.p_raw >= thr).astype(int)
    pred = pd.concat([test, oof]).merge(meta[cols], on="conversation_id", how="left")
    pred["crisis"] = (pred.risk_level == "crisis").astype(int)
    return pred


def main() -> None:
    set_seed()
    ensure_dirs()
    thr = json.loads((RESULTS / "decision_config.json").read_text())["threshold_raw"]
    pred = collect(thr)
    fn = pred[(pred.crisis == 1) & (pred.flag == 0)].copy().reset_index(drop=True)

    negs = {d: g.loc[g.crisis == 0, "p_raw"].to_numpy() for d, g in pred.groupby("dataset")}
    fn["neg_scoring_higher"] = [int((negs[r.dataset] >= r.p_raw).sum()) for r in fn.itertuples()]
    fn["n_negatives"] = [len(negs[r.dataset]) for r in fn.itertuples()]

    m2 = joblib.load(RESULTS / "models" / "M2.joblib")
    m1 = M1Rules()
    print(f"false-negative crisis cases at the frozen threshold ({thr:.4f}): {len(fn)} "
          f"({(fn.dataset == 'test').sum()} test, {(fn.dataset == 'dev_oof').sum()} dev out-of-fold)")

    hint_rows = []
    for i, r in enumerate(fn.itertuples(), 1):
        toks = tokenize(r.text)
        neg = [t for t in toks if _is_negation(t)]
        hedge = [t for t in toks if t in HEDGE_WORDS] + [m.group(0).lower() for m in HEDGE_PHRASES.finditer(r.text)]
        slang = sorted({m.group(0).lower() for m in SLANG.finditer(r.text)})
        kw = m1.explain(r.text)["crisis"]
        hint_rows.append({"negation_words": " ".join(neg), "hedge_words": " ".join(hedge),
                          "slang_tokens": " ".join(slang), "m1_crisis_keywords": " | ".join(kw)})
        wer = f"{r.gen_asr_wer:.2f}" if pd.notna(r.gen_asr_wer) else "n/a"
        print(f"\n[{r.dataset} #{i}] {r.conversation_id} | week {r.week} | {r.channel} | {r.language} | "
              f"age {r.age_band} | family {r.template_family}")
        print(f"  recorded={r.risk_level} clean={r.risk_level_clean} | gen_asr_wer={wer}")
        print(f"  P(crisis)={r.p_raw:.4f} vs threshold {thr:.4f} | non-crisis cases scoring at least as high: "
              f"{r.neg_scoring_higher} of {r.n_negatives}")
        if r.dataset == "test":
            probs = m2.predict_proba(pd.DataFrame({"text": [r.text]}))[0]
            print("  M2 class probabilities: " + ", ".join(f"{c}={p:.3f}" for c, p in zip(CLASSES, probs)))
        print(f"  hints: negation={neg or '-'} | hedge={hedge or '-'} | slang={slang or '-'} | "
              f"M1 crisis keywords still present={kw or '-'}")
        print(f"  TEXT: {r.text}")

    out = pd.concat([fn.reset_index(drop=True), pd.DataFrame(hint_rows)], axis=1)
    out["tag_primary"], out["tag_secondary"], out["note"] = "", "", ""
    out.to_csv(RESULTS / "fn_cases.csv", index=False)
    print("\nwrote results/fn_cases.csv (columns tag_primary / tag_secondary / note are empty, for hand-tags)")


if __name__ == "__main__":
    main()
