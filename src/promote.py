"""
Pre-registered promotion test (docs/model_plan.md): promote M3 over M2 only if the
paired family-cluster-bootstrap 95% CI of the crisis PR-AUC difference excludes 0
AND the point gain is >= 0.02 absolute.

Decision data: out-of-fold predictions on train+val (GroupKFold(5) by template_family),
each model refit per fold with the hyperparameters already tuned on train.
Secondary (cannot overturn the decision): validation-only comparison.
The TEST split is never loaded here.

ALL DATA IS SYNTHETIC. Run after `python -m src.models --rungs M0 M1 M2 M3`:
    python -m src.promote
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import DATA_PROCESSED, RESULTS, ensure_dirs, set_seed
from src.evaluate import CLASSES, CRISIS_IDX, cluster_bootstrap_pr_auc, crisis_pr_auc, percentile_ci
from src.models import M1Rules, M2TfidfLR, M3XGB

MIN_GAIN = 0.02
N_BOOT = 1000


def dev_oof_proba(dev: pd.DataFrame, factories: dict) -> dict:
    gkf = GroupKFold(n_splits=5)
    oof = {name: np.zeros((len(dev), len(CLASSES))) for name in factories}
    for k, (tr, te) in enumerate(gkf.split(dev, groups=dev.template_family), start=1):
        print(f"  fold {k}/5 ...", flush=True)
        for name, make in factories.items():
            oof[name][te] = make().fit(dev.iloc[tr]).predict_proba(dev.iloc[te])
    return oof


def paired_summary(y, scores: dict, groups, a: str, b: str) -> dict:
    """b minus a: point estimate and paired cluster-bootstrap CI of the crisis PR-AUC gap."""
    boot, skipped = cluster_bootstrap_pr_auc(y, {a: scores[a], b: scores[b]}, groups, N_BOOT)
    diff = boot[b] - boot[a]
    lo, hi = percentile_ci(diff)
    point = crisis_pr_auc(y, scores[b]) - crisis_pr_auc(y, scores[a])
    return {"a": a, "b": b, "point_diff": point, "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_0": bool(lo > 0 or hi < 0), "n_boot_used": int(len(diff)), "skipped": skipped}


def main() -> None:
    set_seed()
    ensure_dirs()
    df = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    train, val = df[df.split == "train"], df[df.split == "val"]
    dev = df[df.split.isin(["train", "val"])].reset_index(drop=True)   # test is excluded

    m2, m3 = joblib.load(RESULTS / "models" / "M2.joblib"), joblib.load(RESULTS / "models" / "M3.joblib")
    print(f"tuned on train: M2 C={m2.best_C_} | M3 params={m3.best_params_}\n")

    print(f"Out-of-fold predictions on dev set (rows={len(dev)}, crisis={int((dev.risk_level == 'crisis').sum())}):")
    oof = dev_oof_proba(dev, {
        "M1_rules": lambda: M1Rules(),
        "M2_tfidf_lr": lambda: M2TfidfLR(fixed_C=m2.best_C_),
        "M3_xgb": lambda: M3XGB(fixed_params=m3.best_params_),
    })
    y = (dev.risk_level == "crisis").to_numpy().astype(int)
    scores = {n: p[:, CRISIS_IDX] for n, p in oof.items()}
    groups = dev.template_family.to_numpy()

    boot, _ = cluster_bootstrap_pr_auc(y, scores, groups, N_BOOT)
    print("\nDev out-of-fold crisis PR-AUC (family-cluster bootstrap 95% CI):")
    for n in scores:
        lo, hi = percentile_ci(boot[n])
        print(f"  {n:12s} {crisis_pr_auc(y, scores[n]):.3f}  [{lo:.3f}, {hi:.3f}]")

    cmp_m3 = paired_summary(y, scores, groups, "M2_tfidf_lr", "M3_xgb")
    cmp_m1 = paired_summary(y, scores, groups, "M1_rules", "M2_tfidf_lr")
    promote = cmp_m3["ci_excludes_0"] and cmp_m3["ci_lo"] > 0 and cmp_m3["point_diff"] >= MIN_GAIN

    print("\nPaired differences (second minus first), crisis PR-AUC:")
    for c in (cmp_m3, cmp_m1):
        print(f"  {c['b']} - {c['a']}: {c['point_diff']:+.3f}  95% CI [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]"
              f"  excludes 0: {c['ci_excludes_0']}")
    winner = "M3_xgb" if promote else "M2_tfidf_lr"
    print(f"\nRULE: promote M3 only if CI excludes 0 (in M3's favour) AND gain >= {MIN_GAIN}")
    print(f"DECISION: {'PROMOTE M3' if promote else 'KEEP M2'}  ->  winner = {winner}")

    # secondary: validation only (models fit on train); cannot overturn the decision
    yv = (val.risk_level == "crisis").to_numpy().astype(int)
    sv = {"M2_tfidf_lr": m2.predict_proba(val)[:, CRISIS_IDX], "M3_xgb": m3.predict_proba(val)[:, CRISIS_IDX]}
    cmp_val = paired_summary(yv, sv, val.template_family.to_numpy(), "M2_tfidf_lr", "M3_xgb")
    print(f"\n(secondary, validation only, {int(yv.sum())} crisis cases) M3 - M2: {cmp_val['point_diff']:+.3f}"
          f"  95% CI [{cmp_val['ci_lo']:+.3f}, {cmp_val['ci_hi']:+.3f}]")

    out = pd.DataFrame({"conversation_id": dev.conversation_id, "template_family": dev.template_family,
                        "risk_level": dev.risk_level, **{f"p_crisis_{n}": s for n, s in scores.items()}})
    out.to_csv(RESULTS / "oof_dev_predictions.csv", index=False)
    (RESULTS / "promotion.json").write_text(json.dumps({
        "winner": winner, "promote_m3": bool(promote), "min_gain": MIN_GAIN, "n_boot": N_BOOT,
        "m3_vs_m2_dev_oof": cmp_m3, "m2_vs_m1_dev_oof": cmp_m1, "m3_vs_m2_val_secondary": cmp_val,
        "dev_oof_pr_auc": {n: crisis_pr_auc(y, s) for n, s in scores.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
