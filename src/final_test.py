"""
THE one-time evaluation on the held-out TEST split.

Uses only things frozen before the test was touched: results/models/*.joblib,
results/decision_config.json (Platt parameters, thresholds). Refuses to run twice.

ALL DATA IS SYNTHETIC. Run exactly once:
    python -m src.final_test
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve

import src.models  # noqa: F401  (needed so joblib can unpickle the saved models)
from src.calibrate import apply_platt
from src.config import (COST_FN_CRISIS, COST_FP_CRISIS, DATA_PROCESSED, FIGURES, RESULTS,
                        ensure_dirs, set_seed)
from src.evaluate import (CLASSES, CRISIS_IDX, brier, choose_threshold, cluster_bootstrap_pr_auc,
                          cost_curve, crisis_pr_auc, crisis_report, ece_quantile, mcnemar_exact,
                          percentile_ci, recall_at_precision)
from src.models import M0Prior, M1Rules

LOCK = RESULTS / "TEST_RUN.lock"
N_BOOT = 1000
NAMES = ["M0_prior", "M1_rules", "M2_tfidf_lr", "M3_xgb"]


def main() -> None:
    if LOCK.exists():
        sys.exit(f"STOP: the test split was already evaluated ({LOCK.read_text().splitlines()[0]}). "
                 "Re-running would break the 'touched once' rule. Delete the lock only for a "
                 "deliberate, clearly-labelled exploratory run.")
    set_seed()
    ensure_dirs()

    cfg_path = RESULTS / "decision_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    thr2, thr3 = cfg["threshold_raw"], cfg["m3"]["threshold_raw"]
    a, b = cfg["platt"]["a"], cfg["platt"]["b"]

    df = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    train, val = df[df.split == "train"], df[df.split == "val"]
    test = df[df.split == "test"].reset_index(drop=True)
    y = (test.risk_level == "crisis").to_numpy().astype(int)
    groups = test.template_family.to_numpy()
    print(f"TEST split: n={len(test)}, crisis={int(y.sum())}, families={len(set(groups))}")
    print(f"frozen config sha256: {cfg_sha[:16]}...  (frozen at {cfg['frozen_at_utc']})")
    print(f"frozen thresholds: M2 raw>={thr2:.4f} (calibrated {cfg['threshold_calibrated']:.4f}), "
          f"M3 raw>={thr3:.4f}")

    m2 = joblib.load(RESULTS / "models" / "M2.joblib")
    m3 = joblib.load(RESULTS / "models" / "M3.joblib")
    P = {
        "M0_prior": M0Prior().fit(train).predict_proba(test),
        "M1_rules": M1Rules().predict_proba(test),
        "M2_tfidf_lr": m2.predict_proba(test),
        "M3_xgb": m3.predict_proba(test),
    }
    scores = {n: P[n][:, CRISIS_IDX] for n in NAMES}
    flags = {
        "M0_prior": np.zeros(len(test), dtype=bool),            # argmax is always no_risk
        "M1_rules": scores["M1_rules"] == 1.0,
        "M2_tfidf_lr": scores["M2_tfidf_lr"] >= thr2,
        "M3_xgb": scores["M3_xgb"] >= thr3,
    }

    # ---- headline metrics -----------------------------------------------------------
    boot, skipped = cluster_bootstrap_pr_auc(y, scores, groups, N_BOOT)
    rows = []
    for n in NAMES:
        rep = crisis_report(y, flags[n].astype(int))
        lo, hi = percentile_ci(boot[n])
        pred4 = np.array(CLASSES)[P[n].argmax(axis=1)]
        cost = COST_FN_CRISIS * rep["fn"] + COST_FP_CRISIS * rep["fp"]
        rows.append({
            "model": n, "n": len(test), "n_crisis": int(y.sum()),
            "pr_auc": crisis_pr_auc(y, scores[n]), "pr_auc_lo": lo, "pr_auc_hi": hi,
            "recall": rep["recall"], "recall_lo": rep["recall_lo"], "recall_hi": rep["recall_hi"],
            "precision": rep["precision"], "precision_lo": rep["precision_lo"],
            "precision_hi": rep["precision_hi"], "f2": rep["f2"],
            "macro_f1_argmax": float(f1_score(test.risk_level, pred4, labels=CLASSES,
                                              average="macro", zero_division=0)),
            "tp": rep["tp"], "fp": rep["fp"], "fn": rep["fn"], "tn": rep["tn"],
            "expected_cost": cost, "cost_per_1000": 1000 * cost / len(test),
        })
    metrics = pd.DataFrame(rows)
    print(f"\n=== TEST metrics at frozen thresholds (PR-AUC CI: family-cluster bootstrap, "
          f"{N_BOOT} resamples, {skipped} skipped) ===")
    print(metrics[["model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "f2", "macro_f1_argmax"]]
          .round(3).to_string(index=False))
    print()
    print(metrics[["model", "recall", "recall_lo", "recall_hi", "precision", "precision_lo",
                   "precision_hi"]].round(3).to_string(index=False))
    print()
    print(metrics[["model", "tp", "fp", "fn", "tn", "expected_cost", "cost_per_1000"]]
          .round(1).to_string(index=False))
    nobody = COST_FN_CRISIS * y.sum()
    print(f"reference: flag nobody = {nobody:.0f} ({1000 * nobody / len(test):.1f} per 1000) | "
          f"flag everybody = {len(test) - y.sum()} ({1000 * (len(test) - y.sum()) / len(test):.1f} per 1000)")

    w = metrics[metrics.model == "M2_tfidf_lr"].iloc[0]
    print("\n--- Pre-registered success criterion 1 (M2, frozen threshold) ---")
    print(f"crisis recall {w.recall:.3f} (95% Wilson {w.recall_lo:.3f}-{w.recall_hi:.3f}) vs target >= 0.95: "
          f"{'MET (point estimate)' if w.recall >= 0.95 else 'NOT MET'}"
          f"{'; CI lower bound >= 0.95: ' + str(bool(w.recall_lo >= 0.95))}")
    print(f"crisis precision {w.precision:.3f} (95% Wilson {w.precision_lo:.3f}-{w.precision_hi:.3f}) "
          f"vs target >= 0.30: {'MET' if w.precision >= 0.30 else 'NOT MET'}")
    print("(criterion 2, slice gaps, is Phase 6)")

    print("\nTest-curve diagnostic: best recall with precision >= 0.30 (threshold-free, uses test labels):")
    for n in NAMES:
        r = recall_at_precision(cost_curve(y, scores[n], COST_FN_CRISIS, COST_FP_CRISIS), 0.30)
        print(f"  {n:12s} recall {r['recall']:.3f} at precision {r['precision']:.3f}")

    # ---- confusion matrices ---------------------------------------------------------
    conf = []
    for n in NAMES:
        pred4 = np.array(CLASSES)[P[n].argmax(axis=1)]
        cm = confusion_matrix(test.risk_level, pred4, labels=CLASSES)
        for i, act in enumerate(CLASSES):
            for j, prd in enumerate(CLASSES):
                conf.append({"model": n, "view": "4class_argmax", "actual": act,
                             "predicted": prd, "count": int(cm[i, j])})
        r = metrics[metrics.model == n].iloc[0]
        for act, prd, cnt in [("crisis", "flagged", r.tp), ("crisis", "not_flagged", r.fn),
                              ("not_crisis", "flagged", r.fp), ("not_crisis", "not_flagged", r.tn)]:
            conf.append({"model": n, "view": "crisis_at_threshold", "actual": act,
                         "predicted": prd, "count": int(cnt)})
    pd.DataFrame(conf).to_csv(RESULTS / "confusion_long_test.csv", index=False)
    print("\nM2 4-class confusion on test (rows = recorded label, cols = predicted):")
    cm2 = confusion_matrix(test.risk_level, np.array(CLASSES)[P["M2_tfidf_lr"].argmax(axis=1)], labels=CLASSES)
    print(pd.DataFrame(cm2, index=CLASSES, columns=CLASSES).to_string())

    # ---- calibration (M2) -----------------------------------------------------------
    p_raw = scores["M2_tfidf_lr"]
    p_cal = apply_platt(p_raw, a, b)
    print(f"\n=== Calibration on TEST (Platt fitted on validation only) ===")
    print(f"constant-prevalence Brier baseline: {y.mean() * (1 - y.mean()):.4f}")
    cal_bins, cal_stats = [], {}
    for label, p in [("raw_M2", p_raw), ("platt_M2", p_cal)]:
        ece, tab = ece_quantile(y, p)
        cal_stats[label] = {"brier": brier(y, p), "ece": ece, "mean_pred": float(p.mean())}
        cal_bins.append(tab.assign(model=label))
        print(f"  {label:9s} Brier {brier(y, p):.4f} | ECE {ece:.4f} | mean predicted {p.mean():.4f} "
              f"(observed prevalence {y.mean():.4f})")
    pd.concat(cal_bins).to_csv(RESULTS / "calibration_bins_test.csv", index=False)

    # ---- paired McNemar on crisis detection -----------------------------------------
    pos = y == 1
    mc = []
    for other in ("M1_rules", "M3_xgb"):
        res = mcnemar_exact(flags["M2_tfidf_lr"][pos], flags[other][pos])
        mc.append({"a": "M2_tfidf_lr", "b": other, **res})
    mc = pd.DataFrame(mc)
    mc.to_csv(RESULTS / "mcnemar_test.csv", index=False)
    print("\n=== McNemar (exact) on detection of the true crisis cases, at frozen thresholds ===")
    print(mc.rename(columns={"a_only": "a_detects_b_misses", "b_only": "b_detects_a_misses"})
          .round(4).to_string(index=False))

    # ---- optimism gap: validation vs test for M2 ------------------------------------
    yv = (val.risk_level == "crisis").to_numpy().astype(int)
    pv = m2.predict_proba(val)[:, CRISIS_IDX]
    rv = crisis_report(yv, (pv >= thr2).astype(int))
    print("\n=== Validation vs test, M2 at the frozen threshold ===")
    print(f"  PR-AUC     val {crisis_pr_auc(yv, pv):.3f}  ->  test {w.pr_auc:.3f}")
    print(f"  recall     val {rv['recall']:.3f}  ->  test {w.recall:.3f}")
    print(f"  precision  val {rv['precision']:.3f}  ->  test {w.precision:.3f}")

    # ---- sensitivity to the cost ratio, thresholds chosen on dev out-of-fold only ----
    oof = pd.read_csv(RESULTS / "oof_dev_predictions.csv")
    y_oof = (oof.risk_level == "crisis").to_numpy().astype(int)
    p_oof = oof["p_crisis_M2_tfidf_lr"].to_numpy()
    sens = []
    for ratio in (5, 10, 20, 50, 100):
        t = choose_threshold(cost_curve(y_oof, p_oof, float(ratio), COST_FP_CRISIS))
        rep = crisis_report(y, (p_raw >= t).astype(int))
        sens.append({"fn_to_fp_ratio": ratio, "raw_threshold_from_dev_oof": round(t, 4),
                     "test_recall": round(rep["recall"], 3), "test_precision": round(rep["precision"], 3),
                     "flagged": rep["n_flagged"], "fn": rep["fn"], "fp": rep["fp"]})
    sens = pd.DataFrame(sens)
    sens.to_csv(RESULTS / "cost_ratio_sensitivity_test.csv", index=False)
    print("\n=== Sensitivity to the assumed cost ratio (thresholds chosen on dev out-of-fold, applied to test) ===")
    print(sens.to_string(index=False))

    # ---- save predictions and figures -----------------------------------------------
    pred = test[["conversation_id", "week", "channel", "language", "age_band", "template_family",
                 "risk_level", "escalate_needed", "text"]].copy()
    pred["p_raw_M2"], pred["p_cal_M2"], pred["flag_M2"] = p_raw, p_cal, flags["M2_tfidf_lr"].astype(int)
    pred["p_raw_M3"], pred["flag_M3"] = scores["M3_xgb"], flags["M3_xgb"].astype(int)
    pred["flag_M1"] = flags["M1_rules"].astype(int)
    pred["threshold_raw_M2"] = thr2
    pred.to_csv(RESULTS / "test_predictions.csv", index=False)
    metrics.to_csv(RESULTS / "metrics_by_model_test.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color="#9a9a96", lw=1, ls="--")
    for label, color in [("raw_M2", "#2a78d6"), ("platt_M2", "#eb6834")]:
        t = pd.concat(cal_bins)
        t = t[t.model == label]
        ax.plot(t.bin_mean_pred, t.bin_frac_pos, "o-", color=color, lw=2, ms=5, label=label)
    ax.set_xlabel("mean predicted P(crisis)")
    ax.set_ylabel("observed crisis fraction")
    ax.set_title("Reliability, TEST (synthetic data)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "reliability_test.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for n, color in [("M1_rules", "#2a78d6"), ("M2_tfidf_lr", "#eb6834"), ("M3_xgb", "#1baf7a")]:
        pr, rc, _ = precision_recall_curve(y, scores[n])
        ax.plot(rc, pr, color=color, lw=2, label=n)
        r = metrics[metrics.model == n].iloc[0]
        if r.tp + r.fp > 0:
            ax.plot([r.recall], [r.precision], "o", color=color, ms=8, mec="white", mew=1.5)
    ax.axhline(0.30, color="#9a9a96", lw=1, ls="--")
    ax.axvline(0.95, color="#9a9a96", lw=1, ls="--")
    ax.set_xlabel("crisis recall")
    ax.set_ylabel("crisis precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall, TEST (dots = frozen thresholds)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "pr_curve_test.png", dpi=150)

    summary = {
        "split": "test", "n": len(test), "n_crisis": int(y.sum()), "n_families": len(set(groups)),
        "frozen_config_sha256": cfg_sha, "n_boot": N_BOOT,
        "models": metrics.set_index("model").to_dict(orient="index"),
        "m2_calibration": cal_stats, "mcnemar": mc.to_dict(orient="records"),
        "m2_val_vs_test": {"val_pr_auc": crisis_pr_auc(yv, pv), "test_pr_auc": float(w.pr_auc),
                           "val_recall": rv["recall"], "test_recall": float(w.recall),
                           "val_precision": rv["precision"], "test_precision": float(w.precision)},
    }
    (RESULTS / "test_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    LOCK.write_text(f"test evaluated at {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
                    f"decision_config sha256 {cfg_sha}\n")
    print("\nwrote results/test_predictions.csv, metrics_by_model_test.csv, confusion_long_test.csv, "
          "mcnemar_test.csv, calibration_bins_test.csv, cost_ratio_sensitivity_test.csv, "
          "test_summary.json, reliability_test.png, pr_curve_test.png, and TEST_RUN.lock")


if __name__ == "__main__":
    main()
