"""
Calibrate the winning model (M2) with Platt scaling fitted on VALIDATION only, choose the
decision threshold by minimising expected cost on VALIDATION, and FREEZE everything
into results/decision_config.json before the test split is ever touched.

Uses: results/models/M2.joblib, results/models/M3.joblib, results/oof_dev_predictions.csv.
The TEST split is never loaded here.

ALL DATA IS SYNTHETIC. Run once:
    python -m src.calibrate
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import src.models  # noqa: F401  (needed so joblib can unpickle the saved models)
from src.config import (COST_FN_CRISIS, COST_FP_CRISIS, DATA_PROCESSED, FIGURES, RESULTS,
                        ensure_dirs, set_seed)
from src.evaluate import (CRISIS_IDX, brier, choose_threshold, cost_curve, ece_quantile,
                          recall_at_precision)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p_raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling: P(crisis) = sigmoid(a * logit(p_raw) + b). Returns (a, b)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p_raw, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * _logit(p_raw) + b)))


def main() -> None:
    set_seed()
    ensure_dirs()
    df = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    val = df[df.split == "val"].reset_index(drop=True)          # test is never loaded
    yv = (val.risk_level == "crisis").to_numpy().astype(int)

    m2 = joblib.load(RESULTS / "models" / "M2.joblib")
    m3 = joblib.load(RESULTS / "models" / "M3.joblib")
    p2 = m2.predict_proba(val)[:, CRISIS_IDX]
    p3 = m3.predict_proba(val)[:, CRISIS_IDX]
    n, n_pos = len(val), int(yv.sum())
    print(f"validation: n={n}, crisis={n_pos} (prevalence {yv.mean():.4f})")

    # ---- calibration ----------------------------------------------------------------
    a, b = fit_platt(p2, yv)
    p2_cal = apply_platt(p2, a, b)
    print(f"\nPlatt scaling fitted on validation: a={a:.4f}, b={b:.4f}")
    print("Calibration on validation (NOTE: the Platt row is IN-SAMPLE, so it flatters the "
          "calibrated model; the honest check is on test, once):")
    print(f"  constant-prevalence Brier baseline: {yv.mean() * (1 - yv.mean()):.4f}")
    bins = {}
    for name, p in [("raw_M2", p2), ("platt_M2", p2_cal)]:
        ece, tab = ece_quantile(yv, p)
        bins[name] = tab
        print(f"  {name:9s} Brier {brier(yv, p):.4f} | ECE {ece:.4f} | mean predicted {p.mean():.4f}")

    pd.concat([t.assign(model=k) for k, t in bins.items()]).to_csv(
        RESULTS / "calibration_bins_val.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color="#9a9a96", lw=1, ls="--")
    ax.plot(bins["raw_M2"].bin_mean_pred, bins["raw_M2"].bin_frac_pos, "o-",
            color="#2a78d6", lw=2, ms=5, label="raw M2")
    ax.plot(bins["platt_M2"].bin_mean_pred, bins["platt_M2"].bin_frac_pos, "o-",
            color="#eb6834", lw=2, ms=5, label="Platt (in-sample)")
    ax.set_xlabel("mean predicted P(crisis)")
    ax.set_ylabel("observed crisis fraction")
    ax.set_title("Reliability, validation (synthetic data)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "reliability_val.png", dpi=150)

    # ---- threshold by expected cost on validation -----------------------------------
    curve = cost_curve(yv, p2, COST_FN_CRISIS, COST_FP_CRISIS)
    thr_raw = choose_threshold(curve)
    thr_cal = float(apply_platt(np.array([thr_raw]), a, b)[0])
    row = curve.loc[curve.threshold == thr_raw].iloc[0]
    theory = COST_FP_CRISIS / (COST_FP_CRISIS + COST_FN_CRISIS)
    print(f"\nCost-minimising threshold on validation (FN={COST_FN_CRISIS:g}, FP={COST_FP_CRISIS:g}):")
    print(f"  raw P(crisis) >= {thr_raw:.4f}   | Platt-calibrated >= {thr_cal:.4f}"
          f"   | textbook value for perfectly calibrated scores = {theory:.4f}")
    print(f"  on validation: TP={int(row.tp)} FP={int(row.fp)} FN={int(row.fn)} "
          f"recall={row.recall:.3f} precision={row.precision:.3f} "
          f"expected cost per 1000 conversations={row.cost_per_1000:.1f}")
    print(f"  reference costs on validation: flag nobody = {COST_FN_CRISIS * n_pos:.0f} "
          f"({1000 * COST_FN_CRISIS * n_pos / n:.1f} per 1000) | flag everybody = {n - n_pos} "
          f"({1000 * (n - n_pos) / n:.1f} per 1000)")
    rp = recall_at_precision(curve, 0.30)
    print(f"  best recall with precision >= 0.30 on validation: {rp['recall']:.3f} "
          f"(at raw threshold {rp['threshold']:.4f})")

    print("\nSensitivity of the threshold to the assumed cost ratio (validation):")
    sens = []
    for ratio in (5, 10, 20, 50):
        c = cost_curve(yv, p2, float(ratio), COST_FP_CRISIS)
        t = choose_threshold(c)
        r = c.loc[c.threshold == t].iloc[0]
        sens.append({"fn_to_fp_ratio": ratio, "raw_threshold": round(t, 4), "tp": int(r.tp),
                     "fp": int(r.fp), "fn": int(r.fn), "recall": round(r.recall, 3),
                     "precision": round(r.precision, 3)})
    print(pd.DataFrame(sens).to_string(index=False))

    # ---- robustness: threshold chosen on family-disjoint out-of-fold predictions ----
    oof = pd.read_csv(RESULTS / "oof_dev_predictions.csv")
    y_oof = (oof.risk_level == "crisis").to_numpy().astype(int)
    p_oof = oof["p_crisis_M2_tfidf_lr"].to_numpy()
    c_oof = cost_curve(y_oof, p_oof, COST_FN_CRISIS, COST_FP_CRISIS)
    thr_oof = choose_threshold(c_oof)
    r_oof = c_oof.loc[c_oof.threshold == thr_oof].iloc[0]
    f_val = p2 >= thr_oof
    print(f"\nRobustness check (threshold chosen on {len(oof)} out-of-fold dev rows, "
          f"{int(y_oof.sum())} crisis, family-disjoint):")
    print(f"  raw threshold {thr_oof:.4f} | OOF recall={r_oof.recall:.3f} precision={r_oof.precision:.3f}"
          f" | on validation: recall={(f_val & (yv == 1)).sum() / n_pos:.3f}"
          f" precision={((f_val & (yv == 1)).sum() / max(f_val.sum(), 1)):.3f}")

    # ---- M3 threshold (needed later for the paired McNemar comparison) --------------
    thr_m3 = choose_threshold(cost_curve(yv, p3, COST_FN_CRISIS, COST_FP_CRISIS))
    print(f"\nM3 cost-minimising raw threshold on validation: {thr_m3:.4f}")

    # ---- save the curve for the BI export and FREEZE the decision config -------------
    out = pd.concat([
        curve.assign(model="M2_tfidf_lr", split="val"),
        cost_curve(yv, p3, COST_FN_CRISIS, COST_FP_CRISIS).assign(model="M3_xgb", split="val"),
    ])
    out.to_csv(RESULTS / "threshold_cost_curve_val.csv", index=False)

    config = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Frozen BEFORE the test split was evaluated. Do not edit after the test run.",
        "model": "M2_tfidf_lr", "model_file": "results/models/M2.joblib",
        "cost_matrix": {"fn": COST_FN_CRISIS, "fp": COST_FP_CRISIS},
        "platt": {"a": a, "b": b, "fitted_on": "validation", "n": n, "n_crisis": n_pos},
        "threshold_raw": thr_raw, "threshold_calibrated": thr_cal,
        "threshold_source": "min expected cost on validation (ties -> lowest threshold)",
        "sensitivity_threshold_raw_dev_oof": thr_oof,
        "m3": {"model_file": "results/models/M3.joblib", "threshold_raw": thr_m3},
        "m1": {"rule": "flag crisis if the rule-based classifier outputs 'crisis'"},
    }
    (RESULTS / "decision_config.json").write_text(json.dumps(config, indent=2))
    print("\nwrote results/decision_config.json (FROZEN), results/threshold_cost_curve_val.csv, "
          "results/calibration_bins_val.csv, reports/figures/reliability_val.png")


if __name__ == "__main__":
    main()
