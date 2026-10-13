"""
Phase 11: clean CSV exports for the BI tool (bi/). No raw text is exported.
Reconciles predictions.csv against metrics_by_model_test.csv before writing.

ALL DATA IS SYNTHETIC.
    python -m src.export_bi
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.config import BI, DATA_PROCESSED, RESULTS, ensure_dirs
from src.evaluate import CRISIS_IDX
from src.models import M2TfidfLR  # noqa: F401  (needed to unpickle M2)

ORIGIN = "SYNTHETIC DATA"


def need(name: str):
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"missing {p} - run the earlier phase that produces it")
    return pd.read_csv(p)


def write(df: pd.DataFrame, name: str, written: dict) -> None:
    df = df.copy()
    df["data_origin"] = ORIGIN
    df.to_csv(BI / name, index=False)
    written[name] = df
    print(f"bi/{name}: {len(df)} rows x {df.shape[1]} cols")


def main() -> None:
    ensure_dirs()
    BI.mkdir(parents=True, exist_ok=True)
    written: dict[str, pd.DataFrame] = {}

    # ---- dim_model ------------------------------------------------------------------
    dim = pd.DataFrame([
        ("M0_prior", 0, "Class prior (flags nobody)", "baseline"),
        ("M1_rules", 1, "Hand-written regex rules, 30-char negation window", "baseline"),
        ("M2_tfidf_lr", 2, "TF-IDF word+char n-grams + logistic regression, Platt-calibrated", "winner (pre-registered rule)"),
        ("M3_xgb", 3, "XGBoost on TF-IDF-SVD + 6 engineered features", "challenger (not promoted)"),
    ], columns=["model", "rung", "description", "status"])
    write(dim, "dim_model.csv", written)

    # ---- metrics_by_model + reconciliation -------------------------------------------
    metrics = need("metrics_by_model_test.csv")
    metrics.insert(0, "dataset", "test (pre-registered)")
    write(metrics, "metrics_by_model.csv", written)

    # ---- predictions (long: conversation x model) --------------------------------------
    tp = need("test_predictions.csv")
    tp["is_crisis"] = (tp.risk_level == "crisis").astype(int)
    base = ["conversation_id", "week", "channel", "language", "age_band", "risk_level", "is_crisis"]
    parts = []
    for model, flag_col, score_col in [("M1_rules", "flag_M1", None),
                                       ("M2_tfidf_lr", "flag_M2", "p_raw_M2"),
                                       ("M3_xgb", "flag_M3", "p_raw_M3")]:
        d = tp[base].copy()
        d["model"] = model
        d["flag"] = tp[flag_col].astype(int)
        d["score_raw"] = tp[score_col] if score_col else np.nan
        d["score_calibrated"] = tp["p_cal_M2"] if model == "M2_tfidf_lr" else np.nan
        d["outcome"] = np.select(
            [(d.is_crisis == 1) & (d.flag == 1), (d.is_crisis == 1) & (d.flag == 0),
             (d.is_crisis == 0) & (d.flag == 1)], ["TP", "FN", "FP"], default="TN")
        parts.append(d)
    pred = pd.concat(parts, ignore_index=True)
    pred.insert(0, "dataset", "test (pre-registered)")

    counts = pred.pivot_table(index="model", columns="outcome", values="conversation_id",
                              aggfunc="count", fill_value=0)
    for c in ["TP", "FP", "FN", "TN"]:
        if c not in counts:
            counts[c] = 0
    for m in counts.index:
        r = metrics[metrics.model == m].iloc[0]
        got = (int(counts.loc[m, "TP"]), int(counts.loc[m, "FP"]), int(counts.loc[m, "FN"]), int(counts.loc[m, "TN"]))
        want = (int(r.tp), int(r.fp), int(r.fn), int(r.tn))
        assert got == want, f"reconciliation FAILED for {m}: predictions {got} vs metrics {want}"
    print("reconciliation predictions.csv vs metrics_by_model_test.csv: OK (TP/FP/FN/TN identical for M1, M2, M3)")
    write(pred, "predictions.csv", written)

    # ---- confusion_long ----------------------------------------------------------------
    conf = need("confusion_long_test.csv")
    conf.insert(0, "dataset", "test (pre-registered)")
    write(conf, "confusion_long.csv", written)

    # ---- slice_errors ------------------------------------------------------------------
    sl = need("slice_metrics.csv")
    sl.insert(0, "model", "M2_tfidf_lr")
    sl["breach_5pts"] = (sl.gap_vs_overall_pts < -5).astype(int)
    sl["in_pre_registered_criterion"] = sl.variable.isin(["channel", "language"]).astype(int)
    write(sl, "slice_errors.csv", written)

    # ---- weekly_monitoring (+ control limits and alert markers) --------------------------
    wk = need("drift_weekly.csv").rename(
        columns={"share_D_families_(hidden_in_prod)": "share_d_families_hidden_in_prod"})
    ds = json.loads((RESULTS / "drift_summary.json").read_text())
    rules = ds["rules"]
    df = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    val = df[df.split == "val"]
    m2 = joblib.load(RESULTS / "models" / "M2.joblib")
    ref_flag = float((m2.predict_proba(val)[:, CRISIS_IDX] >= ds["threshold_frozen"]).mean())
    se = np.sqrt(ref_flag * (1 - ref_flag) / wk.n)
    wk["ref_flag_rate"] = ref_flag
    wk["flag_rate_ucl"] = ref_flag + 3 * se
    wk["flag_rate_lcl"] = np.maximum(ref_flag - 3 * se, 0)
    wk["flag_rate_out_of_control"] = ((wk.flag_rate > wk.flag_rate_ucl) | (wk.flag_rate < wk.flag_rate_lcl)).astype(int)
    wk["psi_limit"] = rules["psi_limit"]
    wk["oov_ref"] = ds["ref_oov"]
    wk["oov_limit"] = rules["oov_mult"] * ds["ref_oov"]
    wk["r1_condition"] = ((wk.psi_score > wk.psi_limit) | (wk.oov_rate > wk.oov_limit)).astype(int)
    wk["recall_target"] = 0.95
    wk["recall_floor"] = rules["recall_floor"]
    wk["alert"] = np.where(wk.week == ds["r2_week"], "R2 retrain",
                           np.where(wk.week == ds["r1_week"], "R1 warning", ""))
    write(wk, "weekly_monitoring.csv", written)

    # ---- threshold_cost_curve & calibration_bins -------------------------------------------
    write(need("threshold_cost_curve_val.csv"), "threshold_cost_curve.csv", written)
    cb_val = need("calibration_bins_val.csv").assign(split="val")
    cb_test = need("calibration_bins_test.csv").assign(split="test")
    write(pd.concat([cb_val, cb_test], ignore_index=True), "calibration_bins.csv", written)

    # ---- optional: LLM comparison (Phase 9) --------------------------------------------------
    if (RESULTS / "llm_metrics.csv").exists():
        lm = pd.read_csv(RESULTS / "llm_metrics.csv")
        lm.insert(0, "dataset", "test golden subset (39 crisis + 261 non-crisis; precision/cost re-weighted)")
        write(lm, "llm_comparison.csv", written)

    # ---- optional Excel workbook for the Excel fallback ---------------------------------------
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(BI / "dashboard_data.xlsx", engine="openpyxl") as xw:
            for name, d in written.items():
                d.to_excel(xw, sheet_name=name.replace(".csv", "")[:31], index=False)
        print("bi/dashboard_data.xlsx written (openpyxl)")
    except ImportError:
        print("openpyxl not installed: skipped dashboard_data.xlsx (pip install openpyxl if you want the Excel version)")

    print("\nheadline numbers in the exports (all SYNTHETIC):")
    print(metrics[["model", "recall", "precision", "fn", "fp", "cost_per_1000"]].round(3).to_string(index=False))
    print(f"weekly alerts: {wk.loc[wk.alert != '', ['week', 'alert']].to_dict('records')} | "
          f"baseline flagged rate {ref_flag:.4f}")


if __name__ == "__main__":
    main()
