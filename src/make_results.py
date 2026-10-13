"""
Phase 12: build results/RESULTS.md and results/resume_facts.json FROM the result files.
No number is typed by hand. ALL DATA IS SYNTHETIC.
    python -m src.make_results
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

import joblib
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, RESULTS, ROOT
from src.models import M2TfidfLR, M3XGB  # noqa: F401  (needed to unpickle the saved models)

DISCLAIMER = ("ALL NUMBERS BELOW ARE ON SYNTHETIC DATA. They are not clinical validation and say "
              "nothing about real patients or real deployments.")


def jload(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def cload(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


INT_COLS = {"week", "n", "n_crisis", "win_n_crisis", "retro_n_crisis", "tp", "fp", "fn", "tn",
            "flagged", "fn_to_fp_ratio", "discordant", "a_only", "b_only"}


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            if v != v:
                return "n/a"
            if v != 0 and abs(v) < 0.001:
                return f"{v:.2e}"
            return f"{v:.{nd}f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for rec in df.to_dict("records"):
        cells = []
        for c in cols:
            v = rec[c]
            if c in INT_COLS and isinstance(v, (int, float)) and v == v:
                cells.append(str(int(v)))
            else:
                cells.append(fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plain(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))


class Doc:
    def __init__(self):
        self.parts, self.rows, self.title, self.printed = [], [], None, True

    def section(self, title):
        self.flush()
        self.title, self.printed = title, False

    def row(self, quantity, value, script, file):
        self.rows.append((quantity, value, script, file))

    def flush(self):
        if self.title and not self.printed:
            self.parts.append(f"## {self.title}\n")
            self.printed = True
        if self.rows:
            df = pd.DataFrame(self.rows, columns=["Quantity", "Value", "Script", "File"])
            self.parts.append(md_table(df) + "\n")
            self.rows = []

    def table(self, caption, df, script, file):
        self.flush()
        self.parts.append(f"**{caption}**  \n_script:_ `{script}` | _file:_ `{file}`\n\n{md_table(df)}\n")

    def render(self):
        self.flush()
        return "\n".join(self.parts)


def nan_to_none(o):
    if isinstance(o, dict):
        return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan_to_none(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


def tools_used() -> dict:
    mapping = {"pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "statsmodels": "statsmodels",
               "sklearn": "scikit-learn", "xgboost": "xgboost", "matplotlib": "matplotlib",
               "seaborn": "seaborn", "pydantic": "pydantic", "joblib": "joblib", "pytest": "pytest",
               "openpyxl": "openpyxl", "nbformat": "nbformat", "certifi": "certifi"}
    imported = set()
    files = (list((ROOT / "src").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
             + list((ROOT / "notebooks").glob("*.py")))
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module.split(".")[0])
    out = {}
    for mod, dist in mapping.items():
        if mod in imported:
            try:
                out[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                pass
    if "sqlite3" in imported:
        out["sqlite3 (Python standard library)"] = "stdlib"
    if (ROOT / "notebooks" / "01_eda.ipynb").exists():
        try:
            out["jupyter nbconvert"] = metadata.version("nbconvert")
        except metadata.PackageNotFoundError:
            pass
    out["Python"] = sys.version.split()[0]
    return out


def run_pytest() -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    passed = re.search(r"(\d+) passed", tail)
    failed = re.search(r"(\d+) failed", tail)
    return {"passed": int(passed.group(1)) if passed else 0,
            "failed": int(failed.group(1)) if failed else 0, "summary_line": tail}


def main() -> None:
    doc, facts = Doc(), {}
    facts["disclaimer"] = DISCLAIMER
    facts["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ------------------------------------------------------------------ data
    raw = pd.read_csv(DATA_RAW / "conversations.csv")
    sp = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    rec_crisis = int((raw.risk_level == "crisis").sum())
    oracle_tp = int(((raw.risk_level == "crisis") & (raw.risk_level_clean == "crisis")).sum())
    noise = float((raw.risk_level != raw.risk_level_clean).mean())
    mix = raw.risk_level.value_counts(normalize=True).round(4).to_dict()
    splits = {s: {"n": int((sp.split == s).sum()),
                  "n_crisis": int(((sp.split == s) & (sp.risk_level == "crisis")).sum()),
                  "weeks": f"{int(sp[sp.split == s].week.min())}-{int(sp[sp.split == s].week.max())}"}
              for s in ["train", "val", "test", "monitor"]}
    facts["data"] = {"n_conversations": len(raw), "weeks": "1-26", "n_template_families": int(raw.template_family.nunique()),
                     "recorded_class_mix": mix, "recorded_crisis_n": rec_crisis,
                     "label_noise_rate": round(noise, 4),
                     "oracle_ceiling_crisis_recall": f"{oracle_tp}/{rec_crisis} = {oracle_tp / rec_crisis:.3f}",
                     "synthetic": True}
    facts["splits"] = splits
    g = "generate_data.py / split.py"
    doc.section("1. Data and splits (synthetic)")
    doc.row("Conversations", len(raw), g, "data/raw/conversations.csv")
    doc.row("Template families", facts["data"]["n_template_families"], g, "data/raw/conversations.csv")
    doc.row("Recorded class mix", ", ".join(f"{k} {v:.3f}" for k, v in mix.items()), g, "data/raw/conversations.csv")
    doc.row("Label-noise rate (recorded != clean)", f"{noise:.4f}", g, "data/raw/conversations.csv")
    doc.row("Oracle ceiling, crisis recall (clean label vs recorded)", facts["data"]["oracle_ceiling_crisis_recall"], g, "data/raw/conversations.csv")
    for s, v in splits.items():
        doc.row(f"Split {s}", f"n={v['n']}, crisis={v['n_crisis']}, weeks {v['weeks']}", "split.py", "data/processed/conversations_split.csv")

    # ------------------------------------------------------------ selection
    prom = jload("promotion.json") or {}
    sel = {"winner": prom.get("winner"), "promote_m3": prom.get("promote_m3"),
           "m3_minus_m2_dev_oof_pr_auc": prom.get("m3_vs_m2_dev_oof"),
           "m2_minus_m1_dev_oof_pr_auc": prom.get("m2_vs_m1_dev_oof")}
    try:
        m2 = joblib.load(RESULTS / "models" / "M2.joblib")
        m3 = joblib.load(RESULTS / "models" / "M3.joblib")
        sel["m2_selected_C"] = m2.best_C_
        sel["m3_best_params"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in m3.best_params_.items()}
    except Exception as e:  # noqa: BLE001
        sel["model_load_error"] = str(e)
    facts["model_selection"] = sel
    doc.section("2. Model selection (dev out-of-fold, pre-registered rule)")
    doc.row("Winner", sel["winner"], "promote.py", "results/promotion.json")
    for key, label in [("m3_minus_m2_dev_oof_pr_auc", "M3 minus M2 crisis PR-AUC"), ("m2_minus_m1_dev_oof_pr_auc", "M2 minus M1 crisis PR-AUC")]:
        c = sel.get(key)
        if c:
            doc.row(label, f"{c['point_diff']:+.3f}, 95% CI [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]", "promote.py", "results/promotion.json")
    if "m2_selected_C" in sel:
        doc.row("M2 selected C (GroupKFold by family, train only)", sel["m2_selected_C"], "models.py", "results/models/M2.joblib")
        doc.row("M3 best params (25-candidate random search, train only)", json.dumps(sel["m3_best_params"]), "models.py", "results/models/M3.joblib")

    # --------------------------------------------------------- frozen config
    cfg = jload("decision_config.json") or {}
    facts["frozen_decision"] = {"model": cfg.get("model"), "threshold_raw": cfg.get("threshold_raw"),
                                "threshold_calibrated": cfg.get("threshold_calibrated"),
                                "platt": cfg.get("platt"), "cost_matrix": cfg.get("cost_matrix"),
                                "m3_threshold_raw": (cfg.get("m3") or {}).get("threshold_raw")}
    doc.section("3. Frozen decision rule (set on validation before the test run)")
    doc.row("Model", cfg.get("model"), "calibrate.py", "results/decision_config.json")
    doc.row("Raw threshold (min expected cost, FN=20, FP=1)", f"{cfg.get('threshold_raw', float('nan')):.4f}", "calibrate.py", "results/decision_config.json")
    if cfg.get("platt"):
        doc.row("Platt a, b (validation)", f"{cfg['platt']['a']:.4f}, {cfg['platt']['b']:.4f}", "calibrate.py", "results/decision_config.json")

    # ------------------------------------------------------------------ test
    mt = cload("metrics_by_model_test.csv")
    keep = ["pr_auc", "pr_auc_lo", "pr_auc_hi", "recall", "recall_lo", "recall_hi", "precision",
            "tp", "fp", "fn", "tn", "expected_cost", "cost_per_1000"]
    facts["test_results"] = {r.model: {k: getattr(r, k) for k in keep} for r in mt.itertuples()}
    facts["test_set"] = {"n": splits["test"]["n"], "n_crisis": splits["test"]["n_crisis"]}
    doc.section("4. Test set, evaluated once at frozen thresholds")
    doc.table("Headline metrics (PR-AUC CI: family-cluster bootstrap; recall/precision CI: Wilson)",
              mt[["model", "n_crisis", "pr_auc", "pr_auc_lo", "pr_auc_hi", "recall", "recall_lo", "recall_hi",
                  "precision", "fn", "fp", "cost_per_1000"]], "final_test.py", "results/metrics_by_model_test.csv")
    mc = cload("mcnemar_test.csv")
    if mc is not None:
        doc.table("Exact McNemar tests on paired crisis detection (M2 vs the others)", mc, "final_test.py", "results/mcnemar_test.csv")
    sens = cload("cost_ratio_sensitivity_test.csv")
    if sens is not None:
        doc.table("Sensitivity to the assumed cost ratio (thresholds from dev out-of-fold)", sens, "final_test.py", "results/cost_ratio_sensitivity_test.csv")
    ts = jload("test_summary.json") or {}
    if ts.get("m2_calibration"):
        cal = pd.DataFrame(ts["m2_calibration"]).T.reset_index().rename(columns={"index": "variant"})
        doc.table("M2 calibration on test (Platt fitted on validation only)", cal, "final_test.py", "results/test_summary.json")
        facts["m2_calibration_test"] = ts["m2_calibration"]
    if ts.get("m2_val_vs_test"):
        facts["m2_val_vs_test"] = ts["m2_val_vs_test"]
        doc.table("M2 validation vs test (validation shares families with train, so it is optimistic)",
                  pd.DataFrame([ts["m2_val_vs_test"]]), "final_test.py", "results/test_summary.json")

    # ---------------------------------------------------------- criteria
    m2row = facts["test_results"].get("M2_tfidf_lr", {})
    c1_met = bool(m2row.get("recall", 0) >= 0.95 and m2row.get("precision", 0) >= 0.30)
    sl = cload("slice_metrics.csv")
    crit2 = {}
    for ds, grp in sl.groupby("dataset"):
        gg = grp[grp.variable.isin(["channel", "language"])]
        w = gg.loc[gg.gap_vs_overall_pts.idxmin()]
        crit2[ds] = {"met_point_estimate": bool(w.gap_vs_overall_pts >= -5), "worst_slice": f"{w.variable}={w.level}",
                     "gap_pts": round(float(w.gap_vs_overall_pts), 1), "slice_recall": round(float(w.recall), 3),
                     "slice_n_crisis": int(w.n_crisis)}
    facts["success_criteria"] = {
        "criterion_1_recall_ge_0.95_at_precision_ge_0.30_on_test": {"met": c1_met, "recall": m2row.get("recall"), "precision": m2row.get("precision")},
        "criterion_2_no_channel_or_language_slice_more_than_5pts_below_overall": crit2}
    doc.section("5. Pre-registered success criteria")
    doc.row("Criterion 1: crisis recall >= 0.95 at precision >= 0.30 (M2, test)",
            f"{'MET' if c1_met else 'NOT MET'} (recall {m2row.get('recall', float('nan')):.3f}, precision {m2row.get('precision', float('nan')):.3f})",
            "final_test.py", "results/metrics_by_model_test.csv")
    for ds, v in crit2.items():
        doc.row(f"Criterion 2 on {ds}", f"{'MET' if v['met_point_estimate'] else 'VIOLATED'} (point estimate); worst {v['worst_slice']}, "
                f"{v['gap_pts']} pts, recall {v['slice_recall']} on {v['slice_n_crisis']} crisis cases", "slices.py", "results/slice_metrics.csv")
    facts["slices"] = {"table_file": "results/slice_metrics.csv"}
    doc.table("Slice metrics (recall with Wilson CI; gap vs overall in points)",
              sl[["dataset", "variable", "level", "n", "n_crisis", "tp", "fn", "recall", "recall_lo", "recall_hi", "gap_vs_overall_pts"]],
              "slices.py", "results/slice_metrics.csv")

    # ----------------------------------------------------------------- drift
    ds_ = jload("drift_summary.json")
    if ds_:
        rc = ds_["retrain_comparison"]
        stale, new = rc["stale (frozen)"], rc["retrained"]
        facts["drift"] = {"r1_warning_week": ds_["r1_week"], "r2_retrain_week": ds_["r2_week"], "drift_start_week": 17,
                          "validation_recall_frozen_m2": ds_["validation_recall"], "monitor_recall_frozen_m2": ds_["monitor_recall"],
                          "retrain_week": ds_["retrain_week"], "retrain_is_scheduled_fallback": ds_["retrain_is_scheduled_fallback"],
                          "stale": stale, "retrained": new,
                          "pr_auc_diff_retrained_minus_stale": round(new["pr_auc"] - stale["pr_auc"], 4),
                          "pr_auc_diff_ci": ds_["pr_auc_diff_ci"],
                          "rules": ds_["rules"]}
        doc.section("6. Drift monitoring and retraining simulation (weeks 17-26)")
        doc.row("Alert rules", json.dumps(ds_["rules"]), "drift.py", "results/drift_summary.json")
        doc.row("R1 warning week / R2 retrain week (drift starts week 17)", f"{ds_['r1_week']} / {ds_['r2_week']}", "drift.py", "results/drift_summary.json")
        doc.row("Frozen M2 crisis recall: validation vs monitor", f"{ds_['validation_recall']:.3f} vs {ds_['monitor_recall']:.3f}", "drift.py", "results/drift_summary.json")
        doc.row(f"Retrain at week {ds_['retrain_week']}: recall stale -> retrained", f"{stale['recall']:.3f} -> {new['recall']:.3f}", "drift.py", "results/drift_summary.json")
        doc.row("Retrain: cost per 1000 stale -> retrained", f"{stale['cost_per_1000']:.1f} -> {new['cost_per_1000']:.1f}", "drift.py", "results/drift_summary.json")
        doc.row("Retrain: PR-AUC difference (retrained - stale), 95% family-cluster CI",
                f"{new['pr_auc'] - stale['pr_auc']:+.3f} [{ds_['pr_auc_diff_ci'][0]:+.3f}, {ds_['pr_auc_diff_ci'][1]:+.3f}]", "drift.py", "results/drift_summary.json")
        fam = cload("drift_by_family_type.csv")
        if fam is not None:
            doc.table("Frozen M2 on the monitor window by family type", fam[["group", "n", "n_crisis", "recall", "recall_lo", "recall_hi", "precision"]],
                      "drift.py", "results/drift_by_family_type.csv")
        wk = cload("drift_weekly.csv")
        if wk is not None:
            doc.table("Weekly monitoring table", wk[["week", "n", "voice_share", "oov_rate", "flag_rate", "psi_score", "win_n_crisis", "win_recall", "retro_n_crisis", "retro_recall"]],
                      "drift.py", "results/drift_weekly.csv")
    else:
        facts["drift"] = {"status": "not run"}

    # ------------------------------------------------------------------- LLM
    ls = jload("llm_summary.json")
    if ls:
        facts["llm"] = {"model": ls["model"], "prompt_sha": ls["prompt_sha"], "n_golden": ls["n_golden"], "n_crisis": ls["n_crisis"],
                        "fallbacks": ls["fallbacks"], "mcnemar_llm_vs_m2": ls["mcnemar_llm_vs_m2"], "metrics": ls["metrics"],
                        "note": "zero-shot, one prompt, one run; precision and cost re-weighted from a crisis-enriched sample"}
        doc.section("7. Optional LLM variant (zero-shot, golden subset of the test split)")
        lm = cload("llm_metrics.csv")
        doc.table("Crisis decisions on the 300-case golden subset (recall exact; precision and cost re-weighted)",
                  lm[["method", "tp", "fn", "recall", "recall_lo", "recall_hi", "precision_reweighted", "cost_per_1000_reweighted"]],
                  "llm_agent.py", "results/llm_metrics.csv")
        doc.row("Model / prompt sha / fallbacks", f"{ls['model']} / {ls['prompt_sha']} / {ls['fallbacks']} of {ls['n_golden']}", "llm_agent.py", "results/llm_summary.json")
        m = ls["mcnemar_llm_vs_m2"]
        doc.row("McNemar LLM vs M2 on the true crises", f"LLM-only {m['a_only']}, M2-only {m['b_only']}, p={m['p_value']:.3f}", "llm_agent.py", "results/llm_summary.json")
    else:
        facts["llm"] = {"status": "not run"}

    # ----------------------------------------------------------------- audit
    ad = jload("audit_demo.json")
    if ad:
        facts["audit"] = ad
        doc.section("8. Governance audit trail")
        doc.row("Decisions logged / escalations", f"{ad['n_logged']} / {ad['n_escalations']}", "audit.py", "results/audit_demo.json")
        doc.row("UPDATE attempt", ad["update_attempt"], "audit.py", "results/audit_demo.json")
        doc.row("DELETE attempt", ad["delete_attempt"], "audit.py", "results/audit_demo.json")
        doc.row("Intact chain verifies", str(ad["intact_chain"]["ok"]), "audit.py", "results/audit_demo.json")

    # ------------------------------------------------------- tests and tools
    facts["tests"] = run_pytest()
    facts["tools_used"] = tools_used()
    dash = sorted(p.name for p in (ROOT / "reports" / "dashboard").glob("*.png")) if (ROOT / "reports" / "dashboard").exists() else []
    facts["dashboard"] = {"bi_exports_ready": (ROOT / "bi" / "predictions.csv").exists(),
                          "screenshots": dash, "built": len(dash) >= 4}
    facts["claims_not_supported"] = [
        "clinical validity or any real-patient performance",
        "that the zero-shot LLM or XGBoost beat the simpler model",
        "that retraining demonstrably recovered performance (its PR-AUC confidence interval included 0)",
        "crisis recall of 0.95 or higher on test (it was just below)",
        "any tool not listed under tools_used"]
    doc.section("9. Tests and tools")
    doc.row("pytest", facts["tests"]["summary_line"], "pytest", "tests/")
    doc.row("Tools imported by the code that produced these results", ", ".join(f"{k} {v}" for k, v in facts["tools_used"].items()), "make_results.py", "src/, tests/, notebooks/")
    doc.row("Dashboard built (screenshots found in reports/dashboard)", str(facts["dashboard"]["built"]), "make_results.py", "reports/dashboard/")

    text = (f"# Results\n\n> {DISCLAIMER}\n\nGenerated {facts['generated_utc']} by `src/make_results.py` from the files in `results/`. "
            f"Every number is read from the file named in its row.\n\n" + doc.render())
    (RESULTS / "RESULTS.md").write_text(text)
    (RESULTS / "resume_facts.json").write_text(json.dumps(nan_to_none(facts), indent=2, default=plain))
    print(f"wrote results/RESULTS.md ({len(text)} chars) and results/resume_facts.json")
    print(f"tests: {facts['tests']['summary_line']} | tools: {', '.join(facts['tools_used'])}")
    print(f"criterion 1 met: {c1_met} | LLM section: {'yes' if ls else 'no'} | dashboard built: {facts['dashboard']['built']}")


if __name__ == "__main__":
    main()
