"""
Evaluation metrics for the crisis-detection task.

This is the Phase 4 subset (rung reports). Phase 5 extends this file with the
cost-based threshold, bootstrap CIs and McNemar test.

ALL DATA IS SYNTHETIC.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

CLASSES = ["no_risk", "low", "moderate", "crisis"]
CRISIS_IDX = CLASSES.index("crisis")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def crisis_pr_auc(y_crisis: np.ndarray, score: np.ndarray) -> float:
    """Average precision for the crisis class. Chance level = prevalence."""
    return float(average_precision_score(y_crisis, score))


def crisis_confusion(y_crisis: np.ndarray, flagged: np.ndarray) -> dict:
    y, f = np.asarray(y_crisis).astype(bool), np.asarray(flagged).astype(bool)
    tp, fp = int((y & f).sum()), int((~y & f).sum())
    fn, tn = int((y & ~f).sum()), int((~y & ~f).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def crisis_report(y_crisis: np.ndarray, flagged: np.ndarray, beta: float = 2.0) -> dict:
    c = crisis_confusion(y_crisis, flagged)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    b2 = beta**2
    fbeta = ((1 + b2) * precision * recall / (b2 * precision + recall)
             if (precision == precision and recall == recall and (b2 * precision + recall) > 0)
             else 0.0)
    r_lo, r_hi = wilson_ci(tp, tp + fn)
    p_lo, p_hi = wilson_ci(tp, tp + fp)
    return {**c, "recall": recall, "recall_lo": r_lo, "recall_hi": r_hi,
            "precision": precision, "precision_lo": p_lo, "precision_hi": p_hi,
            "f2": fbeta, "n_flagged": tp + fp}


def rung_report(name: str, df: pd.DataFrame, proba: np.ndarray, split_name: str) -> dict:
    """Summary of one model on one split, using the model's own argmax decision.

    (Phase 5 replaces argmax with a cost-minimising threshold chosen on validation.)"""
    y_true = df.risk_level.to_numpy()
    y_pred = np.array(CLASSES)[proba.argmax(axis=1)]
    y_crisis = (y_true == "crisis").astype(int)
    flagged = (y_pred == "crisis").astype(int)
    rep = crisis_report(y_crisis, flagged)
    return {
        "model": name, "split": split_name, "n": len(df), "n_crisis": int(y_crisis.sum()),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "crisis_pr_auc": crisis_pr_auc(y_crisis, proba[:, CRISIS_IDX]),
        "prevalence": float(y_crisis.mean()),
        **{k: rep[k] for k in ["recall", "recall_lo", "recall_hi", "precision", "f2",
                               "tp", "fp", "fn", "n_flagged"]},
        "confusion": confusion_matrix(y_true, y_pred, labels=CLASSES),
    }


def print_rung(rep: dict) -> None:
    print(f"\n=== {rep['model']} on {rep['split']} (n={rep['n']}, crisis={rep['n_crisis']}) ===")
    print(f"accuracy        {rep['accuracy']:.3f}")
    print(f"macro-F1        {rep['macro_f1']:.3f}")
    print(f"crisis PR-AUC   {rep['crisis_pr_auc']:.3f}   (chance = prevalence = {rep['prevalence']:.3f})")
    p = rep["precision"]
    print(f"crisis recall   {rep['recall']:.3f}  [95% Wilson {rep['recall_lo']:.3f}, {rep['recall_hi']:.3f}]"
          f"   (TP={rep['tp']}, FN={rep['fn']})")
    print(f"crisis precision {p:.3f}   (FP={rep['fp']}, flagged={rep['n_flagged']})" if p == p
          else "crisis precision n/a (nothing flagged)")
    print(f"crisis F2       {rep['f2']:.3f}")
    print("confusion matrix (rows = recorded label, cols = predicted):")
    print(pd.DataFrame(rep["confusion"], index=CLASSES, columns=CLASSES).to_string())


# ---------------------------------------------------------------------------
# Family-clustered bootstrap (used by the promotion test and, in Phase 5, the CIs)
# ---------------------------------------------------------------------------
def cluster_indices(groups: np.ndarray) -> list[np.ndarray]:
    """Row indices of each group (template family)."""
    groups = np.asarray(groups)
    return [np.where(groups == g)[0] for g in np.unique(groups)]


def cluster_bootstrap_pr_auc(y_crisis: np.ndarray, scores: dict, groups: np.ndarray,
                             n_boot: int = 1000, seed: int = 42) -> tuple[dict, int]:
    """Resample whole families with replacement; return crisis PR-AUC per model per resample.

    All models are scored on the SAME resamples, so differences are paired.
    Resamples with no crisis case are skipped (PR-AUC is undefined); the count is returned."""
    rng = np.random.default_rng(seed)
    fam_idx = cluster_indices(groups)
    k = len(fam_idx)
    y_crisis = np.asarray(y_crisis)
    out = {name: [] for name in scores}
    skipped = 0
    for _ in range(n_boot):
        idx = np.concatenate([fam_idx[i] for i in rng.integers(0, k, k)])
        yb = y_crisis[idx]
        if yb.sum() == 0:
            skipped += 1
            continue
        for name, sc in scores.items():
            out[name].append(average_precision_score(yb, np.asarray(sc)[idx]))
    return {n: np.asarray(v) for n, v in out.items()}, skipped


def percentile_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return (float(np.percentile(x, 100 * alpha / 2)), float(np.percentile(x, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------
# Phase 5: calibration metrics, cost-based threshold, McNemar
# ---------------------------------------------------------------------------
def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece_quantile(y, p, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Expected calibration error with equal-frequency bins, plus the per-bin table."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    rows, ece = [], 0.0
    for idx in np.array_split(np.argsort(p, kind="stable"), n_bins):
        if len(idx) == 0:
            continue
        conf, obs = float(p[idx].mean()), float(y[idx].mean())
        ece += len(idx) / len(p) * abs(conf - obs)
        rows.append({"bin_mean_pred": conf, "bin_frac_pos": obs, "n": int(len(idx))})
    return float(ece), pd.DataFrame(rows)


def cost_curve(y, p, c_fn: float = 20.0, c_fp: float = 1.0) -> pd.DataFrame:
    """Confusion counts and expected cost at every useful threshold (flag if p >= t)."""
    y, p = np.asarray(y).astype(bool), np.asarray(p, dtype=float)
    uniq = np.unique(p)
    thresholds = np.concatenate([[uniq.min() - 1e-9], (uniq[:-1] + uniq[1:]) / 2, [uniq.max() + 1e-9]])
    rows = []
    for t in thresholds:
        f = p >= t
        tp, fp, fn = int((y & f).sum()), int((~y & f).sum()), int((y & ~f).sum())
        cost = c_fn * fn + c_fp * fp
        rows.append({"threshold": float(t), "tp": tp, "fp": fp, "fn": fn,
                     "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
                     "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                     "cost": cost, "cost_per_1000": 1000 * cost / len(y)})
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame) -> float:
    """Minimum expected cost. Ties go to the LOWEST threshold (favours recall)."""
    return float(curve.loc[curve.cost == curve.cost.min(), "threshold"].min())


def recall_at_precision(curve: pd.DataFrame, min_precision: float = 0.30) -> dict:
    """Best recall reachable while precision stays >= min_precision."""
    ok = curve[(curve.precision >= min_precision) & (curve.tp > 0)]
    if ok.empty:
        return {"recall": 0.0, "precision": float("nan"), "threshold": float("nan")}
    best = ok.loc[ok.recall == ok.recall.max()].sort_values("threshold").iloc[-1]
    return {"recall": float(best.recall), "precision": float(best.precision),
            "threshold": float(best.threshold)}


def mcnemar_exact(detected_a, detected_b) -> dict:
    """Exact McNemar test on paired crisis detection (inputs: detected? for each TRUE crisis case)."""
    from scipy.stats import binomtest
    a, b = np.asarray(detected_a).astype(bool), np.asarray(detected_b).astype(bool)
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_value": p}
