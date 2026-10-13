"""
Slice analysis for the frozen M2 model: recall / false-negative rate by channel, language,
age band, with association tests, effect sizes and Holm correction.

Reads saved predictions only (results/test_predictions.csv, results/oof_dev_predictions.csv).
The test split is NOT re-evaluated here.

ALL DATA IS SYNTHETIC. Run once:
    python -m src.slices
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

from src.config import DATA_PROCESSED, FIGURES, RESULTS, SEED, ensure_dirs, set_seed
from src.evaluate import wilson_ci

SLICE_VARS = ["channel", "language", "age_band"]
GAP = 0.05          # pre-registered: no channel/language slice > 5 points below overall recall
N_PERM = 10000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_test() -> pd.DataFrame:
    t = pd.read_csv(RESULTS / "test_predictions.csv")
    t["crisis"] = (t.risk_level == "crisis").astype(int)
    t["flag"] = t["flag_M2"].astype(int)
    return t


def load_dev_oof(thr: float) -> pd.DataFrame:
    oof = pd.read_csv(RESULTS / "oof_dev_predictions.csv")
    meta = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")[
        ["conversation_id", "channel", "language", "age_band"]]
    d = oof.merge(meta, on="conversation_id", how="left")
    d["crisis"] = (d.risk_level == "crisis").astype(int)
    d["flag"] = (d["p_crisis_M2_tfidf_lr"] >= thr).astype(int)
    return d


# ---------------------------------------------------------------------------
# Slice metrics
# ---------------------------------------------------------------------------
def slice_table(d: pd.DataFrame, dataset: str) -> pd.DataFrame:
    overall = d.loc[d.crisis == 1, "flag"].mean()
    rows = []
    for var in SLICE_VARS:
        for level, g in d.groupby(var):
            pos = g[g.crisis == 1]
            tp = int(pos.flag.sum())
            fn = len(pos) - tp
            fp = int(((g.crisis == 0) & (g.flag == 1)).sum())
            err = int((((g.crisis == 1) & (g.flag == 0)) | ((g.crisis == 0) & (g.flag == 1))).sum())
            lo, hi = wilson_ci(tp, len(pos))
            recall = tp / len(pos) if len(pos) else float("nan")
            rows.append({
                "dataset": dataset, "variable": var, "level": str(level), "n": len(g),
                "n_crisis": len(pos), "tp": tp, "fn": fn, "fp": fp,
                "recall": recall, "recall_lo": lo, "recall_hi": hi,
                "fnr": fn / len(pos) if len(pos) else float("nan"),
                "decision_error_rate": err / len(g),
                "overall_recall": overall, "gap_vs_overall_pts": 100 * (recall - overall),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Association tests
# ---------------------------------------------------------------------------
def _mc_chi2_p(table: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    """Monte Carlo p-value for the chi-square statistic of a (levels x [event, no-event]) table."""
    n_i = table.sum(axis=1).astype(float)
    events, total = float(table[:, 0].sum()), float(n_i.sum())
    e_i = n_i * events / total
    w = 1.0 / e_i + 1.0 / (n_i - e_i)
    levels = np.repeat(np.arange(table.shape[0]), n_i.astype(int))
    outcome = np.concatenate([np.r_[np.ones(table[i, 0], dtype=int), np.zeros(table[i, 1], dtype=int)]
                              for i in range(table.shape[0])])
    stat_obs = float((((table[:, 0] - e_i) ** 2) * w).sum())
    hits = 0
    for _ in range(n_perm):
        o = np.bincount(levels, weights=rng.permutation(outcome), minlength=table.shape[0])
        hits += (((o - e_i) ** 2) * w).sum() >= stat_obs - 1e-9
    return (1 + hits) / (n_perm + 1)


def association_test(table: np.ndarray, rng: np.random.Generator) -> dict:
    """table: rows = slice levels, columns = [event count, non-event count]."""
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2 or table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        return {"method": "not testable (no events or a single level)", "chi2": np.nan,
                "dof": np.nan, "p": np.nan, "cramers_v": np.nan}
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    n, k = table.sum(), min(table.shape) - 1
    v = float(np.sqrt(chi2 / (n * k)))
    if (expected < 5).any():
        if table.shape == (2, 2):
            p, method = float(fisher_exact(table)[1]), "Fisher exact (expected count < 5)"
        else:
            p, method = _mc_chi2_p(table, rng), f"chi-square, Monte Carlo {N_PERM} perms (expected count < 5)"
    else:
        method = "chi-square"
    return {"method": method, "chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": v}


def run_tests(d: pd.DataFrame, dataset: str, rng: np.random.Generator) -> pd.DataFrame:
    d = d.assign(error=(d.crisis != d.flag).astype(int))
    rows = []
    for outcome, sub, event in [("missed_crisis (among true crises)", d[d.crisis == 1], lambda x: 1 - x.flag),
                                ("wrong_decision (FN or FP, all rows)", d, lambda x: x.error)]:
        block = []
        for var in SLICE_VARS:
            ev = event(sub)
            tab = (pd.DataFrame({"level": sub[var].to_numpy(), "event": ev.to_numpy()})
                   .groupby("level").event.agg(["sum", "count"]))
            table = np.column_stack([tab["sum"], tab["count"] - tab["sum"]]).astype(int)
            res = association_test(table, rng)
            block.append({"dataset": dataset, "outcome": outcome, "variable": var,
                          "events": int(tab["sum"].sum()), "n": int(tab["count"].sum()), **res})
        ps = np.array([r["p"] for r in block], dtype=float)
        ok = ~np.isnan(ps)
        adj = np.full(len(ps), np.nan)
        if ok.sum():
            adj[ok] = multipletests(ps[ok], method="holm")[1]
        for r, a in zip(block, adj):
            r["p_holm"] = float(a) if not np.isnan(a) else np.nan
        rows.extend(block)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def forest(tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(10, 4.4))
    for ax, (name, t) in zip(np.atleast_1d(axes), tables.items()):
        t = t[t.n_crisis > 0].reset_index(drop=True)
        y = np.arange(len(t))[::-1]
        ax.errorbar(t.recall, y, xerr=[t.recall - t.recall_lo, t.recall_hi - t.recall], fmt="o",
                    color="#2a78d6", ecolor="#2a78d6", elinewidth=1.5, capsize=3, ms=6, mec="white")
        ov = float(t.overall_recall.iloc[0])
        ax.axvline(ov, color="#52514e", lw=1)
        ax.axvline(ov - GAP, color="#9a9a96", lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{v}={l}  (crisis n={n})" for v, l, n in zip(t.variable, t.level, t.n_crisis)],
                           fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("crisis recall (95% Wilson CI)")
        ax.set_title(f"{name}: overall {ov:.3f} (solid), -5 pts (dashed)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "slice_recall_forest.png", dpi=150)


def main() -> None:
    set_seed()
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    cfg = json.loads((RESULTS / "decision_config.json").read_text())
    thr = cfg["threshold_raw"]

    datasets = {"test (pre-registered)": load_test(), "dev out-of-fold (supplementary)": load_dev_oof(thr)}
    slice_tabs, test_tabs = {}, []
    for name, d in datasets.items():
        st = slice_table(d, name)
        slice_tabs[name] = st
        test_tabs.append(run_tests(d, name, rng))
        print(f"\n===== {name}: n={len(d)}, crisis={int(d.crisis.sum())}, "
              f"overall recall={st.overall_recall.iloc[0]:.3f} =====")
        show = st[["variable", "level", "n", "n_crisis", "tp", "fn", "fp", "recall", "recall_lo",
                   "recall_hi", "fnr", "decision_error_rate", "gap_vs_overall_pts"]]
        print(show.round(3).to_string(index=False))

    all_slices = pd.concat(slice_tabs.values())
    all_tests = pd.concat(test_tabs)
    all_slices.to_csv(RESULTS / "slice_metrics.csv", index=False)
    all_tests.to_csv(RESULTS / "slice_tests.csv", index=False)

    print("\n===== Association tests (Holm-adjusted across the 3 slice variables within each block) =====")
    print(all_tests[["dataset", "outcome", "variable", "events", "n", "method", "chi2", "dof", "p",
                     "p_holm", "cramers_v"]].round(4).to_string(index=False))

    print("\n===== Pre-registered criterion 2: no channel/language slice > 5 points below overall recall =====")
    for name, st in slice_tabs.items():
        sub = st[st.variable.isin(["channel", "language"]) & (st.n_crisis > 0)]
        worst = sub.loc[sub.gap_vs_overall_pts.idxmin()]
        breach = sub[sub.gap_vs_overall_pts < -100 * GAP]
        verdict = "VIOLATED (point estimate)" if len(breach) else "MET (point estimates)"
        print(f"{name}: {verdict}. Worst slice: {worst.variable}={worst.level}, recall {worst.recall:.3f} "
              f"[{worst.recall_lo:.3f}, {worst.recall_hi:.3f}], gap {worst.gap_vs_overall_pts:+.1f} pts "
              f"on {int(worst.n_crisis)} crisis cases")
        for _, r in breach.iterrows():
            print(f"   breach: {r.variable}={r.level}: {r.gap_vs_overall_pts:+.1f} pts, "
                  f"CI [{r.recall_lo:.3f}, {r.recall_hi:.3f}] "
                  "(descriptive only; see the Fisher/Holm tests for significance)")

    forest(slice_tabs)
    print("\nwrote results/slice_metrics.csv, results/slice_tests.csv, reports/figures/slice_recall_forest.png")


if __name__ == "__main__":
    main()
