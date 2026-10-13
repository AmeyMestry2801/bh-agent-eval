"""
Phase 8: drift monitoring and retraining simulation on the monitor window (weeks 17-26).
Uses the FROZEN M2 and its frozen raw threshold. The TEST split is never loaded.
Rules are pre-registered in docs/monitoring_plan.md.

ALL DATA IS SYNTHETIC. Run once:
    python -m src.drift
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import (COST_FN_CRISIS, COST_FP_CRISIS, DATA_PROCESSED, FIGURES, RESULTS,
                        ensure_dirs, set_seed)
from src.evaluate import (CRISIS_IDX, choose_threshold, cluster_bootstrap_pr_auc, cost_curve,
                          crisis_pr_auc, crisis_report, percentile_ci, wilson_ci)
from src.features import _is_negation, tokenize
from src.models import M2TfidfLR

PSI_LIMIT = 0.25
OOV_MULT = 2.0
RECALL_FLOOR = 0.90
MIN_WINDOW_CRISIS = 10
WINDOW_WEEKS = 4
FALLBACK_RETRAIN_WEEK = 21
N_BOOT = 1000


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def psi(ref, cur, n_bins: int = 10) -> float:
    lr, lc = _logit(ref), _logit(cur)
    edges = np.unique(np.quantile(lr, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bins = np.concatenate([[-np.inf], edges, [np.inf]])
    r = np.clip(np.histogram(lr, bins)[0] / len(lr), 1e-4, None)
    c = np.clip(np.histogram(lc, bins)[0] / len(lc), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def oov_rate(text: str, vocab: set) -> float:
    toks = tokenize(text)
    return sum(t not in vocab for t in toks) / max(len(toks), 1)


def neg_rate(text: str) -> float:
    toks = tokenize(text)
    return sum(_is_negation(t) for t in toks) / max(len(toks), 1)


def window_stats(d: pd.DataFrame) -> dict:
    n_crisis = int(d.y.sum())
    tp = int(((d.y == 1) & (d.flag == 1)).sum())
    fp = int(((d.y == 0) & (d.flag == 1)).sum())
    rec = tp / n_crisis if n_crisis else float("nan")
    lo, hi = wilson_ci(tp, n_crisis) if n_crisis else (float("nan"), float("nan"))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    return {"n": len(d), "n_crisis": n_crisis, "tp": tp, "fp": fp,
            "recall": rec, "recall_lo": lo, "recall_hi": hi, "precision": prec}


def consecutive_hit(flags: list[bool], weeks: list[int]) -> int | None:
    """First week at which the flag has been true for 2 consecutive weeks."""
    for i in range(1, len(flags)):
        if flags[i] and flags[i - 1]:
            return weeks[i]
    return None


def fit_with_threshold(train_df: pd.DataFrame, c_value: float):
    y = (train_df.risk_level == "crisis").to_numpy().astype(int)
    model = M2TfidfLR(fixed_C=c_value).fit(train_df)
    oof = np.zeros(len(train_df))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(train_df, y, groups=train_df.template_family):
        m = M2TfidfLR(fixed_C=c_value).fit(train_df.iloc[tr])
        oof[te] = m.predict_proba(train_df.iloc[te])[:, CRISIS_IDX]
    thr = choose_threshold(cost_curve(y, oof, COST_FN_CRISIS, COST_FP_CRISIS))
    return model, float(thr)


def plain(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def main() -> None:
    set_seed()
    ensure_dirs()
    cfg = json.loads((RESULTS / "decision_config.json").read_text())
    thr = float(cfg["threshold_raw"])
    m2 = joblib.load(RESULTS / "models" / "M2.joblib")

    df = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    train = df[df.split == "train"].copy()
    val = df[df.split == "val"].copy()
    mon = df[df.split == "monitor"].copy().reset_index(drop=True)
    print(f"frozen M2, raw threshold {thr:.4f} | val n={len(val)} | monitor n={len(mon)} "
          f"weeks {int(mon.week.min())}-{int(mon.week.max())}")

    # ---- scores and reference statistics --------------------------------
    p_val = m2.predict_proba(val)[:, CRISIS_IDX]
    mon["p"] = m2.predict_proba(mon)[:, CRISIS_IDX]
    mon["y"] = (mon.risk_level == "crisis").astype(int)
    mon["flag"] = (mon.p >= thr).astype(int)
    mon["arrival"] = mon.week + mon.label_delay_weeks
    vocab = set(t for s in train.text for t in tokenize(s))
    mon["oov"] = [oov_rate(s, vocab) for s in mon.text]
    mon["neg"] = [neg_rate(s) for s in mon.text]
    mon["is_voice"] = (mon.channel == "voice").astype(int)
    mon["is_D"] = mon.template_family.astype(str).str.startswith("D-")
    ref_oov = float(np.mean([oov_rate(s, vocab) for s in val.text]))
    ref_voice = float((val.channel == "voice").mean())
    ref_flag = float((p_val >= thr).mean())
    print(f"reference (validation): OOV rate {ref_oov:.4f} | voice share {ref_voice:.3f} "
          f"| flagged rate {ref_flag:.4f}")

    # ---- weekly monitoring table ----------------------------------------
    weeks = sorted(int(w) for w in mon.week.unique())
    rows = []
    for t in weeks:
        cur = mon[mon.week == t]
        win = mon[(mon.arrival <= t) & (mon.arrival > t - WINDOW_WEEKS)]
        ws = window_stats(win)
        retro = window_stats(cur)
        rows.append({
            "week": t, "n": len(cur),
            "voice_share": cur.is_voice.mean(), "oov_rate": cur.oov.mean(),
            "neg_rate": cur.neg.mean(), "flag_rate": cur.flag.mean(),
            "psi_score": psi(p_val, cur.p.to_numpy()),
            "share_D_families_(hidden_in_prod)": cur.is_D.mean(),
            "win_n_crisis": ws["n_crisis"], "win_recall": ws["recall"],
            "win_recall_lo": ws["recall_lo"], "win_recall_hi": ws["recall_hi"],
            "retro_n_crisis": retro["n_crisis"], "retro_recall": retro["recall"],
            "retro_recall_lo": retro["recall_lo"], "retro_recall_hi": retro["recall_hi"],
            "retro_precision": retro["precision"],
        })
    wk = pd.DataFrame(rows)
    wk.to_csv(RESULTS / "drift_weekly.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n--- weekly monitoring (win_* = what is KNOWN at week t given label delay; "
          "retro_* = truth once all labels are in) ---")
    print(wk.round(3).to_string(index=False))

    # ---- rules -----------------------------------------------------------
    r1_flags = [(r.psi_score > PSI_LIMIT) or (r.oov_rate > OOV_MULT * ref_oov)
                for r in wk.itertuples()]
    r1_week = consecutive_hit(r1_flags, weeks)
    ev = wk[wk.win_n_crisis >= MIN_WINDOW_CRISIS]
    r2_week = consecutive_hit([bool(x < RECALL_FLOOR) for x in ev.win_recall],
                              [int(w) for w in ev.week])
    print(f"\nR1 warning (PSI>{PSI_LIMIT} or OOV>{OOV_MULT}x ref, 2 consecutive weeks): "
          f"{'week ' + str(r1_week) if r1_week else 'never fired'}")
    print(f"R2 retrain  (windowed recall<{RECALL_FLOOR}, >= {MIN_WINDOW_CRISIS} labelled crisis, "
          f"2 consecutive weeks): {'week ' + str(r2_week) if r2_week else 'never fired'}")
    print(f"weeks with enough labelled crisis cases to evaluate R2: "
          f"{[int(w) for w in ev.week]}")
    print("true drift start in the generator: week 17 (known only because data is synthetic)")

    # ---- retrospective: did performance really degrade? -------------------
    print("\n--- frozen M2 at frozen threshold: validation vs monitor (all labels known) ---")
    yv = (val.risk_level == "crisis").to_numpy().astype(int)
    rv = crisis_report(yv, (p_val >= thr).astype(int))
    print(f"validation weeks 13-16: recall {rv['recall']:.3f} [{rv['recall_lo']:.3f}, "
          f"{rv['recall_hi']:.3f}] n_crisis={rv['tp'] + rv['fn']} precision {rv['precision']:.3f}")
    allm = window_stats(mon)
    print(f"monitor weeks {weeks[0]}-{weeks[-1]}: recall {allm['recall']:.3f} "
          f"[{allm['recall_lo']:.3f}, {allm['recall_hi']:.3f}] n_crisis={allm['n_crisis']} "
          f"precision {allm['precision']:.3f}")
    fam_rows = []
    for name, d in [("new D-* families", mon[mon.is_D]), ("old families", mon[~mon.is_D])]:
        s = window_stats(d)
        fam_rows.append({"group": name, **s})
        print(f"  {name:18s} n={s['n']:5d} n_crisis={s['n_crisis']:3d} recall {s['recall']:.3f} "
              f"[{s['recall_lo']:.3f}, {s['recall_hi']:.3f}] precision {s['precision']:.3f}")
    pd.DataFrame(fam_rows).to_csv(RESULTS / "drift_by_family_type.csv", index=False)

    # ---- retraining simulation -------------------------------------------
    scheduled = r2_week is None
    T = int(r2_week if r2_week else FALLBACK_RETRAIN_WEEK)
    print(f"\n--- retraining simulation at week {T}"
          f"{' (SCHEDULED FALLBACK: R2 never fired)' if scheduled else ' (R2 trigger)'} ---")
    avail = mon[mon.arrival <= T]
    fut = mon[mon.week > T].copy().reset_index(drop=True)
    retrain_df = pd.concat([train, val, avail.drop(columns=[c for c in mon.columns
                                                              if c not in df.columns])],
                           ignore_index=True)
    print(f"retrain set: {len(retrain_df)} rows ({len(avail)} from monitor with labels arrived "
          f"by week {T}); evaluation on monitor weeks > {T}: {len(fut)} rows, "
          f"{int(fut.y.sum())} crisis")
    new_model, new_thr = fit_with_threshold(retrain_df, m2.best_C_)
    p_new = new_model.predict_proba(fut)[:, CRISIS_IDX]
    p_old = fut.p.to_numpy()
    y = fut.y.to_numpy()
    res = {}
    for name, p, th in [("stale (frozen)", p_old, thr), ("retrained", p_new, new_thr)]:
        rep = crisis_report(y, (p >= th).astype(int))
        cost = COST_FN_CRISIS * rep["fn"] + COST_FP_CRISIS * rep["fp"]
        res[name] = {"threshold": th, "pr_auc": crisis_pr_auc(y, p), "recall": rep["recall"],
                     "recall_lo": rep["recall_lo"], "recall_hi": rep["recall_hi"],
                     "precision": rep["precision"], "fn": rep["fn"], "fp": rep["fp"],
                     "cost_per_1000": 1000 * cost / len(y)}
        r = res[name]
        print(f"  {name:15s} thr {th:.4f} | PR-AUC {r['pr_auc']:.3f} | recall {r['recall']:.3f} "
              f"[{r['recall_lo']:.3f}, {r['recall_hi']:.3f}] | precision {r['precision']:.3f} "
              f"| FN={r['fn']} FP={r['fp']} | cost/1000 {r['cost_per_1000']:.1f}")
    boots, skipped = cluster_bootstrap_pr_auc(y, {"stale": p_old, "retrained": p_new},
                                              fut.template_family.to_numpy(), n_boot=N_BOOT)
    diff = boots["retrained"] - boots["stale"]
    lo, hi = percentile_ci(diff)
    print(f"  PR-AUC retrained - stale: {res['retrained']['pr_auc'] - res['stale (frozen)']['pr_auc']:+.3f} "
          f"95% family-cluster bootstrap CI [{lo:+.3f}, {hi:+.3f}] (skipped resamples: {skipped})")
    print("  caveat: drift families seen before the retrain week also appear after it, so this "
          "measures recovery from a persisting shift, not generalisation to unseen slang.")

    out = {"threshold_frozen": thr, "ref_oov": ref_oov, "ref_voice": ref_voice,
           "r1_week": r1_week, "r2_week": r2_week, "retrain_week": T,
           "retrain_is_scheduled_fallback": scheduled,
           "validation_recall": rv["recall"], "monitor_recall": allm["recall"],
           "retrain_comparison": res,
           "pr_auc_diff_ci": [lo, hi], "n_boot": N_BOOT,
           "rules": {"psi_limit": PSI_LIMIT, "oov_mult": OOV_MULT,
                     "recall_floor": RECALL_FLOOR, "min_window_crisis": MIN_WINDOW_CRISIS,
                     "window_weeks": WINDOW_WEEKS}}
    (RESULTS / "drift_summary.json").write_text(json.dumps(out, indent=2, default=plain))

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    w = wk.week
    ax[0].plot(w, wk.psi_score, "o-"); ax[0].axhline(PSI_LIMIT, ls="--", c="r")
    ax[0].set_ylabel("PSI of crisis score")
    ax[1].plot(w, wk.oov_rate, "o-"); ax[1].axhline(OOV_MULT * ref_oov, ls="--", c="r")
    ax[1].set_ylabel("OOV rate")
    ax[2].plot(w, wk.voice_share, "o-", label="voice share")
    ax[2].plot(w, wk.flag_rate, "s-", label="flagged rate"); ax[2].legend()
    ax[2].set_ylabel("share")
    ax[3].plot(w, wk.retro_recall, "o-", label="retrospective (all labels)")
    ax[3].fill_between(w, wk.retro_recall_lo, wk.retro_recall_hi, alpha=0.2)
    ax[3].axhline(0.95, ls=":", c="g", label="target 0.95")
    ax[3].axhline(RECALL_FLOOR, ls="--", c="r", label="R2 floor 0.90"); ax[3].legend(fontsize=8)
    ax[3].set_ylabel("crisis recall")
    ax[4].plot(w, wk.win_recall, "o-", c="C1", label="known at week t (label delay)")
    ax[4].fill_between(w, wk.win_recall_lo, wk.win_recall_hi, alpha=0.2, color="C1")
    ax[4].axhline(RECALL_FLOOR, ls="--", c="r"); ax[4].legend(fontsize=8)
    ax[4].set_ylabel("windowed recall"); ax[4].set_xlabel("week")
    for a in ax:
        a.axvline(16.5, c="gray", ls=":")
        for mark, col in [(r1_week, "orange"), (r2_week, "red")]:
            if mark:
                a.axvline(mark, c=col, alpha=0.5)
    fig.suptitle("Drift monitoring, weeks 17-26 (SYNTHETIC data; gray=drift start, "
                 "orange=R1, red=R2)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "drift_timeline.png", dpi=130)
    print(f"\nwrote results/drift_weekly.csv, drift_by_family_type.csv, drift_summary.json, "
          f"{FIGURES.name}/drift_timeline.png")


if __name__ == "__main__":
    main()
