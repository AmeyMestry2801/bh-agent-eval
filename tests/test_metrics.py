import math

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (brier, choose_threshold, cluster_bootstrap_pr_auc, cost_curve,
                          crisis_confusion, crisis_pr_auc, crisis_report, ece_quantile,
                          mcnemar_exact, percentile_ci, recall_at_precision, wilson_ci)


def test_wilson_known_values():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4901, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == pytest.approx(0.0, abs=1e-9) and hi0 == pytest.approx(0.2775, abs=1e-3)
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0, abs=1e-9)
    assert all(math.isnan(x) for x in wilson_ci(0, 0))


def test_confusion_and_report_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0])
    f = np.array([1, 1, 0, 1, 0, 0])
    assert crisis_confusion(y, f) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    rep = crisis_report(y, f)
    assert rep["recall"] == pytest.approx(2 / 3)
    assert rep["precision"] == pytest.approx(2 / 3)
    assert rep["f2"] == pytest.approx(2 / 3)
    assert rep["n_flagged"] == 3


def test_report_when_nothing_flagged():
    rep = crisis_report(np.array([1, 0]), np.array([0, 0]))
    assert rep["recall"] == 0.0 and math.isnan(rep["precision"]) and rep["f2"] == 0.0


def test_cost_curve_and_threshold_hand_computed():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.35, 0.2, 0.1])
    curve = cost_curve(y, p, 20.0, 1.0)
    assert (curve.tp + curve.fn == 2).all()
    assert curve.cost.min() == 0.0
    assert choose_threshold(curve) == pytest.approx(0.375)


def test_threshold_ties_go_to_lowest():
    curve = pd.DataFrame({"threshold": [0.3, 0.1, 0.2], "cost": [5.0, 5.0, 9.0]})
    assert choose_threshold(curve) == 0.1


def test_recall_at_precision():
    curve = pd.DataFrame({"threshold": [0.1, 0.3, 0.6], "tp": [5, 4, 2],
                          "precision": [0.2, 0.5, 0.9], "recall": [1.0, 0.8, 0.5]})
    best = recall_at_precision(curve, 0.30)
    assert best["recall"] == 0.8 and best["threshold"] == 0.3
    none = recall_at_precision(curve.assign(precision=0.1), 0.30)
    assert none["recall"] == 0.0 and math.isnan(none["threshold"])


def test_mcnemar_exact():
    a = [1, 1, 1, 1, 1, 1, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 0, 0]
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"], r["discordant"]) == (6, 0, 6)
    assert r["p_value"] == pytest.approx(0.03125)
    assert mcnemar_exact(b, a)["p_value"] == pytest.approx(0.03125)
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_brier():
    assert brier([1, 0], [1.0, 0.0]) == 0.0
    assert brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_ece():
    y = np.r_[np.zeros(50), np.ones(50)]
    assert ece_quantile(y, y)[0] == pytest.approx(0.0)
    assert ece_quantile(y, np.full(100, 0.9))[0] == pytest.approx(0.5)


def test_cluster_bootstrap_is_paired_deterministic_and_sane():
    groups = np.repeat(np.arange(20), 3)
    y = np.tile([1, 0, 0], 20)
    scores = {"perfect": y.astype(float), "random": np.random.default_rng(0).random(60)}
    out, skipped = cluster_bootstrap_pr_auc(y, scores, groups, n_boot=50, seed=1)
    assert skipped == 0 and len(out["perfect"]) == len(out["random"]) == 50
    assert np.allclose(out["perfect"], 1.0)
    assert out["random"].mean() < 0.9
    out2, _ = cluster_bootstrap_pr_auc(y, scores, groups, n_boot=50, seed=1)
    assert np.array_equal(out["random"], out2["random"])


def test_percentile_ci_and_pr_auc():
    assert percentile_ci(np.arange(101)) == (pytest.approx(2.5), pytest.approx(97.5))
    assert crisis_pr_auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx(1.0)
