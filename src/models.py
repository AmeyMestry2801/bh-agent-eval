"""
Model ladder. Each rung is reported on the VALIDATION split before moving up.
The TEST split is never loaded here.

    M0  majority class / class prior
    M1  hand-written keyword + regex rules (the "naive agent")
    M2  TF-IDF (word + char n-grams) + Logistic Regression, C tuned by GroupKFold
    M3  XGBoost on TF-IDF-SVD + engineered features, tuned by RandomizedSearchCV

ALL DATA IS SYNTHETIC. Run from the project root:
    python -m src.models --rungs M0 M1 M2 M3
"""
from __future__ import annotations

import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import DATA_PROCESSED, RESULTS, SEED, ensure_dirs, set_seed
from src.evaluate import CLASSES, CRISIS_IDX, crisis_pr_auc, print_rung, rung_report
from src.features import TextStats
from src.split import NON_FEATURE_COLUMNS  # noqa: F401  (models must never use these as inputs)


# ---------------------------------------------------------------------------
# M0: majority class and class prior
# ---------------------------------------------------------------------------
class M0Prior:
    """Ignores the text. Predicts the training class prior for every row."""
    name = "M0_prior"

    def fit(self, train: pd.DataFrame) -> "M0Prior":
        self.prior_ = (train.risk_level.value_counts(normalize=True)
                       .reindex(CLASSES).fillna(0.0).to_numpy())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.prior_, (len(df), 1))


# ---------------------------------------------------------------------------
# M1: hand-written rules (the "naive agent")
# ---------------------------------------------------------------------------
# Written from general knowledge of crisis language BEFORE looking at model results.
# Deliberately English-only and without the post-launch slang ("unalive", "kms"):
# a rules agent is written once, before deployment.
# CAVEAT: the author also wrote the data generator, so this baseline is probably
# stronger than a rule set written by someone who had never seen the data.
CRISIS_PATTERNS = [
    r"\bkill(?:ing)? my ?self\b",
    r"\bsuicid\w*",
    r"\bend(?:ing)? (?:my life|it all)\b",
    r"\bwant(?:ed)? to die\b",
    r"\b(?:don'?t|do not) want to (?:be alive|live|wake up)\b",
    r"\bhurt(?:ing)? my ?self\b",
    r"\boverdos\w*",
    r"\bgoodbye note\b",
    r"\bbetter off without me\b",
    r"\b(?:took|swallowed)\b.{0,30}\bpills\b",
]
MODERATE_PATTERNS = [
    r"\bhopeless\w*",
    r"\bburden\b",
    r"\bdisappear\b|\bvanish\b",
    r"\bthoughts? of (?:death|dying)\b|\bthink(?:ing)? about dying\b",
    r"\bcutting\b|\bscratching myself\b",
    r"\bpanic\w*",
    r"\b(?:haven'?t|barely) slept\b|\bcan'?t sleep\b",
    r"\bwish i (?:was|were|could)\b",
    r"\bnot exist\b|\bsleep forever\b",
]
LOW_PATTERNS = [
    r"\banxi\w*", r"\bdepress\w*", r"\bstress\w*", r"\bcoping\b",
    r"\b(?:sad|down|flat|nervous|tense|worr\w*|burnout|overwhelmed|exhausted)\b",
]
_CRISIS = [re.compile(p, re.I) for p in CRISIS_PATTERNS]
_MODERATE = [re.compile(p, re.I) for p in MODERATE_PATTERNS]
_LOW = [re.compile(p, re.I) for p in LOW_PATTERNS]
_NEGATION = re.compile(r"\b(?:not|never|no|without)\b|n't\b", re.I)
NEG_WINDOW_CHARS = 30  # look this many characters before a keyword for a negation word


def _hits(text: str, patterns: list[re.Pattern]) -> list[str]:
    """Keyword matches that are NOT preceded by a negation word within the window."""
    out = []
    for pat in patterns:
        for m in pat.finditer(text):
            window = text[max(0, m.start() - NEG_WINDOW_CHARS):m.start()]
            if not _NEGATION.search(window):
                out.append(m.group(0).lower())
    return out


class M1Rules:
    """Keyword rules with a simple negation window. No training, no learned weights."""
    name = "M1_rules"

    def fit(self, train: pd.DataFrame) -> "M1Rules":
        return self

    def explain(self, text: str) -> dict:
        return {"crisis": _hits(text, _CRISIS), "moderate": _hits(text, _MODERATE),
                "low": _hits(text, _LOW)}

    def predict_class(self, text: str) -> str:
        if _hits(text, _CRISIS):
            return "crisis"
        if _hits(text, _MODERATE):
            return "moderate"
        if _hits(text, _LOW):
            return "low"
        return "no_risk"

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        idx = {c: i for i, c in enumerate(CLASSES)}
        out = np.zeros((len(df), len(CLASSES)))
        for r, text in enumerate(df.text.to_numpy()):
            out[r, idx[self.predict_class(text)]] = 1.0
        return out


# ---------------------------------------------------------------------------
# M2: TF-IDF (word + char n-grams) + Logistic Regression
# ---------------------------------------------------------------------------
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
N_FOLDS = 5


def make_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    return word, char


def _fit_transform(vecs, texts):
    return hstack([v.fit_transform(texts) for v in vecs]).tocsr()


def _transform(vecs, texts):
    return hstack([v.transform(texts) for v in vecs]).tocsr()


def _ordered_proba(clf: LogisticRegression, X) -> np.ndarray:
    """predict_proba columns re-ordered to CLASSES."""
    raw = clf.predict_proba(X)
    return raw[:, [list(clf.classes_).index(c) for c in CLASSES]]


class M2TfidfLR:
    """TF-IDF word(1-2) + char_wb(2-5) -> multinomial LR, class_weight='balanced'.
    C is chosen by GroupKFold (groups = template_family) on the training split only."""
    name = "M2_tfidf_lr"

    def __init__(self, c_grid=C_GRID, n_folds: int = N_FOLDS, fixed_C: float | None = None):
        self.c_grid, self.n_folds, self.fixed_C = c_grid, n_folds, fixed_C

    def _lr(self, C: float) -> LogisticRegression:
        return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)

    def tune(self, train: pd.DataFrame) -> pd.DataFrame:
        y = train.risk_level.to_numpy()
        y_cr = (y == "crisis").astype(int)
        gkf = GroupKFold(n_splits=self.n_folds)
        rows = []
        folds = list(gkf.split(train.text, y, groups=train.template_family))
        fold_data = []
        for tr, va in folds:  # fit TF-IDF inside each fold (no leakage), reuse across C values
            vecs = make_vectorizers()
            Xtr = _fit_transform(vecs, train.text.iloc[tr])
            Xva = _transform(vecs, train.text.iloc[va])
            fold_data.append((Xtr, y[tr], Xva, y_cr[va]))
        for C in self.c_grid:
            scores = []
            for Xtr, ytr, Xva, yva_cr in fold_data:
                clf = self._lr(C).fit(Xtr, ytr)
                scores.append(crisis_pr_auc(yva_cr, _ordered_proba(clf, Xva)[:, CRISIS_IDX]))
            rows.append({"C": C, "mean_oof_crisis_pr_auc": float(np.mean(scores)),
                         "sd": float(np.std(scores, ddof=1)),
                         **{f"fold{i + 1}": sc for i, sc in enumerate(scores)}})
        return pd.DataFrame(rows)

    def fit(self, train: pd.DataFrame) -> "M2TfidfLR":
        if self.fixed_C is None:   # tune; otherwise reuse a C that was already tuned on train
            self.cv_ = self.tune(train)
            self.best_C_ = float(self.cv_.loc[self.cv_.mean_oof_crisis_pr_auc.idxmax(), "C"])
        else:
            self.best_C_ = float(self.fixed_C)
        self.vecs_ = make_vectorizers()
        X = _fit_transform(self.vecs_, train.text)
        self.clf_ = self._lr(self.best_C_).fit(X, train.risk_level.to_numpy())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return _ordered_proba(self.clf_, _transform(self.vecs_, df.text))


# ---------------------------------------------------------------------------
# M3: XGBoost on TF-IDF-SVD + engineered features
# ---------------------------------------------------------------------------
N_ITER = 25          # fixed random-search budget (pre-registered in docs/model_plan.md)
SVD_COMPONENTS = 100
PARAM_SPACE = {
    "xgb__n_estimators": [100, 200, 300, 400],
    "xgb__max_depth": [2, 3, 4, 5, 6],
    "xgb__learning_rate": [0.02, 0.05, 0.1, 0.2, 0.3],
    "xgb__subsample": [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.5, 0.7, 1.0],
    "xgb__min_child_weight": [1, 3, 5],
    "xgb__reg_lambda": [0.5, 1, 5, 10],
    "xgb__gamma": [0, 0.5, 1],
}


def _encode(labels) -> np.ndarray:
    return np.array([CLASSES.index(c) for c in labels])


def crisis_ap_scorer(estimator, X, y) -> float:
    """Search metric: crisis-class average precision (y is encoded in CLASSES order)."""
    proba = estimator.predict_proba(X)
    return float(average_precision_score(y == CRISIS_IDX, proba[:, CRISIS_IDX]))


def build_m3_pipeline(seed: int = SEED, memory=None, **xgb_params) -> Pipeline:
    svd_branch = Pipeline([
        ("tfidf", FeatureUnion(list(zip(["word", "char"], make_vectorizers())))),
        ("svd", TruncatedSVD(n_components=SVD_COMPONENTS, random_state=seed)),
    ])
    features = ColumnTransformer([
        ("svd", svd_branch, "text"),
        ("stats", TextStats(), ["text", "channel"]),
    ])
    xgb = XGBClassifier(objective="multi:softprob", num_class=len(CLASSES), tree_method="hist",
                        random_state=seed, n_jobs=4, verbosity=0, **xgb_params)
    return Pipeline([("features", features), ("xgb", xgb)], memory=memory)


class M3XGB:
    """XGBoost on [TF-IDF -> SVD(100)] + 6 engineered features.
    RandomizedSearchCV: 25 candidates, GroupKFold(5) by template_family, crisis-AP metric,
    class-balanced sample weights. TF-IDF/SVD are refit inside every fold."""
    name = "M3_xgb"

    def __init__(self, n_iter: int = N_ITER, n_folds: int = N_FOLDS,
                 fixed_params: dict | None = None):
        self.n_iter, self.n_folds, self.fixed_params = n_iter, n_folds, fixed_params

    def fit(self, train: pd.DataFrame) -> "M3XGB":
        X, y = train[["text", "channel"]], _encode(train.risk_level)
        w = compute_sample_weight("balanced", y)
        if self.fixed_params is not None:   # reuse parameters that were already tuned on train
            self.best_params_ = dict(self.fixed_params)
            self.pipe_ = build_m3_pipeline(**{k.split("__", 1)[1]: v
                                              for k, v in self.best_params_.items()})
            self.pipe_.fit(X, y, xgb__sample_weight=w)
            return self
        cache = joblib.Memory(RESULTS / "cache", verbose=0)  # TF-IDF/SVD cached per CV fold
        search = RandomizedSearchCV(
            build_m3_pipeline(memory=cache), PARAM_SPACE, n_iter=self.n_iter,
            scoring=crisis_ap_scorer,
            cv=GroupKFold(n_splits=self.n_folds).split(X, y, groups=train.template_family),
            random_state=SEED, n_jobs=1, refit=True, error_score="raise",
        )
        search.fit(X, y, xgb__sample_weight=w)
        self.pipe_ = search.best_estimator_
        self.best_params_ = {k: (v.item() if hasattr(v, "item") else v)
                             for k, v in search.best_params_.items()}
        res = pd.DataFrame(search.cv_results_)
        self.cv_ = (res[["mean_test_score", "std_test_score", "rank_test_score"]
                        + [c for c in res.columns if c.startswith("param_")]]
                    .rename(columns={"mean_test_score": "mean_oof_crisis_pr_auc",
                                     "std_test_score": "sd"})
                    .sort_values("rank_test_score").reset_index(drop=True))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(df[["text", "channel"]])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
REGISTRY = {"M0": M0Prior, "M1": M1Rules, "M2": M2TfidfLR, "M3": M3XGB}


def load_dev() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and validation only. The test split is deliberately not returned."""
    df = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    return df[df.split == "train"].copy(), df[df.split == "val"].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=["M0", "M1", "M2", "M3"], choices=sorted(REGISTRY))
    args = ap.parse_args()

    set_seed()
    ensure_dirs()
    train, val = load_dev()
    print(f"train rows={len(train)} (crisis={int((train.risk_level == 'crisis').sum())}) | "
          f"val rows={len(val)} (crisis={int((val.risk_level == 'crisis').sum())})")

    for key in args.rungs:
        model = REGISTRY[key]().fit(train)
        if hasattr(model, "cv_"):
            print(f"\n--- {model.name}: GroupKFold({N_FOLDS}) by template_family on train ---")
            if key == "M2":
                print(model.cv_[["C", "mean_oof_crisis_pr_auc", "sd"]].round(4).to_string(index=False))
                print(f"selected C = {model.best_C_}")
            else:
                print(model.cv_[["rank_test_score", "mean_oof_crisis_pr_auc", "sd"]]
                      .head(5).round(4).to_string(index=False))
                print(f"best of {N_ITER} candidates: {model.best_params_}")
            model.cv_.to_csv(RESULTS / f"cv_{key}.csv", index=False)
        (RESULTS / "models").mkdir(exist_ok=True)
        joblib.dump(model, RESULTS / "models" / f"{key}.joblib")
        for split_name, d in [("val", val), ("train", train)]:
            rep = rung_report(model.name, d, model.predict_proba(d), split_name)
            print_rung(rep)
            if split_name == "val":
                slim = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rep.items()}
                (RESULTS / f"rung_{key}_val.json").write_text(json.dumps(slim, indent=2))


if __name__ == "__main__":
    # Re-import under the real module name so saved models unpickle from any script.
    from src.models import main as _main
    _main()
