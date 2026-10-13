"""
Train / validation / test / monitoring split, grouped by template family.

    train    = weeks 1-12, families NOT in the test set
    val      = weeks 13-16, families NOT in the test set
    test     = weeks 1-16, held-out families  (touched ONCE, at the very end)
    monitor  = weeks 17-26, all families      (the post-deployment stream)

ALL DATA IS SYNTHETIC. Run from the project root:
    python -m src.split
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_PROCESSED, DATA_RAW, RESULTS, SEED, ensure_dirs, set_seed

CLASSES = ["no_risk", "low", "moderate", "crisis"]
TEST_FRAC = 0.20   # target share of weeks 1-16 rows (per class) held out as test families
TRAIN_LAST_WEEK = 12
REF_LAST_WEEK = 16

# Columns that must NEVER be used as model inputs (labels, generator internals, ids, time).
NON_FEATURE_COLUMNS = [
    "conversation_id", "week", "risk_level", "escalate_needed", "label_delay_weeks",
    "template_family", "risk_level_clean", "gen_asr_wer", "split",
]


def choose_test_families(ref: pd.DataFrame, test_frac: float, rng: np.random.Generator) -> set[str]:
    """Pick whole families until ~test_frac of each (clean) class's rows are held out.

    Done per class so that rare classes (crisis) get a test share close to the target."""
    chosen: set[str] = set()
    for cls in CLASSES:
        counts = ref[ref.risk_level_clean == cls].groupby("template_family").size()
        order = rng.permutation(counts.index.to_numpy())
        target, acc = test_frac * counts.sum(), 0
        for fam in order:
            if acc >= target:
                break
            chosen.add(str(fam))
            acc += int(counts[fam])
    return chosen


def make_splits(df: pd.DataFrame, seed: int = SEED, test_frac: float = TEST_FRAC) -> pd.Series:
    rng = np.random.default_rng(seed)
    ref = df[df.week <= REF_LAST_WEEK]
    test_fams = choose_test_families(ref, test_frac, rng)

    split = pd.Series("monitor", index=df.index, dtype="object")
    in_ref = df.week <= REF_LAST_WEEK
    is_test = in_ref & df.template_family.isin(test_fams)
    split[in_ref & ~is_test & (df.week <= TRAIN_LAST_WEEK)] = "train"
    split[in_ref & ~is_test & (df.week > TRAIN_LAST_WEEK)] = "val"
    split[is_test] = "test"
    return split


def check_split_integrity(df: pd.DataFrame) -> dict:
    """Raise AssertionError if any split rule is broken. Returns exact-text overlap counts."""
    s = df.split
    assert set(s.unique()) == {"train", "val", "test", "monitor"}, "missing split"
    assert len(df) == (s != "").sum(), "unassigned rows"
    assert df[s == "train"].week.between(1, TRAIN_LAST_WEEK).all(), "train weeks out of range"
    assert df[s == "val"].week.between(TRAIN_LAST_WEEK + 1, REF_LAST_WEEK).all(), "val weeks out of range"
    assert df[s == "test"].week.between(1, REF_LAST_WEEK).all(), "test weeks out of range"
    assert (df[s == "monitor"].week > REF_LAST_WEEK).all(), "monitor weeks out of range"

    test_fams = set(df[s == "test"].template_family)
    dev_fams = set(df[s.isin(["train", "val"])].template_family)
    straddle = test_fams & dev_fams
    assert not straddle, f"{len(straddle)} families straddle test and train/val: {sorted(straddle)[:5]}"
    assert not any(f.startswith("D-") for f in test_fams), "drift families must not be in test"

    train_text = set(df[s == "train"].text)
    return {
        "test_texts_also_in_train": int(df[s == "test"].text.isin(train_text).sum()),
        "val_texts_also_in_train": int(df[s == "val"].text.isin(train_text).sum()),
        "families_shared_train_val": len(set(df[s == "train"].template_family)
                                         & set(df[s == "val"].template_family)),
    }


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["train", "val", "test", "monitor"]:
        d = df[df.split == name]
        rows.append({
            "split": name,
            "weeks": f"{d.week.min()}-{d.week.max()}",
            "rows": len(d),
            "families": d.template_family.nunique(),
            "crisis": int((d.risk_level == "crisis").sum()),
            "moderate": int((d.risk_level == "moderate").sum()),
            "crisis_pct": round(100 * (d.risk_level == "crisis").mean(), 2),
            "voice_pct": round(100 * (d.channel == "voice").mean(), 1),
            "spanish_pct": round(100 * (d.language == "en_es").mean(), 1),
        })
    return pd.DataFrame(rows).set_index("split")


def _max_sim_to_pool(X_query, X_pool, chunk: int = 500) -> np.ndarray:
    out = np.zeros(X_query.shape[0])
    for a in range(0, X_query.shape[0], chunk):
        b = min(a + chunk, X_query.shape[0])
        out[a:b] = (X_query[a:b] @ X_pool.T).toarray().max(axis=1)
    return out


def leakage_demo(df: pd.DataFrame, seed: int = SEED, thr: float = 0.8) -> pd.DataFrame:
    """How similar is each held-out row to its closest training row?
    Compare our family split with a naive random row split of the same size."""
    rng = np.random.default_rng(seed)
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(df.text)
    train = df[df.split == "train"]
    fam_test = df[df.split == "test"]

    # naive: pick the same number of random test rows from weeks 1-16, train on the rest of weeks 1-12
    ref = df[df.week <= REF_LAST_WEEK]
    naive_idx = rng.choice(ref.index.to_numpy(), size=len(fam_test), replace=False)
    naive_test = df.loc[naive_idx]
    naive_train = ref[(ref.week <= TRAIN_LAST_WEEK) & ~ref.index.isin(naive_idx)]

    val = df[df.split == "val"]
    rows = []
    for name, q, pool in [("test vs train (family split, ours)", fam_test, train),
                          ("val vs train (shared families, by design)", val, train),
                          ("random row split (naive)", naive_test, naive_train)]:
        sim = _max_sim_to_pool(vec.transform(q.text), vec.transform(pool.text))
        rows.append({
            "scheme": name,
            "test_rows": len(q),
            "mean_max_cosine": round(float(sim.mean()), 3),
            f"share_with_train_twin_>={thr}": round(float((sim >= thr).mean()), 3),
        })
    return pd.DataFrame(rows).set_index("scheme")


def main() -> None:
    set_seed()
    ensure_dirs()
    df = pd.read_csv(DATA_RAW / "conversations.csv")
    df["split"] = make_splits(df)
    overlap = check_split_integrity(df)

    out = DATA_PROCESSED / "conversations_split.csv"
    df.to_csv(out, index=False)

    summ = split_summary(df)
    summ.to_csv(RESULTS / "split_summary.csv")
    print(f"wrote {out}\n")
    print("split integrity checks: PASS\n")
    print(summ.to_string(), "\n")
    print("exact-text / family overlap:", overlap, "\n")
    print("leakage demo (how close is each held-out row to its nearest training row?):")
    print(leakage_demo(df).to_string())


if __name__ == "__main__":
    main()
