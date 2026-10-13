import pytest

from src.split import check_split_integrity, make_splits


@pytest.fixture(scope="module")
def split_df(full):
    d = full.copy()
    d["split"] = make_splits(d)
    return d


def test_integrity_passes(split_df):
    out = check_split_integrity(split_df)
    assert set(out) == {"test_texts_also_in_train", "val_texts_also_in_train", "families_shared_train_val"}


def test_week_boundaries(split_df):
    s = split_df
    assert s[s.split == "train"].week.max() <= 12
    assert s[s.split == "val"].week.between(13, 16).all()
    assert s[s.split == "test"].week.max() <= 16
    assert (s[s.split == "monitor"].week >= 17).all()
    assert (s[s.week >= 17].split == "monitor").all()


def test_test_families_disjoint_from_dev(split_df):
    s = split_df
    test_f = set(s[s.split == "test"].template_family)
    dev_f = set(s[s.split.isin(["train", "val"])].template_family)
    assert test_f and not (test_f & dev_f)


def test_no_drift_family_in_test(split_df):
    assert not split_df[split_df.split == "test"].template_family.str.startswith("D-").any()


def test_split_is_deterministic(full):
    assert make_splits(full).equals(make_splits(full))


def test_straddling_family_is_detected(split_df):
    d = split_df.copy()
    train = d[d.split == "train"]
    fam = train.template_family.value_counts().index[0]
    d.loc[train[train.template_family == fam].index[0], "split"] = "test"
    with pytest.raises(AssertionError, match="straddle"):
        check_split_integrity(d)
