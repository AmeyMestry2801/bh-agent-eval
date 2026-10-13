import pandas as pd

from src.generate_data import generate
from src.validate import CLASS_COUNTS_CLEAN, EXPECTED_COLUMNS


def test_columns_and_size(full):
    assert list(full.columns) == EXPECTED_COLUMNS
    assert len(full) == 6000


def test_same_seed_reproduces(full):
    pd.testing.assert_frame_equal(full, generate(seed=42, n=6000))


def test_different_seed_differs():
    a, b = generate(seed=42, n=600), generate(seed=7, n=600)
    assert not a.text.equals(b.text)


def test_clean_class_counts_exact(full):
    assert full.risk_level_clean.value_counts().to_dict() == CLASS_COUNTS_CLEAN


def test_label_noise_rate_in_range(full):
    noise = (full.risk_level != full.risk_level_clean).mean()
    assert 0.01 <= noise <= 0.05


def test_drift_regime_present(full):
    pre, post = full[full.week <= 16], full[full.week >= 17]
    assert post.channel.eq("voice").mean() > pre.channel.eq("voice").mean()
    assert post.gen_asr_wer.mean() > pre.gen_asr_wer.mean()
    assert not pre.template_family.str.startswith("D-").any()
    assert post.template_family.str.startswith("D-").any()


def test_escalate_matches_recorded_label(full):
    assert (full.escalate_needed == (full.risk_level == "crisis").astype(int)).all()


def test_label_delay_range(full):
    assert full.label_delay_weeks.between(1, 4).all()
