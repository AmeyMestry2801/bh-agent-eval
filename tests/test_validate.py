import numpy as np
import pytest

from src.validate import validate_dataset, validate_frame, validate_rows


def _msgs(errors) -> str:
    return " | ".join(errors)


def test_full_dataset_passes(full):
    assert validate_frame(full)["status"] == "PASS"


def test_clean_rows_have_no_row_errors(full):
    assert validate_rows(full.head(200)) == []


def test_rejects_escalate_mismatch(full):
    r = full.head(1).copy()
    r["escalate_needed"] = 1 - r["escalate_needed"]
    assert "escalate_needed" in _msgs(validate_rows(r))


def test_rejects_voice_row_without_wer(full):
    r = full[full.channel == "voice"].head(1).copy()
    r["gen_asr_wer"] = np.nan
    assert "voice rows need gen_asr_wer" in _msgs(validate_rows(r))


def test_rejects_text_row_with_wer(full):
    r = full[full.channel == "text"].head(1).copy()
    r["gen_asr_wer"] = 0.1
    assert "text rows must not have gen_asr_wer" in _msgs(validate_rows(r))


def test_rejects_drift_family_before_week_17(full):
    r = full[full.template_family.str.startswith("D-")].head(1).copy()
    r["week"] = 5
    assert "drift families" in _msgs(validate_rows(r))


def test_rejects_family_class_mismatch(full):
    r = full.head(1).copy()
    r["risk_level_clean"] = "low" if r.iloc[0].risk_level_clean == "crisis" else "crisis"
    assert "template_family class must match" in _msgs(validate_rows(r))


def test_rejects_bad_category_extra_column_bad_id_bad_week(full):
    r = full.head(1).copy()
    r["channel"] = "chat"
    assert "channel" in _msgs(validate_rows(r))
    r = full.head(1).copy()
    r["unexpected"] = 1
    assert "unexpected" in _msgs(validate_rows(r))
    r = full.head(1).copy()
    r["conversation_id"] = "X1"
    assert "conversation_id" in _msgs(validate_rows(r))
    r = full.head(1).copy()
    r["week"] = 27
    assert "week" in _msgs(validate_rows(r))


def test_dataset_detects_duplicate_ids(full):
    d = full.copy()
    d.loc[1, "conversation_id"] = d.loc[0, "conversation_id"]
    assert "not unique" in _msgs(validate_dataset(d))


def test_dataset_detects_missing_week(full):
    assert "weeks 1-26" in _msgs(validate_dataset(full[full.week != 5]))


def test_dataset_detects_zero_label_noise(full):
    d = full.copy()
    d["risk_level"] = d["risk_level_clean"]
    assert "label noise" in _msgs(validate_dataset(d))


def test_dataset_detects_wrong_columns(full):
    assert "columns differ" in _msgs(validate_dataset(full.drop(columns=["age_band"])))


def test_validate_frame_raises_on_bad_data(full):
    d = full.copy()
    d.loc[1, "conversation_id"] = d.loc[0, "conversation_id"]
    with pytest.raises(ValueError, match="validation failed"):
        validate_frame(d)
