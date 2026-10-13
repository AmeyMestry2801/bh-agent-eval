"""
Schema and value validation for the synthetic conversations file.

Row-level checks use a pydantic model; dataset-level checks use plain asserts
collected into a report. ALL DATA IS SYNTHETIC.

Run from the project root:
    python -m src.validate
"""
from __future__ import annotations

import math
import sys
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.config import DATA_RAW

RISK = Literal["no_risk", "low", "moderate", "crisis"]
CLASS_COUNTS_CLEAN = {"no_risk": 4200, "low": 1020, "moderate": 540, "crisis": 240}
EXPECTED_COLUMNS = [
    "conversation_id", "week", "channel", "language", "age_band", "text",
    "risk_level", "escalate_needed", "label_delay_weeks", "template_family",
    "risk_level_clean", "gen_asr_wer",
]


class Conversation(BaseModel):
    """One row of data/raw/conversations.csv."""
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(pattern=r"^C\d{5}$")
    week: int = Field(ge=1, le=26)
    channel: Literal["voice", "text"]
    language: Literal["en", "en_es"]
    age_band: Literal["18-24", "25-34", "35-49", "50-64", "65+"]
    text: str = Field(min_length=5, max_length=2000)
    risk_level: RISK
    escalate_needed: Literal[0, 1]
    label_delay_weeks: int = Field(ge=1, le=4)
    template_family: str = Field(pattern=r"^[BD]-(no_risk|low|moderate|crisis)-\d{3}$")
    risk_level_clean: RISK
    gen_asr_wer: Optional[float] = Field(default=None, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def _cross_field_rules(self) -> "Conversation":
        if self.escalate_needed != int(self.risk_level == "crisis"):
            raise ValueError("escalate_needed must equal (risk_level == 'crisis')")
        if self.channel == "voice" and self.gen_asr_wer is None:
            raise ValueError("voice rows need gen_asr_wer")
        if self.channel == "text" and self.gen_asr_wer is not None:
            raise ValueError("text rows must not have gen_asr_wer")
        if self.template_family.split("-")[1] != self.risk_level_clean:
            raise ValueError("template_family class must match risk_level_clean")
        if self.template_family.startswith("D-") and self.week < 17:
            raise ValueError("drift families (D-*) may only appear in weeks 17-26")
        return self


def _clean(rec: dict) -> dict:
    """pandas NaN -> None so pydantic sees a missing value."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}


def validate_rows(df: pd.DataFrame, max_errors: int = 10) -> list[str]:
    errors: list[str] = []
    for i, rec in enumerate(df.to_dict("records")):
        try:
            Conversation.model_validate(_clean(rec))
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in e.errors())
            errors.append(f"row {i} ({rec.get('conversation_id')}): {msg}")
            if len(errors) >= max_errors:
                break
    return errors


def validate_dataset(df: pd.DataFrame, expect_full_size: bool = True) -> list[str]:
    """Dataset-level checks. Returns a list of failure messages (empty = pass)."""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    check(list(df.columns) == EXPECTED_COLUMNS, f"columns differ from spec: {list(df.columns)}")
    check(df.conversation_id.is_unique, "conversation_id is not unique")
    check(not df.drop(columns=["gen_asr_wer"]).isna().any().any(), "unexpected missing values")
    check(set(df.week.unique()) == set(range(1, 27)), "weeks 1-26 are not all present")
    check((df.text.str.strip().str.len() > 0).all(), "empty text found")

    if expect_full_size:
        check(len(df) == 6000, f"expected 6000 rows, found {len(df)}")
        clean_counts = df.risk_level_clean.value_counts().to_dict()
        check(clean_counts == CLASS_COUNTS_CLEAN,
              f"clean class counts {clean_counts} != {CLASS_COUNTS_CLEAN}")

    noise = float((df.risk_level != df.risk_level_clean).mean())
    check(0.01 <= noise <= 0.05, f"label noise rate {noise:.3f} outside [0.01, 0.05]")

    pre, post = df[df.week <= 16], df[df.week >= 17]
    check(post.channel.eq("voice").mean() > pre.channel.eq("voice").mean(),
          "voice share should be higher in weeks 17-26")
    check(post.gen_asr_wer.mean() > pre.gen_asr_wer.mean(),
          "mean ASR WER should be higher in weeks 17-26")
    check(0.05 <= df.language.eq("en_es").mean() <= 0.15, "Spanish-mixed share outside [5%, 15%]")
    return fails


def validate_frame(df: pd.DataFrame, expect_full_size: bool = True) -> dict:
    """Run every check. Raises ValueError on any failure; returns a small report."""
    problems = validate_rows(df) + validate_dataset(df, expect_full_size)
    if problems:
        raise ValueError("validation failed:\n  " + "\n  ".join(problems))
    return {
        "rows": len(df),
        "weeks": f"{df.week.min()}-{df.week.max()}",
        "families": int(df.template_family.nunique()),
        "label_noise_rate": round(float((df.risk_level != df.risk_level_clean).mean()), 4),
        "status": "PASS",
    }


def main() -> None:
    path = DATA_RAW / "conversations.csv"
    df = pd.read_csv(path)
    try:
        report = validate_frame(df)
    except ValueError as e:
        print(e)
        sys.exit(1)
    for k, v in report.items():
        print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()
