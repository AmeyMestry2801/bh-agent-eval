"""Fill hand-assigned failure tags into results/fn_cases.csv. ALL DATA IS SYNTHETIC."""
import pandas as pd
from src.config import RESULTS

TAGS = [  # (primary, secondary, note) in the printed order of cases 1..7
    ("negation", "", "recorded crisis, clean moderate"),
    ("negation", "", "recorded crisis, clean moderate"),
    ("negation", "language_mix", "recorded crisis, clean moderate"),
    ("negation", "near_threshold", "P=0.0926 vs threshold 0.0927"),
    ("other", "indirect_phrasing", "recorded crisis, clean moderate"),
    ("negation", "voice_low_wer", "WER 0.09, text clear; not an ASR failure"),
    ("hedging", "language_mix", "recorded crisis, clean moderate"),
]

path = RESULTS / "fn_cases.csv"
df = pd.read_csv(path)
assert len(df) == len(TAGS), f"expected {len(TAGS)} rows, got {len(df)}"
df["tag_primary"], df["tag_secondary"], df["note"] = zip(*TAGS)
df.to_csv(path, index=False)
print(df.tag_primary.value_counts().to_string())
print(df[["tag_primary", "tag_secondary", "note"]].to_string())
