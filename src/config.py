"""Global configuration: one seed, one set of paths. Everything imports from here."""
import os
import random
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
BI = ROOT / "bi"
DOCS = ROOT / "docs"

# Cost matrix assumption (see docs/problem_framing.md)
COST_FN_CRISIS = 20.0  # missed crisis
COST_FP_CRISIS = 1.0   # false alarm


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we use so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, BI, DOCS):
        p.mkdir(parents=True, exist_ok=True)
