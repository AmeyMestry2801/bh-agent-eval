import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def full():
    """The full 6000-row synthetic dataset, generated once per test session. Do not mutate."""
    return generate(seed=42, n=6000)
