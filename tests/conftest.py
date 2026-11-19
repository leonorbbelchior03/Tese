import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import clean  # noqa: E402
from src.data.standardize import standardize  # noqa: E402
from src.synthetic.fictitious import make_fictitious  # noqa: E402


@pytest.fixture(scope="session")
def tables():
    t, _ = clean(standardize(make_fictitious(n_patients=60, seed=1)))
    return t
