"""Loader for the fictitious test data (data/synthetic/fictitious/*.csv)."""
import pandas as pd

from src.config import ROOT, datasets
from src.data.standardize import standardize

TABLES = ["patients", "visits", "treatments", "comorbidities"]


def load() -> dict:
    folder = ROOT / datasets()["fictitious"]["path"]
    if not (folder / "patients.csv").exists():
        from src.synthetic.fictitious import make_fictitious, save_fictitious
        save_fictitious(make_fictitious())
    return standardize({t: pd.read_csv(folder / f"{t}.csv") for t in TABLES})
