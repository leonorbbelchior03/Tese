"""
Study populations and reference date T0 for each model.

Model 1: patients whose FIRST recorded treatment is topical. T0 = start of that episode.
Model 2: every start of a systemic / targeted treatment (level >= min_level).
         T0 = start_date of the episode. One patient can contribute several episodes
         (so CV must be grouped by patient).
"""
import pandas as pd

from src.config import params


def model1_cohort(tables: dict) -> pd.DataFrame:
    t = tables["treatments"].sort_values(["patient_id", "start_date"])
    first = t.groupby("patient_id").head(1)
    cohort = first[first.level == 1][["patient_id", "episode_id", "start_date"]]
    return cohort.rename(columns={"start_date": "t0"}).reset_index(drop=True)


def model2_cohort(tables: dict) -> pd.DataFrame:
    min_level = params()["model2"]["min_level"]
    t = tables["treatments"]
    c = t[t.level >= min_level].copy()
    c["t0"] = c["start_date"]
    return c[["patient_id", "episode_id", "drug", "drug_class", "level", "t0",
              "end_date", "stop_reason"]].reset_index(drop=True)
