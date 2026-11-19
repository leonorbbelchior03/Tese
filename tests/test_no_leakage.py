"""
Predictors must use ONLY information up to T0.
Test: change everything that happens AFTER T0 and confirm the features do not change.
"""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.data.cohorts import model1_cohort, model2_cohort
from src.features.baseline import model1_features, model2_features


def _scramble_future(tables, cohort):
    t = {k: df.copy() for k, df in tables.items()}
    first_t0 = cohort.groupby("patient_id").t0.min()
    v = t["visits"]
    after = v.visit_date > v.patient_id.map(first_t0)
    rng = np.random.default_rng(0)
    v.loc[after, "pasi"] = rng.uniform(0, 72, after.sum())
    v.loc[after, "dlqi"] = rng.integers(0, 31, after.sum()).astype(float)
    # add a comorbidity diagnosed in the future
    fut = pd.DataFrame({"patient_id": first_t0.index, "condition": "diabetes",
                        "diagnosis_date": first_t0.values + pd.Timedelta(days=30)})
    t["comorbidities"] = pd.concat([t["comorbidities"], fut], ignore_index=True)
    return t


def test_model2_features_ignore_future(tables):
    c = model2_cohort(tables)
    c_first = c.loc[c.groupby("patient_id").t0.idxmin()]   # first episode per patient
    f1 = model2_features(tables, c_first)
    f2 = model2_features(_scramble_future(tables, c_first), c_first)
    pdt.assert_frame_equal(f1, f2)


def test_model1_features_ignore_future(tables):
    c = model1_cohort(tables)
    f1 = model1_features(tables, c)
    f2 = model1_features(_scramble_future(tables, c), c)
    pdt.assert_frame_equal(f1, f2)
