"""
Model 1 outcome: therapeutic escalation after starting topical treatment.

Event   = first start of a treatment with level >= 3 (or >= 2 if phototherapy counts)
          after T0.
Time    = days from T0 to the event, or to the last visit (censored).
Horizon = binary escalation within 6 / 12 months; missing (NaN) if censored before
          the horizon without an event (not enough follow-up to know).
"""
import numpy as np
import pandas as pd

from src.config import params


def compute_escalation(tables: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    P = params()["model1"]
    min_level = 2 if P["phototherapy_counts_as_escalation"] else 3
    t, v = tables["treatments"], tables["visits"]
    last_visit = v.groupby("patient_id").visit_date.max()
    rows = []
    for c in cohort.itertuples(index=False):
        later = t[(t.patient_id == c.patient_id) & (t.start_date > c.t0) & (t.level >= min_level)]
        if len(later):
            first = later.sort_values("start_date").iloc[0]
            rows.append({"patient_id": c.patient_id, "event": 1,
                         "time_days": (first.start_date - c.t0).days,
                         "escalation_drug": first.drug, "escalation_class": first.drug_class})
        else:
            end = last_visit.get(c.patient_id, c.t0)
            rows.append({"patient_id": c.patient_id, "event": 0,
                         "time_days": (end - c.t0).days,
                         "escalation_drug": None, "escalation_class": None})
    out = pd.DataFrame(rows)
    for h in P["horizons_days"]:
        out[f"escalation_{h}d"] = np.where(
            (out.event == 1) & (out.time_days <= h), 1.0,
            np.where(out.time_days >= h, 0.0, np.nan))
    return out
