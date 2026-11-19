"""
Predictors at T0. RULE: only information dated ON OR BEFORE T0.
(tested in tests/test_no_leakage.py)
"""
import numpy as np
import pandas as pd

from src.config import params


def build_features(tables: dict, cohort: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    p = tables["patients"].set_index("patient_id")
    v = tables["visits"].sort_values("visit_date")
    t = tables["treatments"]
    cm = tables["comorbidities"]
    rows = []
    for c in cohort.itertuples(index=False):
        pid, t0 = c.patient_id, c.t0
        pat = p.loc[pid]
        age = t0.year - pat.birth_year
        r = {"patient_id": pid, "episode_id": c.episode_id,
             "age": age, "sex_male": float(pat.sex == "M"), "bmi": pat.bmi,
             "disease_duration": age - pat.onset_age if pd.notna(pat.onset_age) else np.nan,
             "family_history": float(pat.family_history) if pd.notna(pat.family_history) else np.nan}

        # Baseline visit: last visit within [T0 - lookback, T0]
        pv = v[(v.patient_id == pid) & (v.visit_date <= t0) &
               (v.visit_date >= t0 - pd.Timedelta(days=lookback_days))]
        b = pv.iloc[-1] if len(pv) else None
        for k in ["pasi", "bsa", "dlqi"]:
            r[f"{k}_baseline"] = b[k] if b is not None else np.nan
        r["difficult_sites"] = float(b.difficult_sites) if b is not None and pd.notna(b.difficult_sites) else np.nan

        # Comorbidities diagnosed up to T0
        pc = cm[(cm.patient_id == pid) & (cm.diagnosis_date <= t0)]
        for cond in ["psoriatic_arthritis", "obesity", "diabetes", "cardiovascular"]:
            r[cond] = float((pc.condition == cond).any())

        # Treatment history: episodes STARTED BEFORE T0
        prev = t[(t.patient_id == pid) & (t.start_date < t0)]
        r["n_prev_systemic"] = int((prev.level == 3).sum())
        r["n_prev_biologic"] = int((prev.level == 4).sum())
        r["prev_failure_inefficacy"] = float(((prev.stop_reason == "inefficacy") & (prev.end_date <= t0)).any())

        if hasattr(c, "drug_class"):
            r["drug_class"] = c.drug_class
        rows.append(r)
    return pd.DataFrame(rows)


def model1_features(tables, cohort):
    return build_features(tables, cohort, params()["model1"]["baseline_lookback_days"])


def model2_features(tables, cohort):
    return build_features(tables, cohort, params()["model2"]["baseline_lookback_days"])
