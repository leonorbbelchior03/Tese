"""
Quality control (QC) of the common tables.

run_qc() never raises an error: it returns a report with every problem found,
so that each problem can be inspected and decided on (correct, set to missing, exclude).
"""
import pandas as pd

from src.data.schemas import SCHEMAS
from pandera.errors import SchemaErrors



def schema_issues(tables: dict) -> pd.DataFrame:
    """Values out of range, wrong categories, wrong types, duplicated ids."""
    rows = []
    for name, schema in SCHEMAS.items():
        if name not in tables:
            continue
        try:
            schema.validate(tables[name], lazy=True)
        except SchemaErrors as e:
            fc = e.failure_cases.copy()
            fc["table"] = name
            rows.append(fc[["table", "column", "check", "failure_case", "index"]])
    if not rows:
        return pd.DataFrame(columns=["table", "column", "check", "failure_case", "index"])
    return pd.concat(rows, ignore_index=True)


def coherence_issues(tables: dict) -> pd.DataFrame:
    """Clinical / logical inconsistencies that a schema cannot see."""
    issues = []
    v, t, p = tables["visits"], tables["treatments"], tables["patients"]

    def add(table, idx, problem):
        for i in idx:
            issues.append({"table": table, "index": i, "problem": problem})

    # Duplicate visits (same patient, same date)
    dup = v[v.duplicated(["patient_id", "visit_date"], keep="first")].index
    add("visits", dup, "duplicate visit (same patient and date)")

    # PASI = 0 but BSA > 0 (or the opposite)
    add("visits", v[(v.pasi == 0) & (v.bsa > 0)].index, "PASI = 0 but BSA > 0")
    add("visits", v[(v.pasi > 0) & (v.bsa == 0)].index, "PASI > 0 but BSA = 0")

    # Treatment ends before it starts
    add("treatments", t[t.end_date < t.start_date].index, "end_date before start_date")

    # Stop reason without end date
    add("treatments", t[t.stop_reason.notna() & t.end_date.isna()].index, "stop_reason without end_date")

    # Visits / treatments of patients not in the patients table
    known = set(p.patient_id)
    add("visits", v[~v.patient_id.isin(known)].index, "patient not in patients table")
    add("treatments", t[~t.patient_id.isin(known)].index, "patient not in patients table")

    # Visit before the patient was born
    vy = v.merge(p[["patient_id", "birth_year"]], on="patient_id", how="left")
    add("visits", v.index[(vy.visit_date.dt.year < vy.birth_year).values], "visit before birth year")

    # Disease onset after the age at the first visit
    first = v.groupby("patient_id").visit_date.min().dt.year
    age_first = p.patient_id.map(first) - p.birth_year
    add("patients", p[p.onset_age > age_first].index, "onset_age after age at first visit")

    # Patients without any visit
    no_visit = p[~p.patient_id.isin(v.patient_id)].index
    add("patients", no_visit, "patient without visits")

    return pd.DataFrame(issues, columns=["table", "index", "problem"])


def missingness(tables: dict) -> pd.DataFrame:
    """% missing per table and column."""
    rows = []
    for name, df in tables.items():
        for c in df.columns:
            rows.append({"table": name, "column": c, "n": len(df),
                         "n_missing": int(df[c].isna().sum()),
                         "pct_missing": round(100 * df[c].isna().mean(), 1)})
    return pd.DataFrame(rows)


def summary(tables: dict) -> pd.Series:
    v, t = tables["visits"], tables["treatments"]
    vpp = v.groupby("patient_id").size()
    return pd.Series({
        "patients": len(tables["patients"]),
        "visits": len(v),
        "treatment episodes": len(t),
        "visits per patient (median)": float(vpp.median()) if len(vpp) else 0,
        "patients with >= 3 visits": int((vpp >= 3).sum()),
        "first visit": v.visit_date.min(),
        "last visit": v.visit_date.max(),
    })


def run_qc(tables: dict) -> dict:
    return {
        "summary": summary(tables),
        "schema_issues": schema_issues(tables),
        "coherence_issues": coherence_issues(tables),
        "missingness": missingness(tables),
    }
