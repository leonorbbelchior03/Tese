"""Force the common tables into the dtypes expected by the schemas."""
import pandas as pd

DATE_COLS = {
    "visits": ["visit_date"],
    "treatments": ["start_date", "end_date"],
    "comorbidities": ["diagnosis_date"],
}
FLOAT_COLS = {
    "patients": ["onset_age", "bmi"],
    "visits": ["pasi", "bsa", "dlqi"],
}
BOOL_COLS = {
    "patients": ["family_history"],
    "visits": ["difficult_sites"],
}


def standardize(tables: dict) -> dict:
    out = {}
    for name, df in tables.items():
        df = df.copy()
        if "patient_id" in df:
            df["patient_id"] = df["patient_id"].astype(str)
        for c in DATE_COLS.get(name, []):
            df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
        for c in FLOAT_COLS.get(name, []):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in BOOL_COLS.get(name, []):
            df[c] = df[c].astype("boolean")
        if name == "patients":
            df["birth_year"] = df["birth_year"].astype(int)
        if name == "treatments":
            df["episode_id"] = df["episode_id"].astype(int)
            df["level"] = df["level"].astype(int)
            df["stop_reason"] = df["stop_reason"].astype(object).where(df["stop_reason"].notna(), None)
        out[name] = df
    return out
