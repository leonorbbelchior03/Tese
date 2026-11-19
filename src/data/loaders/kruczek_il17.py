"""
Loader for the Polish real-world IL-17 dataset (Kruczek et al., Mendeley Data,
doi:10.17632/t8mnz5cyzy.1, CC BY 4.0).

Assumes ONE ROW PER PATIENT with PASI/BSA/DLQI at weeks 0, 4 and 12 (wide format).
BEFORE USING: open the file, then fill in configs/datasets.yaml -> kruczek_il17
(file path, sheet, real column names and value codes).

The dataset has no calendar dates: they are reconstructed as relative dates
(week 0 = 2020-01-01), which is enough for windows and for the leakage rules.
"""
import numpy as np
import pandas as pd

from src.config import ROOT, datasets, params
from src.data.standardize import standardize

REF = pd.Timestamp("2020-01-01")
WEEKS = [0, 4, 12]
CLASS = "anti_il17"


def load() -> dict:
    cfg = datasets()["kruczek_il17"]
    f = ROOT / cfg["path"]
    if not f.exists():
        raise FileNotFoundError(
            f"File not found: {f}\nDownload the dataset from Mendeley, put it in "
            "data/raw/external/kruczek_il17/ and update configs/datasets.yaml.")
    raw = pd.read_excel(f, sheet_name=cfg["sheet"]) if f.suffix in (".xlsx", ".xls") else pd.read_csv(f)
    return from_wide(raw, cfg)


def from_wide(raw: pd.DataFrame, cfg: dict) -> dict:
    col = cfg["columns"]
    missing = [v for v in col.values() if v not in raw.columns]
    if missing:
        raise KeyError(f"Columns not found in the file: {missing}. "
                       f"Available columns: {list(raw.columns)}")
    df = raw.rename(columns={v: k for k, v in col.items()})
    df["patient_id"] = "K" + df["patient_id"].astype(str)
    age = pd.to_numeric(df["age"], errors="coerce")

    patients = pd.DataFrame({
        "patient_id": df.patient_id,
        "sex": df["sex"].map(cfg["sex_map"]),
        "birth_year": (REF.year - age).round().astype("Int64").fillna(REF.year - 45).astype(int),
        "onset_age": age - pd.to_numeric(df["disease_duration_years"], errors="coerce"),
        "family_history": pd.NA,
        "bmi": pd.to_numeric(df["bmi"], errors="coerce"),
    })

    visits = []
    for w in WEEKS:
        visits.append(pd.DataFrame({
            "patient_id": df.patient_id,
            "visit_date": REF + pd.Timedelta(weeks=w),
            "pasi": df.get(f"pasi_w{w}"), "bsa": df.get(f"bsa_w{w}"), "dlqi": df.get(f"dlqi_w{w}"),
            "difficult_sites": pd.NA,
        }))
    visits = pd.concat(visits, ignore_index=True).sort_values(["patient_id", "visit_date"])

    drug = df["drug"].astype(str).str.strip().str.lower().map(cfg["drug_map"])
    treatments = pd.DataFrame({
        "patient_id": df.patient_id, "episode_id": 1, "drug": drug, "drug_class": CLASS,
        "level": params()["levels"][CLASS], "start_date": REF,
        "end_date": pd.NaT, "stop_reason": None,
    })

    psa = df["psa"].astype(str).str.strip().str.lower().isin(["1", "yes", "y", "true", "tak"])
    comorbidities = pd.DataFrame({"patient_id": df.patient_id[psa], "condition": "psoriatic_arthritis",
                                  "diagnosis_date": REF})

    return standardize({"patients": patients, "visits": visits,
                        "treatments": treatments, "comorbidities": comorbidities})
