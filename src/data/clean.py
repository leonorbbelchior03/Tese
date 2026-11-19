"""
Simple, documented cleaning rules applied after the QC.
Each rule is logged so it can be reported in the thesis.
"""
import pandas as pd

from src.config import params


def clean(tables: dict) -> tuple[dict, pd.DataFrame]:
    R = params()["ranges"]
    log = []
    t = {k: df.copy() for k, df in tables.items()}

    # 1. Out-of-range scores -> missing (not deleted: the rest of the visit is kept)
    for col in ["pasi", "bsa", "dlqi"]:
        lo, hi = R[col]
        bad = t["visits"][col].notna() & ~t["visits"][col].between(lo, hi)
        if bad.any():
            log.append({"rule": f"{col} out of [{lo}, {hi}] -> NaN", "n_rows": int(bad.sum())})
            t["visits"].loc[bad, col] = float("nan")

    # 2. Duplicate visits -> keep the first
    dup = t["visits"].duplicated(["patient_id", "visit_date"], keep="first")
    if dup.any():
        log.append({"rule": "duplicate visits removed (kept first)", "n_rows": int(dup.sum())})
        t["visits"] = t["visits"][~dup]

    # 3. end_date before start_date -> end_date unknown
    bad = t["treatments"].end_date < t["treatments"].start_date
    if bad.any():
        log.append({"rule": "end_date < start_date -> end_date = NaT", "n_rows": int(bad.sum())})
        t["treatments"].loc[bad, "end_date"] = pd.NaT

    for k in t:
        t[k] = t[k].reset_index(drop=True)
    return t, pd.DataFrame(log, columns=["rule", "n_rows"])
