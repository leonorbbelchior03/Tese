"""
Model 2 outcomes: response to a systemic / targeted treatment.

For each episode:
  - baseline visit = last visit in [T0 - lookback, T0]
  - follow-up visit = visit closest to T0 + window, within +/- tolerance
    (window by drug from configs/treatment_windows.yaml, else class default)
  - PASI75 / PASI90: relative PASI reduction >= 75% / 90%
  - DLQI improvement >= MCID (only if baseline DLQI >= MCID)
If the treatment was stopped for inefficacy before the window, the episode counts
as a failure (0). If stopped for another reason before the window -> missing outcome.
"""
import numpy as np
import pandas as pd

from src.config import params, windows


def window_weeks(drug: str, drug_class: str, source: str = "primary") -> float:
    w = windows()
    if drug in w["drugs"]:
        return w["drugs"][drug][source]
    if drug_class in w["class_defaults"]:
        return w["class_defaults"][drug_class][source]
    return np.nan


def _baseline_visit(v_pat, t0, lookback_days):
    m = v_pat[(v_pat.visit_date <= t0) & (v_pat.visit_date >= t0 - pd.Timedelta(days=lookback_days))]
    return m.iloc[-1] if len(m) else None


def _followup_visit(v_pat, target, tol_days):
    m = v_pat[(v_pat.visit_date - target).abs() <= pd.Timedelta(days=tol_days)]
    if not len(m):
        return None
    return m.loc[(m.visit_date - target).abs().idxmin()]


def compute_response(tables: dict, cohort: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    P = params()["model2"]
    source = source or P["window_source"]
    tol_days = P["window_tolerance_weeks"] * 7
    mcid = P["dlqi_mcid"]
    visits = tables["visits"].sort_values("visit_date")
    by_pat = dict(tuple(visits.groupby("patient_id")))
    rows = []

    for ep in cohort.itertuples(index=False):
        v = by_pat.get(ep.patient_id, visits.iloc[0:0])
        wk = window_weeks(ep.drug, ep.drug_class, source)
        target = ep.t0 + pd.Timedelta(weeks=wk)
        base = _baseline_visit(v, ep.t0, P["baseline_lookback_days"])
        row = {"patient_id": ep.patient_id, "episode_id": ep.episode_id, "window_weeks": wk,
               "pasi_base": np.nan, "pasi_fu": np.nan, "dlqi_base": np.nan, "dlqi_fu": np.nan,
               "fu_date": pd.NaT, "status": None}

        stopped_early = pd.notna(ep.end_date) and ep.end_date < target - pd.Timedelta(days=tol_days)
        if base is not None:
            row.update(pasi_base=base.pasi, dlqi_base=base.dlqi)

        if stopped_early:
            if ep.stop_reason == "inefficacy" and P["stop_for_inefficacy_is_failure"]:
                row["status"] = "failure_stopped_inefficacy"
            else:
                row["status"] = "stopped_other_reason"
        elif base is None or pd.isna(base.pasi) or base.pasi == 0:
            row["status"] = "no_baseline_pasi"
        else:
            # only visits while still on this treatment
            on_tx = v if pd.isna(ep.end_date) else v[v.visit_date <= ep.end_date]
            fu = _followup_visit(on_tx[on_tx.visit_date > ep.t0], target, tol_days)
            if fu is None or pd.isna(fu.pasi):
                row["status"] = "no_followup_in_window"
            else:
                row.update(pasi_fu=fu.pasi, dlqi_fu=fu.dlqi, fu_date=fu.visit_date, status="assessed")
        rows.append(row)

    out = pd.DataFrame(rows)
    red = (out.pasi_base - out.pasi_fu) / out.pasi_base
    out["pasi_reduction"] = red
    assessed = out.status == "assessed"
    failed = out.status == "failure_stopped_inefficacy"
    for k, thr in [("pasi75", 0.75), ("pasi90", 0.90)]:
        out[k] = np.where(assessed, (red >= thr).astype(float), np.where(failed, 0.0, np.nan))
    dlqi_ok = assessed & out.dlqi_base.ge(mcid) & out.dlqi_fu.notna()
    out["dlqi_improved"] = np.where(dlqi_ok, ((out.dlqi_base - out.dlqi_fu) >= mcid).astype(float),
                                    np.where(failed & out.dlqi_base.ge(mcid), 0.0, np.nan))
    return out
