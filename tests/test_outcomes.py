"""Outcomes computed by hand on small, known cases."""
import pandas as pd

from src.data.cohorts import model2_cohort
from src.data.standardize import standardize
from src.outcomes.response import compute_response, window_weeks


def _case(pasi_fu, fu_week, end_week=None, reason=None, drug="secukinumab", cls="anti_il17"):
    t0 = pd.Timestamp("2022-01-01")
    tables = standardize({
        "patients": pd.DataFrame([{"patient_id": "P1", "sex": "F", "birth_year": 1980,
                                   "onset_age": 20.0, "family_history": False, "bmi": 25.0}]),
        "visits": pd.DataFrame([
            {"patient_id": "P1", "visit_date": t0, "pasi": 20.0, "bsa": 25.0, "dlqi": 15.0, "difficult_sites": False},
            {"patient_id": "P1", "visit_date": t0 + pd.Timedelta(weeks=fu_week), "pasi": pasi_fu,
             "bsa": 2.0, "dlqi": 3.0, "difficult_sites": False}]),
        "treatments": pd.DataFrame([{"patient_id": "P1", "episode_id": 1, "drug": drug, "drug_class": cls,
                                     "level": 4, "start_date": t0,
                                     "end_date": t0 + pd.Timedelta(weeks=end_week) if end_week else pd.NaT,
                                     "stop_reason": reason}]),
        "comorbidities": pd.DataFrame(columns=["patient_id", "condition", "diagnosis_date"]),
    })
    return compute_response(tables, model2_cohort(tables)).iloc[0]


def test_window_from_config():
    assert window_weeks("secukinumab", "anti_il17") == 16
    assert window_weeks("ustekinumab", "anti_il12_23") == 28
    assert window_weeks("unknown_drug", "anti_il23") == 16   # class default


def test_pasi90_responder():
    r = _case(pasi_fu=1.0, fu_week=16)        # 95% reduction
    assert r.pasi75 == 1 and r.pasi90 == 1 and r.dlqi_improved == 1


def test_pasi75_not_90():
    r = _case(pasi_fu=4.0, fu_week=16)        # 80% reduction
    assert r.pasi75 == 1 and r.pasi90 == 0


def test_visit_outside_window_is_missing():
    r = _case(pasi_fu=1.0, fu_week=30)        # 16 wk window, +/-4 wk tolerance
    assert r.status == "no_followup_in_window" and pd.isna(r.pasi90)


def test_stop_for_inefficacy_is_failure():
    r = _case(pasi_fu=15.0, fu_week=6, end_week=6, reason="inefficacy")
    assert r.status == "failure_stopped_inefficacy" and r.pasi75 == 0


def test_stop_for_adverse_event_is_missing():
    r = _case(pasi_fu=15.0, fu_week=6, end_week=6, reason="adverse_event")
    assert pd.isna(r.pasi75)
