"""
Small FICTITIOUS dataset in the common format, used ONLY to test the code
end to end while real data are not available.

- It is not the evidence-informed synthetic dataset (plan B); numbers are rough.
- Includes topical patients who escalate (to test Model 1, absent from the
  public dataset) and systemic/biologic starts (to test Model 2).
- Contains a few DELIBERATE errors so that the QC has something to find.
Results obtained on these data must never be interpreted.
"""
import numpy as np
import pandas as pd

from src.config import params, path

DRUGS = {  # drug -> (class, mean fraction of PASI reduction at plateau)
    "methotrexate": ("conventional_systemic", 0.65),
    "ciclosporin": ("conventional_systemic", 0.70),
    "apremilast": ("oral_small_molecule", 0.60),
    "adalimumab": ("anti_tnf", 0.82),
    "ustekinumab": ("anti_il12_23", 0.86),
    "secukinumab": ("anti_il17", 0.92),
    "bimekizumab": ("anti_il17", 0.96),
    "guselkumab": ("anti_il23", 0.93),
    "risankizumab": ("anti_il23", 0.95),
}
FOLLOW_UP_WEEKS = [4, 12, 16, 24, 28, 40, 52]


def make_fictitious(n_patients: int = 120, seed: int | None = None, inject_errors: bool = True) -> dict:
    P = params()
    rng = np.random.default_rng(P["seed"] if seed is None else seed)
    levels = P["levels"]
    patients, visits, treatments, comorb = [], [], [], []

    def add_visit(pid, date, pasi, difficult):
        pasi = round(float(np.clip(pasi, 0, 72)), 1)
        bsa = float(np.clip(pasi * 1.3 + rng.normal(0, 3), 0.5, 100)) if pasi > 0 else 0.0
        dlqi = float(np.clip(round(0.6 * pasi + rng.normal(2, 3)), 0, 30))
        visits.append({
            "patient_id": pid, "visit_date": date, "pasi": round(pasi, 1),
            "bsa": np.nan if rng.random() < 0.05 else round(bsa, 1),
            "dlqi": np.nan if rng.random() < 0.10 else dlqi,
            "difficult_sites": difficult,
        })

    def pasi_curve(pasi0, frac, weeks):
        return pasi0 * (1 - frac * (1 - np.exp(-weeks / 5.0))) + rng.normal(0, 0.8)

    def run_systemic(pid, ep, start, pasi0, difficult, prior_bio, bmi, psa, allow_switch=True,
                     baseline_visit=True):
        drug = rng.choice(list(DRUGS))
        cls, frac_mean = DRUGS[drug]
        frac = frac_mean - 0.006 * max(bmi - 25, 0) - 0.08 * prior_bio - 0.05 * psa + rng.normal(0, 0.15)
        frac = float(np.clip(frac, 0.0, 1.0))
        if baseline_visit:  # on a switch, the stopping visit is also the new baseline
            add_visit(pid, start, pasi0, difficult)
        n_fu = rng.integers(3, len(FOLLOW_UP_WEEKS) + 1)
        end, reason = pd.NaT, None
        for w in FOLLOW_UP_WEEKS[:n_fu]:
            d = start + pd.Timedelta(days=int(w * 7 + rng.integers(-10, 11)))
            add_visit(pid, d, pasi_curve(pasi0, frac, w), difficult)
            if w >= 16 and frac < 0.45 and allow_switch:   # non-responder stops and switches
                end, reason = d, "inefficacy"
                break
        if pd.isna(end) and rng.random() < 0.08:
            end, reason = start + pd.Timedelta(days=int(rng.integers(30, 200))), "adverse_event"
        treatments.append({"patient_id": pid, "episode_id": ep, "drug": drug, "drug_class": cls,
                           "level": levels[cls], "start_date": start, "end_date": end, "stop_reason": reason})
        if reason == "inefficacy":
            new_pasi0 = visits[-1]["pasi"]
            run_systemic(pid, ep + 1, end, new_pasi0, difficult,
                         prior_bio or levels[cls] == 4, bmi, psa, allow_switch=False,
                         baseline_visit=False)

    for i in range(n_patients):
        pid = f"F{i + 1:03d}"
        birth = int(rng.integers(1950, 2001))
        bmi = float(np.clip(rng.normal(28, 5), 17, 50))
        psa = bool(rng.random() < 0.25)
        t0 = pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 1100)))
        age_t0 = t0.year - birth
        patients.append({"patient_id": pid, "sex": rng.choice(["M", "F"]), "birth_year": birth,
                         "onset_age": float(rng.integers(8, max(9, min(45, age_t0 - 1)))),
                         "family_history": bool(rng.random() < 0.3), "bmi": round(bmi, 1)})
        if psa:
            comorb.append({"patient_id": pid, "condition": "psoriatic_arthritis",
                           "diagnosis_date": t0 - pd.Timedelta(days=int(rng.integers(-300, 1500)))})
        if bmi >= 30:
            comorb.append({"patient_id": pid, "condition": "obesity", "diagnosis_date": t0 - pd.Timedelta(days=400)})
        if rng.random() < 0.1:
            comorb.append({"patient_id": pid, "condition": "diabetes", "diagnosis_date": t0 - pd.Timedelta(days=800)})
        difficult = bool(rng.random() < 0.35)

        if rng.random() < 0.45:
            # ---- Starts on topical (Model 1 population) ----
            pasi0 = float(rng.uniform(2, 14))
            risk = 0.15 + 0.04 * pasi0 + 0.2 * difficult + 0.15 * psa
            escalates = rng.random() < min(risk, 0.9)
            t_esc = int(rng.integers(60, 700)) if escalates else None
            fu_end = int(rng.integers(200, 760))
            d, pasi = 0, pasi0
            while d <= fu_end and (t_esc is None or d < t_esc):
                add_visit(pid, t0 + pd.Timedelta(days=d), pasi, difficult)
                d += int(rng.integers(80, 130))
                pasi = pasi + rng.normal(0.8 if escalates else -0.3, 1.5)
            if escalates and t_esc <= fu_end:
                esc_date = t0 + pd.Timedelta(days=t_esc)
                treatments.append({"patient_id": pid, "episode_id": 1, "drug": "topical_corticosteroid",
                                   "drug_class": "topical", "level": 1, "start_date": t0,
                                   "end_date": esc_date, "stop_reason": "inefficacy"})
                run_systemic(pid, 2, esc_date, max(pasi + 4, 10), difficult, False, bmi, psa)
            else:
                treatments.append({"patient_id": pid, "episode_id": 1, "drug": "topical_corticosteroid",
                                   "drug_class": "topical", "level": 1, "start_date": t0,
                                   "end_date": pd.NaT, "stop_reason": None})
        else:
            # ---- Starts directly on systemic/biologic (Model 2 population) ----
            prior_bio = bool(rng.random() < 0.3)
            if prior_bio:
                prev_start = t0 - pd.Timedelta(days=int(rng.integers(300, 900)))
                treatments.append({"patient_id": pid, "episode_id": 1, "drug": "adalimumab",
                                   "drug_class": "anti_tnf", "level": 4, "start_date": prev_start,
                                   "end_date": t0 - pd.Timedelta(days=30), "stop_reason": "inefficacy"})
            run_systemic(pid, 2 if prior_bio else 1, t0, float(rng.uniform(10, 30)),
                         difficult, prior_bio, bmi, psa)

    tables = {
        "patients": pd.DataFrame(patients),
        "visits": pd.DataFrame(visits).sort_values(["patient_id", "visit_date"]).reset_index(drop=True),
        "treatments": pd.DataFrame(treatments).sort_values(["patient_id", "episode_id"]).reset_index(drop=True),
        "comorbidities": pd.DataFrame(comorb, columns=["patient_id", "condition", "diagnosis_date"]),
    }

    if inject_errors:  # deliberate errors, for the QC to catch
        v, t = tables["visits"], tables["treatments"]
        v.loc[3, "pasi"] = 80.0                                  # PASI > 72
        v.loc[10, "dlqi"] = 35.0                                 # DLQI > 30
        tables["visits"] = pd.concat([v, v.iloc[[20]]], ignore_index=True)   # duplicate visit
        t.loc[t.end_date.notna().idxmax(), "end_date"] = pd.Timestamp("2000-01-01")  # end < start

    return tables


def save_fictitious(tables: dict) -> None:
    out = path("synthetic") / "fictitious"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)


if __name__ == "__main__":
    save_fictitious(make_fictitious())
    print("Fictitious data saved to data/synthetic/fictitious/")
