# Psoriasis: prediction of therapeutic escalation and treatment response

MSc thesis code. Machine-learning models to predict, from routine clinical data:

- **Model 1: therapeutic escalation**: risk of moving from topical to systemic/biologic treatment (6 and 12 months).
- **Model 2: treatment response**: PASI90 (primary), PASI75 and DLQI improvement at a class-specific window after starting a systemic/targeted treatment.

Validation is leakage-free: predictors use only information up to the reference date (T0), preprocessing is fitted inside each fold, and cross-validation is grouped by patient.

> ⚠️ No real patient data are stored in this repository (`data/raw`, `data/interim` and `data/processed` are git-ignored).

## Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.synthetic.fictitious      # creates the fictitious test data
python -m pytest                        # all tests must pass
```

Then open the notebooks in `notebooks/`, in order. Run everything from the repository root.

## Structure

```
configs/
  params.yaml              # all parameters (seeds, ranges, tolerances, thresholds)
  treatment_windows.yaml   # assessment window per drug, with literature source
  datasets.yaml            # file paths and column mapping per data source
data/
  raw/                     # original files, never edited (git-ignored)
  interim/                 # cleaned common tables (git-ignored)
  processed/               # modelling datasets (git-ignored)
  synthetic/               # fictitious test data (can be committed)
notebooks/
  01_data_validation       # import + quality control + cleaning
  03_eda                   # cohort description, outcomes, trajectories
  04_model1_escalation     # Model 1 baseline (pipeline test)
  05_model2_response       # Model 2 baseline: null model vs logistic regression
src/
  data/                    # loaders (one per source), schemas, QC, cleaning, cohorts
  outcomes/                # escalation (Model 1), response (Model 2)
  features/                # predictors at T0
  models/                  # model pipelines
  evaluation/              # grouped CV, metrics, calibration
  longitudinal/            # prediction update over time (future)
  synthetic/               # fictitious test data generator
tests/                     # leakage, CV grouping, outcome definitions
reports/                   # figures and tables for the thesis
```

## Common data format

Every data source is converted by its loader into four tables:

| Table | One row per | Main columns |
|---|---|---|
| patients | patient | patient_id, sex, birth_year, onset_age, family_history, bmi |
| visits | visit | patient_id, visit_date, pasi, bsa, dlqi, difficult_sites |
| treatments | treatment episode | patient_id, episode_id, drug, drug_class, level (1–4), start_date, end_date, stop_reason |
| comorbidities | diagnosis | patient_id, condition, diagnosis_date |

Adding a new source = writing a new loader in `src/data/loaders/` that returns these tables.

## Data sources

| Source | Status | Use |
|---|---|---|
| Clinical data (Dr. Paulo Ferreira) | pending | main dataset (Models 1 and 2) |
| Kruczek et al., IL-17 real-world data, Mendeley Data, doi:10.17632/t8mnz5cyzy.1 (CC BY 4.0) | public | pilot / external validation of Model 2 (anti-IL-17 only, weeks 0, 4, 12) |
| Fictitious data (`src/synthetic/fictitious.py`) | generated | code testing only, results must not be interpreted |
