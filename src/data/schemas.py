"""
Expected format of the common tables (data dictionary in code).

Each loader must return these 4 tables. The schemas are used by the QC
(src/data/qc.py) to find out-of-range values and wrong types.
"""
try:  # pandera >= 0.24
    import pandera.pandas as pa
except ImportError:  # older pandera
    import pandera as pa

from src.config import params

R = params()["ranges"]
CLASSES = list(params()["levels"].keys())
STOP_REASONS = ["inefficacy", "adverse_event", "remission", "patient_choice", "other"]
CONDITIONS = ["psoriatic_arthritis", "obesity", "diabetes", "cardiovascular", "depression"]

patients_schema = pa.DataFrameSchema(
    {
        "patient_id": pa.Column(str, unique=True),
        "sex": pa.Column(str, pa.Check.isin(["M", "F"])),
        "birth_year": pa.Column(int, pa.Check.in_range(1900, 2010)),
        "onset_age": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
        "family_history": pa.Column("boolean", nullable=True),
        "bmi": pa.Column(float, pa.Check.in_range(*R["bmi"]), nullable=True),
    },
    name="patients",
)

visits_schema = pa.DataFrameSchema(
    {
        "patient_id": pa.Column(str),
        "visit_date": pa.Column("datetime64[ns]"),
        "pasi": pa.Column(float, pa.Check.in_range(*R["pasi"]), nullable=True),
        "bsa": pa.Column(float, pa.Check.in_range(*R["bsa"]), nullable=True),
        "dlqi": pa.Column(float, pa.Check.in_range(*R["dlqi"]), nullable=True),
        "difficult_sites": pa.Column("boolean", nullable=True),
    },
    name="visits",
)

treatments_schema = pa.DataFrameSchema(
    {
        "patient_id": pa.Column(str),
        "episode_id": pa.Column(int),
        "drug": pa.Column(str),
        "drug_class": pa.Column(str, pa.Check.isin(CLASSES)),
        "level": pa.Column(int, pa.Check.in_range(1, 4)),
        "start_date": pa.Column("datetime64[ns]"),
        "end_date": pa.Column("datetime64[ns]", nullable=True),
        "stop_reason": pa.Column(str, pa.Check.isin(STOP_REASONS), nullable=True),
    },
    name="treatments",
)

comorbidities_schema = pa.DataFrameSchema(
    {
        "patient_id": pa.Column(str),
        "condition": pa.Column(str, pa.Check.isin(CONDITIONS)),
        "diagnosis_date": pa.Column("datetime64[ns]", nullable=True),
    },
    name="comorbidities",
)

SCHEMAS = {
    "patients": patients_schema,
    "visits": visits_schema,
    "treatments": treatments_schema,
    "comorbidities": comorbidities_schema,
}
