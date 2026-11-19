"""
Baseline models. All preprocessing lives INSIDE the Pipeline, so it is fitted
only on the training fold (no leakage from the test fold).
"""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessor(numeric: list, categorical: list) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def null_model() -> Pipeline:
    """Always predicts the outcome prevalence. Minimum reference to beat."""
    return Pipeline([("model", DummyClassifier(strategy="prior"))])


def logistic(numeric: list, categorical: list, C: float = 1.0, seed: int = 42) -> Pipeline:
    return Pipeline([("prep", preprocessor(numeric, categorical)),
                     ("model", LogisticRegression(C=C, max_iter=2000, random_state=seed))])
