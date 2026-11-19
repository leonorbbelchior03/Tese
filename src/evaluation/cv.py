"""
Repeated, stratified cross-validation GROUPED BY PATIENT:
the same patient is never in training and test at the same time.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def grouped_cv_splits(y, groups, n_splits=5, n_repeats=10, seed=42):
    """Yields (repeat, fold, train_idx, test_idx)."""
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            yield r, f, tr, te


def evaluate(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
             n_splits=5, n_repeats=10, seed=42):
    """Returns (per-fold metrics, out-of-fold predictions)."""
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    metrics, preds = [], []
    for r, f, tr, te in grouped_cv_splits(y, groups, n_splits, n_repeats, seed):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        yt = y.iloc[te]
        auc = roc_auc_score(yt, p) if yt.nunique() == 2 else np.nan
        metrics.append({"repeat": r, "fold": f, "n_test": len(te), "events_test": int(yt.sum()),
                        "auc": auc, "brier": brier_score_loss(yt, p)})
        preds.append(pd.DataFrame({"repeat": r, "row": te, "y": yt.values, "p": p}))
    return pd.DataFrame(metrics), pd.concat(preds, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.Series:
    """Mean and 2.5-97.5 percentiles over folds."""
    out = {}
    for k in ["auc", "brier"]:
        s = metrics[k].dropna()
        out[f"{k}_mean"] = s.mean()
        out[f"{k}_p2.5"] = s.quantile(0.025)
        out[f"{k}_p97.5"] = s.quantile(0.975)
    return pd.Series(out).round(3)


def calibration_table(preds: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Observed vs predicted risk by quantile of predicted risk."""
    d = preds.copy()
    d["bin"] = pd.qcut(d.p, q=n_bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"),
                                               n=("y", "size")).reset_index()
