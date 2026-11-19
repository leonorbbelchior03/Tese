"""The same patient must never be in training and test in the same fold."""
import numpy as np

from src.evaluation.cv import grouped_cv_splits


def test_no_patient_in_train_and_test():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), rng.integers(1, 4, 40))   # 1-3 episodes per patient
    y = rng.integers(0, 2, len(groups))
    for _, _, tr, te in grouped_cv_splits(y, groups, n_splits=5, n_repeats=3):
        assert set(groups[tr]).isdisjoint(groups[te])
