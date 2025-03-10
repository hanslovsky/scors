import numpy as np
import pytest
from numpy.typing import DTypeLike

from sklearn.metrics import average_precision_score as ap_skl

from scors import average_precision

def test_sklearn_dosctring():
    # This example is taken from sklearn dosctring
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    y_true = np.array([0, 0, 1, 1], dtype=np.uint8)
    y_scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    expected = 0.8333333333333333
    actual = average_precision(y_true, y_scores, weights=np.ones_like(y_scores, dtype=np.float64))
    assert np.isclose
