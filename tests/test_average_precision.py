import time

import numpy as np
import pytest
from numpy.typing import DTypeLike

from sklearn.metrics import average_precision_score as ap_skl

from scors import average_precision, average_precision_on_two_sorted_samples

def test_sklearn_dosctring():
    # This example is taken from sklearn dosctring
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    y_true = np.array([0, 0, 1, 1], dtype=np.uint8)
    y_scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    expected = 0.8333333333333333
    actual = average_precision(y_true, y_scores, weights=np.ones_like(y_scores, dtype=np.float64))
    assert np.isclose


def test_average_precision_on_two_sorted():
    rng = np.random.default_rng(42)
    n = 1_000_001
    n1 = 900_002
    n2 = n - n1
    y_true = np.require(rng.random(n) > 0.5, dtype=np.uint8)
    y_scores = rng.random(n)
    weights = rng.random(n)
    t0 = time.perf_counter()
    expected = ap_skl(y_true=y_true, y_score=y_scores, sample_weight=weights)
    dt_skl = time.perf_counter() - t0
    print(f"{dt_skl=}")

    y_scores1, y_scores2 = y_scores[:n1], y_scores[n1:]
    indices1 = np.argsort(y_scores1)[::-1]
    indices2 = np.argsort(y_scores2)[::-1]

    y_scores1, y_scores2 = y_scores1[indices1], y_scores2[indices2]
    y_true1, y_true2 = y_true[:n1][indices1], y_true[n1:][indices2]
    weights1, weights2 = weights[:n1][indices1], weights[n1:][indices2]

    t0 = time.perf_counter()
    double_check = average_precision(
        np.concatenate([y_true1, y_true2]),
        np.concatenate([y_scores1, y_scores2]),
        weights=np.concatenate([weights1, weights2])
    )
    dt_reg = time.perf_counter() - t0
    assert np.isclose(double_check, expected)
    print(f"{dt_reg=}")

    t0 = time.perf_counter()
    actual = average_precision_on_two_sorted_samples(
        labels1=y_true1,
        predictions1=y_scores1,
        weights1=weights1,
        labels2=y_true2,
        predictions2=y_scores2,
        weights2=weights2,
    )
    dt_act = time.perf_counter() - t0
    assert np.isclose(actual, expected)
    print(f"{dt_act}")

