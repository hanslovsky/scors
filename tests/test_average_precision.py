import time

import numba
import numpy as np
import pytest
from numpy.typing import DTypeLike

from sklearn.metrics import average_precision_score as ap_skl

from scors import Order, average_precision, average_precision_on_two_sorted_samples #_deprecated as average_precision_on_two_sorted_samples

def test_sklearn_dosctring():
    # This example is taken from sklearn dosctring
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    y_true = np.array([0, 0, 1, 1], dtype=np.uint8)
    y_scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    expected = 0.8333333333333333
    actual = average_precision(y_true, y_scores, weights=np.ones_like(y_scores, dtype=np.float64))
    assert np.isclose


@numba.njit
def merge_by_scores_descending(
        scores1: np.ndarray,
        scores2: np.ndarray,
        true1: np.ndarray,
        true2: np.ndarray,
        weights1: np.ndarray,
        weights2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n1 = scores1.shape[0]
    n2 = scores2.shape[0]
    n = n1 + n2
    scores = np.empty((n,), dtype=scores1.dtype)
    true = np.empty((n,), dtype=true1.dtype)
    weights = np.ones_like(scores)  # np.empty((n,), dtype=weights1.dtype)
    i,k = 0, 0
    for m in range(n):
        if i == n1:
            s, t, w, idx = scores2, true2, weights2, k
            k += 1
        elif k == n2 or scores1[i] >= scores2[k]:
            s, t, w, idx = scores1, true1, weights1, i
            i += 1
        else:
            s, t, w, idx = scores2, true2, weights2, k
            k += 1
        scores[m] = s[idx]
        true[m] = t[idx]
        weights[m] = w[idx]
    return scores, true, weights


def test_average_precision_on_two_sorted():
    rng = np.random.default_rng(42)
    n = 10_000_001
    n1 =  9_000_002
    n2 = n - n1
    y_true = np.require(rng.random(n) > 0.5, dtype=bool)
    y_scores = rng.random(n)
    weights = rng.random(n)

    def split(array: np.ndarray, where=n1) -> tuple[np.ndarray, np.ndarray]:
        return array[:where], array[where:]
    
    t0 = time.perf_counter()
    # concatenation should be measured as part of the runtime
    expected = ap_skl(y_true=np.concatenate(split(y_true)), y_score=np.concatenate(split(y_scores)), sample_weight=np.concatenate(split(weights)))
    dt_skl = time.perf_counter() - t0
    print()
    print(f"{dt_skl=}")

    indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = y_scores[indices]
    y_true_sorted = y_true[indices]
    weights_sorted = weights[indices]

    t0 = time.perf_counter()
    double_check_sorted = average_precision(y_true_sorted, y_scores_sorted, weights=weights_sorted, order=Order.DESCENDING)
    dt_desc = time.perf_counter() - t0
    assert np.isclose(double_check_sorted, expected)
    print(f"{dt_desc=} (lower bound to that does not consider time required for merging the two arrays)")

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

    # trigger jit
    y_scores_merged, y_true_merged, weights_merged = merge_by_scores_descending(
        y_scores1,
        y_scores2,
        y_true1,
        y_true2,
        weights1,
        weights2
    )
    t0 = time.perf_counter()
    y_scores_merged, y_true_merged, weights_merged = merge_by_scores_descending(
        y_scores1,
        y_scores2,
        y_true1,
        y_true2,
        weights1,
        weights2
    )
    dt_merge = time.perf_counter() - t0
    t0 = time.perf_counter()
    check_merged = average_precision(y_true_merged, y_scores_merged, weights=weights_merged, order=Order.DESCENDING)
    dt_merged = time.perf_counter() - t0
    assert np.isclose(check_merged, expected)
    print(f"dt_sum={dt_merge+dt_merged} {dt_merge=} {dt_merged=}")

    import logging
    logging.basicConfig(level="INFO")
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
    print(f"{dt_act=}")

