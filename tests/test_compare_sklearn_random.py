from functools import cache, reduce

import numpy as np
import pytest
from numpy.typing import DTypeLike

from sklearn.metrics import (
    average_precision_score as average_precision_skl,
    roc_auc_score as roc_auc_skl
)

from scors import average_precision, roc_auc


def all_parameters(func):
    decorators = [
        pytest.mark.parametrize("fraction_negative", [0.0, 0.5, 1.0], ids=[f"fn={x}" for x in ["0", "0.5", "1"]]),
        pytest.mark.parametrize("n", [100, 10_000]),
        pytest.mark.parametrize("with_weights", [True, False]),
        pytest.mark.parametrize("dtype", [bool, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]),
    ]
    return reduce(lambda a, b: b(a), decorators[::-1], func)


@cache
def _make_data(n: int, fraction_negative: float, dtype: DTypeLike, with_weights: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(42)
    labels = np.require((np.arange(n) / n) >= fraction_negative, dtype=dtype)
    predictions = rng.random(labels.shape, dtype=np.float64)
    weights = rng.random(labels.shape, dtype=np.float64) if with_weights else None
    return labels, predictions, weights


def _run_and_assert(n: int, fraction_negative: float, with_weights: bool, dtype: DTypeLike, f_skl, f, **kwargs):
    labels, predictions, weights = _make_data(n, fraction_negative, dtype=dtype, with_weights=with_weights)
    score_skl = f_skl(labels, predictions, sample_weight=weights, **kwargs)
    score = f(labels, predictions, weights=weights, **kwargs)
    assert np.isclose(score_skl, score, equal_nan=True), f"{score} != {score_skl}"


@all_parameters
def test_average_precision(n: int, fraction_negative: float, with_weights: bool, dtype: DTypeLike):
    _run_and_assert(n, fraction_negative, with_weights, dtype, average_precision_skl, average_precision)


@pytest.mark.parametrize("max_fpr", [None, 0.5, 0.9, 1.0], ids=[f"{max_fpr=}" for max_fpr in ["None", "0.5", "0.9", "1"]])
@all_parameters
def test_roc_auc(max_fpr: float | None, n: int, fraction_negative: float, with_weights: bool, dtype: DTypeLike):
    _run_and_assert(n, fraction_negative, with_weights, dtype, roc_auc_skl, roc_auc, max_fpr=max_fpr)

