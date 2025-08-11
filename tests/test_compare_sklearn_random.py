import math
from functools import cache, partial, reduce

import numpy as np
import pytest
from numpy.typing import DTypeLike

from sklearn.metrics import (
    average_precision_score as average_precision_skl,
    roc_auc_score as roc_auc_skl
)

from scors import average_precision, average_precision_on_two_sorted_samples, roc_auc, roc_auc_on_two_sorted_samples


def _all_parameters(func, with_weight_dtype: bool):
    decorators = [
        pytest.mark.parametrize("split_fraction", [None, 0.5], ids=["none", "half"]),
        pytest.mark.parametrize("n", [100, 10_000]),
        pytest.mark.parametrize("with_weights", [True, False]),
        pytest.mark.parametrize("dtype", [bool, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]),
        pytest.mark.parametrize("score_dtype", [np.float32, np.float64]),
        pytest.mark.parametrize("weight_dtype", [np.float32, np.float64]),
    ]
    if not with_weight_dtype:
        decorators = decorators[:-1]
    return reduce(lambda a, b: b(a), decorators[::-1], func)


all_parameters = partial(_all_parameters, with_weight_dtype=False)


@cache
def _make_data(n: int, fraction_negative: float, dtype: DTypeLike, score_dtype: DTypeLike, with_weights: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(42)
    labels = np.require((np.arange(n) / n) >= fraction_negative, dtype=dtype)
    predictions = np.require(rng.random(labels.shape, dtype=np.float64), dtype=score_dtype)
    weights = np.require(rng.random(labels.shape, dtype=np.float64), dtype=score_dtype) if with_weights else None
    return labels, predictions, weights


def _run_and_assert(n: int, fraction_negative: float, with_weights: bool, dtype: DTypeLike, score_dtype: DTypeLike, f_skl, f, split_fraction: float | None = None, **kwargs):
    labels, predictions, weights = _make_data(n, fraction_negative, dtype=dtype, score_dtype=score_dtype, with_weights=with_weights)
    score_skl = f_skl(labels, predictions, sample_weight=weights, **kwargs)
    if split_fraction is None:
        score = f(labels, predictions, weights=weights, **kwargs)
    else:
        sort_indices = np.argsort(predictions)
        labels = labels[sort_indices]
        predictions = predictions[sort_indices]
        weights = None if weights is None else weights[sort_indices]
        split_index = math.floor(split_fraction * n)
        sl1 = slice(None, split_index, None)
        sl2 = slice(split_index, None, None)
        score = f(labels[sl1], predictions[sl1], None if weights is None else weights[sl1],
                  labels[sl2], predictions[sl2], None if weights is None else weights[sl2])
    assert np.isclose(score_skl, score, equal_nan=True), f"{score} != {score_skl}"


@all_parameters
def test_average_precision(split_fraction: float | None, n: int, with_weights: bool, dtype: DTypeLike, score_dtype: DTypeLike):
    fraction_negative = 0.5
    _run_and_assert(n, fraction_negative, with_weights, dtype, score_dtype, average_precision_skl, average_precision)


@pytest.mark.parametrize("max_fpr", [None, 0.5, 0.9, 1.0], ids=[f"{max_fpr=}" for max_fpr in ["None", "0.5", "0.9", "1"]])
@all_parameters
def test_roc_auc(max_fpr: float | None, split_fraction: float | None, n: int, with_weights: bool, dtype: DTypeLike, score_dtype: DTypeLike):
    fraction_negative = 0.5
    _run_and_assert(n, fraction_negative, with_weights, dtype, score_dtype, roc_auc_skl, roc_auc, max_fpr=max_fpr)

