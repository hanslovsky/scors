import numpy as np
import pytest
from numpy.typing import DTypeLike

from sklearn.metrics import roc_auc_score as roc_auc_skl

from scors import roc_auc


@pytest.mark.parametrize("max_fpr", [None, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize("with_weights", [True, False])
@pytest.mark.parametrize("dtype", [bool, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64])
def test_compare_sklearn_random(max_fpr: float | None, n: int, with_weights: bool, dtype: DTypeLike):
    rng = np.random.default_rng(42)
    labels = np.require((np.arange(n) / n) >= 0.9, dtype=dtype)
    predictions = rng.random(labels.shape, dtype=np.float64)
    weights = rng.random(labels.shape, dtype=np.float64) if with_weights else None
    score_skl = roc_auc_skl(labels, predictions, sample_weight=weights, max_fpr=max_fpr)
    score = roc_auc(labels, predictions, weights=weights, max_fpr=max_fpr)
    assert np.isclose(score_skl, score), f"{score} != {score_skl}"

