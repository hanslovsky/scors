import numpy as np
import pytest

from sklearn.metrics import roc_auc_score as roc_auc_skl

from scors import roc_auc


@pytest.mark.parametrize("max_fpr", [None, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize("with_weights", [True, False])
def test_compare_sklearn_random(max_fpr: float | None, n: int, with_weights: bool):
    rng = np.random.default_rng(42)
    labels = np.require((np.arange(n) / n) >= 0.9, dtype=np.uint8)
    predictions = rng.random(labels.shape, dtype=np.float64)
    weights = rng.random(labels.shape, dtype=np.float64) if with_weights else None
    score_skl = roc_auc_skl(labels, predictions, sample_weight=weights, max_fpr=max_fpr)
    score = roc_auc(labels, predictions, weights=weights, max_false_positive_rate=max_fpr)
    assert np.isclose(score_skl, score), f"{score} != {score_skl}"

