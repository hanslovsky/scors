import numpy as np
import pytest

from sklearn.metrics import roc_auc_score as roc_auc_skl

from scors import roc_auc


@pytest.mark.parametrize("n", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize("with_weights", [True, False])
def test_compare_sklearn_random(n: int, with_weights: bool):
    rng = np.random.default_rng(42)
    labels = np.require((np.arange(n) / n) >= 0.9, dtype=np.uint8)
    predictions = rng.random(labels.shape, dtype=np.float64)
    weights = rng.random(labels.shape, dtype=np.float64) if with_weights else np.ones_like(predictions)
    score_skl = roc_auc_skl(labels, predictions, sample_weight=weights)
    score = roc_auc(labels, predictions, weights=weights)
    assert np.isclose(score_skl, score), f"{score} != {score_skl}"

