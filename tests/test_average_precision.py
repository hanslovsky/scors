import numpy as np
import pytest

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


@pytest.mark.parametrize("n", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize("with_weights", [True, False])
def test_compare_sklearn_random(n: int, with_weights: bool):
    rng = np.random.default_rng(42)
    # TODO sklearn returns 0.0 if all labels are negative,
    #      my implementation returns nan.
    #      Figure out if we want to diverge from sklearn here.
    labels = np.require((np.arange(n) / n) >= 0.9, dtype=np.uint8)
    predictions = rng.random(labels.shape, dtype=np.float64)
    weights = rng.random(labels.shape, dtype=np.float64) if with_weights else None
    score_skl = ap_skl(labels, predictions, sample_weight=weights)
    score = average_precision(labels, predictions, weights=weights)
    assert np.isclose(score_skl, score), f"{score} != {score_skl}"

