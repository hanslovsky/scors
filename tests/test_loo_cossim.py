import numpy as np

import scors

def test_loo_cossim_single():
    data = np.random.default_rng(42).random((2, 3))
    expected = 0.95385941
    actual = scors.loo_cossim(data)
    assert np.allclose(expected, actual, rtol=1e-8)

def test_loo_cossim_many():
    data = np.random.default_rng(42).random((4, 2, 3))
    expected = np.asarray([0.95385941, 0.62417001, 0.92228589, 0.90025417])
    actual = scors.loo_cossim_many(data)
    assert actual.shape == expected.shape, f"{actual.shape=} != {expected.shape=}"
    assert np.allclose(expected, actual, rtol=1e-8), f"{actual=} != {expected=}"

