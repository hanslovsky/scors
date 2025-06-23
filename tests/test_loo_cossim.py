import numpy as np
import pytest

import scors

@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_loo_cossim_single(dtype):
    data = np.random.default_rng(42).random((2, 3))
    expected = 0.95385941
    actual = scors.loo_cossim(np.require(data, dtype=dtype))
    assert np.allclose(expected, actual, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_loo_cossim_many(dtype):
    data = np.random.default_rng(42).random((4, 2, 3))
    expected = np.asarray([0.95385941, 0.62417001, 0.92228589, 0.90025417])
    actual = scors.loo_cossim_many(np.require(data, dtype=dtype))
    assert actual.shape == expected.shape, f"{actual.shape=} != {expected.shape=}"
    assert np.allclose(expected, actual, rtol=1e-6), f"{actual=} != {expected=}"

