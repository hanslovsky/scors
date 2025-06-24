import numpy as np
import pytest

import scors

@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_loo_cossim_single(dtype):
    data = np.asarray([[0.77395605, 0.43887844, 0.85859792],
                       [0.69736803, 0.09417735, 0.97562235]])
    expected = 0.95385941
    actual = scors.loo_cossim(np.require(data, dtype=dtype))
    assert np.allclose(expected, actual, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_loo_cossim_many_2d(dtype):
    data = np.asarray([[0.77395605, 0.43887844, 0.85859792],
                       [0.69736803, 0.09417735, 0.97562235]])
    expected = np.full((), 0.95385941)
    actual = scors.loo_cossim_many(np.require(data, dtype=dtype))
    assert actual.shape == expected.shape, f"{actual.shape=} != {expected.shape=}"
    assert np.allclose(expected, actual, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_loo_cossim_many(dtype):
    data = np.asarray([[[0.77395605, 0.43887844, 0.85859792],
                        [0.69736803, 0.09417735, 0.97562235]],
                       [[0.7611397 , 0.78606431, 0.12811363],
                        [0.45038594, 0.37079802, 0.92676499]],
                       [[0.64386512, 0.82276161, 0.4434142 ],
                        [0.22723872, 0.55458479, 0.06381726]],
                       [[0.82763117, 0.6316644 , 0.75808774],
                        [0.35452597, 0.97069802, 0.89312112]]])
    expected = np.asarray([0.95385941, 0.62417001, 0.92228589, 0.90025417])
    actual = scors.loo_cossim_many(np.require(data, dtype=dtype))
    assert actual.shape == expected.shape, f"{actual.shape=} != {expected.shape=}"
    assert np.allclose(expected, actual, rtol=1e-6), f"{actual=} != {expected=}"


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"])
def test_loo_cossim_wrong_type(dtype):
    data = np.ones((2, 3))
    with pytest.raises(TypeError, match="Only float32 and float64"):
        scors.loo_cossim(np.require(data, dtype=dtype))


@pytest.mark.parametrize("ndim", [0, 1, 3, 4, 7])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_loo_cossim_wrong_shape(ndim, dtype):
    shape = tuple(np.arange(ndim) + 1)
    data = np.ones(shape, dtype=dtype)
    with pytest.raises(TypeError, match="Expected 2-dimensional array for data"):
        scors.loo_cossim(data)


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "float64"])
def test_loo_cossim_many_f32_wrong_type(dtype):
    data = np.ones((1, 2, 3))
    with pytest.raises(TypeError, match="Only float32 data supported"):
        scors.loo_cossim_many_f32(np.require(data, dtype=dtype))

        
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "float32"])
def test_loo_cossim_many_f64_wrong_type(dtype):
    data = np.ones((1, 2, 3))
    with pytest.raises(TypeError, match="Only float64 data supported"):
        scors.loo_cossim_many_f64(np.require(data, dtype=dtype))

        
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"])
def test_loo_cossim_many_wrong_type(dtype):
    data = np.ones((1, 2, 3))
    with pytest.raises(TypeError, match="Only float32 and float64 data supported"):
        scors.loo_cossim_many(np.require(data, dtype=dtype))


@pytest.mark.parametrize("ndim", [0, 1, 2, 4, 7])
@pytest.mark.parametrize(
    ("f", "dtype"),
    [(scors.loo_cossim_many_f32, "float32"),
     (scors.loo_cossim_many_f64, "float64")],
    ids=["float32", "float64"]
)
def test_loo_cossim_many_typed_wrong_shape(ndim, f, dtype):
    shape = tuple(np.arange(ndim) + 1)
    data = np.ones(shape, dtype=dtype)
    with pytest.raises(TypeError, match="Expected 3-dimensional array for data"):
        f(data)


@pytest.mark.parametrize("ndim", [0, 1])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_loo_cossim_many_wrong_shape(ndim, dtype):
    shape = tuple(np.arange(ndim) + 2)
    data = np.ones(shape, dtype=dtype)
    with pytest.raises(TypeError, match="Expected 3-dimensional array for data"):
        scors.loo_cossim_many(data)
        
