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


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_loo_cossim_many_4d(dtype):
    data = np.asarray([[[[0.77395605, 0.43887844, 0.85859792, 0.69736803, 0.09417735],
                         [0.97562235, 0.7611397 , 0.78606431, 0.12811363, 0.45038594],
                         [0.37079802, 0.92676499, 0.64386512, 0.82276161, 0.4434142 ],
                         [0.22723872, 0.55458479, 0.06381726, 0.82763117, 0.6316644 ]],

                        [[0.75808774, 0.35452597, 0.97069802, 0.89312112, 0.7783835 ],
                         [0.19463871, 0.466721  , 0.04380377, 0.15428949, 0.68304895],
                         [0.74476216, 0.96750973, 0.32582536, 0.37045971, 0.46955581],
                         [0.18947136, 0.12992151, 0.47570493, 0.22690935, 0.66981399]],

                        [[0.43715192, 0.8326782 , 0.7002651 , 0.31236664, 0.8322598 ],
                         [0.80476436, 0.38747838, 0.2883281 , 0.6824955 , 0.13975248],
                         [0.1999082 , 0.00736227, 0.78692438, 0.66485086, 0.70516538],
                         [0.78072903, 0.45891578, 0.5687412 , 0.139797  , 0.11453007]]],

                       [[[0.66840296, 0.47109621, 0.56523611, 0.76499886, 0.63471832],
                         [0.5535794 , 0.55920716, 0.3039501 , 0.03081783, 0.43671739],
                         [0.21458467, 0.40852864, 0.85340307, 0.23393949, 0.05830274],
                         [0.28138389, 0.29359376, 0.66191651, 0.55703215, 0.78389821]],

                        [[0.66431354, 0.40638686, 0.81402038, 0.16697292, 0.02271207],
                         [0.09004786, 0.72235935, 0.46187723, 0.16127178, 0.50104478],
                         [0.1523121 , 0.69632038, 0.44615628, 0.38102123, 0.30151209],
                         [0.63028259, 0.36181261, 0.08764992, 0.1180059 , 0.96189766]],

                        [[0.90858069, 0.69970713, 0.26586996, 0.96917638, 0.7787509 ],
                         [0.71689019, 0.4493615 , 0.27224156, 0.09639096, 0.9026024 ],
                         [0.45577629, 0.20236336, 0.30595662, 0.57921957, 0.17677278],
                         [0.85661428, 0.75851953, 0.71946296, 0.43209304, 0.62730884]]]])
    expected = np.asarray([[0.84580024, 0.83749452, 0.8000156 ],
                           [0.85025678, 0.77409066, 0.90218724]])
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
        scors.scors.loo_cossim_many_f32(np.require(data, dtype=dtype))

        
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "float32"])
def test_loo_cossim_many_f64_wrong_type(dtype):
    data = np.ones((1, 2, 3))
    with pytest.raises(TypeError, match="Only float64 data supported"):
        scors.scors.loo_cossim_many_f64(np.require(data, dtype=dtype))

        
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"])
def test_loo_cossim_many_wrong_type(dtype):
    data = np.ones((1, 2, 3))
    with pytest.raises(TypeError, match="Only float32 and float64 data supported"):
        scors.loo_cossim_many(np.require(data, dtype=dtype))


@pytest.mark.parametrize("ndim", [0, 1, 2, 4, 7])
@pytest.mark.parametrize(
    ("f", "dtype"),
    [(scors.scors.loo_cossim_many_f32, "float32"),
     (scors.scors.loo_cossim_many_f64, "float64")],
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
        
