import numpy as np

from ._scors import *

__doc__ = _scors.__doc__


def _loo_cossim_many(data: np.ndarray):
    if data.dtype == np.float32:
        return loo_cossim_many_f32(data)
    if data.dtype == np.float64:
        return loo_cossim_many_f64(data)
    raise TypeError(f"Only float32 and float64 data supported, but found {data.dtype}")


def loo_cossim_many(data: np.ndarray):
    sim = _loo_cossim_many(np.reshape(data, (-1, *data.shape[-2:])))
    sim_reshaped = np.reshape(sim, data.shape[:-2])
    return sim_reshaped


__all__ = ["loo_cossim_many", *_scors.__all__]
