import numpy as np

from ._scors import *

__doc__ = _scors.__doc__
if hasattr(_scors, "__all__"):
    def loo_cossim_many(data: np.ndarray):
        if data.dtype == np.float32:
            return loo_cossim_many_f32(data)
        if data.dtype == np.float64:
            return loo_cossim_many_f64(data)
        raise ValueError(f"Only float32 and float64 data supported, but found {data.dtype}")
    __all__ = _scors.__all__
