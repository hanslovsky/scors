import logging

import numpy as np

from . import _scors as scors
from ._scors import *

__doc__ = _scors.__doc__


_logger = logging.getLogger("scors")


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


average_precision_on_two_sorted_samples_deprecated = average_precision_on_two_sorted_samples


def average_precision_on_two_sorted_samples(
    labels1: np.ndarray,    
    predictions1: np.ndarray,
    weights1: np.ndarray | None,
    labels2: np.ndarray,    
    predictions2: np.ndarray,
    weights2: np.ndarray | None,
):
    l1, p1, w1 = labels1, predictions1, weights1
    l2, p2, w2 = labels2, predictions2, weights2

    if l1.dtype != l2.dtype:
        raise TypeError(f"Label arrays must have the same dtype but found: {l1.dtype=} != {l2.dtype=}")

    if p1.dtype != p2.dtype:
        raise TypeError(f"Predictions arrays must have the same dtype but found: {p1.dtype=} != {p2.dtype=}")

    if w1 is not None and w1.dtype != p1.dtype:
        raise TypeError(f"Weight array must have the same dtype as predictions but found: {w1.dtype=} != {p1.dtype=}")

    if w2 is not None and w2.dtype != p2.dtype:
        raise TypeError(f"Weight array must have the same dtype as predictions but found: {w2.dtype=} != {p2.dtype=}")

    supported_label_types = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
    supported_score_types = ("float32", "float64")

    if l1.dtype.name not in supported_label_types:
        raise TypeError(f"Unsupported type for labels: {l1.dtype}. Supported types: {supported_label_types}")
    l_dtype = l1.dtype.name.replace("uint", "u").replace("int", "i")

    if w1.dtype.name not in supported_score_types:
        raise TypeError(f"Unsupported type for predictions/weights: {w1.dtype}. Supported types: {supported_score_types}")
    p_dtype = p1.dtype.name.replace("float", "f")

    func_name = f"average_precision_on_two_sorted_samples_{l_dtype}_{p_dtype}"
    _logger.info(f"{func_name=}")
    func = getattr(scors, func_name)
    return func(l1, p1, w1, l2, p2, w2)
    


__all__ = [
    "average_precision_on_two_sorted_samples",
    "loo_cossim_many",
    scors.__all__
]
