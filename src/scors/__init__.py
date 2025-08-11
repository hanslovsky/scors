import logging
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Literal

import numpy as np

from . import _scors as scors
from ._scors import loo_cossim

__doc__ = scors.__doc__
_logger = logging.getLogger("scors")
_supported_label_types = (
    ("bool", "bool",),
    ("int8", "i8",),
    ("int16", "i16"),
    ("int32", "i32"),
    ("int64", "i64"),
    ("uint8", "u8",),
    ("uint16", "u16"),
    ("uint32", "u32"),
    ("uint64", "u64"),
)
_supported_score_types = (
    ("float32", "f32"),
    ("float64", "f64"),
)

Order = scors.Order


def _lookup_supported_type(dtype: str | np.dtype, supported_type_dict: dict[str, str]) -> str:
    dtype_name = dtype if isinstance(dtype, str) else dtype.name
    try:
        return supported_type_dict[dtype_name]
    except KeyError as e:
        raise TypeError(f"Unsupported {dtype=} ({dtype_name=}). Supported types: {tuple(supported_type_dict.keys())}")


def _lookup_supported_label_type(dtype: str | np.dtype) -> str:
    return _lookup_supported_type(dtype, dict(_supported_label_types))


def _lookup_supported_score_type(dtype: str | np.dtype) -> str:
    return _lookup_supported_type(dtype, dict(_supported_score_types))


def _lookup_supported_score_func(
        func_name: str,
        label_dtype: str | np.dtype,
        score_dtype: np.dtype | str
) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]:
    specific_func_name = f"{func_name}_{_lookup_supported_label_type(label_dtype)}_{_lookup_supported_score_type(score_dtype)}"
    return getattr(scors, specific_func_name)


def _from_generic_score(scors_name: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None, Order | None], float]:
    def scors_func(labels: np.ndarray, predictions: np.ndarray, *args, weights: np.ndarray | None = None, order: Order | None = None, **kwargs):
        if weights is not None and predictions.dtype != weights.dtype:
            raise ValueError(f"Weights must be the same dtype as predictions, if provided, but found {weights.dtype=} != {predictions.dtype=}")
        return _lookup_supported_score_func(scors_name, labels.dtype, predictions.dtype)(labels, predictions, *args, weights=weights, order=order, **kwargs)
    return scors_func

    
def _loo_cossim_many(data: np.ndarray):
    if data.dtype == np.float32:
        return scors.loo_cossim_many_f32(data)
    if data.dtype == np.float64:
        return scors.loo_cossim_many_f64(data)
    raise TypeError(f"Only float32 and float64 data supported, but found {data.dtype}")


def loo_cossim_many(data: np.ndarray):
    sim = _loo_cossim_many(np.reshape(data, (-1, *data.shape[-2:])))
    sim_reshaped = np.reshape(sim, data.shape[:-2])
    return sim_reshaped


def average_precision(labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None, order: Order | None = None):
    return _from_generic_score("average_precision")(labels=labels, predictions=predictions, weights=weights, order=order)


def roc_auc(labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None, order: Order | None = None, max_fpr: float | None = None):
    return _from_generic_score("roc_auc")(labels=labels, predictions=predictions, weights=weights, order=order, max_fpr=max_fpr)


def _score_two_sorted_samples(name: Literal["average_precision", "roc_auc"]):
    def decorator(func):
        @wraps(func)
        def _func(
                labels1: np.ndarray,    
                predictions1: np.ndarray,
                weights1: np.ndarray | None,
                labels2: np.ndarray,    
                predictions2: np.ndarray,
                weights2: np.ndarray | None,
                *args,
                **kwargs,
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

            l_dtype = _lookup_supported_label_type(l1.dtype)
            p_dtype = _lookup_supported_score_type(p1.dtype)

            func_name = f"{name}_on_two_sorted_samples_{l_dtype}_{p_dtype}"
            _logger.info(f"{func_name=}")
            func = getattr(scors, func_name)
            return func(l1, p1, w1, l2, p2, w2, *args, **kwargs)
        return _func
    return decorator


@_score_two_sorted_samples(name="average_precision")
def average_precision_on_two_sorted_samples(
    labels1: np.ndarray,    
    predictions1: np.ndarray,
    weights1: np.ndarray | None,
    labels2: np.ndarray,    
    predictions2: np.ndarray,
    weights2: np.ndarray | None,
):
    raise NotImplementedError()


@_score_two_sorted_samples(name="roc_auc")
def roc_auc_on_two_sorted_samples(
    labels1: np.ndarray,    
    predictions1: np.ndarray,
    weights1: np.ndarray | None,
    labels2: np.ndarray,    
    predictions2: np.ndarray,
    weights2: np.ndarray | None,
    max_fpr: float | None = None,
):
    raise NotImplementedError()
    

__all__ = sorted([
    "Order",
    "average_precision",
    "average_precision_on_two_sorted_samples",
    "loo_cossim",
    "loo_cossim_many",
    "roc_auc",
    "roc_auc_on_two_sorted_samples",
    "scors",
])
