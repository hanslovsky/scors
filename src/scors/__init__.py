import logging
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Literal

import numpy as np

from . import _scors as scors
from ._scors import loo_cossim as _loo_cossim_rust


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
"""Enum indicating the sort order of predictions passed to scoring functions.

Pass to the ``order`` parameter of :func:`average_precision` and
:func:`roc_auc` to skip the internal sort when the data is already ordered.

- ``Order.ASCENDING`` — predictions are sorted smallest-first.
- ``Order.DESCENDING`` — predictions are sorted largest-first (highest score first).
"""


def loo_cossim(data: np.ndarray) -> float:
    """Leave-one-out cosine similarity for a matrix of replicates.

    For each replicate row, computes the cosine similarity between that row
    and the mean of the remaining rows (the leave-one-out mean), then averages
    over all replicates.

    The inner loops are SIMD-vectorized (NEON on ARM, SSE2 on x86-64) for
    C-contiguous input.  For batched computation over multiple samples use
    :func:`loo_cossim_many`.

    :param data: 2-D array of shape ``(replicates, features)`` with dtype
        ``float32`` or ``float64``.  Requires at least 2 replicates.
    :return: Scalar LOO cosine similarity as ``float64``.
    """
    return _loo_cossim_rust(data)


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


_supported_sortable_types = {
    "float32": "f32",
    "float64": "f64",
    "int8":    "i8",
    "int16":   "i16",
    "int32":   "i32",
    "int64":   "i64",
    "uint8":   "u8",
    "uint16":  "u16",
    "uint32":  "u32",
    "uint64":  "u64",
    "bool":    "bool",
}


def argsort(
    a: np.ndarray,
    *,
    stable: bool = False,
    num_threads: int | None = None,
) -> np.ndarray:
    """Return indices that sort `a` in ascending order (1D only).

    Differences from numpy.argsort:
    - 1D only — no ``axis`` parameter.
    - ``stable`` bool instead of ``kind`` string.
    - ``num_threads``: ``None`` → global Rayon pool (all CPUs); ``n`` → pool of n threads.
    - Always returns int64 (numpy returns intp).
    - NaN sorts to end for float dtypes (via total_cmp).
    """
    dtype_name = a.dtype.name
    suffix = _supported_sortable_types.get(dtype_name)
    if suffix is None:
        raise TypeError(f"argsort: unsupported dtype {a.dtype}. Supported: {tuple(_supported_sortable_types)}")
    fn = getattr(scors, f"argsort_{suffix}")
    return fn(a, stable=stable, num_threads=num_threads)


def sort_inplace(
    a: np.ndarray,
    *,
    stable: bool = False,
    num_threads: int | None = None,
) -> None:
    """Sort `a` in place in ascending order (1D only, C-contiguous).

    Differences from numpy.ndarray.sort:
    - 1D only — no ``axis`` parameter.
    - ``stable`` bool instead of ``kind`` string.
    - ``num_threads``: ``None`` → global Rayon pool (all CPUs); ``n`` → pool of n threads.
    - NaN sorts to end for float dtypes (via total_cmp).
    - Raises TypeError for non-C-contiguous arrays (use ``numpy.ascontiguousarray`` first).
    """
    dtype_name = a.dtype.name
    suffix = _supported_sortable_types.get(dtype_name)
    if suffix is None:
        raise TypeError(f"sort_inplace: unsupported dtype {a.dtype}. Supported: {tuple(_supported_sortable_types)}")
    fn = getattr(scors, f"sort_inplace_{suffix}")
    fn(a, stable=stable, num_threads=num_threads)


def sort(
    a: np.ndarray,
    *,
    stable: bool = False,
    num_threads: int | None = None,
) -> np.ndarray:
    """Return a sorted copy of `a` in ascending order (1D only).

    Differences from numpy.sort:
    - 1D only — no ``axis`` parameter.
    - ``stable`` bool instead of ``kind`` string.
    - ``num_threads``: ``None`` → global Rayon pool (all CPUs); ``n`` → pool of n threads.
    - NaN sorts to end for float dtypes (via total_cmp).
    """
    out = a.copy()
    sort_inplace(out, stable=stable, num_threads=num_threads)
    return out


def loo_cossim_many(data: np.ndarray) -> np.ndarray:
    """Batched leave-one-out cosine similarity.

    Computes :func:`loo_cossim` over the last two axes of *data*, treating all
    leading axes as independent batch dimensions.

    :param data: Array of shape ``(..., replicates, features)`` with dtype
        ``float32`` or ``float64``.  At least 2-D.
    :return: Array of shape ``data.shape[:-2]`` containing the LOO cosine
        similarity for each entry along the leading axes.
    """
    sim = _loo_cossim_many(np.reshape(data, (-1, *data.shape[-2:])))
    sim_reshaped = np.reshape(sim, data.shape[:-2])
    return sim_reshaped


def average_precision(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
    order: Order | None = None,
) -> float:
    """Compute Average Precision (area under the precision-recall curve).

    Re-implements :func:`sklearn.metrics.average_precision_score` for binary
    classification.  Differences from sklearn:

    - Parameter names differ: ``labels`` / ``predictions`` / ``weights``
      instead of ``y_true`` / ``y_score`` / ``sample_weight``.
    - ``order`` parameter skips the internal sort for pre-sorted data.
    - No input validation (no ``np.unique`` checks) — caller is responsible.
    - Output is always ``float64`` regardless of input dtype.

    :param labels: 1-D array of binary labels.  Supported dtypes: ``bool``,
        ``int8``–``int64``, ``uint8``–``uint64``.
    :param predictions: 1-D array of scores, same length as *labels*.
        Supported dtypes: ``float32``, ``float64``.  Must have the same dtype
        as *weights* when *weights* is provided.
    :param weights: Optional 1-D array of sample weights, same dtype and
        length as *predictions*.
    :param order: If ``None`` the data is sorted internally (descending by
        score).  Pass :attr:`Order.DESCENDING` or :attr:`Order.ASCENDING` to
        skip the sort when the data is already ordered.
    :return: Average Precision as a ``float64`` scalar.
    """
    return _from_generic_score("average_precision")(labels=labels, predictions=predictions, weights=weights, order=order)


def roc_auc(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
    order: Order | None = None,
    max_fpr: float | None = None,
) -> float:
    """Compute ROC-AUC, optionally truncated at a maximum false-positive rate.

    Re-implements :func:`sklearn.metrics.roc_auc_score` for binary
    classification.  When *max_fpr* is given the score is the normalised
    partial AUC following McClish (1989), scaled to ``[0, 1]`` so that 0.5
    corresponds to a random classifier and 1.0 to a perfect one.

    Differences from sklearn:

    - Parameter names differ: ``labels`` / ``predictions`` / ``weights``
      instead of ``y_true`` / ``y_score`` / ``sample_weight``.
    - ``order`` parameter skips the internal sort for pre-sorted data.
    - No input validation — caller is responsible.
    - Output is always ``float64`` regardless of input dtype.
    - ``max_fpr`` accepts ``float64``; sklearn accepts ``float``.

    :param labels: 1-D array of binary labels.  Supported dtypes: ``bool``,
        ``int8``–``int64``, ``uint8``–``uint64``.
    :param predictions: 1-D array of scores, same length as *labels*.
        Supported dtypes: ``float32``, ``float64``.  Must have the same dtype
        as *weights* when *weights* is provided.
    :param weights: Optional 1-D array of sample weights, same dtype and
        length as *predictions*.
    :param order: If ``None`` the data is sorted internally (descending by
        score).  Pass :attr:`Order.DESCENDING` or :attr:`Order.ASCENDING` to
        skip the sort when the data is already ordered.
    :param max_fpr: If given, truncate the ROC curve at this false-positive
        rate and return the normalised partial AUC.
    :return: ROC-AUC as a ``float64`` scalar.
    """
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
) -> float:
    """Compute Average Precision by merging two pre-sorted samples on the fly.

    Both samples must be sorted in **descending** order by their prediction
    scores.  The two sorted streams are merged without materialising a combined
    array, which avoids an O(n) allocation when scoring a large background
    sample against a small foreground sample.

    For workloads with multiple metric calls per pair (e.g. AP + ROC-AUC),
    separate :func:`average_precision` calls on the concatenated array are
    typically faster because the merge cost is paid once per metric.

    :param labels1: 1-D binary label array, sorted descending by *predictions1*.
    :param predictions1: 1-D score array, descending order.
    :param weights1: Optional sample weights for sample 1, same dtype as
        *predictions1*.
    :param labels2: 1-D binary label array, sorted descending by *predictions2*.
    :param predictions2: 1-D score array, descending order.
    :param weights2: Optional sample weights for sample 2, same dtype as
        *predictions2*.
    :return: Average Precision as a ``float64`` scalar.
    """
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
) -> float:
    """Compute ROC-AUC by merging two pre-sorted samples on the fly.

    Both samples must be sorted in **descending** order by their prediction
    scores.  See :func:`average_precision_on_two_sorted_samples` for the
    trade-off discussion on when to use the two-sample variant.

    :param labels1: 1-D binary label array, sorted descending by *predictions1*.
    :param predictions1: 1-D score array, descending order.
    :param weights1: Optional sample weights for sample 1, same dtype as
        *predictions1*.
    :param labels2: 1-D binary label array, sorted descending by *predictions2*.
    :param predictions2: 1-D score array, descending order.
    :param weights2: Optional sample weights for sample 2, same dtype as
        *predictions2*.
    :param max_fpr: If given, truncate the ROC curve at this false-positive
        rate and return the normalised partial AUC.
    :return: ROC-AUC as a ``float64`` scalar.
    """
    raise NotImplementedError()
    

__all__ = sorted([
    "Order",
    "argsort",
    "average_precision",
    "average_precision_on_two_sorted_samples",
    "loo_cossim",
    "loo_cossim_many",
    "roc_auc",
    "roc_auc_on_two_sorted_samples",
    "sort",
    "sort_inplace",
])
