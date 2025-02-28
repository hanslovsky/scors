import sys
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap_skl

from scors import average_precision as ap, Order

def timer(func, n) -> np.ndarray:
    dts = np.empty((n,), dtype=np.float64)
    for idx in range(n):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        dt = t1 - t0
        dts[idx] = dt
    return dts

X, y = load_breast_cancer(return_X_y=True)
X = np.concatenate([X] * 100)
y = np.concatenate([y] * 100)
print(f"{X.shape=} {y.shape}")
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
pred = clf.predict_proba(X)[:, 1]
pred = np.require(pred, dtype=np.float64)
y = np.require(y, dtype=np.uint8)
weights = np.ones_like(y, dtype=np.float64)

def run_ap_skl() -> float:
    return ap_skl(y, pred)

def run_full_sort() -> float:
    t0 = time.time()
    res = ap(y, pred, weights=weights)
    t1 = time.time()
    dt = t1 - t0
    return res

sort_indices = np.argsort(pred)[::-1]
y_sorted = y[sort_indices]
pred_sorted = pred[sort_indices]

def run_presorted() -> float:
    return ap(y_sorted, pred_sorted, weights=weights, order=Order.DESCENDING)

def run_np_presort() -> float:
    idx = np.argsort(y)[::-1]
    return ap(y[idx], pred[idx], weights=weights, order=Order.DESCENDING)

stop = int(np.floor(0.99 * y_sorted.size))
sort_indices_partial = np.argsort(pred[:stop])[::-1]
y_sorted_partial = y[sort_indices_partial]
pred_sorted_partial = pred[sort_indices_partial]

# def run_presorted_partial() -> float:
#     y_full = np.empty_like(y, dtype=np.uint8)
#     pred_full = np.empty_like()
    

sc_skl = run_ap_skl()
sc = run_full_sort()
sc_presorted = run_presorted()

assert np.isclose(sc_skl, sc), f"{sc_skl=} != {sc=}"
assert np.isclose(sc_skl, sc_presorted), f"{sc_skl=} != {sc_presorted=}"

# sys.exit(0)

def _argsort():
    idxs = np.argsort(y)[::-1]
    _y = y[idxs]
    _pred = pred[idxs]
n = 20
t_argsort = timer(_argsort, n=n)
print(f"np argsort:  {np.mean(t_argsort)} {np.std(t_argsort)}")
t_skl = timer(run_ap_skl, n=n)
print(f"skl:         {np.mean(t_skl)} {np.std(t_skl)}")
t_sort = timer(run_full_sort, n=n)
print(f"full sort:   {np.mean(t_sort)} {np.std(t_sort)}")
t_presorted = timer(run_presorted, n=n)
print(f"pre-sorted:  {np.mean(t_presorted)} {np.std(t_presorted)}")
t_np_presort = timer(run_np_presort, n=n)
print(f"np-pre-sort: {np.mean(t_np_presort)} {np.std(t_np_presort)}")

# y.dtype
# y.shape
# sc = ap(np.require(y, dtype=np.uint8), pred, weights=np.ones_like(y))
# pred.dtype
# sc = ap(np.require(y, dtype=np.uint8), pred, weights=np.ones_like(pred, dtype=np.float64)))
# sc = ap(np.require(y, dtype=np.uint8), pred, weights=np.ones_like(pred, dtype=np.float64))
# sc
# sc_skl
# np.isclose(sc, sc_skl)
# %timeit sc_skl = ap_skl(y, pred)
# %timeit sc = ap(np.require(y, dtype=np.uint8), pred, weights=np.ones_like(pred, dtype=np.float64))
# sort_indices = np.argsort(pred)
# y_sorted = y[sort_indices]
# pred_sorted = pred[sort_indices]
# %timeit sc_skl = ap_skl(y_sorted, pred_sorted)
# %timeit sc = ap(np.require(y_sorted, dtype=np.uint8), pred_sorted, weights=np.ones_like(pred, dtype=np.float64))
# %timeit sc = ap(np.require(y_sorted, dtype=np.uint8), pred_sorted, weights=np.ones_like(pred, dtype=np.float64), Order.ASCENDING)
# %timeit sc = ap(np.require(y_sorted, dtype=np.uint8), pred_sorted, weights=np.ones_like(pred, dtype=np.float64), order=Order.ASCENDING)
# y_sorted = y[sort_indices[::-1]]
# pred_sorted = pred[sort_indices[::-1]]
# %timeit sc = ap(np.require(y_sorted, dtype=np.uint8), pred_sorted, weights=np.ones_like(pred, dtype=np.float64), order=Order.ASCENDING)
# np.isclose(sc, sc_skl)
# %timeit sc = ap(np.require(y_sorted, dtype=np.uint8), pred_sorted, weights=np.ones_like(pred, dtype=np.float64), order=Order.DESCENDING)
# %timeit sc_skl = ap_skl(y_sorted, pred_sorted)
# np.isclose(sc, sc_skl)

