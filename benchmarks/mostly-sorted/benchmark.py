import pandas as pd
import time
from functools import partial
from typing import Callable, TypeVar

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas
from scors import average_precision, roc_auc
from sklearn.metrics import average_precision_score, roc_auc_score

T = TypeVar("T")

def timeit(func: Callable[[], T], n: int, n_warmup: int):
    result = None
    for _ in range(n_warmup):
        result = func()

    dts = np.empty((n,), dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        r = func()
        t1 = time.perf_counter()
        result = r
        dts[i] = t1 - t0
    return result, dts


@numba.njit(nogil=True)
def sort_combine_weighted(
    
):


if __name__ == "__main__":
    ns = 10, 100, 1_000, 10_000, 100_000, 1_000_000
    repeats = 100
    n_warmup = 10
    records = []
    rng = np.random.default_rng(42)
    labels = rng.random(max(ns), dtype=np.float64) > 0.7
    predictions = rng.random(max(ns), dtype=np.float64)
    weights = rng.random(max(ns), dtype=np.float64)

    benchmark_data_fn = "statistics.parquet"

    try:
        df = pd.read_parquet(benchmark_data_fn)
    except FileNotFoundError:
        for n in ns[:]:
            print(f"{n=}")
            l = labels[:n]
            p = predictions[:n]
            for w in (None, weights):
                use_weights = w is not None
                w = w if w is None else w[:n]
                record = dict(n=n, n_warmup=n_warmup, repeats=repeats, weights=use_weights)
                skl_kwargs = dict(y_true=l, y_score=p, sample_weight=w)
                sc_kwargs = dict(labels=l, predictions=p, weights=w, order=None)
                ap_skl, dt_ap_skl = timeit(partial(average_precision_score, **skl_kwargs), n=repeats, n_warmup=n_warmup)
                ap_sc, dt_ap_sc = timeit(partial(average_precision, **sc_kwargs), n=repeats, n_warmup=n_warmup)
                assert np.isclose(ap_skl, ap_sc)
                auc_skl, dt_auc_skl = timeit(partial(roc_auc_score, **skl_kwargs), n=repeats, n_warmup=n_warmup)
                auc_sc, dt_auc_sc = timeit(partial(roc_auc, **sc_kwargs), n=repeats, n_warmup=n_warmup)
                assert np.isclose(auc_skl, auc_sc)
                records.append(record | dict(metric="AP", package="sklearn", dt_mean=np.mean(dt_ap_skl), dt_std=np.std(dt_ap_skl), value=ap_skl))
                records.append(record | dict(metric="AP", package="scors", dt_mean=np.mean(dt_ap_sc), dt_std=np.std(dt_ap_sc), value=ap_sc))
                records.append(record | dict(metric="AUROC", package="sklearn", dt_mean=np.mean(dt_auc_skl), dt_std=np.std(dt_auc_skl), value=auc_skl))
                records.append(record | dict(metric="AUROC", package="scors", dt_mean=np.mean(dt_auc_sc), dt_std=np.std(dt_auc_sc), value=auc_sc))


        df = pd.DataFrame.from_records(records)
        df.to_parquet(benchmark_data_fn, index=False)

    for (metric, use_weights), gdf in df.groupby(["metric", "weights"]):
        use_weights = bool(use_weights)
        gdf = gdf.sort_values(by="n")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"{metric} {use_weights=}")
        ref = gdf[gdf.package=="sklearn"].dt_mean.to_numpy()
        for p, pdf in gdf.groupby("package"):
            ax1.errorbar(pdf.n, pdf.dt_mean, yerr=pdf.dt_std, label=p, ls="none", marker="d")
            ax2.plot(pdf.n, pdf.dt_mean.to_numpy() / ref, label=p, ls="none", marker="d")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax2.set_xscale("log")
        ax1.legend()
        ax2.legend()
        ax1.set_ylabel("Mean Runtime [s]")
        ax1.set_xlabel("Sample Size")
        ax2.set_ylabel("Mean Runtime / Mean Runtime sklearn")
        ax2.set_xlabel("Sample Size")

    plt.show()
    
