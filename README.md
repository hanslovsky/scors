![Build Status](https://github.com/hanslovsky/scors/actions/workflows/CI.yml/badge.svg)
![Crates.io Version](https://img.shields.io/crates/v/scors)
![PyPI - Version](https://img.shields.io/pypi/v/scors)

# Scors

Scors is a Rust re-implementation with Python bindings of selected
[binary classification scores from scikit-learn](https://scikit-learn.org/stable/api/sklearn.metrics.html),
plus parallel sort utilities.  All computation releases the GIL and uses
[Rayon](https://github.com/rayon-rs/rayon) for parallelism where beneficial.

## Scores

Scors implements a subset of sklearn's metrics, restricted to **binary
classification** only.  Parameter names differ slightly:

| sklearn           | scors         |
|-------------------|---------------|
| `y_true`          | `labels`      |
| `y_score`         | `predictions` |
| `sample_weight`   | `weights`     |

| sklearn function           | scors function      |
|----------------------------|---------------------|
| `average_precision_score`  | `average_precision` |
| `roc_auc_score`            | `roc_auc`           |

### Differences from sklearn

- **`order` parameter** — pass `Order.DESCENDING` / `Order.ASCENDING` to skip
  the internal sort when data is already ordered.  Default `None` sorts
  internally.
- **No data validation** — scors places responsibility for valid inputs on the
  caller (no `np.unique` checks, no label type coercion).  This avoids
  overhead that matters when calling the same metric thousands of times.
- **No multi-class support** — binary labels only.
- **Output is always `float64`** regardless of input dtype.
- **Parallel sort** — the internal argsort uses Rayon's global thread pool by
  default, giving ~10× speedup over sequential sort at 1M elements on
  multi-core hardware.

### Scores on two sorted samples

For workloads that combine a large sorted background sample with many small
foreground samples, scors exposes:

- `average_precision_on_two_sorted_samples`
- `roc_auc_on_two_sorted_samples`

These merge the two sorted iterators on the fly without materialising a
combined array.  When there is only one metric call per pair this is faster
than concatenating; for multiple metrics per pair, separate sorted passes
are typically faster.

## Sort utilities

scors also exposes parallel sort as a first-class API — useful as a drop-in
replacement for `parallel-sort` (which has limited platform/version coverage):

| function         | description                                      |
|------------------|--------------------------------------------------|
| `argsort`        | Returns indices that sort `a` ascending (int64)  |
| `sort`           | Returns a sorted copy                            |
| `sort_inplace`   | Sorts in place, returns `None`                   |

All three accept `stable: bool = False` and `num_threads: int | None = None`
(`None` → global Rayon pool).

### Differences from `numpy.argsort` / `numpy.sort`

- **1D only** — no `axis` parameter.
- **`stable` bool** instead of `kind` string.
- **`num_threads`** for explicit parallelism control.
- **NaN sorts to end** for ascending; for descending NaN appears first (unlike
  numpy).  If your data may contain NaN use numpy directly.
- **Always returns `int64`** (`numpy.argsort` returns `intp`, platform-dependent).

## LOO cosine similarity

`loo_cossim` and `loo_cossim_many` compute leave-one-out cosine similarity
over replicate matrices.  These have no sklearn equivalent.  The inner loops
are SIMD-vectorized (NEON on ARM, SSE2 on x86-64) for C-contiguous input.

## Benchmarks

At 1M elements on Apple M-series (multi-core):

| path                  | time     | vs baseline |
|-----------------------|----------|-------------|
| Sorted (no argsort)   | ~40 ms   | —           |
| Unsorted, sequential  | ~1950 ms | 1×          |
| Unsorted, parallel    | ~200 ms  | **~10×**    |

`loo_cossim_many` (1000 × 300 × 500, f32):

| layout       | time    |
|--------------|---------|
| C-contiguous | ~84 ms  |
| F-order      | ~520 ms |

## Installation

```
pip install scors
```

Wheels are published for Python 3.11–3.14 on Linux x86-64/aarch64,
macOS ARM64 and Intel, and Windows x64.
