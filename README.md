![Build Status](https://github.com/hanslovsky/scors/actions/workflows/CI.yml/badge.svg)
![Crates.io Version](https://img.shields.io/crates/v/scors)
![PyPI - Version](https://img.shields.io/pypi/v/scors)

# Scors

This package is a Rust re-implementation with Python bindings of some of the [classification scores from scikit-learn](https://scikit-learn.org/stable/api/sklearn.metrics.html) (sklearn),
restricted to binary classification only. Scores generally have 3 input parameters for labels, predictions, and weights, with slightly different names in `sklearn`:

| **sklearn**     | **scors**     |
| ----------------| --------------|
| `y_true`        | `labels`      |
| `y_score`       | `predictions` |
| `sample_weight` | `weights`     |

Functions in `scors` have an additional parameter `order` that can be 
 1. *(default)* `None` to indicate unsorted data,
 2. `Order.ASCENDING` to indicate that the input data is sorted in ascending order wrt `predictions`, or
 3. `Order.DESCENDING` to indicate that the input data is sorted in descending order wrt `predictions`.
 
Other parameters that may be present (e.g. `max_fprs` in `roc_auc`) follow the naming and meaning as defined in the respective sklearn counterpart

## Why?

I want to improve runtime performance of scores for my use case. I have a single large background sample that I combine and score with each of many small foreground smaples.
For the rank-based metrics (e.g. [`average_precision-score`_](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)),
the data is sorted, which has complexity `n*log(n)`. 
Exploiting the structure of my data helps me avoid this cost to boost performance.
But even without assumptions about structure in the data, I found ways to improve performance. 
This is a summary of all the optimizations I implemented (or plan to):

 1. Add option to assume the data is sorted. This allows the caller to exploit structure in data that is already sorted/mostly sorted.
 2. Remove checks on labels. When stepping through the debugger, I noticed that the sklearn imlementation uses safeguards like [`np.unique`](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) to check the validity of the data.
    This can be helpful to ensure that assumptions are always met, especially in a library a huge audience and general scope like sklearn.
    But it also has a performance penalty.
    I decided, to place the responsibility for data validation completely on the caller.
    The caller can add or leave out data validation as appropriate
 3. Minimize data allocation:
    1. All current scores are implemented as single pass over the data (double pass in case of ROC AUC **with** max fprs).
    2  For ordered input data, no allocations are made.
    3. If the optional weights parameter is not provided, no extra constant array filled with `1` is created. Instead, the Rust implementation uses a constant value iterator.
    4. **TODO**: For unordered input data, I currently create sorted copies of all of the input data. That is a total of 4 (3 if weights are not provided) extra allocations for
       the index array (for sorting), labels, predictions, and weights. Instead of creating copies of the input arrays, I consider creating index views that simply index
       into the original arrays through the sorted index array. This may provide another performance benefit, but I still have to benchmark this.

## Currently Implemented Scores

| **sklearn**               | **scors**           |
| ------------------------- | ------------------- |
| `average_precision_score` | `average_precision` |
| `roc_auc_score`           | `roc_auc`           |

## Is It Actually Faster?

**TODO**: Benchmarks

