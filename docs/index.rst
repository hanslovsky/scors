Scors
=====

Scors provides fast binary classifier evaluation scores implemented in Rust
with Python bindings via PyO3.  All heavy computation releases the GIL and
uses Rayon for parallelism where beneficial.

**Scores**

- :func:`~scors.average_precision` — Average Precision (area under the
  precision-recall curve).
- :func:`~scors.roc_auc` — ROC-AUC, optionally truncated at a maximum FPR.

**Sorting**

- :func:`~scors.argsort` — Parallel argsort, ascending (returns ``int64``).
- :func:`~scors.sort` — Parallel sort, returns a sorted copy.
- :func:`~scors.sort_inplace` — Parallel in-place sort.

**Utilities**

- :func:`~scors.loo_cossim` — Leave-one-out cosine similarity for a matrix
  of replicates.
- :func:`~scors.loo_cossim_many` — Batched leave-one-out cosine similarity.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
