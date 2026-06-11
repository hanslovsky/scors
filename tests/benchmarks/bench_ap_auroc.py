"""Benchmark AP and AUROC: compare f32 vs f64 accumulation.

The refactoring plan fixes output to f64 regardless of input dtype.
This measures the cost of widening f32 inputs to f64 for accumulation.

Run:
    uv run python tests/benchmarks/bench_ap_auroc.py

Shape: 10M elements, matching the existing two-sorted-samples benchmark scale.
Both sorted (pre-sorted input, order=DESCENDING) and unsorted paths are tested.
"""

import time

import numpy as np
import scors

N         = 10_000_000
SEED      = 42
REPEATS   = 5
LABEL_POS = 0.1  # 10% positive rate


def bench(fn, *args, label: str, **kwargs) -> float:
    # warm-up
    fn(*args, **kwargs)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    best = min(times)
    mean = sum(times) / len(times)
    print(f"  {label:<40}  best={best*1e3:7.1f} ms   mean={mean*1e3:7.1f} ms")
    return best


def main() -> None:
    rng = np.random.default_rng(SEED)

    y_true = rng.random(N) < LABEL_POS  # bool labels

    # Pre-sort descending so we can use order=DESCENDING (skip argsort cost)
    scores_f64 = rng.random(N)
    idx = np.argsort(scores_f64)[::-1]
    scores_f64 = scores_f64[idx].astype(np.float64)
    scores_f32 = scores_f64.astype(np.float32)
    y_true_sorted = y_true[idx]

    print(f"scors {getattr(scors, '__version__', 'dev')}")
    print(f"N={N:,}  pos_rate={LABEL_POS:.0%}  repeats={REPEATS}\n")

    print("Average Precision (pre-sorted, order=DESCENDING):")
    bench(scors.average_precision, y_true_sorted, scores_f64,
          order=scors.Order.DESCENDING, label="f64 input → f64 accum")
    bench(scors.average_precision, y_true_sorted, scores_f32,
          order=scors.Order.DESCENDING, label="f32 input → f32 accum (current)")

    print("\nROC-AUC (pre-sorted, order=DESCENDING):")
    bench(scors.roc_auc, y_true_sorted, scores_f64,
          order=scors.Order.DESCENDING, label="f64 input → f64 accum")
    bench(scors.roc_auc, y_true_sorted, scores_f32,
          order=scors.Order.DESCENDING, label="f32 input → f32 accum (current)")

    print("\nROC-AUC max_fpr=0.1 (pre-sorted, order=DESCENDING):")
    bench(scors.roc_auc, y_true_sorted, scores_f64,
          order=scors.Order.DESCENDING, max_fpr=0.1, label="f64 input → f64 accum")
    bench(scors.roc_auc, y_true_sorted, scores_f32,
          order=scors.Order.DESCENDING, max_fpr=0.1, label="f32 input → f32 accum (current)")

    print()
    print("Note: after refactoring, f32 input will widen to f64 accumulators.")
    print("      The f64 rows above show the post-refactor cost for f32 inputs.")


if __name__ == "__main__":
    main()
