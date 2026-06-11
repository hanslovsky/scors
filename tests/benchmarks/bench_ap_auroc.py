"""Benchmark AP and AUROC: sorted vs unsorted paths, f32 vs f64 inputs.

Covers two distinct code paths:
  - Sorted (order=DESCENDING): no argsort, pure scoring loop — memory-bound.
  - Unsorted (order=None):     argsort + select copies + scoring loop.
    This is where parallel argsort will show improvement.

Run:
    uv run python tests/benchmarks/bench_ap_auroc.py

Shape: 10M elements. Both f32 and f64 inputs tested.
"""

import time

import numpy as np
import scors

N         = 1_000_000
SEED      = 42
REPEATS   = 3
LABEL_POS = 0.1  # 10% positive rate


def bench(fn, *args, label: str, **kwargs) -> float:
    fn(*args, **kwargs)  # warm-up
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    best = min(times)
    mean = sum(times) / len(times)
    print(f"  {label:<44}  best={best*1e3:7.1f} ms   mean={mean*1e3:7.1f} ms")
    return best


def main() -> None:
    rng = np.random.default_rng(SEED)

    scores_f64 = rng.random(N).astype(np.float64)
    scores_f32 = scores_f64.astype(np.float32)
    y_true = (rng.random(N) < LABEL_POS)  # bool, unsorted

    # pre-sorted versions (descending) for the sorted-path benchmarks
    idx = np.argsort(scores_f64)[::-1]
    scores_f64_sorted = np.ascontiguousarray(scores_f64[idx])
    scores_f32_sorted = np.ascontiguousarray(scores_f32[idx])
    y_true_sorted = np.ascontiguousarray(y_true[idx])

    print(f"scors {getattr(scors, '__version__', 'dev')}")
    print(f"N={N:,}  pos_rate={LABEL_POS:.0%}  repeats={REPEATS}\n")

    # ── Sorted path (no argsort) ──────────────────────────────────────────────
    print("── sorted path (order=DESCENDING, no argsort) ──")

    print("Average Precision:")
    bench(scors.average_precision, y_true_sorted, scores_f64_sorted,
          order=scors.Order.DESCENDING, label="f64")
    bench(scors.average_precision, y_true_sorted, scores_f32_sorted,
          order=scors.Order.DESCENDING, label="f32")

    print("ROC-AUC:")
    bench(scors.roc_auc, y_true_sorted, scores_f64_sorted,
          order=scors.Order.DESCENDING, label="f64")
    bench(scors.roc_auc, y_true_sorted, scores_f32_sorted,
          order=scors.Order.DESCENDING, label="f32")

    print("ROC-AUC max_fpr=0.1:")
    bench(scors.roc_auc, y_true_sorted, scores_f64_sorted,
          order=scors.Order.DESCENDING, max_fpr=0.1, label="f64")
    bench(scors.roc_auc, y_true_sorted, scores_f32_sorted,
          order=scors.Order.DESCENDING, max_fpr=0.1, label="f32")

    # ── Unsorted path (argsort fires) ────────────────────────────────────────
    print("\n── unsorted path (order=None, argsort + select + score) ──")
    print("Note: this is where parallel argsort will improve performance.\n")

    print("Average Precision:")
    bench(scors.average_precision, y_true, scores_f64,
          label="f64")
    bench(scors.average_precision, y_true, scores_f32,
          label="f32")

    print("ROC-AUC:")
    bench(scors.roc_auc, y_true, scores_f64,
          label="f64")
    bench(scors.roc_auc, y_true, scores_f32,
          label="f32")

    print("ROC-AUC max_fpr=0.1:")
    bench(scors.roc_auc, y_true, scores_f64,
          max_fpr=0.1, label="f64")
    bench(scors.roc_auc, y_true, scores_f32,
          max_fpr=0.1, label="f32")


if __name__ == "__main__":
    main()
