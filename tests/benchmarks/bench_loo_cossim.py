"""Benchmark loo_cossim_many: compare SIMD-optimised build vs baseline.

Run twice:
    # with the current (optimised) scors:
    uv run python tests/benchmarks/bench_loo_cossim.py

    # with the published release (e.g. 0.2.3):
    uv run --with scors==0.2.3 --no-project python tests/benchmarks/bench_loo_cossim.py

Shape: (1000 samples, 300 replicates, 500 features) — representative of the
mezzo-forte univariate metric call pattern.  Both float32 and float64 are
tested since the SIMD codegen differs between them.
"""

import time

import numpy as np
import scors

SAMPLES     = 1_000
REPLICATES  = 300
FEATURES    = 500
SEED        = 42
REPEATS     = 5


def bench(data: np.ndarray, label: str) -> None:
    # warm-up
    scors.loo_cossim_many(data)

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        scors.loo_cossim_many(data)
        times.append(time.perf_counter() - t0)

    best = min(times)
    mean = sum(times) / len(times)
    print(f"  {label:<12}  best={best*1e3:7.1f} ms   mean={mean*1e3:7.1f} ms   ({REPEATS} runs)")


def main() -> None:
    version = getattr(scors, "__version__", "dev")
    print(f"scors {version}")
    print(f"shape: ({SAMPLES}, {REPLICATES}, {FEATURES})\n")

    rng = np.random.default_rng(SEED)

    for dtype in (np.float32, np.float64):
        # Case 1: fully C-contiguous 3D array
        data = rng.standard_normal((SAMPLES, REPLICATES, FEATURES)).astype(dtype)
        assert data.flags["C_CONTIGUOUS"]
        bench(data, f"{dtype.__name__}")

        # Case 2: slice along axis 1 — 3D array is NOT C-contiguous (axis-0
        # stride is 321*500, not 300*500), but each 2D slice yielded by
        # outer_iter() has strides (500, 1) and IS C-contiguous.
        # So the SIMD fast path still fires inside loo_cossim — same speed.
        data_big = rng.standard_normal((SAMPLES, REPLICATES + 21, FEATURES)).astype(dtype)
        data_sliced = data_big[:, :REPLICATES, :]
        assert not data_sliced.flags["C_CONTIGUOUS"], "3D array should not be C-contiguous after slice"
        bench(data_sliced, f"{dtype.__name__} (:300)")

        # Case 3: Fortran-order — feature axis is non-unit stride, so
        # as_slice() returns None for every 2D slice → strided fallback.
        data_f = np.asfortranarray(data)
        bench(data_f, f"{dtype.__name__} (F-order)")

    print()
    print("Expected: C-contiguous ≈ :300 slice (both hit SIMD fast path)")
    print("          F-order is slower (strided fallback, no SIMD on scoring loop)")


if __name__ == "__main__":
    main()
