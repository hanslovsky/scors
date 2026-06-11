# Scors Development Guidelines

## Build & Workflow

- `make develop` — debug build in-place (fast, for iteration)
- `make develop-release` — release build in-place (needed for accurate benchmarks)
- `make test` — run the full test suite
- `make bench` — build release extension then run all benchmarks
- `uv sync` — install deps + build (used by CI; equivalent to `make develop`)
- Do NOT use `maturin develop` directly — it can't find `cargo` unless rustup's
  `~/.cargo/bin` is on PATH; `uv sync` / `make develop` are the reliable entry points
- Rust toolchain: **stable** (nightly is not required)

## Architecture

- Pure Rust extension via PyO3 + maturin. Python wrapper in `src/scors/__init__.py`.
- Two scores: **Average Precision** and **ROC-AUC** (with optional `max_fpr`).
- One utility: **LOO cosine similarity** (`loo_cossim`, `loo_cossim_many`).
- Sort utilities: **`argsort`**, **`sort`**, **`sort_inplace`** — parallel via Rayon.
- The Python layer dispatches to typed Rust functions via a macro-generated matrix
  of label × prediction type combinations (9 label types × 2 float types = 18 per metric).

## Rust Code Structure (`rust/src/lib.rs`)

- `IntoF64: Into<f64> + Copy + PartialOrd` — bound for score/weight inputs. Output is
  always `f64`; `ScoreAccumulator` and `IntoScore<S>` were removed in the generics cleanup.
- `TotalCmp` — local trait for total ordering: floats use `f32/f64::total_cmp` (NaN sorts
  last), integers/bool use `Ord::cmp`. Used by `do_argsort`, `do_sort`, `SortableData`.
- `ScoreSortedDescending` — core scoring protocol. `_score` takes `(f64, (bool, f64))` items;
  `score` converts generic `(P, (B, W))` tuples via `IntoF64`.
- `Positives` — fixed-`f64` accumulator for TPs/FPs; used by AP and AUROC.
- `ConstWeight` — fixed-`f64` infinite iterator of 1.0; used for the no-weights path.
  No longer generic — `W::one()` was eliminated when `W` type param was dropped.
- `CombineIterDescending` (in `combine.rs`) — merge-sort merge of two sorted-descending
  iterators; used by `score_two_sorted_samples`.
- `PyScoreGeneric` trait — bridges Rust generics to PyO3. `W` type param was dropped;
  weights are converted to `f64` iterators at the Python boundary.
- `SortableData<T: TotalCmp + Clone>: Data<T> + Sync` — default `argsort_unstable`
  uses `as_contiguous_slice()` fast path (LLVM-optimized) with `get_at` fallback for
  strided data. No copies needed in either path — Rayon sorts the index Vec.
- `do_argsort` / `do_sort` — pool-aware helpers. `None` → global Rayon pool (parallel);
  `Some(pool)` → sized pool. `stable` flag selects stable vs unstable sort.
- All Python-facing computation releases the GIL via `py.detach()`.

## Generics Design

Outputs are always `f64` for AP/AUROC — this eliminated `ScoreAccumulator`, `IntoScore<S>`,
and the `SA` type parameter from every scoring function. The reduction:
- `Positives<P>` → `Positives` (fixed f64)
- `AveragePrecision{}` / `RocAuc{}` → unit structs `AveragePrecision;` / `RocAuc;`
- `ConstWeight<F>` → `ConstWeight` (fixed f64, `W::one()` gone)
- `score_py` `W` type param dropped — weights are `Vec<f64>` converted at Python boundary

`loo_cossim` stays generic over `F: num::Float + AddAssign` — fixing to f64 would halve
SIMD throughput for f32 inputs (4×f32 vs 2×f64 per NEON/SSE2 instruction).

## Parallel Sort (Rayon)

`argsort_unstable(None)` uses Rayon's global pool by default (parallel). `score_maybe_sorted_sample`
passes `None` — the internal argsort is parallel automatically. The Python `argsort`/`sort`
functions expose `num_threads: int | None = None` where `None` = global pool, `n` = sized pool.

Measured speedup on Apple M-series at 1M elements:
- Sequential baseline: ~1950ms (unsorted AP/AUROC path)
- With global Rayon pool: ~200ms (**9.7× speedup**)

The `as_contiguous_slice()` dispatch in `SortableData` is critical: `get_at`-based comparators
in `do_argsort` are not slow for random-access sort (the bottleneck is the algorithm, not
element access), so strided arrays work correctly without copies. Contiguous arrays additionally
benefit from LLVM optimizing the comparator to a plain slice index.

## SIMD Optimization in `loo_cossim`

- **Accumulation loop** (building `replicate_sum`): auto-vectorizes via NEON/SSE2.
- **Scoring loop** (`prod_sum`, `m_sqs`, `l_sqs`): vectorizes only with contiguous access.
  Runtime dispatch via `mat.as_slice()`:

```rust
if let Some(mat_slice) = mat.as_slice() {
    loo_cossim_loops(mat_slice.chunks(ncols).map(|s| s.iter().copied()), ...)
} else {
    loo_cossim_loops(mat.outer_iter().map(|r| r.into_iter().copied()), ...)
}
```

`impl Fn(&ArrayView1) -> I` was considered but rejected: `as_slice()` ties its `&[F]` return
to `&self` (the view reference lifetime), causing a use-after-free. The iterator solution
(`chunks()` / `into_iter()`) transfers the data lifetime correctly.

In the primary mezzo-forte call path (`loo_cossim_many` on `sub[:, :n, :]`), each 2D slice
has strides `(ncols, 1)` — fully C-contiguous — so the fast path always fires.

Benchmarks (Apple M-series, 1000 samples × 300 replicates × 500 features):
- f32 C-contiguous: ~84ms | f32 F-order (strided fallback): ~520ms

## AVX-512 (future)

CI wheels target generic x86-64 baseline (SSE2). The SLURM cluster has Intel Xeon 6975P-C
(Granite Rapids) with AVX-512 — 8 doubles per instruction vs 2 with SSE2. Options:

- **Build on cluster**: set `RUSTFLAGS=-C target-cpu=native` in the opsflow env.
- **Runtime dispatch** (preferred): `is_x86_feature_detected!("avx512f")` +
  `#[target_feature(enable="avx512f")]` — one wheel, automatic on AVX-512 hardware.
- PyPI has no microarchitecture tag slot; two-wheel distribution requires a custom index.

## Dependency Notes

- `ipython` 9.x requires Python ≥3.11 → `requires-python = ">=3.11"`.
- `requires-python` upper bound tracks NumPy: currently `<3.15` (NumPy 2.4.x).
  When **NumPy 2.5.0** releases: (1) relax to `<3.16`, add 3.15 to CI; (2) bump lower
  bound to `>=3.12` (NumPy 2.5 drops 3.11); (3) bump `numpy>=2.5.0` in dev deps.
- `numba` (benchmarks only) has no macOS x86-64 wheels → gated with
  `sys_platform != 'darwin' or platform_machine != 'x86_64'`.
- `rayon = "1.12"` — minimum version tested. `"1"` would work but is less explicit.

## PyO3 Migration Notes (0.23 → 0.28)

- `allow_threads` → `detach`
- `downcast::<T>()` → `cast::<T>()`
- `#[pyclass(eq, eq_int)]` → add `from_py_object` for `Clone` pyclasses
- `#![feature(trait_alias)]` removed; replaced with supertrait bounds + blanket impls
- `GILProtected` compile error with Python 3.13 came from `numpy` (rust-numpy) 0.23.0,
  not scors itself — fixed by upgrading to rust-numpy 0.28

## argsort / sort API

Differences from numpy:
- 1D only (no `axis`)
- `stable: bool = False` instead of `kind` string
- `num_threads: int | None = None` (None → global Rayon pool)
- `argsort` always returns `int64` (numpy returns `intp`, platform-dependent)
- NaN sorts to end for ascending; for **descending NaN appears first** (unlike numpy) —
  documented difference, callers with NaN should use numpy directly
- No `order` parameter (no structured arrays)

`sort` returns a copy; `sort_inplace` modifies in place and returns `None` (like `list.sort()`).
`sort_inplace` requires C-contiguous input.

## Known Design Notes

- `score_two_sorted_samples` is not used in mezzo-forte: for multiple metrics (AP + AUROC +
  multiple `max_fpr`), separate sorted passes are faster than the merge iterator.
- `ScoreSortedDescending::_score` requires `Clone` on the iterator but only
  `RocAucWithMaxFPR` actually calls `.clone()` on it.
- `score_maybe_sorted_sample` passes `None` to `argsort_unstable` — the pool parameter
  is there as a hook for future caller-controlled parallelism, not yet wired to Python.
