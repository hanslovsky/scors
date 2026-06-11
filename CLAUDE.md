# Scors Development Guidelines

## Build & Workflow

- `uv sync` — install dependencies and build the Rust extension in-place (uses maturin internally)
- `uv run pytest` — run the full test suite (791 tests)
- `uv run pytest tests/test_loo_cossim.py` — run only the LOO cosine similarity tests
- Do NOT use `maturin develop` directly — it can't find `cargo` unless rustup's `~/.cargo/bin` is on PATH; `uv sync` is the reliable entry point
- Rust toolchain: **stable** (nightly is not required; `#![feature(trait_alias)]` was removed)

## Architecture

- Pure Rust extension via PyO3 + maturin. Python wrapper in `src/scors/__init__.py`.
- Two scores: **Average Precision** and **ROC-AUC** (with optional `max_fpr`).
- One utility: **LOO cosine similarity** (`loo_cossim`, `loo_cossim_many`).
- The Python layer dispatches to typed Rust functions (e.g. `average_precision_bool_f32`) via a macro-generated matrix of label × prediction × weight type combinations.

## Rust Code Structure (`rust/src/lib.rs`)

- `ScoreAccumulator` / `IntoScore<S>` — supertrait bounds (converted from nightly trait aliases in the PyO3 0.28 upgrade; blanket impls cover all numeric types automatically).
- `ScoreSortedDescending` — core scoring protocol. `_score` takes typed `(S, (bool, S))` items; `score` converts generic `(P, (B, W))` tuples.
- `Positives<P>` — accumulates TPs and FPs; used by both AP and AUROC implementations.
- `ConstWeight<F>` — infinite iterator of a constant value; used when no weights are supplied.
- `CombineIterDescending` (in `combine.rs`) — merge-sort merge of two sorted-descending iterators; used by `score_two_sorted_samples`.
- `PyScoreGeneric` trait — bridges Rust generics to PyO3; `score_py` and `score_two_sorted_samples_py_generic` are the two Python-facing entry points.
- All Python-facing computation releases the GIL via `py.detach()`.

## SIMD Optimization in `loo_cossim`

The two inner loops in `loo_cossim` have different vectorization behaviour:

- **Accumulation loop** (building `replicate_sum`): LLVM auto-vectorizes via NEON/SSE2 from the first version.
- **Scoring loop** (`prod_sum`, `m_sqs`, `l_sqs`): only vectorizes when the input is proven contiguous. Uses a runtime dispatch:

```rust
if let Some(mat_slice) = mat.as_slice() {
    // contiguous path: chunks() → &[F] → LLVM emits SIMD
} else {
    // strided path: outer_iter() → into_iter() → scalar fallback
}
```

`mat.as_slice()` succeeds iff the 2D array is fully C-contiguous (strides == `[ncols, 1]`). In the primary call path from mezzo-forte (`loo_cossim_many` on `sub[:, :n, :]`), each 2D slice has strides `(ncols, 1)` — fully C-contiguous — so the fast path always fires.

`impl Fn(&ArrayView1) -> I` was considered but rejected: `as_slice()` ties its `&[F]` return to `&self` (the view reference, not the data lifetime), causing a use-after-free in the closure. The iterator solution (`chunks()` + `into_iter()`) sidesteps this correctly.

The shared helper is `loo_cossim_loops<F, Row>(rows: impl Iterator<Item=Row> + Clone, ...)`. The `Clone` on the outer iterator enables two-pass iteration without allocation (clone cost ≈ 24 bytes copied on the stack).

## AVX-512 (future)

The manylinux CI wheels target generic x86-64 baseline (SSE2). The SLURM cluster has Intel Xeon 6975P-C (Granite Rapids) with AVX-512 — 8 doubles per instruction vs 2 with SSE2. To enable:

- **Build on cluster**: set `RUSTFLAGS=-C target-cpu=native` in the opsflow environment build.
- **Runtime dispatch** (preferred long-term): use `is_x86_feature_detected!("avx512f")` + `#[target_feature(enable="avx512f")]` to select the fast path at import time — one wheel, automatic on AVX-512 hardware.
- PyPI has no microarchitecture tag slot, so two-wheel PyPI distribution is not straightforward.

## PyO3 Migration Notes (0.23 → 0.28)

- `allow_threads` → `detach`
- `downcast::<T>()` → `cast::<T>()`
- `#[pyclass(eq, eq_int)]` → add `from_py_object` to preserve `FromPyObject` derive for `Clone` pyclasses
- `#![feature(trait_alias)]` removed; replaced with explicit supertrait bounds + blanket impls
- `rust-toolchain.toml` not needed (stable suffices); nightly was only required for trait aliases
- `GILProtected` compile error with Python 3.13 came from `numpy` (rust-numpy) 0.23.0, not scors itself — fixed by upgrading to rust-numpy 0.28

## Dependency Notes

- `numba` is a dev dependency used only for benchmarks (`tests/benchmarks/`). It has no cp315 wheel, so it is gated with `python_version < '3.15'` — benchmarks are simply unavailable on Python 3.15+.
- `ipython` 9.x requires Python ≥3.11, which is why `requires-python = ">=3.11"` (Python 3.10 support was dropped when bumping to ipython 9.x).

## Known Design Notes

- `score_two_sorted_samples` is not used in mezzo-forte: for multiple metrics (AP + AUROC + multiple `max_fpr`), separate sorted passes are faster than the merge iterator because the merge cost is paid once per metric call. The merge approach avoids copying the similarity matrix but is only a win for exactly one metric.
- `SortableData` is only implemented for `f64` on `Vec`/`&[T]`/arrays, but generically for `ArrayView`. This asymmetry is a trap if `Vec<f32>` is ever passed to `score_sample`.
- `B2` type parameter in `score_two_sorted_samples_py_generic` is declared but `labels2` uses `B1` — effectively unused.
- `ScoreSortedDescending::_score` requires `Clone` on the iterator but only `RocAucWithMaxFPR` uses it.
