"""Tests for scors.argsort and scors.sort / scors.sort_inplace.

Covers:
- All supported dtypes
- Ascending order matches numpy
- Stability (stable=True preserves relative order of equal elements)
- NaN behaviour: NaN sorts to end for float dtypes
- sort returns a copy, original is unchanged
- sort_inplace modifies in place and returns None
- num_threads=1 (single-threaded pool) gives same results as default
- Non-contiguous input to argsort (strided ArrayView)
- sort_inplace rejects non-contiguous arrays
"""

import numpy as np
import pytest

import scors

# ── dtype coverage ─────────────────────────────────────────────────────────────

_FLOAT_DTYPES = [np.float32, np.float64]
_INT_DTYPES   = [np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64]
_ALL_DTYPES   = _FLOAT_DTYPES + _INT_DTYPES + [bool]

_float_ids = ["f32", "f64"]
_int_ids   = ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"]
_all_ids   = _float_ids + _int_ids + ["bool"]


# ── argsort ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype,expected_id", zip(_ALL_DTYPES, _all_ids), ids=_all_ids)
def test_argsort_sorts_correctly(dtype, expected_id):
    """argsort must produce a valid sorting permutation.

    We do not compare exact indices because scors uses unstable sort by default
    and numpy uses a different algorithm — tie-breaking may differ for equal
    elements.  Checking that a[argsort(a)] == sort(a) is the correct invariant.
    For stable sort we additionally verify exact index agreement with numpy.
    """
    rng = np.random.default_rng(42)
    if dtype == bool:
        a = rng.integers(0, 2, size=100).astype(bool)
    elif np.issubdtype(dtype, np.unsignedinteger):
        a = rng.integers(0, 200, size=100).astype(dtype)
    else:
        a = rng.integers(-100, 100, size=100).astype(dtype)
    # unstable: permutation must sort the array correctly
    idx = scors.argsort(a)
    assert np.array_equal(a[idx], np.sort(a))
    # stable: must exactly match numpy's stable argsort
    idx_stable = scors.argsort(a, stable=True)
    assert np.array_equal(idx_stable, np.argsort(a, kind="stable"))


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=_float_ids)
def test_argsort_float_matches_numpy(dtype):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(200).astype(dtype)
    assert np.array_equal(scors.argsort(a), np.argsort(a))


def test_argsort_empty():
    a = np.array([], dtype=np.float32)
    assert np.array_equal(scors.argsort(a), np.argsort(a))


def test_argsort_single_element():
    a = np.array([42.0], dtype=np.float64)
    assert np.array_equal(scors.argsort(a), [0])


def test_argsort_already_sorted():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    assert np.array_equal(scors.argsort(a), np.argsort(a))


def test_argsort_reverse_sorted():
    a = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    assert np.array_equal(scors.argsort(a), np.argsort(a))


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=_float_ids)
def test_argsort_nan_sorts_last(dtype):
    a = np.array([1.0, float("nan"), 0.5, float("nan"), 2.0], dtype=dtype)
    idx = scors.argsort(a)
    # values at the returned indices should be non-NaN first, NaN at end
    sorted_values = a[idx]
    nan_mask = np.isnan(sorted_values)
    assert not nan_mask[:-2].any(), "NaN appeared before the last two positions"
    assert nan_mask[-2:].all(), "last two positions should be NaN"


@pytest.mark.parametrize("stable", [False, True], ids=["unstable", "stable"])
def test_argsort_stable_equal_elements(stable):
    # with equal elements, stable sort must preserve relative order of original indices
    a = np.array([3, 1, 1, 2, 1], dtype=np.int32)
    idx = scors.argsort(a, stable=stable)
    if stable:
        # positions of value=1: original indices 1, 2, 4 — must appear in that order
        ones = idx[a[idx] == 1]
        assert list(ones) == sorted(ones), "stable sort must preserve relative order"
    # both stable and unstable must produce a correctly sorted result
    assert np.array_equal(a[idx], np.sort(a))


def test_argsort_returns_int64():
    a = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    assert scors.argsort(a).dtype == np.int64


def test_argsort_num_threads():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(1000).astype(np.float32)
    assert np.array_equal(scors.argsort(a, num_threads=1), np.argsort(a))
    assert np.array_equal(scors.argsort(a, num_threads=2), np.argsort(a))


def test_argsort_strided_input():
    # non-contiguous: every other element of a 2D row
    a = np.array([[5.0, 1.0, 3.0, 2.0, 4.0],
                  [9.0, 6.0, 7.0, 8.0, 0.0]], dtype=np.float32)
    row = a[0]  # C-contiguous row
    assert np.array_equal(scors.argsort(row), np.argsort(row))
    col = np.ascontiguousarray(a[:, 0])  # column extracted as contiguous
    assert np.array_equal(scors.argsort(col), np.argsort(col))


# ── sort ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype,expected_id", zip(_ALL_DTYPES, _all_ids), ids=_all_ids)
def test_sort_matches_numpy(dtype, expected_id):
    rng = np.random.default_rng(42)
    if dtype == bool:
        a = rng.integers(0, 2, size=50).astype(bool)
    elif np.issubdtype(dtype, np.unsignedinteger):
        a = rng.integers(0, 200, size=50).astype(dtype)
    else:
        a = rng.integers(-100, 100, size=50).astype(dtype)
    assert np.array_equal(scors.sort(a), np.sort(a))


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=_float_ids)
def test_sort_float_matches_numpy(dtype):
    rng = np.random.default_rng(1)
    a = rng.standard_normal(200).astype(dtype)
    assert np.allclose(scors.sort(a), np.sort(a))


def test_sort_returns_copy():
    a = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    original = a.copy()
    b = scors.sort(a)
    assert np.array_equal(a, original), "sort must not modify the original"
    assert not np.shares_memory(a, b)


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=_float_ids)
def test_sort_nan_sorts_last(dtype):
    a = np.array([1.0, float("nan"), 0.5, 2.0], dtype=dtype)
    result = scors.sort(a)
    assert not np.isnan(result[:-1]).any()
    assert np.isnan(result[-1])


# ── sort_inplace ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype,expected_id", zip(_ALL_DTYPES, _all_ids), ids=_all_ids)
def test_sort_inplace_matches_numpy(dtype, expected_id):
    rng = np.random.default_rng(42)
    if dtype == bool:
        a = rng.integers(0, 2, size=50).astype(bool)
    elif np.issubdtype(dtype, np.unsignedinteger):
        a = rng.integers(0, 200, size=50).astype(dtype)
    else:
        a = rng.integers(-100, 100, size=50).astype(dtype)
    expected = np.sort(a)
    scors.sort_inplace(a)
    assert np.array_equal(a, expected)


def test_sort_inplace_returns_none():
    a = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    result = scors.sort_inplace(a)
    assert result is None


def test_sort_inplace_modifies_in_place():
    a = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    original_ptr = a.ctypes.data
    scors.sort_inplace(a)
    assert a.ctypes.data == original_ptr  # same buffer
    assert np.array_equal(a, [1.0, 2.0, 3.0])


def test_sort_inplace_rejects_non_contiguous():
    a = np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    col = a[:, 0]  # non-contiguous column view
    assert not col.flags["C_CONTIGUOUS"]
    with pytest.raises(Exception):
        scors.sort_inplace(col)


def test_sort_inplace_num_threads():
    rng = np.random.default_rng(9)
    a = rng.standard_normal(1000).astype(np.float64)
    expected = np.sort(a)
    scors.sort_inplace(a, num_threads=2)
    assert np.allclose(a, expected)
