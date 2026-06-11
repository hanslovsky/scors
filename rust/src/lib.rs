mod combine;

use ndarray::{Array1,ArrayView,ArrayView2,ArrayView3,ArrayViewMut1,Ix1};
use num;
use num::traits::float::TotalOrder;
use numpy::{Element,PyArray,PyArray1,PyArray2,PyArray3,PyArrayDescrMethods,PyArrayDyn,PyArrayMethods,PyReadonlyArray1,PyUntypedArray,PyUntypedArrayMethods,dtype};
use pyo3::Bound;
use pyo3::exceptions::PyTypeError;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cmp::{Ordering,PartialOrd};
use std::iter::{DoubleEndedIterator,repeat};
use std::ops::AddAssign;

/// Total ordering for sorting — floats use `total_cmp` (NaN sorts last),
/// integers and bool use their natural `Ord`.
pub trait TotalCmp: Copy + Send + Sync {
    fn total_cmp(&self, other: &Self) -> Ordering;
}

macro_rules! impl_total_cmp_float {
    ($t:ty) => {
        impl TotalCmp for $t {
            fn total_cmp(&self, other: &Self) -> Ordering { <$t>::total_cmp(self, other) }
        }
    };
}
macro_rules! impl_total_cmp_ord {
    ($t:ty) => {
        impl TotalCmp for $t {
            fn total_cmp(&self, other: &Self) -> Ordering { self.cmp(other) }
        }
    };
}

impl_total_cmp_float!(f32);
impl_total_cmp_float!(f64);
impl_total_cmp_ord!(i8);
impl_total_cmp_ord!(i16);
impl_total_cmp_ord!(i32);
impl_total_cmp_ord!(i64);
impl_total_cmp_ord!(u8);
impl_total_cmp_ord!(u16);
impl_total_cmp_ord!(u32);
impl_total_cmp_ord!(u64);
impl_total_cmp_ord!(bool);

/// Parallel-aware sort helper.  `None` → sequential; `Some(pool)` → Rayon pool.
fn do_argsort<T, D>(data: &D, len: usize, stable: bool, pool: Option<&rayon::ThreadPool>) -> Vec<usize>
where T: TotalCmp, D: Data<T> + Sync + ?Sized
{
    let mut indices: Vec<usize> = (0..len).collect();
    let cmp = |i: &usize, k: &usize| data.get_at(*i).total_cmp(&data.get_at(*k));
    match pool {
        None => {
            // No specific pool requested — use the global Rayon pool (parallel).
            if stable { indices.par_sort_by(cmp); } else { indices.par_sort_unstable_by(cmp); }
        }
        Some(p) => {
            if stable { p.install(|| indices.par_sort_by(cmp)); }
            else      { p.install(|| indices.par_sort_unstable_by(cmp)); }
        }
    }
    indices
}

fn do_sort<T: TotalCmp>(data: &mut [T], stable: bool, pool: Option<&rayon::ThreadPool>) {
    let cmp = |a: &T, b: &T| a.total_cmp(b);
    match pool {
        None => {
            // No specific pool requested — use the global Rayon pool (parallel).
            if stable { data.par_sort_by(cmp); } else { data.par_sort_unstable_by(cmp); }
        }
        Some(p) => {
            if stable { p.install(|| data.par_sort_by(cmp)); }
            else      { p.install(|| data.par_sort_unstable_by(cmp)); }
        }
    }
}

fn build_pool(num_threads: Option<usize>) -> Option<rayon::ThreadPool> {
    num_threads.map(|n| ThreadPoolBuilder::new().num_threads(n).build().unwrap())
}

// Convenience bound for types that can be widened into f64 scores/weights.
pub trait IntoF64: Into<f64> + Copy + PartialOrd {}
impl<T: Into<f64> + Copy + PartialOrd> IntoF64 for T {}

#[derive(Clone, Copy)]
pub enum Order {
    ASCENDING,
    DESCENDING
}

/// Infinite iterator of a constant f64 weight — used for the no-weights path.
#[derive(Clone, Copy)]
struct ConstWeight {
    value: f64,
}

impl ConstWeight {
    fn one() -> Self {
        return ConstWeight { value: 1.0 };
    }
}

impl Iterator for ConstWeight {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        return Some(self.value);
    }
}

impl DoubleEndedIterator for ConstWeight {
    fn next_back(&mut self) -> Option<f64> {
        return Some(self.value);
    }
}

impl Data<f64> for ConstWeight {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = f64> + Clone {
        return *self;
    }

    fn get_at(&self, _index: usize) -> f64 {
        return self.value;
    }
}

pub trait Data<T: Clone> {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> + Clone;
    fn get_at(&self, index: usize) -> T;
}

/// Marker trait: implementors of `Data<T>` that also support sorting.
///
/// The default `argsort_unstable` uses `get_at` for element access so no
/// copy is needed regardless of memory layout.  Implementors that can provide
/// a contiguous `&[T]` should override `as_contiguous_slice` to enable a
/// faster slice-based comparator that LLVM can optimize more aggressively.
pub trait SortableData<T: TotalCmp + Clone>: Data<T> + Sync {
    /// Return a contiguous slice if the data is laid out contiguously in
    /// memory, `None` otherwise.  The default returns `None`.
    fn as_contiguous_slice(&self) -> Option<&[T]> { None }

    fn argsort_unstable(&self, pool: Option<&rayon::ThreadPool>) -> Vec<usize> {
        let len = self.get_iterator().count();
        if let Some(slice) = self.as_contiguous_slice() {
            do_argsort(slice, slice.len(), false, pool)
        } else {
            do_argsort(self, len, false, pool)
        }
    }
}

impl <T: Clone> Data<T> for Vec<T> {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> + Clone {
        return self.iter().cloned();
    }
    fn get_at(&self, index: usize) -> T {
        return self[index].clone();
    }
}

impl<F: TotalCmp + num::Float + TotalOrder + Clone + Sync> SortableData<F> for Vec<F> {
    fn as_contiguous_slice(&self) -> Option<&[F]> { Some(self.as_slice()) }
}

impl <T: Clone> Data<T> for [T] {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> + Clone {
        return self.iter().cloned();
    }
    fn get_at(&self, index: usize) -> T {
        return self[index].clone();
    }
}

impl <T: Clone> Data<T> for &[T] {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> + Clone {
        return self.iter().cloned();
    }
    fn get_at(&self, index: usize) -> T {
        return self[index].clone();
    }
}

impl<F: TotalCmp + num::Float + TotalOrder + Clone + Sync> SortableData<F> for &[F] {
    fn as_contiguous_slice(&self) -> Option<&[F]> { Some(self) }
}

impl <T: Clone, const N: usize> Data<T> for [T; N] {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> + Clone {
        return self.iter().cloned();
    }
    fn get_at(&self, index: usize) -> T {
        return self[index].clone();
    }
}

impl<F: TotalCmp + num::Float + TotalOrder + Clone + Sync, const N: usize> SortableData<F> for [F; N] {
    fn as_contiguous_slice(&self) -> Option<&[F]> { Some(self.as_slice()) }
}

impl <T: Clone> Data<T> for ArrayView<'_, T, Ix1> {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> + Clone {
        return self.iter().cloned();
    }
    fn get_at(&self, index: usize) -> T {
        return self[index].clone();
    }
}

impl <F> SortableData<F> for ArrayView<'_, F, Ix1>
where F: TotalCmp + num::Float + TotalOrder + Clone + Sync {
    fn as_contiguous_slice(&self) -> Option<&[F]> { self.as_slice() }
}

pub trait BinaryLabel: Clone + Copy {
    fn get_value(&self) -> bool;
}

impl BinaryLabel for bool {
    fn get_value(&self) -> bool {
        return self.clone();
    }
}

impl BinaryLabel for u8 {
    fn get_value(&self) -> bool {
        return (self & 1u8) == 1u8;
    }
}

impl BinaryLabel for u16 {
    fn get_value(&self) -> bool {
        return (self & 1u16) == 1u16;
    }
}

impl BinaryLabel for u32 {
    fn get_value(&self) -> bool {
        return (self & 1u32) == 1u32;
    }
}

impl BinaryLabel for u64 {
    fn get_value(&self) -> bool {
        return (self & 1u64) == 1u64;
    }
}

impl BinaryLabel for i8 {
    fn get_value(&self) -> bool {
        return (self & 1i8) == 1i8;
    }
}

impl BinaryLabel for i16 {
    fn get_value(&self) -> bool {
        return (self & 1i16) == 1i16;
    }
}

impl BinaryLabel for i32 {
    fn get_value(&self) -> bool {
        return (self & 1i32) == 1i32;
    }
}

impl BinaryLabel for i64 {
    fn get_value(&self) -> bool {
        return (self & 1i64) == 1i64;
    }
}

fn select<T, I>(slice: &I, indices: &[usize]) -> Vec<T>
where T: Copy, I: Data<T>
{
    let mut selection: Vec<T> = Vec::new();
    selection.reserve_exact(indices.len());
    for index in indices {
        selection.push(slice.get_at(*index));
    }
    return selection;
}

pub trait ScoreSortedDescending {
    fn _score(&self, labels_with_weights: impl Iterator<Item = (f64, (bool, f64))> + Clone) -> f64;
    fn score<P, B, W>(&self, labels_with_weights: impl Iterator<Item = (P, (B, W))> + Clone) -> f64
    where P: IntoF64, B: BinaryLabel, W: IntoF64
    {
        return self._score(
            labels_with_weights.map(|(p, (b, w))| -> (f64, (bool, f64)) { (p.into(), (b.get_value(), w.into()))})
        )
    }
}


pub fn score_sorted_iterators<S, P, B, W>(
    score: S,
    predictions: impl Iterator<Item = P> + Clone,
    labels: impl Iterator<Item = B> + Clone,
    weights: impl Iterator<Item = W> + Clone,
) -> f64
where S: ScoreSortedDescending, P: IntoF64, B: BinaryLabel, W: IntoF64 {
    let zipped = predictions.zip(labels.zip(weights));
    return score.score(zipped);
}


pub fn score_sorted_sample<S, P, B, W>(
    score: S,
    predictions: &impl Data<P>,
    labels: &impl Data<B>,
    weights: &impl Data<W>,
    order: Order,
) -> f64
where S: ScoreSortedDescending, P: IntoF64, B: BinaryLabel, W: IntoF64 + Clone {
    let p = predictions.get_iterator();
    let l = labels.get_iterator();
    let w = weights.get_iterator();
    return match order {
        Order::ASCENDING => score_sorted_iterators(score, p.rev(), l.rev(), w.rev()),
        Order::DESCENDING => score_sorted_iterators(score, p, l, w),
    };
}


pub fn score_maybe_sorted_sample<S, P, B, W>(
    score: S,
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
    order: Option<Order>,
) -> f64
where S: ScoreSortedDescending, P: IntoF64 + TotalCmp + num::Float + TotalOrder, B: BinaryLabel, W: IntoF64
{
    return match order {
        Some(o) => {
            match weights {
                Some(w) => score_sorted_sample(score, predictions, labels, w, o),
                None => score_sorted_sample(score, predictions, labels, &ConstWeight::one(), o),
            }
        }
        None => {
            let mut indices = predictions.argsort_unstable(None);
            indices.reverse(); // score_maybe_sorted_sample needs descending order
            let sorted_labels = select(labels, &indices);
            let sorted_predictions = select(predictions, &indices);
            match weights {
                Some(w) => {
                    let sorted_weights = select(w, &indices);
                    score_sorted_sample(score, &sorted_predictions, &sorted_labels, &sorted_weights, Order::DESCENDING)
                }
                None => score_sorted_sample(score, &sorted_predictions, &sorted_labels, &ConstWeight::one(), Order::DESCENDING)
            }
        }
    };
}


pub fn score_two_sorted_samples<S, P, B, W>(
    score: S,
    predictions1: impl Iterator<Item = P> + Clone,
    label1: impl Iterator<Item = B> + Clone,
    weight1: impl Iterator<Item = W> + Clone,
    predictions2: impl Iterator<Item = P> + Clone,
    label2: impl Iterator<Item = B> + Clone,
    weight2: impl Iterator<Item = W> + Clone,
) -> f64
where S: ScoreSortedDescending, P: IntoF64, B: BinaryLabel + PartialOrd, W: IntoF64
{
    return score_two_sorted_samples_zipped(
        score,
        predictions1.zip(label1.zip(weight1)),
        predictions2.zip(label2.zip(weight2)),
    );
}


pub fn score_two_sorted_samples_zipped<S, P, B, W>(
    score: S,
    iter1: impl Iterator<Item = (P, (B, W))> + Clone,
    iter2: impl Iterator<Item = (P, (B, W))> + Clone,
) -> f64
where S: ScoreSortedDescending, P: IntoF64, B: BinaryLabel + PartialOrd, W: IntoF64
{
    let combined_iter = combine::combine::CombineIterDescending::new(iter1, iter2);
    return score.score(combined_iter);
}


struct AveragePrecision;


#[derive(Clone,Copy,Debug)]
struct Positives {
    tps: f64,
    fps: f64,
}

impl Positives {
    fn new(tps: f64, fps: f64) -> Self {
        return Positives { tps, fps };
    }

    fn zero() -> Self {
        return Positives::new(0.0, 0.0);
    }

    fn add(&mut self, label: bool, weight: f64) {
        let tp = weight * (label as u8) as f64;
        let fp = weight - tp;
        self.tps += tp;
        self.fps += fp;
    }

    fn positives_sum(&self) -> f64 {
        return self.tps + self.fps;
    }

    fn precision(&self) -> f64 {
        return self.tps / self.positives_sum();
    }
}


impl ScoreSortedDescending for AveragePrecision {
    fn _score(&self, mut labels_with_weights: impl Iterator<Item = (f64, (bool, f64))> + Clone) -> f64
    {
        let mut positives = Positives::zero();
        let mut last_p = f64::NAN;
        let mut last_tps = 0.0f64;
        let mut ap = 0.0f64;

        // TODO can we unify this preparation step with the loop?
        match labels_with_weights.next() {
            None => (), // TODO: Should we return an error in this case?
            Some((p, (label, w))) => {
                positives.add(label, w);
                last_p = p;
            }
        }

        for (p, (label, w)) in labels_with_weights {
            if last_p != p {
                ap += (positives.tps - last_tps) * positives.precision();
                last_p = p;
                last_tps = positives.tps;
            }
            positives.add(label, w);
        }

        ap += (positives.tps - last_tps) * positives.precision();

        // Special case for tps == 0 following sklearn
        // https://github.com/scikit-learn/scikit-learn/blob/5cce87176a530d2abea45b5a7e5a4d837c481749/sklearn/metrics/_ranking.py#L1032-L1039
        // I.e. if tps is 0.0, there are no positive samples in labels: Either all labels are 0, or all weights (for positive labels) are 0
        return if positives.tps == 0.0 { 0.0 } else { ap / positives.tps };
    }
}


struct RocAuc;


impl ScoreSortedDescending for RocAuc {
    fn _score(&self, mut labels_with_weights: impl Iterator<Item = (f64, (bool, f64))> + Clone) -> f64
    {
        let mut positives = Positives::zero();
        let mut last_p = f64::NAN;
        let mut last_counted_fp = 0.0f64;
        let mut last_counted_tp = 0.0f64;
        let mut area_under_curve = 0.0f64;

        // TODO can we unify this preparation step with the loop?
        match labels_with_weights.next() {
            None => (), // TODO: Should we return an error in this case?
            Some((p, (label, w))) => {
                positives.add(label, w);
                last_p = p;
            }
        }

        for (p, (label, w)) in labels_with_weights {
            if last_p != p {
                area_under_curve += area_under_line_segment(
                    last_counted_fp, positives.fps,
                    last_counted_tp, positives.tps,
                );
                last_counted_fp = positives.fps;
                last_counted_tp = positives.tps;
                last_p = p;
            }
            positives.add(label, w);
        }
        area_under_curve += area_under_line_segment(
            last_counted_fp, positives.fps,
            last_counted_tp, positives.tps,
        );
        return area_under_curve / (positives.tps * positives.fps);
    }
}


struct RocAucWithMaxFPR {
    max_fpr: f64,
}


impl RocAucWithMaxFPR {
    fn new(max_fpr: f64) -> Self {
        return RocAucWithMaxFPR { max_fpr };
    }

    fn get_positive_sum(labels_with_weights: impl Iterator<Item = (bool, f64)>) -> Positives {
        let mut positives = Positives::zero();
        for (label, weight) in labels_with_weights {
            positives.add(label, weight);
        }
        return positives;
    }
}


impl ScoreSortedDescending for RocAucWithMaxFPR {
    fn _score(&self, mut labels_with_weights: impl Iterator<Item = (f64, (bool, f64))> + Clone) -> f64
    {
        let total_positives = Self::get_positive_sum(labels_with_weights.clone().map(|(_p, (b, w))| (b, w)));
        let max_fpr = self.max_fpr;
        let false_positive_cutoff = max_fpr * total_positives.fps;

        let mut positives = Positives::zero();
        let mut last_p = f64::NAN;
        let mut last_counted_fp = 0.0f64;
        let mut last_counted_tp = 0.0f64;
        let mut area_under_curve = 0.0f64;

        // TODO can we unify this preparation step with the loop?
        match labels_with_weights.next() {
            None => (), // TODO: Should we return an error in this case?
            Some((p, (label, w))) => {
                positives.add(label, w);
                last_p = p;
            }
        }

        for (p, (label, w)) in labels_with_weights {
            if last_p != p {
                area_under_curve += area_under_line_segment(
                    last_counted_fp, positives.fps,
                    last_counted_tp, positives.tps,
                );
                last_counted_fp = positives.fps;
                last_counted_tp = positives.tps;
                last_p = p;
            }
            let mut next_pos = positives;
            next_pos.add(label, w);
            if next_pos.fps > false_positive_cutoff {
                let dx = next_pos.fps - positives.fps;
                let dy = next_pos.tps - positives.tps;
                positives = Positives::new(
                    positives.tps + dy * false_positive_cutoff / dx,
                    false_positive_cutoff,
                );
                break;
            } else {
                positives = next_pos;
            }
        }

        area_under_curve += area_under_line_segment(
            last_counted_fp, positives.fps,
            last_counted_tp, positives.tps,
        );

        let normalized_area_under_curve = area_under_curve / (total_positives.tps * total_positives.fps);
        let min_area = 0.5 * max_fpr * max_fpr;
        let max_area = max_fpr;
        return 0.5 * (1.0 + (normalized_area_under_curve - min_area) / (max_area - min_area));
    }
}


struct RocAucWithOptionalMaxFPR {
    max_fpr: Option<f64>,
}

impl RocAucWithOptionalMaxFPR {
    fn new(max_fpr: Option<f64>) -> Self {
        return Self { max_fpr };
    }
}


impl ScoreSortedDescending for RocAucWithOptionalMaxFPR {
    fn _score(&self, labels_with_weights: impl Iterator<Item = (f64, (bool, f64))> + Clone) -> f64
    {
        return match self.max_fpr {
            Some(mfpr) => RocAucWithMaxFPR::new(mfpr).score(labels_with_weights),
            None => RocAuc.score(labels_with_weights),
        }
    }
}


pub fn average_precision<P, B, W>(
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
    order: Option<Order>,
) -> f64
where P: IntoF64 + TotalCmp + num::Float + TotalOrder, B: BinaryLabel, W: IntoF64
{
    return score_maybe_sorted_sample(AveragePrecision, predictions, labels, weights, order);
}


pub fn roc_auc<P, B, W>(
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
    order: Option<Order>,
    max_fpr: Option<f64>,
) -> f64
where P: IntoF64 + TotalCmp + num::Float + TotalOrder, B: BinaryLabel, W: IntoF64
{
    return score_maybe_sorted_sample(RocAucWithOptionalMaxFPR::new(max_fpr), predictions, labels, weights, order);
}


fn area_under_line_segment(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    return dx * y0 + dy * dx * 0.5;
}


/// Accumulate `row` into `sum` element-wise.
///
/// When monomorphized with `Copied<slice::Iter>` LLVM proves unit stride and
/// emits SIMD.  When called with ndarray's strided `Iter` it stays scalar.
#[inline]
fn accum_row<F: num::Float + AddAssign>(row: impl Iterator<Item = F>, sum: &mut [F]) {
    for (f, s) in row.zip(sum.iter_mut()) {
        *s += f;
    }
}

/// Compute (prod_sum, m_sqs, l_sqs) for one replicate row.
///
/// Same vectorization note as `accum_row`.
#[inline]
fn score_row<F: num::Float + AddAssign>(
    row: impl Iterator<Item = F>,
    sum: impl Iterator<Item = F>,
    loo_weight_factor: F,
) -> (F, F, F) {
    let mut prod_sum = F::zero();
    let mut m_sqs = F::zero();
    let mut l_sqs = F::zero();
    for (f, s) in row.zip(sum) {
        let m_f = f;
        let l_f = (s - f) * loo_weight_factor;
        prod_sum += m_f * l_f;
        m_sqs += m_f * m_f;
        l_sqs += l_f * l_f;
    }
    (prod_sum, m_sqs, l_sqs)
}

/// Core two-pass loop shared by the contiguous and strided paths.
///
/// `rows` must be `Clone` because the algorithm makes two passes: one to
/// accumulate `sum`, one to score.  The `Clone` is free — the iterator is
/// just a pointer and two `usize`s on the stack.
///
/// LLVM monomorphizes separately for each `Row` type: when called with
/// `Chunks`-derived slice iterators it emits SIMD; when called with
/// ndarray's strided iterators it stays scalar.
///
/// Note: `impl Fn(&ArrayView1) -> I` would be cleaner at the call site but
/// hits a Rust lifetime limitation — `as_slice()` ties its `&[F]` return to
/// `&self` (the view reference), not to the underlying data lifetime, so the
/// returned iterator borrows a local and the compiler rejects it.  The
/// iterator solution avoids this: `chunks()` borrows directly from `mat`
/// and `into_iter()` transfers the data lifetime from the view.
#[inline]
fn loo_cossim_loops<F, Row>(
    rows: impl Iterator<Item = Row> + Clone,
    sum: &mut [F],
    loo_weight_factor: F,
    num_replicates: usize,
) -> F
where
    F: num::Float + AddAssign,
    Row: Iterator<Item = F>,
{
    for row in rows.clone() {
        accum_row(row, sum);
    }
    let mut result = F::zero();
    for row in rows {
        let (prod_sum, m_sqs, l_sqs) = score_row(row, sum.iter().copied(), loo_weight_factor);
        result += prod_sum / (m_sqs * l_sqs).sqrt();
    }
    result / F::from(num_replicates).unwrap()
}

pub fn loo_cossim<F: num::Float + AddAssign>(mat: &ArrayView2<'_, F>, replicate_sum: &mut ArrayViewMut1<'_, F>) -> F {
    let num_replicates = mat.shape()[0];
    let loo_weight_factor = F::from(1).unwrap() / F::from(num_replicates - 1).unwrap();
    let ncols = mat.shape()[1];
    // replicate_sum is always created from Array1::zeros() inside Rust, so
    // it is guaranteed to be contiguous.
    let sum = replicate_sum.as_slice_mut().unwrap();

    // mat.as_slice() succeeds iff the matrix is fully C-contiguous (strides
    // == [ncols, 1]).  chunks() then yields &[F] slices that borrow directly
    // from mat's data so LLVM sees plain pointer arithmetic and emits SIMD.
    // For non-contiguous input (strided or Fortran-order) outer_iter() +
    // into_iter() transfer the data lifetime from the view correctly.
    //
    // In the primary call path (loo_cossim_many, univariate sampling) each
    // 2-D slice has strides (ncols, 1) and as_slice() always succeeds.
    if let Some(mat_slice) = mat.as_slice() {
        loo_cossim_loops(mat_slice.chunks(ncols).map(|s| s.iter().copied()), sum, loo_weight_factor, num_replicates)
    } else {
        loo_cossim_loops(mat.outer_iter().map(|r| r.into_iter().copied()), sum, loo_weight_factor, num_replicates)
    }
}


pub fn loo_cossim_single<F: num::Float + AddAssign>(mat: &ArrayView2<'_, F>) -> F {
    let mut replicate_sum = Array1::<F>::zeros(mat.shape()[1]);
    return loo_cossim(mat, &mut replicate_sum.view_mut());
}


pub fn loo_cossim_many<F: num::Float + AddAssign>(mat: &ArrayView3<'_, F>) -> Array1<F> {
    let mut cossims = Array1::<F>::zeros(mat.shape()[0]);
    let mut replicate_sum = Array1::<F>::zeros(mat.shape()[2]);
    for (m, c) in mat.outer_iter().zip(cossims.iter_mut()) {
        replicate_sum.fill(F::zero());
        *c = loo_cossim(&m, &mut replicate_sum.view_mut());
    }
    return cossims;
}


// Python bindings
#[pyclass(eq, eq_int, from_py_object, name="Order")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyOrder {
    ASCENDING,
    DESCENDING
}

fn py_order_as_order(order: PyOrder) -> Order {
    return match order {
        PyOrder::ASCENDING => Order::ASCENDING,
        PyOrder::DESCENDING => Order::DESCENDING,
    }
}

trait PyScoreGeneric<S: ScoreSortedDescending>: Ungil + Sync {

    fn get_score(&self) -> S;

    fn score_py<'py, P, B, W>(
        &self,
        py: Python<'py>,
        labels: PyReadonlyArray1<'py, B>,
        predictions: PyReadonlyArray1<'py, P>,
        weights: Option<PyReadonlyArray1<'py, W>>,
        order: Option<PyOrder>,
    ) -> f64
    where P: IntoF64 + TotalCmp + Element + num::Float + TotalOrder, B: BinaryLabel + Element, W: IntoF64 + Element
    {
        let labels = labels.as_array();
        let predictions = predictions.as_array();
        let order = order.map(py_order_as_order);
        return match weights {
            Some(weight) => {
                let w = weight.as_array();
                py.detach(move || {
                    score_maybe_sorted_sample(self.get_score(), &predictions, &labels, Some(&w), order)
                })
            },
            None => py.detach(move || {
                score_maybe_sorted_sample(self.get_score(), &predictions, &labels, None::<&Vec<W>>, order)
            })
        };
    }

    fn score_two_sorted_samples_py_generic<'py, B, F, B1, F1, F2, W1, W2>(
        &self,
        py: Python<'py>,
        labels1: PyReadonlyArray1<'py, B1>,
        predictions1: PyReadonlyArray1<'py, F1>,
        weights1: Option<PyReadonlyArray1<'py, W1>>,
        labels2: PyReadonlyArray1<'py, B1>,
        predictions2: PyReadonlyArray1<'py, F2>,
        weights2: Option<PyReadonlyArray1<'py, W2>>,
    ) -> f64
    where B: BinaryLabel + PartialOrd + Ungil, F: IntoF64 + Ungil,
          B1: Element + Into<B> + Clone,
          F1: Element + Into<F> + Clone, F2: Element + Into<F> + Clone,
          W1: Element + Into<f64> + Clone, W2: Element + Into<f64> + Clone
    {
        let l1 = labels1.as_array().into_iter().cloned().map(|l| -> B { l.into() });
        let l2 = labels2.as_array().into_iter().cloned().map(|l| -> B { l.into() });
        let p1 = predictions1.as_array().into_iter().cloned().map(|f| -> F { f.into() });
        let p2 = predictions2.as_array().into_iter().cloned().map(|f| -> F { f.into() });

        return match (weights1, weights2) {
            (None, None) => {
                py.detach(move || {
                    score_two_sorted_samples(self.get_score(), p1, l1, repeat(1.0f64), p2, l2, repeat(1.0f64))
                })
            }
            (Some(w1), None) => {
                let w1i = w1.as_array().into_iter().cloned().map(|w| -> f64 { w.into() });
                py.detach(move || {
                    score_two_sorted_samples(self.get_score(), p1, l1, w1i, p2, l2, repeat(1.0f64))
                })
            }
            (None, Some(w2)) => {
                let w2i = w2.as_array().into_iter().cloned().map(|w| -> f64 { w.into() });
                py.detach(move || {
                    score_two_sorted_samples(self.get_score(), p1, l1, repeat(1.0f64), p2, l2, w2i)
                })
            }
            (Some(w1), Some(w2)) => {
                let w1i = w1.as_array().into_iter().cloned().map(|w| -> f64 { w.into() });
                let w2i = w2.as_array().into_iter().cloned().map(|w| -> f64 { w.into() });
                py.detach(move || {
                    score_two_sorted_samples(self.get_score(), p1, l1, w1i, p2, l2, w2i)
                })
            }
        };
    }
}

struct AveragePrecisionPyGeneric;

impl PyScoreGeneric<AveragePrecision> for AveragePrecisionPyGeneric {
    fn get_score(&self) -> AveragePrecision {
        return AveragePrecision;
    }
}

struct RocAucPyGeneric {
    max_fpr: Option<f64>,
}

impl RocAucPyGeneric {
    fn new(max_fpr: Option<f64>) -> Self {
        return RocAucPyGeneric { max_fpr };
    }
}

impl PyScoreGeneric<RocAucWithOptionalMaxFPR> for RocAucPyGeneric {
    fn get_score(&self) -> RocAucWithOptionalMaxFPR {
        return RocAucWithOptionalMaxFPR::new(self.max_fpr);
    }
}

// https://stackoverflow.com/questions/70128978/how-to-define-different-function-names-with-a-macro
// https://stackoverflow.com/questions/70872059/using-a-rust-macro-to-generate-a-function-with-variable-parameters
// https://doc.rust-lang.org/rust-by-example/macros/designators.html
// https://users.rust-lang.org/t/is-there-a-way-to-convert-given-identifier-to-a-string-in-a-macro/42907
macro_rules! average_precision_py {
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (labels, predictions, *, weights=None, order=None))]
        pub fn $fname<'py>(
            py: Python<'py>,
            labels: PyReadonlyArray1<'py, $label_type>,
            predictions: PyReadonlyArray1<'py, $prediction_type>,
            weights: Option<PyReadonlyArray1<'py, $prediction_type>>,
            order: Option<PyOrder>
        ) -> f64
        {
            return AveragePrecisionPyGeneric.score_py(py, labels, predictions, weights, order);
        }
    };
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty, $py_module:ident) => {
        average_precision_py!($fname, $pyname, $label_type, $prediction_type);
        $py_module.add_function(wrap_pyfunction!($fname, $py_module)?).unwrap();
    };
}


macro_rules! roc_auc_py {
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (labels, predictions, *, weights=None, order=None, max_fpr=None))]
        pub fn $fname<'py>(
            py: Python<'py>,
            labels: PyReadonlyArray1<'py, $label_type>,
            predictions: PyReadonlyArray1<'py, $prediction_type>,
            weights: Option<PyReadonlyArray1<'py, $prediction_type>>,
            order: Option<PyOrder>,
            max_fpr: Option<f64>,
        ) -> f64
        {
            return RocAucPyGeneric::new(max_fpr).score_py(py, labels, predictions, weights, order);
        }
    };
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty, $py_module:ident) => {
        roc_auc_py!($fname, $pyname, $label_type, $prediction_type);
        $py_module.add_function(wrap_pyfunction!($fname, $py_module)?).unwrap();
    };
}


macro_rules! average_precision_on_two_sorted_samples_py {
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2, *))]
        pub fn $fname<'py>(
            py: Python<'py>,
            labels1: PyReadonlyArray1<'py, $label_type>,
            predictions1: PyReadonlyArray1<'py, $prediction_type>,
            weights1: Option<PyReadonlyArray1<'py, $prediction_type>>,
            labels2: PyReadonlyArray1<'py, $label_type>,
            predictions2: PyReadonlyArray1<'py, $prediction_type>,
            weights2: Option<PyReadonlyArray1<'py, $prediction_type>>,
        ) -> f64
        {
            return AveragePrecisionPyGeneric.score_two_sorted_samples_py_generic::<$label_type, $prediction_type, $label_type, $prediction_type, $prediction_type, $prediction_type, $prediction_type>(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
        }
    };
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty, $py_module:ident) => {
        average_precision_on_two_sorted_samples_py!($fname, $pyname, $label_type, $prediction_type);
        $py_module.add_function(wrap_pyfunction!($fname, $py_module)?).unwrap();
    };
}


macro_rules! roc_auc_on_two_sorted_samples_py {
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2, *, max_fpr=None))]
        pub fn $fname<'py>(
            py: Python<'py>,
            labels1: PyReadonlyArray1<'py, $label_type>,
            predictions1: PyReadonlyArray1<'py, $prediction_type>,
            weights1: Option<PyReadonlyArray1<'py, $prediction_type>>,
            labels2: PyReadonlyArray1<'py, $label_type>,
            predictions2: PyReadonlyArray1<'py, $prediction_type>,
            weights2: Option<PyReadonlyArray1<'py, $prediction_type>>,
            max_fpr: Option<f64>,
        ) -> f64
        {
            return RocAucPyGeneric::new(max_fpr).score_two_sorted_samples_py_generic::<$label_type, $prediction_type, $label_type, $prediction_type, $prediction_type, $prediction_type, $prediction_type>(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
        }
    };
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty, $py_module:ident) => {
        roc_auc_on_two_sorted_samples_py!($fname, $pyname, $label_type, $prediction_type);
        $py_module.add_function(wrap_pyfunction!($fname, $py_module)?).unwrap();
    };
}


#[pyfunction(name = "loo_cossim")]
#[pyo3(signature = (data))]
pub fn loo_cossim_py<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyUntypedArray>
) -> PyResult<f64> {
    if data.ndim() != 2 {
        return Err(PyTypeError::new_err(format!("Expected 2-dimensional array for data (samples x features) but found {} dimenisons.", data.ndim())));
    }

    let dt = data.dtype();
    if dt.is_equiv_to(&dtype::<f32>(py)) {
        let typed_data = data.cast::<PyArray2<f32>>().unwrap().readonly();
        let array = typed_data.as_array();
        let score = py.detach(move || {
            loo_cossim_single(&array)
        });
        return Ok(score as f64);
    }
    if dt.is_equiv_to(&dtype::<f64>(py)) {
        let typed_data = data.cast::<PyArray2<f64>>().unwrap().readonly();
        let array = typed_data.as_array();
        let score = py.detach(move || {
            loo_cossim_single(&array)
        });
        return Ok(score);
    }
    return Err(PyTypeError::new_err(format!("Only float32 and float64 data supported, but found {}", dt)));
}

pub fn loo_cossim_many_generic_py<'py, F: num::Float + AddAssign + Element>(
    py: Python<'py>,
    data: &Bound<'py, PyArrayDyn<F>>
) -> PyResult<Bound<'py, PyArray1<F>>> {
    if data.ndim() != 3 {
        return Err(PyTypeError::new_err(format!("Expected 3-dimensional array for data (outer(?) x samples x features) but found {} dimenisons.", data.ndim())));
    }
    let typed_data = data.cast::<PyArray3<F>>().unwrap().readonly();
    let array = typed_data.as_array();
    let score = py.detach(move || {
        loo_cossim_many(&array)
    });
    // TODO how can we return this generically without making a copy at the end?
    let score_py = PyArray::from_owned_array(py, score);
    return Ok(score_py);
}

#[pyfunction(name = "loo_cossim_many_f64")]
#[pyo3(signature = (data))]
pub fn loo_cossim_many_py_f64<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyUntypedArray>
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if data.ndim() != 3 {
        return Err(PyTypeError::new_err(format!("Expected 3-dimensional array for data (outer(?) x samples x features) but found {} dimenisons.", data.ndim())));
    }

    let dt = data.dtype();
    if !dt.is_equiv_to(&dtype::<f64>(py)) {
        return Err(PyTypeError::new_err(format!("Only float64 data supported, but found {}", dt)));
    }
    let typed_data = data.cast::<PyArrayDyn<f64>>().unwrap();
    return loo_cossim_many_generic_py(py, typed_data);
}

#[pyfunction(name = "loo_cossim_many_f32")]
#[pyo3(signature = (data))]
pub fn loo_cossim_many_py_f32<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyUntypedArray>
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if data.ndim() != 3 {
        return Err(PyTypeError::new_err(format!("Expected 3-dimensional array for data (outer(?) x samples x features) but found {} dimenisons.", data.ndim())));
    }

    let dt = data.dtype();
    if !dt.is_equiv_to(&dtype::<f32>(py)) {
        return Err(PyTypeError::new_err(format!("Only float32 data supported, but found {}", dt)));
    }
    let typed_data = data.cast::<PyArrayDyn<f32>>().unwrap();
    return loo_cossim_many_generic_py(py, typed_data);
}

// ── argsort / sort Python API ────────────────────────────────────────────────
//
// Differences from numpy.argsort / numpy.sort:
//   - 1D only (no `axis` parameter)
//   - `stable: bool = False` instead of numpy's `kind` string
//   - `num_threads: int | None = None`  (None → sequential; n → Rayon pool of n)
//   - No `order` parameter (no structured arrays)
//   - argsort always returns int64 (numpy returns intp, platform-dependent)
//   - NaN sorts to end (consistent with numpy float behaviour via total_cmp)

macro_rules! argsort_py {
    ($fname:ident, $pyname:literal, $type:ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (data, *, stable=false, num_threads=None))]
        pub fn $fname<'py>(
            py: Python<'py>,
            data: PyReadonlyArray1<'py, $type>,
            stable: bool,
            num_threads: Option<usize>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>> {
            let arr = data.as_array();
            let len = arr.len();
            let pool = build_pool(num_threads);
            let indices = py.detach(|| do_argsort(&arr, len, stable, pool.as_ref()));
            let out: Vec<i64> = indices.into_iter().map(|i| i as i64).collect();
            Ok(PyArray1::from_vec(py, out))
        }
    };
    ($fname:ident, $pyname:literal, $type:ty, $module:ident) => {
        argsort_py!($fname, $pyname, $type);
        $module.add_function(wrap_pyfunction!($fname, $module)?).unwrap();
    };
}

macro_rules! sort_inplace_py {
    ($fname:ident, $pyname:literal, $type:ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (data, *, stable=false, num_threads=None))]
        pub fn $fname<'py>(
            py: Python<'py>,
            data: &Bound<'py, PyArray1<$type>>,
            stable: bool,
            num_threads: Option<usize>,
        ) -> PyResult<()> {
            let mut arr = unsafe { data.as_array_mut() };
            let slice = arr.as_slice_mut()
                .ok_or_else(|| PyTypeError::new_err("sort_inplace requires a C-contiguous array"))?;
            let pool = build_pool(num_threads);
            py.detach(|| do_sort(slice, stable, pool.as_ref()));
            Ok(())
        }
    };
    ($fname:ident, $pyname:literal, $type:ty, $module:ident) => {
        sort_inplace_py!($fname, $pyname, $type);
        $module.add_function(wrap_pyfunction!($fname, $module)?).unwrap();
    };
}

#[pymodule(name = "_scors")]
fn scors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    average_precision_py!(average_precision_bool_f32, "average_precision_bool_f32", bool, f32, m);
    average_precision_py!(average_precision_i8_f32, "average_precision_i8_f32", i8, f32, m);
    average_precision_py!(average_precision_i16_f32, "average_precision_i16_f32", i16, f32, m);
    average_precision_py!(average_precision_i32_f32, "average_precision_i32_f32", i32, f32, m);
    average_precision_py!(average_precision_i64_f32, "average_precision_i64_f32", i64, f32, m);
    average_precision_py!(average_precision_u8_f32, "average_precision_u8_f32", u8, f32, m);
    average_precision_py!(average_precision_u16_f32, "average_precision_u16_f32", u16, f32, m);
    average_precision_py!(average_precision_u32_f32, "average_precision_u32_f32", u32, f32, m);
    average_precision_py!(average_precision_u64_f32, "average_precision_u64_f32", u64, f32, m);
    average_precision_py!(average_precision_bool_f64, "average_precision_bool_f64", bool, f64, m);
    average_precision_py!(average_precision_i8_f64, "average_precision_i8_f64", i8, f64, m);
    average_precision_py!(average_precision_i16_f64, "average_precision_i16_f64", i16, f64, m);
    average_precision_py!(average_precision_i32_f64, "average_precision_i32_f64", i32, f64, m);
    average_precision_py!(average_precision_i64_f64, "average_precision_i64_f64", i64, f64, m);
    average_precision_py!(average_precision_u8_f64, "average_precision_u8_f64", u8, f64, m);
    average_precision_py!(average_precision_u16_f64, "average_precision_u16_f64", u16, f64, m);
    average_precision_py!(average_precision_u32_f64, "average_precision_u32_f64", u32, f64, m);
    average_precision_py!(average_precision_u64_f64, "average_precision_u64_f64", u64, f64, m);

    roc_auc_py!(roc_auc_bool_f32, "roc_auc_bool_f32", bool, f32, m);
    roc_auc_py!(roc_auc_i8_f32, "roc_auc_i8_f32", i8, f32, m);
    roc_auc_py!(roc_auc_i16_f32, "roc_auc_i16_f32", i16, f32, m);
    roc_auc_py!(roc_auc_i32_f32, "roc_auc_i32_f32", i32, f32, m);
    roc_auc_py!(roc_auc_i64_f32, "roc_auc_i64_f32", i64, f32, m);
    roc_auc_py!(roc_auc_u8_f32, "roc_auc_u8_f32", u8, f32, m);
    roc_auc_py!(roc_auc_u16_f32, "roc_auc_u16_f32", u16, f32, m);
    roc_auc_py!(roc_auc_u32_f32, "roc_auc_u32_f32", u32, f32, m);
    roc_auc_py!(roc_auc_u64_f32, "roc_auc_u64_f32", u64, f32, m);
    roc_auc_py!(roc_auc_bool_f64, "roc_auc_bool_f64", bool, f64, m);
    roc_auc_py!(roc_auc_i8_f64, "roc_auc_i8_f64", i8, f64, m);
    roc_auc_py!(roc_auc_i16_f64, "roc_auc_i16_f64", i16, f64, m);
    roc_auc_py!(roc_auc_i32_f64, "roc_auc_i32_f64", i32, f64, m);
    roc_auc_py!(roc_auc_i64_f64, "roc_auc_i64_f64", i64, f64, m);
    roc_auc_py!(roc_auc_u8_f64, "roc_auc_u8_f64", u8, f64, m);
    roc_auc_py!(roc_auc_u16_f64, "roc_auc_u16_f64", u16, f64, m);
    roc_auc_py!(roc_auc_u32_f64, "roc_auc_u32_f64", u32, f64, m);
    roc_auc_py!(roc_auc_u64_f64, "roc_auc_u64_f64", u64, f64, m);

    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_bool_f32, "average_precision_on_two_sorted_samples_bool_f32", bool, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i8_f32, "average_precision_on_two_sorted_samples_i8_f32", i8, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i16_f32, "average_precision_on_two_sorted_samples_i16_f32", i16, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i32_f32, "average_precision_on_two_sorted_samples_i32_f32", i32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i64_f32, "average_precision_on_two_sorted_samples_i64_f32", i64, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u8_f32, "average_precision_on_two_sorted_samples_u8_f32", u8, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u16_f32, "average_precision_on_two_sorted_samples_u16_f32", u16, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u32_f32, "average_precision_on_two_sorted_samples_u32_f32", u32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u64_f32, "average_precision_on_two_sorted_samples_u64_f32", u64, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_bool_f64, "average_precision_on_two_sorted_samples_bool_f64", bool, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i8_f64, "average_precision_on_two_sorted_samples_i8_f64", i8, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i16_f64, "average_precision_on_two_sorted_samples_i16_f64", i16, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i32_f64, "average_precision_on_two_sorted_samples_i32_f64", i32, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i64_f64, "average_precision_on_two_sorted_samples_i64_f64", i64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u8_f64, "average_precision_on_two_sorted_samples_u8_f64", u8, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u16_f64, "average_precision_on_two_sorted_samples_u16_f64", u16, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u32_f64, "average_precision_on_two_sorted_samples_u32_f64", u32, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u64_f64, "average_precision_on_two_sorted_samples_u64_f64", u64, f64, m);

    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_bool_f32, "roc_auc_on_two_sorted_samples_bool_f32", bool, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i8_f32, "roc_auc_on_two_sorted_samples_i8_f32", i8, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i16_f32, "roc_auc_on_two_sorted_samples_i16_f32", i16, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i32_f32, "roc_auc_on_two_sorted_samples_i32_f32", i32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i64_f32, "roc_auc_on_two_sorted_samples_i64_f32", i64, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u8_f32, "roc_auc_on_two_sorted_samples_u8_f32", u8, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u16_f32, "roc_auc_on_two_sorted_samples_u16_f32", u16, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u32_f32, "roc_auc_on_two_sorted_samples_u32_f32", u32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u64_f32, "roc_auc_on_two_sorted_samples_u64_f32", u64, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_bool_f64, "roc_auc_on_two_sorted_samples_bool_f64", bool, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i8_f64, "roc_auc_on_two_sorted_samples_i8_f64", i8, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i16_f64, "roc_auc_on_two_sorted_samples_i16_f64", i16, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i32_f64, "roc_auc_on_two_sorted_samples_i32_f64", i32, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i64_f64, "roc_auc_on_two_sorted_samples_i64_f64", i64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u8_f64, "roc_auc_on_two_sorted_samples_u8_f64", u8, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u16_f64, "roc_auc_on_two_sorted_samples_u16_f64", u16, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u32_f64, "roc_auc_on_two_sorted_samples_u32_f64", u32, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u64_f64, "roc_auc_on_two_sorted_samples_u64_f64", u64, f64, m);

    m.add_function(wrap_pyfunction!(loo_cossim_py, m)?).unwrap();
    m.add_function(wrap_pyfunction!(loo_cossim_many_py_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(loo_cossim_many_py_f32, m)?).unwrap();
    m.add_class::<PyOrder>().unwrap();

    argsort_py!(argsort_f32,  "argsort_f32",  f32,  m);
    argsort_py!(argsort_f64,  "argsort_f64",  f64,  m);
    argsort_py!(argsort_i8,   "argsort_i8",   i8,   m);
    argsort_py!(argsort_i16,  "argsort_i16",  i16,  m);
    argsort_py!(argsort_i32,  "argsort_i32",  i32,  m);
    argsort_py!(argsort_i64,  "argsort_i64",  i64,  m);
    argsort_py!(argsort_u8,   "argsort_u8",   u8,   m);
    argsort_py!(argsort_u16,  "argsort_u16",  u16,  m);
    argsort_py!(argsort_u32,  "argsort_u32",  u32,  m);
    argsort_py!(argsort_u64,  "argsort_u64",  u64,  m);
    argsort_py!(argsort_bool, "argsort_bool", bool, m);

    sort_inplace_py!(sort_inplace_f32,  "sort_inplace_f32",  f32,  m);
    sort_inplace_py!(sort_inplace_f64,  "sort_inplace_f64",  f64,  m);
    sort_inplace_py!(sort_inplace_i8,   "sort_inplace_i8",   i8,   m);
    sort_inplace_py!(sort_inplace_i16,  "sort_inplace_i16",  i16,  m);
    sort_inplace_py!(sort_inplace_i32,  "sort_inplace_i32",  i32,  m);
    sort_inplace_py!(sort_inplace_i64,  "sort_inplace_i64",  i64,  m);
    sort_inplace_py!(sort_inplace_u8,   "sort_inplace_u8",   u8,   m);
    sort_inplace_py!(sort_inplace_u16,  "sort_inplace_u16",  u16,  m);
    sort_inplace_py!(sort_inplace_u32,  "sort_inplace_u32",  u32,  m);
    sort_inplace_py!(sort_inplace_u64,  "sort_inplace_u64",  u64,  m);
    sort_inplace_py!(sort_inplace_bool, "sort_inplace_bool", bool, m);

    return Ok(());
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_precision_on_sorted() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual: f64 = score_sorted_sample(AveragePrecision, &predictions, &labels, &weights, Order::DESCENDING);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_on_sorted_double() {
        let labels: [u8; 8] = [1, 1, 0, 0, 1, 1, 0, 0];
        let predictions: [f64; 8] = [0.8, 0.8, 0.4, 0.4, 0.35, 0.35, 0.1, 0.1];
        let weights: [f64; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let actual: f64 = score_sorted_sample(AveragePrecision, &predictions, &labels, &weights, Order::DESCENDING);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_unsorted() {
        let labels: [u8; 4] = [0, 0, 1, 1];
        let predictions: [f64; 4] = [0.1, 0.4, 0.35, 0.8];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual: f64 = average_precision(&predictions, &labels, Some(&weights), None);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_sorted() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual: f64 = average_precision(&predictions, &labels, Some(&weights), Some(Order::DESCENDING));
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_sorted_pair() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual: f64 = score_two_sorted_samples(
            AveragePrecision,
            predictions.iter().cloned(),
            labels.iter().cloned(),
            weights.iter().cloned(),
            predictions.iter().cloned(),
            labels.iter().cloned(),
            weights.iter().cloned()
        );
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_roc_auc() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual: f64 = roc_auc(&predictions, &labels, Some(&weights), Some(Order::DESCENDING), None);
        assert_eq!(actual, 0.75);
    }

    #[test]
    fn test_roc_auc_double() {
        let labels: [u8; 8] = [1, 0, 1, 0, 1, 0, 1, 0];
        let predictions: [f64; 8] = [0.8, 0.4, 0.35, 0.1, 0.8, 0.4, 0.35, 0.1];
        let actual: f64 = roc_auc(&predictions, &labels, None::<&[f64; 8]>, None, None);
        assert_eq!(actual, 0.75);
    }

    #[test]
    fn test_roc_sorted_pair() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual: f64 = score_two_sorted_samples(
            RocAuc,
            predictions.iter().cloned(),
            labels.iter().cloned(),
            weights.iter().cloned(),
            predictions.iter().cloned(),
            labels.iter().cloned(),
            weights.iter().cloned()
        );
        assert_eq!(actual, 0.75);
    }

    #[test]
    fn test_roc_auc_max_fpr() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual: f64 = roc_auc(&predictions, &labels, Some(&weights), Some(Order::DESCENDING), Some(0.25));
        assert_eq!(actual, 0.7142857142857143);
    }

    #[test]
    fn test_roc_auc_max_fpr_double() {
        let labels: [u8; 8] = [1, 0, 1, 0, 1, 0, 1, 0];
        let predictions: [f64; 8] = [0.8, 0.4, 0.35, 0.1, 0.8, 0.4, 0.35, 0.1];
        let actual: f64 = roc_auc(&predictions, &labels, None::<&[f64; 8]>, None, Some(0.25));
        assert_eq!(actual, 0.7142857142857143);
    }

    #[test]
    fn test_roc_auc_max_fpr_sorted_pair() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual: f64 = score_two_sorted_samples(
            RocAucWithMaxFPR::new(0.25f64),
            predictions.iter().cloned(),
            labels.iter().cloned(),
            weights.iter().cloned(),
            predictions.iter().cloned(),
            labels.iter().cloned(),
            weights.iter().cloned()
        );
        assert_eq!(actual, 0.7142857142857143);
    }
}
