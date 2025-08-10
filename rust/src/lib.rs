#![feature(trait_alias)]

mod combine;

use ndarray::{Array1,ArrayView,ArrayView2,ArrayView3,ArrayViewMut1,Ix1};
use num;
use num::traits::float::TotalOrder;
use numpy::{Element,PyArray,PyArray1,PyArray2,PyArray3,PyArrayDescrMethods,PyArrayDyn,PyArrayMethods,PyReadonlyArray1,PyUntypedArray,PyUntypedArrayMethods,dtype};
use pyo3::Bound;
use pyo3::exceptions::PyTypeError;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use std::cmp::PartialOrd;
use std::iter::{DoubleEndedIterator,repeat};
use std::ops::AddAssign;

#[derive(Clone, Copy)]
pub enum Order {
    ASCENDING,
    DESCENDING
}

#[derive(Clone, Copy)]
struct ConstWeight<F: num::Float> {
    value: F
}

impl <F: num::Float> ConstWeight<F> {
    fn new(value: F) -> Self {
        return ConstWeight { value: value };
    }
    fn one() -> Self {
        return Self::new(F::one());
    }
}

pub trait Data<T: Clone>: {
    // TODO This is necessary because it seems that there is no trait like that in rust
    //      Maybe I am just not aware, but for now use my own trait.
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> + Clone;
    fn get_at(&self, index: usize) -> T;
}

pub trait SortableData<T> {
    fn argsort_unstable(&self) -> Vec<usize>;
}

impl <F: num::Float> Iterator for ConstWeight<F> {
    type Item = F;
    fn next(&mut self) -> Option<F> {
        return Some(self.value);
    }
}

impl <F: num::Float> DoubleEndedIterator for ConstWeight<F> {
    fn next_back(&mut self) -> Option<F> {
        return Some(self.value);
    }
}

impl <F: num::Float> Data<F> for ConstWeight<F> {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = F> + Clone {
        return ConstWeight::new(self.value);
    }

    fn get_at(&self, _index: usize) -> F {
        return self.value.clone();
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

impl SortableData<f64> for Vec<f64> {
    fn argsort_unstable(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|i, k| self[*k].total_cmp(&self[*i]));
        // indices.sort_unstable_by_key(|i| self[*i]);
        return indices;
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

impl SortableData<f64> for &[f64] {
    fn argsort_unstable(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|i, k| self[*k].total_cmp(&self[*i]));
        return indices;
    }
}

impl <T: Clone, const N: usize> Data<T> for [T; N] {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> + Clone {
        return self.iter().cloned();
    }
    fn get_at(&self, index: usize) -> T {
        return self[index].clone();
    }
}

impl <const N: usize> SortableData<f64> for [f64; N] {
    fn argsort_unstable(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|i, k| self[*k].total_cmp(&self[*i]));
        return indices;
    }
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
where F: num::Float + TotalOrder
{
    fn argsort_unstable(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|i, k| self[*k].total_cmp(&self[*i]));
        return indices;
    }
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

pub trait ScoreAccumulator = num::Float + AddAssign + From<bool> + From<f32>;
pub trait IntoScore<S: ScoreAccumulator> =  Into<S> + num::Float;



pub trait ScoreSortedDescending {
    fn _score<S: ScoreAccumulator>(&self, labels_with_weights: impl Iterator<Item = (S, (bool, S))> + Clone) -> S;
    fn score<S, P, B, W>(&self, labels_with_weights: impl Iterator<Item = (P, (B, W))> + Clone) -> S
    where S: ScoreAccumulator, P: IntoScore<S>, B: BinaryLabel, W: IntoScore<S>
    {
        return self._score(
            labels_with_weights.map(|(p, (b, w))| -> (S, (bool, S)) { (p.into(), (b.get_value(), w.into()))})
        )
    }
}


pub fn score_sorted_iterators<S, SA, P, B, W>(
    score: S,
    predictions: impl Iterator<Item = P> + Clone,
    labels: impl Iterator<Item = B> + Clone,
    weights: impl Iterator<Item = W> + Clone,
) -> SA
where S: ScoreSortedDescending, SA: ScoreAccumulator, P: IntoScore<SA>, B: BinaryLabel, W: IntoScore<SA> {
    let zipped = predictions.zip(labels.zip(weights));
    return score.score(zipped);
}


pub fn score_sorted_sample<S, SA, P, B, W>(
    score: S,
    predictions: &impl Data<P>,
    labels: &impl Data<B>,
    weights: &impl Data<W>,
    order: Order,
) -> SA
where S: ScoreSortedDescending, SA: ScoreAccumulator, P: IntoScore<SA>, B: BinaryLabel, W: IntoScore<SA> + Clone {
    let p = predictions.get_iterator();
    let l = labels.get_iterator();
    let w = weights.get_iterator();
    return match order {
        Order::ASCENDING => score_sorted_iterators(score, p.rev(), l.rev(), w.rev()),
        Order::DESCENDING => score_sorted_iterators(score, p, l, w),
    };
}


pub fn score_maybe_sorted_sample<S, SA, P, B, W>(
    score: S,
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
    order: Option<Order>,
) -> SA
where S: ScoreSortedDescending, SA: ScoreAccumulator, P: IntoScore<SA>, B: BinaryLabel, W: IntoScore<SA> + Clone
{
    return match order {
        Some(o) => {
            match weights {
                Some(w) => score_sorted_sample(score, predictions, labels, w, o),
                None => score_sorted_sample(score, predictions, labels, &ConstWeight::<W>::one(), o),
            }
        }
        None => {
            let indices = predictions.argsort_unstable();
            let sorted_labels = select(labels, &indices);
            let sorted_predictions = select(predictions, &indices);
            match weights {
                Some(w) => {
                    let sorted_weights = select(w, &indices);
                    score_sorted_sample(score, &sorted_predictions, &sorted_labels, &sorted_weights, Order::DESCENDING)
                }
                None => score_sorted_sample(score, &sorted_predictions, &sorted_labels, &ConstWeight::<W>::one(), Order::DESCENDING)
            }
        }
    };
}


pub fn score_sample<S, SA, P, B, W>(
    score: S,
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
) -> SA

where S: ScoreSortedDescending, SA: ScoreAccumulator, P: IntoScore<SA>, B: BinaryLabel, W: IntoScore<SA> + Clone {
    return score_maybe_sorted_sample(score, predictions, labels, weights, None);
}


pub fn score_two_sorted_samples<S, SA, P, B, W>(
    score: S,
    predictions1: impl Iterator<Item = P> + Clone,
    label1: impl Iterator<Item = B> + Clone,
    weight1: impl Iterator<Item = W> + Clone,
    predictions2: impl Iterator<Item = P> + Clone,
    label2: impl Iterator<Item = B> + Clone,
    weight2: impl Iterator<Item = W> + Clone,
) -> SA
where S: ScoreSortedDescending, SA: ScoreAccumulator, P: IntoScore<SA>, B: BinaryLabel + PartialOrd, W: IntoScore<SA>
{
    return score_two_sorted_samples_zipped(
        score,
        predictions1.zip(label1.zip(weight1)),
        predictions2.zip(label2.zip(weight2)),
    );
}


pub fn score_two_sorted_samples_zipped<S, SA, P, B, W>(
    score: S,
    iter1: impl Iterator<Item = (P, (B, W))> + Clone,
    iter2: impl Iterator<Item = (P, (B, W))> + Clone,
) -> SA
where S: ScoreSortedDescending, SA: ScoreAccumulator, P: IntoScore<SA>, B: BinaryLabel + PartialOrd, W: IntoScore<SA>
{
    let combined_iter = combine::combine::CombineIterDescending::new(iter1, iter2);
    return score.score(combined_iter);
}


struct AveragePrecision {
    
}


impl AveragePrecision {
    fn new() -> Self {
        return AveragePrecision{};
    }
}


#[derive(Clone,Copy,Debug)]
struct Positives<P>
where P: num::Float + From<bool> + AddAssign
{
    tps: P,
    fps: P,
}

impl <P> Positives<P>
where P: num::Float + From<bool> + AddAssign
{
    fn new(tps: P, fps: P) -> Self {
        return Positives { tps, fps };
    }

    fn zero() -> Self {
        return Positives::new(P::zero(), P::zero());
    }

    fn add(&mut self, label: bool, weight: P) {
        let label: P = label.into();
        let tp = weight * label;
        let fp = weight - tp;  // (weight*(1 -label) = weight - weight * label = weight - tp)
        self.tps += tp;
        self.fps += fp;
    }

    fn positives_sum(&self) -> P {
        return self.tps + self.fps;
    }

    fn precision(&self) -> P {
        return self.tps / self.positives_sum();
    }
}


impl ScoreSortedDescending for AveragePrecision {
    fn _score<S: ScoreAccumulator>(&self, mut labels_with_weights: impl Iterator<Item = (S, (bool, S))> + Clone) -> S
    {
        let mut positives: Positives<S> = Positives::zero();
        let mut last_p: S = f32::NAN.into();
        let mut last_tps: S = S::zero();
        let mut ap: S = S::zero();

        // TODO can we unify this preparation step with the loop?
        match labels_with_weights.next() {
            None => (), // TODO: Sohuld we return an error in this case?
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
            positives.add(label.get_value(), w.into());
        }

        ap += (positives.tps - last_tps) * positives.precision();
        
        // Special case for tps == 0 following sklearn
        // https://github.com/scikit-learn/scikit-learn/blob/5cce87176a530d2abea45b5a7e5a4d837c481749/sklearn/metrics/_ranking.py#L1032-L1039
        // I.e. if tps is 0.0, there are no positive samples in labels: Either all labels are 0, or all weights (for positive labels) are 0
        return if positives.tps == S::zero() {
            S::zero()
        } else {
            ap / positives.tps
        };
    }
}


struct RocAuc {

}


impl RocAuc {
    fn new() -> Self {
        return RocAuc { };
    }
}


impl ScoreSortedDescending for RocAuc {
    fn _score<S: ScoreAccumulator>(&self, mut labels_with_weights: impl Iterator<Item = (S, (bool, S))> + Clone) -> S
    {
        let mut positives: Positives<S> = Positives::zero();
        let mut last_p: S = f32::NAN.into();
        let mut last_counted_fp = S::zero();
        let mut last_counted_tp = S::zero();
        let mut area_under_curve = S::zero();

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
                    last_counted_fp,
                    positives.fps,
                    last_counted_tp,
                    positives.tps,
                );
                last_counted_fp = positives.fps;
                last_counted_tp = positives.tps;
                last_p = p;
            }
            positives.add(label, w);
        }
        area_under_curve += area_under_line_segment(
            last_counted_fp,
            positives.fps,
            last_counted_tp,
            positives.tps,
        );
        return area_under_curve / (positives.tps * positives.fps);
    }
}


struct RocAucWithMaxFPR {
    max_fpr: f32,
}


impl RocAucWithMaxFPR {
    fn new(max_fpr: f32) -> Self {
        return RocAucWithMaxFPR { max_fpr };
    }

    fn get_positive_sum<B, W>(labels_with_weights: impl Iterator<Item = (B, W)>) -> Positives<W>
    where B: BinaryLabel, W: num::Float + From::<bool> + AddAssign
    {
        let mut positives: Positives<W>  = Positives::zero();
        for (label, weight) in labels_with_weights {
            positives.add(label.get_value(), weight);
        }
        return positives;
    }
}


impl ScoreSortedDescending for RocAucWithMaxFPR {
    fn _score<S: ScoreAccumulator>(&self, mut labels_with_weights: impl Iterator<Item = (S, (bool, S))> + Clone) -> S
    {
        let total_positives = Self::get_positive_sum(labels_with_weights.clone().map(|(_a, b)| b));
        let max_fpr: S = self.max_fpr.into();
        let false_positive_cutoff = max_fpr * total_positives.fps;

        let mut positives: Positives<S> = Positives::zero();
        let mut last_p: S = f32::NAN.into();
        let mut last_counted_fp = S::zero();
        let mut last_counted_tp = S::zero();
        let mut area_under_curve = S::zero();

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
                    last_counted_fp,
                    positives.fps,
                    last_counted_tp,
                    positives.tps,
                );
                last_counted_fp = positives.fps;
                last_counted_tp = positives.tps;
                last_p = p;
            }
            let mut next_pos = positives.clone();
            next_pos.add(label, w);
            if next_pos.fps > false_positive_cutoff {
                let dx = next_pos.fps - positives.fps;
                let dy = next_pos.tps - positives.tps;
                positives = Positives::new(
                    positives.tps + dy * false_positive_cutoff / dx,
                    false_positive_cutoff,
                );
                break;
            }
            else {
                positives = next_pos;
            }
        }

        area_under_curve += area_under_line_segment(
            last_counted_fp,
            positives.fps,
            last_counted_tp,
            positives.tps,
        );
        
        let normalized_area_under_curve = area_under_curve / (total_positives.tps * total_positives.fps);
        let one_half: S = 0.5f32.into(); 
        let min_area = one_half * max_fpr * max_fpr;
        let max_area = max_fpr;
        return one_half * (S::one() + (normalized_area_under_curve - min_area) / (max_area - min_area));
    }
}


struct RocAucWithOptionalMaxFPR {
    // TODO: Can we have a single implementation for this and RocAuc?
    //       This would add an unncessary check to RocAuc but performance
    //       penalty may be negligible.
    max_fpr: Option<f32>,
}

impl RocAucWithOptionalMaxFPR {
    fn new(max_fpr: Option<f32>) -> Self {
        return Self { max_fpr };
    }
}


impl ScoreSortedDescending for RocAucWithOptionalMaxFPR {
    fn _score<S: ScoreAccumulator>(&self, labels_with_weights: impl Iterator<Item = (S, (bool, S))> + Clone) -> S
    {
        return match self.max_fpr {
            Some(mfpr) => RocAucWithMaxFPR::new(mfpr).score(labels_with_weights),
            None => RocAuc::new().score(labels_with_weights),
        }
    }
}


pub fn average_precision<S, P, B, W>(
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
    order: Option<Order>,
) -> S
where S: ScoreAccumulator, P: IntoScore<S>, B: BinaryLabel, W: IntoScore<S> + Clone
{
    return score_maybe_sorted_sample(AveragePrecision::new(), predictions, labels, weights, order);
}


pub fn roc_auc<S, P, B, W>(
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
    order: Option<Order>,
    max_fpr: Option<f32>,
) -> S
where S: ScoreAccumulator, P: IntoScore<S>, B: BinaryLabel, W: IntoScore<S> + Clone
{
    return score_maybe_sorted_sample(RocAucWithOptionalMaxFPR::new(max_fpr), predictions, labels, weights, order);
}


fn area_under_line_segment<P>(x0: P, x1: P, y0: P, y1: P) -> P
where P: num::Float + From<f32>
{
    let dx = x1 - x0;
    let dy = y1 - y0;
    let one_half: P = 0.5f32.into();
    return dx * y0 + dy * dx * one_half;
}


pub fn loo_cossim<F: num::Float + AddAssign>(mat: &ArrayView2<'_, F>, replicate_sum: &mut ArrayViewMut1<'_, F>) -> F {
    let num_replicates = mat.shape()[0];
    let loo_weight = F::from(num_replicates - 1).unwrap();
    let loo_weight_factor = F::from(1).unwrap() / loo_weight;
    for mat_replicate in mat.outer_iter() {
        for (feature, feature_sum) in mat_replicate.iter().zip(replicate_sum.iter_mut()) {
            *feature_sum += *feature;
        }
    }

    let mut result = F::zero();

    for mat_replicate in mat.outer_iter() {
        let mut m_sqs = F::zero();
        let mut l_sqs = F::zero();
        let mut prod_sum = F::zero();
        for (feature, feature_sum) in mat_replicate.iter().zip(replicate_sum.iter()) {
            let m_f = *feature;
            let l_f = (*feature_sum - *feature) * loo_weight_factor;
            prod_sum += m_f * l_f;
            m_sqs += m_f * m_f;
            l_sqs += l_f * l_f;
        }
        result += prod_sum / (m_sqs * l_sqs).sqrt();
    }

    return result / F::from(num_replicates).unwrap();
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
#[pyclass(eq, eq_int, name="Order")]
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
    ) -> P
    where P: ScoreAccumulator + Element + TotalOrder, B: BinaryLabel + Element, W: IntoScore<P> + Element
    {
        let labels = labels.as_array();
        let predictions = predictions.as_array();
        let order = order.map(py_order_as_order);
        let score = match weights {
            Some(weight) => {
                let w = weight.as_array();
                py.allow_threads(move || {
                    score_maybe_sorted_sample(self.get_score(), &predictions, &labels, Some(&w), order)
                })
            },
            None => py.allow_threads(move || {
                score_maybe_sorted_sample(self.get_score(), &predictions, &labels, None::<&Vec<W>>, order)
            })
        };
        return score;
    }

    fn score_two_sorted_samples_py_generic<'py, B, F, W, B1, B2, F1, F2, W1, W2>(
        &self,
        py: Python<'py>,
        labels1: PyReadonlyArray1<'py, B1>,
        predictions1: PyReadonlyArray1<'py, F1>,
        weights1: Option<PyReadonlyArray1<'py, W1>>,
        labels2: PyReadonlyArray1<'py, B1>,
        predictions2: PyReadonlyArray1<'py, F2>,
        weights2: Option<PyReadonlyArray1<'py, W2>>,
    ) -> F
    where B: BinaryLabel + PartialOrd, F: ScoreAccumulator + TotalOrder + Ungil, W: IntoScore<F>, B1: Element + Into<B> + Clone, B2: Element + Into<B> + Clone, F1: Element + Into<F> + Clone, F2: Element + Into<F> + Clone, W1: Element + Into<W> + Clone, W2: Element + Into<W> + Clone
    {
        let l1 = labels1.as_array().into_iter().cloned().map(|l| -> B { l.into() });
        let l2 = labels2.as_array().into_iter().cloned().map(|l| -> B { l.into() });
        let p1 = predictions1.as_array().into_iter().cloned().map(|f| -> F { f.into() });
        let p2 = predictions2.as_array().into_iter().cloned().map(|f| -> F { f.into() });


        return match (weights1, weights2) {
            (None, None) => {
                py.allow_threads(move || {
                    score_two_sorted_samples(self.get_score(), p1, l1, repeat(W::one()), p2, l2, repeat(W::one()))
                })
            }
            (Some(w1), None) => {
                let w1i = w1.as_array().into_iter().cloned().map(|w| -> W { w.into() });
                py.allow_threads(move || {
                    score_two_sorted_samples(self.get_score(), p1, l1, w1i, p2, l2, repeat(W::one()))
                })
            }
            (None, Some(w2)) => {
                let w2i = w2.as_array().into_iter().cloned().map(|w| -> W { w.into() });
                py.allow_threads(move || {
                    score_two_sorted_samples(self.get_score(), p1, l1, repeat(W::one()), p2, l2, w2i)
                })
            }
            (Some(w1), Some(w2)) =>  {
                let w1i = w1.as_array().into_iter().cloned().map(|w| -> W { w.into() });
                let w2i = w2.as_array().into_iter().cloned().map(|w| -> W { w.into() });
                py.allow_threads(move || {
                    score_two_sorted_samples(self.get_score(), p1, l1, w1i, p2, l2, w2i)
                })
            }
        };
    }
}

struct AveragePrecisionPyGeneric {

}

impl AveragePrecisionPyGeneric {
    fn new() -> Self {
        return AveragePrecisionPyGeneric {};
    }
}

impl PyScoreGeneric<AveragePrecision> for AveragePrecisionPyGeneric {
    fn get_score(&self) -> AveragePrecision {
        return AveragePrecision::new();
    }
}

struct RocAucPyGeneric {
    max_fpr: Option<f32>,
}

impl RocAucPyGeneric {
    fn new(max_fpr: Option<f32>) -> Self {
        return RocAucPyGeneric { max_fpr: max_fpr };
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
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty, $weight_type:ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (labels, predictions, *, weights=None, order=None))]
        pub fn $fname<'py>(
            py: Python<'py>,
            labels: PyReadonlyArray1<'py, $label_type>,
            predictions: PyReadonlyArray1<'py, $prediction_type>,
            weights: Option<PyReadonlyArray1<'py, $weight_type>>,
            order: Option<PyOrder>
        ) -> $prediction_type
        {
            return AveragePrecisionPyGeneric::new().score_py(py, labels, predictions, weights, order);
        }
    };
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty, $weight_type:ty, $py_module:ident) => {
        average_precision_py!($fname, $pyname, $label_type, $prediction_type, $weight_type);
        $py_module.add_function(wrap_pyfunction!($fname, $py_module)?).unwrap();
    };
}


macro_rules! roc_auc_py {
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty, $weight_type:ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (labels, predictions, *, weights=None, order=None, max_fpr=None))]
        pub fn $fname<'py>(
            py: Python<'py>,
            labels: PyReadonlyArray1<'py, $label_type>,
            predictions: PyReadonlyArray1<'py, $prediction_type>,
            weights: Option<PyReadonlyArray1<'py, $weight_type>>,
            order: Option<PyOrder>,
            max_fpr: Option<f32>,
        ) -> $prediction_type
        {
            return RocAucPyGeneric::new(max_fpr).score_py(py, labels, predictions, weights, order);
        }
    };
    ($fname: ident, $pyname:literal, $label_type:ty, $prediction_type:ty, $weight_type: ty, $py_module:ident) => {
        roc_auc_py!($fname, $pyname, $label_type, $prediction_type, $weight_type);
        $py_module.add_function(wrap_pyfunction!($fname, $py_module)?).unwrap();
    };
}


macro_rules! average_precision_on_two_sorted_samples_py {
    ($fname: ident, $pyname:literal, $lt:ty, $pt:ty, $wt:ty, $lt1:ty, $pt1:ty, $wt1:ty, $lt2:ty, $pt2:ty, $wt2: ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2, *))]
        pub fn $fname<'py>(
            py: Python<'py>,
            labels1: PyReadonlyArray1<'py, $lt1>,
            predictions1: PyReadonlyArray1<'py, $pt1>,
            weights1: Option<PyReadonlyArray1<'py, $wt1>>,
            labels2: PyReadonlyArray1<'py, $lt2>,
            predictions2: PyReadonlyArray1<'py, $pt2>,
            weights2: Option<PyReadonlyArray1<'py, $wt2>>,
        ) -> $pt
        {
            return AveragePrecisionPyGeneric::new().score_two_sorted_samples_py_generic::<$lt, $pt, $wt, $lt1, $lt2, $pt1, $pt2, $wt1, $wt2>(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
        }
    };
    ($fname: ident, $pyname:literal, $lt:ty, $pt:ty, $wt:ty, $lt1:ty, $pt1:ty, $wt1:ty, $lt2:ty, $pt2:ty, $wt2: ty, $py_module:ident) => {
        average_precision_on_two_sorted_samples_py!($fname, $pyname, $lt, $pt, $wt, $lt1, $pt1, $wt1, $lt2, $pt2, $wt2);
        $py_module.add_function(wrap_pyfunction!($fname, $py_module)?).unwrap();
    };
}


macro_rules! roc_auc_on_two_sorted_samples_py {
    ($fname: ident, $pyname:literal, $lt:ty, $pt:ty, $wt:ty, $lt1:ty, $pt1:ty, $wt1:ty, $lt2:ty, $pt2:ty, $wt2: ty) => {
        #[pyfunction(name = $pyname)]
        #[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2, *, max_fpr=None))]
        pub fn $fname<'py>(
            py: Python<'py>,
            labels1: PyReadonlyArray1<'py, $lt1>,
            predictions1: PyReadonlyArray1<'py, $pt1>,
            weights1: Option<PyReadonlyArray1<'py, $wt1>>,
            labels2: PyReadonlyArray1<'py, $lt2>,
            predictions2: PyReadonlyArray1<'py, $pt2>,
            weights2: Option<PyReadonlyArray1<'py, $wt2>>,
            max_fpr: Option<f32>,
        ) -> $pt
        {
            return RocAucPyGeneric::new(max_fpr).score_two_sorted_samples_py_generic::<$lt, $pt, $wt, $lt1, $lt2, $pt1, $pt2, $wt1, $wt2>(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
        }
    };
    ($fname: ident, $pyname:literal, $lt:ty, $pt:ty, $wt:ty, $lt1:ty, $pt1:ty, $wt1:ty, $lt2:ty, $pt2:ty, $wt2: ty, $py_module:ident) => {
        roc_auc_on_two_sorted_samples_py!($fname, $pyname, $lt, $pt, $wt, $lt1, $pt1, $wt1, $lt2, $pt2, $wt2);
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
        let typed_data = data.downcast::<PyArray2<f32>>().unwrap().readonly();
        let array = typed_data.as_array();
        let score = py.allow_threads(move || {
            loo_cossim_single(&array)
        });
        return Ok(score as f64);
    }
    if dt.is_equiv_to(&dtype::<f64>(py)) {
        let typed_data = data.downcast::<PyArray2<f64>>().unwrap().readonly();
        let array = typed_data.as_array();
        let score = py.allow_threads(move || {
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
    let typed_data = data.downcast::<PyArray3<F>>().unwrap().readonly();
    let array = typed_data.as_array();
    let score = py.allow_threads(move || {
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
    let typed_data = data.downcast::<PyArrayDyn<f64>>().unwrap();
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
    let typed_data = data.downcast::<PyArrayDyn<f32>>().unwrap();
    return loo_cossim_many_generic_py(py, typed_data);
}

#[pymodule(name = "_scors")]
fn scors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    average_precision_py!(average_precision_bool_f32, "average_precision_bool_f32", bool, f32, f32, m);
    average_precision_py!(average_precision_i8_f32, "average_precision_i8_f32", i8, f32, f32, m);
    average_precision_py!(average_precision_i16_f32, "average_precision_i16_f32", i16, f32, f32, m);
    average_precision_py!(average_precision_i32_f32, "average_precision_i32_f32", i32, f32, f32, m);
    average_precision_py!(average_precision_i64_f32, "average_precision_i64_f32", i64, f32, f32, m);
    average_precision_py!(average_precision_u8_f32, "average_precision_u8_f32", u8, f32, f32, m);
    average_precision_py!(average_precision_u16_f32, "average_precision_u16_f32", u16, f32, f32, m);
    average_precision_py!(average_precision_u32_f32, "average_precision_u32_f32", u32, f32, f32, m);
    average_precision_py!(average_precision_u64_f32, "average_precision_u64_f32", u64, f32, f32, m);
    average_precision_py!(average_precision_bool_f64, "average_precision_bool_f64", bool, f64, f64, m);
    average_precision_py!(average_precision_i8_f64, "average_precision_i8_f64", i8, f64, f64, m);
    average_precision_py!(average_precision_i16_f64, "average_precision_i16_f64", i16, f64, f64, m);
    average_precision_py!(average_precision_i32_f64, "average_precision_i32_f64", i32, f64, f64, m);
    average_precision_py!(average_precision_i64_f64, "average_precision_i64_f64", i64, f64, f64, m);
    average_precision_py!(average_precision_u8_f64, "average_precision_u8_f64", u8, f64, f64, m);
    average_precision_py!(average_precision_u16_f64, "average_precision_u16_f64", u16, f64, f64, m);
    average_precision_py!(average_precision_u32_f64, "average_precision_u32_f64", u32, f64, f64, m);
    average_precision_py!(average_precision_u64_f64, "average_precision_u64_f64", u64, f64, f64, m);

    roc_auc_py!(roc_auc_bool_f32, "roc_auc_bool_f32", bool, f32, f32, m);
    roc_auc_py!(roc_auc_i8_f32, "roc_auc_i8_f32", i8, f32, f32, m);
    roc_auc_py!(roc_auc_i16_f32, "roc_auc_i16_f32", i16, f32, f32, m);
    roc_auc_py!(roc_auc_i32_f32, "roc_auc_i32_f32", i32, f32, f32, m);
    roc_auc_py!(roc_auc_i64_f32, "roc_auc_i64_f32", i64, f32, f32, m);
    roc_auc_py!(roc_auc_u8_f32, "roc_auc_u8_f32", u8, f32, f32, m);
    roc_auc_py!(roc_auc_u16_f32, "roc_auc_u16_f32", u16, f32, f32, m);
    roc_auc_py!(roc_auc_u32_f32, "roc_auc_u32_f32", u32, f32, f32, m);
    roc_auc_py!(roc_auc_u64_f32, "roc_auc_u64_f32", u64, f32, f32, m);
    roc_auc_py!(roc_auc_bool_f64, "roc_auc_bool_f64", bool, f64, f64, m);
    roc_auc_py!(roc_auc_i8_f64, "roc_auc_i8_f64", i8, f64, f64, m);
    roc_auc_py!(roc_auc_i16_f64, "roc_auc_i16_f64", i16, f64, f64, m);
    roc_auc_py!(roc_auc_i32_f64, "roc_auc_i32_f64", i32, f64, f64, m);
    roc_auc_py!(roc_auc_i64_f64, "roc_auc_i64_f64", i64, f64, f64, m);
    roc_auc_py!(roc_auc_u8_f64, "roc_auc_u8_f64", u8, f64, f64, m);
    roc_auc_py!(roc_auc_u16_f64, "roc_auc_u16_f64", u16, f64, f64, m);
    roc_auc_py!(roc_auc_u32_f64, "roc_auc_u32_f64", u32, f64, f64, m);
    roc_auc_py!(roc_auc_u64_f64, "roc_auc_u64_f64", u64, f64, f64, m);

    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_bool_f32, "average_precision_on_two_sorted_samples_bool_f32", bool, f32, f32, bool, f32, f32, bool, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i8_f32, "average_precision_on_two_sorted_samples_i8_f32", i8, f32, f32, i8, f32, f32, i8, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i16_f32, "average_precision_on_two_sorted_samples_i16_f32", i16, f32, f32, i16, f32, f32, i16, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i32_f32, "average_precision_on_two_sorted_samples_i32_f32", i32, f32, f32, i32, f32, f32, i32, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i64_f32, "average_precision_on_two_sorted_samples_i64_f32", i64, f32, f32, i64, f32, f32, i64, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u8_f32, "average_precision_on_two_sorted_samples_u8_f32", u8, f32, f32, u8, f32, f32, u8, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u16_f32, "average_precision_on_two_sorted_samples_u16_f32", u16, f32, f32, u16, f32, f32, u16, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u32_f32, "average_precision_on_two_sorted_samples_u32_f32", u32, f32, f32, u32, f32, f32, u32, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u64_f32, "average_precision_on_two_sorted_samples_u64_f32", u64, f32, f32, u64, f32, f32, u64, f32, f32, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_bool_f64, "average_precision_on_two_sorted_samples_bool_f64", bool, f64, f64, bool, f64, f64, bool, f64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i8_f64, "average_precision_on_two_sorted_samples_i8_f64", i8, f64, f64, i8, f64, f64, i8, f64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i16_f64, "average_precision_on_two_sorted_samples_i16_f64", i16, f64, f64, i16, f64, f64, i16, f64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i32_f64, "average_precision_on_two_sorted_samples_i32_f64", i32, f64, f64, i16, f64, f64, i16, f64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_i64_f64, "average_precision_on_two_sorted_samples_i64_f64", i64, f64, f64, i64, f64, f64, i64, f64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u8_f64, "average_precision_on_two_sorted_samples_u8_f64", u8, f64, f64, u8, f64, f64, u8, f64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u16_f64, "average_precision_on_two_sorted_samples_u16_f64", u16, f64, f64, u16, f64, f64, u16, f64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u32_f64, "average_precision_on_two_sorted_samples_u32_f64", u32, f64, f64, u32, f64, f64, u32, f64, f64, m);
    average_precision_on_two_sorted_samples_py!(average_precision_on_two_sorted_samples_u64_f64, "average_precision_on_two_sorted_samples_u64_f64", u64, f64, f64, u64, f64, f64, u64, f64, f64, m);

    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_bool_f32, "roc_auc_on_two_sorted_samples_bool_f32", bool, f32, f32, bool, f32, f32, bool, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i8_f32, "roc_auc_on_two_sorted_samples_i8_f32", i8, f32, f32, i8, f32, f32, i8, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i16_f32, "roc_auc_on_two_sorted_samples_i16_f32", i16, f32, f32, i16, f32, f32, i16, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i32_f32, "roc_auc_on_two_sorted_samples_i32_f32", i32, f32, f32, i32, f32, f32, i32, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i64_f32, "roc_auc_on_two_sorted_samples_i64_f32", i64, f32, f32, i64, f32, f32, i64, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u8_f32, "roc_auc_on_two_sorted_samples_u8_f32", u8, f32, f32, u8, f32, f32, u8, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u16_f32, "roc_auc_on_two_sorted_samples_u16_f32", u16, f32, f32, u16, f32, f32, u16, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u32_f32, "roc_auc_on_two_sorted_samples_u32_f32", u32, f32, f32, u32, f32, f32, u32, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u64_f32, "roc_auc_on_two_sorted_samples_u64_f32", u64, f32, f32, u64, f32, f32, u64, f32, f32, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_bool_f64, "roc_auc_on_two_sorted_samples_bool_f64", bool, f64, f64, bool, f64, f64, bool, f64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i8_f64, "roc_auc_on_two_sorted_samples_i8_f64", i8, f64, f64, i8, f64, f64, i8, f64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i16_f64, "roc_auc_on_two_sorted_samples_i16_f64", i16, f64, f64, i16, f64, f64, i16, f64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i32_f64, "roc_auc_on_two_sorted_samples_i32_f64", i32, f64, f64, i16, f64, f64, i16, f64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_i64_f64, "roc_auc_on_two_sorted_samples_i64_f64", i64, f64, f64, i64, f64, f64, i64, f64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u8_f64, "roc_auc_on_two_sorted_samples_u8_f64", u8, f64, f64, u8, f64, f64, u8, f64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u16_f64, "roc_auc_on_two_sorted_samples_u16_f64", u16, f64, f64, u16, f64, f64, u16, f64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u32_f64, "roc_auc_on_two_sorted_samples_u32_f64", u32, f64, f64, u32, f64, f64, u32, f64, f64, m);
    roc_auc_on_two_sorted_samples_py!(roc_auc_on_two_sorted_samples_u64_f64, "roc_auc_on_two_sorted_samples_u64_f64", u64, f64, f64, u64, f64, f64, u64, f64, f64, m);

    m.add_function(wrap_pyfunction!(loo_cossim_py, m)?).unwrap();
    m.add_function(wrap_pyfunction!(loo_cossim_many_py_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(loo_cossim_many_py_f32, m)?).unwrap();
    m.add_class::<PyOrder>().unwrap();
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
        let actual: f64 = score_sorted_sample(AveragePrecision::new(), &predictions, &labels, &weights, Order::DESCENDING);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_on_sorted_double() {
        let labels: [u8; 8] = [1, 1, 0, 0, 1, 1, 0, 0];
        let predictions: [f64; 8] = [0.8, 0.8, 0.4, 0.4, 0.35, 0.35, 0.1, 0.1];
        let weights: [f64; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let actual: f64 = score_sorted_sample(AveragePrecision::new(), &predictions, &labels, &weights, Order::DESCENDING);
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
            AveragePrecision::new(),
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
            RocAuc::new(),
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
            RocAucWithMaxFPR::new(0.25),
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
