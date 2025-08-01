mod combine;

use ndarray::{Array1,ArrayView1,ArrayView,ArrayView2,ArrayView3,ArrayViewMut1,Ix1};
use num;
use num::traits::float::TotalOrder;
use numpy::{Element,PyArray,PyArray1,PyArray2,PyArray3,PyArrayDescr,PyArrayDescrMethods,PyArrayDyn,PyArrayMethods,PyReadonlyArray1,PyUntypedArray,PyUntypedArrayMethods,dtype};
use pyo3::Bound;
use pyo3::exceptions::PyTypeError;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use std::cmp::PartialOrd;
use std::iter::{DoubleEndedIterator,repeat};
use std::marker::PhantomData;
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
        // let t0 = Instant::now();
        let mut indices: Vec<usize> = (0..self.len()).collect::<Vec<_>>();
        // println!("Creating indices took {}ms", t0.elapsed().as_millis());
        // let t1 = Instant::now();
        indices.sort_unstable_by(|i, k| self[*k].total_cmp(&self[*i]));
        // println!("Sorting took {}ms", t0.elapsed().as_millis());
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
        return (self & 1) == 1;
    }
}

impl BinaryLabel for u16 {
    fn get_value(&self) -> bool {
        return (self & 1) == 1;
    }
}

impl BinaryLabel for u32 {
    fn get_value(&self) -> bool {
        return (self & 1) == 1;
    }
}

impl BinaryLabel for u64 {
    fn get_value(&self) -> bool {
        return (self & 1) == 1;
    }
}

impl BinaryLabel for i8 {
    fn get_value(&self) -> bool {
        return (self & 1) == 1;
    }
}

impl BinaryLabel for i16 {
    fn get_value(&self) -> bool {
        return (self & 1) == 1;
    }
}

impl BinaryLabel for i32 {
    fn get_value(&self) -> bool {
        return (self & 1) == 1;
    }
}

impl BinaryLabel for i64 {
    fn get_value(&self) -> bool {
        return (self & 1) == 1;
    }
}

struct SortedSampleDescending<'a, B, L, P, W>
where B: BinaryLabel + Clone + 'a, &'a L: IntoIterator<Item = &'a B>, &'a P: IntoIterator<Item = &'a f64>, &'a W: IntoIterator<Item = &'a f64>
{
    labels: &'a L,
    predictions: &'a P,
    weights: &'a W,
    label_type: PhantomData<B>,
}

impl <'a, B, L, P, W> SortedSampleDescending<'a, B, L, P, W>
where B: BinaryLabel + Clone + 'a, &'a L: IntoIterator<Item = &'a B>, &'a P: IntoIterator<Item = &'a f64>, &'a W: IntoIterator<Item = &'a f64>
{
    fn new(labels: &'a L, predictions: &'a P, weights: &'a W) -> Self {
        return SortedSampleDescending {
            labels: labels,
            predictions: predictions,
            weights: weights,
            label_type: PhantomData
        }
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


pub fn average_precision_on_sorted_labels<B, L, F, W>(labels: &L, weights: Option<&W>, order: Order) -> f64
where B: BinaryLabel, L: Data<B>, F: num::Float, W: Data<F>
{
    return match weights {
        None => average_precision_on_iterator(labels.get_iterator(), ConstWeight::<F>::one(), order),
        Some(w) => average_precision_on_iterator(labels.get_iterator(), w.get_iterator(), order)
    };
}

pub fn average_precision_on_iterator<B, L, F, W>(labels: L, weights: W, order: Order) -> f64
where B: BinaryLabel, L: DoubleEndedIterator<Item = B>, F: num::Float, W: DoubleEndedIterator<Item = F>
{
    return match order {
        Order::ASCENDING => average_precision_on_descending_iterator(labels.rev(), weights.rev()),
        Order::DESCENDING => average_precision_on_descending_iterator(labels, weights)
    };
}

pub fn average_precision_on_descending_iterator<B: BinaryLabel, F: num::Float>(labels: impl Iterator<Item = B>, weights: impl Iterator<Item = F>) -> f64 {
    return average_precision_on_descending_iterators(labels.zip(weights));
}

pub fn average_precision_on_sorted_samples<'a, B, L, F, P, W>(l1: &'a L, p1: &'a P, w1: &'a W, l2: &'a L, p2: &'a P, w2: &'a W) -> f64
where B: BinaryLabel + Clone + 'a, &'a L: IntoIterator<Item = &'a B>, F: num::Float + Clone + 'a, &'a P: IntoIterator<Item = &'a F>, &'a W: IntoIterator<Item = &'a F>
{
    // let mut it1 = p1.into_iter();
    let i1 = p1.into_iter().cloned().zip(l1.into_iter().cloned().zip(w1.into_iter().cloned()));
    let i2 = p2.into_iter().cloned().zip(l2.into_iter().cloned().zip(w2.into_iter().cloned()));
    let labels_and_weights = i1.zip(i2).map(|(t1, t2)| {
        if t1.0 > t2.0 {
            t1.1
        } else {
            t2.1
        }
    });
    return average_precision_on_descending_iterators(labels_and_weights);
}



trait ScoreSortedDescending {
    fn score<B: BinaryLabel>(&self, labels_with_weights: impl Iterator<Item = (f64, (B, f64))> + Clone) -> f64;
    fn score_generic<P, B, W>(&self, labels_with_weights: impl Iterator<Item = (P, (B, W))> + Clone) -> f64
    where P: num::Float + Into<f64>, B: BinaryLabel, W: num::Float + Into<f64>
    {
        return self.score(labels_with_weights.map(|(p, (b, w))| (p.into(), (b, w.into()))));
    }
}

pub fn score_sorted_iterators<S, P, B, W>(
    score: S,
    predictions: impl Iterator<Item = P> + Clone,
    labels: impl Iterator<Item = B> + Clone,
    weights: impl Iterator<Item = W> + Clone,
) -> f64
where S: ScoreSortedDescending, P: num::Float + Into<f64>, B: BinaryLabel, W: num::Float + Into<f64> {
    let mut zipped = predictions.zip(labels.zip(weights));
    return score.score_generic(zipped);
}

pub fn score_sorted_sample<S, P, B, W>(
    score: S,
    predictions: &impl Data<P>,
    labels: &impl Data<B>,
    weights: &impl Data<W>,
    order: Order,
) -> f64
where S: ScoreSortedDescending, P: num::Float + Into<f64>, B: BinaryLabel, W: num::Float + Into<f64> {
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
where S: ScoreSortedDescending, P: num::Float + Into<f64>, B: BinaryLabel, W: num::Float + Into<f64>
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

pub fn score_sample<S, P, B, W>(
    score: S,
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
) -> f64
where S: ScoreSortedDescending, P: num::Float + Into<f64>, B: BinaryLabel, W: num::Float + Into<f64> {
    return score_maybe_sorted_sample(score, predictions, labels, weights, None);
}

struct AveragePrecision {
    
}

impl AveragePrecision {
    fn new() -> Self {
        return AveragePrecision{};
    }
}

impl ScoreSortedDescending for AveragePrecision {
    fn score<B: BinaryLabel>(&self, labels_with_weights: impl Iterator<Item = (f64, (B, f64))> + Clone) -> f64 {
        let mut ap: f64 = 0.0;
        let mut tps: f64 = 0.0;
        let mut fps: f64 = 0.0;
        for (_, (label, weight)) in labels_with_weights {
            let w: f64 = weight.into();
            let l: bool = label.get_value();
            let tp = w * f64::from(l);
            tps += tp;
            fps += w - tp;
            let ps = tps + fps;
            let precision = tps / ps;
            ap += tp * precision;
        }
        // Special case for tps == 0 following sklearn
        // https://github.com/scikit-learn/scikit-learn/blob/5cce87176a530d2abea45b5a7e5a4d837c481749/sklearn/metrics/_ranking.py#L1032-L1039
        // I.e. if tps is 0.0, there are no positive samples in labels: Either all labels are 0, or all weights (for positive labels) are 0
        return if tps == 0.0 {
            0.0
        } else {
            ap / tps
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
    fn score<B: BinaryLabel>(&self, labels_with_weights: impl Iterator<Item = (f64, (B, f64))> + Clone) -> f64 {
        let mut false_positives: f64 = 0.0;
        let mut true_positives: f64 = 0.0;
        let mut last_counted_fp = 0.0;
        let mut last_counted_tp = 0.0;
        let mut area_under_curve = 0.0;
        let mut lww = labels_with_weights.peekable();
        loop {
            match lww.next() {
                None => break,
                Some((p, (l_binary, w))) => {
                    let l = f64::from(l_binary.get_value());
                    let wl = l * w;
                    true_positives += wl;
                    false_positives += w - wl;
                    if lww.peek().map(|x| x.0 != p).unwrap_or(true) {
                        area_under_curve += area_under_line_segment(last_counted_fp, false_positives, last_counted_tp, true_positives);
                        last_counted_fp = false_positives;
                        last_counted_tp = true_positives;
                    }
                }
            };
        }
        return area_under_curve / (true_positives * false_positives);
    }
}

struct RocAucWithMaxFPR {
    max_fpr: f64,
}

impl RocAucWithMaxFPR {
    fn new(max_fpr: f64) -> Self {
        return RocAucWithMaxFPR { max_fpr };
    }

    fn get_positive_sum<B: BinaryLabel>(labels_with_weights: impl Iterator<Item = (B, f64)>) -> (f64, f64) {
        let mut false_positives = 0f64;
        let mut true_positives = 0f64;
        for (label, weight) in labels_with_weights {
            let lw = weight * f64::from(label.get_value());
            false_positives += weight - lw;
            true_positives += lw;
        }
        return (false_positives, true_positives);
    }
}

impl ScoreSortedDescending for RocAucWithMaxFPR {
    fn score<B: BinaryLabel>(&self, labels_with_weights: impl Iterator<Item = (f64, (B, f64))> + Clone) -> f64 {
        let mut false_positives: f64 = 0.0;
        let mut true_positives: f64 = 0.0;
        let mut last_counted_fp = 0.0;
        let mut last_counted_tp = 0.0;
        let mut area_under_curve = 0.0;
        let (false_positive_sum, true_positive_sum) = Self::get_positive_sum(labels_with_weights.clone().map(|(a, b)| b));
        let false_positive_cutoff = self.max_fpr * false_positive_sum;
        let mut lww = labels_with_weights.peekable();
        loop {
            match lww.next() {
                None => break,
                Some((p, (l_binary, w))) => {
                    let l = f64::from(l_binary.get_value());
                    let wl = l * w;
                    let next_tp = true_positives + wl;
                    let next_fp = false_positives + (w - wl);
                    let is_above_max = next_fp > false_positive_cutoff;
                    if is_above_max {
                        let dx = next_fp  - false_positives;
                        let dy = next_tp - true_positives;
                        true_positives += dy * false_positive_cutoff / dx;
                        false_positives = false_positive_cutoff;
                    } else {
                        true_positives = next_tp;
                        false_positives = next_fp;
                    }
                    if lww.peek().map(|x| x.0 != p).unwrap_or(true) || is_above_max {
                        area_under_curve += area_under_line_segment(last_counted_fp, false_positives, last_counted_tp, true_positives);
                        last_counted_fp = false_positives;
                        last_counted_tp = true_positives;
                    }
                    if is_above_max {
                        break;
                    }                
                }
            };
        }
        let normalized_area_under_curve = area_under_curve / (true_positive_sum * false_positive_sum);
        let min_area = 0.5 * self.max_fpr * self.max_fpr;
        let max_area = self.max_fpr;
        return 0.5 * (1.0 + (normalized_area_under_curve - min_area) / (max_area - min_area));
    }
}

pub fn average_precision<P, B, W>(
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
    order: Option<Order>,
) -> f64
where P: num::Float + Into<f64>, B: BinaryLabel, W: num::Float + Into<f64>
{
    return score_maybe_sorted_sample(AveragePrecision::new(), predictions, labels, weights, order);
}

pub fn roc_auc<P, B, W>(
    predictions: &(impl Data<P> + SortableData<P>),
    labels: &impl Data<B>,
    weights: Option<&impl Data<W>>,
    order: Option<Order>,
    max_fpr: Option<f64>,
) -> f64
where P: num::Float + Into<f64>, B: BinaryLabel, W: num::Float + Into<f64>
{
    return match max_fpr {
        Some(mfpr) => score_maybe_sorted_sample(RocAucWithMaxFPR::new(mfpr), predictions, labels, weights, order),
        None => score_maybe_sorted_sample(RocAuc::new(), predictions, labels, weights, order),
    };
}


pub fn average_precision_on_descending_iterators<B: BinaryLabel, F: num::Float>(
    labels_and_weights: impl Iterator<Item = (B, F)>
) -> f64 {
    let mut ap: f64 = 0.0;
    let mut tps: f64 = 0.0;
    let mut fps: f64 = 0.0;
    for (label, weight) in labels_and_weights {
        let w: f64 = weight.to_f64().unwrap();
        let l: bool = label.get_value();
        let tp = w * f64::from(l);
        tps += tp;
        fps += w - tp;
        let ps = tps + fps;
        let precision = tps / ps;
        ap += tp * precision;
    }
    // Special case for tps == 0 following sklearn
    // https://github.com/scikit-learn/scikit-learn/blob/5cce87176a530d2abea45b5a7e5a4d837c481749/sklearn/metrics/_ranking.py#L1032-L1039
    // I.e. if tps is 0.0, there are no positive samples in labels: Either all labels are 0, or all weights (for positive labels) are 0
    return if tps == 0.0 {
        0.0
    } else {
        ap / tps
    };
}

pub fn average_precision_on_two_sorted_samples<'a, B, L, P, W>(
    sample1: SortedSampleDescending<'a, B, L, P, W>,
    sample2: SortedSampleDescending<'a, B, L, P, W>
) -> f64
where B: BinaryLabel + Clone + PartialOrd + 'a, &'a L: IntoIterator<Item = &'a B>, &'a P: IntoIterator<Item = &'a f64>, &'a W: IntoIterator<Item = &'a f64>
{
    let iter1 = sample1.predictions.into_iter().cloned().zip(
        sample1.labels.into_iter().cloned().zip(sample1.weights.into_iter().cloned())
    );
    let iter2 = sample2.predictions.into_iter().cloned().zip(
        sample2.labels.into_iter().cloned().zip(sample2.weights.into_iter().cloned())
    );
    return average_precision_on_two_sorted_descending_iterators(iter1, iter2,);
}


pub fn average_precision_on_two_sorted_descending_iterators_unzipped<B>(
    label1: impl Iterator<Item = B>,
    score1: impl Iterator<Item = f64>,
    weight1: impl Iterator<Item = f64>,
    label2: impl Iterator<Item = B>,
    score2: impl Iterator<Item = f64>,
    weight2: impl Iterator<Item = f64>,
) -> f64 
where B: BinaryLabel + Clone + PartialOrd
{
    return  average_precision_on_two_sorted_descending_iterators(
        score1.zip(label1.zip(weight1)),
        score2.zip(label2.zip(weight2)),
    );
}


pub fn average_precision_on_two_sorted_descending_iterators<B, F>(
    iter1: impl Iterator<Item = (F, (B, F))>,
    iter2: impl Iterator<Item = (F, (B, F))>,
) -> f64
where B: BinaryLabel + Clone + PartialOrd, F: num::Float
{
    let combined_iter = combine::combine::CombineIterDescending::new(iter1, iter2);
    let label_weight_iter = combined_iter.map(|(a, b)| b);
    return average_precision_on_descending_iterators(label_weight_iter);
}


pub fn roc_auc_max_fpr<B, L, F, P, W>(labels: &L, predictions: &P, weights: Option<&W>, max_false_positive_rate: Option<f64>) -> f64
where B: BinaryLabel, L: Data<B>, F: num::Float, P: SortableData<F> + Data<F>, W: Data<F>
{
    return roc_auc_with_order(labels, predictions, weights, None, max_false_positive_rate);
}

pub fn roc_auc_with_order<B, L, F, P, W>(labels: &L, predictions: &P, weights: Option<&W>, order: Option<Order>, max_false_positive_rate: Option<f64>) -> f64
where B: BinaryLabel, L: Data<B>, F: num::Float, P: SortableData<F> + Data<F>, W: Data<F>
{
    return match order {
        Some(o) => roc_auc_on_sorted_labels(labels, predictions, weights, o, max_false_positive_rate),
        None => {
            let indices = predictions.argsort_unstable();
            let sorted_labels = select(labels, &indices);
            let sorted_predictions = select(predictions, &indices);
            let roc_auc_score = match weights {
                Some(w) => {
                    let sorted_weights = select(w, &indices);
                    roc_auc_on_sorted_labels(&sorted_labels, &sorted_predictions, Some(&sorted_weights), Order::DESCENDING, max_false_positive_rate)
                },
                None => {
                    roc_auc_on_sorted_labels(&sorted_labels, &sorted_predictions, None::<&W>, Order::DESCENDING, max_false_positive_rate)
                }
            };
            roc_auc_score
        }
    };
}
pub fn roc_auc_on_sorted_labels<B, L, F, P, W>(labels: &L, predictions: &P, weights: Option<&W>, order: Order, max_false_positive_rate: Option<f64>) -> f64
where B: BinaryLabel, L: Data<B>, F: num::Float, P: Data<F>, W: Data<F> {
    return match max_false_positive_rate {
        None => {
            let mut lit = labels.get_iterator();
            let mut pit = predictions.get_iterator().map(|p| p.to_f64().unwrap());
            match weights {
                Some(w) => roc_auc_on_sorted_iterator(&mut lit, &mut pit, &mut w.get_iterator().map(|x| x.to_f64().unwrap()), order),
                None => roc_auc_on_sorted_iterator(&mut lit, &mut pit, &mut ConstWeight::<F>::one().get_iterator().map(|x| x.to_f64().unwrap()), order),
            }
        }
        Some(max_fpr) => match weights {
            Some(w) => roc_auc_on_sorted_with_fp_cutoff(labels, predictions, w, order, max_fpr),
            None => roc_auc_on_sorted_with_fp_cutoff(labels, predictions, &ConstWeight::<F>::one(), order, max_fpr),
        }
    };
}

pub fn roc_auc_on_sorted_iterator<B: BinaryLabel>(
    labels: &mut impl DoubleEndedIterator<Item = B>,
    predictions: &mut impl DoubleEndedIterator<Item = f64>,
    weights: &mut impl DoubleEndedIterator<Item = f64>,
    order: Order
) -> f64 {
    return match order {
        Order::ASCENDING => roc_auc_on_descending_iterator(&mut labels.rev(), &mut predictions.rev(), &mut weights.rev()),
        Order::DESCENDING => roc_auc_on_descending_iterator(labels, predictions, weights)
    }
}

pub fn roc_auc_on_descending_iterator<B: BinaryLabel>(
    labels: &mut impl Iterator<Item = B>,
    predictions: &mut impl Iterator<Item = f64>,
    weights: &mut impl Iterator<Item = f64>
) -> f64 {
    let mut false_positives: f64 = 0.0;
    let mut true_positives: f64 = 0.0;
    let mut last_counted_fp = 0.0;
    let mut last_counted_tp = 0.0;
    let mut area_under_curve = 0.0;
    let mut zipped = labels.zip(predictions).zip(weights).peekable();
    loop {
        match zipped.next() {
            None => break,
            Some(actual) => {
                let l = f64::from(actual.0.0.get_value());
                let w = actual.1;
                let wl = l * w;
                true_positives += wl;
                false_positives += w - wl;
                if zipped.peek().map(|x| x.0.1 != actual.0.1).unwrap_or(true) {
                    area_under_curve += area_under_line_segment(last_counted_fp, false_positives, last_counted_tp, true_positives);
                    last_counted_fp = false_positives;
                    last_counted_tp = true_positives;
                }
            }
        };
    }
    return area_under_curve / (true_positives * false_positives);
}

fn area_under_line_segment(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    return dx * y0 + dy * dx * 0.5;
}

fn get_positive_sum<B: BinaryLabel>(
    labels: impl Iterator<Item = B>,
    weights: impl Iterator<Item = f64>
) -> (f64, f64) {
    let mut false_positives = 0f64;
    let mut true_positives = 0f64;
    for (label, weight) in labels.zip(weights) {
        let lw = weight * f64::from(label.get_value());
        false_positives += weight - lw;
        true_positives += lw;
    }
    return (false_positives, true_positives);
}

pub fn roc_auc_on_sorted_with_fp_cutoff<B, L, F, P, W>(labels: &L, predictions: &P, weights: &W, order: Order, max_false_positive_rate: f64) -> f64
where B: BinaryLabel, L: Data<B>, F: num::Float, P: Data<F>, W: Data<F> {
    // TODO validate max_fpr
    let (fps, tps) = get_positive_sum(labels.get_iterator(), weights.get_iterator().map(|x| x.to_f64().unwrap()));
    let mut l_it = labels.get_iterator();
    let mut p_it = predictions.get_iterator().map(|p| p.to_f64().unwrap());
    let mut w_it = weights.get_iterator().map(|w| w.to_f64().unwrap());
    return match order {
        Order::ASCENDING => roc_auc_on_descending_iterator_with_fp_cutoff(&mut l_it.rev(), &mut p_it.rev(), &mut w_it.rev(), fps, tps, max_false_positive_rate),
        Order::DESCENDING => roc_auc_on_descending_iterator_with_fp_cutoff(&mut l_it, &mut p_it, &mut w_it, fps, tps, max_false_positive_rate)
    };
}
    

fn roc_auc_on_descending_iterator_with_fp_cutoff<B: BinaryLabel>(
    labels: &mut impl Iterator<Item = B>,
    predictions: &mut impl Iterator<Item = f64>,
    weights: &mut impl Iterator<Item = f64>,
    false_positive_sum: f64,
    true_positive_sum: f64,
    max_false_positive_rate: f64
) -> f64 {
    let mut false_positives: f64 = 0.0;
    let mut true_positives: f64 = 0.0;
    let mut last_counted_fp = 0.0;
    let mut last_counted_tp = 0.0;
    let mut area_under_curve = 0.0;
    let mut zipped = labels.zip(predictions).zip(weights).peekable();
    let false_positive_cutoff = max_false_positive_rate * false_positive_sum;
    loop {
        match zipped.next() {
            None => break,
            Some(actual) => {
                let l = f64::from(actual.0.0.get_value());
                let w = actual.1;
                let wl = l * w;
                let next_tp = true_positives + wl;
                let next_fp = false_positives + (w - wl);
                let is_above_max = next_fp > false_positive_cutoff;
                if is_above_max {
                    let dx = next_fp  - false_positives;
                    let dy = next_tp - true_positives;
                    true_positives += dy * false_positive_cutoff / dx;
                    false_positives = false_positive_cutoff;
                } else {
                    true_positives = next_tp;
                    false_positives = next_fp;
                }
                if zipped.peek().map(|x| x.0.1 != actual.0.1).unwrap_or(true) || is_above_max {
                    area_under_curve += area_under_line_segment(last_counted_fp, false_positives, last_counted_tp, true_positives);
                    last_counted_fp = false_positives;
                    last_counted_tp = true_positives;
                }
                if is_above_max {
                    break;
                }                
            }
        };
    }
    let normalized_area_under_curve = area_under_curve / (true_positive_sum * false_positive_sum);
    let min_area = 0.5 * max_false_positive_rate * max_false_positive_rate;
    let max_area = max_false_positive_rate;
    return 0.5 * (1.0 + (normalized_area_under_curve - min_area) / (max_area - min_area));
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

trait PyScoreGeneric<B, F>: Ungil + Sync
where B: BinaryLabel + Element, F: num::Float + Element + TotalOrder
{

    fn score_py<'py>(
        &self,
        py: Python<'py>,
        labels: PyReadonlyArray1<'py, B>,
        predictions: PyReadonlyArray1<'py, F>,
        weights: Option<PyReadonlyArray1<'py, F>>,
        order: Option<PyOrder>,
    ) -> f64
    {
        let labels = labels.as_array();
        let predictions = predictions.as_array();
        let order = order.map(py_order_as_order);
        let score = match weights {
            Some(weight) => {
                let w = weight.as_array();
                py.allow_threads(move || {
                    self.score(labels, predictions, Some(w), order)
                })
            },
            None => py.allow_threads(move || {
                self.score(labels, predictions, None::<Vec<F>>, order)
            })
        };
        return score;
    }

    fn score<L: Data<B>, P: SortableData<F> + Data<F>, W: Data<F>>(
        &self,
        labels: L,
        predictions: P,
        weights: Option<W>,
        order: Option<Order>
    ) -> f64;

}

struct AveragePrecisionPyGeneric {

}

impl AveragePrecisionPyGeneric {
    fn new() -> Self {
        return AveragePrecisionPyGeneric {};
    }
}

impl <B, F> PyScoreGeneric<B, F> for AveragePrecisionPyGeneric
where B: BinaryLabel + Element, F: num::Float + Element + TotalOrder + Into<f64>
{
    fn score<L: Data<B>, P: SortableData<F> + Data<F>, W: Data<F>>(
        &self,
        labels: L,
        predictions: P,
        weights: Option<W>,
        order: Option<Order>
    ) -> f64 {
        return average_precision(&predictions, &labels, weights.as_ref(), order);
    }
}

struct RocAucPyGeneric {
    max_fpr: Option<f64>,
}

impl RocAucPyGeneric {
    fn new(max_fpr: Option<f64>) -> Self {
        return RocAucPyGeneric { max_fpr: max_fpr };
    }
}

impl <B, F> PyScoreGeneric<B, F> for RocAucPyGeneric
where B: BinaryLabel + Element, F: num::Float + Element + TotalOrder + Into<f64>
{
    fn score<L: Data<B>, P: SortableData<F> + Data<F>, W: Data<F>>(
        &self,
        labels: L,
        predictions: P,
        weights: Option<W>,
        order: Option<Order>
    ) -> f64 {
        return roc_auc(&predictions, &labels, weights.as_ref(), order, self.max_fpr);
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
            return AveragePrecisionPyGeneric::new().score_py(py, labels, predictions, weights, order);
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


pub fn average_precision_on_two_sorted_samples_py_generic<'py, B, F>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, B>,
    predictions1: PyReadonlyArray1<'py, F>,
    weights1: Option<PyReadonlyArray1<'py, F>>,
    labels2: PyReadonlyArray1<'py, B>,
    predictions2: PyReadonlyArray1<'py, F>,
    weights2: Option<PyReadonlyArray1<'py, F>>,
) -> f64
where B: BinaryLabel + Clone + PartialOrd + Element, F: num::Float + Element
{
    let l1 = labels1.as_array().into_iter().cloned();
    let l2 = labels2.as_array().into_iter().cloned();
    let p1 = predictions1.as_array().into_iter().map(|f| f.to_f64().unwrap());
    let p2 = predictions2.as_array().into_iter().map(|f| f.to_f64().unwrap());


    return match (weights1, weights2) {
        (None, None) => {
            py.allow_threads(move || {
                average_precision_on_two_sorted_descending_iterators_unzipped(l1, p1, repeat(1.0f64), l2, p2, repeat(1.0f64))
            })
        }
        (Some(w1), None) => {
            let w1i = w1.as_array().into_iter().map(|f| f.to_f64().unwrap());
            py.allow_threads(move || {
                average_precision_on_two_sorted_descending_iterators_unzipped(l1, p1, w1i, l2, p2, repeat(1.0f64))
            })
        }
        (None, Some(w2)) => {
            let w2i = w2.as_array().into_iter().map(|f| f.to_f64().unwrap());
            py.allow_threads(move || {
                average_precision_on_two_sorted_descending_iterators_unzipped(l1, p1, repeat(1.0f64), l2, p2, w2i)
            })
        }
        (Some(w1), Some(w2)) =>  {
            let w1i = w1.as_array().into_iter().map(|f| f.to_f64().unwrap());
            let w2i = w2.as_array().into_iter().map(|f| f.to_f64().unwrap());
            py.allow_threads(move || {
                average_precision_on_two_sorted_descending_iterators_unzipped(l1, p1, w1i, l2, p2, w2i)
            })
        }
    };
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_bool_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_bool_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, bool>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, bool>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_i8_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_i8_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, i8>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, i8>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_i16_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_i16_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, i16>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, i16>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_i32_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_i32_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, i32>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, i32>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_i64_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_i64_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, i64>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, i64>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_u8_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_u8_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, u8>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, u8>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_u16_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_u16_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, u16>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, u16>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_u32_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_u32_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, u32>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, u32>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_u64_f32")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_u64_f32<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, u64>,
    predictions1: PyReadonlyArray1<'py, f32>,
    weights1: Option<PyReadonlyArray1<'py, f32>>,
    labels2: PyReadonlyArray1<'py, u64>,
    predictions2: PyReadonlyArray1<'py, f32>,
    weights2: Option<PyReadonlyArray1<'py, f32>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_bool_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_bool_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, bool>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, bool>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_i8_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_i8_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, i8>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, i8>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_i16_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_i16_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, i16>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, i16>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_i32_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_i32_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, i32>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, i32>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_i64_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_i64_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, i64>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, i64>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_u8_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_u8_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, u8>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, u8>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_u16_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_u16_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, u16>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, u16>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_u32_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_u32_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, u32>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, u32>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples_u64_f64")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_u64_f64<'py>(
    py: Python<'py>,
    labels1: PyReadonlyArray1<'py, u64>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: PyReadonlyArray1<'py, u64>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> f64 {
    return average_precision_on_two_sorted_samples_py_generic(py, labels1, predictions1, weights1, labels2, predictions2, weights2);
}


#[pyfunction(name = "average_precision_on_two_sorted_samples")]
#[pyo3(signature = (labels1, predictions1, weights1, labels2, predictions2, weights2))]
pub fn average_precision_on_two_sorted_samples_py<'py>(
    py: Python<'py>,
    labels1: &Bound<'py, PyUntypedArray>,
    predictions1: PyReadonlyArray1<'py, f64>,
    weights1: Option<PyReadonlyArray1<'py, f64>>,
    labels2: &Bound<'py, PyUntypedArray>,
    predictions2: PyReadonlyArray1<'py, f64>,
    weights2: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<f64> {
    let dtype1 = labels1.dtype();
    let dtype2 = labels2.dtype();
    if dtype1.is_equiv_to(&dtype::<u8>(py)) && dtype2.is_equiv_to(&dtype::<u8>(py)) {
        let labels1_dc = labels1.downcast::<PyArray1<u8>>().unwrap().readonly();
        let labels2_dc = labels2.downcast::<PyArray1<u8>>().unwrap().readonly();
        let labels1_arr = labels1_dc.as_array();
        let labels2_arr = labels2_dc.as_array();
        let predictions1_arr = predictions1.as_array();
        let predictions2_arr = predictions2.as_array();
        return match (weights1, weights2) {
            (Some(w1), Some(w2)) => {
                let w1_arr = w1.as_array();
                let w2_arr = w2.as_array();
                let score = py.allow_threads(move || {
                    average_precision_on_two_sorted_samples(
                        SortedSampleDescending::new(&labels1_arr, &predictions1_arr, &w1_arr),
                        SortedSampleDescending::new(&labels2_arr, &predictions2_arr, &w2_arr),
                    )
                });
                Ok(score)
            }
            (Some(w1), None) => Err(PyTypeError::new_err("Only supporting weights and u8 labels as of now.")),
            (None, Some(w2)) => Err(PyTypeError::new_err("Only supporting weights and u8 labels as of now.")),
            (None, None) => Err(PyTypeError::new_err("Only supporting weights and u8 labels as of now.")),
        };
    }
    return Err(PyTypeError::new_err("Only supporting weights and u8 labels as of now."));
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

    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_py, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_bool_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_i8_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_i16_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_i32_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_i64_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_u8_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_u16_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_u32_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_u64_f32, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_bool_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_i8_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_i16_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_i32_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_i64_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_u8_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_u16_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_u32_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(average_precision_on_two_sorted_samples_u64_f64, m)?).unwrap();

    m.add_function(wrap_pyfunction!(loo_cossim_py, m)?).unwrap();
    m.add_function(wrap_pyfunction!(loo_cossim_many_py_f64, m)?).unwrap();
    m.add_function(wrap_pyfunction!(loo_cossim_many_py_f32, m)?).unwrap();
    m.add_class::<PyOrder>().unwrap();
    return Ok(());
}


#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq};
    use super::*;

    #[test]
    fn test_average_precision_on_sorted() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision_on_sorted_labels(&labels, Some(&weights), Order::DESCENDING);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_unsorted() {
        let labels: [u8; 4] = [0, 0, 1, 1];
        let predictions: [f64; 4] = [0.1, 0.4, 0.35, 0.8];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision(&predictions, &labels, Some(&weights), None);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_sorted() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision(&predictions, &labels, Some(&weights), Some(Order::DESCENDING));
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_sorted_pair() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision_on_sorted_samples(&labels, &predictions, &weights, &labels, &predictions, &weights);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_roc_auc() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = roc_auc_with_order(&labels, &predictions, Some(&weights), Some(Order::DESCENDING), None);
        assert_eq!(actual, 0.75);
    }

    // #[test]
    // fn test_loo_cossim_single() {
    //     let data = arr2(&[[0.77395605, 0.43887844, 0.85859792],
    //                       [0.69736803, 0.09417735, 0.97562235]]);
    //     let cossim = loo_cossim_single(&data.view());
    //     let expected = 0.95385941;
    //     assert_relative_eq!(cossim, expected);
    // }

    // #[test]
    // fn test_loo_cossim_many() {
    //     let data = arr3(&[[[0.77395605, 0.43887844, 0.85859792],
    //                        [0.69736803, 0.09417735, 0.97562235]],
    //                       [[0.7611397 , 0.78606431, 0.12811363],
    //                        [0.45038594, 0.37079802, 0.92676499]],
    //                       [[0.64386512, 0.82276161, 0.4434142 ],
    //                        [0.22723872, 0.55458479, 0.06381726]],
    //                       [[0.82763117, 0.6316644 , 0.75808774],
    //                        [0.35452597, 0.97069802, 0.89312112]]]);
    //     let cossim = loo_cossim_many(&data.view());
    //     let expected = arr1(&[0.95385941, 0.62417001, 0.92228589, 0.90025417]);
    //     assert_eq!(cossim.shape(), expected.shape());
    //     for (c, e) in cossim.iter().zip(expected.iter()) {
    //         assert_relative_eq!(c, e);
    //     }
    // }
}
