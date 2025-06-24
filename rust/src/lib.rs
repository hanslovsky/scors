use ndarray::{Array1,ArrayView,ArrayView2,ArrayView3,ArrayViewMut1,Ix1};
use num;
use numpy::{Element,PyArray,PyArray1,PyArray2,PyArray3,PyArrayDescr,PyArrayDescrMethods,PyArrayDyn,PyArrayMethods,PyReadonlyArray1,PyUntypedArray,PyUntypedArrayMethods,dtype};
use pyo3::Bound;
use pyo3::exceptions::PyTypeError;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use std::iter::DoubleEndedIterator;
use std::marker::PhantomData;
use std::ops::AddAssign;

#[derive(Clone, Copy)]
pub enum Order {
    ASCENDING,
    DESCENDING
}

struct ConstWeight {
    value: f64
}

impl ConstWeight {
    fn new(value: f64) -> Self {
        return ConstWeight { value: value };
    }
    fn one() -> Self {
        return Self::new(1.0);
    }
}

pub trait Data<T: Clone>: {
    // TODO This is necessary because it seems that there is no trait like that in rust
    //      Maybe I am just not aware, but for now use my own trait.
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T>;
    fn get_at(&self, index: usize) -> T;
}

pub trait SortableData<T> {
    fn argsort_unstable(&self) -> Vec<usize>;
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
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = f64> {
        return ConstWeight::new(self.value);
    }

    fn get_at(&self, _index: usize) -> f64 {
        return self.value.clone();
    }
}

impl <T: Clone> Data<T> for Vec<T> {
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> {
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
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> {
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
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> {
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
    fn get_iterator(&self) -> impl DoubleEndedIterator<Item = T> {
        return self.iter().cloned();
    }
    fn get_at(&self, index: usize) -> T {
        return self[index].clone();
    }
}

impl SortableData<f64> for ArrayView<'_, f64, Ix1> {
    fn argsort_unstable(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|i, k| self[*k].total_cmp(&self[*i]));
        return indices;
    }
}

// struct IndexView<'a, T, D> where T: Clone, D: Data<T>{
//     data: &'a D,
//     indices: &'a Vec<usize>,
// }

// impl <'a, T: Clone> Data<T> for IndexView<'a, T> {
// }

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

// struct SortedSampleIterator<'a, B, L, P, W>
// where B: BinaryLabel + Clone + 'a, L: Iterator<Item = &'a B>, P: Iterator<Item = &'a f64>, W: Iterator<Item = &' f64>
// {
//     labels: &'a mut L,
//     predictions: &'a mut P,
//     weights: &'a mut W,
// }


// impl <'a, B, L, P, W> Iterator for SortedSampleIterator<'a, B, L, P, W>
// where B: BinaryLabel, L: Iterator<Item = B>, P: Iterator<Item = f64>, W: Iterator<Item = f64>
// {
//     type Item = (B, f64, f64);
//     fn next(&mut self) -> Option<(B, f64, f64)> {
//         return match (self.labels.next(), self.predictions.next(), self.weights.next()) {
//             (Some(l), Some(p), Some(w)) => Some((l, p, w)),
//             _ => None
//         };
//     }
// }

// impl <'a, B, L, P, W> IntoIterator for &SortedSampleDescending<'a, B, L, P, W>
// where B: BinaryLabel, L: IntoIterator<Item = B>, P: IntoIterator<Item = f64>, W: IntoIterator<Item = f64> {
//     type Item = (B, f64, f64);
//     type IntoIter = SortedSampleIterator<'a, B, L::IntoIter, P::IntoIter, W::IntoIter>;

//     fn into_iter(self) -> Self::IntoIter {
//         let l = self.labels.into_iter();
//         // return SortedSampleIterator {
//         //     labels: self.labels.into_iter(),
//         //     predictions: self.labels.into_iter(),
//         //     weights: self.weights.into_iter(),
//         // }
//     }
// }

// struct CombineView<'a, B, L, P, W> 
// where B: BinaryLabel, L: Data<B>, P: Data<f64>, W: Data<f64>
// {
//     sample1: &'a SortedSampleDescending<'a, B, L, P, W>,
//     sample2: &'a SortedSampleDescending<'a, B, L, P, W>,
// }

// impl <'a, B, L, P, W> CombineView<'a, B, L, P, W>
// where B: BinaryLabel, L: Data<B>, P: Data<f64>, W: Data<f64>
// {
//     fn new(sample1: &'a SortedSampleDescending<'a, B, L, P, W>, sample2: &'a SortedSampleDescending<'a, B, L, P, W>,) -> Self {
//         return PairedSortedSampleDescending { sample1: sample1, sample2: sample2 };
//     }
// }

// struct CombineViewIterator<'a, B, L, P, W>
// where B: BinaryLabel, L: Data<B>, P: Data<f64>, W: Data<f64>
// {
//     view: &'a CombineView<'a, B, L, P, W>,
//     iterator1: impl Iterator<Item: ((B, f64), f64),
//     iterator2: impl Iterator<Item: ((B, f64), f64),
//     current_index: usize,
//     num_elements: usize
// }

// impl <'a, B, L, P, W> CombineViewIterator<'a, B, L, P, W>
// where B: BinaryLabel, L: Data<B>, P: Data<f64>, W: Data<f64>
// {
//     fn new(view: &'a CombineView<'a, B, L, P, W>) -> Self {
//         return CombineViewIerator {
//             view: view,
//             iterator1: view.sample1.iterator(),
//             iterator2: view.sample2.iterator(),
//             current_index: 0,
//             num_elements: usize
//         };
//     }
// }


// impl <'a, B, L, P, W> Iterator for CombineViewIterator<'a, B, L, P, W>
// where B: BinaryLabel, L: Data<B>, P: Data<f64>, W: Data<f64>
// {
//     type Item = (B, f64, f64);
//     fn next(&mut self) -> Option<(B, f64, f64)> {
//         if self.current_index == self.num_elements {
//             return None;
//         }
        
//         return Some(self.value);
//     }
// }

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

pub fn average_precision<B, L, P, W>(labels: &L, predictions: &P, weights: Option<&W>) -> f64
where B: BinaryLabel, L: Data<B>, P: SortableData<f64>, W: Data<f64>
{
    return average_precision_with_order(labels, predictions, weights, None);
}

pub fn average_precision_with_order<B, L, P, W>(labels: &L, predictions: &P, weights: Option<&W>, order: Option<Order>) -> f64
where B: BinaryLabel, L: Data<B>, P: SortableData<f64>, W: Data<f64>
{
    return match order {
        Some(o) => average_precision_on_sorted_labels(labels, weights, o),
        None => {
            let indices = predictions.argsort_unstable();
            let sorted_labels = select(labels, &indices);
            let ap = match weights {
                None => {
                    // let w: Oepion<&
                    average_precision_on_sorted_labels(&sorted_labels, weights, Order::DESCENDING)
                },
                Some(w) => average_precision_on_sorted_labels(&sorted_labels, Some(&select(w, &indices)), Order::DESCENDING),
            };
            ap
        }
    };
}

pub fn average_precision_on_sorted_labels<B, L, W>(labels: &L, weights: Option<&W>, order: Order) -> f64
where B: BinaryLabel, L: Data<B>, W: Data<f64>
{
    return match weights {
        None => average_precision_on_iterator(labels.get_iterator(), ConstWeight::one(), order),
        Some(w) => average_precision_on_iterator(labels.get_iterator(), w.get_iterator(), order)
    };
}

pub fn average_precision_on_iterator<B, L, W>(labels: L, weights: W, order: Order) -> f64
where B: BinaryLabel, L: DoubleEndedIterator<Item = B>, W: DoubleEndedIterator<Item = f64>
{
    return match order {
        Order::ASCENDING => average_precision_on_descending_iterator(labels.rev(), weights.rev()),
        Order::DESCENDING => average_precision_on_descending_iterator(labels, weights)
    };
}

pub fn average_precision_on_descending_iterator<B: BinaryLabel>(labels: impl Iterator<Item = B>, weights: impl Iterator<Item = f64>) -> f64 {
    return average_precision_on_descending_iterators(labels.zip(weights));
}

// impl <'a, B, L, P, W> SortedSampleDescending<'a, B, L, P, W>
// where B: BinaryLabel, L: IntoIterator<Item = B>, P: IntoIterator<Item = f64>, W: IntoIterator<Item = f64>

pub fn average_precision_on_sorted_samples<'a, B, L, P, W>(l1: &'a L, p1: &'a P, w1: &'a W, l2: &'a L, p2: &'a P, w2: &'a W) -> f64
where B: BinaryLabel + Clone + 'a, &'a L: IntoIterator<Item = &'a B>, &'a P: IntoIterator<Item = &'a f64>, &'a W: IntoIterator<Item = &'a f64>
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

pub fn average_precision_on_descending_iterators<B: BinaryLabel>(labels_and_weights: impl Iterator<Item = (B, f64)>) -> f64 {
    let mut ap: f64 = 0.0;
    let mut tps: f64 = 0.0;
    let mut fps: f64 = 0.0;
    for (label, weight) in labels_and_weights {
        let w: f64 = weight;
        let l: bool = label.get_value();
        let tp = w * f64::from(l);
        tps += tp;
        fps += weight - tp;
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



// ROC AUC score
pub fn roc_auc<B, L, P, W>(labels: &L, predictions: &P, weights: Option<&W>) -> f64
where B: BinaryLabel, L: Data<B>, P: SortableData<f64> + Data<f64>, W: Data<f64>
{
    return roc_auc_with_order(labels, predictions, weights, None, None);
}

pub fn roc_auc_max_fpr<B, L, P, W>(labels: &L, predictions: &P, weights: Option<&W>, max_false_positive_rate: Option<f64>) -> f64
where B: BinaryLabel, L: Data<B>, P: SortableData<f64> + Data<f64>, W: Data<f64>
{
    return roc_auc_with_order(labels, predictions, weights, None, max_false_positive_rate);
}

pub fn roc_auc_with_order<B, L, P, W>(labels: &L, predictions: &P, weights: Option<&W>, order: Option<Order>, max_false_positive_rate: Option<f64>) -> f64
where B: BinaryLabel, L: Data<B>, P: SortableData<f64> + Data<f64>, W: Data<f64>
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
pub fn roc_auc_on_sorted_labels<B, L, P, W>(labels: &L, predictions: &P, weights: Option<&W>, order: Order, max_false_positive_rate: Option<f64>) -> f64
where B: BinaryLabel, L: Data<B>, P: Data<f64>, W: Data<f64> {
    return match max_false_positive_rate {
        None => match weights {
            Some(w) => roc_auc_on_sorted_iterator(&mut labels.get_iterator(), &mut predictions.get_iterator(), &mut w.get_iterator(), order),
            None => roc_auc_on_sorted_iterator(&mut labels.get_iterator(), &mut predictions.get_iterator(), &mut ConstWeight::one().get_iterator(), order),
        }
        Some(max_fpr) => match weights {
            Some(w) => roc_auc_on_sorted_with_fp_cutoff(labels, predictions, w, order, max_fpr),
            None => roc_auc_on_sorted_with_fp_cutoff(labels, predictions, &ConstWeight::one(), order, max_fpr),
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

pub fn roc_auc_on_sorted_with_fp_cutoff<B, L, P, W>(labels: &L, predictions: &P, weights: &W, order: Order, max_false_positive_rate: f64) -> f64
where B: BinaryLabel, L: Data<B>, P: Data<f64>, W: Data<f64> {
    // TODO validate max_fpr
    let (fps, tps) = get_positive_sum(labels.get_iterator(), weights.get_iterator());
    let mut l_it = labels.get_iterator();
    let mut p_it = predictions.get_iterator();
    let mut w_it = weights.get_iterator();
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


trait PyScore: Ungil + Sync {

    fn score<B, L, P, W>(&self, labels: &L, predictions: &P, weights: Option<&W>, order: Option<Order>) -> f64
    where B: BinaryLabel, L: Data<B>, P: SortableData<f64> + Data<f64>, W: Data<f64>;

    fn score_py_generic<'py, B>(
        &self,
        py: Python<'py>,
        labels: &PyReadonlyArray1<'py, B>,
        predictions: &PyReadonlyArray1<'py, f64>,
        weights: &Option<PyReadonlyArray1<'py, f64>>,
        order: &Option<PyOrder>,
    ) -> f64
    where B: BinaryLabel + Element
    {
        let labels = labels.as_array();
        let predictions = predictions.as_array();
        let order = order.map(py_order_as_order);
        let score = match weights {
            Some(weight) => {
                let weights = weight.as_array();
                py.allow_threads(move || {
                    self.score(&labels, &predictions, Some(&weights), order)
                })
            },
            None => py.allow_threads(move || {
                self.score(&labels, &predictions, None::<&Vec<f64>>, order)
            })
        };
        return score;
    }

    fn score_py_match_run<'py, T>(
        &self,
        py: Python<'py>,
        labels: &Bound<'py, PyUntypedArray>,
        predictions: &PyReadonlyArray1<'py, f64>,
        weights: &Option<PyReadonlyArray1<'py, f64>>,
        order: &Option<PyOrder>,
        dt: &Bound<'py, PyArrayDescr>
    ) -> Option<f64>
    where T: Element + BinaryLabel
    {
        return if dt.is_equiv_to(&dtype::<T>(py)) {
            let labels = labels.downcast::<PyArray1<T>>().unwrap().readonly();
            Some(self.score_py_generic(py, &labels.readonly(), predictions, weights, order))
        } else {
            None
        };
    }
    
    fn score_py<'py>(
        &self,
        py: Python<'py>,
        labels: &Bound<'py, PyUntypedArray>,
        predictions: PyReadonlyArray1<'py, f64>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
        order: Option<PyOrder>,
    ) -> PyResult<f64> {
        if labels.ndim() != 1 {
            return Err(PyTypeError::new_err(format!("Expected 1-dimensional array for labels but found {} dimenisons.", labels.ndim())));
        }
        let label_dtype = labels.dtype();
        if let Some(score) = self.score_py_match_run::<bool>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        else if let Some(score) = self.score_py_match_run::<u8>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        else if let Some(score) = self.score_py_match_run::<i8>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        else if let Some(score) = self.score_py_match_run::<u16>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        else if let Some(score) = self.score_py_match_run::<i16>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        else if let Some(score) = self.score_py_match_run::<u32>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        else if let Some(score) = self.score_py_match_run::<i32>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        else if let Some(score) = self.score_py_match_run::<u64>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        else if let Some(score) = self.score_py_match_run::<i64>(py, &labels, &predictions, &weights, &order, &label_dtype) {
            return Ok(score)
        }
        return Err(PyTypeError::new_err(format!("Unsupported dtype for labels: {}. Supported dtypes are bool, uint8, uint16, uint32, uint64, in8, int16, int32, int64", label_dtype)));
    }
}

struct PyAveragePrecision {
    
}

impl PyAveragePrecision{
    fn new() -> Self {
        return PyAveragePrecision {};
    }
}

impl PyScore for PyAveragePrecision {
    fn score<B, L, P, W>(&self, labels: &L, predictions: &P, weights: Option<&W>, order: Option<Order>) -> f64
    where B: BinaryLabel, L: Data<B>, P: SortableData<f64> + Data<f64>, W: Data<f64> {
        return average_precision_with_order(labels, predictions, weights, order);
    }
}

struct PyRocAuc {
    max_fpr: Option<f64>
}

impl PyRocAuc {
    fn new(max_fpr: Option<f64>) -> Self {
        return PyRocAuc { max_fpr: max_fpr };
    }
}

impl PyScore for PyRocAuc {
    fn score<B, L, P, W>(&self, labels: &L, predictions: &P, weights: Option<&W>, order: Option<Order>) -> f64
    where B: BinaryLabel, L: Data<B>, P: SortableData<f64> + Data<f64>, W: Data<f64> {
        return roc_auc_with_order(labels, predictions, weights, order, self.max_fpr);
    }
}


#[pyfunction(name = "average_precision")]
#[pyo3(signature = (labels, predictions, *, weights=None, order=None))]
pub fn average_precision_py<'py>(
    py: Python<'py>,
    labels: &Bound<'py, PyUntypedArray>,
    predictions: PyReadonlyArray1<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    order: Option<PyOrder>
) -> PyResult<f64> {
    return PyAveragePrecision::new().score_py(py, labels, predictions, weights, order);
}

#[pyfunction(name = "roc_auc")]
#[pyo3(signature = (labels, predictions, *, weights=None, order=None, max_fpr=None))]
pub fn roc_auc_py<'py>(
    py: Python<'py>,
    labels: &Bound<'py, PyUntypedArray>,
    predictions: PyReadonlyArray1<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    order: Option<PyOrder>,
    max_fpr: Option<f64>,
) -> PyResult<f64> {
    return PyRocAuc::new(max_fpr).score_py(py, labels, predictions, weights, order);
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
    m.add_function(wrap_pyfunction!(average_precision_py, m)?).unwrap();
    m.add_function(wrap_pyfunction!(roc_auc_py, m)?).unwrap();
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
        // let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision_on_sorted_labels(&labels, Some(&weights), Order::DESCENDING);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_unsorted() {
        let labels: [u8; 4] = [0, 0, 1, 1];
        let predictions: [f64; 4] = [0.1, 0.4, 0.35, 0.8];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision_with_order(&labels, &predictions, Some(&weights), None);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_sorted() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision_with_order(&labels, &predictions, Some(&weights), Some(Order::DESCENDING));
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

    #[test]
    fn test_loo_cossim_single() {
        let data = arr2(&[[0.77395605, 0.43887844, 0.85859792],
                          [0.69736803, 0.09417735, 0.97562235]]);
        let cossim = loo_cossim_single(&data.view());
        let expected = 0.95385941;
        assert_relative_eq!(cossim, expected);
    }

    #[test]
    fn test_loo_cossim_many() {
        let data = arr3(&[[[0.77395605, 0.43887844, 0.85859792],
                           [0.69736803, 0.09417735, 0.97562235]],
                          [[0.7611397 , 0.78606431, 0.12811363],
                           [0.45038594, 0.37079802, 0.92676499]],
                          [[0.64386512, 0.82276161, 0.4434142 ],
                           [0.22723872, 0.55458479, 0.06381726]],
                          [[0.82763117, 0.6316644 , 0.75808774],
                           [0.35452597, 0.97069802, 0.89312112]]]);
        let cossim = loo_cossim_many(&data.view());
        let expected = arr1(&[0.95385941, 0.62417001, 0.92228589, 0.90025417]);
        assert_eq!(cossim.shape(), expected.shape());
        for (c, e) in cossim.iter().zip(expected.iter()) {
            assert_relative_eq!(c, e);
        }
    }
}
