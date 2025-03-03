use ndarray::{ArrayView,Ix1};
use numpy::{NotContiguousError,PyReadonlyArray1};
use pyo3::prelude::*; // {PyModule,PyResult,Python,pymodule};
use std::iter::DoubleEndedIterator;

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

pub fn average_precision<L, P, W>(labels: &L, predictions: &P, weights: Option<&W>) -> f64
where L: Data<u8>, P: SortableData<f64>, W: Data<f64>
{
    return average_precision_with_order(labels, predictions, weights, None);
}

pub fn average_precision_with_order<L, P, W>(labels: &L, predictions: &P, weights: Option<&W>, order: Option<Order>) -> f64
where L: Data<u8>, P: SortableData<f64>, W: Data<f64>
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

pub fn average_precision_on_sorted_labels<L, W>(labels: &L, weights: Option<&W>, order: Order) -> f64
where L: Data<u8>, W: Data<f64>
{
    return match weights {
        None => average_precision_on_iterator(labels.get_iterator(), ConstWeight::one(), order),
        Some(w) => average_precision_on_iterator(labels.get_iterator(), w.get_iterator(), order)
    };
}

pub fn average_precision_on_iterator<L, W>(labels: L, weights: W, order: Order) -> f64
where L: DoubleEndedIterator<Item = u8>, W: DoubleEndedIterator<Item = f64>
{
    return match order {
        Order::ASCENDING => average_precision_on_descending_iterator(labels.rev(), weights.rev()),
        Order::DESCENDING => average_precision_on_descending_iterator(labels, weights)
    };
}

pub fn average_precision_on_descending_iterator(labels: impl Iterator<Item = u8>, weights: impl Iterator<Item = f64>) -> f64 {
    let mut ap: f64 = 0.0;
    let mut tps: f64 = 0.0;
    let mut fps: f64 = 0.0;
    for (label, weight) in labels.zip(weights) {
        let w: f64 = weight;
        let l: u8 = label;
        let tp = w * (l as f64);
        tps += tp;
        fps += weight - tp;
        let ps = tps + fps;
        let precision = tps / ps;
        ap += tp * precision;
    }
    return ap / tps;
}



// ROC AUC score
pub fn roc_auc<L, P, W>(labels: &L, predictions: &P, weights: &W) -> f64
where L: Data<u8>, P: SortableData<f64> + Data<f64>, W: Data<f64>
{
    return roc_auc_with_order(labels, predictions, weights, None, None);
}

pub fn roc_auc_max_fpr<L, P, W>(labels: &L, predictions: &P, weights: &W, max_false_positive_rate: Option<f64>) -> f64
where L: Data<u8>, P: SortableData<f64> + Data<f64>, W: Data<f64>
{
    return roc_auc_with_order(labels, predictions, weights, None, max_false_positive_rate);
}

pub fn roc_auc_with_order<L, P, W>(labels: &L, predictions: &P, weights: &W, order: Option<Order>, max_false_positive_rate: Option<f64>) -> f64
where L: Data<u8>, P: SortableData<f64> + Data<f64>, W: Data<f64>
{
    return match order {
        Some(o) => roc_auc_on_sorted_labels(labels, predictions, weights, o, max_false_positive_rate),
        None => {
            let indices = predictions.argsort_unstable();
            let sorted_labels = select(labels, &indices);
            let sorted_predictions = select(predictions, &indices);
            let sorted_weights = select(weights, &indices);
            let roc_auc_score = roc_auc_on_sorted_labels(&sorted_labels, &sorted_predictions, &sorted_weights, Order::DESCENDING, max_false_positive_rate);
            roc_auc_score
        }
    };
}
pub fn roc_auc_on_sorted_labels<L, P, W>(labels: &L, predictions: &P, weights: &W, order: Order, max_false_positive_rate: Option<f64>) -> f64
where L: Data<u8>, P: Data<f64>, W: Data<f64> {
    return match max_false_positive_rate {
        None => roc_auc_on_sorted_iterator(&mut labels.get_iterator(), &mut predictions.get_iterator(), &mut weights.get_iterator(), order),
        Some(max_fpr) => roc_auc_on_sorted_with_fp_cutoff(labels, predictions, weights, order, max_fpr)
    };
}

pub fn roc_auc_on_sorted_iterator(
    labels: &mut impl DoubleEndedIterator<Item = u8>,
    predictions: &mut impl DoubleEndedIterator<Item = f64>,
    weights: &mut impl DoubleEndedIterator<Item = f64>,
    order: Order
) -> f64 {
    return match order {
        Order::ASCENDING => roc_auc_on_descending_iterator(&mut labels.rev(), &mut predictions.rev(), &mut weights.rev()),
        Order::DESCENDING => roc_auc_on_descending_iterator(labels, predictions, weights)
    }
}

pub fn roc_auc_on_descending_iterator(
    labels: &mut impl Iterator<Item = u8>,
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
                let l = actual.0.0 as f64;
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

fn get_positive_sum(
    labels: impl Iterator<Item = u8>,
    weights: impl Iterator<Item = f64>
) -> (f64, f64) {
    let mut false_positives = 0f64;
    let mut true_positives = 0f64;
    for (label, weight) in labels.zip(weights) {
        let lw = weight * (label as f64);
        false_positives += weight - lw;
        true_positives += lw;
    }
    return (false_positives, true_positives);
}

pub fn roc_auc_on_sorted_with_fp_cutoff<L, P, W>(labels: &L, predictions: &P, weights: &W, order: Order, max_false_positive_rate: f64) -> f64
where L: Data<u8>, P: Data<f64>, W: Data<f64> {
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
    

fn roc_auc_on_descending_iterator_with_fp_cutoff(
    labels: &mut impl Iterator<Item = u8>,
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
                let l = actual.0.0 as f64;
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


// Python bindings
#[pyclass(eq, eq_int, name="Order")]
#[derive(PartialEq)]
pub enum PyOrder {
    ASCENDING,
    DESCENDING
}

impl Clone for PyOrder {
    fn clone(&self) -> Self {
        match self {
            PyOrder::ASCENDING => PyOrder::ASCENDING,
            PyOrder::DESCENDING => PyOrder::DESCENDING
        }
    }
}

fn py_order_as_order(order: PyOrder) -> Order {
    return match order {
        PyOrder::ASCENDING => Order::ASCENDING,
        PyOrder::DESCENDING => Order::DESCENDING,
    }
}

#[pyfunction(name = "average_precision")]
#[pyo3(signature = (labels, predictions, *, weights=None, order=None))]
pub fn average_precision_py<'py>(
    _py: Python<'py>,
    labels: PyReadonlyArray1<'py, u8>,
    predictions: PyReadonlyArray1<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    order: Option<PyOrder>
) -> Result<f64, NotContiguousError> {
    // TODO benchmark if slice has any benefits over just as_array
    let o = order.map(py_order_as_order);
    let ap = match weights {
        None => {
            if let (Ok(l), Ok(p)) = (labels.as_slice(), predictions.as_slice()) {
                average_precision_with_order(&l, &p, None::<&Vec<f64>>, o)  
            } else {
                average_precision_with_order(&labels.as_array(), &predictions.as_array(), None::<&Vec<f64>>, o)
            }
        },
        Some(weight) => {
            if let (Ok(l), Ok(p), Ok(w)) = (labels.as_slice(), predictions.as_slice(), weight.as_slice()) {
                average_precision_with_order(&l, &p, Some(&w), o)  
            } else {
                average_precision_with_order(&labels.as_array(), &predictions.as_array(), Some(&weight.as_array()), o)
            }
        }
    };
    return Ok(ap);
}

#[pyfunction(name = "roc_auc")]
#[pyo3(signature = (labels, predictions, *, weights, order=None, max_false_positive_rate=None))]
pub fn roc_auc_py<'py>(
    _py: Python<'py>,
    labels: PyReadonlyArray1<'py, u8>,
    predictions: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    order: Option<PyOrder>,
    max_false_positive_rate: Option<f64>,
) -> Result<f64, NotContiguousError> {
    let o = order.map(py_order_as_order);
    let ap = if let (Ok(l), Ok(p), Ok(w)) = (labels.as_slice(), predictions.as_slice(), weights.as_slice()) {
        roc_auc_with_order(&l, &p, &w, o, max_false_positive_rate)
    } else {
        roc_auc_with_order(&labels.as_array(), &predictions.as_array(), &weights.as_array(), o, max_false_positive_rate)
    };

    return Ok(ap);
}

#[pymodule(name = "_scors")]
fn scors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(average_precision_py, m)?).unwrap();
    m.add_function(wrap_pyfunction!(roc_auc_py, m)?).unwrap();
    m.add_class::<PyOrder>().unwrap();
    return Ok(());
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_precision_on_sorted() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        // let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision_on_sorted_labels(&labels, &weights, Order::DESCENDING);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_unsorted() {
        let labels: [u8; 4] = [0, 0, 1, 1];
        let predictions: [f64; 4] = [0.1, 0.4, 0.35, 0.8];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision_with_order(&labels, &predictions, &weights, None);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_sorted() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision_with_order(&labels, &predictions, &weights, Some(Order::DESCENDING));
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_roc_auc() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = roc_auc_with_order(&labels, &predictions, &weights, Some(Order::DESCENDING));
        assert_eq!(actual, 0.75);
    }
}
