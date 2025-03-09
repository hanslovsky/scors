use ndarray::{ArrayView,Ix1};
use numpy::{Element,NotContiguousError,PyArray1,PyArrayDescr,PyArrayDescrMethods,PyArrayMethods,PyReadonlyArray1,PyUntypedArray,PyUntypedArrayMethods,dtype};
use pyo3::Bound;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::iter::DoubleEndedIterator;

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
    let mut ap: f64 = 0.0;
    let mut tps: f64 = 0.0;
    let mut fps: f64 = 0.0;
    for (label, weight) in labels.zip(weights) {
        let w: f64 = weight;
        let l: bool = label.get_value();
        let tp = w * f64::from(l);
        tps += tp;
        fps += weight - tp;
        let ps = tps + fps;
        let precision = tps / ps;
        ap += tp * precision;
    }
    return ap / tps;
}



// ROC AUC score
pub fn roc_auc<L, P, W>(labels: &L, predictions: &P, weights: Option<&W>) -> f64
where L: Data<u8>, P: SortableData<f64> + Data<f64>, W: Data<f64>
{
    return roc_auc_with_order(labels, predictions, weights, None, None);
}

pub fn roc_auc_max_fpr<L, P, W>(labels: &L, predictions: &P, weights: Option<&W>, max_false_positive_rate: Option<f64>) -> f64
where L: Data<u8>, P: SortableData<f64> + Data<f64>, W: Data<f64>
{
    return roc_auc_with_order(labels, predictions, weights, None, max_false_positive_rate);
}

pub fn roc_auc_with_order<L, P, W>(labels: &L, predictions: &P, weights: Option<&W>, order: Option<Order>, max_false_positive_rate: Option<f64>) -> f64
where L: Data<u8>, P: SortableData<f64> + Data<f64>, W: Data<f64>
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
pub fn roc_auc_on_sorted_labels<L, P, W>(labels: &L, predictions: &P, weights: Option<&W>, order: Order, max_false_positive_rate: Option<f64>) -> f64
where L: Data<u8>, P: Data<f64>, W: Data<f64> {
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

fn average_precision_py_generic<'py, B>(
    py: Python<'py>,
    labels: &PyReadonlyArray1<'py, B>,
    predictions: &PyReadonlyArray1<'py, f64>,
    weights: &Option<PyReadonlyArray1<'py, f64>>,
    order: &Option<PyOrder>
) -> f64
where B: BinaryLabel + Element
{
    let labels = labels.as_array();
    let predictions = predictions.as_array();
    let order = order.map(py_order_as_order);
    let ap = match weights {
        Some(w) => {
            let weights = w.as_array();
            py.allow_threads(move || {
                average_precision_with_order(&labels, &predictions, Some(&weights), order)
            })
        },
        None => {
            py.allow_threads(move || {
                average_precision_with_order(&labels, &predictions, None::<&ArrayView<'_, f64, Ix1>>, order)
            })
        }
    };
    return ap;
}

fn average_precision_py_match_run<'py, T>(
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
        Some(average_precision_py_generic(py, &labels.readonly(), predictions, weights, order))
    } else {
        None
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
    if labels.ndim() != 1 {
        return Err(PyTypeError::new_err(format!("Expected 1-dimensional array for labels but found {} dimenisons.", labels.ndim())));
    }
    let label_dtype = labels.dtype();
    if let Some(ap) = average_precision_py_match_run::<bool>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    else if let Some(ap) = average_precision_py_match_run::<u8>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    else if let Some(ap) = average_precision_py_match_run::<i8>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    else if let Some(ap) = average_precision_py_match_run::<u16>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    else if let Some(ap) = average_precision_py_match_run::<i16>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    else if let Some(ap) = average_precision_py_match_run::<u32>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    else if let Some(ap) = average_precision_py_match_run::<i32>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    else if let Some(ap) = average_precision_py_match_run::<u64>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    else if let Some(ap) = average_precision_py_match_run::<i64>(py, &labels, &predictions, &weights, &order, &label_dtype) {
        return Ok(ap)
    }
    return Err(PyTypeError::new_err(format!("Unsupported dtype for labels: {}. Supported dtypes are bool, uint8, uint16, uint32, uint64, in8, int16, int32, int64", label_dtype)));
}

#[pyfunction(name = "roc_auc")]
#[pyo3(signature = (labels, predictions, *, weights=None, order=None, max_false_positive_rate=None))]
pub fn roc_auc_py<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray1<'py, u8>,
    predictions: PyReadonlyArray1<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    order: Option<PyOrder>,
    max_false_positive_rate: Option<f64>,
) -> Result<f64, NotContiguousError> {
    let labels = labels.as_array();
    let predictions = predictions.as_array();
    let order = order.map(py_order_as_order);
    let ap = match weights {
        Some(weight) => {
            let weights = weight.as_array();
            py.allow_threads(move || {
                roc_auc_with_order(&labels, &predictions, Some(&weights), order, max_false_positive_rate)
            })
        },
        None => py.allow_threads(move || {
            roc_auc_with_order(&labels, &predictions, None::<&Vec<f64>>, order, max_false_positive_rate)
        })
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
    fn test_roc_auc() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = roc_auc_with_order(&labels, &predictions, Some(&weights), Some(Order::DESCENDING), None);
        assert_eq!(actual, 0.75);
    }
}
