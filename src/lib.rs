use ndarray::{ArrayView,Ix1};
use numpy::{NotContiguousError,PyReadonlyArray1};
use pyo3::prelude::*; // {PyModule,PyResult,Python,pymodule};
use std::iter::DoubleEndedIterator;
use std::time::Instant;

pub enum Order {
    ASCENDING,
    DESCENDING
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

pub fn average_precision<L, P, W>(labels: &L, predictions: &P, weights: &W) -> f64
where L: Data<u8>, P: SortableData<f64>, W: Data<f64>
{
    return average_precision_with_order(labels, predictions, weights, None);
}

pub fn average_precision_with_order<L, P, W>(labels: &L, predictions: &P, weights: &W, order: Option<Order>) -> f64
where L: Data<u8>, P: SortableData<f64>, W: Data<f64>
{
    return match order {
        Some(o) => average_precision_on_sorted_labels(labels, weights, o),
        None => {
            // let timer1 = Instant::now();
            let indices = predictions.argsort_unstable();
            // let dt1 = timer1.elapsed();
            // println!("Sorting took {}ms", dt1.as_millis());
            let sorted_labels = select(labels, &indices);
            let sorted_weights = select(weights, &indices);
            let ap = average_precision_on_sorted_labels(&sorted_labels, &sorted_weights, Order::DESCENDING);
            ap
        }
    };
}

pub fn average_precision_on_sorted_labels<L, W>(labels: &L, weights: &W, order: Order) -> f64
where L: Data<u8>, W: Data<f64>
{
    return average_precision_on_iterator(labels.get_iterator(), weights.get_iterator(), order);
}

pub fn average_precision_on_iterator<L, W>(labels: L, weights: W, order: Order) -> f64
where L: DoubleEndedIterator<Item = u8>, W: DoubleEndedIterator<Item = f64>
{
    return match order {
        Order::ASCENDING => average_precision_on_descending_iterator(labels.rev(), weights.rev()),
        Order::DESCENDING => average_precision_on_descending_iterator(labels, weights)
    };
}

pub fn average_precision_on_descending_iterator(labels: impl Iterator<Item = u8>, weights: impl Iterator<Item = f64>) -> f64
{
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
#[pyo3(signature = (labels, predictions, *, weights, order=None))]
pub fn average_precision_py<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray1<'py, u8>,
    predictions: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    order: Option<PyOrder>
) -> Result<f64, NotContiguousError> {
    // let timer0 = Instant::now();
    let o = order.map(py_order_as_order);
    let ap = if let (Ok(l), Ok(p), Ok(w)) = (labels.as_slice(), predictions.as_slice(), weights.as_slice()) {
        // let timer = Instant::now();
        let ap = average_precision_with_order(&l, &p, &w, o);
        // let dt = timer.elapsed();
        // println!("AP took {}ms", dt.as_millis());
        ap
    } else {
        average_precision_with_order(&labels.as_array(), &predictions.as_array(), &weights.as_array(), o)
    };
    // let dt_total = timer0.elapsed();
    // println!("AP with overhead took {}ms", dt_total.as_millis());
    return Ok(ap);
}

#[pymodule(name = "scors")]
fn scors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(average_precision_py, m)?).unwrap();
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
}
