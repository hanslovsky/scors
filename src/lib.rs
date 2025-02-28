use numpy::{NotContiguousError,PyReadonlyArray1};
use pyo3::prelude::*; // {PyModule,PyResult,Python,pymodule};
use std::iter::DoubleEndedIterator;

pub enum Order {
    ASCENDING,
    DESCENDING
}

fn argsort_descending(slice: &[f64]) -> Vec<usize> {
    let mut indices = (0..slice.len()).collect::<Vec<_>>();
    indices.sort_by(|i, k| slice[*k].total_cmp(&slice[*i]));
    return indices;
}

fn select<T>(slice: &[T], indices: &[usize]) -> Vec<T> where T: Copy {
    let mut selection: Vec<T> = Vec::new();
    selection.reserve_exact(indices.len());
    for index in indices {
        selection.push(slice[*index]);
    }
    return selection;
}

pub fn average_precision(labels: &[u8], predictions: &[f64], weights: &[f64]) -> f64 {
    return average_precision_with_order(&labels, &predictions, &weights, None);
}

pub fn average_precision_with_order(labels: &[u8], predictions: &[f64], weights: &[f64], order: Option<Order>) -> f64
{
    return match order {
        Some(o) => average_precision_on_sorted_labels(labels, weights, o),
        None => {
            let indices = argsort_descending(predictions);
            average_precision_on_sorted_labels(
                &select(&labels, &indices),
                &select(&weights, &indices),
                Order::DESCENDING
            )
        }
    };
}

pub fn average_precision_on_sorted_labels(labels: &[u8], weights: &[f64], order: Order) -> f64
{
    return average_precision_on_iterator(labels.iter().cloned(), weights.iter().cloned(), order);
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
    let labels_sl = match labels.as_slice() {
        Err(e) => return Err(e),
        Ok(sl) => sl
    };
    let predictions_sl = match predictions.as_slice() {
        Err(e) => return Err(e),
        Ok(sl) => sl
    };
    let weights_sl = match weights.as_slice() {
        Err(e) => return Err(e),
        Ok(sl) => sl
    };
    let o = order.map(py_order_as_order);
    return Ok(average_precision_with_order(&labels_sl, &predictions_sl, &weights_sl, o));
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
        let actual = average_precision_on_sorted_labels(&labels, &weights);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_unsorted() {
        let labels: [u8; 4] = [0, 0, 1, 1];
        let predictions: [f64; 4] = [0.1, 0.4, 0.35, 0.8];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision(&labels, &predictions, &weights, false);
        assert_eq!(actual, 0.8333333333333333);
    }

    #[test]
    fn test_average_precision_sorted() {
        let labels: [u8; 4] = [1, 0, 1, 0];
        let predictions: [f64; 4] = [0.8, 0.4, 0.35, 0.1];
        let weights: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        let actual = average_precision(&labels, &predictions, &weights, true);
        assert_eq!(actual, 0.8333333333333333);
    }
}
