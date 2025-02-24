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

pub fn average_precision(labels: &[u8], predictions: &[f64], weights: &[f64], is_descending: bool) -> f64
{
    if is_descending {
        return average_precision_on_sorted_labels(labels, weights);
    }
    let indices = argsort_descending(predictions);
    return average_precision(
        &select(&labels, &indices),
        &predictions,
        &select(&weights, &indices),
        true
    );
}

pub fn average_precision_on_sorted_labels(labels: &[u8], weights: &[f64]) -> f64
{
    let mut ap: f64 = 0.0;
    let mut tps: f64 = 0.0;
    let mut fps: f64 = 0.0;
    for (label, weight) in labels.iter().zip(weights.iter()) {
        let tp = weight * (*label as f64);
        tps += tp;
        fps += weight - tp;
        println!("tps={} fps={}", tps, fps);
        let ps = tps + fps;
        let precision = tps / ps;
        ap += tp * precision;
    }
    return ap / tps;
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
