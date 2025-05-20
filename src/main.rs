mod normal_distribution_hypothesis;
mod same_variance_hypothesis;

use normal_distribution_hypothesis::*;
use same_variance_hypothesis::*;

fn main() {
    let into_vec_f64 = |seq: &[i32]| seq.iter().map(|i| *i as f64).collect::<Vec<f64>>();

    // Normal Distribution Hypothesis
    let e: Vec<_> = into_vec_f64(&[7, 12, 49, 66, 83, 67, 23, 13]);
    let t: Vec<_> = into_vec_f64(&[5, 9, 46, 60, 89, 81, 19, 11]);
    let ndh = NormalDistributionHypothesis::new(&e, &t, 0.05).unwrap();
    println!("NDH: {:?}", ndh.solve());

    // Same Variance Hypothesis
    let x = [100.0f64, 100.5, 99.5, 90.0, 100.0].to_vec();
    let y = [85.4f64, 80.6, 83.0, 81.0].to_vec();
    let svh = SameVarianceHypothesis::new(&x, &y, 0.05);
    println!("SVH: {:?}", svh.solve());
}
