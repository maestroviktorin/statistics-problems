mod normal_distribution_hypothesis;
mod same_variance_hypothesis;

use normal_distribution_hypothesis::*;
use same_variance_hypothesis::*;

fn main() {
    let into_vec_f64 = |seq: &[i32]| seq.iter().map(|i| *i as f64).collect::<Vec<f64>>();

    // Normal Distribution Hypothesis with a Complete Problem Situation
    let e: Vec<_> = into_vec_f64(&[7, 12, 49, 66, 83, 67, 23, 13]);
    let t: Vec<_> = into_vec_f64(&[5, 9, 46, 60, 89, 81, 19, 11]);
    let situation = Box::new(CompleteNDHProblemSituation::new(&e, &t, 0.05).unwrap());
    let ndh = NormalDistributionHypothesis::new(situation).unwrap();
    println!("NDH Complete: {:?}", ndh.solve());

    // Normal Distribution Hypothesis with an Incomplete Problem Situation
    let rv_ranges = vec![
        (22.0, 24.0),
        (24.0, 26.0),
        (26.0, 28.0),
        (28.0, 30.0),
        (30.0, 32.0),
        (32.0, 34.0),
    ];
    let e: Vec<_> = into_vec_f64(&[2, 12, 34, 40, 10, 2]);
    let situation = Box::new(IncompleteNDHProblemSituation::new(&rv_ranges, &e, 0.01).unwrap());
    let ndh = NormalDistributionHypothesis::new(situation).unwrap();
    println!("NDH Incomplete: {:?}", ndh.solve());

    // Same Variance Hypothesis
    let x = [100.0f64, 100.5, 99.5, 90.0, 100.0].to_vec();
    let y = [85.4f64, 80.6, 83.0, 81.0].to_vec();
    let svh = SameVarianceHypothesis::new(&x, &y, 0.05);
    println!("SVH: {:?}", svh.solve());
}
