//! Algorithm for solving problems of the following kind.
//!
//! **Given**: *significance ratio*, *sample of a random variable **X***, *sample of a random variable **Y***.
//! **To figure out**: Is it appropriate to **assume** `Var(X) = Var(Y)`?

use statrs::distribution::{ContinuousCDF, FisherSnedecor};

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum SVHError {
    SignificanceInvalid,
    FreedomDegreesInvalid,
}

impl std::fmt::Display for SVHError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SVHError::SignificanceInvalid => {
                write!(f, "Significance must be between 0.0 and 1.0")
            }
            SVHError::FreedomDegreesInvalid => {
                write!(
                    f,
                    "Freedom Degrees led to fail in initialization of Fisher-Snedecor"
                )
            }
        }
    }
}

pub struct SameVarianceHypothesis {
    x_sample: Vec<f64>,
    y_sample: Vec<f64>,
    significance: f64,
}

impl SameVarianceHypothesis {
    pub fn new(x_sample: &[f64], y_sample: &[f64], significance: f64) -> Self {
        Self {
            x_sample: x_sample.to_owned(),
            y_sample: y_sample.to_owned(),
            significance,
        }
    }

    pub fn solve(&self) -> Result<bool, SVHError> {
        let sample_mean =
            |sample: &[f64]| (1f64 / sample.len() as f64) * sample.iter().sum::<f64>();

        let unbiased_sample_variance = |sample: &[f64]| {
            let mean = sample_mean(&sample);
            (1f64 / (sample.len() as f64 - 1f64))
                * sample
                    .iter()
                    .map(|value| (value - mean).abs().powi(2))
                    .sum::<f64>()
        };
        let (x_usv, y_usv) = (
            unbiased_sample_variance(&self.x_sample),
            unbiased_sample_variance(&self.y_sample),
        );

        let (max_usv, min_usv) = (x_usv.max(y_usv), x_usv.min(y_usv));
        let (freedom_degrees_1, freedom_degrees_2) = if max_usv == x_usv {
            (
                self.x_sample.len() as f64 - 1f64,
                self.y_sample.len() as f64 - 1f64,
            )
        } else {
            (
                self.y_sample.len() as f64 - 1f64,
                self.x_sample.len() as f64 - 1f64,
            )
        };

        let fisher_snedecor_observed = max_usv / min_usv;
        let fisher_snedecor_critical_value = calculate_fished_snedecor_critical_value(
            freedom_degrees_1,
            freedom_degrees_2,
            self.significance,
        )?;

        println!("{fisher_snedecor_observed} < {fisher_snedecor_critical_value}");

        Ok(fisher_snedecor_observed < fisher_snedecor_critical_value)
    }
}

fn calculate_fished_snedecor_critical_value(
    freedom_degrees_1: f64,
    freedom_degrees_2: f64,
    significance: f64,
) -> Result<f64, SVHError> {
    if !(significance > 0.0 && significance < 1.0) {
        return Err(SVHError::SignificanceInvalid);
    }

    let fisher_snedecor_dist = FisherSnedecor::new(freedom_degrees_1, freedom_degrees_2)
        .map_err(|_| SVHError::FreedomDegreesInvalid)?;

    // Critical values corresponds to `(1 - significance)`-quantile.
    // `inverse_cdf(p)` finds `x` such that `P(X <= x) = p`.
    // We need `P(X > x_crit) = significance`, which is equivalent `P(X <= x_crit) = 1 - significance`
    let probability = 1.0 - significance;
    let critical_value = fisher_snedecor_dist.inverse_cdf(probability);

    Ok(critical_value)
}
