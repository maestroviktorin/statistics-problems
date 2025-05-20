//! Algorithm for solving problems of the following kind.
//!
//! Given: `significance` ratio, empirical sample, theoretical sample.
//! To figure out: Is it appropriate to assume that the sample is a sample of a Normal Distribution?

use statrs::distribution::{ChiSquared, ContinuousCDF};

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum NDHError {
    NonEqualSamplesLengths,
    SignificanceInvalid,
    FreedomDegreesInvalid,
}

impl std::fmt::Display for NDHError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NDHError::NonEqualSamplesLengths => {
                write!(f, "Lengths of samples are different")
            }
            NDHError::SignificanceInvalid => {
                write!(f, "Significance must be between 0.0 and 1.0")
            }
            NDHError::FreedomDegreesInvalid => {
                write!(
                    f,
                    "Freedom Degrees led to fail in initialization of Gamma underlying Chi"
                )
            }
        }
    }
}

pub struct NormalDistributionHypothesis {
    empirical_sample: Vec<f64>,
    theoretical_sample: Vec<f64>,
    significance: f64,
}

impl NormalDistributionHypothesis {
    pub fn new(
        empirical_sample: &[f64],
        theoretical_sample: &[f64],
        significance: f64,
    ) -> Result<Self, NDHError> {
        if empirical_sample.len() != theoretical_sample.len() {
            return Err(NDHError::NonEqualSamplesLengths);
        }

        Ok(Self {
            empirical_sample: empirical_sample.to_owned(),
            theoretical_sample: theoretical_sample.to_owned(),
            significance,
        })
    }

    pub fn solve(&self) -> Result<bool, NDHError> {
        let freedom_degrees = self.empirical_sample.len() as f64 - 2.0 - 1.0;
        let chi_squared_critical_value =
            calculate_chi_squared_critical_value(freedom_degrees, self.significance)?;

        let chi_squared_observed: f64 = self
            .empirical_sample
            .iter()
            .zip(self.theoretical_sample.iter())
            .map(|(e, t)| (e - t).powi(2) / t)
            .sum();

        println!("{chi_squared_observed} < {chi_squared_critical_value}");

        Ok(chi_squared_observed < chi_squared_critical_value)
    }
}

fn calculate_chi_squared_critical_value(
    freedom_degrees: f64,
    significance: f64,
) -> Result<f64, NDHError> {
    if !(significance > 0.0 && significance < 1.0) {
        return Err(NDHError::SignificanceInvalid);
    }

    let chi_squared_dist =
        ChiSquared::new(freedom_degrees).map_err(|_| NDHError::FreedomDegreesInvalid)?;

    // Critical values corresponds to `(1 - significance)`-quantile.
    // `inverse_cdf(p)` finds `x` such that `P(X <= x) = p`.
    // We need `P(X > x_crit) = significance`, which is equivalent `P(X <= x_crit) = 1 - significance`
    let probability = 1.0 - significance;
    let critical_value = chi_squared_dist.inverse_cdf(probability);

    Ok(critical_value)
}
