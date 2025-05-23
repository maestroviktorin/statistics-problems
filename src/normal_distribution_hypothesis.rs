//! Algorithm for solving problems of the following kinds.
//!
//! -  **Given**: *significance ratio*, *empirical frequency sample*, *theoretical frequency sample*.
//!    **To figure out**: Is it appropriate to **assume** that the empirical sample is a sample of a **Normal Distribution**?
//!
//! -  **Given**: *significance ratio*, *empirical frequency sample*, *random value
//!             ranges* corresponding to the *empirical frequency sample*.  
//!    **To figure out**: Is it appropriate to **assume** that the sample is a sample of a **Normal Distribution**?
//!     

use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

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

pub trait NDHProblemSituation {
    fn empirical_sample(&self) -> Vec<f64>;
    fn theoretical_sample(&self) -> Vec<f64>;
    fn significance(&self) -> f64;
}

pub struct CompleteNDHProblemSituation {
    empirical_sample: Vec<f64>,
    theoretical_sample: Vec<f64>,
    significance: f64,
}

impl CompleteNDHProblemSituation {
    pub fn new(
        empirical_sample: &[f64],
        theoretical_sample: &[f64],
        significance: f64,
    ) -> Result<Self, NDHError> {
        if empirical_sample.len() != theoretical_sample.len() {
            return Err(NDHError::NonEqualSamplesLengths);
        }

        if !(significance > 0. && significance < 1.) {
            return Err(NDHError::SignificanceInvalid);
        }

        Ok(Self {
            empirical_sample: empirical_sample.to_owned(),
            theoretical_sample: theoretical_sample.to_owned(),
            significance,
        })
    }
}

impl NDHProblemSituation for CompleteNDHProblemSituation {
    fn empirical_sample(&self) -> Vec<f64> {
        self.empirical_sample.to_owned()
    }

    fn theoretical_sample(&self) -> Vec<f64> {
        self.theoretical_sample.to_owned()
    }

    fn significance(&self) -> f64 {
        self.significance
    }
}

pub struct IncompleteNDHProblemSituation {
    random_value_ranges: Vec<(f64, f64)>,
    empirical_sample: Vec<f64>,
    significance: f64,
}

// Possible improvement: Implement merging intervals with small frequencies.
impl IncompleteNDHProblemSituation {
    pub fn new(
        random_value_ranges: &[(f64, f64)],
        empirical_sample: &[f64],
        significance: f64,
    ) -> Result<Self, NDHError> {
        if random_value_ranges.len() != empirical_sample.len() {
            return Err(NDHError::NonEqualSamplesLengths);
        }

        if !(significance > 0. && significance < 1.) {
            return Err(NDHError::SignificanceInvalid);
        }

        Ok(Self {
            random_value_ranges: random_value_ranges.to_owned(),
            empirical_sample: empirical_sample.to_owned(),
            significance,
        })
    }
}

impl NDHProblemSituation for IncompleteNDHProblemSituation {
    fn empirical_sample(&self) -> Vec<f64> {
        self.empirical_sample.to_owned()
    }

    fn theoretical_sample(&self) -> Vec<f64> {
        let mean = (1. / self.empirical_sample.iter().sum::<f64>())
            * self
                .random_value_ranges
                .iter()
                .zip(self.empirical_sample.iter())
                .map(|((x_1, x_2), m)| m * (x_2 + x_1) / 2.)
                .sum::<f64>();

        let variance = (1. / (self.empirical_sample.iter().sum::<f64>()))
            * self
                .random_value_ranges
                .iter()
                .zip(self.empirical_sample.iter())
                .map(|((x_1, x_2), m)| m * (((x_2 + x_1) / 2.) - mean).powi(2))
                .sum::<f64>();
        let std_dev = variance.sqrt();

        let normal_distribution = Normal::new(mean, std_dev).unwrap();
        let result = self
            .random_value_ranges
            .iter()
            .map(|(x1, x2)| {
                self.empirical_sample.iter().sum::<f64>()
                    * (normal_distribution.cdf(*x2) - normal_distribution.cdf(*x1))
            })
            .collect();
        println!("{result:?}");
        result
    }

    fn significance(&self) -> f64 {
        self.significance
    }
}

pub struct NormalDistributionHypothesis {
    situation: Box<dyn NDHProblemSituation>,
}

impl NormalDistributionHypothesis {
    pub fn new(situation: Box<dyn NDHProblemSituation>) -> Result<Self, NDHError> {
        Ok(Self { situation })
    }

    pub fn solve(&self) -> Result<bool, NDHError> {
        let freedom_degrees = self.situation.empirical_sample().len() as f64 - 2.0 - 1.0;
        let chi_squared_critical_value =
            calculate_chi_squared_critical_value(freedom_degrees, self.situation.significance())?;

        let chi_squared_observed: f64 = self
            .situation
            .empirical_sample()
            .iter()
            .zip(self.situation.theoretical_sample().iter())
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
