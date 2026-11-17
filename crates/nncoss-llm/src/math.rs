use std::f64::consts::{FRAC_1_SQRT_2, LN_2};

const LN2_HIGH: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LOW: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
const HALF_PI_HIGH: f64 = f64::from_bits(0x3ff9_21fb_5440_0000);
const HALF_PI_LOW: f64 = f64::from_bits(0x3dd0_b461_1a62_6331);
const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;
const MAX_ANGLE: f64 = 1.0e6;

pub(crate) fn exp(value: f64) -> f64 {
    if !value.is_finite() {
        return f64::NAN;
    }
    if value > 709.0 {
        return f64::INFINITY;
    }
    if value < -708.0 {
        return 0.0;
    }
    let steps = (value / LN_2).round_ties_even();
    let reduced = (value - steps * LN2_HIGH) - steps * LN2_LOW;
    let mut series = 1.0;
    for order in (1..=14).rev() {
        series = 1.0 + series * reduced / f64::from(order);
    }
    let power = f64::from_bits(((steps as i64 + 1023) as u64) << 52);
    series * power
}

pub(crate) fn ln(value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 || !value.is_normal() {
        return f64::NAN;
    }
    let bits = value.to_bits();
    let mut exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut mantissa = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
    if mantissa > 2.0 * FRAC_1_SQRT_2 {
        mantissa /= 2.0;
        exponent += 1;
    }
    let ratio = (mantissa - 1.0) / (mantissa + 1.0);
    let square = ratio * ratio;
    let mut series = 0.0;
    for order in (0..15).rev() {
        series = 1.0 / f64::from(2 * order + 1) + square * series;
    }
    let exponent = exponent as f64;
    exponent * LN2_HIGH + (2.0 * ratio * series + exponent * LN2_LOW)
}

pub(crate) fn pow(base: f64, power: f64) -> f64 {
    exp(power * ln(base))
}

pub(crate) fn sin_cos(value: f64) -> (f64, f64) {
    if !value.is_finite() || value.abs() > MAX_ANGLE {
        return (f64::NAN, f64::NAN);
    }
    let quadrant = (value * TWO_OVER_PI).round_ties_even();
    let reduced = (value - quadrant * HALF_PI_HIGH) - quadrant * HALF_PI_LOW;
    let square = reduced * reduced;
    let mut sine = 0.0;
    for order in (1..=9).rev() {
        sine = 1.0 - square * sine / f64::from((2 * order) * (2 * order + 1));
    }
    let sine = reduced * sine;
    let mut cosine = 0.0;
    for order in (1..=10).rev() {
        cosine = 1.0 - square * cosine / f64::from((2 * order - 1) * (2 * order));
    }
    match (quadrant as i64).rem_euclid(4) {
        0 => (sine, cosine),
        1 => (cosine, -sine),
        2 => (-sine, -cosine),
        _ => (-cosine, sine),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
        let scale = expected.abs().max(f64::MIN_POSITIVE);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "{actual:e} != {expected:e}"
        );
    }

    #[test]
    fn exp_and_ln_match_std() {
        let mut value = -700.0;
        while value < 700.0 {
            assert_relative(exp(value), value.exp(), 4e-15);
            value += 1.37;
        }
        for value in [1e-300, 1e-10, 0.3, 0.7, 1.0, 1.5, 2.0, 1e4, 1e6, 1e300] {
            let expected = f64::ln(value);
            assert!((ln(value) - expected).abs() <= 4e-16 * expected.abs().max(1.0));
        }
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(ln(1.0), 0.0);
        assert!(exp(f64::NAN).is_nan() && ln(-1.0).is_nan() && ln(0.0).is_nan());
        assert_eq!(exp(1000.0), f64::INFINITY);
        assert_eq!(exp(-1000.0), 0.0);
    }

    #[test]
    fn pow_matches_std_for_rope_frequencies() {
        for base in [100.0_f64, 10_000.0, 500_000.0, 1_000_000.0] {
            for index in 0..64 {
                let power = f64::from(2 * index) / 128.0;
                assert_relative(pow(base, power), base.powf(power), 1e-14);
            }
        }
    }

    #[test]
    fn sin_cos_match_std_across_quadrants_and_large_angles() {
        let mut value = -1.0e5;
        while value < 1.0e5 {
            let (sine, cosine) = sin_cos(value);
            assert!((sine - value.sin()).abs() < 1e-11, "sin({value})");
            assert!((cosine - value.cos()).abs() < 1e-11, "cos({value})");
            value += 7.123_456_789;
        }
        for value in [0.0, 1e-8, 0.5, 0.785, 1.0, 3.0, 4.0, 6.0, -2.5] {
            let (sine, cosine) = sin_cos(value);
            assert_relative(sine, f64::sin(value), 4e-16);
            assert_relative(cosine, f64::cos(value), 4e-16);
        }
        assert_eq!(sin_cos(0.0), (0.0, 1.0));
        assert!(sin_cos(f64::INFINITY).0.is_nan() && sin_cos(2e6).1.is_nan());
    }
}
