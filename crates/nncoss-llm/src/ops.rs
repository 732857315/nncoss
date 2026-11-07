use crate::RopeMode;

pub(crate) fn linear(
    weights: &[f32],
    rows: usize,
    columns: usize,
    input: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    debug_assert_eq!(weights.len(), rows * columns);
    debug_assert_eq!(input.len(), columns);
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let weights = &weights[row * columns..(row + 1) * columns];
        let mut sum = bias.map_or(0.0, |values| values[row]);
        for (weight, value) in weights.iter().zip(input) {
            sum += weight * value;
        }
        output.push(sum);
    }
    output
}

pub(crate) fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    debug_assert_eq!(input.len(), weight.len());
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + epsilon).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * weight)
        .collect()
}

pub(crate) fn rms_norm_heads(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
) {
    for head in 0..heads {
        let start = head * head_dim;
        let normalized = rms_norm(&values[start..start + head_dim], weight, epsilon);
        values[start..start + head_dim].copy_from_slice(&normalized);
    }
}

pub(crate) fn apply_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
    mode: RopeMode,
) {
    let half = head_dim / 2;
    for head in 0..heads {
        let base = head * head_dim;
        for index in 0..half {
            let frequency = theta.powf(-((2 * index) as f32) / head_dim as f32);
            let angle = position as f32 * frequency;
            let (sine, cosine) = angle.sin_cos();
            match mode {
                RopeMode::HalfSplit => {
                    let left = base + index;
                    let right = left + half;
                    let first = values[left];
                    let second = values[right];
                    values[left] = first * cosine - second * sine;
                    values[right] = second * cosine + first * sine;
                }
                RopeMode::Interleaved => {
                    let left = base + 2 * index;
                    let right = left + 1;
                    let first = values[left];
                    let second = values[right];
                    values[left] = first * cosine - second * sine;
                    values[right] = second * cosine + first * sine;
                }
            }
        }
    }
}

pub(crate) fn softmax(values: &[f32]) -> Vec<f32> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut output: Vec<f32> = values.iter().map(|value| (value - maximum).exp()).collect();
    let sum = output.iter().sum::<f32>();
    for value in &mut output {
        *value /= sum;
    }
    output
}

pub(crate) fn silu(value: f32) -> f32 {
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        value * exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn linear_uses_row_major_weights_and_bias() {
        let output = linear(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[2.0, 3.0], Some(&[0.5, -0.5]));
        assert_close(&output, &[8.5, 17.5]);
    }

    #[test]
    fn rms_norm_matches_direct_computation() {
        let output = rms_norm(&[1.0, -2.0], &[2.0, 3.0], 0.0);
        let scale = 1.0 / 2.5_f32.sqrt();
        assert_close(&output, &[2.0 * scale, -6.0 * scale]);
    }

    #[test]
    fn rope_preserves_position_zero() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        for mode in [RopeMode::HalfSplit, RopeMode::Interleaved] {
            let mut values = original.clone();
            apply_rope(&mut values, 1, 4, 0, 10_000.0, mode);
            assert_close(&values, &original);
        }
    }

    #[test]
    fn softmax_is_normalized() {
        let output = softmax(&[1.0, 2.0, 3.0]);
        assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(output[2] > output[1] && output[1] > output[0]);
    }
}
