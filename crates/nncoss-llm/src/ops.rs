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

    fn rotate(first: f32, second: f32, angle: f32) -> (f32, f32) {
        let (sine, cosine) = angle.sin_cos();
        (
            first * cosine - second * sine,
            second * cosine + first * sine,
        )
    }

    #[test]
    fn rope_matches_hand_computed_rotations_at_nonzero_position() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let (slow, fast) = (3.0_f32, 0.3_f32);

        let mut half = input.to_vec();
        apply_rope(&mut half, 1, 4, 3, 100.0, RopeMode::HalfSplit);
        let (a0, a2) = rotate(1.0, 3.0, slow);
        let (a1, a3) = rotate(2.0, 4.0, fast);
        assert_close(&half, &[a0, a1, a2, a3]);

        let mut interleaved = input.to_vec();
        apply_rope(&mut interleaved, 1, 4, 3, 100.0, RopeMode::Interleaved);
        let (b0, b1) = rotate(1.0, 2.0, slow);
        let (b2, b3) = rotate(3.0, 4.0, fast);
        assert_close(&interleaved, &[b0, b1, b2, b3]);

        assert!(
            half.iter()
                .zip(&interleaved)
                .any(|(l, r)| (l - r).abs() > 1e-3)
        );
    }

    #[test]
    fn rope_depends_on_theta_and_rotates_every_head() {
        let input = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let mut low = input.to_vec();
        let mut high = input.to_vec();
        apply_rope(&mut low, 2, 4, 3, 100.0, RopeMode::HalfSplit);
        apply_rope(&mut high, 2, 4, 3, 10_000.0, RopeMode::HalfSplit);
        assert!(low.iter().zip(&high).any(|(l, r)| (l - r).abs() > 1e-3));
        assert_close(&low[..4], &low[4..]);
        assert_close(&high[..4], &high[4..]);
    }

    #[test]
    fn rope_preserves_vector_norm() {
        let mut values = vec![0.5, -1.5, 2.0, 0.25, 1.0, -0.75, 0.0, 3.0];
        let before: f32 = values.iter().map(|v| v * v).sum();
        for mode in [RopeMode::HalfSplit, RopeMode::Interleaved] {
            apply_rope(&mut values, 1, 8, 17, 10_000.0, mode);
            let after: f32 = values.iter().map(|v| v * v).sum();
            assert!((before - after).abs() < 1e-4);
        }
    }

    #[test]
    fn silu_matches_definition_on_both_branches() {
        for value in [-20.0_f32, -1.5, 0.0, 0.7, 12.0] {
            let expected = value / (1.0 + (-value).exp());
            assert!((silu(value) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_is_normalized() {
        let output = softmax(&[1.0, 2.0, 3.0]);
        assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(output[2] > output[1] && output[1] > output[0]);
    }
}
