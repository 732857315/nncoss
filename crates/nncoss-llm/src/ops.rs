use std::ops::Range;

use crate::RopeMode;
use crate::cache::{F16, I8, KvValue, KvView, RowScales};
use crate::pool::{SharedPtr, parallel_for};

macro_rules! simd_fn {
    ($(#[$meta:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        $(#[$meta])*
        $vis fn $name($($arg: $ty),*) {
            #[inline(always)]
            fn body($($arg: $ty),*) $body
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn accelerated($($arg: $ty),*) {
                    body($($arg),*)
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    return unsafe { accelerated($($arg),*) };
                }
            }
            body($($arg),*)
        }
    };
}

pub(crate) unsafe fn linear_rows(
    weights: &[f32],
    columns: usize,
    inputs: &[f32],
    rows: Range<usize>,
    bias: Option<&[f32]>,
    output: *mut f32,
    stride: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { linear_rows_avx2(weights, columns, inputs, rows, bias, output, stride) };
        return;
    }
    unsafe {
        linear_rows_with(
            weights,
            columns,
            inputs,
            rows,
            bias,
            output,
            stride,
            dot,
            |_| {},
        )
    };
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn linear_rows_with(
    weights: &[f32],
    columns: usize,
    inputs: &[f32],
    rows: Range<usize>,
    bias: Option<&[f32]>,
    output: *mut f32,
    stride: usize,
    dot: impl Fn(&[f32], &[f32]) -> f32,
    prefetch: impl Fn(&[f32]),
) {
    for (offset, row) in rows.clone().enumerate() {
        if row + 1 < rows.end {
            prefetch(&weights[(row + 1) * columns..(row + 2) * columns]);
        }
        let weights = &weights[row * columns..(row + 1) * columns];
        for (token, input) in inputs.chunks_exact(columns).enumerate() {
            let product = dot(input, weights);
            let value = match bias {
                Some(bias) => product + bias[row],
                None => product,
            };
            unsafe { output.add(token * stride + offset).write(value) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn linear_rows_avx2(
    weights: &[f32],
    columns: usize,
    inputs: &[f32],
    rows: Range<usize>,
    bias: Option<&[f32]>,
    output: *mut f32,
    stride: usize,
) {
    unsafe {
        linear_rows_with(
            weights,
            columns,
            inputs,
            rows,
            bias,
            output,
            stride,
            |left, right| dot_avx2(left, right),
            |row| {
                use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                for line in (0..row.len()).step_by(16) {
                    _mm_prefetch::<_MM_HINT_T0>(row[line..].as_ptr().cast());
                }
            },
        )
    }
}

#[cfg(test)]
pub(crate) fn linear(
    weights: &[f32],
    rows: usize,
    columns: usize,
    input: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    debug_assert_eq!(weights.len(), rows * columns);
    let mut output = vec![0.0; rows];
    unsafe {
        linear_rows(
            weights,
            columns,
            input,
            0..rows,
            bias,
            output.as_mut_ptr(),
            rows,
        )
    };
    output
}

const LANES: usize = 8;

#[inline(always)]
fn sum_lanes(lanes: [f32; LANES]) -> f32 {
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
}

#[inline(always)]
fn sum(values: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let rest = chunks.remainder();
    for chunk in chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane += value;
        }
    }
    let mut total = sum_lanes(lanes);
    for value in rest {
        total += value;
    }
    total
}

#[inline(always)]
fn sum_squares(values: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let rest = chunks.remainder();
    for chunk in chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane += value * value;
        }
    }
    let mut total = sum_lanes(lanes);
    for value in rest {
        total += value * value;
    }
    total
}

#[inline(always)]
fn dot<K: KvValue>(left: &[f32], right: &[K]) -> f32 {
    const WIDE: usize = LANES * 2;
    let mut lanes = [0.0_f32; WIDE];
    let left_chunks = left.chunks_exact(WIDE);
    let right_chunks = right.chunks_exact(WIDE);
    let rest = left_chunks.remainder().iter().zip(right_chunks.remainder());
    for (left, right) in left_chunks.zip(right_chunks) {
        for ((lane, left), right) in lanes.iter_mut().zip(left).zip(right) {
            *lane += left * right.to_f32();
        }
    }
    let mut folded = [0.0_f32; LANES];
    for (index, lane) in folded.iter_mut().enumerate() {
        *lane = lanes[index] + lanes[index + LANES];
    }
    let mut total = sum_lanes(folded);
    for (left, right) in rest {
        total += left * right.to_f32();
    }
    total
}

#[inline(always)]
fn axpy<K: KvValue>(output: &mut [f32], scale: f32, values: &[K]) {
    for (output, value) in output.iter_mut().zip(values) {
        *output += scale * value.to_f32();
    }
}

#[inline(always)]
pub(crate) fn exp(value: f32) -> f32 {
    use std::f32::consts::LOG2_E;
    const LN2_HIGH: f32 = 0.693_359_4;
    const LN2_LOW: f32 = -2.121_944_4e-4;
    #[allow(clippy::manual_clamp)]
    let clamped = value.max(-87.0).min(88.0);
    let exponent = (clamped * LOG2_E).round_ties_even();
    let reduced = clamped - exponent * LN2_HIGH - exponent * LN2_LOW;
    let mut series = 1.987_569_1e-4_f32;
    series = series * reduced + 1.398_199_9e-3;
    series = series * reduced + 8.333_452e-3;
    series = series * reduced + 4.166_579_6e-2;
    series = series * reduced + 1.666_666_5e-1;
    series = series * reduced + 5.0e-1;
    let result = series * (reduced * reduced) + reduced + 1.0;
    let exponent = unsafe { exponent.to_int_unchecked::<i32>() };
    let power = f32::from_bits(((exponent + 127) as u32) << 23);
    result * power + value * 0.0
}

#[inline(always)]
pub(crate) fn silu(value: f32) -> f32 {
    value / (1.0 + exp(-value))
}

simd_fn! {
    pub(crate) fn rms_norm_into(input: &[f32], weight: &[f32], epsilon: f32, output: &mut [f32]) {
        debug_assert_eq!(input.len(), weight.len());
        debug_assert_eq!(input.len(), output.len());
        let mean_square = sum_squares(input) / input.len() as f32;
        let scale = (mean_square + epsilon).sqrt().recip();
        for ((output, value), weight) in output.iter_mut().zip(input).zip(weight) {
            *output = value * scale * weight;
        }
    }
}

simd_fn! {
    pub(crate) fn rms_norm_heads(
        values: &mut [f32],
        heads: usize,
        head_dim: usize,
        weight: &[f32],
        epsilon: f32,
    ) {
        for head in values[..heads * head_dim].chunks_exact_mut(head_dim) {
            let mean_square = sum_squares(head) / head_dim as f32;
            let scale = (mean_square + epsilon).sqrt().recip();
            for (value, weight) in head.iter_mut().zip(weight) {
                *value = *value * scale * weight;
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    rms_norm_into(input, weight, epsilon, &mut output);
    output
}

simd_fn! {
    pub(crate) fn swiglu(gate: &mut [f32], up: &[f32]) {
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = silu(*gate) * up;
        }
    }
}

simd_fn! {
    pub(crate) fn add_assign(state: &mut [f32], delta: &[f32]) {
        for (state, delta) in state.iter_mut().zip(delta) {
            *state += delta;
        }
    }
}

/// 归一化的 Walsh–Hadamard 变换，逐头原地进行。它是正交且自逆的，长度须为 2 的幂。
pub(crate) fn hadamard_heads(values: &mut [f32], head_dim: usize) {
    debug_assert!(head_dim.is_power_of_two());
    let norm = (head_dim as f32).sqrt().recip();
    for head in values.chunks_exact_mut(head_dim) {
        let mut width = 1;
        while width < head_dim {
            for start in (0..head_dim).step_by(width * 2) {
                for index in start..start + width {
                    let (left, right) = (head[index], head[index + width]);
                    head[index] = left + right;
                    head[index + width] = left - right;
                }
            }
            width *= 2;
        }
        for value in head.iter_mut() {
            *value *= norm;
        }
    }
}

pub(crate) fn rope_frequencies(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|index| {
            let power = (2 * index) as f32 / head_dim as f32;
            1.0 / crate::math::pow(f64::from(theta), f64::from(power)) as f32
        })
        .collect()
}

pub(crate) fn rope_table(frequencies: &[f32], position: usize, table: &mut Vec<(f32, f32)>) {
    table.clear();
    table.extend(frequencies.iter().map(|frequency| {
        let angle = position as f32 * frequency;
        let (sine, cosine) = crate::math::sin_cos(f64::from(angle));
        (sine as f32, cosine as f32)
    }));
}

pub(crate) fn apply_rope_table(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    table: &[(f32, f32)],
    mode: RopeMode,
) {
    let half = head_dim / 2;
    debug_assert_eq!(table.len(), half);
    for head in values[..heads * head_dim].chunks_exact_mut(head_dim) {
        match mode {
            RopeMode::HalfSplit => {
                let (left, right) = head.split_at_mut(half);
                for ((left, right), (sine, cosine)) in left.iter_mut().zip(right).zip(table) {
                    let (first, second) = (*left, *right);
                    *left = first * cosine - second * sine;
                    *right = second * cosine + first * sine;
                }
            }
            RopeMode::Interleaved => {
                for (pair, (sine, cosine)) in head.chunks_exact_mut(2).zip(table) {
                    let (first, second) = (pair[0], pair[1]);
                    pair[0] = first * cosine - second * sine;
                    pair[1] = second * cosine + first * sine;
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn apply_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
    mode: RopeMode,
) {
    let mut table = Vec::new();
    rope_table(&rope_frequencies(head_dim, theta), position, &mut table);
    apply_rope_table(values, heads, head_dim, &table, mode);
}

#[cfg(test)]
fn softmax_in_place(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for value in values.iter_mut() {
        *value = exp(*value - maximum);
    }
    let total = sum(values);
    for value in values.iter_mut() {
        *value /= total;
    }
}

#[cfg(test)]
pub(crate) fn softmax(values: &[f32]) -> Vec<f32> {
    let mut output = values.to_vec();
    softmax_in_place(&mut output);
    output
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionShape {
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
}

pub(crate) const ATTENTION_CHUNK: usize = if cfg!(test) { 8 } else { 256 };
const PARTIAL_LIMIT: usize = if cfg!(test) { 4096 } else { 1 << 22 };

pub(crate) trait AttendValue: KvValue {
    #[allow(clippy::too_many_arguments)]
    fn chunk(
        queries: &[f32],
        keys: &[Self],
        values: &[Self],
        scales: RowScales<'_>,
        head_dim: usize,
        count: usize,
        scores: &mut [f32],
        partials: &mut [f32],
    );
}

impl AttendValue for f32 {
    fn chunk(
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        scales: RowScales<'_>,
        head_dim: usize,
        count: usize,
        scores: &mut [f32],
        partials: &mut [f32],
    ) {
        let rows = Rows {
            keys,
            values,
            scales,
            head_dim,
            count,
        };
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                if head_dim % (LANES * 2) == 0 {
                    chunk_blocked_f32(queries, rows, scores, partials);
                } else {
                    chunk_f32_avx2(queries, rows, scores, partials);
                }
            }
            return;
        }
        chunk_with(queries, rows, scores, partials, dot, axpy);
    }
}

impl AttendValue for F16 {
    fn chunk(
        queries: &[f32],
        keys: &[F16],
        values: &[F16],
        scales: RowScales<'_>,
        head_dim: usize,
        count: usize,
        scores: &mut [f32],
        partials: &mut [f32],
    ) {
        let rows = Rows {
            keys,
            values,
            scales,
            head_dim,
            count,
        };
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            unsafe {
                if head_dim % (LANES * 2) == 0 {
                    chunk_blocked_f16(queries, rows, scores, partials);
                } else {
                    chunk_f16_avx2(queries, rows, scores, partials);
                }
            }
            return;
        }
        chunk_with(queries, rows, scores, partials, dot, axpy);
    }
}

impl AttendValue for I8 {
    fn chunk(
        queries: &[f32],
        keys: &[I8],
        values: &[I8],
        scales: RowScales<'_>,
        head_dim: usize,
        count: usize,
        scores: &mut [f32],
        partials: &mut [f32],
    ) {
        let rows = Rows {
            keys,
            values,
            scales,
            head_dim,
            count,
        };
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && head_dim % (LANES * 2) == 0 {
            unsafe { chunk_blocked_i8(queries, rows, scores, partials) };
            return;
        }
        chunk_with(queries, rows, scores, partials, dot, axpy);
    }
}

/// 一段连续位置的 K/V 行及其缩放系数。
#[derive(Clone, Copy)]
struct Rows<'a, K> {
    keys: &'a [K],
    values: &'a [K],
    scales: RowScales<'a>,
    head_dim: usize,
    count: usize,
}

/// 键的缩放系数乘到注意力分数上。顺序与各实现保持一致，保证逐位相同。
#[inline(always)]
fn scale_scores(scores: &mut [f32], scales: RowScales<'_>) {
    if let Some((keys, _)) = scales {
        for (score, factor) in scores.iter_mut().zip(keys) {
            *score *= factor;
        }
    }
}

/// 值的缩放系数乘到注意力权重上；权重之和须在此之前算好。
#[inline(always)]
fn scale_weights(weights: &mut [f32], scales: RowScales<'_>) {
    if let Some((_, values)) = scales {
        for (weight, factor) in weights.iter_mut().zip(values) {
            *weight *= factor;
        }
    }
}

const NEGLIGIBLE_EXPONENT: f32 = -41.0;

#[inline(always)]
fn attention_weight(value: f32) -> f32 {
    if value < NEGLIGIBLE_EXPONENT {
        0.0
    } else {
        exp(value)
    }
}

#[inline(always)]
fn exponentiate(scores: &mut [f32]) -> (f32, f32) {
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for value in scores.iter_mut() {
        *value = attention_weight(*value - maximum);
    }
    (maximum, sum(scores))
}

fn merge_partial(state: &mut [f32], partial: &[f32]) {
    let maximum = state[0].max(partial[0]);
    let old = attention_weight(state[0] - maximum);
    let new = attention_weight(partial[0] - maximum);
    state[0] = maximum;
    state[1] = state[1] * old + partial[1] * new;
    for (value, update) in state[2..].iter_mut().zip(&partial[2..]) {
        *value = *value * old + update * new;
    }
}

fn finish_partial(state: &[f32], output: &mut [f32]) {
    let total = state[1];
    for (output, value) in output.iter_mut().zip(&state[2..]) {
        *output = value / total;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let length = left.len().min(right.len());
    let blocks = length / (LANES * 2);
    let mut low = _mm256_setzero_ps();
    let mut high = _mm256_setzero_ps();
    for block in 0..blocks {
        unsafe {
            let left = left.as_ptr().add(block * LANES * 2);
            let right = right.as_ptr().add(block * LANES * 2);
            low = _mm256_add_ps(
                low,
                _mm256_mul_ps(_mm256_loadu_ps(left), _mm256_loadu_ps(right)),
            );
            high = _mm256_add_ps(
                high,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.add(LANES)),
                    _mm256_loadu_ps(right.add(LANES)),
                ),
            );
        }
    }
    let mut folded = [0.0_f32; LANES];
    unsafe { _mm256_storeu_ps(folded.as_mut_ptr(), _mm256_add_ps(low, high)) };
    let mut total = sum_lanes(folded);
    for index in blocks * LANES * 2..length {
        total += left[index] * right[index];
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn dot_f16_avx2(left: &[f32], right: &[F16]) -> f32 {
    use std::arch::x86_64::*;

    let length = left.len().min(right.len());
    let blocks = length / (LANES * 2);
    let mut low = _mm256_setzero_ps();
    let mut high = _mm256_setzero_ps();
    for block in 0..blocks {
        unsafe {
            let left = left.as_ptr().add(block * LANES * 2);
            let right = right.as_ptr().add(block * LANES * 2).cast::<__m128i>();
            low = _mm256_add_ps(
                low,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left),
                    _mm256_cvtph_ps(_mm_loadu_si128(right)),
                ),
            );
            high = _mm256_add_ps(
                high,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.add(LANES)),
                    _mm256_cvtph_ps(_mm_loadu_si128(right.add(1))),
                ),
            );
        }
    }
    let mut folded = [0.0_f32; LANES];
    unsafe { _mm256_storeu_ps(folded.as_mut_ptr(), _mm256_add_ps(low, high)) };
    let mut total = sum_lanes(folded);
    for index in blocks * LANES * 2..length {
        total += left[index] * right[index].to_f32();
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn axpy_f16_avx2(output: &mut [f32], scale: f32, values: &[F16]) {
    use std::arch::x86_64::*;

    let length = output.len().min(values.len());
    let blocks = length / LANES;
    let factor = _mm256_set1_ps(scale);
    for block in 0..blocks {
        unsafe {
            let output = output.as_mut_ptr().add(block * LANES);
            let value = _mm256_cvtph_ps(_mm_loadu_si128(values.as_ptr().add(block * LANES).cast()));
            _mm256_storeu_ps(
                output,
                _mm256_add_ps(_mm256_loadu_ps(output), _mm256_mul_ps(factor, value)),
            );
        }
    }
    for index in blocks * LANES..length {
        output[index] += scale * values[index].to_f32();
    }
}

#[inline(always)]
fn chunk_with<K: KvValue>(
    queries: &[f32],
    rows: Rows<'_, K>,
    scores: &mut [f32],
    partials: &mut [f32],
    dot: impl Fn(&[f32], &[K]) -> f32,
    axpy: impl Fn(&mut [f32], f32, &[K]),
) {
    let Rows {
        keys,
        values,
        scales,
        head_dim,
        count,
    } = rows;
    let scale = (head_dim as f32).sqrt().recip();
    let width = head_dim + 2;
    for index in 0..count {
        let key = &keys[index * head_dim..(index + 1) * head_dim];
        for (query, scores) in queries
            .chunks_exact(head_dim)
            .zip(scores.chunks_exact_mut(count))
        {
            scores[index] = dot(query, key) * scale;
        }
    }
    for (scores, partial) in scores
        .chunks_exact_mut(count)
        .zip(partials.chunks_exact_mut(width))
    {
        scale_scores(scores, scales);
        let (maximum, total) = exponentiate(scores);
        scale_weights(scores, scales);
        partial[0] = maximum;
        partial[1] = total;
        partial[2..].fill(0.0);
    }
    for index in 0..count {
        let value = &values[index * head_dim..(index + 1) * head_dim];
        for (partial, scores) in partials
            .chunks_exact_mut(width)
            .zip(scores.chunks_exact(count))
        {
            axpy(&mut partial[2..], scores[index], value);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn chunk_f32_avx2(
    queries: &[f32],
    rows: Rows<'_, f32>,
    scores: &mut [f32],
    partials: &mut [f32],
) {
    chunk_with(
        queries,
        rows,
        scores,
        partials,
        |left, right| unsafe { dot_avx2(left, right) },
        axpy,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn chunk_f16_avx2(
    queries: &[f32],
    rows: Rows<'_, F16>,
    scores: &mut [f32],
    partials: &mut [f32],
) {
    chunk_with(
        queries,
        rows,
        scores,
        partials,
        |left, right| unsafe { dot_f16_avx2(left, right) },
        |output, scale, values| unsafe { axpy_f16_avx2(output, scale, values) },
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn horizontal_sum(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let folded = _mm256_add_ps(value, _mm256_permute2f128_ps::<1>(value, value));
        let low = _mm256_castps256_ps128(folded);
        let pairs = _mm_add_ps(low, _mm_movehl_ps(low, low));
        _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<1>(pairs, pairs)))
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_rows<K, const ROWS: usize>(
    query: *const f32,
    keys: [*const K; ROWS],
    head_dim: usize,
    load: &impl Fn(*const K) -> std::arch::x86_64::__m256,
) -> [f32; ROWS] {
    use std::arch::x86_64::*;

    unsafe {
        let mut low = [_mm256_setzero_ps(); ROWS];
        let mut high = [_mm256_setzero_ps(); ROWS];
        for offset in (0..head_dim).step_by(LANES * 2) {
            let first = _mm256_loadu_ps(query.add(offset));
            let second = _mm256_loadu_ps(query.add(offset + LANES));
            for row in 0..ROWS {
                low[row] =
                    _mm256_add_ps(low[row], _mm256_mul_ps(first, load(keys[row].add(offset))));
                high[row] = _mm256_add_ps(
                    high[row],
                    _mm256_mul_ps(second, load(keys[row].add(offset + LANES))),
                );
            }
        }
        std::array::from_fn(|row| horizontal_sum(_mm256_add_ps(low[row], high[row])))
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn accumulate_values<K, const VECTORS: usize>(
    probabilities: &[f32],
    values: *const K,
    head_dim: usize,
    output: *mut f32,
    load: &impl Fn(*const K) -> std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;

    unsafe {
        let mut sums = [_mm256_setzero_ps(); VECTORS];
        for (index, probability) in probabilities.iter().enumerate() {
            let factor = _mm256_set1_ps(*probability);
            let row = values.add(index * head_dim);
            for (vector, sum) in sums.iter_mut().enumerate() {
                *sum = _mm256_add_ps(*sum, _mm256_mul_ps(factor, load(row.add(vector * LANES))));
            }
        }
        for (vector, sum) in sums.iter().enumerate() {
            _mm256_storeu_ps(output.add(vector * LANES), *sum);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn chunk_blocked<K: KvValue>(
    queries: &[f32],
    rows: Rows<'_, K>,
    scores: &mut [f32],
    partials: &mut [f32],
    load: impl Fn(*const K) -> std::arch::x86_64::__m256,
) {
    let Rows {
        keys,
        values,
        scales,
        head_dim,
        count,
    } = rows;
    const ROWS: usize = 4;
    const WIDE: usize = 8;
    let heads = queries.len() / head_dim;
    let width = head_dim + 2;
    assert!(head_dim % (LANES * 2) == 0);
    assert!(keys.len() >= count * head_dim && values.len() >= count * head_dim);
    assert!(scores.len() >= heads * count && partials.len() >= heads * width);
    let scale = (head_dim as f32).sqrt().recip();
    for head in 0..heads {
        let query = queries[head * head_dim..].as_ptr();
        let scores = &mut scores[head * count..(head + 1) * count];
        let mut index = 0;
        while index + ROWS <= count {
            let rows = std::array::from_fn(|row| keys[(index + row) * head_dim..].as_ptr());
            let dots = unsafe { dot_rows::<K, ROWS>(query, rows, head_dim, &load) };
            for (row, dot) in dots.iter().enumerate() {
                scores[index + row] = dot * scale;
            }
            index += ROWS;
        }
        while index < count {
            let rows = [keys[index * head_dim..].as_ptr()];
            scores[index] = unsafe { dot_rows::<K, 1>(query, rows, head_dim, &load) }[0] * scale;
            index += 1;
        }
        scale_scores(scores, scales);
        let (maximum, total) = exponentiate(scores);
        scale_weights(scores, scales);
        partials[head * width] = maximum;
        partials[head * width + 1] = total;
    }
    for head in 0..heads {
        let probabilities = &scores[head * count..(head + 1) * count];
        let output = partials[head * width + 2..(head + 1) * width].as_mut_ptr();
        let mut start = 0;
        while start + WIDE * LANES <= head_dim {
            unsafe {
                accumulate_values::<K, WIDE>(
                    probabilities,
                    values.as_ptr().add(start),
                    head_dim,
                    output.add(start),
                    &load,
                )
            };
            start += WIDE * LANES;
        }
        while start < head_dim {
            unsafe {
                accumulate_values::<K, 2>(
                    probabilities,
                    values.as_ptr().add(start),
                    head_dim,
                    output.add(start),
                    &load,
                )
            };
            start += LANES * 2;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn chunk_blocked_f32(
    queries: &[f32],
    rows: Rows<'_, f32>,
    scores: &mut [f32],
    partials: &mut [f32],
) {
    use std::arch::x86_64::*;

    unsafe {
        chunk_blocked(queries, rows, scores, partials, |pointer: *const f32| {
            _mm256_loadu_ps(pointer)
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn chunk_blocked_f16(
    queries: &[f32],
    rows: Rows<'_, F16>,
    scores: &mut [f32],
    partials: &mut [f32],
) {
    use std::arch::x86_64::*;

    unsafe {
        chunk_blocked(queries, rows, scores, partials, |pointer: *const F16| {
            _mm256_cvtph_ps(_mm_loadu_si128(pointer.cast()))
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn chunk_blocked_i8(
    queries: &[f32],
    rows: Rows<'_, I8>,
    scores: &mut [f32],
    partials: &mut [f32],
) {
    use std::arch::x86_64::*;

    unsafe {
        chunk_blocked(queries, rows, scores, partials, |pointer: *const I8| {
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(pointer.cast())))
        })
    }
}

pub(crate) fn attention<K: AttendValue>(
    queries: &[f32],
    kv: KvView<'_, K>,
    shape: AttentionShape,
    first: usize,
    scratch: &mut Vec<f32>,
    output: &mut [f32],
) {
    const PARALLEL_PRODUCTS: usize = 32_768;
    let AttentionShape {
        head_count,
        kv_head_count,
        head_dim,
    } = shape;
    let q_dim = head_count * head_dim;
    let tokens = queries.len() / q_dim;
    let group = head_count / kv_head_count;
    let width = head_dim + 2;
    let longest = first + tokens;
    let output = &mut output[..tokens * q_dim];
    let parallel = tokens * q_dim * longest >= PARALLEL_PRODUCTS;
    let chunks = longest.div_ceil(ATTENTION_CHUNK);
    let partial_length = tokens * chunks * head_count * width;
    if chunks > 1 && partial_length <= PARTIAL_LIMIT {
        let tasks = kv_head_count * chunks;
        let task_scores = group * ATTENTION_CHUNK;
        scratch.resize(partial_length + tasks * task_scores, 0.0);
        let (partials, scores) = scratch.split_at_mut(partial_length);
        let partial_pointer = SharedPtr(partials.as_mut_ptr());
        let score_pointer = SharedPtr(scores.as_mut_ptr());
        let output_pointer = SharedPtr(output.as_mut_ptr());
        let run = |task: usize| {
            let (kv_head, chunk) = (task / chunks, task % chunks);
            let begin = chunk * ATTENTION_CHUNK;
            let (keys, values) = kv.head(kv_head, longest, head_dim);
            let row_scales = kv.scales(kv_head, longest);
            let scores = unsafe {
                std::slice::from_raw_parts_mut(
                    score_pointer.get().add(task * task_scores),
                    task_scores,
                )
            };
            for token in 0..tokens {
                let count = first + token + 1;
                if count <= begin {
                    continue;
                }
                let end = (begin + ATTENTION_CHUNK).min(count);
                let length = end - begin;
                let query = token * q_dim + kv_head * group * head_dim;
                let slot = ((token * chunks + chunk) * head_count + kv_head * group) * width;
                let partial = unsafe {
                    std::slice::from_raw_parts_mut(partial_pointer.get().add(slot), group * width)
                };
                K::chunk(
                    &queries[query..query + group * head_dim],
                    &keys[begin * head_dim..end * head_dim],
                    &values[begin * head_dim..end * head_dim],
                    row_scales.map(|(keys, values)| (&keys[begin..end], &values[begin..end])),
                    head_dim,
                    length,
                    &mut scores[..group * length],
                    partial,
                );
            }
        };
        if parallel {
            parallel_for(tasks, &run);
        } else {
            for task in 0..tasks {
                run(task);
            }
        }
        if chunks > 1 {
            let merge = |task: usize| {
                let (token, head) = (task / head_count, task % head_count);
                let used = (first + token + 1).div_ceil(ATTENTION_CHUNK);
                let slot = |chunk: usize| ((token * chunks + chunk) * head_count + head) * width;
                let state = unsafe {
                    std::slice::from_raw_parts_mut(partial_pointer.get().add(slot(0)), width)
                };
                for chunk in 1..used {
                    let partial = unsafe {
                        std::slice::from_raw_parts(partial_pointer.get().add(slot(chunk)), width)
                    };
                    merge_partial(state, partial);
                }
                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        output_pointer.get().add(token * q_dim + head * head_dim),
                        head_dim,
                    )
                };
                finish_partial(state, output);
            };
            if parallel {
                parallel_for(tokens * head_count, &merge);
            } else {
                for task in 0..tokens * head_count {
                    merge(task);
                }
            }
        }
    } else {
        let tasks = kv_head_count * tokens;
        let per_task = group * (ATTENTION_CHUNK + 2 * width);
        scratch.resize(tasks * per_task, 0.0);
        let scratch_pointer = SharedPtr(scratch.as_mut_ptr());
        let output_pointer = SharedPtr(output.as_mut_ptr());
        let run = |task: usize| {
            let (token, kv_head) = (task / kv_head_count, task % kv_head_count);
            let local = unsafe {
                std::slice::from_raw_parts_mut(scratch_pointer.get().add(task * per_task), per_task)
            };
            let (scores, rest) = local.split_at_mut(group * ATTENTION_CHUNK);
            let (state, partial) = rest.split_at_mut(group * width);
            let count = first + token + 1;
            let (keys, values) = kv.head(kv_head, count, head_dim);
            let row_scales = kv.scales(kv_head, count);
            let query = token * q_dim + kv_head * group * head_dim;
            let queries = &queries[query..query + group * head_dim];
            for chunk in 0..count.div_ceil(ATTENTION_CHUNK) {
                let begin = chunk * ATTENTION_CHUNK;
                let end = (begin + ATTENTION_CHUNK).min(count);
                let length = end - begin;
                let target = if chunk == 0 {
                    &mut *state
                } else {
                    &mut *partial
                };
                K::chunk(
                    queries,
                    &keys[begin * head_dim..end * head_dim],
                    &values[begin * head_dim..end * head_dim],
                    row_scales.map(|(keys, values)| (&keys[begin..end], &values[begin..end])),
                    head_dim,
                    length,
                    &mut scores[..group * length],
                    target,
                );
                if chunk > 0 {
                    for head in 0..group {
                        merge_partial(
                            &mut state[head * width..(head + 1) * width],
                            &partial[head * width..(head + 1) * width],
                        );
                    }
                }
            }
            let output = unsafe {
                std::slice::from_raw_parts_mut(output_pointer.get().add(query), group * head_dim)
            };
            for head in 0..group {
                finish_partial(
                    &state[head * width..(head + 1) * width],
                    &mut output[head * head_dim..(head + 1) * head_dim],
                );
            }
        };
        if parallel {
            parallel_for(tasks, &run);
        } else {
            for task in 0..tasks {
                run(task);
            }
        }
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
    fn f32_linear_paths_match_bit_for_bit_and_track_f64() {
        for (rows, columns, tokens) in [(7, 37, 3), (5, 64, 1), (9, 1024, 4)] {
            let weights = wave(rows * columns, 0.071);
            let inputs = wave(tokens * columns, 0.13);
            let bias = wave(rows, 0.5);
            for bias in [None, Some(&bias[..])] {
                let mut dispatched = vec![0.0; tokens * rows];
                let mut generic = vec![0.0; tokens * rows];
                unsafe {
                    linear_rows(
                        &weights,
                        columns,
                        &inputs,
                        0..rows,
                        bias,
                        dispatched.as_mut_ptr(),
                        rows,
                    );
                    linear_rows_with(
                        &weights,
                        columns,
                        &inputs,
                        0..rows,
                        bias,
                        generic.as_mut_ptr(),
                        rows,
                        dot,
                        |_| {},
                    );
                }
                assert_eq!(
                    dispatched.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    generic.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                for token in 0..tokens {
                    for row in 0..rows {
                        let expected: f64 = weights[row * columns..(row + 1) * columns]
                            .iter()
                            .zip(&inputs[token * columns..(token + 1) * columns])
                            .map(|(w, x)| f64::from(*w) * f64::from(*x))
                            .sum::<f64>()
                            + bias.map_or(0.0, |b| f64::from(b[row]));
                        let actual = f64::from(dispatched[token * rows + row]);
                        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
                    }
                }
            }
        }
    }

    #[test]
    fn exp_matches_std_within_relative_tolerance() {
        let mut value = -87.0_f32;
        while value <= 88.0 {
            let expected = value.exp();
            let actual = exp(value);
            assert!(
                (actual - expected).abs() <= expected * 4e-7,
                "exp({value}) = {actual}, expected {expected}"
            );
            value += 0.137;
        }
        assert_eq!(exp(0.0), 1.0);
        assert!(exp(-1000.0) > 0.0 && exp(-1000.0) < 1e-37);
        assert!(exp(1000.0).is_finite());
        assert!(exp(f32::NAN).is_nan() && exp(f32::INFINITY).is_nan());
    }

    #[test]
    fn lane_reductions_match_sequential_sums() {
        let values: Vec<f32> = (0..37).map(|index| (index as f32 - 18.0) / 7.0).collect();
        let other: Vec<f32> = (0..37).map(|index| (index as f32 * 0.31).sin()).collect();
        let expected_sum: f32 = values.iter().sum();
        let expected_squares: f32 = values.iter().map(|value| value * value).sum();
        let expected_dot: f32 = values.iter().zip(&other).map(|(l, r)| l * r).sum();
        assert!((sum(&values) - expected_sum).abs() < 1e-4);
        assert!((sum_squares(&values) - expected_squares).abs() < 1e-4);
        assert!((dot(&values, &other) - expected_dot).abs() < 1e-4);
    }

    fn wave(count: usize, step: f32) -> Vec<f32> {
        (0..count)
            .map(|index| (index as f32 * step).sin())
            .collect()
    }

    fn reference_chunks<K: AttendValue>(
        queries: &[f32],
        view: KvView<'_, K>,
        shape: AttentionShape,
        first: usize,
    ) -> Vec<f32> {
        let head_dim = shape.head_dim;
        let width = head_dim + 2;
        let q_dim = shape.head_count * head_dim;
        let tokens = queries.len() / q_dim;
        let group = shape.head_count / shape.kv_head_count;
        let mut output = vec![0.0; tokens * q_dim];
        for token in 0..tokens {
            let count = first + token + 1;
            for head in 0..shape.head_count {
                let (keys, values) = view.head(head / group, count, head_dim);
                let row_scales = view.scales(head / group, count);
                let query = &queries[token * q_dim + head * head_dim..][..head_dim];
                let mut state = vec![0.0; width];
                for chunk in 0..count.div_ceil(ATTENTION_CHUNK) {
                    let begin = chunk * ATTENTION_CHUNK;
                    let end = (begin + ATTENTION_CHUNK).min(count);
                    let mut scores = vec![0.0; end - begin];
                    let mut partial = vec![0.0; width];
                    let rows = Rows {
                        keys: &keys[begin * head_dim..end * head_dim],
                        values: &values[begin * head_dim..end * head_dim],
                        scales: row_scales
                            .map(|(keys, values)| (&keys[begin..end], &values[begin..end])),
                        head_dim,
                        count: end - begin,
                    };
                    chunk_with(query, rows, &mut scores, &mut partial, dot, axpy);
                    if chunk == 0 {
                        state = partial;
                    } else {
                        merge_partial(&mut state, &partial);
                    }
                }
                finish_partial(
                    &state,
                    &mut output[token * q_dim + head * head_dim..][..head_dim],
                );
            }
        }
        output
    }

    fn naive_attention<K: AttendValue>(
        queries: &[f32],
        view: KvView<'_, K>,
        shape: AttentionShape,
        first: usize,
    ) -> Vec<f32> {
        let head_dim = shape.head_dim;
        let q_dim = shape.head_count * head_dim;
        let tokens = queries.len() / q_dim;
        let group = shape.head_count / shape.kv_head_count;
        let mut output = vec![0.0; tokens * q_dim];
        for token in 0..tokens {
            let count = first + token + 1;
            for head in 0..shape.head_count {
                let (keys, values) = view.head(head / group, count, head_dim);
                let (key_scale, value_scale) = match view.scales(head / group, count) {
                    Some((keys, values)) => (keys.to_vec(), values.to_vec()),
                    None => (vec![1.0; count], vec![1.0; count]),
                };
                let query = &queries[token * q_dim + head * head_dim..][..head_dim];
                let scores: Vec<f64> = (0..count)
                    .map(|index| {
                        query
                            .iter()
                            .zip(&keys[index * head_dim..(index + 1) * head_dim])
                            .map(|(q, k)| f64::from(*q) * f64::from(key_scale[index] * k.to_f32()))
                            .sum::<f64>()
                            / (head_dim as f64).sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let total: f64 = weights.iter().sum();
                for dimension in 0..head_dim {
                    let value: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(index, weight)| {
                            weight
                                * f64::from(
                                    value_scale[index]
                                        * values[index * head_dim + dimension].to_f32(),
                                )
                        })
                        .sum();
                    output[token * q_dim + head * head_dim + dimension] = (value / total) as f32;
                }
            }
        }
        output
    }

    fn attention_case<K: AttendValue>(head_dim: usize, group: usize, first: usize, tokens: usize) {
        let shape = AttentionShape {
            head_count: 2 * group,
            kv_head_count: 2,
            head_dim,
        };
        let q_dim = shape.head_count * head_dim;
        let longest = first + tokens;
        let capacity = longest + 3;
        let encode = |values: Vec<f32>| {
            let mut stored = vec![K::default(); values.len()];
            let mut scales = Vec::new();
            for (row, output) in values.chunks(head_dim).zip(stored.chunks_mut(head_dim)) {
                scales.push(K::store_row(row, output));
            }
            if !K::SCALED {
                scales.clear();
            }
            (stored, scales)
        };
        let (keys, key_scales) = encode(
            wave(capacity * 2 * head_dim, 0.37)
                .into_iter()
                .enumerate()
                .map(|(index, value)| value * (1.0 + (index / head_dim % 5) as f32))
                .collect(),
        );
        let (values, value_scales) = encode(wave(capacity * 2 * head_dim, 0.11));
        let view = KvView {
            keys: &keys,
            values: &values,
            key_scales: &key_scales,
            value_scales: &value_scales,
            capacity,
        };
        let queries = wave(tokens * q_dim, 0.23);
        let mut scratch = Vec::new();
        let mut dispatched = vec![0.0; tokens * q_dim];
        attention(&queries, view, shape, first, &mut scratch, &mut dispatched);
        let reference = reference_chunks(&queries, view, shape, first);
        assert_eq!(
            dispatched.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "head_dim={head_dim} group={group} first={first} tokens={tokens}"
        );
        let naive = naive_attention(&queries, view, shape, first);
        for (actual, expected) in dispatched.iter().zip(&naive) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        assert!(dispatched.iter().any(|value| value.abs() > 1e-3));
        for token in 0..tokens {
            let mut single = vec![0.0; q_dim];
            attention(
                &queries[token * q_dim..(token + 1) * q_dim],
                view,
                shape,
                first + token,
                &mut scratch,
                &mut single,
            );
            assert_eq!(&single[..], &dispatched[token * q_dim..(token + 1) * q_dim]);
        }
    }

    #[test]
    fn attention_paths_match_bit_for_bit() {
        for (head_dim, group, first, tokens) in [
            (40, 2, 6, 3),
            (64, 2, 6, 3),
            (128, 3, 10, 3),
            (32, 1, 0, 3),
            (64, 2, 21, 2),
            (128, 2, 3, 9),
            (40, 2, 30, 6),
        ] {
            attention_case::<f32>(head_dim, group, first, tokens);
            attention_case::<F16>(head_dim, group, first, tokens);
            attention_case::<I8>(head_dim, group, first, tokens);
        }
    }

    #[test]
    fn hadamard_is_orthogonal_and_self_inverse() {
        let original = wave(3 * 16, 0.7);
        let mut rotated = original.clone();
        hadamard_heads(&mut rotated, 16);
        for (before, after) in original.chunks(16).zip(rotated.chunks(16)) {
            let norm = |values: &[f32]| values.iter().map(|v| v * v).sum::<f32>();
            assert!((norm(before) - norm(after)).abs() < 1e-4);
        }
        let mut other = wave(3 * 16, 0.3);
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let expected = dot(&original[..16], &other[..16]);
        hadamard_heads(&mut other, 16);
        assert!((dot(&rotated[..16], &other[..16]) - expected).abs() < 1e-4);
        hadamard_heads(&mut rotated, 16);
        for (back, value) in rotated.iter().zip(&original) {
            assert!((back - value).abs() < 1e-5);
        }
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
