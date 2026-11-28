use std::ops::Range;

use crate::pool::{SharedPtr, parallel_for};
use crate::q8::{Activation8, Q8Matrix};
use crate::{Error, Result};

pub(crate) const GROUP_SIZE: usize = 32;
const NIBBLE_BYTES: usize = GROUP_SIZE / 2;
const SCALE_BYTES: usize = 4;
const LANES: usize = 8;
const MIN_PEAK: f32 = 1.0e-30;
const CHUNKS_PER_THREAD: usize = 4;
#[cfg(target_arch = "x86_64")]
const PREFETCH_BYTES: usize = 4096;
const PARALLEL_WEIGHTS: usize = 65_536;
const MAX_JOBS: usize = 3;

#[derive(Debug, Clone)]
pub(crate) enum Weight {
    F32(Vec<f32>),
    Q4(Q4Matrix),
    Q8(Q8Matrix),
}

impl Default for Weight {
    fn default() -> Self {
        Self::F32(Vec::new())
    }
}

impl Weight {
    /// 全部权重为零。Q4 只需检查每行的 scale：scale 为零时整行还原值都是零。
    pub(crate) fn is_zero(&self) -> bool {
        match self {
            Self::F32(values) => values.iter().all(|value| *value == 0.0),
            Self::Q4(matrix) => (0..matrix.rows).all(|row| matrix.row_scale(row) == 0.0),
            Self::Q8(matrix) => matrix.is_zero(),
        }
    }

    pub fn into_f32(self) -> Vec<f32> {
        match self {
            Self::F32(values) => values,
            Self::Q4(matrix) => {
                let mut output = vec![0.0; matrix.rows * matrix.columns];
                if matrix.columns > 0 {
                    for (row, values) in output.chunks_exact_mut(matrix.columns).enumerate() {
                        matrix.dequantize_row(row, values);
                    }
                }
                output
            }
            Self::Q8(matrix) => {
                let mut output = vec![0.0; matrix.rows * matrix.columns];
                if matrix.columns > 0 {
                    for (row, values) in output.chunks_exact_mut(matrix.columns).enumerate() {
                        matrix.dequantize_row(row, values);
                    }
                }
                output
            }
        }
    }

    pub fn row_into(&self, row: usize, output: &mut [f32]) {
        match self {
            Self::F32(values) => {
                let columns = output.len();
                output.copy_from_slice(&values[row * columns..(row + 1) * columns]);
            }
            Self::Q4(matrix) => matrix.dequantize_row(row, output),
            Self::Q8(matrix) => matrix.dequantize_row(row, output),
        }
    }

    #[cfg(test)]
    pub fn row(&self, row: usize, columns: usize) -> Vec<f32> {
        let mut output = vec![0.0; columns];
        self.row_into(row, &mut output);
        output
    }
}

pub(crate) const fn groups(columns: usize) -> usize {
    columns.div_ceil(GROUP_SIZE)
}

pub(crate) fn matrix_bytes(rows: usize, columns: usize) -> Option<usize> {
    groups(columns)
        .checked_mul(1 + NIBBLE_BYTES)?
        .checked_add(SCALE_BYTES)?
        .checked_mul(rows)?
        .checked_next_multiple_of(4)
}

#[derive(Debug, Clone)]
pub(crate) struct Q4Matrix {
    pub rows: usize,
    pub columns: usize,
    groups: usize,
    bytes: Vec<u8>,
}

impl Q4Matrix {
    pub fn new(rows: usize, columns: usize, bytes: Vec<u8>) -> Result<Self> {
        let expected = matrix_bytes(rows, columns).ok_or(Error::InvalidModel("Q4 张量尺寸溢出"))?;
        if bytes.len() != expected {
            return Err(Error::InvalidModel("Q4 张量尺寸不匹配"));
        }
        let matrix = Self {
            rows,
            columns,
            groups: groups(columns),
            bytes,
        };
        for row in 0..rows {
            let scale = matrix.row_scale(row);
            if !scale.is_finite() || scale < 0.0 {
                return Err(Error::InvalidModel("Q4 缩放因子无效"));
            }
        }
        Ok(matrix)
    }

    pub(crate) fn row_scale(&self, row: usize) -> f32 {
        let offset = row * SCALE_BYTES;
        f32::from_le_bytes(
            self.bytes[offset..offset + SCALE_BYTES]
                .try_into()
                .expect("validated Q4 scale"),
        )
    }

    fn codes(&self) -> &[u8] {
        let start = self.rows * SCALE_BYTES;
        &self.bytes[start..start + self.rows * self.groups]
    }

    fn nibbles(&self) -> &[u8] {
        let start = self.rows * (SCALE_BYTES + self.groups);
        &self.bytes[start..start + self.rows * self.groups * NIBBLE_BYTES]
    }

    pub(crate) fn row_codes(&self, row: usize) -> &[u8] {
        &self.codes()[row * self.groups..(row + 1) * self.groups]
    }

    pub(crate) fn row_nibbles(&self, row: usize) -> &[u8] {
        let width = self.groups * NIBBLE_BYTES;
        &self.nibbles()[row * width..(row + 1) * width]
    }

    pub fn dequantize_row(&self, row: usize, output: &mut [f32]) {
        debug_assert_eq!(output.len(), self.columns);
        let scale = self.row_scale(row);
        let codes = self.row_codes(row);
        let nibbles = self.row_nibbles(row);
        for (index, value) in output.iter_mut().enumerate() {
            let step = scale * group_code(codes[index / GROUP_SIZE]);
            let byte = nibbles[index / 2];
            let nibble = if index % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            *value = step * f32::from(i16::from(nibble) - 8);
        }
    }

    unsafe fn compute_rows(
        &self,
        activations: &[Activation],
        rows: Range<usize>,
        bias: Option<&[f32]>,
        output: *mut f32,
        stride: usize,
    ) {
        assert!(rows.end <= self.rows);
        for activation in activations {
            assert!(activation.codes.len() >= self.groups);
            assert!(activation.offsets.len() >= self.groups);
            assert!(activation.scales.len() >= self.groups);
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { self.compute_rows_avx2(activations, rows, bias, output, stride) };
            return;
        }
        for (offset, row) in rows.enumerate() {
            for (token, activation) in activations.iter().enumerate() {
                let value = self.row_scalar(activation, row, bias);
                unsafe { output.add(token * stride + offset).write(value) };
            }
        }
    }

    #[cfg(test)]
    fn compute_rows_into(
        &self,
        activation: &Activation,
        rows: Range<usize>,
        bias: Option<&[f32]>,
        output: &mut [f32],
    ) {
        assert_eq!(output.len(), rows.len());
        unsafe {
            self.compute_rows(
                std::slice::from_ref(activation),
                rows,
                bias,
                output.as_mut_ptr(),
                0,
            )
        };
    }

    fn row_scalar(&self, activation: &Activation, row: usize, bias: Option<&[f32]>) -> f32 {
        let codes = self.row_codes(row);
        let nibbles = self.row_nibbles(row);
        let mut lanes = [0.0_f32; LANES];
        for group in 0..self.groups {
            let packed = &nibbles[group * NIBBLE_BYTES..(group + 1) * NIBBLE_BYTES];
            let inputs = &activation.codes[group].0;
            let weight = group_code(codes[group]) * activation.scales[group];
            for (lane, total) in lanes.iter_mut().enumerate() {
                let mut sum = 0_i32;
                for slot in lane * 4..lane * 4 + 4 {
                    let nibble = if slot < NIBBLE_BYTES {
                        packed[slot] & 0x0f
                    } else {
                        packed[slot - NIBBLE_BYTES] >> 4
                    };
                    sum += (i32::from(nibble) - 8) * i32::from(inputs[slot]);
                }
                *total += sum as f32 * weight;
            }
        }
        finish_row(self.row_scale(row), sum_lanes(lanes), bias, row)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_rows_avx2(
        &self,
        activations: &[Activation],
        rows: Range<usize>,
        bias: Option<&[f32]>,
        output: *mut f32,
        stride: usize,
    ) {
        use std::arch::x86_64::*;

        // 外层按 token 分组（每组最多 4 个），内层逐行：每组 token 的激活约 4 KB，留在 L1 里，
        // 每组 token 都要重读一遍的是这一段行的权重（每行约 0.5 KB），从 L2 读。
        // 反过来（逐行、行内遍历全部 token）时，prefill 一块 32 个 token 的激活约 40 KB，
        // 超出 L1，每一行都要从 L2 重读全部激活。每个 (行, token) 的求和顺序不变，结果逐位相同。
        let row_bytes = self.groups * NIBBLE_BYTES;
        let mut first = 0;
        while first < activations.len() {
            let tile = &activations[first..];
            let width = tile.len().min(4);
            for (offset, row) in rows.clone().enumerate() {
                let nibbles = self.row_nibbles(row).as_ptr();
                for line in 0..row_bytes.div_ceil(64) {
                    _mm_prefetch::<_MM_HINT_T0>(
                        nibbles.wrapping_add(PREFETCH_BYTES + line * 64).cast(),
                    );
                }
                let target = unsafe { output.add(first * stride + offset) };
                unsafe {
                    match width {
                        1 => self.row_tile_avx2::<1>(tile, row, bias, target, stride),
                        2 => self.row_tile_avx2::<2>(tile, row, bias, target, stride),
                        3 => self.row_tile_avx2::<3>(tile, row, bias, target, stride),
                        _ => self.row_tile_avx2::<4>(tile, row, bias, target, stride),
                    }
                };
            }
            first += width;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    #[allow(clippy::needless_range_loop)]
    unsafe fn row_tile_avx2<const TOKENS: usize>(
        &self,
        activations: &[Activation],
        row: usize,
        bias: Option<&[f32]>,
        output: *mut f32,
        stride: usize,
    ) -> usize {
        use std::arch::x86_64::*;

        let groups = self.groups;
        let codes = self.row_codes(row).as_ptr();
        let nibbles = self.row_nibbles(row).as_ptr();
        let inputs: [*const __m256i; TOKENS] =
            std::array::from_fn(|token| activations[token].codes.as_ptr().cast());
        let offsets: [*const __m256i; TOKENS] =
            std::array::from_fn(|token| activations[token].offsets.as_ptr().cast());
        let scales: [*const f32; TOKENS] =
            std::array::from_fn(|token| activations[token].scales.as_ptr());
        let mask = _mm256_set1_epi8(0x0f);
        let shifts = _mm256_setr_epi64x(0, 0, 4, 4);
        let ones = _mm256_set1_epi16(1);
        let blocks = groups / LANES;
        let mut lanes = [_mm256_setzero_ps(); TOKENS];
        let mut weights = [[0.0_f32; LANES]; TOKENS];
        let mut group = 0;
        for block in 0..=blocks {
            let count = if block < blocks {
                unsafe {
                    let codes = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        codes.add(group).cast(),
                    )));
                    for token in 0..TOKENS {
                        let scale = _mm256_loadu_ps(scales[token].add(group));
                        _mm256_storeu_ps(weights[token].as_mut_ptr(), _mm256_mul_ps(codes, scale));
                    }
                }
                LANES
            } else {
                let count = groups - group;
                for token in 0..TOKENS {
                    for (index, weight) in weights[token][..count].iter_mut().enumerate() {
                        *weight = unsafe {
                            group_code(*codes.add(group + index))
                                * *scales[token].add(group + index)
                        };
                    }
                }
                count
            };
            for index in 0..count {
                unsafe {
                    let packed = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        nibbles.add(group * NIBBLE_BYTES).cast::<__m128i>(),
                    ));
                    let quartets = _mm256_and_si256(_mm256_srlv_epi64(packed, shifts), mask);
                    for token in 0..TOKENS {
                        let input = _mm256_load_si256(inputs[token].add(group));
                        let offset = _mm256_load_si256(offsets[token].add(group));
                        let pairs = _mm256_sub_epi16(_mm256_maddubs_epi16(quartets, input), offset);
                        let products = _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, ones));
                        lanes[token] = _mm256_add_ps(
                            lanes[token],
                            _mm256_mul_ps(products, _mm256_broadcast_ss(&weights[token][index])),
                        );
                    }
                }
                group += 1;
            }
        }
        let scale = self.row_scale(row);
        for (token, lanes) in lanes.iter().enumerate() {
            let mut stored = [0.0_f32; LANES];
            unsafe {
                _mm256_storeu_ps(stored.as_mut_ptr(), *lanes);
                output
                    .add(token * stride)
                    .write(finish_row(scale, sum_lanes(stored), bias, row));
            }
        }
        TOKENS
    }
}

fn group_code(byte: u8) -> f32 {
    f32::from(byte as i8)
}

fn sum_lanes(lanes: [f32; LANES]) -> f32 {
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
}

fn finish_row(scale: f32, sum: f32, bias: Option<&[f32]>, row: usize) -> f32 {
    let value = scale * sum;
    match bias {
        Some(bias) => value + bias[row],
        None => value,
    }
}

#[derive(Debug, Default)]
pub(crate) struct Activation {
    codes: Vec<CodeBlock>,
    offsets: Vec<OffsetBlock>,
    scales: Vec<f32>,
    /// INT8 权重使用的 64 元素分块激活，只在有 INT8 矩阵时量化。
    pub(crate) int8: Activation8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(32))]
struct CodeBlock([i8; GROUP_SIZE]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(32))]
struct OffsetBlock([i16; NIBBLE_BYTES]);

impl Activation {
    pub fn quantize(&mut self, input: &[f32]) {
        let groups = groups(input.len());
        self.reset(groups);
        let full = input.len() / GROUP_SIZE;
        #[cfg(target_arch = "x86_64")]
        let vectorized = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let vectorized = false;
        if vectorized {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                self.quantize_groups_avx2(&input[..full * GROUP_SIZE]);
            }
        } else {
            for group in 0..full {
                self.quantize_group(group, &input[group * GROUP_SIZE..(group + 1) * GROUP_SIZE]);
            }
        }
        if full < groups {
            self.quantize_group(full, &input[full * GROUP_SIZE..]);
        }
    }

    fn reset(&mut self, groups: usize) {
        self.codes.clear();
        self.codes.resize(groups, CodeBlock([0; GROUP_SIZE]));
        self.offsets.clear();
        self.offsets.resize(groups, OffsetBlock([0; NIBBLE_BYTES]));
        self.scales.clear();
        self.scales.resize(groups, 0.0);
    }

    fn quantize_group(&mut self, group: usize, values: &[f32]) {
        let codes = &mut self.codes[group].0;
        let offsets = &mut self.offsets[group].0;
        let mut peak = 0.0_f32;
        let mut invalid = 0_u32;
        for value in values {
            peak = peak.max(value.abs());
            invalid |= u32::from(!value.is_finite());
        }
        let Some(inverse) = group_inverse(peak, invalid) else {
            self.scales[group] = if invalid != 0 { f32::NAN } else { 0.0 };
            return;
        };
        for (index, value) in values.iter().enumerate() {
            let slot = index / 2 + (index % 2) * NIBBLE_BYTES;
            codes[slot] = (value * inverse).round_ties_even() as i8;
        }
        for (pair, offset) in offsets.iter_mut().enumerate() {
            *offset = 8 * (i16::from(codes[pair * 2]) + i16::from(codes[pair * 2 + 1]));
        }
        self.scales[group] = peak / 127.0;
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_groups_avx2(&mut self, input: &[f32]) {
        use std::arch::x86_64::*;

        let groups = input.len() / GROUP_SIZE;
        assert!(self.codes.len() >= groups);
        assert!(self.offsets.len() >= groups);
        assert!(self.scales.len() >= groups);
        let absolute = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let natural = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let split = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3,
            5, 7, 9, 11, 13, 15,
        );
        let eight = _mm256_set1_epi8(8);
        for group in 0..groups {
            unsafe {
                let source = input.as_ptr().add(group * GROUP_SIZE);
                let values = [
                    _mm256_loadu_ps(source),
                    _mm256_loadu_ps(source.add(8)),
                    _mm256_loadu_ps(source.add(16)),
                    _mm256_loadu_ps(source.add(24)),
                ];
                let mut peaks = _mm256_setzero_ps();
                let mut invalid = _mm256_setzero_ps();
                for value in values {
                    peaks = _mm256_max_ps(peaks, _mm256_and_ps(value, absolute));
                    invalid = _mm256_or_ps(invalid, _mm256_sub_ps(value, value));
                }
                let mut stored = [0.0_f32; LANES];
                _mm256_storeu_ps(stored.as_mut_ptr(), peaks);
                let peak = stored.into_iter().fold(0.0_f32, f32::max);
                let invalid = _mm256_castps_si256(invalid);
                let invalid = u32::from(_mm256_testz_si256(invalid, invalid) == 0);
                let codes = self.codes.as_mut_ptr().add(group);
                let offsets = self.offsets.as_mut_ptr().add(group);
                let Some(inverse) = group_inverse(peak, invalid) else {
                    _mm256_store_si256(codes.cast::<__m256i>(), _mm256_setzero_si256());
                    _mm256_store_si256(offsets.cast::<__m256i>(), _mm256_setzero_si256());
                    *self.scales.get_unchecked_mut(group) =
                        if invalid != 0 { f32::NAN } else { 0.0 };
                    continue;
                };
                let inverse = _mm256_set1_ps(inverse);
                let rounded = values.map(|value| _mm256_cvtps_epi32(_mm256_mul_ps(value, inverse)));
                let low = _mm256_packs_epi32(rounded[0], rounded[1]);
                let high = _mm256_packs_epi32(rounded[2], rounded[3]);
                let bytes = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(low, high), natural);
                let arranged = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(bytes, split), 0xd8);
                _mm256_store_si256(codes.cast::<__m256i>(), arranged);
                _mm256_store_si256(
                    offsets.cast::<__m256i>(),
                    _mm256_maddubs_epi16(eight, arranged),
                );
                *self.scales.get_unchecked_mut(group) = peak / 127.0;
            }
        }
    }
}

fn group_inverse(peak: f32, invalid: u32) -> Option<f32> {
    if invalid != 0 || peak < MIN_PEAK {
        None
    } else {
        Some(127.0 / peak)
    }
}

pub(crate) struct Linear<'a> {
    pub weight: &'a Weight,
    pub bias: Option<&'a [f32]>,
    pub output: &'a mut [f32],
}

struct Plan<'a> {
    weight: &'a Weight,
    bias: Option<&'a [f32]>,
    output: SharedPtr<f32>,
    rows: usize,
    chunk: usize,
    first_task: usize,
}

pub(crate) fn linear_many(inputs: &[f32], activations: &mut [Activation], jobs: &mut [Linear<'_>]) {
    assert!(jobs.len() <= MAX_JOBS);
    let tokens = activations.len();
    assert!(tokens > 0 && inputs.len() % tokens == 0);
    let columns = inputs.len() / tokens;
    if jobs
        .iter()
        .any(|job| matches!(job.weight, Weight::Q4(_)) && !job.output.is_empty())
    {
        for (activation, input) in activations.iter_mut().zip(inputs.chunks_exact(columns)) {
            activation.quantize(input);
        }
    }
    let int8_block = jobs.iter().find_map(|job| match job.weight {
        Weight::Q8(matrix) if !job.output.is_empty() => Some(matrix.block()),
        _ => None,
    });
    if let Some(block) = int8_block {
        // 同一次调用中的 INT8 矩阵共用一份激活量化，块长必须相同。
        assert!(jobs.iter().all(|job| match job.weight {
            Weight::Q8(matrix) => matrix.block() == block,
            _ => true,
        }));
        for (activation, input) in activations.iter_mut().zip(inputs.chunks_exact(columns)) {
            activation.int8.quantize(input, block);
        }
    }
    let total_rows: usize = jobs.iter().map(|job| job.output.len() / tokens).sum();
    // 8 的倍数：INT8 矩阵按 8 行一组计算，任务从组的边界开始；其他格式逐行计算，不受影响。
    let chunk = total_rows
        .div_ceil(crate::pool::threads() * CHUNKS_PER_THREAD)
        .max(1)
        .next_multiple_of(8);
    let mut plans: [Option<Plan<'_>>; MAX_JOBS] = [None, None, None];
    let mut tasks = 0;
    let mut weights = 0_usize;
    for (plan, job) in plans.iter_mut().zip(jobs.iter_mut()) {
        assert_eq!(job.output.len() % tokens, 0);
        let rows = job.output.len() / tokens;
        match job.weight {
            Weight::F32(values) => assert_eq!(values.len(), rows * columns),
            Weight::Q4(matrix) => assert!(matrix.rows == rows && matrix.columns == columns),
            Weight::Q8(matrix) => assert!(matrix.rows == rows && matrix.columns == columns),
        }
        if let Some(bias) = job.bias {
            assert_eq!(bias.len(), rows);
        }
        *plan = Some(Plan {
            weight: job.weight,
            bias: job.bias,
            output: SharedPtr(job.output.as_mut_ptr()),
            rows,
            chunk,
            first_task: tasks,
        });
        tasks += rows.div_ceil(chunk);
        weights = weights.saturating_add(rows.saturating_mul(columns));
    }
    let activations = &*activations;
    let run = |task: usize| {
        let plan = plans
            .iter()
            .flatten()
            .rev()
            .find(|plan| plan.first_task <= task)
            .expect("task belongs to a plan");
        let start = (task - plan.first_task) * plan.chunk;
        let end = (start + plan.chunk).min(plan.rows);
        let output = unsafe { plan.output.get().add(start) };
        match plan.weight {
            Weight::F32(values) => unsafe {
                crate::ops::linear_rows(
                    values,
                    columns,
                    inputs,
                    start..end,
                    plan.bias,
                    output,
                    plan.rows,
                )
            },
            Weight::Q4(matrix) => unsafe {
                matrix.compute_rows(activations, start..end, plan.bias, output, plan.rows)
            },
            Weight::Q8(matrix) => unsafe {
                matrix.compute_rows(activations, start..end, plan.bias, output, plan.rows)
            },
        }
    };
    if weights < PARALLEL_WEIGHTS {
        for task in 0..tasks {
            run(task);
        }
    } else {
        parallel_for(tasks, &run);
    }
}

pub(crate) fn linear_into(weight: &Weight, input: &[f32], output: &mut [f32]) {
    linear_many(
        input,
        &mut [Activation::default()],
        &mut [Linear {
            weight,
            bias: None,
            output,
        }],
    );
}

#[cfg(test)]
pub(crate) fn linear(
    weight: &Weight,
    rows: usize,
    columns: usize,
    input: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), columns);
    let mut output = vec![0.0; rows];
    linear_many(
        input,
        &mut [Activation::default()],
        &mut [Linear {
            weight,
            bias,
            output: &mut output,
        }],
    );
    output
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn encode(values: &[f32], rows: usize, columns: usize) -> Vec<u8> {
        let groups = groups(columns);
        let mut scales = Vec::new();
        let mut codes = Vec::new();
        let mut nibbles = Vec::new();
        for row in 0..rows {
            let row_values = &values[row * columns..(row + 1) * columns];
            let steps: Vec<f32> = (0..groups)
                .map(|group| {
                    let end = ((group + 1) * GROUP_SIZE).min(columns);
                    row_values[group * GROUP_SIZE..end]
                        .iter()
                        .fold(0.0_f32, |peak, value| peak.max(value.abs()))
                        / 7.0
                })
                .collect();
            let scale = steps.iter().copied().fold(0.0_f32, f32::max) / 127.0;
            scales.extend_from_slice(&scale.to_le_bytes());
            for (group, step) in steps.iter().enumerate() {
                let magnitude = if *step > 0.0 {
                    (step / scale).ceil().clamp(1.0, 127.0) as i8
                } else {
                    0
                };
                let code = if group % 2 == 1 {
                    -magnitude
                } else {
                    magnitude
                };
                codes.push(code as u8);
                let step = scale * f32::from(code);
                let mut packed = [0x88_u8; NIBBLE_BYTES];
                let end = ((group + 1) * GROUP_SIZE).min(columns);
                for (index, value) in row_values[group * GROUP_SIZE..end].iter().enumerate() {
                    let quantized = if step != 0.0 {
                        (value / step).round_ties_even().clamp(-8.0, 7.0) as i8
                    } else {
                        0
                    };
                    let nibble = (quantized + 8) as u8;
                    if index % 2 == 0 {
                        packed[index / 2] = (packed[index / 2] & 0xf0) | nibble;
                    } else {
                        packed[index / 2] = (packed[index / 2] & 0x0f) | (nibble << 4);
                    }
                }
                nibbles.extend_from_slice(&packed);
            }
        }
        let mut bytes = scales;
        bytes.extend_from_slice(&codes);
        bytes.extend_from_slice(&nibbles);
        bytes.resize(matrix_bytes(rows, columns).unwrap(), 0);
        bytes
    }

    pub(crate) fn pseudo_random(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1_u32 << 23) as f32 - 1.0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{encode, pseudo_random};
    use super::*;

    fn matrix(rows: usize, columns: usize, seed: u64) -> (Vec<f32>, Q4Matrix) {
        let values = pseudo_random(rows * columns, seed);
        let matrix = Q4Matrix::new(rows, columns, encode(&values, rows, columns)).unwrap();
        (values, matrix)
    }

    #[test]
    fn layout_places_scale_code_and_nibbles() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.25_f32.to_le_bytes());
        bytes.push(2);
        bytes.push(0xa9);
        bytes.extend_from_slice(&[0x88; 15]);
        bytes.extend_from_slice(&[0; 3]);
        let matrix = Q4Matrix::new(1, 32, bytes).unwrap();
        let mut row = vec![0.0; 32];
        matrix.dequantize_row(0, &mut row);
        assert_eq!(&row[..3], &[0.5, 1.0, 0.0]);
        let mut input = vec![0.0; 32];
        input[0] = 2.0;
        input[1] = 3.0;
        let output = linear(&Weight::Q4(matrix), 1, 32, &input, None);
        assert!((output[0] - 4.0).abs() < 0.05, "{}", output[0]);
    }

    #[test]
    fn negative_group_codes_flip_the_group() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.25_f32.to_le_bytes());
        bytes.push((-2_i8) as u8);
        bytes.push(0xa9);
        bytes.extend_from_slice(&[0x88; 15]);
        bytes.extend_from_slice(&[0; 3]);
        let matrix = Q4Matrix::new(1, 32, bytes).unwrap();
        let mut row = vec![0.0; 32];
        matrix.dequantize_row(0, &mut row);
        assert_eq!(&row[..3], &[-0.5, -1.0, 0.0]);
    }

    #[test]
    fn dequantized_rows_stay_within_half_a_step() {
        let (values, matrix) = matrix(3, 70, 7);
        let restored = Weight::Q4(matrix.clone()).into_f32();
        for row in 0..3 {
            let scale = matrix.row_scale(row);
            for column in 0..70 {
                let step = scale * group_code(matrix.row_codes(row)[column / GROUP_SIZE]).abs();
                let index = row * 70 + column;
                assert!(
                    (restored[index] - values[index]).abs() <= step * 0.5001,
                    "row={row} column={column}"
                );
            }
        }
    }

    #[test]
    fn quantized_linear_tracks_dequantized_reference() {
        for (rows, columns) in [(5, 32), (4, 70), (9, 257), (3, 1024)] {
            let (_, matrix) = matrix(rows, columns, 11 + columns as u64);
            let input = pseudo_random(columns, 99);
            let bias = pseudo_random(rows, 5);
            let restored = Weight::Q4(matrix.clone()).into_f32();
            let expected = crate::ops::linear(&restored, rows, columns, &input, Some(&bias));
            let actual = linear(&Weight::Q4(matrix), rows, columns, &input, Some(&bias));
            let norm = restored
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < 0.02 * norm.max(1.0),
                    "{rows}x{columns}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn vectorized_paths_match_scalar_bit_for_bit() {
        for (rows, columns) in [(6, 32), (5, 70), (7, 513), (4, 1024)] {
            let (_, matrix) = matrix(rows, columns, 3 + rows as u64);
            let mut input = pseudo_random(columns, 17);
            input[columns / 2] = 250.0;
            input[0] = -0.0;
            let mut reference = Activation::default();
            reference.reset(groups(columns));
            for group in 0..groups(columns) {
                let end = ((group + 1) * GROUP_SIZE).min(columns);
                reference.quantize_group(group, &input[group * GROUP_SIZE..end]);
            }
            let mut activation = Activation::default();
            activation.quantize(&input);
            assert_eq!(activation.codes, reference.codes);
            assert_eq!(activation.offsets, reference.offsets);
            assert_eq!(
                activation
                    .scales
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .scales
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            let bias = pseudo_random(rows, 23);
            let scalar: Vec<f32> = (0..rows)
                .map(|row| matrix.row_scalar(&reference, row, Some(&bias)))
                .collect();
            let mut dispatched = vec![0.0; rows];
            matrix.compute_rows_into(&activation, 0..rows, Some(&bias), &mut dispatched);
            assert_eq!(
                scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                dispatched.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn parallel_dispatch_matches_serial_rows() {
        let (rows, columns) = (300, 1024);
        let (_, matrix) = matrix(rows, columns, 41);
        let input = pseudo_random(columns, 43);
        let mut activation = Activation::default();
        activation.quantize(&input);
        let mut serial = vec![0.0; rows];
        matrix.compute_rows_into(&activation, 0..rows, None, &mut serial);
        let weight = Weight::Q4(matrix);
        let mut first = vec![0.0; rows];
        let mut second = vec![0.0; rows];
        linear_many(
            &input,
            std::slice::from_mut(&mut activation),
            &mut [
                Linear {
                    weight: &weight,
                    bias: None,
                    output: &mut first,
                },
                Linear {
                    weight: &weight,
                    bias: None,
                    output: &mut second,
                },
            ],
        );
        assert_eq!(first, serial);
        assert_eq!(second, serial);
    }

    #[test]
    fn batched_tokens_match_single_token_bit_for_bit() {
        for (rows, columns, tokens) in [(300, 1024, 7), (9, 70, 5), (40, 96, 4), (300, 1024, 1)] {
            let (_, matrix) = matrix(rows, columns, 50 + tokens as u64);
            let weight = Weight::Q4(matrix);
            let bias = pseudo_random(rows, 3);
            let inputs = pseudo_random(columns * tokens, 61);
            let mut batched = vec![0.0; rows * tokens];
            let mut activations: Vec<Activation> =
                (0..tokens).map(|_| Activation::default()).collect();
            linear_many(
                &inputs,
                &mut activations,
                &mut [Linear {
                    weight: &weight,
                    bias: Some(&bias),
                    output: &mut batched,
                }],
            );
            for (token, input) in inputs.chunks_exact(columns).enumerate() {
                let single = linear(&weight, rows, columns, input, Some(&bias));
                assert_eq!(
                    single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    batched[token * rows..(token + 1) * rows]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "{rows}x{columns} token {token}"
                );
            }
        }
    }

    #[test]
    fn zero_and_non_finite_inputs_are_handled() {
        let (_, matrix) = matrix(2, 64, 5);
        let weight = Weight::Q4(matrix);
        assert_eq!(linear(&weight, 2, 64, &[0.0; 64], None), vec![0.0, 0.0]);
        let mut input = vec![1.0; 64];
        input[40] = f32::NAN;
        assert!(
            linear(&weight, 2, 64, &input, None)
                .iter()
                .all(|v| v.is_nan())
        );
        input[40] = f32::INFINITY;
        assert!(
            linear(&weight, 2, 64, &input, None)
                .iter()
                .all(|v| v.is_nan())
        );
    }

    #[test]
    fn rejects_invalid_scale_and_size() {
        let mut bytes = vec![0x88; matrix_bytes(1, 32).unwrap()];
        bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Q4Matrix::new(1, 32, bytes.clone()).is_err());
        bytes[..4].copy_from_slice(&(-1.0_f32).to_le_bytes());
        assert!(Q4Matrix::new(1, 32, bytes.clone()).is_err());
        bytes[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        assert!(Q4Matrix::new(1, 32, bytes.clone()).is_ok());
        bytes.push(0);
        assert!(Q4Matrix::new(1, 32, bytes).is_err());
        assert_eq!(matrix_bytes(2, 33), Some(2 * (4 + 2 * 17)));
        assert_eq!(matrix_bytes(usize::MAX, 64), None);
    }
}
