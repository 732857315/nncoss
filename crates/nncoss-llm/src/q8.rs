//! INT8 分块权重：每行按 32 或 64 个元素分块，每块一个 f32 还原系数，元素为 [-127, 127] 的 int8，
//! 还原值 = 码 × 还原系数。64 元素一块与 ncnn 的 weight block quant（8 bit、block_size 64）相同，
//! 32 元素一块与 GGUF 的 Q8_0 相同（Q8_0 的 f16 系数无损展开成 f32）。
//! 激活按同样的块长量化为 int8（absmax / 127，四舍五入远离零），与 ncnn 和 llama.cpp 的做法一致。

use std::ops::Range;

use crate::q4::Activation;
use crate::{Error, Result};

/// 一个 AVX2 寄存器的字节数；每块由 1 个（32 元素）或 2 个（64 元素）这样的分段组成。
const CHUNK: usize = 32;
const SCALE_BYTES: usize = 4;
const LANES: usize = 8;
const MIN_PEAK: f32 = 1.0e-30;
#[cfg(target_arch = "x86_64")]
const PREFETCH_BYTES: usize = 4096;

pub(crate) const fn supported_block(block: usize) -> bool {
    block == 32 || block == 64
}

pub(crate) const fn blocks(columns: usize, block: usize) -> usize {
    columns.div_ceil(block)
}

/// 字节布局：先是 rows × blocks 个 f32 还原系数（小端），再是 rows × blocks × block 个 int8 码；
/// 每行补齐到整块，补齐部分为零。
pub(crate) fn matrix_bytes(rows: usize, columns: usize, block: usize) -> Option<usize> {
    blocks(columns, block)
        .checked_mul(SCALE_BYTES + block)?
        .checked_mul(rows)
}

#[derive(Debug, Clone)]
pub(crate) struct Q8Matrix {
    pub rows: usize,
    pub columns: usize,
    block: usize,
    blocks: usize,
    bytes: Vec<u8>,
}

impl Q8Matrix {
    pub fn new(rows: usize, columns: usize, block: usize, bytes: Vec<u8>) -> Result<Self> {
        if !supported_block(block) {
            return Err(Error::Unsupported("INT8 块长"));
        }
        let expected =
            matrix_bytes(rows, columns, block).ok_or(Error::InvalidModel("INT8 张量尺寸溢出"))?;
        if bytes.len() != expected {
            return Err(Error::InvalidModel("INT8 张量尺寸不匹配"));
        }
        let matrix = Self {
            rows,
            columns,
            block,
            blocks: blocks(columns, block),
            bytes,
        };
        for index in 0..rows * matrix.blocks {
            if !matrix.descale_at(index).is_finite() {
                return Err(Error::InvalidModel("INT8 还原系数无效"));
            }
        }
        // -128 会让符号技巧的点积溢出；ncnn 与 llama.cpp 的量化也只产生 [-127, 127]。
        if matrix.codes().contains(&0x80) {
            return Err(Error::InvalidModel("INT8 权重码超出 [-127, 127]"));
        }
        for row in 0..rows {
            if matrix.row_codes(row)[columns..]
                .iter()
                .any(|code| *code != 0)
            {
                return Err(Error::InvalidModel("INT8 行末补齐必须为零"));
            }
        }
        Ok(matrix)
    }

    pub(crate) fn block(&self) -> usize {
        self.block
    }

    fn descale_at(&self, index: usize) -> f32 {
        let offset = index * SCALE_BYTES;
        f32::from_le_bytes(
            self.bytes[offset..offset + SCALE_BYTES]
                .try_into()
                .expect("validated INT8 descale"),
        )
    }

    pub(crate) fn descale(&self, row: usize, block: usize) -> f32 {
        self.descale_at(row * self.blocks + block)
    }

    fn codes(&self) -> &[u8] {
        &self.bytes[self.rows * self.blocks * SCALE_BYTES..]
    }

    pub(crate) fn row_codes(&self, row: usize) -> &[i8] {
        let width = self.blocks * self.block;
        let bytes = &self.codes()[row * width..(row + 1) * width];
        // u8 与 i8 的大小和对齐相同。
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) }
    }

    /// 每一块都是还原系数为零或码全为零时，整个矩阵的还原值都是零。
    pub(crate) fn is_zero(&self) -> bool {
        (0..self.rows).all(|row| {
            let codes = self.row_codes(row);
            (0..self.blocks).all(|block| {
                self.descale(row, block) == 0.0
                    || codes[block * self.block..(block + 1) * self.block]
                        .iter()
                        .all(|code| *code == 0)
            })
        })
    }

    pub fn dequantize_row(&self, row: usize, output: &mut [f32]) {
        debug_assert_eq!(output.len(), self.columns);
        let codes = self.row_codes(row);
        for (index, value) in output.iter_mut().enumerate() {
            *value = f32::from(codes[index]) * self.descale(row, index / self.block);
        }
    }

    pub(crate) unsafe fn compute_rows(
        &self,
        activations: &[Activation],
        rows: Range<usize>,
        bias: Option<&[f32]>,
        output: *mut f32,
        stride: usize,
    ) {
        assert!(rows.end <= self.rows);
        for activation in activations {
            assert_eq!(activation.int8.block, self.block);
            assert!(activation.int8.descales.len() >= self.blocks);
            assert!(activation.int8.chunks.len() >= self.blocks * self.block / CHUNK);
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { self.compute_rows_avx2(activations, rows, bias, output, stride) };
            return;
        }
        for (offset, row) in rows.enumerate() {
            for (token, activation) in activations.iter().enumerate() {
                let value = self.row_scalar(&activation.int8, row, bias);
                unsafe { output.add(token * stride + offset).write(value) };
            }
        }
    }

    /// 标量参考：8 条 lane 的划分与 AVX2 路径相同（lane i 是每 32 字节分段中的第 4i..4i+3 个），
    /// 每块的整数点积精确，浮点累加顺序一致，所以两条路径逐位相同。
    pub(crate) fn row_scalar(
        &self,
        activation: &Activation8,
        row: usize,
        bias: Option<&[f32]>,
    ) -> f32 {
        let codes = self.row_codes(row);
        let chunks = self.block / CHUNK;
        let mut lanes = [0.0_f32; LANES];
        for block in 0..self.blocks {
            let scale = self.descale(row, block) * activation.descales[block];
            for (lane, total) in lanes.iter_mut().enumerate() {
                let mut sum = 0_i32;
                for chunk in block * chunks..(block + 1) * chunks {
                    let weights = &codes[chunk * CHUNK..(chunk + 1) * CHUNK];
                    let inputs = &activation.chunks[chunk].0;
                    for slot in lane * 4..lane * 4 + 4 {
                        sum += i32::from(weights[slot]) * i32::from(inputs[slot]);
                    }
                }
                *total += sum as f32 * scale;
            }
        }
        finish_row(sum_lanes(lanes), bias, row)
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

        // 与 Q4 相同：外层每组最多 4 个 token，内层逐行，每行权重读一次供这组 token 共用。
        let row_bytes = self.blocks * self.block;
        let mut first = 0;
        while first < activations.len() {
            let tile = &activations[first..];
            let width = tile.len().min(4);
            for (offset, row) in rows.clone().enumerate() {
                let codes = self.row_codes(row).as_ptr();
                for line in 0..row_bytes.div_ceil(64) {
                    _mm_prefetch::<_MM_HINT_T0>(
                        codes.wrapping_add(PREFETCH_BYTES + line * 64).cast(),
                    );
                }
                let target = unsafe { output.add(first * stride + offset) };
                unsafe {
                    match (width, self.block) {
                        (1, 64) => self.row_tile_avx2::<1, 2>(tile, row, bias, target, stride),
                        (2, 64) => self.row_tile_avx2::<2, 2>(tile, row, bias, target, stride),
                        (3, 64) => self.row_tile_avx2::<3, 2>(tile, row, bias, target, stride),
                        (_, 64) => self.row_tile_avx2::<4, 2>(tile, row, bias, target, stride),
                        (1, _) => self.row_tile_avx2::<1, 1>(tile, row, bias, target, stride),
                        (2, _) => self.row_tile_avx2::<2, 1>(tile, row, bias, target, stride),
                        (3, _) => self.row_tile_avx2::<3, 1>(tile, row, bias, target, stride),
                        (_, _) => self.row_tile_avx2::<4, 1>(tile, row, bias, target, stride),
                    }
                }
            }
            first += width;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn row_tile_avx2<const TOKENS: usize, const CHUNKS: usize>(
        &self,
        activations: &[Activation],
        row: usize,
        bias: Option<&[f32]>,
        output: *mut f32,
        stride: usize,
    ) {
        use std::arch::x86_64::*;

        let codes = self.row_codes(row).as_ptr().cast::<__m256i>();
        let inputs: [*const __m256i; TOKENS] =
            std::array::from_fn(|token| activations[token].int8.chunks.as_ptr().cast());
        let descales: [*const f32; TOKENS] =
            std::array::from_fn(|token| activations[token].int8.descales.as_ptr());
        let ones = _mm256_set1_epi16(1);
        let mut lanes = [_mm256_setzero_ps(); TOKENS];
        for block in 0..self.blocks {
            unsafe {
                let weights: [__m256i; CHUNKS] = std::array::from_fn(|chunk| {
                    _mm256_loadu_si256(codes.add(block * CHUNKS + chunk))
                });
                let row_descale = self.descale(row, block);
                for token in 0..TOKENS {
                    let mut sums = _mm256_setzero_si256();
                    for (chunk, weight) in weights.iter().enumerate() {
                        let input = _mm256_load_si256(inputs[token].add(block * CHUNKS + chunk));
                        // |a| × (w 带上 a 的符号) = a × w；|a|、|w| ≤ 127，相邻两项之和不会使 i16 饱和。
                        let products = _mm256_maddubs_epi16(
                            _mm256_sign_epi8(input, input),
                            _mm256_sign_epi8(*weight, input),
                        );
                        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(products, ones));
                    }
                    let scale = row_descale * *descales[token].add(block);
                    lanes[token] = _mm256_add_ps(
                        lanes[token],
                        _mm256_mul_ps(_mm256_cvtepi32_ps(sums), _mm256_set1_ps(scale)),
                    );
                }
            }
        }
        for (token, lanes) in lanes.iter().enumerate() {
            let mut stored = [0.0_f32; LANES];
            unsafe {
                _mm256_storeu_ps(stored.as_mut_ptr(), *lanes);
                output
                    .add(token * stride)
                    .write(finish_row(sum_lanes(stored), bias, row));
            }
        }
    }
}

fn sum_lanes(lanes: [f32; LANES]) -> f32 {
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
}

fn finish_row(sum: f32, bias: Option<&[f32]>, row: usize) -> f32 {
    match bias {
        Some(bias) => sum + bias[row],
        None => sum,
    }
}

/// 一个 token 的 INT8 激活：按块长分块，码按原顺序存放在 32 字节对齐的分段中，补齐部分为零。
#[derive(Debug, Default)]
pub(crate) struct Activation8 {
    block: usize,
    chunks: Vec<Chunk>,
    descales: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(32))]
struct Chunk([i8; CHUNK]);

impl Activation8 {
    pub fn quantize(&mut self, input: &[f32], block: usize) {
        assert!(supported_block(block));
        let blocks = blocks(input.len(), block);
        self.block = block;
        self.chunks.clear();
        self.chunks
            .resize(blocks * block / CHUNK, Chunk([0; CHUNK]));
        self.descales.clear();
        self.descales.resize(blocks, 0.0);
        for index in 0..blocks {
            let end = ((index + 1) * block).min(input.len());
            self.quantize_block(index, &input[index * block..end]);
        }
    }

    fn quantize_block(&mut self, index: usize, values: &[f32]) {
        let mut peak = 0.0_f32;
        let mut invalid = false;
        for value in values {
            peak = peak.max(value.abs());
            invalid |= !value.is_finite();
        }
        if invalid {
            // 输出随之成为 NaN，与 f32 路径遇到非有限输入时一致。
            self.descales[index] = f32::NAN;
            return;
        }
        if peak < MIN_PEAK {
            return;
        }
        let scale = 127.0 / peak;
        let first = index * self.block / CHUNK;
        for (offset, value) in values.iter().enumerate() {
            let code = (value * scale).round().clamp(-127.0, 127.0) as i8;
            self.chunks[first + offset / CHUNK].0[offset % CHUNK] = code;
        }
        self.descales[index] = peak / 127.0;
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// 与 ncnn 的 minmax 分块量化和 llama.cpp 的 Q8_0 相同：每块 absmax / 127 为还原系数，
    /// 码四舍五入并限制在 ±127。
    pub(crate) fn encode(values: &[f32], rows: usize, columns: usize, block: usize) -> Vec<u8> {
        let blocks = blocks(columns, block);
        let mut descales = Vec::with_capacity(rows * blocks * SCALE_BYTES);
        let mut codes = vec![0_u8; rows * blocks * block];
        for row in 0..rows {
            let row_values = &values[row * columns..(row + 1) * columns];
            for index in 0..blocks {
                let end = ((index + 1) * block).min(columns);
                let part = &row_values[index * block..end];
                let peak = part
                    .iter()
                    .fold(0.0_f32, |peak, value| peak.max(value.abs()));
                let descale = peak / 127.0;
                descales.extend_from_slice(&descale.to_le_bytes());
                if peak > 0.0 {
                    for (offset, value) in part.iter().enumerate() {
                        let code = (value / descale).round().clamp(-127.0, 127.0) as i8;
                        codes[(row * blocks + index) * block + offset] = code as u8;
                    }
                }
            }
        }
        descales.extend_from_slice(&codes);
        descales
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::encode;
    use super::*;
    use crate::q4::test_support::pseudo_random;
    use crate::q4::{Linear, Weight, linear, linear_many};

    const BLOCKS: [usize; 2] = [32, 64];

    fn matrix(rows: usize, columns: usize, block: usize, seed: u64) -> (Vec<f32>, Q8Matrix) {
        let values = pseudo_random(rows * columns, seed);
        let bytes = encode(&values, rows, columns, block);
        (values, Q8Matrix::new(rows, columns, block, bytes).unwrap())
    }

    fn activation(input: &[f32], block: usize) -> Activation {
        let mut activation = Activation::default();
        activation.int8.quantize(input, block);
        activation
    }

    #[test]
    fn layout_places_descales_before_codes() {
        for block in BLOCKS {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&0.5_f32.to_le_bytes());
            let mut codes = vec![0_u8; block];
            codes[0] = 2;
            codes[1] = (-3_i8) as u8;
            bytes.extend_from_slice(&codes);
            let matrix = Q8Matrix::new(1, 3, block, bytes).unwrap();
            let mut row = vec![0.0; 3];
            matrix.dequantize_row(0, &mut row);
            assert_eq!(row, vec![1.0, -1.5, 0.0]);
            let output = linear(&Weight::Q8(matrix), 1, 3, &[2.0, 4.0, 1.0], None);
            assert!((output[0] - (2.0 - 6.0)).abs() < 0.05, "{}", output[0]);
        }
    }

    #[test]
    fn dequantized_rows_stay_within_half_a_step() {
        for block in BLOCKS {
            let (values, matrix) = matrix(3, 150, block, 7);
            let restored = Weight::Q8(matrix.clone()).into_f32();
            for row in 0..3 {
                for column in 0..150 {
                    let step = matrix.descale(row, column / block);
                    let index = row * 150 + column;
                    assert!(
                        (restored[index] - values[index]).abs() <= step * 0.5001,
                        "block={block} row={row} column={column}"
                    );
                }
            }
        }
    }

    #[test]
    fn quantized_linear_tracks_dequantized_reference() {
        for block in BLOCKS {
            for (rows, columns) in [(5, 64), (4, 70), (9, 257), (3, 1024)] {
                let (_, matrix) = matrix(rows, columns, block, 11 + columns as u64);
                let input = pseudo_random(columns, 99);
                let bias = pseudo_random(rows, 5);
                let restored = Weight::Q8(matrix.clone()).into_f32();
                let expected = crate::ops::linear(&restored, rows, columns, &input, Some(&bias));
                let actual = linear(&Weight::Q8(matrix), rows, columns, &input, Some(&bias));
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        (actual - expected).abs() < 0.01 * (columns as f32).sqrt(),
                        "block={block} {rows}x{columns}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn activation_matches_ncnn_rounding() {
        for block in BLOCKS {
            let mut input = vec![0.0_f32; block];
            input[0] = 2.54;
            input[1] = -1.0;
            input[2] = 0.01;
            let mut activation = Activation8::default();
            activation.quantize(&input, block);
            let scale = 127.0_f32 / 2.54;
            assert_eq!(activation.chunks[0].0[0], 127);
            assert_eq!(activation.chunks[0].0[1], (-scale).round() as i8);
            assert_eq!(activation.chunks[0].0[2], (0.01 * scale).round() as i8);
            assert_eq!(activation.descales[0], 2.54 / 127.0);
        }
    }

    #[test]
    fn vectorized_paths_match_scalar_bit_for_bit() {
        for block in BLOCKS {
            for (rows, columns) in [(6, 64), (5, 70), (7, 513), (4, 1024)] {
                let (_, matrix) = matrix(rows, columns, block, 3 + rows as u64);
                let mut input = pseudo_random(columns, 17);
                input[columns / 2] = 250.0;
                input[0] = -0.0;
                let activation = activation(&input, block);
                let bias = pseudo_random(rows, 23);
                let scalar: Vec<f32> = (0..rows)
                    .map(|row| matrix.row_scalar(&activation.int8, row, Some(&bias)))
                    .collect();
                let mut dispatched = vec![0.0; rows];
                unsafe {
                    matrix.compute_rows(
                        std::slice::from_ref(&activation),
                        0..rows,
                        Some(&bias),
                        dispatched.as_mut_ptr(),
                        0,
                    )
                };
                assert_eq!(
                    scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    dispatched.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "block={block} {rows}x{columns}"
                );
            }
        }
    }

    #[test]
    fn batched_tokens_match_single_token_bit_for_bit() {
        for block in BLOCKS {
            for (rows, columns, tokens) in
                [(300, 1024, 7), (9, 70, 5), (40, 128, 4), (300, 1024, 1)]
            {
                let (_, matrix) = matrix(rows, columns, block, 50 + tokens as u64);
                let weight = Weight::Q8(matrix);
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
                        "block={block} {rows}x{columns} token {token}"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_and_non_finite_inputs_are_handled() {
        for block in BLOCKS {
            let (_, matrix) = matrix(2, 128, block, 5);
            let weight = Weight::Q8(matrix);
            assert_eq!(linear(&weight, 2, 128, &[0.0; 128], None), vec![0.0, 0.0]);
            let mut input = vec![1.0; 128];
            input[40] = f32::NAN;
            assert!(
                linear(&weight, 2, 128, &input, None)
                    .iter()
                    .all(|v| v.is_nan())
            );
            input[40] = f32::INFINITY;
            assert!(
                linear(&weight, 2, 128, &input, None)
                    .iter()
                    .all(|v| v.is_nan())
            );
        }
    }

    #[test]
    fn zero_detection_covers_zero_descales_and_zero_codes() {
        for block in BLOCKS {
            let zero = Q8Matrix::new(2, 70, block, encode(&[0.0; 140], 2, 70, block)).unwrap();
            assert!(Weight::Q8(zero).is_zero());
            let (_, nonzero) = matrix(2, 70, block, 9);
            assert!(!Weight::Q8(nonzero).is_zero());
        }
    }

    #[test]
    fn rejects_invalid_descale_code_padding_size_and_block() {
        for block in BLOCKS {
            let valid = encode(&pseudo_random(70, 3), 1, 70, block);
            assert!(Q8Matrix::new(1, 70, block, valid.clone()).is_ok());
            for descale in [f32::NAN, f32::INFINITY] {
                let mut bytes = valid.clone();
                bytes[..4].copy_from_slice(&descale.to_le_bytes());
                assert!(Q8Matrix::new(1, 70, block, bytes).is_err());
            }
            let codes = blocks(70, block) * SCALE_BYTES;
            let mut bytes = valid.clone();
            bytes[codes] = 0x80;
            assert!(Q8Matrix::new(1, 70, block, bytes).is_err());
            let mut bytes = valid.clone();
            bytes[codes + 80] = 1;
            assert!(Q8Matrix::new(1, 70, block, bytes).is_err());
            let mut bytes = valid;
            bytes.push(0);
            assert!(Q8Matrix::new(1, 70, block, bytes).is_err());
        }
        assert_eq!(
            Q8Matrix::new(1, 16, 16, vec![0; 20]).unwrap_err(),
            Error::Unsupported("INT8 块长")
        );
        assert_eq!(matrix_bytes(2, 65, 64), Some(2 * 2 * (4 + 64)));
        assert_eq!(matrix_bytes(2, 65, 32), Some(2 * 3 * (4 + 32)));
        assert_eq!(matrix_bytes(usize::MAX, 64, 64), None);
    }
}
