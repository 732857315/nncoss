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

/// 每 8 行交错成一个面板，按 k 每 4 个一步：一步 32 字节 = 8 行 × 4 个码。AVX2 内核一次读一步，
/// 8 个 lane 分别累加 8 行，块末的缩放与累加也一次处理 8 行。
const PANEL_ROWS: usize = 8;
const STEP: usize = 4;

#[derive(Debug, Clone)]
pub(crate) struct Q8Matrix {
    pub rows: usize,
    pub columns: usize,
    block: usize,
    blocks: usize,
    /// 每行按块补齐后的列数（blocks × block）。
    width: usize,
    /// groups × (width / 4) 步，每步 32 字节；补齐的行与列为零。
    panels: Vec<i8>,
    /// groups × blocks × 8 个还原系数；补齐的行为零。
    descales: Vec<f32>,
}

impl Q8Matrix {
    /// `bytes` 为文件中的行主序布局（见 matrix_bytes），载入时改排成面板。
    pub fn new(rows: usize, columns: usize, block: usize, bytes: Vec<u8>) -> Result<Self> {
        if !supported_block(block) {
            return Err(Error::Unsupported("INT8 块长"));
        }
        let expected =
            matrix_bytes(rows, columns, block).ok_or(Error::InvalidModel("INT8 张量尺寸溢出"))?;
        if bytes.len() != expected {
            return Err(Error::InvalidModel("INT8 张量尺寸不匹配"));
        }
        let blocks = blocks(columns, block);
        let width = blocks * block;
        let scales = &bytes[..rows * blocks * SCALE_BYTES];
        let codes = &bytes[rows * blocks * SCALE_BYTES..];
        // -128 会让符号技巧的点积溢出；ncnn 与 llama.cpp 的量化也只产生 [-127, 127]。
        if codes.contains(&0x80) {
            return Err(Error::InvalidModel("INT8 权重码超出 [-127, 127]"));
        }
        for row in 0..rows {
            if codes[row * width + columns..(row + 1) * width]
                .iter()
                .any(|code| *code != 0)
            {
                return Err(Error::InvalidModel("INT8 行末补齐必须为零"));
            }
        }
        let groups = rows.div_ceil(PANEL_ROWS);
        let steps = width / STEP;
        let mut panels = Vec::new();
        panels
            .try_reserve_exact(groups * steps * PANEL_ROWS * STEP)
            .map_err(|_| Error::Allocation)?;
        panels.resize(groups * steps * PANEL_ROWS * STEP, 0_i8);
        let mut descales = vec![0.0_f32; groups * blocks * PANEL_ROWS];
        for row in 0..rows {
            let (group, lane) = (row / PANEL_ROWS, row % PANEL_ROWS);
            for index in 0..blocks {
                let offset = (row * blocks + index) * SCALE_BYTES;
                let descale = f32::from_le_bytes(
                    scales[offset..offset + SCALE_BYTES]
                        .try_into()
                        .expect("validated INT8 descale"),
                );
                if !descale.is_finite() {
                    return Err(Error::InvalidModel("INT8 还原系数无效"));
                }
                descales[(group * blocks + index) * PANEL_ROWS + lane] = descale;
            }
            let source = &codes[row * width..(row + 1) * width];
            for step in 0..steps {
                let target = ((group * steps + step) * PANEL_ROWS + lane) * STEP;
                for slot in 0..STEP {
                    panels[target + slot] = source[step * STEP + slot] as i8;
                }
            }
        }
        Ok(Self {
            rows,
            columns,
            block,
            blocks,
            width,
            panels,
            descales,
        })
    }

    pub(crate) fn block(&self) -> usize {
        self.block
    }

    pub(crate) fn descale(&self, row: usize, block: usize) -> f32 {
        self.descales[((row / PANEL_ROWS) * self.blocks + block) * PANEL_ROWS + row % PANEL_ROWS]
    }

    /// 第 `row` 行第 `step` 步的 4 个码。
    fn step_codes(&self, row: usize, step: usize) -> &[i8] {
        let steps = self.width / STEP;
        let start = (((row / PANEL_ROWS) * steps + step) * PANEL_ROWS + row % PANEL_ROWS) * STEP;
        &self.panels[start..start + STEP]
    }

    /// 每一块都是还原系数为零或码全为零时，整个矩阵的还原值都是零。
    pub(crate) fn is_zero(&self) -> bool {
        let per_block = self.block / STEP;
        (0..self.rows).all(|row| {
            (0..self.blocks).all(|index| {
                self.descale(row, index) == 0.0
                    || (index * per_block..(index + 1) * per_block)
                        .all(|step| self.step_codes(row, step).iter().all(|code| *code == 0))
            })
        })
    }

    pub fn dequantize_row(&self, row: usize, output: &mut [f32]) {
        debug_assert_eq!(output.len(), self.columns);
        for (index, value) in output.iter_mut().enumerate() {
            let code = self.step_codes(row, index / STEP)[index % STEP];
            *value = f32::from(code) * self.descale(row, index / self.block);
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
            assert!(activation.int8.chunks.len() * CHUNK >= self.width);
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

    /// 标量参考：每块的整数点积是精确的（|点积| ≤ 64 × 127 × 127 < 2^24，转成 f32 也精确），
    /// 再按块的顺序累加 `点积 × (权重系数 × 激活系数)`。AVX2 内核对每一行做同样的运算，
    /// 与分组和 token 分块无关，所以两条路径逐位相同。
    pub(crate) fn row_scalar(
        &self,
        activation: &Activation8,
        row: usize,
        bias: Option<&[f32]>,
    ) -> f32 {
        let per_block = self.block / STEP;
        let mut total = 0.0_f32;
        for index in 0..self.blocks {
            let mut dot = 0_i32;
            for step in index * per_block..(index + 1) * per_block {
                let weights = self.step_codes(row, step);
                for (slot, weight) in weights.iter().enumerate() {
                    let k = step * STEP + slot;
                    let input = activation.chunks[k / CHUNK].0[k % CHUNK];
                    dot += i32::from(*weight) * i32::from(input);
                }
            }
            total += dot as f32 * (self.descale(row, index) * activation.descales[index]);
        }
        finish_row(total, bias, row)
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
        // 外层每组最多 4 个 token，内层逐个 8 行面板：一步权重读一次，供这组 token 共用。
        let mut first = 0;
        while first < activations.len() {
            let width = (activations.len() - first).min(4);
            let tile = &activations[first..first + width];
            for group in rows.start / PANEL_ROWS..rows.end.div_ceil(PANEL_ROWS) {
                let mut lanes = [[0.0_f32; PANEL_ROWS]; 4];
                unsafe {
                    match width {
                        1 => self.group_avx2::<1>(tile, group, &mut lanes),
                        2 => self.group_avx2::<2>(tile, group, &mut lanes),
                        3 => self.group_avx2::<3>(tile, group, &mut lanes),
                        _ => self.group_avx2::<4>(tile, group, &mut lanes),
                    }
                }
                for (lane, row) in (group * PANEL_ROWS..(group + 1) * PANEL_ROWS).enumerate() {
                    if row < rows.start || row >= rows.end {
                        continue;
                    }
                    for (token, values) in lanes.iter().enumerate().take(width) {
                        let value = finish_row(values[lane], bias, row);
                        unsafe {
                            output
                                .add((first + token) * stride + row - rows.start)
                                .write(value)
                        };
                    }
                }
            }
            first += width;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn group_avx2<const TOKENS: usize>(
        &self,
        activations: &[Activation],
        group: usize,
        lanes: &mut [[f32; PANEL_ROWS]; 4],
    ) {
        use std::arch::x86_64::*;

        let steps = self.width / STEP;
        let per_block = self.block / STEP;
        let panel = unsafe { self.panels.as_ptr().add(group * steps * PANEL_ROWS * STEP) };
        let descales = unsafe { self.descales.as_ptr().add(group * self.blocks * PANEL_ROWS) };
        let codes: [*const i8; TOKENS] =
            std::array::from_fn(|token| activations[token].int8.chunks.as_ptr().cast());
        let magnitudes: [*const i8; TOKENS] =
            std::array::from_fn(|token| activations[token].int8.magnitudes.as_ptr().cast());
        let scales: [*const f32; TOKENS] =
            std::array::from_fn(|token| activations[token].int8.descales.as_ptr());
        let ones = _mm256_set1_epi16(1);
        let mut totals = [_mm256_setzero_ps(); TOKENS];
        for index in 0..self.blocks {
            let mut sums = [_mm256_setzero_si256(); TOKENS];
            for step in index * per_block..(index + 1) * per_block {
                unsafe {
                    let source = panel.add(step * PANEL_ROWS * STEP);
                    if step % 2 == 0 {
                        _mm_prefetch::<_MM_HINT_T0>(source.wrapping_add(PREFETCH_BYTES).cast());
                    }
                    let weights = _mm256_loadu_si256(source.cast());
                    for token in 0..TOKENS {
                        // 4 个激活码广播到每一行；同一位置的码与各行同一位置的权重相乘。
                        let input = _mm256_set1_epi32(
                            codes[token].add(step * STEP).cast::<i32>().read_unaligned(),
                        );
                        let magnitude = _mm256_set1_epi32(
                            magnitudes[token]
                                .add(step * STEP)
                                .cast::<i32>()
                                .read_unaligned(),
                        );
                        // |a| × (w 带上 a 的符号) = a × w；|a|、|w| ≤ 127，相邻两项之和不会使 i16 饱和。
                        let products =
                            _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(weights, input));
                        sums[token] =
                            _mm256_add_epi32(sums[token], _mm256_madd_epi16(products, ones));
                    }
                }
            }
            let row_scales = unsafe { _mm256_loadu_ps(descales.add(index * PANEL_ROWS)) };
            for token in 0..TOKENS {
                let scale = _mm256_mul_ps(
                    row_scales,
                    _mm256_set1_ps(unsafe { *scales[token].add(index) }),
                );
                totals[token] = _mm256_add_ps(
                    totals[token],
                    _mm256_mul_ps(_mm256_cvtepi32_ps(sums[token]), scale),
                );
            }
        }
        for token in 0..TOKENS {
            unsafe { _mm256_storeu_ps(lanes[token].as_mut_ptr(), totals[token]) };
        }
    }
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
    /// 每个码的绝对值（码在 ±127 内，不会溢出）。AVX2 内核直接拿它做 vpmaddubsw 的无符号操作数，
    /// 不必对每一行权重重新算一遍。
    magnitudes: Vec<Chunk>,
    descales: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(32))]
struct Chunk([i8; CHUNK]);

impl Activation8 {
    pub fn quantize(&mut self, input: &[f32], block: usize) {
        assert!(supported_block(block));
        let blocks = blocks(input.len(), block);
        let chunks = blocks * block / CHUNK;
        self.block = block;
        self.chunks.clear();
        self.chunks.resize(chunks, Chunk([0; CHUNK]));
        self.magnitudes.clear();
        self.magnitudes.resize(chunks, Chunk([0; CHUNK]));
        self.descales.clear();
        self.descales.resize(blocks, 0.0);
        let full = input.len() / block;
        #[cfg(target_arch = "x86_64")]
        let vectorized = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let vectorized = false;
        if vectorized {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                self.quantize_blocks_avx2(&input[..full * block]);
            }
        } else {
            for index in 0..full {
                self.quantize_block(index, &input[index * block..(index + 1) * block]);
            }
        }
        if full < blocks {
            self.quantize_block(full, &input[full * block..]);
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
            self.magnitudes[first + offset / CHUNK].0[offset % CHUNK] = code.abs();
        }
        self.descales[index] = peak / 127.0;
    }

    /// 与 quantize_block 逐位相同：乘法同为 f32，取整用截断加符号修正实现“四舍五入远离零”。
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_blocks_avx2(&mut self, input: &[f32]) {
        use std::arch::x86_64::*;

        let block = self.block;
        let absolute = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let sign = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let upper = _mm256_set1_ps(127.0);
        let lower = _mm256_set1_ps(-127.0);
        // packs 按 128 位分半交错，按 32 位重排回原顺序。
        let natural = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for index in 0..input.len() / block {
            unsafe {
                let source = input.as_ptr().add(index * block);
                let mut peaks = _mm256_setzero_ps();
                let mut invalid = _mm256_setzero_ps();
                for vector in 0..block / 8 {
                    let value = _mm256_loadu_ps(source.add(vector * 8));
                    peaks = _mm256_max_ps(peaks, _mm256_and_ps(value, absolute));
                    // 有限值相减得 +0，无穷大与 NaN 得 NaN。
                    invalid = _mm256_or_ps(invalid, _mm256_sub_ps(value, value));
                }
                let invalid = _mm256_castps_si256(invalid);
                if _mm256_testz_si256(invalid, invalid) == 0 {
                    *self.descales.get_unchecked_mut(index) = f32::NAN;
                    continue;
                }
                let mut stored = [0.0_f32; 8];
                _mm256_storeu_ps(stored.as_mut_ptr(), peaks);
                let peak = stored.into_iter().fold(0.0_f32, f32::max);
                if peak < MIN_PEAK {
                    continue;
                }
                let scale = _mm256_set1_ps(127.0 / peak);
                let first = index * block / CHUNK;
                for chunk in 0..block / CHUNK {
                    let integers: [__m256i; 4] = std::array::from_fn(|part| {
                        let value = _mm256_mul_ps(
                            _mm256_loadu_ps(source.add(chunk * CHUNK + part * 8)),
                            scale,
                        );
                        let truncated =
                            _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(value);
                        let fraction = _mm256_and_ps(_mm256_sub_ps(value, truncated), absolute);
                        let away = _mm256_and_ps(
                            _mm256_cmp_ps::<_CMP_GE_OQ>(fraction, half),
                            _mm256_or_ps(one, _mm256_and_ps(value, sign)),
                        );
                        let rounded = _mm256_add_ps(truncated, away);
                        _mm256_cvtps_epi32(_mm256_max_ps(_mm256_min_ps(rounded, upper), lower))
                    });
                    let low = _mm256_packs_epi32(integers[0], integers[1]);
                    let high = _mm256_packs_epi32(integers[2], integers[3]);
                    let bytes = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(low, high), natural);
                    _mm256_store_si256(self.chunks.as_mut_ptr().add(first + chunk).cast(), bytes);
                    _mm256_store_si256(
                        self.magnitudes.as_mut_ptr().add(first + chunk).cast(),
                        _mm256_abs_epi8(bytes),
                    );
                }
                *self.descales.get_unchecked_mut(index) = peak / 127.0;
            }
        }
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
    fn vectorized_activation_matches_scalar_quantization() {
        for block in BLOCKS {
            for columns in [block, 70, 513, 1024, 3072] {
                let mut input = pseudo_random(columns, 7 + columns as u64);
                input[columns / 3] = 250.0;
                input[0] = -0.0;
                for (index, value) in input.iter_mut().enumerate().skip(5).step_by(97) {
                    // 恰好落在 .5 上的值检验“远离零”的取整方向。
                    *value = (index % 7) as f32 + 0.5;
                }
                let mut vectorized = Activation8::default();
                vectorized.quantize(&input, block);
                let count = blocks(columns, block);
                let mut reference = Activation8 {
                    block,
                    chunks: vec![Chunk([0; CHUNK]); count * block / CHUNK],
                    magnitudes: vec![Chunk([0; CHUNK]); count * block / CHUNK],
                    descales: vec![0.0; count],
                };
                for index in 0..count {
                    let end = ((index + 1) * block).min(columns);
                    reference.quantize_block(index, &input[index * block..end]);
                }
                assert_eq!(
                    vectorized.chunks, reference.chunks,
                    "block={block} columns={columns}"
                );
                assert_eq!(vectorized.magnitudes, reference.magnitudes);
                assert_eq!(
                    vectorized
                        .descales
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    reference
                        .descales
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            let mut invalid = vec![1.0_f32; block * 2];
            invalid[3] = f32::INFINITY;
            invalid[block + 1] = f32::NAN;
            let mut activation = Activation8::default();
            activation.quantize(&invalid, block);
            assert!(activation.descales.iter().all(|v| v.is_nan()));
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
