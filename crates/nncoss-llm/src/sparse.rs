//! decode 时逐 token 稀疏化矩阵输入通道。
//!
//! 每个 token 只保留各矩阵乘法输入中绝对值最大的一部分通道，被丢弃的通道对应的权重列不再读取。
//! decode 受内存带宽限制，少读多少列就省多少时间。为此给 Q4 矩阵另存一份按列连续的副本：
//! 量化数值与原矩阵完全相同（每行一个 scale、每行每 32 列一个 code），只是半字节改为按列存放，
//! 因此结果等于把被丢弃的通道置零后再用原矩阵计算，只有浮点求和顺序不同。
//! 共享同一输入的矩阵（q/k/v，gate/up）按行拼成一个副本，使每列成为 2–3 KB 的连续数据：
//! 连续段太短时，硬件预取器会把跳过的数据也读进来，省不下带宽（见 examples/sparse_read.rs）。
//! o 与 down 只有 1024 行，按线程分行后每列每个线程只有 128 字节，实测比稠密内核更慢，
//! 质量代价也最高（丢一半通道平均 KL 约 0.04），因此始终完整计算，不建立按列副本。

use std::cell::RefCell;
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::str::FromStr;

use crate::model::Model;
use crate::pool::{SharedPtr, parallel_for, threads};
use crate::q4::{GROUP_SIZE, Q4Matrix, Weight, groups};
use crate::{Error, Result};

const BLOCK_ROWS: usize = 64;
const BLOCK_BYTES: usize = BLOCK_ROWS / 2;
const VECTOR_ROWS: usize = 16;
const MIN_TASK_BLOCKS: usize = 4;
/// 每个任务最多 4096 行：16 位与 32 位累加器共 24 KB，留在 L1 中；每列每次仍连续读 2 KB。
/// 输出层有 151,936 行，不设上限时每个任务的累加器有 150 KB，每读一列都要在 L2 来回一遍。
const MAX_TASK_BLOCKS: usize = 64;
const MIN_PEAK: f32 = 1.0e-30;

thread_local! {
    static SCRATCH: RefCell<(Vec<f32>, Vec<i16>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// 64 行一块，块内第 `within` 行的半字节位置：（字节序号，是否高半字节）。
/// 这种排列让 AVX2 的 unpacklo/unpackhi 直接得到按行顺序排列的 16 位整数。
const fn slot(within: usize) -> (usize, bool) {
    let vector = within / VECTOR_ROWS;
    let element = within % VECTOR_ROWS;
    let byte = if element < 8 { element } else { 8 + element };
    (byte + 8 * (vector % 2), vector >= 2)
}

/// 按列存放半字节的 Q4 矩阵副本，可由几个列数相同的矩阵按行拼接而成。
#[derive(Debug)]
pub(crate) struct ColumnMatrix {
    rows: usize,
    padded: usize,
    columns: usize,
    scales: Vec<f32>,
    /// [组][行]：每行每 32 列一个 code，与原矩阵相同。
    codes: Vec<i8>,
    /// [列][块][32 字节]：每列 padded / 2 字节连续存放。
    nibbles: Vec<u8>,
}

impl ColumnMatrix {
    pub(crate) fn from_parts(parts: &[&Q4Matrix]) -> Self {
        let columns = parts[0].columns;
        assert!(parts.iter().all(|part| part.columns == columns));
        let sources: Vec<(&Q4Matrix, usize)> = parts
            .iter()
            .flat_map(|part| (0..part.rows).map(move |row| (*part, row)))
            .collect();
        let rows = sources.len();
        let padded = rows.next_multiple_of(BLOCK_ROWS);
        let group_count = groups(columns);
        let stride = padded / 2;
        let mut scales = vec![0.0; padded];
        let mut codes = vec![0_i8; group_count * padded];
        for (row, (part, local)) in sources.iter().enumerate() {
            scales[row] = part.row_scale(*local);
            for (group, code) in part.row_codes(*local).iter().enumerate() {
                codes[group * padded + row] = *code as i8;
            }
        }
        let mut nibbles = vec![0_u8; columns * stride];
        let slots: [(usize, bool); BLOCK_ROWS] = std::array::from_fn(slot);
        for block in 0..padded / BLOCK_ROWS {
            let first = block * BLOCK_ROWS;
            let packed: Vec<&[u8]> = sources[first..(first + BLOCK_ROWS).min(rows)]
                .iter()
                .map(|(part, row)| part.row_nibbles(*row))
                .collect();
            for column in 0..columns {
                let target = &mut nibbles[column * stride + block * BLOCK_BYTES..][..BLOCK_BYTES];
                for (within, packed) in packed.iter().enumerate() {
                    let value = (packed[column / 2] >> (4 * (column % 2))) & 0x0f;
                    let (byte, high) = slots[within];
                    target[byte] |= if high { value << 4 } else { value };
                }
            }
        }
        Self {
            rows,
            padded,
            columns,
            scales,
            codes,
            nibbles,
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// output = W·x，只读取 `input` 中保留的列。
    pub(crate) fn multiply(&self, input: &SparseInput, output: &mut [f32]) {
        assert_eq!(output.len(), self.rows);
        assert_eq!(input.columns, self.columns);
        let blocks = self.padded / BLOCK_ROWS;
        let tasks = threads()
            .min(blocks.div_ceil(MIN_TASK_BLOCKS))
            .max(blocks.div_ceil(MAX_TASK_BLOCKS))
            .max(1);
        let per_task = blocks.div_ceil(tasks);
        let target = SharedPtr(output.as_mut_ptr());
        let run = |task: usize| {
            let first = task * per_task;
            let last = ((task + 1) * per_task).min(blocks);
            if first >= last {
                return;
            }
            let rows = first * BLOCK_ROWS..(last * BLOCK_ROWS).min(self.rows);
            // 各任务写互不重叠的行。
            let output =
                unsafe { std::slice::from_raw_parts_mut(target.get().add(rows.start), rows.len()) };
            self.multiply_blocks(input, first..last, output);
        };
        if tasks == 1 {
            run(0);
        } else {
            parallel_for(tasks, &run);
        }
    }

    fn multiply_blocks(&self, input: &SparseInput, blocks: Range<usize>, output: &mut [f32]) {
        SCRATCH.with(|scratch| {
            let (wide, narrow) = &mut *scratch.borrow_mut();
            let count = blocks.len() * BLOCK_ROWS;
            wide.clear();
            wide.resize(count, 0.0);
            narrow.resize(count, 0);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe { self.accumulate_avx2(input, blocks.clone(), wide, narrow) };
            } else {
                self.accumulate_scalar(input, blocks.clone(), wide, narrow);
            }
            #[cfg(not(target_arch = "x86_64"))]
            self.accumulate_scalar(input, blocks.clone(), wide, narrow);
            let first = blocks.start * BLOCK_ROWS;
            for (row, value) in output.iter_mut().enumerate() {
                *value = self.scales[first + row] * wide[row];
            }
        });
    }

    fn column_bytes(&self, column: u32, blocks: &Range<usize>) -> &[u8] {
        let start = column as usize * (self.padded / 2) + blocks.start * BLOCK_BYTES;
        &self.nibbles[start..start + blocks.len() * BLOCK_BYTES]
    }

    fn group_codes(&self, group: u32, blocks: &Range<usize>) -> &[i8] {
        let start = group as usize * self.padded + blocks.start * BLOCK_ROWS;
        &self.codes[start..start + blocks.len() * BLOCK_ROWS]
    }

    /// 每组内先用 16 位整数累加 Σ n·q（可能回绕），减去 8·Σq 后恰为 Σ (n−8)·q，
    /// 其绝对值不超过 32·8·127 = 32512，因此回绕不影响结果；再乘组 code 与激活 scale 累加到浮点。
    fn accumulate_scalar(
        &self,
        input: &SparseInput,
        blocks: Range<usize>,
        wide: &mut [f32],
        narrow: &mut [i16],
    ) {
        let slots: [(usize, bool); BLOCK_ROWS] = std::array::from_fn(slot);
        for span in &input.spans {
            narrow.fill(0);
            for &(column, factor) in &input.entries[span.entries.clone()] {
                let bytes = self.column_bytes(column, &blocks);
                for (block, packed) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
                    let sums = &mut narrow[block * BLOCK_ROWS..(block + 1) * BLOCK_ROWS];
                    for (sum, (byte, high)) in sums.iter_mut().zip(slots) {
                        let nibble = if high {
                            packed[byte] >> 4
                        } else {
                            packed[byte] & 0x0f
                        };
                        *sum = sum.wrapping_add(i16::from(nibble).wrapping_mul(factor));
                    }
                }
            }
            let codes = self.group_codes(span.group, &blocks);
            for ((total, sum), code) in wide.iter_mut().zip(narrow.iter()).zip(codes) {
                *total += f32::from(*code) * span.scale * f32::from(sum.wrapping_sub(span.offset));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn accumulate_avx2(
        &self,
        input: &SparseInput,
        blocks: Range<usize>,
        wide: &mut [f32],
        narrow: &mut [i16],
    ) {
        use std::arch::x86_64::*;

        let mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let vectors = blocks.len() * BLOCK_ROWS / VECTOR_ROWS;
        let sums = narrow.as_mut_ptr().cast::<__m256i>();
        for span in &input.spans {
            for vector in 0..vectors {
                unsafe { _mm256_storeu_si256(sums.add(vector), zero) };
            }
            // 每次取 4 列（两对）：两列的半字节交错成 (A, B) 字节对，一条 maddubs 得到 A·qa + B·qb
            // （不超过 2·15·127，不会饱和），累加器每 4 列才读写一次。不足 4 列时用系数 0 补齐。
            let entries = &input.entries[span.entries.clone()];
            for (chunk_index, chunk) in entries.chunks(4).enumerate() {
                let pick = |slot: usize| chunk.get(slot).copied().unwrap_or((chunk[0].0, 0));
                let columns: [*const u8; 4] =
                    std::array::from_fn(|slot| self.column_bytes(pick(slot).0, &blocks).as_ptr());
                let factors: [__m256i; 2] = std::array::from_fn(|pair| {
                    let first = pick(2 * pair).1 as i8 as u8;
                    let second = pick(2 * pair + 1).1 as i8 as u8;
                    _mm256_set1_epi16(i16::from_le_bytes([first, second]))
                });
                let pairs = chunk.len().div_ceil(2);
                if let Some(next) = entries.get((chunk_index + 1) * 4..) {
                    for &(column, _) in next.iter().take(4) {
                        let bytes = self.column_bytes(column, &blocks).as_ptr();
                        _mm_prefetch::<_MM_HINT_T0>(bytes.cast());
                        _mm_prefetch::<_MM_HINT_T0>(bytes.wrapping_add(64).cast());
                    }
                }
                for block in 0..blocks.len() {
                    unsafe {
                        let target = sums.add(block * 4);
                        let mut totals: [__m256i; 4] =
                            std::array::from_fn(|part| _mm256_loadu_si256(target.add(part)));
                        for pair in 0..pairs {
                            let split = |column: *const u8| {
                                let packed =
                                    _mm256_loadu_si256(column.add(block * BLOCK_BYTES).cast());
                                (
                                    _mm256_and_si256(packed, mask),
                                    _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask),
                                )
                            };
                            let (low_a, high_a) = split(columns[2 * pair]);
                            let (low_b, high_b) = split(columns[2 * pair + 1]);
                            let parts = [
                                _mm256_unpacklo_epi8(low_a, low_b),
                                _mm256_unpackhi_epi8(low_a, low_b),
                                _mm256_unpacklo_epi8(high_a, high_b),
                                _mm256_unpackhi_epi8(high_a, high_b),
                            ];
                            for (total, nibbles) in totals.iter_mut().zip(parts) {
                                *total = _mm256_add_epi16(
                                    *total,
                                    _mm256_maddubs_epi16(nibbles, factors[pair]),
                                );
                            }
                        }
                        for (part, total) in totals.into_iter().enumerate() {
                            _mm256_storeu_si256(target.add(part), total);
                        }
                    }
                }
            }
            let offset = _mm256_set1_epi16(span.offset);
            let scale = _mm256_set1_ps(span.scale);
            let codes = self.group_codes(span.group, &blocks).as_ptr();
            let totals = wide.as_mut_ptr();
            for vector in 0..vectors {
                unsafe {
                    let sum = _mm256_sub_epi16(_mm256_loadu_si256(sums.add(vector)), offset);
                    let code = _mm_loadu_si128(codes.add(vector * VECTOR_ROWS).cast());
                    let halves = [
                        (
                            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum)),
                            _mm256_cvtepi8_epi32(code),
                        ),
                        (
                            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum, 1)),
                            _mm256_cvtepi8_epi32(_mm_srli_si128(code, 8)),
                        ),
                    ];
                    for (half, (sum, code)) in halves.into_iter().enumerate() {
                        let target = totals.add(vector * VECTOR_ROWS + half * 8);
                        let weight = _mm256_mul_ps(_mm256_cvtepi32_ps(code), scale);
                        let value = _mm256_mul_ps(weight, _mm256_cvtepi32_ps(sum));
                        _mm256_storeu_ps(target, _mm256_add_ps(_mm256_loadu_ps(target), value));
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Span {
    group: u32,
    scale: f32,
    /// 8·Σq，按 16 位回绕。
    offset: i16,
    entries: Range<usize>,
}

/// 一个 token 的稀疏输入：保留的列及其 int8 量化值（与稠密路径的 Activation 舍入方式相同）。
#[derive(Debug, Default)]
pub(crate) struct SparseInput {
    columns: usize,
    magnitudes: Vec<u32>,
    entries: Vec<(u32, i16)>,
    spans: Vec<Span>,
}

impl SparseInput {
    /// 保留 |x| 最大的 `keep` 个通道（并列时可能多留几个），每 32 个通道一组量化成 int8，只记录非零项。
    /// 输入含非有限值时返回 false，调用方应改走稠密路径，让 NaN 照常传播。
    pub(crate) fn select(&mut self, input: &[f32], keep: usize) -> bool {
        self.columns = input.len();
        self.entries.clear();
        self.spans.clear();
        if input.iter().any(|value| !value.is_finite()) {
            return false;
        }
        let threshold = self.threshold(input, keep);
        for (group, values) in input.chunks(GROUP_SIZE).enumerate() {
            let peak = values
                .iter()
                .filter(|value| value.abs().to_bits() >= threshold)
                .fold(0.0_f32, |peak, value| peak.max(value.abs()));
            if peak < MIN_PEAK {
                continue;
            }
            let inverse = 127.0 / peak;
            let start = self.entries.len();
            let mut total = 0_i16;
            for (index, value) in values.iter().enumerate() {
                if value.abs().to_bits() < threshold {
                    continue;
                }
                let code = (value * inverse).round_ties_even() as i16;
                if code != 0 {
                    self.entries
                        .push(((group * GROUP_SIZE + index) as u32, code));
                    total += code;
                }
            }
            if self.entries.len() > start {
                self.spans.push(Span {
                    group: group as u32,
                    scale: peak / 127.0,
                    offset: total.wrapping_mul(8),
                    entries: start..self.entries.len(),
                });
            }
        }
        true
    }

    /// 第 keep 大的 |x| 的位模式；非负浮点数的位模式与数值同序。
    fn threshold(&mut self, input: &[f32], keep: usize) -> u32 {
        let keep = keep.max(1);
        if keep >= input.len() {
            return 0;
        }
        self.magnitudes.clear();
        self.magnitudes
            .extend(input.iter().map(|value| value.abs().to_bits()));
        *self.magnitudes.select_nth_unstable(input.len() - keep).1
    }

    #[cfg(test)]
    fn kept(&self) -> usize {
        self.entries.len()
    }
}

/// 一层中共享输入的矩阵各拼成一个按列副本。
#[derive(Debug)]
pub(crate) struct ColumnLayer {
    pub qkv: ColumnMatrix,
    pub gate_up: ColumnMatrix,
}

#[derive(Debug)]
pub(crate) struct ColumnWeights {
    pub layers: Vec<ColumnLayer>,
    pub head: ColumnMatrix,
}

impl ColumnWeights {
    /// 全部矩阵都是 Q4 时才建立；f32 模型返回 None。
    pub(crate) fn build(model: &Model) -> Option<Self> {
        fn q4(weight: &Weight) -> Option<&Q4Matrix> {
            match weight {
                Weight::Q4(matrix) => Some(matrix),
                Weight::F32(_) | Weight::Q8(_) => None,
            }
        }
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            layers.push(ColumnLayer {
                qkv: ColumnMatrix::from_parts(&[
                    q4(&layer.q_proj)?,
                    q4(&layer.k_proj)?,
                    q4(&layer.v_proj)?,
                ]),
                gate_up: ColumnMatrix::from_parts(&[q4(&layer.gate_proj)?, q4(&layer.up_proj)?]),
            });
        }
        Some(Self {
            layers,
            head: ColumnMatrix::from_parts(&[q4(model.head())?]),
        })
    }
}

/// decode 时各部分矩阵输入丢弃的通道比例，0 表示这一部分完整计算；o 与 down 始终完整计算。
/// 文本形式为单个比例（三部分相同），或 `attn=0.15,mlp=0.15,head=0.3`（未列出的部分为 0）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InputSparsity {
    /// q、k、v 投影的输入（注意力前的 RMSNorm 输出）。
    pub attention: f32,
    /// gate、up 投影的输入（MLP 前的 RMSNorm 输出）。
    pub mlp: f32,
    /// 输出层的输入（最后的 RMSNorm 输出）。
    pub head: f32,
}

const PART_NAMES: [&str; 3] = ["attn", "mlp", "head"];

impl InputSparsity {
    pub const NONE: Self = Self {
        attention: 0.0,
        mlp: 0.0,
        head: 0.0,
    };

    pub fn uniform(drop: f32) -> Self {
        Self {
            attention: drop,
            mlp: drop,
            head: drop,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.levels().iter().any(|level| *level > 0.0)
    }

    fn levels(&self) -> [f32; 3] {
        [self.attention, self.mlp, self.head]
    }

    fn level_mut(&mut self, part: usize) -> &mut f32 {
        match part {
            0 => &mut self.attention,
            1 => &mut self.mlp,
            _ => &mut self.head,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.levels().iter().all(|level| (0.0..1.0).contains(level)) {
            Ok(())
        } else {
            Err(Error::InvalidArgument("输入稀疏比例须在 [0, 1) 内"))
        }
    }
}

impl FromStr for InputSparsity {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        let invalid = Error::InvalidArgument("输入稀疏比例写作单个数，或 attn=…,mlp=…,head=…");
        let text = text.trim();
        let sparsity = if let Ok(drop) = text.parse::<f32>() {
            Self::uniform(drop)
        } else {
            let mut sparsity = Self::NONE;
            for item in text.split(',') {
                let (name, value) = item.split_once('=').ok_or(invalid.clone())?;
                let part = PART_NAMES
                    .iter()
                    .position(|candidate| *candidate == name.trim())
                    .ok_or(invalid.clone())?;
                *sparsity.level_mut(part) = value.trim().parse().map_err(|_| invalid.clone())?;
            }
            sparsity
        };
        sparsity.validate()?;
        Ok(sparsity)
    }
}

impl Display for InputSparsity {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = PART_NAMES
            .iter()
            .zip(self.levels())
            .map(|(name, level)| format!("{name}={level}"))
            .collect();
        formatter.write_str(&parts.join(","))
    }
}

/// 丢弃比例换算成保留的通道数，至少保留一个。
pub(crate) fn kept_channels(width: usize, drop: f32) -> usize {
    ((width as f32 * (1.0 - drop)).round() as usize).clamp(1, width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q4::test_support::{encode, pseudo_random};
    use crate::q4::{Weight, linear};

    fn matrix(rows: usize, columns: usize, seed: u64) -> Q4Matrix {
        let values = pseudo_random(rows * columns, seed);
        Q4Matrix::new(rows, columns, encode(&values, rows, columns)).unwrap()
    }

    fn masked(input: &[f32], keep: usize) -> Vec<f32> {
        let mut sorted: Vec<f32> = input.iter().map(|value| value.abs()).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let threshold = sorted[keep.max(1) - 1];
        input
            .iter()
            .map(|value| {
                if value.abs() >= threshold {
                    *value
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        let spread = expected
            .iter()
            .fold(1.0e-6_f32, |peak, value| peak.max(value.abs()));
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-4 * spread,
                "第 {row} 行：{actual} != {expected}"
            );
        }
    }

    #[test]
    fn block_slots_cover_every_nibble_once() {
        let mut seen = [[false; 2]; BLOCK_BYTES];
        for within in 0..BLOCK_ROWS {
            let (byte, high) = slot(within);
            assert!(!seen[byte][usize::from(high)]);
            seen[byte][usize::from(high)] = true;
        }
    }

    #[test]
    fn keeping_every_channel_matches_the_dense_kernel() {
        for (rows, columns) in [(64, 32), (100, 72), (130, 256), (8, 40)] {
            let weight = matrix(rows, columns, rows as u64 * 31 + columns as u64);
            let input = pseudo_random(columns, 7);
            let dense = linear(&Weight::Q4(weight.clone()), rows, columns, &input, None);
            let column = ColumnMatrix::from_parts(&[&weight]);
            let mut sparse = SparseInput::default();
            assert!(sparse.select(&input, columns));
            let mut output = vec![0.0; rows];
            column.multiply(&sparse, &mut output);
            assert_close(&output, &dense);
        }
    }

    #[test]
    fn dropped_channels_act_like_zero_inputs() {
        let (rows, columns) = (200, 96);
        let weight = matrix(rows, columns, 3);
        let input = pseudo_random(columns, 11);
        let column = ColumnMatrix::from_parts(&[&weight]);
        for keep in [1, 17, 48, 72] {
            let mut sparse = SparseInput::default();
            assert!(sparse.select(&input, keep));
            assert!(sparse.kept() <= keep);
            let mut output = vec![0.0; rows];
            column.multiply(&sparse, &mut output);
            let dense = linear(
                &Weight::Q4(weight.clone()),
                rows,
                columns,
                &masked(&input, keep),
                None,
            );
            assert_close(&output, &dense);
        }
    }

    #[test]
    fn fused_parts_stack_their_rows() {
        let columns = 64;
        let parts = [
            matrix(48, columns, 1),
            matrix(32, columns, 2),
            matrix(70, columns, 3),
        ];
        let input = pseudo_random(columns, 5);
        let fused = ColumnMatrix::from_parts(&parts.iter().collect::<Vec<_>>());
        let mut sparse = SparseInput::default();
        assert!(sparse.select(&input, 40));
        let mut output = vec![0.0; fused.rows()];
        fused.multiply(&sparse, &mut output);
        let mut expected = Vec::new();
        for part in &parts {
            let single = ColumnMatrix::from_parts(&[part]);
            let mut values = vec![0.0; part.rows];
            single.multiply(&sparse, &mut values);
            expected.extend(values);
        }
        assert_close(&output, &expected);
    }

    #[test]
    fn many_rows_split_across_tasks() {
        let (rows, columns) = (4096 + 64 * 3 + 5, 64);
        let weight = matrix(rows, columns, 9);
        let input = pseudo_random(columns, 13);
        let column = ColumnMatrix::from_parts(&[&weight]);
        let mut sparse = SparseInput::default();
        assert!(sparse.select(&input, 48));
        let mut output = vec![0.0; rows];
        column.multiply(&sparse, &mut output);
        let dense = linear(
            &Weight::Q4(weight),
            rows,
            columns,
            &masked(&input, 48),
            None,
        );
        assert_close(&output, &dense);
    }

    #[test]
    fn scalar_kernel_matches_the_vector_kernel() {
        let (rows, columns) = (192, 96);
        let weight = matrix(rows, columns, 21);
        let input = pseudo_random(columns, 4);
        let column = ColumnMatrix::from_parts(&[&weight]);
        let mut sparse = SparseInput::default();
        assert!(sparse.select(&input, 70));
        let mut output = vec![0.0; rows];
        column.multiply(&sparse, &mut output);
        let mut wide = vec![0.0; column.padded];
        let mut narrow = vec![0; column.padded];
        column.accumulate_scalar(
            &sparse,
            0..column.padded / BLOCK_ROWS,
            &mut wide,
            &mut narrow,
        );
        let scalar: Vec<f32> = (0..rows)
            .map(|row| column.scales[row] * wide[row])
            .collect();
        assert_close(&scalar, &output);
    }

    #[test]
    fn non_finite_inputs_fall_back_to_the_dense_path() {
        let mut input = pseudo_random(64, 1);
        input[5] = f32::NAN;
        assert!(!SparseInput::default().select(&input, 32));
    }

    #[test]
    fn sparsity_parses_uniform_and_per_part_forms() {
        assert_eq!(
            "0.25".parse::<InputSparsity>().unwrap(),
            InputSparsity::uniform(0.25)
        );
        let parsed: InputSparsity = "mlp=0.15, head=0.3".parse().unwrap();
        assert_eq!(
            parsed,
            InputSparsity {
                mlp: 0.15,
                head: 0.3,
                ..InputSparsity::NONE
            }
        );
        assert_eq!(parsed.to_string().parse::<InputSparsity>().unwrap(), parsed);
        for invalid in ["1.0", "mlp=1", "down=0.1", "head", "-0.1"] {
            assert!(invalid.parse::<InputSparsity>().is_err(), "{invalid}");
        }
        assert!(!InputSparsity::NONE.is_enabled());
    }

    #[test]
    fn kept_channel_count_rounds_and_keeps_at_least_one() {
        assert_eq!(kept_channels(1024, 0.25), 768);
        assert_eq!(kept_channels(1024, 0.0), 1024);
        assert_eq!(kept_channels(3, 0.99), 1);
    }
}
