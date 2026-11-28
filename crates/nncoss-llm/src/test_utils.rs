use crate::model::MODEL_MAGIC;
use crate::q4::test_support::encode;
use crate::q4::{Q4Matrix, Weight};
use crate::{Model, RopeMode};

#[derive(Debug, Clone, Copy)]
pub(crate) struct TinySpec {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_context: usize,
    pub tied: bool,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    pub rope_mode: RopeMode,
    pub rope_theta: f32,
}

impl TinySpec {
    pub const BASE: Self = Self {
        vocab: 5,
        hidden: 4,
        intermediate: 8,
        layers: 1,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        max_context: 16,
        tied: false,
        qkv_bias: true,
        qk_norm: true,
        rope_mode: RopeMode::HalfSplit,
        rope_theta: 10_000.0,
    };

    pub const WIDE: Self = Self {
        vocab: 11,
        hidden: 16,
        intermediate: 24,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        max_context: 32,
        tied: false,
        qkv_bias: true,
        qk_norm: true,
        rope_mode: RopeMode::HalfSplit,
        rope_theta: 10_000.0,
    };

    pub fn flags(&self) -> u32 {
        u32::from(self.tied) | (u32::from(self.qkv_bias) << 1) | (u32::from(self.qk_norm) << 2)
    }

    pub fn bytes(&self) -> Vec<u8> {
        let q_dim = self.heads * self.head_dim;
        let kv_dim = self.kv_heads * self.head_dim;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MODEL_MAGIC);
        push_u32(&mut bytes, 1);
        push_u32(&mut bytes, 1);
        for value in [
            self.vocab,
            self.hidden,
            self.intermediate,
            self.layers,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.max_context,
        ] {
            push_u32(&mut bytes, value as u32);
        }
        push_u32(&mut bytes, self.flags());
        push_u32(&mut bytes, self.rope_mode as u32);
        push_f32(&mut bytes, 1e-5);
        push_f32(&mut bytes, self.rope_theta);
        let mut salt = 0;
        tensor(&mut bytes, self.vocab * self.hidden, &mut salt);
        for _ in 0..self.layers {
            tensor(&mut bytes, self.hidden, &mut salt);
            tensor(&mut bytes, q_dim * self.hidden, &mut salt);
            tensor(&mut bytes, kv_dim * self.hidden, &mut salt);
            tensor(&mut bytes, kv_dim * self.hidden, &mut salt);
            if self.qkv_bias {
                tensor(&mut bytes, q_dim, &mut salt);
                tensor(&mut bytes, kv_dim, &mut salt);
                tensor(&mut bytes, kv_dim, &mut salt);
            }
            if self.qk_norm {
                tensor(&mut bytes, self.head_dim, &mut salt);
                tensor(&mut bytes, self.head_dim, &mut salt);
            }
            tensor(&mut bytes, self.hidden * q_dim, &mut salt);
            tensor(&mut bytes, self.hidden, &mut salt);
            tensor(&mut bytes, self.intermediate * self.hidden, &mut salt);
            tensor(&mut bytes, self.intermediate * self.hidden, &mut salt);
            tensor(&mut bytes, self.hidden * self.intermediate, &mut salt);
        }
        tensor(&mut bytes, self.hidden, &mut salt);
        if !self.tied {
            tensor(&mut bytes, self.vocab * self.hidden, &mut salt);
        }
        bytes
    }

    pub fn model(&self) -> Model {
        Model::from_bytes(&self.bytes()).unwrap()
    }

    pub fn quantized_model(&self) -> Model {
        let mut model = self.model();
        model.config.quantized = true;
        let config = model.config.clone();
        let (hidden, q_dim, kv_dim) = (config.hidden_size, config.q_dim(), config.kv_dim());
        let inner = config.intermediate_size;
        quantize(&mut model.token_embedding, config.vocab_size, hidden);
        if let Some(head) = &mut model.lm_head {
            quantize(head, config.vocab_size, hidden);
        }
        for layer in &mut model.layers {
            quantize(&mut layer.q_proj, q_dim, hidden);
            quantize(&mut layer.k_proj, kv_dim, hidden);
            quantize(&mut layer.v_proj, kv_dim, hidden);
            quantize(&mut layer.o_proj, hidden, q_dim);
            quantize(&mut layer.gate_proj, inner, hidden);
            quantize(&mut layer.up_proj, inner, hidden);
            quantize(&mut layer.down_proj, hidden, inner);
        }
        model
    }

    /// 7 个投影矩阵改为 64 元素分块 INT8，嵌入与 LM Head 保持 f32（ncnn 的 INT8 模型）。
    pub fn int8_model(&self) -> Model {
        self.int8_model_with(64, false)
    }

    /// 按给定块长量化投影矩阵；`embeddings` 时嵌入与 LM Head 也量化（GGUF Q8_0 模型）。
    pub fn int8_model_with(&self, block: usize, embeddings: bool) -> Model {
        let mut model = self.model();
        model.config.int8 = true;
        model.config.int8_block = block;
        model.config.int8_embeddings = embeddings;
        let config = model.config.clone();
        let (hidden, q_dim, kv_dim) = (config.hidden_size, config.q_dim(), config.kv_dim());
        let inner = config.intermediate_size;
        if embeddings {
            quantize_int8(&mut model.token_embedding, config.vocab_size, hidden, block);
            if let Some(head) = &mut model.lm_head {
                quantize_int8(head, config.vocab_size, hidden, block);
            }
        }
        for layer in &mut model.layers {
            quantize_int8(&mut layer.q_proj, q_dim, hidden, block);
            quantize_int8(&mut layer.k_proj, kv_dim, hidden, block);
            quantize_int8(&mut layer.v_proj, kv_dim, hidden, block);
            quantize_int8(&mut layer.o_proj, hidden, q_dim, block);
            quantize_int8(&mut layer.gate_proj, inner, hidden, block);
            quantize_int8(&mut layer.up_proj, inner, hidden, block);
            quantize_int8(&mut layer.down_proj, hidden, inner, block);
        }
        model
    }

    pub fn wide_variants() -> Vec<Self> {
        let mut variants = Vec::new();
        for tied in [false, true] {
            for qkv_bias in [false, true] {
                for qk_norm in [false, true] {
                    for rope_mode in [RopeMode::HalfSplit, RopeMode::Interleaved] {
                        variants.push(Self {
                            tied,
                            qkv_bias,
                            qk_norm,
                            rope_mode,
                            ..Self::WIDE
                        });
                    }
                }
            }
        }
        variants
    }
}

fn quantize(weight: &mut Weight, rows: usize, columns: usize) {
    let values = std::mem::take(weight).into_f32();
    let bytes = encode(&values, rows, columns);
    *weight = Weight::Q4(Q4Matrix::new(rows, columns, bytes).unwrap());
}

fn quantize_int8(weight: &mut Weight, rows: usize, columns: usize, block: usize) {
    let values = std::mem::take(weight).into_f32();
    let bytes = crate::q8::test_support::encode(&values, rows, columns, block);
    *weight = Weight::Q8(crate::q8::Q8Matrix::new(rows, columns, block, bytes).unwrap());
}

pub(crate) fn tiny_model() -> Model {
    TinySpec::BASE.model()
}

pub(crate) fn tiny_model_bytes() -> Vec<u8> {
    TinySpec::BASE.bytes()
}

fn tensor(bytes: &mut Vec<u8>, length: usize, salt: &mut usize) {
    *salt += 1;
    for index in 0..length {
        let value = ((index * 17 + *salt * 31) % 97) as f32;
        push_f32(bytes, (value - 48.0) / 192.0);
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/tiny_v1.nncossllm");

    #[test]
    fn base_spec_matches_checked_in_fixture() {
        assert_eq!(tiny_model_bytes(), FIXTURE);
    }

    #[test]
    fn every_wide_variant_loads() {
        for spec in TinySpec::wide_variants() {
            let model = spec.model();
            assert_eq!(model.config().tied_embeddings, spec.tied);
            assert_eq!(model.config().rope_mode, spec.rope_mode);
            assert_eq!(model.lm_head.is_none(), spec.tied);
        }
    }
}
