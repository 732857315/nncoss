use crate::{Error, MODEL_FORMAT_VERSION, Result};

pub(crate) const MODEL_MAGIC: &[u8; 8] = b"NNCSLLM\0";
const MODEL_ARCHITECTURE: u32 = 1;
const FLAG_TIED_EMBEDDINGS: u32 = 1;
const FLAG_QKV_BIAS: u32 = 2;
const FLAG_QK_NORM: u32 = 4;
const KNOWN_FLAGS: u32 = FLAG_TIED_EMBEDDINGS | FLAG_QKV_BIAS | FLAG_QK_NORM;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RopeMode {
    HalfSplit = 0,
    Interleaved = 1,
}

impl TryFrom<u32> for RopeMode {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::HalfSplit),
            1 => Ok(Self::Interleaved),
            _ => Err(Error::Unsupported("RoPE 模式")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub max_context: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub rope_mode: RopeMode,
    pub tied_embeddings: bool,
    pub qkv_bias: bool,
    pub qk_norm: bool,
}

impl Config {
    fn validate(&self) -> Result<()> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.layer_count == 0
            || self.head_count == 0
            || self.kv_head_count == 0
            || self.head_dim == 0
            || self.max_context == 0
        {
            return Err(Error::InvalidModel("模型维度必须为非零值"));
        }
        if self.head_dim % 2 != 0 {
            return Err(Error::InvalidModel("RoPE head_dim 必须为偶数"));
        }
        if self.head_count % self.kv_head_count != 0 {
            return Err(Error::InvalidModel(
                "head_count 必须能被 kv_head_count 整除",
            ));
        }
        self.head_count
            .checked_mul(self.head_dim)
            .ok_or(Error::InvalidModel("Q 维度溢出"))?;
        self.kv_head_count
            .checked_mul(self.head_dim)
            .ok_or(Error::InvalidModel("K/V 维度溢出"))?;
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(Error::InvalidModel("norm_eps 必须为正有限值"));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::InvalidModel("rope_theta 必须为正有限值"));
        }
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.head_count * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_head_count * self.head_dim
    }
}

#[derive(Debug)]
pub(crate) struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o_proj: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

#[derive(Debug)]
pub struct Model {
    pub(crate) config: Config,
    pub(crate) token_embedding: Vec<f32>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) lm_head: Option<Vec<f32>>,
}

impl Model {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        if reader.take(MODEL_MAGIC.len())? != MODEL_MAGIC {
            return Err(Error::InvalidModel("magic 不匹配"));
        }
        let version = reader.read_u32()?;
        if version != MODEL_FORMAT_VERSION {
            return Err(Error::Unsupported("模型格式版本"));
        }
        let architecture = reader.read_u32()?;
        if architecture != MODEL_ARCHITECTURE {
            return Err(Error::Unsupported("模型架构"));
        }
        let vocab_size = reader.read_u32()? as usize;
        let hidden_size = reader.read_u32()? as usize;
        let intermediate_size = reader.read_u32()? as usize;
        let layer_count = reader.read_u32()? as usize;
        let head_count = reader.read_u32()? as usize;
        let kv_head_count = reader.read_u32()? as usize;
        let head_dim = reader.read_u32()? as usize;
        let max_context = reader.read_u32()? as usize;
        let flags = reader.read_u32()?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(Error::Unsupported("模型 flags"));
        }
        let rope_mode = RopeMode::try_from(reader.read_u32()?)?;
        let norm_eps = reader.read_f32()?;
        let rope_theta = reader.read_f32()?;
        let config = Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            layer_count,
            head_count,
            kv_head_count,
            head_dim,
            max_context,
            norm_eps,
            rope_theta,
            rope_mode,
            tied_embeddings: flags & FLAG_TIED_EMBEDDINGS != 0,
            qkv_bias: flags & FLAG_QKV_BIAS != 0,
            qk_norm: flags & FLAG_QK_NORM != 0,
        };
        config.validate()?;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let token_embedding = reader.read_f32s(area(vocab_size, hidden_size)?)?;
        let mut layers = Vec::new();
        for _ in 0..layer_count {
            let attn_norm = reader.read_f32s(hidden_size)?;
            let q_proj = reader.read_f32s(area(q_dim, hidden_size)?)?;
            let k_proj = reader.read_f32s(area(kv_dim, hidden_size)?)?;
            let v_proj = reader.read_f32s(area(kv_dim, hidden_size)?)?;
            let (q_bias, k_bias, v_bias) = if config.qkv_bias {
                (
                    Some(reader.read_f32s(q_dim)?),
                    Some(reader.read_f32s(kv_dim)?),
                    Some(reader.read_f32s(kv_dim)?),
                )
            } else {
                (None, None, None)
            };
            let (q_norm, k_norm) = if config.qk_norm {
                (
                    Some(reader.read_f32s(head_dim)?),
                    Some(reader.read_f32s(head_dim)?),
                )
            } else {
                (None, None)
            };
            let o_proj = reader.read_f32s(area(hidden_size, q_dim)?)?;
            let ffn_norm = reader.read_f32s(hidden_size)?;
            let gate_proj = reader.read_f32s(area(intermediate_size, hidden_size)?)?;
            let up_proj = reader.read_f32s(area(intermediate_size, hidden_size)?)?;
            let down_proj = reader.read_f32s(area(hidden_size, intermediate_size)?)?;
            layers.push(LayerWeights {
                attn_norm,
                q_proj,
                k_proj,
                v_proj,
                q_bias,
                k_bias,
                v_bias,
                q_norm,
                k_norm,
                o_proj,
                ffn_norm,
                gate_proj,
                up_proj,
                down_proj,
            });
        }
        let final_norm = reader.read_f32s(hidden_size)?;
        let lm_head = if config.tied_embeddings {
            None
        } else {
            Some(reader.read_f32s(area(vocab_size, hidden_size)?)?)
        };
        reader.finish()?;
        Ok(Self {
            config,
            token_embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

fn area(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error::InvalidModel("张量尺寸溢出"))
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(Error::InvalidModel("模型长度溢出"))?;
        if end > self.bytes.len() {
            return Err(Error::InvalidModel("模型数据被截断"));
        }
        let output = &self.bytes[self.position..end];
        self.position = end;
        Ok(output)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f32s(&mut self, count: usize) -> Result<Vec<f32>> {
        let byte_count = count
            .checked_mul(4)
            .ok_or(Error::InvalidModel("张量尺寸溢出"))?;
        let bytes = self.take(byte_count)?;
        let mut values = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !value.is_finite() {
                return Err(Error::InvalidModel("权重包含非有限值"));
            }
            values.push(value);
        }
        Ok(values)
    }

    fn finish(&self) -> Result<()> {
        if self.position != self.bytes.len() {
            return Err(Error::InvalidModel("模型末尾存在多余数据"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{tiny_model, tiny_model_bytes};

    #[test]
    fn loads_valid_model() {
        let model = tiny_model();
        assert_eq!(model.config().vocab_size, 5);
        assert_eq!(model.config().head_count, 2);
        assert!(model.config().qkv_bias);
        assert!(model.config().qk_norm);
    }

    #[test]
    fn rejects_invalid_and_truncated_models() {
        let bytes = tiny_model_bytes();
        for length in [0, 7, 8, 63, 64, bytes.len() - 1] {
            assert!(Model::from_bytes(&bytes[..length]).is_err());
        }
        let mut invalid = bytes.clone();
        invalid[0] ^= 1;
        assert!(Model::from_bytes(&invalid).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(Model::from_bytes(&trailing).is_err());
    }

    #[test]
    fn rejects_invalid_head_combination() {
        let mut bytes = tiny_model_bytes();
        bytes[36..40].copy_from_slice(&4_u32.to_le_bytes());
        assert!(Model::from_bytes(&bytes).is_err());
    }
}
