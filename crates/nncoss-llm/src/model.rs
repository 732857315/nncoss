use std::sync::OnceLock;

use crate::q4::{Weight, matrix_bytes as q4_matrix_bytes};
use crate::sparse::ColumnWeights;
use crate::{Error, MODEL_FORMAT_VERSION, Result};

pub(crate) const MODEL_MAGIC: &[u8; 8] = b"NNCSLLM\0";
const MODEL_ARCHITECTURE: u32 = 1;
const FLAG_TIED_EMBEDDINGS: u32 = 1;
const FLAG_QKV_BIAS: u32 = 2;
const FLAG_QK_NORM: u32 = 4;
const FLAG_Q4_WEIGHTS: u32 = 8;
const KNOWN_FLAGS: u32 = FLAG_TIED_EMBEDDINGS | FLAG_QKV_BIAS | FLAG_QK_NORM | FLAG_Q4_WEIGHTS;
const HEADER_LENGTH: usize = 64;
const VALUE_LENGTH: usize = 4;

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
    pub quantized: bool,
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

#[derive(Debug, Default)]
pub(crate) struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o_proj: Weight,
    pub ffn_norm: Vec<f32>,
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
    /// o_proj 全为零：注意力子层的输出恒为零，前向时整块跳过，也不写入这一层的 KV Cache。
    pub attn_skipped: bool,
    /// down_proj 全为零：MLP 子层的输出恒为零，前向时整块跳过。
    pub mlp_skipped: bool,
}

impl LayerWeights {
    /// 根据输出投影是否全为零，标记可以整块跳过的子层。删去子层可用 nncoss-edit 把输出投影置零。
    pub(crate) fn refresh_skips(&mut self) {
        self.attn_skipped = self.o_proj.is_zero();
        self.mlp_skipped = self.down_proj.is_zero();
    }
}

#[derive(Debug)]
pub struct Model {
    pub(crate) config: Config,
    pub(crate) token_embedding: Weight,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) lm_head: Option<Weight>,
    /// 按列存放的权重副本，第一次启用输入稀疏化时建立（见 sparse.rs）。
    pub(crate) columns: OnceLock<Option<ColumnWeights>>,
}

impl Model {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut loader = ModelLoader::new(bytes.len());
        loader.write(bytes)?;
        loader.finish()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn head(&self) -> &Weight {
        self.lm_head.as_ref().unwrap_or(&self.token_embedding)
    }

    /// 按列存放的权重副本；只有 Q4 模型才有。建立一次约占与 Q4 权重相同的内存。
    pub(crate) fn column_weights(&self) -> Option<&ColumnWeights> {
        self.columns
            .get_or_init(|| ColumnWeights::build(self))
            .as_ref()
    }
}

fn parse_header(bytes: &[u8]) -> Result<Config> {
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
        quantized: flags & FLAG_Q4_WEIGHTS != 0,
    };
    config.validate()?;
    Ok(config)
}

fn area(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error::InvalidModel("张量尺寸溢出"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    TokenEmbedding,
    AttnNorm,
    QProj,
    KProj,
    VProj,
    QBias,
    KBias,
    VBias,
    QNorm,
    KNorm,
    OProj,
    FfnNorm,
    GateProj,
    UpProj,
    DownProj,
    FinalNorm,
    LmHead,
}

impl Slot {
    fn shape(self, config: &Config) -> (usize, usize) {
        let hidden = config.hidden_size;
        match self {
            Self::TokenEmbedding | Self::LmHead => (config.vocab_size, hidden),
            Self::AttnNorm | Self::FfnNorm | Self::FinalNorm => (1, hidden),
            Self::QProj => (config.q_dim(), hidden),
            Self::KProj | Self::VProj => (config.kv_dim(), hidden),
            Self::QBias => (1, config.q_dim()),
            Self::KBias | Self::VBias => (1, config.kv_dim()),
            Self::QNorm | Self::KNorm => (1, config.head_dim),
            Self::OProj => (hidden, config.q_dim()),
            Self::GateProj | Self::UpProj => (config.intermediate_size, hidden),
            Self::DownProj => (hidden, config.intermediate_size),
        }
    }

    fn value_count(self, config: &Config) -> Result<usize> {
        let (rows, columns) = self.shape(config);
        area(rows, columns)
    }

    fn is_matrix(self) -> bool {
        matches!(
            self,
            Self::TokenEmbedding
                | Self::QProj
                | Self::KProj
                | Self::VProj
                | Self::OProj
                | Self::GateProj
                | Self::UpProj
                | Self::DownProj
                | Self::LmHead
        )
    }

    fn byte_count(self, config: &Config) -> Result<usize> {
        let (rows, columns) = self.shape(config);
        if config.quantized && self.is_matrix() {
            q4_matrix_bytes(rows, columns).ok_or(Error::InvalidModel("Q4 张量尺寸溢出"))
        } else {
            self.value_count(config)?
                .checked_mul(VALUE_LENGTH)
                .ok_or(Error::InvalidModel("张量尺寸溢出"))
        }
    }

    fn next(self, config: &Config, finished_layers: usize) -> Option<Self> {
        let after_projections = if config.qk_norm {
            Self::QNorm
        } else {
            Self::OProj
        };
        Some(match self {
            Self::TokenEmbedding => Self::AttnNorm,
            Self::AttnNorm => Self::QProj,
            Self::QProj => Self::KProj,
            Self::KProj => Self::VProj,
            Self::VProj if config.qkv_bias => Self::QBias,
            Self::VProj => after_projections,
            Self::QBias => Self::KBias,
            Self::KBias => Self::VBias,
            Self::VBias => after_projections,
            Self::QNorm => Self::KNorm,
            Self::KNorm => Self::OProj,
            Self::OProj => Self::FfnNorm,
            Self::FfnNorm => Self::GateProj,
            Self::GateProj => Self::UpProj,
            Self::UpProj => Self::DownProj,
            Self::DownProj if finished_layers < config.layer_count => Self::AttnNorm,
            Self::DownProj => Self::FinalNorm,
            Self::FinalNorm if config.tied_embeddings => return None,
            Self::FinalNorm => Self::LmHead,
            Self::LmHead => return None,
        })
    }
}

impl Slot {
    fn name(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "embed_tokens",
            Self::AttnNorm => "input_layernorm",
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::QBias => "q_bias",
            Self::KBias => "k_bias",
            Self::VBias => "v_bias",
            Self::QNorm => "q_norm",
            Self::KNorm => "k_norm",
            Self::OProj => "o_proj",
            Self::FfnNorm => "post_attention_layernorm",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
            Self::FinalNorm => "norm",
            Self::LmHead => "lm_head",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    pub kind: &'static str,
    pub layer: Option<usize>,
    pub offset: usize,
    pub byte_count: usize,
    pub rows: usize,
    pub columns: usize,
    pub quantized: bool,
}

pub fn tensor_layout(bytes: &[u8]) -> Result<(Config, Vec<TensorLayout>)> {
    let header = bytes
        .get(..HEADER_LENGTH)
        .ok_or(Error::InvalidModel("模型数据被截断"))?;
    let config = parse_header(header)?;
    let mut tensors = Vec::new();
    let mut offset = HEADER_LENGTH;
    let mut finished = 0;
    let mut slot = Some(Slot::TokenEmbedding);
    while let Some(current) = slot {
        let (rows, columns) = current.shape(&config);
        let byte_count = current.byte_count(&config)?;
        let global = matches!(
            current,
            Slot::TokenEmbedding | Slot::FinalNorm | Slot::LmHead
        );
        tensors.push(TensorLayout {
            kind: current.name(),
            layer: (!global).then_some(finished),
            offset,
            byte_count,
            rows,
            columns,
            quantized: config.quantized && current.is_matrix(),
        });
        offset = offset
            .checked_add(byte_count)
            .ok_or(Error::InvalidModel("模型长度溢出"))?;
        if current == Slot::DownProj {
            finished += 1;
        }
        slot = current.next(&config, finished);
    }
    Ok((config, tensors))
}

#[derive(Debug)]
struct Tensor {
    slot: Slot,
    byte_count: usize,
    rows: usize,
    columns: usize,
    quantized: bool,
    values: Vec<u8>,
}

impl Tensor {
    fn open(slot: Slot, config: &Config, position: usize, total: usize) -> Result<Self> {
        let byte_count = slot.byte_count(config)?;
        let end = position
            .checked_add(byte_count)
            .ok_or(Error::InvalidModel("模型长度溢出"))?;
        if end > total {
            return Err(Error::InvalidModel("模型数据被截断"));
        }
        let (rows, columns) = slot.shape(config);
        let mut values = Vec::new();
        values
            .try_reserve_exact(byte_count)
            .map_err(|_| Error::Allocation)?;
        Ok(Self {
            slot,
            byte_count,
            rows,
            columns,
            quantized: config.quantized && slot.is_matrix(),
            values,
        })
    }

    fn missing(&self) -> usize {
        self.byte_count - self.values.len()
    }

    fn push(&mut self, bytes: &[u8]) {
        self.values.extend_from_slice(bytes);
    }

    fn into_weight(self) -> Result<Weight> {
        if self.quantized && self.rows * self.columns != 0 {
            let matrix = crate::q4::Q4Matrix::new(self.rows, self.columns, self.values)?;
            return Ok(Weight::Q4(matrix));
        }
        let mut values = Vec::with_capacity(self.byte_count / VALUE_LENGTH);
        for chunk in self.values.chunks_exact(VALUE_LENGTH) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !value.is_finite() {
                return Err(Error::InvalidModel("权重包含非有限值"));
            }
            values.push(value);
        }
        Ok(Weight::F32(values))
    }
}

#[derive(Debug)]
struct Assembly {
    config: Config,
    token_embedding: Weight,
    layers: Vec<LayerWeights>,
    layer: LayerWeights,
    final_norm: Vec<f32>,
    lm_head: Option<Weight>,
    tensor: Option<Tensor>,
}

impl Assembly {
    fn open(config: Config, position: usize, total: usize) -> Result<Self> {
        let tensor = Tensor::open(Slot::TokenEmbedding, &config, position, total)?;
        Ok(Self {
            config,
            token_embedding: Weight::default(),
            layers: Vec::new(),
            layer: LayerWeights::default(),
            final_norm: Vec::new(),
            lm_head: None,
            tensor: Some(tensor),
        })
    }

    fn store(&mut self, slot: Slot, values: Weight) -> Result<()> {
        match slot {
            Slot::TokenEmbedding => self.token_embedding = values,
            Slot::AttnNorm => self.layer.attn_norm = values.into_f32(),
            Slot::QProj => self.layer.q_proj = values,
            Slot::KProj => self.layer.k_proj = values,
            Slot::VProj => self.layer.v_proj = values,
            Slot::QBias => self.layer.q_bias = Some(values.into_f32()),
            Slot::KBias => self.layer.k_bias = Some(values.into_f32()),
            Slot::VBias => self.layer.v_bias = Some(values.into_f32()),
            Slot::QNorm => self.layer.q_norm = Some(values.into_f32()),
            Slot::KNorm => self.layer.k_norm = Some(values.into_f32()),
            Slot::OProj => self.layer.o_proj = values,
            Slot::FfnNorm => self.layer.ffn_norm = values.into_f32(),
            Slot::GateProj => self.layer.gate_proj = values,
            Slot::UpProj => self.layer.up_proj = values,
            Slot::DownProj => {
                self.layer.down_proj = values;
                self.layers.try_reserve(1).map_err(|_| Error::Allocation)?;
                self.layers.push(std::mem::take(&mut self.layer));
            }
            Slot::FinalNorm => self.final_norm = values.into_f32(),
            Slot::LmHead => self.lm_head = Some(values),
        }
        Ok(())
    }

    fn close_tensor(&mut self, position: usize, total: usize) -> Result<()> {
        let Some(tensor) = self.tensor.take() else {
            return Ok(());
        };
        let slot = tensor.slot;
        let values = tensor.into_weight()?;
        self.store(slot, values)?;
        match slot.next(&self.config, self.layers.len()) {
            Some(slot) => {
                self.tensor = Some(Tensor::open(slot, &self.config, position, total)?);
            }
            None if position != total => {
                return Err(Error::InvalidModel("模型末尾存在多余数据"));
            }
            None => {}
        }
        Ok(())
    }

    fn into_model(self) -> Result<Model> {
        if self.tensor.is_some() {
            return Err(Error::InvalidModel("模型数据被截断"));
        }
        let mut layers = self.layers;
        layers.iter_mut().for_each(LayerWeights::refresh_skips);
        Ok(Model {
            config: self.config,
            token_embedding: self.token_embedding,
            layers,
            final_norm: self.final_norm,
            lm_head: self.lm_head,
            columns: OnceLock::new(),
        })
    }
}

#[derive(Debug)]
enum LoaderState {
    Header,
    Tensors(Box<Assembly>),
    Failed(Error),
}

#[derive(Debug)]
pub struct ModelLoader {
    total: usize,
    position: usize,
    carry: [u8; HEADER_LENGTH],
    carry_length: usize,
    state: LoaderState,
}

impl ModelLoader {
    pub fn new(total_length: usize) -> Self {
        Self {
            total: total_length,
            position: 0,
            carry: [0; HEADER_LENGTH],
            carry_length: 0,
            state: LoaderState::Header,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        let result = self.consume(bytes);
        if let Err(error) = &result {
            self.state = LoaderState::Failed(error.clone());
        }
        result
    }

    pub fn finish(self) -> Result<Model> {
        match self.state {
            LoaderState::Failed(error) => Err(error),
            LoaderState::Header => {
                parse_header(&self.carry[..self.carry_length])?;
                Err(Error::InvalidModel("模型数据被截断"))
            }
            LoaderState::Tensors(assembly) => assembly.into_model(),
        }
    }

    fn consume(&mut self, mut bytes: &[u8]) -> Result<()> {
        if let LoaderState::Failed(error) = &self.state {
            return Err(error.clone());
        }
        if bytes.len() > self.total - self.position {
            return Err(Error::InvalidArgument("写入长度超过声明的模型长度"));
        }
        if matches!(self.state, LoaderState::Header) {
            bytes = self.fill_carry(bytes, HEADER_LENGTH);
            if self.carry_length < HEADER_LENGTH {
                return Ok(());
            }
            let config = parse_header(&self.carry)?;
            self.carry_length = 0;
            let assembly = Assembly::open(config, self.position, self.total)?;
            self.state = LoaderState::Tensors(Box::new(assembly));
        }
        while !bytes.is_empty() {
            if self.carry_length > 0 || bytes.len() < VALUE_LENGTH {
                bytes = self.fill_carry(bytes, VALUE_LENGTH);
                if self.carry_length < VALUE_LENGTH {
                    return Ok(());
                }
                self.carry_length = 0;
                let value = self.carry;
                self.push_values(&value[..VALUE_LENGTH])?;
            } else {
                let missing = self.missing_values()?;
                let length = (missing.min(bytes.len()) / VALUE_LENGTH) * VALUE_LENGTH;
                let (values, rest) = bytes.split_at(length);
                self.position += length;
                self.push_values(values)?;
                bytes = rest;
            }
        }
        Ok(())
    }

    fn fill_carry<'a>(&mut self, bytes: &'a [u8], target: usize) -> &'a [u8] {
        let length = (target - self.carry_length).min(bytes.len());
        let (head, rest) = bytes.split_at(length);
        self.carry[self.carry_length..self.carry_length + length].copy_from_slice(head);
        self.carry_length += length;
        self.position += length;
        rest
    }

    fn missing_values(&self) -> Result<usize> {
        match &self.state {
            LoaderState::Tensors(assembly) => match &assembly.tensor {
                Some(tensor) => Ok(tensor.missing()),
                None => Err(Error::InvalidModel("模型末尾存在多余数据")),
            },
            _ => Err(Error::InvalidArgument("模型加载器状态无效")),
        }
    }

    fn push_values(&mut self, bytes: &[u8]) -> Result<()> {
        let (position, total) = (self.position, self.total);
        let LoaderState::Tensors(assembly) = &mut self.state else {
            return Err(Error::InvalidArgument("模型加载器状态无效"));
        };
        let Some(tensor) = &mut assembly.tensor else {
            return Err(Error::InvalidModel("模型末尾存在多余数据"));
        };
        tensor.push(bytes);
        if tensor.missing() == 0 {
            assembly.close_tensor(position, total)?;
        }
        Ok(())
    }
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
    fn tensor_layout_covers_every_byte_in_loader_order() {
        use crate::test_utils::TinySpec;
        for spec in TinySpec::wide_variants() {
            let bytes = spec.bytes();
            let (config, tensors) = tensor_layout(&bytes).unwrap();
            assert_eq!(config, *spec.model().config());
            assert_eq!(tensors[0].kind, "embed_tokens");
            assert_eq!(tensors[0].offset, HEADER_LENGTH);
            for pair in tensors.windows(2) {
                assert_eq!(pair[0].offset + pair[0].byte_count, pair[1].offset);
            }
            let last = tensors.last().unwrap();
            assert_eq!(last.offset + last.byte_count, bytes.len());
            assert_eq!(last.kind, if spec.tied { "norm" } else { "lm_head" });
            let downs: Vec<_> = tensors
                .iter()
                .filter(|t| t.kind == "down_proj")
                .map(|t| t.layer)
                .collect();
            assert_eq!(downs, (0..spec.layers).map(Some).collect::<Vec<_>>());
            assert!(tensors.iter().all(|t| !t.quantized));
        }
        assert!(tensor_layout(&tiny_model_bytes()[..10]).is_err());
    }

    #[test]
    fn rejects_invalid_head_combination() {
        let mut bytes = tiny_model_bytes();
        bytes[36..40].copy_from_slice(&4_u32.to_le_bytes());
        assert!(Model::from_bytes(&bytes).is_err());
    }

    fn patched(offset: usize, value: [u8; 4]) -> Result<Model> {
        let mut bytes = tiny_model_bytes();
        bytes[offset..offset + 4].copy_from_slice(&value);
        Model::from_bytes(&bytes)
    }

    #[test]
    fn rejects_unsupported_header_fields() {
        assert_eq!(
            patched(8, 2_u32.to_le_bytes()).unwrap_err(),
            Error::Unsupported("模型格式版本")
        );
        assert_eq!(
            patched(12, 2_u32.to_le_bytes()).unwrap_err(),
            Error::Unsupported("模型架构")
        );
        assert_eq!(
            patched(48, (6_u32 | 16).to_le_bytes()).unwrap_err(),
            Error::Unsupported("模型 flags")
        );
        assert_eq!(
            patched(52, 2_u32.to_le_bytes()).unwrap_err(),
            Error::Unsupported("RoPE 模式")
        );
    }

    #[test]
    fn rejects_invalid_dimensions_and_scalars() {
        for offset in (16..48).step_by(4) {
            assert!(
                matches!(
                    patched(offset, 0_u32.to_le_bytes()),
                    Err(Error::InvalidModel(_))
                ),
                "zero field at offset {offset} must be rejected"
            );
        }
        assert!(matches!(
            patched(40, 3_u32.to_le_bytes()),
            Err(Error::InvalidModel(_))
        ));
        for value in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                patched(56, value.to_le_bytes()),
                Err(Error::InvalidModel(_))
            ));
            assert!(matches!(
                patched(60, value.to_le_bytes()),
                Err(Error::InvalidModel(_))
            ));
        }
    }

    #[test]
    fn rejects_non_finite_weights() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                patched(64, value.to_le_bytes()).unwrap_err(),
                Error::InvalidModel("权重包含非有限值")
            );
        }
    }

    #[test]
    fn huge_layer_count_fails_as_truncated_model() {
        assert_eq!(
            patched(28, u32::MAX.to_le_bytes()).unwrap_err(),
            Error::InvalidModel("模型数据被截断")
        );
    }

    #[test]
    fn tied_models_omit_the_lm_head_tensor() {
        use crate::test_utils::TinySpec;
        let spec = TinySpec {
            tied: true,
            ..TinySpec::WIDE
        };
        let untied = TinySpec::WIDE.bytes();
        let tied = spec.bytes();
        assert_eq!(untied.len() - tied.len(), spec.vocab * spec.hidden * 4);
        assert!(spec.model().lm_head.is_none());
        assert!(TinySpec::WIDE.model().lm_head.is_some());
    }

    const TRUNCATED: Error = Error::InvalidModel("模型数据被截断");

    fn weight_bytes(model: &Model) -> Vec<u8> {
        let mut tensors: Vec<Vec<f32>> = vec![model.token_embedding.clone().into_f32()];
        for layer in &model.layers {
            tensors.extend([
                layer.attn_norm.clone(),
                layer.q_proj.clone().into_f32(),
                layer.k_proj.clone().into_f32(),
                layer.v_proj.clone().into_f32(),
            ]);
            let optional = [
                &layer.q_bias,
                &layer.k_bias,
                &layer.v_bias,
                &layer.q_norm,
                &layer.k_norm,
            ];
            tensors.extend(optional.into_iter().filter_map(|tensor| tensor.clone()));
            tensors.extend([
                layer.o_proj.clone().into_f32(),
                layer.ffn_norm.clone(),
                layer.gate_proj.clone().into_f32(),
                layer.up_proj.clone().into_f32(),
                layer.down_proj.clone().into_f32(),
            ]);
        }
        tensors.push(model.final_norm.clone());
        tensors.extend(model.lm_head.clone().map(Weight::into_f32));
        tensors
            .into_iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn load_chunked(bytes: &[u8], total: usize, chunk: usize) -> Result<Model> {
        let mut loader = ModelLoader::new(total);
        for part in bytes.chunks(chunk) {
            loader.write(part)?;
        }
        loader.finish()
    }

    #[test]
    fn chunked_loading_reproduces_every_tensor_for_all_variants() {
        use crate::test_utils::TinySpec;
        let mut specs = TinySpec::wide_variants();
        specs.push(TinySpec::BASE);
        for spec in specs {
            let bytes = spec.bytes();
            let expected = spec.model();
            for chunk in [1, 2, 3, 4, 5, 7, 63, 64, 65, 1024, bytes.len()] {
                let model = load_chunked(&bytes, bytes.len(), chunk).unwrap();
                assert_eq!(model.config, expected.config);
                assert_eq!(model.layers.len(), spec.layers);
                assert_eq!(weight_bytes(&model), &bytes[HEADER_LENGTH..]);
            }
        }
    }

    #[test]
    fn every_two_part_split_matches_single_write() {
        let bytes = tiny_model_bytes();
        for split in 0..=bytes.len() {
            let mut loader = ModelLoader::new(bytes.len());
            loader.write(&bytes[..split]).unwrap();
            assert_eq!(loader.position(), split);
            loader.write(&bytes[split..]).unwrap();
            assert_eq!(loader.position(), bytes.len());
            let model = loader.finish().unwrap();
            assert_eq!(weight_bytes(&model), &bytes[HEADER_LENGTH..]);
        }
    }

    #[test]
    fn truncated_streams_fail_at_every_length() {
        let bytes = tiny_model_bytes();
        for length in 0..bytes.len() {
            let prefix = &bytes[..length];
            for chunk in [1, 5, 64, bytes.len()] {
                assert_eq!(
                    load_chunked(prefix, bytes.len(), chunk).unwrap_err(),
                    TRUNCATED,
                    "stream of {length} bytes in chunks of {chunk}"
                );
                assert_eq!(
                    load_chunked(prefix, length, chunk).unwrap_err(),
                    TRUNCATED,
                    "file of {length} bytes in chunks of {chunk}"
                );
            }
        }
    }

    #[test]
    fn stream_errors_are_independent_of_chunking() {
        let bytes = tiny_model_bytes();
        let last = bytes.len() - 4;
        let patches = [
            (0, 0_u32, Error::InvalidModel("magic 不匹配")),
            (8, 2, Error::Unsupported("模型格式版本")),
            (12, 2, Error::Unsupported("模型架构")),
            (16, 0, Error::InvalidModel("模型维度必须为非零值")),
            (48, 22, Error::Unsupported("模型 flags")),
            (52, 2, Error::Unsupported("RoPE 模式")),
            (
                64,
                f32::NAN.to_bits(),
                Error::InvalidModel("权重包含非有限值"),
            ),
            (
                last,
                f32::INFINITY.to_bits(),
                Error::InvalidModel("权重包含非有限值"),
            ),
        ];
        let mut cases: Vec<(Vec<u8>, Error)> = patches
            .into_iter()
            .map(|(offset, value, error)| {
                let mut case = bytes.clone();
                case[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                (case, error)
            })
            .collect();
        let mut trailing = bytes.clone();
        trailing.push(0);
        cases.push((trailing, Error::InvalidModel("模型末尾存在多余数据")));
        for (case, expected) in cases {
            assert_eq!(Model::from_bytes(&case).unwrap_err(), expected);
            for chunk in [1, 3, 4, 64, 100] {
                assert_eq!(
                    load_chunked(&case, case.len(), chunk).unwrap_err(),
                    expected
                );
            }
        }
    }

    #[test]
    fn partial_header_reports_field_errors_before_truncation() {
        let mut bytes = tiny_model_bytes();
        bytes[0] ^= 1;
        let mut loader = ModelLoader::new(bytes.len());
        loader.write(&bytes[..10]).unwrap();
        assert_eq!(
            loader.finish().unwrap_err(),
            Error::InvalidModel("magic 不匹配")
        );
        assert_eq!(
            Model::from_bytes(&bytes[..10]).unwrap_err(),
            Error::InvalidModel("magic 不匹配")
        );
        assert_eq!(Model::from_bytes(&bytes[..7]).unwrap_err(), TRUNCATED);
    }

    #[test]
    fn non_finite_value_is_reported_when_its_tensor_completes() {
        let mut bytes = tiny_model_bytes();
        bytes[64..68].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut loader = ModelLoader::new(bytes.len());
        loader.write(&bytes[..70]).unwrap();
        assert_eq!(loader.finish().unwrap_err(), TRUNCATED);
        let mut loader = ModelLoader::new(bytes.len());
        loader.write(&bytes[..66]).unwrap();
        assert_eq!(
            loader.write(&bytes[66..]).unwrap_err(),
            Error::InvalidModel("权重包含非有限值")
        );
    }

    #[test]
    fn failed_loader_keeps_reporting_its_first_error() {
        let bytes = tiny_model_bytes();
        let mut loader = ModelLoader::new(bytes.len());
        loader.write(&bytes[..100]).unwrap();
        let error = loader.write(&bytes[99..]).unwrap_err();
        assert_eq!(error, Error::InvalidArgument("写入长度超过声明的模型长度"));
        assert_eq!(loader.position(), 100);
        assert_eq!(loader.write(&[]).unwrap_err(), error);
        assert_eq!(loader.finish().unwrap_err(), error);

        let mut invalid = bytes;
        invalid[0] ^= 1;
        let mut loader = ModelLoader::new(invalid.len());
        let error = loader.write(&invalid[..64]).unwrap_err();
        assert_eq!(error, Error::InvalidModel("magic 不匹配"));
        assert_eq!(loader.write(&invalid[64..]).unwrap_err(), error);
        assert_eq!(loader.finish().unwrap_err(), error);
    }

    #[test]
    fn declared_length_rejects_oversized_headers_before_allocating() {
        let mut bytes = tiny_model_bytes();
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut loader = ModelLoader::new(bytes.len());
        assert_eq!(loader.write(&bytes[..64]).unwrap_err(), TRUNCATED);
        assert_eq!(Model::from_bytes(&bytes).unwrap_err(), TRUNCATED);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn unsatisfiable_tensor_allocation_fails_without_aborting() {
        let mut bytes = tiny_model_bytes();
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        bytes[20..24].copy_from_slice(&(1_u32 << 20).to_le_bytes());
        let mut loader = ModelLoader::new(usize::MAX);
        assert_eq!(loader.write(&bytes[..64]).unwrap_err(), Error::Allocation);
        assert_eq!(loader.finish().unwrap_err(), Error::Allocation);
    }
}
