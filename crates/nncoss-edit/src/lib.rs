use std::collections::BTreeSet;

use nncoss_llm::{Config, TensorLayout, tensor_layout};
use serde::{Deserialize, Serialize};

/// 头部中层数字段的偏移，见 nncoss-llm 的模型格式说明。
const LAYER_COUNT_OFFSET: usize = 28;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Mlp,
    KvHead,
    QNorm,
    KNorm,
    AttnNorm,
    FfnNorm,
    FinalNorm,
    Token,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Edit {
    Scale {
        target: Target,
        #[serde(default)]
        layer: Option<usize>,
        #[serde(default)]
        index: Vec<usize>,
        factor: f32,
    },
    /// 删除整层（注意力与 MLP 一起删除），其余张量逐字节保留。
    /// 删除后层号重新从 0 连续编号，之后的改动使用新的层号。
    DropLayers { layers: Vec<usize> },
}

pub struct ModelFile {
    bytes: Vec<u8>,
    config: Config,
    tensors: Vec<TensorLayout>,
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

fn write_f32(bytes: &mut [u8], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

impl ModelFile {
    pub fn parse(bytes: Vec<u8>) -> Result<Self, String> {
        let (config, tensors) = tensor_layout(&bytes).map_err(|error| error.to_string())?;
        let end = tensors
            .last()
            .map(|tensor| tensor.offset + tensor.byte_count)
            .unwrap_or(0);
        if end != bytes.len() {
            return Err(format!("模型长度 {} 与布局 {} 不一致", bytes.len(), end));
        }
        Ok(Self {
            bytes,
            config,
            tensors,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn tensors(&self) -> &[TensorLayout] {
        &self.tensors
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn find(&self, kind: &str, layer: Option<usize>) -> Result<TensorLayout, String> {
        self.tensors
            .iter()
            .find(|tensor| tensor.kind == kind && tensor.layer == layer)
            .cloned()
            .ok_or_else(|| match layer {
                Some(layer) => format!(
                    "第 {layer} 层没有 {kind}（模型共 {} 层）",
                    self.config.layer_count
                ),
                None => format!("模型没有 {kind}"),
            })
    }

    fn multiply(&mut self, offset: usize, factor: f32) {
        let value = read_f32(&self.bytes, offset) * factor;
        write_f32(&mut self.bytes, offset, value);
    }

    pub fn row_scale(&self, kind: &str, layer: Option<usize>, row: usize) -> Result<f32, String> {
        let tensor = self.find(kind, layer)?;
        if !tensor.quantized || row >= tensor.rows {
            return Err(format!("{kind} 不是 Q4 矩阵或行号越界"));
        }
        Ok(read_f32(&self.bytes, tensor.offset + row * 4))
    }

    fn scale_rows(
        &mut self,
        tensor: &TensorLayout,
        rows: &[usize],
        factor: f32,
    ) -> Result<usize, String> {
        if let Some(row) = rows.iter().find(|&&row| row >= tensor.rows) {
            return Err(format!(
                "{} 只有 {} 行，第 {row} 行越界",
                tensor.kind, tensor.rows
            ));
        }
        if tensor.quantized && factor < 0.0 {
            return Err("Q4 矩阵的行缩放必须是非负数".to_string());
        }
        let mut changed = 0;
        for &row in rows {
            if tensor.quantized {
                let offset = tensor.offset + row * 4;
                self.multiply(offset, factor);
                changed += 1;
            } else {
                for column in 0..tensor.columns {
                    let offset = tensor.offset + (row * tensor.columns + column) * 4;
                    self.multiply(offset, factor);
                }
                changed += tensor.columns;
            }
        }
        Ok(changed)
    }

    fn scale_elements(
        &mut self,
        tensor: &TensorLayout,
        indices: &[usize],
        factor: f32,
    ) -> Result<usize, String> {
        let length = tensor.rows * tensor.columns;
        if let Some(index) = indices.iter().find(|&&index| index >= length) {
            return Err(format!(
                "{} 只有 {length} 个元素，第 {index} 个越界",
                tensor.kind
            ));
        }
        for &index in indices {
            let offset = tensor.offset + index * 4;
            self.multiply(offset, factor);
        }
        Ok(indices.len())
    }

    /// 删除指定的层，返回被删除的数值个数。
    fn drop_layers(&mut self, layers: &[usize]) -> Result<usize, String> {
        let count = self.config.layer_count;
        if layers.is_empty() {
            return Err("没有指定要删除的层".to_string());
        }
        if let Some(layer) = layers.iter().find(|&&layer| layer >= count) {
            return Err(format!("模型只有 {count} 层，第 {layer} 层越界"));
        }
        let removed: BTreeSet<usize> = layers.iter().copied().collect();
        if removed.len() >= count {
            return Err("至少要保留一层".to_string());
        }
        let header = self
            .tensors
            .first()
            .map_or(self.bytes.len(), |tensor| tensor.offset);
        let mut bytes = self.bytes[..header].to_vec();
        let kept = u32::try_from(count - removed.len()).map_err(|error| error.to_string())?;
        bytes[LAYER_COUNT_OFFSET..LAYER_COUNT_OFFSET + 4].copy_from_slice(&kept.to_le_bytes());
        let mut values = 0;
        for tensor in &self.tensors {
            if tensor.layer.is_some_and(|layer| removed.contains(&layer)) {
                values += tensor.rows * tensor.columns;
            } else {
                bytes.extend_from_slice(
                    &self.bytes[tensor.offset..tensor.offset + tensor.byte_count],
                );
            }
        }
        *self = Self::parse(bytes)?;
        Ok(values)
    }

    pub fn apply(&mut self, edit: &Edit) -> Result<usize, String> {
        let (target, layer, index, factor) = match edit {
            Edit::DropLayers { layers } => return self.drop_layers(layers),
            Edit::Scale {
                target,
                layer,
                index,
                factor,
            } => (target, layer, index, *factor),
        };
        if !factor.is_finite() {
            return Err("缩放因子必须是有限数".to_string());
        }
        let layered =
            |layer: &Option<usize>| layer.ok_or_else(|| format!("{target:?} 需要指定 layer"));
        let all = |count: usize| -> Vec<usize> {
            if index.is_empty() {
                (0..count).collect()
            } else {
                index.clone()
            }
        };
        match target {
            Target::Mlp => {
                let tensor = self.find("up_proj", Some(layered(layer)?))?;
                self.scale_rows(&tensor, &all(tensor.rows), factor)
            }
            Target::KvHead => {
                let tensor = self.find("v_proj", Some(layered(layer)?))?;
                let width = self.config.head_dim;
                let heads = all(self.config.kv_head_count);
                if let Some(head) = heads
                    .iter()
                    .find(|&&head| head >= self.config.kv_head_count)
                {
                    return Err(format!(
                        "模型只有 {} 个 KV 头，第 {head} 个越界",
                        self.config.kv_head_count
                    ));
                }
                let rows: Vec<usize> = heads
                    .iter()
                    .flat_map(|head| head * width..(head + 1) * width)
                    .collect();
                self.scale_rows(&tensor, &rows, factor)
            }
            Target::Token => {
                let tensor = self.find("embed_tokens", None)?;
                self.scale_rows(&tensor, &all(tensor.rows), factor)
            }
            Target::QNorm | Target::KNorm | Target::AttnNorm | Target::FfnNorm => {
                let kind = match target {
                    Target::QNorm => "q_norm",
                    Target::KNorm => "k_norm",
                    Target::AttnNorm => "input_layernorm",
                    _ => "post_attention_layernorm",
                };
                let tensor = self.find(kind, Some(layered(layer)?))?;
                self.scale_elements(&tensor, &all(tensor.rows * tensor.columns), factor)
            }
            Target::FinalNorm => {
                let tensor = self.find("norm", None)?;
                self.scale_elements(&tensor, &all(tensor.rows * tensor.columns), factor)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nncoss_llm::Model;

    const FIXTURE: &[u8] = include_bytes!("../../nncoss-llm/tests/fixtures/tiny_v1.nncossllm");
    const FLAGS_OFFSET: usize = 48;

    fn values(file: &ModelFile, kind: &str, layer: Option<usize>) -> Vec<f32> {
        let tensor = file.find(kind, layer).unwrap();
        (0..tensor.rows * tensor.columns)
            .map(|index| read_f32(&file.bytes, tensor.offset + index * 4))
            .collect()
    }

    fn quantized() -> ModelFile {
        let mut header = FIXTURE[..64].to_vec();
        let flags =
            u32::from_le_bytes(header[FLAGS_OFFSET..FLAGS_OFFSET + 4].try_into().unwrap()) | 8;
        header[FLAGS_OFFSET..FLAGS_OFFSET + 4].copy_from_slice(&flags.to_le_bytes());
        let (_, tensors) = tensor_layout(&header).unwrap();
        let last = tensors.last().unwrap();
        let mut bytes = header;
        bytes.resize(last.offset + last.byte_count, 0);
        for tensor in &tensors {
            let count = if tensor.quantized {
                tensor.rows
            } else {
                tensor.rows * tensor.columns
            };
            for index in 0..count {
                write_f32(&mut bytes, tensor.offset + index * 4, 1.0);
            }
        }
        ModelFile::parse(bytes).unwrap()
    }

    #[test]
    fn f32_mlp_and_kv_edits_change_only_the_selected_rows() {
        let mut file = ModelFile::parse(FIXTURE.to_vec()).unwrap();
        let before = values(&file, "up_proj", Some(0));
        let columns = file.config().hidden_size;
        let edit = Edit::Scale {
            target: Target::Mlp,
            layer: Some(0),
            index: vec![3],
            factor: 0.0,
        };
        assert_eq!(file.apply(&edit).unwrap(), columns);
        let after = values(&file, "up_proj", Some(0));
        for (index, (old, new)) in before.iter().zip(&after).enumerate() {
            let expected = if index / columns == 3 { 0.0 } else { *old };
            assert_eq!(*new, expected);
        }
        let v_before = values(&file, "v_proj", Some(0));
        let edit = Edit::Scale {
            target: Target::KvHead,
            layer: Some(0),
            index: vec![0],
            factor: 2.0,
        };
        assert_eq!(file.apply(&edit).unwrap(), file.config().head_dim * columns);
        let v_after = values(&file, "v_proj", Some(0));
        for (old, new) in v_before
            .iter()
            .zip(&v_after)
            .take(file.config().head_dim * columns)
        {
            assert_eq!(*new, old * 2.0);
        }
        Model::from_bytes(file.bytes()).unwrap();
    }

    #[test]
    fn norm_edits_scale_single_elements() {
        let mut file = ModelFile::parse(FIXTURE.to_vec()).unwrap();
        let before = values(&file, "k_norm", Some(0));
        let edit: Edit = serde_json::from_str(
            r#"{"op": "scale", "target": "k_norm", "layer": 0, "index": [1], "factor": 0.5}"#,
        )
        .unwrap();
        assert_eq!(file.apply(&edit).unwrap(), 1);
        let after = values(&file, "k_norm", Some(0));
        assert_eq!(after[0], before[0]);
        assert_eq!(after[1], before[1] * 0.5);
        let edit: Edit =
            serde_json::from_str(r#"{"op": "scale", "target": "final_norm", "factor": 2.0}"#)
                .unwrap();
        assert_eq!(file.apply(&edit).unwrap(), file.config().hidden_size);
    }

    #[test]
    fn invalid_edits_are_rejected_without_changes() {
        let mut file = ModelFile::parse(FIXTURE.to_vec()).unwrap();
        let original = file.bytes().to_vec();
        let cases = [
            Edit::Scale {
                target: Target::Mlp,
                layer: None,
                index: vec![0],
                factor: 1.0,
            },
            Edit::Scale {
                target: Target::Mlp,
                layer: Some(7),
                index: vec![0],
                factor: 1.0,
            },
            Edit::Scale {
                target: Target::Mlp,
                layer: Some(0),
                index: vec![99],
                factor: 1.0,
            },
            Edit::Scale {
                target: Target::KvHead,
                layer: Some(0),
                index: vec![5],
                factor: 1.0,
            },
            Edit::Scale {
                target: Target::Mlp,
                layer: Some(0),
                index: vec![0],
                factor: f32::NAN,
            },
        ];
        for edit in &cases {
            assert!(file.apply(edit).is_err(), "{edit:?}");
        }
        assert_eq!(file.bytes(), original.as_slice());
        assert!(ModelFile::parse(FIXTURE[..FIXTURE.len() - 4].to_vec()).is_err());
    }

    /// 把单层的测试模型复制成多层，并给每层的 Norm 乘上不同系数，使各层内容互不相同。
    fn layered(file: &ModelFile, count: usize) -> ModelFile {
        let tensors = file.tensors();
        let first = tensors
            .iter()
            .position(|tensor| tensor.layer.is_some())
            .unwrap();
        let last = tensors
            .iter()
            .rposition(|tensor| tensor.layer.is_some())
            .unwrap();
        let copy = |bytes: &mut Vec<u8>, tensor: &TensorLayout| {
            bytes.extend_from_slice(&file.bytes()[tensor.offset..tensor.offset + tensor.byte_count])
        };
        let mut bytes = file.bytes()[..tensors[0].offset].to_vec();
        bytes[LAYER_COUNT_OFFSET..LAYER_COUNT_OFFSET + 4]
            .copy_from_slice(&u32::try_from(count).unwrap().to_le_bytes());
        tensors[..first]
            .iter()
            .for_each(|tensor| copy(&mut bytes, tensor));
        for _ in 0..count {
            tensors[first..=last]
                .iter()
                .for_each(|tensor| copy(&mut bytes, tensor));
        }
        tensors[last + 1..]
            .iter()
            .for_each(|tensor| copy(&mut bytes, tensor));
        let mut result = ModelFile::parse(bytes).unwrap();
        for layer in 0..count {
            for target in [Target::AttnNorm, Target::FfnNorm] {
                let edit = Edit::Scale {
                    target,
                    layer: Some(layer),
                    index: Vec::new(),
                    factor: (layer + 2) as f32,
                };
                result.apply(&edit).unwrap();
            }
        }
        result
    }

    #[test]
    fn dropping_layers_keeps_the_other_tensors_byte_for_byte() {
        let base = [ModelFile::parse(FIXTURE.to_vec()).unwrap(), quantized()];
        for mut file in base.iter().map(|file| layered(file, 4)) {
            let original = ModelFile::parse(file.bytes().to_vec()).unwrap();
            let edit: Edit =
                serde_json::from_str(r#"{"op": "drop_layers", "layers": [2, 0]}"#).unwrap();
            let removed = file.apply(&edit).unwrap();
            assert_eq!(file.config().layer_count, 2);
            Model::from_bytes(file.bytes()).unwrap();
            let slice = |model: &ModelFile, tensor: &TensorLayout| {
                model.bytes()[tensor.offset..tensor.offset + tensor.byte_count].to_vec()
            };
            let mut expected_removed = 0;
            for tensor in original.tensors() {
                let moved = match tensor.layer {
                    Some(0 | 2) => {
                        expected_removed += tensor.rows * tensor.columns;
                        continue;
                    }
                    Some(1) => Some(0),
                    Some(3) => Some(1),
                    other => other,
                };
                let target = file.find(tensor.kind, moved).unwrap();
                assert_eq!(
                    slice(&original, tensor),
                    slice(&file, &target),
                    "{}",
                    tensor.kind
                );
            }
            assert_eq!(removed, expected_removed);
            assert_ne!(
                slice(&file, &file.find("input_layernorm", Some(0)).unwrap()),
                slice(&file, &file.find("input_layernorm", Some(1)).unwrap())
            );
            for layers in [vec![], vec![1, 2], vec![0, 1]] {
                let before = file.bytes().to_vec();
                assert!(file.apply(&Edit::DropLayers { layers }).is_err());
                assert_eq!(file.bytes(), before.as_slice());
            }
        }
    }

    #[test]
    fn q4_edits_rewrite_row_scales_and_keep_the_model_loadable() {
        let mut file = quantized();
        Model::from_bytes(file.bytes()).unwrap();
        let edit = Edit::Scale {
            target: Target::Mlp,
            layer: Some(0),
            index: vec![2, 5],
            factor: 0.25,
        };
        assert_eq!(file.apply(&edit).unwrap(), 2);
        assert_eq!(file.row_scale("up_proj", Some(0), 2).unwrap(), 0.25);
        assert_eq!(file.row_scale("up_proj", Some(0), 1).unwrap(), 1.0);
        let negative = Edit::Scale {
            target: Target::KvHead,
            layer: Some(0),
            index: vec![0],
            factor: -1.0,
        };
        assert!(file.apply(&negative).is_err());
        Model::from_bytes(file.bytes()).unwrap();
    }
}
