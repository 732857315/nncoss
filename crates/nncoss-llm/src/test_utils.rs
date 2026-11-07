use crate::Model;
use crate::model::MODEL_MAGIC;

pub(crate) fn tiny_model() -> Model {
    Model::from_bytes(&tiny_model_bytes()).unwrap()
}

pub(crate) fn tiny_model_bytes() -> Vec<u8> {
    const VOCAB: usize = 5;
    const HIDDEN: usize = 4;
    const INTERMEDIATE: usize = 8;
    const LAYERS: usize = 1;
    const HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 2;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MODEL_MAGIC);
    push_u32(&mut bytes, 1);
    push_u32(&mut bytes, 1);
    for value in [
        VOCAB,
        HIDDEN,
        INTERMEDIATE,
        LAYERS,
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        16,
    ] {
        push_u32(&mut bytes, value as u32);
    }
    push_u32(&mut bytes, 6);
    push_u32(&mut bytes, 0);
    push_f32(&mut bytes, 1e-5);
    push_f32(&mut bytes, 10_000.0);
    let mut salt = 0;
    tensor(&mut bytes, VOCAB * HIDDEN, &mut salt);
    for _ in 0..LAYERS {
        tensor(&mut bytes, HIDDEN, &mut salt);
        tensor(&mut bytes, HEADS * HEAD_DIM * HIDDEN, &mut salt);
        tensor(&mut bytes, KV_HEADS * HEAD_DIM * HIDDEN, &mut salt);
        tensor(&mut bytes, KV_HEADS * HEAD_DIM * HIDDEN, &mut salt);
        tensor(&mut bytes, HEADS * HEAD_DIM, &mut salt);
        tensor(&mut bytes, KV_HEADS * HEAD_DIM, &mut salt);
        tensor(&mut bytes, KV_HEADS * HEAD_DIM, &mut salt);
        tensor(&mut bytes, HEAD_DIM, &mut salt);
        tensor(&mut bytes, HEAD_DIM, &mut salt);
        tensor(&mut bytes, HIDDEN * HEADS * HEAD_DIM, &mut salt);
        tensor(&mut bytes, HIDDEN, &mut salt);
        tensor(&mut bytes, INTERMEDIATE * HIDDEN, &mut salt);
        tensor(&mut bytes, INTERMEDIATE * HIDDEN, &mut salt);
        tensor(&mut bytes, HIDDEN * INTERMEDIATE, &mut salt);
    }
    tensor(&mut bytes, HIDDEN, &mut salt);
    tensor(&mut bytes, VOCAB * HIDDEN, &mut salt);
    bytes
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
