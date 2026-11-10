use std::sync::Arc;

use nncoss_llm::{Model, RopeMode, Session};

const ABS_TOLERANCE: f32 = 1e-4;
const REL_TOLERANCE: f32 = 1e-4;
const GOLDEN_MAGIC: &[u8; 8] = b"NNCSGLD\0";
const GOLDEN_VERSION: u32 = 1;
const ROPE_MODE_OFFSET: usize = 52;

struct Case {
    name: &'static str,
    model: &'static [u8],
    golden: &'static [u8],
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    tied: bool,
    qkv_bias: bool,
    qk_norm: bool,
}

macro_rules! case {
    ($name:literal, $layers:literal, $heads:literal, $kv_heads:literal, $head_dim:literal, $tied:literal, $qkv_bias:literal, $qk_norm:literal) => {
        Case {
            name: $name,
            model: include_bytes!(concat!("fixtures/golden/", $name, ".nncossllm")),
            golden: include_bytes!(concat!("fixtures/golden/", $name, ".golden")),
            layers: $layers,
            heads: $heads,
            kv_heads: $kv_heads,
            head_dim: $head_dim,
            tied: $tied,
            qkv_bias: $qkv_bias,
            qk_norm: $qk_norm,
        }
    };
}

const CASES: &[Case] = &[
    case!("llama_gqa_untied", 2, 4, 2, 4, false, false, false),
    case!("llama_mha_tied", 3, 2, 2, 8, true, false, false),
    case!("qwen2_gqa_tied", 2, 6, 2, 4, true, true, false),
    case!("qwen2_mqa_untied", 1, 2, 1, 8, false, true, false),
    case!("qwen3_gqa_tied", 2, 4, 2, 8, true, false, true),
    case!("qwen3_narrow_untied", 1, 2, 2, 6, false, false, true),
];

struct Golden {
    vocab: usize,
    prompt: Vec<u32>,
    greedy: Vec<u32>,
    logits: Vec<f32>,
}

impl Golden {
    fn parse(bytes: &[u8]) -> Self {
        assert_eq!(&bytes[..8], GOLDEN_MAGIC);
        let word = |index: usize| {
            let start = 8 + index * 4;
            u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
        };
        assert_eq!(word(0), GOLDEN_VERSION);
        let vocab = word(1) as usize;
        let prompt_length = word(2) as usize;
        let greedy_length = word(3) as usize;
        let count = prompt_length + greedy_length;
        assert!(vocab > 0 && prompt_length >= 2);
        assert_eq!(bytes.len(), 24 + count * 4 + count * vocab * 4);
        let tokens: Vec<u32> = (0..count).map(|index| word(4 + index)).collect();
        let logits = bytes[24 + count * 4..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Self {
            vocab,
            prompt: tokens[..prompt_length].to_vec(),
            greedy: tokens[prompt_length..].to_vec(),
            logits,
        }
    }

    fn tokens(&self) -> Vec<u32> {
        let mut tokens = self.prompt.clone();
        tokens.extend_from_slice(&self.greedy);
        tokens
    }

    fn row(&self, position: usize) -> &[f32] {
        &self.logits[position * self.vocab..(position + 1) * self.vocab]
    }
}

fn first_mismatch(actual: &[f32], expected: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| {
            let error = (**actual - **expected).abs();
            error.is_nan() || error > ABS_TOLERANCE + REL_TOLERANCE * expected.abs()
        })
        .map(|(index, (actual, expected))| (index, *actual, *expected))
}

fn assert_matches(name: &str, stage: &str, position: usize, actual: &[f32], expected: &[f32]) {
    if let Some((index, actual, expected)) = first_mismatch(actual, expected) {
        panic!(
            "{name} {stage} 位置 {position} logit[{index}]={actual} 与 PyTorch 参考 {expected} 不一致"
        );
    }
}

fn load(case: &Case) -> (Arc<Model>, Golden) {
    let model = Arc::new(Model::from_bytes(case.model).unwrap());
    (model, Golden::parse(case.golden))
}

#[test]
fn fixtures_cover_the_supported_llama_and_qwen_configurations() {
    for case in CASES {
        let (model, golden) = load(case);
        let config = model.config();
        assert_eq!(config.vocab_size, golden.vocab, "{}", case.name);
        assert_eq!(config.layer_count, case.layers, "{}", case.name);
        assert_eq!(config.head_count, case.heads, "{}", case.name);
        assert_eq!(config.kv_head_count, case.kv_heads, "{}", case.name);
        assert_eq!(config.head_dim, case.head_dim, "{}", case.name);
        assert_eq!(config.tied_embeddings, case.tied, "{}", case.name);
        assert_eq!(config.qkv_bias, case.qkv_bias, "{}", case.name);
        assert_eq!(config.qk_norm, case.qk_norm, "{}", case.name);
        assert_eq!(config.rope_mode, RopeMode::HalfSplit, "{}", case.name);
        assert!(golden.tokens().len() <= config.max_context, "{}", case.name);
        assert!(!golden.greedy.is_empty(), "{}", case.name);
    }
    for (tied, qkv_bias, qk_norm) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, true, false),
        (false, false, true),
        (true, false, true),
    ] {
        assert!(
            CASES
                .iter()
                .any(|case| (case.tied, case.qkv_bias, case.qk_norm) == (tied, qkv_bias, qk_norm))
        );
    }
    assert!(CASES.iter().any(|case| case.heads == case.kv_heads));
    assert!(
        CASES
            .iter()
            .any(|case| case.kv_heads == 1 && case.heads > 1)
    );
    assert!(CASES.iter().any(|case| {
        let hidden = Model::from_bytes(case.model).unwrap().config().hidden_size;
        case.heads * case.head_dim != hidden
    }));
}

#[test]
fn incremental_decode_matches_pytorch_logits_at_every_position() {
    for case in CASES {
        let (model, golden) = load(case);
        let tokens = golden.tokens();
        let mut session = Session::new(model, 0).unwrap();
        session.prefill(&tokens[..1]).unwrap();
        assert_matches(
            case.name,
            "增量",
            0,
            session.last_logits().unwrap(),
            golden.row(0),
        );
        for (position, token) in tokens.iter().enumerate().skip(1) {
            session.decode(*token).unwrap();
            assert_matches(
                case.name,
                "增量",
                position,
                session.last_logits().unwrap(),
                golden.row(position),
            );
        }
        assert_eq!(session.position(), tokens.len());
    }
}

#[test]
fn prefill_matches_pytorch_logits_for_every_prefix_split() {
    for case in CASES {
        let (model, golden) = load(case);
        let last = golden.prompt.len() - 1;
        for split in 0..golden.prompt.len() {
            let mut session = Session::new(model.clone(), 0).unwrap();
            if split > 0 {
                session.prefill(&golden.prompt[..split]).unwrap();
                assert_matches(
                    case.name,
                    "prefill 前缀",
                    split - 1,
                    session.last_logits().unwrap(),
                    golden.row(split - 1),
                );
            }
            session.prefill(&golden.prompt[split..]).unwrap();
            assert_matches(
                case.name,
                "prefill",
                last,
                session.last_logits().unwrap(),
                golden.row(last),
            );
        }
    }
}

#[test]
fn greedy_generation_matches_pytorch_tokens_and_logits() {
    for case in CASES {
        let (model, golden) = load(case);
        let total = golden.tokens().len();
        let mut session = Session::new(model.clone(), 0).unwrap();
        let generated = session
            .generate(&golden.prompt, golden.greedy.len(), &[])
            .unwrap();
        assert_eq!(generated, golden.greedy, "{}", case.name);
        assert_eq!(session.position(), total);
        assert_matches(
            case.name,
            "generate",
            total - 1,
            session.last_logits().unwrap(),
            golden.row(total - 1),
        );
        let mut stepped = Session::new(model, 0).unwrap();
        stepped.prefill(&golden.prompt).unwrap();
        for (step, expected) in golden.greedy.iter().enumerate() {
            assert_eq!(stepped.sample_next().unwrap(), *expected, "{}", case.name);
            stepped.decode(*expected).unwrap();
            let position = golden.prompt.len() + step;
            assert_matches(
                case.name,
                "贪心",
                position,
                stepped.last_logits().unwrap(),
                golden.row(position),
            );
        }
    }
}

#[test]
fn golden_logits_reject_the_interleaved_rope_layout() {
    for case in CASES {
        let golden = Golden::parse(case.golden);
        let mut bytes = case.model.to_vec();
        bytes[ROPE_MODE_OFFSET..ROPE_MODE_OFFSET + 4].copy_from_slice(&1_u32.to_le_bytes());
        let model = Arc::new(Model::from_bytes(&bytes).unwrap());
        assert_eq!(model.config().rope_mode, RopeMode::Interleaved);
        let mut session = Session::new(model, 0).unwrap();
        session.prefill(&golden.prompt).unwrap();
        let last = golden.prompt.len() - 1;
        assert!(
            first_mismatch(session.last_logits().unwrap(), golden.row(last)).is_some(),
            "{} 改为 Interleaved RoPE 后仍与 PyTorch 参考一致",
            case.name
        );
    }
}
