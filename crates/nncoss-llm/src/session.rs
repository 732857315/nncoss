use std::sync::Arc;

use crate::cache::{KvCache, KvLayer, KvPrecision};
use crate::model::Model;
use crate::ops::{
    AttentionShape, add_assign, apply_rope_table, attention, hadamard_heads, rms_norm_heads,
    rms_norm_into, rope_frequencies, rope_table, swiglu,
};
use crate::q4::{Activation, Linear, linear_into, linear_many};
use crate::sampler::Sampler;
use crate::{Error, Result, SamplingConfig};

const PREFILL_BLOCK: usize = 32;
const DRAFT_NGRAM: usize = 3;
const DRAFT_MIN_NGRAM: usize = 2;

#[derive(Debug, Default)]
struct Scratch {
    state: Vec<f32>,
    normalized: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    attended: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    block_logits: Vec<f32>,
    scores: Vec<f32>,
    activations: Vec<Activation>,
    rope_frequencies: Vec<f32>,
    rope: Vec<(f32, f32)>,
}

impl Scratch {
    fn new(model: &Model) -> Self {
        let config = &model.config;
        Self {
            hidden: vec![0.0; config.hidden_size],
            rope_frequencies: rope_frequencies(config.head_dim, config.rope_theta),
            ..Self::default()
        }
    }

    fn reserve(&mut self, model: &Model, tokens: usize) {
        let config = &model.config;
        let hidden = config.hidden_size;
        for (buffer, width) in [
            (&mut self.state, hidden),
            (&mut self.normalized, hidden),
            (&mut self.query, config.q_dim()),
            (&mut self.key, config.kv_dim()),
            (&mut self.value, config.kv_dim()),
            (&mut self.attended, config.q_dim()),
            (&mut self.projected, hidden),
            (&mut self.gate, config.intermediate_size),
            (&mut self.up, config.intermediate_size),
        ] {
            buffer.resize(tokens * width, 0.0);
        }
        self.activations.resize_with(tokens, Activation::default);
    }
}

#[derive(Debug)]
pub struct Session {
    model: Arc<Model>,
    cache: KvCache,
    max_context: usize,
    logits: Vec<f32>,
    has_logits: bool,
    sampler: Sampler,
    scratch: Scratch,
    history: Vec<u32>,
    speculation: bool,
    drafted: usize,
    accepted: usize,
    #[cfg(test)]
    fail_at_position: Option<usize>,
}

impl Session {
    pub fn new(model: Arc<Model>, max_context: usize) -> Result<Self> {
        let maximum = if max_context == 0 {
            model.config.max_context
        } else {
            max_context
        };
        if maximum > model.config.max_context {
            return Err(Error::ContextLength {
                requested: maximum,
                maximum: model.config.max_context,
            });
        }
        let precision = if model.config.quantized {
            KvPrecision::F16
        } else {
            KvPrecision::F32
        };
        Ok(Self {
            cache: KvCache::new(
                model.config.layer_count,
                model.config.kv_head_count,
                model.config.head_dim,
                precision,
            ),
            scratch: Scratch::new(&model),
            model,
            max_context: maximum,
            logits: Vec::new(),
            has_logits: false,
            sampler: Sampler::new(SamplingConfig::default(), 0)?,
            history: Vec::new(),
            speculation: true,
            drafted: 0,
            accepted: 0,
            #[cfg(test)]
            fail_at_position: None,
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn position(&self) -> usize {
        self.cache.len()
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        self.history.clear();
        self.has_logits = false;
    }

    /// 更换 KV Cache 的存储精度并清空会话。Q4 模型默认 f16，f32 模型默认 f32。
    pub fn set_kv_precision(&mut self, precision: KvPrecision) {
        let config = &self.model.config;
        self.cache = KvCache::new(
            config.layer_count,
            config.kv_head_count,
            config.head_dim,
            precision,
        );
        self.reset();
    }

    pub fn kv_precision(&self) -> KvPrecision {
        self.cache.precision()
    }

    pub fn set_speculation(&mut self, enabled: bool) {
        self.speculation = enabled;
    }

    pub fn speculation_counts(&self) -> (usize, usize) {
        (self.drafted, self.accepted)
    }

    pub fn set_sampler(&mut self, config: SamplingConfig, seed: u64) -> Result<()> {
        self.sampler = Sampler::new(config, seed)?;
        Ok(())
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::InvalidArgument("prefill tokens 不能为空"));
        }
        self.validate_tokens(tokens)?;
        let start = self.cache.len();
        for block in tokens.chunks(PREFILL_BLOCK) {
            if let Err(error) = self.forward(block, false) {
                self.rewind_to(start);
                return Err(error);
            }
        }
        self.commit_hidden();
        Ok(())
    }

    pub fn decode(&mut self, token: u32) -> Result<()> {
        self.validate_tokens(&[token])?;
        self.forward(&[token], false)?;
        self.commit_hidden();
        Ok(())
    }

    pub fn last_logits(&self) -> Result<&[f32]> {
        if self.has_logits {
            Ok(&self.logits)
        } else {
            Err(Error::NoLogits)
        }
    }

    pub fn sample_next(&mut self) -> Result<u32> {
        if !self.has_logits {
            return Err(Error::NoLogits);
        }
        self.sampler.sample(&self.logits)
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        self.prefill(prompt)?;
        let mut generated = Vec::with_capacity(max_new_tokens);
        if max_new_tokens == 0 {
            return Ok(generated);
        }
        let mut pending = self.sample_next()?;
        loop {
            if eos_tokens.contains(&pending) {
                break;
            }
            generated.push(pending);
            let draft = if generated.len() < max_new_tokens {
                self.draft(pending)
            } else {
                None
            };
            let Some(draft) = draft else {
                self.decode(pending)?;
                if generated.len() == max_new_tokens {
                    break;
                }
                pending = self.sample_next()?;
                continue;
            };
            self.verify(pending, draft)?;
            self.drafted += 1;
            let vocab = self.model.config.vocab_size;
            let (accepted, token) = self
                .sampler
                .sample_with_draft(&self.scratch.block_logits[..vocab], draft)?;
            if !accepted || eos_tokens.contains(&draft) {
                self.rewind_to(self.cache.len() - 1);
                self.use_block_logits(0);
                if accepted {
                    break;
                }
                pending = token;
                continue;
            }
            self.accepted += 1;
            generated.push(draft);
            self.use_block_logits(1);
            if generated.len() == max_new_tokens {
                break;
            }
            pending = self.sample_next()?;
        }
        Ok(generated)
    }

    fn draft(&self, last: u32) -> Option<u32> {
        if !self.speculation || self.position() + 2 > self.max_context {
            return None;
        }
        draft_token(&self.history, last)
    }

    fn verify(&mut self, token: u32, draft: u32) -> Result<()> {
        self.validate_tokens(&[token, draft])?;
        self.forward(&[token, draft], true)?;
        let vocab = self.model.config.vocab_size;
        let hidden = self.model.config.hidden_size;
        let Scratch {
            hidden: states,
            block_logits,
            activations,
            ..
        } = &mut self.scratch;
        block_logits.clear();
        block_logits.resize(2 * vocab, 0.0);
        linear_many(
            &states[..2 * hidden],
            &mut activations[..2],
            &mut [Linear {
                weight: self.model.head(),
                bias: None,
                output: block_logits,
            }],
        );
        Ok(())
    }

    fn use_block_logits(&mut self, row: usize) {
        let vocab = self.model.config.vocab_size;
        self.logits.clear();
        self.logits
            .extend_from_slice(&self.scratch.block_logits[row * vocab..(row + 1) * vocab]);
        self.has_logits = true;
    }

    fn rewind_to(&mut self, length: usize) {
        self.cache.truncate(length);
        self.history.truncate(length);
    }

    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
        let requested = self
            .position()
            .checked_add(tokens.len())
            .ok_or(Error::ContextLength {
                requested: usize::MAX,
                maximum: self.max_context,
            })?;
        if requested > self.max_context {
            return Err(Error::ContextLength {
                requested,
                maximum: self.max_context,
            });
        }
        for token in tokens {
            if *token as usize >= self.model.config.vocab_size {
                return Err(Error::TokenOutOfRange(*token));
            }
        }
        Ok(())
    }

    fn commit_hidden(&mut self) {
        let config = &self.model.config;
        self.logits.clear();
        self.logits.resize(config.vocab_size, 0.0);
        let hidden = self.model.config.hidden_size;
        linear_into(
            self.model.head(),
            &self.scratch.hidden[..hidden],
            &mut self.logits,
        );
        self.has_logits = true;
    }

    fn forward(&mut self, tokens: &[u32], every: bool) -> Result<()> {
        #[cfg(test)]
        let fail_at_position = self.fail_at_position;
        let Self {
            model,
            cache,
            scratch,
            max_context,
            history,
            ..
        } = self;
        let config = &model.config;
        let count = tokens.len();
        let start = cache.len();
        let (hidden, q_dim, kv_dim) = (config.hidden_size, config.q_dim(), config.kv_dim());
        let half = config.head_dim / 2;
        // int8 缓存按行缩放，键里少数离群维度会吃掉其余维度的精度。对 Q、K、V 做同一个
        // 正交 Hadamard 旋转把离群值摊开：Q·K 不变，注意力输出再旋转回来即可。
        let rotate = cache.precision() == KvPrecision::I8 && config.head_dim.is_power_of_two();
        let shape = AttentionShape {
            head_count: config.head_count,
            kv_head_count: config.kv_head_count,
            head_dim: config.head_dim,
        };
        scratch.reserve(model, count);
        scratch.rope.clear();
        let mut table = Vec::with_capacity(half);
        for offset in 0..count {
            rope_table(&scratch.rope_frequencies, start + offset, &mut table);
            scratch.rope.extend_from_slice(&table);
        }
        for (token, state) in tokens.iter().zip(scratch.state.chunks_exact_mut(hidden)) {
            model.token_embedding.row_into(*token as usize, state);
        }
        for (layer_index, layer) in model.layers.iter().enumerate() {
            // 输出投影全为零的子层对残差没有贡献，跳过它与结果逐位相同（见 LayerWeights::refresh_skips）。
            if !layer.attn_skipped {
                normalize(
                    &scratch.state,
                    &layer.attn_norm,
                    config.norm_eps,
                    &mut scratch.normalized,
                    hidden,
                );
                linear_many(
                    &scratch.normalized[..count * hidden],
                    &mut scratch.activations[..count],
                    &mut [
                        Linear {
                            weight: &layer.q_proj,
                            bias: layer.q_bias.as_deref(),
                            output: &mut scratch.query[..count * q_dim],
                        },
                        Linear {
                            weight: &layer.k_proj,
                            bias: layer.k_bias.as_deref(),
                            output: &mut scratch.key[..count * kv_dim],
                        },
                        Linear {
                            weight: &layer.v_proj,
                            bias: layer.v_bias.as_deref(),
                            output: &mut scratch.value[..count * kv_dim],
                        },
                    ],
                );
                for offset in 0..count {
                    let query = &mut scratch.query[offset * q_dim..(offset + 1) * q_dim];
                    let key = &mut scratch.key[offset * kv_dim..(offset + 1) * kv_dim];
                    let value = &mut scratch.value[offset * kv_dim..(offset + 1) * kv_dim];
                    if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
                        rms_norm_heads(
                            query,
                            config.head_count,
                            config.head_dim,
                            q_norm,
                            config.norm_eps,
                        );
                        rms_norm_heads(
                            key,
                            config.kv_head_count,
                            config.head_dim,
                            k_norm,
                            config.norm_eps,
                        );
                    }
                    let table = &scratch.rope[offset * half..(offset + 1) * half];
                    apply_rope_table(
                        query,
                        config.head_count,
                        config.head_dim,
                        table,
                        config.rope_mode,
                    );
                    apply_rope_table(
                        key,
                        config.kv_head_count,
                        config.head_dim,
                        table,
                        config.rope_mode,
                    );
                    if rotate {
                        hadamard_heads(query, config.head_dim);
                        hadamard_heads(key, config.head_dim);
                        hadamard_heads(value, config.head_dim);
                    }
                    cache.write(layer_index, start + offset, key, value, *max_context)?;
                }
                #[cfg(test)]
                if layer_index + 1 == model.layers.len()
                    && fail_at_position
                        .is_some_and(|position| (start..start + count).contains(&position))
                {
                    return Err(Error::Allocation);
                }
                let queries = &scratch.query[..count * q_dim];
                let attended = &mut scratch.attended[..count * q_dim];
                match cache.layer(layer_index) {
                    KvLayer::F32(view) => {
                        attention(queries, view, shape, start, &mut scratch.scores, attended)
                    }
                    KvLayer::F16(view) => {
                        attention(queries, view, shape, start, &mut scratch.scores, attended)
                    }
                    KvLayer::I8(view) => {
                        attention(queries, view, shape, start, &mut scratch.scores, attended)
                    }
                }
                if rotate {
                    hadamard_heads(attended, config.head_dim);
                }
                linear_many(
                    &scratch.attended[..count * q_dim],
                    &mut scratch.activations[..count],
                    &mut [Linear {
                        weight: &layer.o_proj,
                        bias: None,
                        output: &mut scratch.projected[..count * hidden],
                    }],
                );
                add_assign(
                    &mut scratch.state[..count * hidden],
                    &scratch.projected[..count * hidden],
                );
            }
            if !layer.mlp_skipped {
                normalize(
                    &scratch.state,
                    &layer.ffn_norm,
                    config.norm_eps,
                    &mut scratch.normalized,
                    hidden,
                );
                let inner = count * config.intermediate_size;
                linear_many(
                    &scratch.normalized[..count * hidden],
                    &mut scratch.activations[..count],
                    &mut [
                        Linear {
                            weight: &layer.gate_proj,
                            bias: None,
                            output: &mut scratch.gate[..inner],
                        },
                        Linear {
                            weight: &layer.up_proj,
                            bias: None,
                            output: &mut scratch.up[..inner],
                        },
                    ],
                );
                swiglu(&mut scratch.gate[..inner], &scratch.up[..inner]);
                linear_many(
                    &scratch.gate[..inner],
                    &mut scratch.activations[..count],
                    &mut [Linear {
                        weight: &layer.down_proj,
                        bias: None,
                        output: &mut scratch.projected[..count * hidden],
                    }],
                );
                add_assign(
                    &mut scratch.state[..count * hidden],
                    &scratch.projected[..count * hidden],
                );
            }
        }
        cache.finish_tokens(count);
        history.extend_from_slice(tokens);
        let first = if every { 0 } else { count - 1 };
        scratch.hidden.resize((count - first).max(1) * hidden, 0.0);
        normalize(
            &scratch.state[first * hidden..count * hidden],
            &model.final_norm,
            config.norm_eps,
            &mut scratch.hidden,
            hidden,
        );
        Ok(())
    }
}

fn draft_token(history: &[u32], last: u32) -> Option<u32> {
    let length = history.len() + 1;
    let at = |index: usize| history.get(index).copied().unwrap_or(last);
    for size in (DRAFT_MIN_NGRAM..=DRAFT_NGRAM).rev() {
        if length <= size {
            continue;
        }
        let suffix = length - size;
        'search: for start in (0..suffix).rev() {
            for offset in 0..size {
                if at(start + offset) != at(suffix + offset) {
                    continue 'search;
                }
            }
            return Some(at(start + size));
        }
    }
    None
}

fn normalize(input: &[f32], weight: &[f32], epsilon: f32, output: &mut [f32], width: usize) {
    for (input, output) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        rms_norm_into(input, weight, epsilon, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RopeMode;
    use crate::ops::{apply_rope, rms_norm, silu, softmax};
    use crate::q4::linear;
    use crate::test_utils::{TinySpec, tiny_model};

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
    }

    #[test]
    fn cached_decode_matches_uncached_reference() {
        let model = Arc::new(tiny_model());
        let tokens = [1, 2, 3, 0];
        let expected = reference_logits(&model, &tokens);
        let mut prefill = Session::new(model.clone(), 0).unwrap();
        prefill.prefill(&tokens).unwrap();
        assert_close(prefill.last_logits().unwrap(), &expected);
        let mut incremental = Session::new(model, 0).unwrap();
        incremental.prefill(&tokens[..2]).unwrap();
        incremental.decode(tokens[2]).unwrap();
        incremental.decode(tokens[3]).unwrap();
        assert_close(incremental.last_logits().unwrap(), &expected);
        assert_eq!(incremental.position(), tokens.len());
    }

    #[test]
    fn zeroed_sublayers_are_skipped_bit_for_bit() {
        use crate::model::LayerWeights;
        use crate::q4::test_support::encode;
        use crate::q4::{Q4Matrix, Weight};

        fn zero(weight: &mut Weight, rows: usize, columns: usize) {
            *weight = match weight {
                Weight::F32(values) => Weight::F32(vec![0.0; values.len()]),
                Weight::Q4(_) => Weight::Q4(
                    Q4Matrix::new(
                        rows,
                        columns,
                        encode(&vec![0.0; rows * columns], rows, columns),
                    )
                    .unwrap(),
                ),
            };
        }
        let tokens = [1, 2, 3, 0, 4, 2];
        for quantized in [false, true] {
            let build = |skip: bool| {
                let mut model = if quantized {
                    TinySpec::WIDE.quantized_model()
                } else {
                    TinySpec::WIDE.model()
                };
                let config = model.config.clone();
                let (hidden, inner) = (config.hidden_size, config.intermediate_size);
                zero(&mut model.layers[0].o_proj, hidden, config.q_dim());
                zero(&mut model.layers[1].down_proj, hidden, inner);
                if skip {
                    model
                        .layers
                        .iter_mut()
                        .for_each(LayerWeights::refresh_skips);
                }
                Arc::new(model)
            };
            let skipped = build(true);
            assert!(skipped.layers[0].attn_skipped && !skipped.layers[0].mlp_skipped);
            assert!(skipped.layers[1].mlp_skipped && !skipped.layers[1].attn_skipped);
            let computed = build(false);
            assert!(!computed.layers[0].attn_skipped && !computed.layers[1].mlp_skipped);
            let run = |model: Arc<Model>| {
                let mut session = Session::new(model, 0).unwrap();
                session.prefill(&tokens[..3]).unwrap();
                for token in &tokens[3..] {
                    session.decode(*token).unwrap();
                }
                session.last_logits().unwrap().to_vec()
            };
            let (fast, full) = (run(skipped), run(computed));
            assert_eq!(
                fast.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                full.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "quantized={quantized}"
            );
        }
    }

    #[test]
    fn int8_kv_cache_tracks_full_precision_logits() {
        let model = Arc::new(tiny_model());
        let tokens = [1, 2, 3, 0, 4, 2, 1];
        let expected = reference_logits(&model, &tokens);
        let mut session = Session::new(model, 0).unwrap();
        session.set_kv_precision(KvPrecision::I8);
        assert_eq!(session.kv_precision(), KvPrecision::I8);
        session.prefill(&tokens[..3]).unwrap();
        for token in &tokens[3..] {
            session.decode(*token).unwrap();
        }
        let actual = session.last_logits().unwrap();
        let spread = expected
            .iter()
            .fold(0.0_f32, |peak, value| peak.max(value.abs()));
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() < 0.02 * spread,
                "{actual} != {expected}"
            );
        }
        assert_ne!(
            actual,
            expected.as_slice(),
            "int8 缓存应当有量化误差，否则没有生效"
        );
        session.set_kv_precision(KvPrecision::F16);
        assert_eq!(session.position(), 0);
    }

    #[test]
    fn reset_restarts_positions_and_logits() {
        let model = Arc::new(tiny_model());
        let mut session = Session::new(model, 0).unwrap();
        session.prefill(&[1, 2]).unwrap();
        let expected = session.last_logits().unwrap().to_vec();
        session.reset();
        assert_eq!(session.position(), 0);
        assert!(session.last_logits().is_err());
        session.prefill(&[1, 2]).unwrap();
        assert_close(session.last_logits().unwrap(), &expected);
    }

    #[test]
    fn validates_context_and_tokens_before_mutating_state() {
        let model = Arc::new(tiny_model());
        let mut session = Session::new(model, 2).unwrap();
        assert!(session.prefill(&[1, 99]).is_err());
        assert_eq!(session.position(), 0);
        session.prefill(&[1, 2]).unwrap();
        assert!(session.decode(3).is_err());
        assert_eq!(session.position(), 2);
    }

    #[test]
    fn generation_stops_before_eos() {
        let model = Arc::new(tiny_model());
        let mut probe = Session::new(model.clone(), 0).unwrap();
        probe.prefill(&[1]).unwrap();
        let first = probe.sample_next().unwrap();
        let mut session = Session::new(model, 0).unwrap();
        let generated = session.generate(&[1], 4, &[first]).unwrap();
        assert!(generated.is_empty());
        assert_eq!(session.position(), 1);
    }

    #[test]
    fn generation_commits_last_token_before_continuation() {
        let model = Arc::new(tiny_model());
        let mut session = Session::new(model.clone(), 0).unwrap();
        let generated = session.generate(&[1], 2, &[]).unwrap();
        assert_eq!(generated.len(), 2);
        assert_eq!(session.position(), 3);
        let expected = reference_logits(&model, &[1, generated[0], generated[1]]);
        assert_close(session.last_logits().unwrap(), &expected);
        session.prefill(&[2]).unwrap();
        let expected = reference_logits(&model, &[1, generated[0], generated[1], 2]);
        assert_close(session.last_logits().unwrap(), &expected);
    }

    #[test]
    fn every_wide_variant_matches_uncached_reference() {
        let tokens = [3, 7, 1, 10, 0, 5];
        for spec in TinySpec::wide_variants() {
            let model = Arc::new(spec.model());
            let expected = reference_logits(&model, &tokens);
            let mut prefill = Session::new(model.clone(), 0).unwrap();
            prefill.prefill(&tokens).unwrap();
            assert_close(prefill.last_logits().unwrap(), &expected);
            let mut incremental = Session::new(model, 0).unwrap();
            incremental.prefill(&tokens[..2]).unwrap();
            for token in &tokens[2..] {
                incremental.decode(*token).unwrap();
            }
            assert_close(incremental.last_logits().unwrap(), &expected);
        }
    }

    const LONG: TinySpec = TinySpec {
        vocab: 23,
        hidden: 64,
        intermediate: 96,
        heads: 4,
        kv_heads: 2,
        head_dim: 16,
        max_context: 80,
        ..TinySpec::WIDE
    };

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn batched_prefill_matches_token_by_token_decode_bit_for_bit() {
        let tokens: Vec<u32> = (0..70).map(|index| (index * 7 + 3) % 23).collect();
        for model in [LONG.model(), LONG.quantized_model()] {
            let model = Arc::new(model);
            let mut batched = Session::new(model.clone(), 0).unwrap();
            batched.prefill(&tokens).unwrap();
            let mut stepped = Session::new(model.clone(), 0).unwrap();
            stepped.prefill(&tokens[..1]).unwrap();
            for token in &tokens[1..] {
                stepped.decode(*token).unwrap();
            }
            assert_eq!(batched.position(), tokens.len());
            assert_eq!(
                bits(batched.last_logits().unwrap()),
                bits(stepped.last_logits().unwrap())
            );
            let mut split = Session::new(model, 0).unwrap();
            split.prefill(&tokens[..5]).unwrap();
            split.prefill(&tokens[5..]).unwrap();
            assert_eq!(
                bits(split.last_logits().unwrap()),
                bits(stepped.last_logits().unwrap())
            );
        }
    }

    fn repetitive_prompt() -> Vec<u32> {
        let phrase = [3, 7, 11, 2, 19, 5];
        phrase.iter().cycle().take(30).copied().collect()
    }

    #[test]
    fn speculative_greedy_generation_matches_plain_generation_exactly() {
        for model in [LONG.model(), LONG.quantized_model()] {
            let model = Arc::new(model);
            let prompt = repetitive_prompt();
            let mut plain = Session::new(model.clone(), 0).unwrap();
            plain.set_speculation(false);
            let expected = plain.generate(&prompt, 40, &[]).unwrap();
            let mut fast = Session::new(model, 0).unwrap();
            let actual = fast.generate(&prompt, 40, &[]).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(fast.position(), plain.position());
            assert_eq!(
                bits(fast.last_logits().unwrap()),
                bits(plain.last_logits().unwrap())
            );
            let (drafted, accepted) = fast.speculation_counts();
            assert!(drafted > 0 && accepted > 0, "{drafted} {accepted}");
            assert_eq!(plain.speculation_counts(), (0, 0));
            let more = fast.generate(&[2], 5, &[]).unwrap();
            assert_eq!(more, plain.generate(&[2], 5, &[]).unwrap());
        }
    }

    #[test]
    fn speculative_sampling_leaves_a_consistent_session() {
        let model = Arc::new(LONG.quantized_model());
        let prompt = repetitive_prompt();
        for seed in 0..4 {
            let mut session = Session::new(model.clone(), 0).unwrap();
            session
                .set_sampler(
                    SamplingConfig {
                        temperature: 0.9,
                        top_k: 8,
                        top_p: 0.95,
                    },
                    seed,
                )
                .unwrap();
            let generated = session.generate(&prompt, 30, &[]).unwrap();
            assert_eq!(generated.len(), 30);
            assert_eq!(session.position(), prompt.len() + 30);
            let mut replay = Session::new(model.clone(), 0).unwrap();
            replay.prefill(&prompt).unwrap();
            replay.prefill(&generated).unwrap();
            assert_eq!(
                bits(session.last_logits().unwrap()),
                bits(replay.last_logits().unwrap())
            );
            assert_eq!(session.history, replay.history);
        }
    }

    #[test]
    fn speculative_generation_handles_limits_and_eos() {
        let model = Arc::new(LONG.model());
        let prompt = repetitive_prompt();
        let mut session = Session::new(model.clone(), 0).unwrap();
        assert!(session.generate(&prompt, 0, &[]).unwrap().is_empty());
        assert_eq!(session.position(), prompt.len());
        let mut plain = Session::new(model.clone(), 0).unwrap();
        plain.set_speculation(false);
        let reference = plain.generate(&prompt, 12, &[]).unwrap();
        for limit in 1..12 {
            let mut session = Session::new(model.clone(), 0).unwrap();
            let generated = session.generate(&prompt, limit, &[]).unwrap();
            assert_eq!(generated, reference[..limit]);
            assert_eq!(session.position(), prompt.len() + limit);
        }
        for stop in 1..8 {
            let eos = reference[stop];
            if reference[..stop].contains(&eos) {
                continue;
            }
            let mut session = Session::new(model.clone(), 0).unwrap();
            let generated = session.generate(&prompt, 12, &[eos]).unwrap();
            assert_eq!(generated, reference[..stop]);
            assert_eq!(session.position(), prompt.len() + stop);
            let mut replay = Session::new(model.clone(), 0).unwrap();
            replay.prefill(&prompt).unwrap();
            replay.prefill(&generated).unwrap();
            assert_eq!(
                bits(session.last_logits().unwrap()),
                bits(replay.last_logits().unwrap())
            );
        }
    }

    #[test]
    fn draft_lookup_prefers_the_longest_recent_match() {
        assert_eq!(draft_token(&[], 4), None);
        assert_eq!(draft_token(&[4], 5), None);
        assert_eq!(draft_token(&[1, 2, 3, 9, 1, 2], 3), Some(9));
        assert_eq!(draft_token(&[5, 2, 3, 8, 1, 2], 3), Some(8));
        assert_eq!(draft_token(&[7, 3, 6, 3, 4, 2], 3), None);
        assert_eq!(draft_token(&[6, 3, 4, 7, 6], 3), Some(4));
        assert_eq!(draft_token(&[1, 1, 1], 1), Some(1));
    }

    #[test]
    fn quantized_model_tracks_f32_reference() {
        let tokens: Vec<u32> = (0..12).map(|index| (index * 5 + 1) % 23).collect();
        let expected = reference_logits(&LONG.model(), &tokens);
        let mut session = Session::new(Arc::new(LONG.quantized_model()), 0).unwrap();
        session.prefill(&tokens).unwrap();
        let actual = session.last_logits().unwrap();
        let spread = expected
            .iter()
            .fold(0.0_f32, |peak, value| peak.max(value.abs()));
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() < 0.1 * spread,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn failure_in_a_later_prefill_block_rolls_back_the_whole_call() {
        let model = Arc::new(LONG.model());
        let tokens: Vec<u32> = (0..50).map(|index| index % 23).collect();
        let mut session = Session::new(model.clone(), 0).unwrap();
        session.prefill(&[1, 2]).unwrap();
        let before = session.last_logits().unwrap().to_vec();
        session.fail_at_position = Some(2 + PREFILL_BLOCK + 3);
        assert_eq!(session.prefill(&tokens), Err(Error::Allocation));
        assert_eq!(session.position(), 2);
        assert_eq!(session.last_logits().unwrap(), &before[..]);
        session.fail_at_position = None;
        session.prefill(&tokens).unwrap();
        let mut expected = Session::new(model, 0).unwrap();
        expected.prefill(&[1, 2]).unwrap();
        for token in &tokens {
            expected.decode(*token).unwrap();
        }
        assert_eq!(
            bits(session.last_logits().unwrap()),
            bits(expected.last_logits().unwrap())
        );
    }

    #[test]
    fn rope_layout_and_theta_change_multi_token_logits() {
        let logits = |spec: TinySpec| {
            let mut session = Session::new(Arc::new(spec.model()), 0).unwrap();
            session.prefill(&[3, 7, 1, 10]).unwrap();
            session.last_logits().unwrap().to_vec()
        };
        let base = logits(TinySpec::WIDE);
        let interleaved = logits(TinySpec {
            rope_mode: RopeMode::Interleaved,
            ..TinySpec::WIDE
        });
        let low_theta = logits(TinySpec {
            rope_theta: 100.0,
            ..TinySpec::WIDE
        });
        let differs =
            |left: &[f32], right: &[f32]| left.iter().zip(right).any(|(l, r)| (l - r).abs() > 1e-5);
        assert!(differs(&base, &interleaved));
        assert!(differs(&base, &low_theta));
    }

    #[test]
    fn grouped_query_heads_read_their_shared_kv_head() {
        let mut cache = KvCache::new(1, 2, 2, KvPrecision::F32);
        cache
            .append(0, &[0.1, 0.2, 0.3, 0.4], &[1.0, 2.0, 3.0, 4.0], 4)
            .unwrap();
        let query = [0.5, -0.5, 1.0, 0.25, -1.0, 2.0, 0.0, 1.0];
        let shape = AttentionShape {
            head_count: 4,
            kv_head_count: 2,
            head_dim: 2,
        };
        let mut scores = vec![0.0; 4];
        let mut output = vec![0.0; 8];
        let KvLayer::F32(view) = cache.layer(0) else {
            unreachable!()
        };
        attention(&query, view, shape, 0, &mut scores, &mut output);
        assert_close(&output, &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn failed_prefill_rolls_back_position_and_keeps_previous_logits() {
        let model = Arc::new(TinySpec::WIDE.model());
        let mut session = Session::new(model.clone(), 0).unwrap();
        session.prefill(&[3, 7]).unwrap();
        let before = session.last_logits().unwrap().to_vec();
        session.fail_at_position = Some(3);
        assert_eq!(session.prefill(&[1, 10, 0]), Err(Error::Allocation));
        assert_eq!(session.position(), 2);
        assert_eq!(session.last_logits().unwrap(), &before[..]);
        session.fail_at_position = None;
        session.prefill(&[1, 10, 0]).unwrap();
        assert_close(
            session.last_logits().unwrap(),
            &reference_logits(&model, &[3, 7, 1, 10, 0]),
        );
    }

    #[test]
    fn failed_decode_keeps_position_and_logits() {
        let model = Arc::new(TinySpec::WIDE.model());
        let mut session = Session::new(model.clone(), 0).unwrap();
        session.prefill(&[3, 7]).unwrap();
        let before = session.last_logits().unwrap().to_vec();
        session.fail_at_position = Some(2);
        assert_eq!(session.decode(1), Err(Error::Allocation));
        assert_eq!(session.position(), 2);
        assert_eq!(session.last_logits().unwrap(), &before[..]);
        session.fail_at_position = None;
        session.decode(1).unwrap();
        assert_close(
            session.last_logits().unwrap(),
            &reference_logits(&model, &[3, 7, 1]),
        );
    }

    fn reference_logits(model: &Model, tokens: &[u32]) -> Vec<f32> {
        let config = &model.config;
        let hidden = config.hidden_size;
        let mut states: Vec<Vec<f32>> = tokens
            .iter()
            .map(|token| model.token_embedding.row(*token as usize, hidden))
            .collect();
        for layer in &model.layers {
            let mut queries = Vec::new();
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for (position, state) in states.iter().enumerate() {
                let normalized = rms_norm(state, &layer.attn_norm, config.norm_eps);
                let mut query = linear(
                    &layer.q_proj,
                    config.q_dim(),
                    hidden,
                    &normalized,
                    layer.q_bias.as_deref(),
                );
                let mut key = linear(
                    &layer.k_proj,
                    config.kv_dim(),
                    hidden,
                    &normalized,
                    layer.k_bias.as_deref(),
                );
                let value = linear(
                    &layer.v_proj,
                    config.kv_dim(),
                    hidden,
                    &normalized,
                    layer.v_bias.as_deref(),
                );
                if let (Some(q_norm), Some(k_norm)) = (&layer.q_norm, &layer.k_norm) {
                    rms_norm_heads(
                        &mut query,
                        config.head_count,
                        config.head_dim,
                        q_norm,
                        config.norm_eps,
                    );
                    rms_norm_heads(
                        &mut key,
                        config.kv_head_count,
                        config.head_dim,
                        k_norm,
                        config.norm_eps,
                    );
                }
                apply_rope(
                    &mut query,
                    config.head_count,
                    config.head_dim,
                    position,
                    config.rope_theta,
                    config.rope_mode,
                );
                apply_rope(
                    &mut key,
                    config.kv_head_count,
                    config.head_dim,
                    position,
                    config.rope_theta,
                    config.rope_mode,
                );
                queries.push(query);
                keys.push(key);
                values.push(value);
            }
            let mut next_states = Vec::new();
            for (position, previous_state) in states.iter().enumerate() {
                let attended = reference_attention(
                    config.head_count,
                    config.kv_head_count,
                    config.head_dim,
                    position,
                    &queries,
                    &keys,
                    &values,
                );
                let projected = linear(&layer.o_proj, hidden, config.q_dim(), &attended, None);
                let mut state: Vec<f32> = previous_state
                    .iter()
                    .zip(projected)
                    .map(|(left, right)| left + right)
                    .collect();
                let normalized = rms_norm(&state, &layer.ffn_norm, config.norm_eps);
                let gate = linear(
                    &layer.gate_proj,
                    config.intermediate_size,
                    hidden,
                    &normalized,
                    None,
                );
                let up = linear(
                    &layer.up_proj,
                    config.intermediate_size,
                    hidden,
                    &normalized,
                    None,
                );
                let activated: Vec<f32> = gate
                    .iter()
                    .zip(up)
                    .map(|(left, right)| silu(*left) * right)
                    .collect();
                let projected = linear(
                    &layer.down_proj,
                    hidden,
                    config.intermediate_size,
                    &activated,
                    None,
                );
                for (state, projected) in state.iter_mut().zip(projected) {
                    *state += projected;
                }
                next_states.push(state);
            }
            states = next_states;
        }
        let normalized = rms_norm(states.last().unwrap(), &model.final_norm, config.norm_eps);
        let weights = model.lm_head.as_ref().unwrap_or(&model.token_embedding);
        linear(weights, config.vocab_size, hidden, &normalized, None)
    }

    fn reference_attention(
        head_count: usize,
        kv_head_count: usize,
        head_dim: usize,
        position: usize,
        queries: &[Vec<f32>],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> Vec<f32> {
        let group_size = head_count / kv_head_count;
        let scale = (head_dim as f32).sqrt().recip();
        let mut output = Vec::new();
        for head in 0..head_count {
            let kv_head = head / group_size;
            let query = &queries[position][head * head_dim..(head + 1) * head_dim];
            let mut scores = Vec::new();
            for key in keys.iter().take(position + 1) {
                let key = &key[kv_head * head_dim..(kv_head + 1) * head_dim];
                scores.push(
                    query
                        .iter()
                        .zip(key)
                        .map(|(left, right)| left * right)
                        .sum::<f32>()
                        * scale,
                );
            }
            let probabilities = softmax(&scores);
            let mut context = vec![0.0; head_dim];
            for (value, probability) in values.iter().take(position + 1).zip(&probabilities) {
                let value = &value[kv_head * head_dim..(kv_head + 1) * head_dim];
                for (context, value) in context.iter_mut().zip(value) {
                    *context += probability * value;
                }
            }
            output.extend_from_slice(&context);
        }
        output
    }
}
