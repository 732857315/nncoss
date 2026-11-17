use std::sync::Arc;

use crate::cache::KvCache;
use crate::model::Model;
use crate::ops::{
    AttentionShape, add_assign, apply_rope_table, attention, rms_norm_heads, rms_norm_into,
    rope_frequencies, rope_table, swiglu,
};
use crate::q4::{Activation, Linear, linear_into, linear_many};
use crate::sampler::Sampler;
use crate::{Error, Result, SamplingConfig};

const PREFILL_BLOCK: usize = 32;

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
        Ok(Self {
            cache: KvCache::new(model.config.layer_count),
            scratch: Scratch::new(&model),
            model,
            max_context: maximum,
            logits: Vec::new(),
            has_logits: false,
            sampler: Sampler::new(SamplingConfig::default(), 0)?,
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
        self.has_logits = false;
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
            if let Err(error) = self.forward(block) {
                self.cache.truncate(start);
                return Err(error);
            }
        }
        self.commit_hidden();
        Ok(())
    }

    pub fn decode(&mut self, token: u32) -> Result<()> {
        self.validate_tokens(&[token])?;
        self.forward(&[token])?;
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
        for _ in 0..max_new_tokens {
            let token = self.sample_next()?;
            if eos_tokens.contains(&token) {
                break;
            }
            generated.push(token);
            self.decode(token)?;
        }
        Ok(generated)
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
        linear_into(self.model.head(), &self.scratch.hidden, &mut self.logits);
        self.has_logits = true;
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<()> {
        #[cfg(test)]
        let fail_at_position = self.fail_at_position;
        let Self {
            model,
            cache,
            scratch,
            max_context,
            ..
        } = self;
        let config = &model.config;
        let count = tokens.len();
        let start = cache.len();
        let (hidden, q_dim, kv_dim) = (config.hidden_size, config.q_dim(), config.kv_dim());
        let half = config.head_dim / 2;
        let shape = |position: usize| AttentionShape {
            head_count: config.head_count,
            kv_head_count: config.kv_head_count,
            head_dim: config.head_dim,
            count: position + 1,
        };
        scratch.reserve(model, count);
        let longest = config.head_count * (start + count);
        if scratch.scores.len() < longest {
            scratch.scores.resize(longest.next_power_of_two(), 0.0);
        }
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
                let value = &scratch.value[offset * kv_dim..(offset + 1) * kv_dim];
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
                cache.write(layer_index, start + offset, key, value, *max_context)?;
            }
            #[cfg(test)]
            if layer_index + 1 == model.layers.len()
                && fail_at_position
                    .is_some_and(|position| (start..start + count).contains(&position))
            {
                return Err(Error::Allocation);
            }
            let (keys, values) = cache.layer(layer_index);
            for offset in 0..count {
                attention(
                    &scratch.query[offset * q_dim..(offset + 1) * q_dim],
                    keys,
                    values,
                    shape(start + offset),
                    &mut scratch.scores,
                    &mut scratch.attended[offset * q_dim..(offset + 1) * q_dim],
                );
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
        cache.finish_tokens(count);
        let last = &scratch.state[(count - 1) * hidden..count * hidden];
        rms_norm_into(
            last,
            &model.final_norm,
            config.norm_eps,
            &mut scratch.hidden,
        );
        Ok(())
    }
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
        let mut cache = KvCache::new(1);
        cache
            .append(0, &[0.1, 0.2, 0.3, 0.4], &[1.0, 2.0, 3.0, 4.0], 4)
            .unwrap();
        let query = [0.5, -0.5, 1.0, 0.25, -1.0, 2.0, 0.0, 1.0];
        let (keys, values) = cache.layer(0);
        let shape = AttentionShape {
            head_count: 4,
            kv_head_count: 2,
            head_dim: 2,
            count: 1,
        };
        let mut scores = vec![0.0; 4];
        let mut output = vec![0.0; 8];
        attention(&query, keys, values, shape, &mut scores, &mut output);
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
