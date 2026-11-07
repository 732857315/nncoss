use std::sync::Arc;

use crate::cache::KvCache;
use crate::model::Model;
use crate::ops::{apply_rope, linear, rms_norm, rms_norm_heads, silu, softmax};
use crate::sampler::Sampler;
use crate::{Error, Result, SamplingConfig};

#[derive(Debug)]
pub struct Session {
    model: Arc<Model>,
    cache: KvCache,
    max_context: usize,
    last_logits: Option<Vec<f32>>,
    sampler: Sampler,
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
            model,
            max_context: maximum,
            last_logits: None,
            sampler: Sampler::new(SamplingConfig::default(), 0)?,
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
        self.last_logits = None;
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
        for token in tokens {
            self.forward_token(*token)?;
        }
        Ok(())
    }

    pub fn decode(&mut self, token: u32) -> Result<()> {
        self.validate_tokens(&[token])?;
        self.forward_token(token)
    }

    pub fn last_logits(&self) -> Result<&[f32]> {
        self.last_logits.as_deref().ok_or(Error::NoLogits)
    }

    pub fn sample_next(&mut self) -> Result<u32> {
        let logits = self.last_logits.as_deref().ok_or(Error::NoLogits)?;
        self.sampler.sample(logits)
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

    fn forward_token(&mut self, token: u32) -> Result<()> {
        let config = &self.model.config;
        let hidden = config.hidden_size;
        let position = self.cache.len();
        let start = token as usize * hidden;
        let mut state = self.model.token_embedding[start..start + hidden].to_vec();
        for (layer_index, layer) in self.model.layers.iter().enumerate() {
            let normalized = rms_norm(&state, &layer.attn_norm, config.norm_eps);
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
            self.cache
                .append(layer_index, &key, &value, self.max_context)?;
            let attended = attention(
                &query,
                &self.cache,
                layer_index,
                config.head_count,
                config.kv_head_count,
                config.head_dim,
                position,
            );
            let projected = linear(&layer.o_proj, hidden, config.q_dim(), &attended, None);
            for (state, projected) in state.iter_mut().zip(projected) {
                *state += projected;
            }
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
                .map(|(gate, up)| silu(*gate) * up)
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
        }
        self.cache.finish_token();
        let normalized = rms_norm(&state, &self.model.final_norm, config.norm_eps);
        let lm_head = self
            .model
            .lm_head
            .as_deref()
            .unwrap_or(&self.model.token_embedding);
        self.last_logits = Some(linear(
            lm_head,
            config.vocab_size,
            hidden,
            &normalized,
            None,
        ));
        Ok(())
    }
}

fn attention(
    query: &[f32],
    cache: &KvCache,
    layer: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    position: usize,
) -> Vec<f32> {
    let group_size = head_count / kv_head_count;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = Vec::with_capacity(head_count * head_dim);
    for head in 0..head_count {
        let kv_head = head / group_size;
        let query = &query[head * head_dim..(head + 1) * head_dim];
        let mut scores = Vec::with_capacity(position + 1);
        for index in 0..=position {
            let key = cache.key(layer, index, kv_head, head_dim);
            let score = query
                .iter()
                .zip(key)
                .map(|(query, key)| query * key)
                .sum::<f32>()
                * scale;
            scores.push(score);
        }
        let probabilities = softmax(&scores);
        let mut context = vec![0.0; head_dim];
        for (index, probability) in probabilities.iter().enumerate() {
            let value = cache.value(layer, index, kv_head, head_dim);
            for (context, value) in context.iter_mut().zip(value) {
                *context += probability * value;
            }
        }
        output.extend_from_slice(&context);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::tiny_model;

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

    fn reference_logits(model: &Model, tokens: &[u32]) -> Vec<f32> {
        let config = &model.config;
        let hidden = config.hidden_size;
        let mut states: Vec<Vec<f32>> = tokens
            .iter()
            .map(|token| {
                let start = *token as usize * hidden;
                model.token_embedding[start..start + hidden].to_vec()
            })
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
        let weights = model.lm_head.as_deref().unwrap_or(&model.token_embedding);
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
