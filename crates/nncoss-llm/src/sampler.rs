use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Sampler {
    config: SamplingConfig,
    rng: SplitMix64,
}

impl Sampler {
    pub fn new(config: SamplingConfig, seed: u64) -> Result<Self> {
        if !config.temperature.is_finite() || config.temperature < 0.0 {
            return Err(Error::InvalidArgument("temperature 必须为非负有限值"));
        }
        if !config.top_p.is_finite() || config.top_p <= 0.0 || config.top_p > 1.0 {
            return Err(Error::InvalidArgument("top_p 必须在 (0, 1] 内"));
        }
        Ok(Self {
            config,
            rng: SplitMix64::new(seed),
        })
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if self.config.temperature == 0.0 {
            check(logits)?;
            return greedy(logits);
        }
        let mut candidates = self.candidates(logits)?;
        let Some(total) = self.weigh(&mut candidates) else {
            return Ok(candidates[0].0);
        };
        Ok(self.pick(&candidates, total))
    }

    pub fn sample_with_draft(&mut self, logits: &[f32], draft: u32) -> Result<(bool, u32)> {
        if self.config.temperature == 0.0 {
            check(logits)?;
            let token = greedy(logits)?;
            return Ok((token == draft, token));
        }
        let mut candidates = self.candidates(logits)?;
        let Some(total) = self.weigh(&mut candidates) else {
            let token = candidates[0].0;
            return Ok((token == draft, token));
        };
        let probability = candidates
            .iter()
            .find(|candidate| candidate.0 == draft)
            .map_or(0.0, |candidate| candidate.1 / total);
        if probability > 0.0 && self.rng.next_f32() < probability {
            return Ok((true, draft));
        }
        candidates.retain(|candidate| candidate.0 != draft);
        if candidates.is_empty() {
            return Ok((true, draft));
        }
        let total = candidates.iter().map(|candidate| candidate.1).sum::<f32>();
        Ok((false, self.pick(&candidates, total)))
    }

    fn candidates(&self, logits: &[f32]) -> Result<Vec<(u32, f32)>> {
        check(logits)?;
        let mut candidates = Vec::new();
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            select_top(logits, self.config.top_k, &mut candidates);
        } else {
            candidates.extend(
                logits
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index as u32, *value)),
            );
            candidates.sort_by(order);
        }
        Ok(candidates)
    }

    fn weigh(&self, candidates: &mut Vec<(u32, f32)>) -> Option<f32> {
        for candidate in candidates.iter_mut() {
            candidate.1 /= self.config.temperature;
            if !candidate.1.is_finite() {
                return None;
            }
        }
        let maximum = candidates[0].1;
        let mut sum = 0.0;
        for candidate in candidates.iter_mut() {
            candidate.1 = crate::ops::exp(candidate.1 - maximum);
            sum += candidate.1;
        }
        let mut retained = 0;
        let mut cumulative = 0.0;
        for candidate in candidates.iter() {
            cumulative += candidate.1 / sum;
            retained += 1;
            if cumulative >= self.config.top_p {
                break;
            }
        }
        candidates.truncate(retained);
        Some(candidates.iter().map(|candidate| candidate.1).sum::<f32>())
    }

    fn pick(&mut self, candidates: &[(u32, f32)], total: f32) -> u32 {
        let mut threshold = self.rng.next_f32() * total;
        for candidate in candidates {
            if threshold < candidate.1 {
                return candidate.0;
            }
            threshold -= candidate.1;
        }
        candidates.last().map(|candidate| candidate.0).unwrap_or(0)
    }
}

fn check(logits: &[f32]) -> Result<()> {
    if logits.is_empty() {
        return Err(Error::InvalidArgument("logits 不能为空"));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteLogit);
    }
    Ok(())
}

fn order(left: &(u32, f32), right: &(u32, f32)) -> std::cmp::Ordering {
    right.1.total_cmp(&left.1).then(left.0.cmp(&right.0))
}

fn select_top(logits: &[f32], count: usize, candidates: &mut Vec<(u32, f32)>) {
    candidates.clear();
    let mut floor = f32::NEG_INFINITY;
    for (index, value) in logits.iter().enumerate() {
        if candidates.len() == count && *value <= floor {
            continue;
        }
        let candidate = (index as u32, *value);
        let position = candidates.partition_point(|existing| order(existing, &candidate).is_lt());
        if position >= count {
            continue;
        }
        if candidates.len() == count {
            candidates.pop();
        }
        candidates.insert(position, candidate);
        if candidates.len() == count {
            floor = candidates[count - 1].1;
        }
    }
}

fn greedy(logits: &[f32]) -> Result<u32> {
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).map_err(|_| Error::InvalidArgument("词表超过 u32 范围"))
}

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1_u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_top_k_are_deterministic() {
        let logits = [0.1, 2.0, 1.0];
        let mut greedy = Sampler::new(SamplingConfig::default(), 0).unwrap();
        assert_eq!(greedy.sample(&logits).unwrap(), 1);
        let mut top_k = Sampler::new(
            SamplingConfig {
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
            },
            42,
        )
        .unwrap();
        assert_eq!(top_k.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn top_k_selection_matches_full_sort() {
        let logits: Vec<f32> = (0..500)
            .map(|index| ((index * 37 % 101) as f32 * 0.25).sin() * 4.0)
            .collect();
        let mut sorted: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(index, value)| (index as u32, *value))
            .collect();
        sorted.sort_by(order);
        for count in [1, 2, 20, 499] {
            let mut selected = Vec::new();
            select_top(&logits, count, &mut selected);
            assert_eq!(selected, sorted[..count]);
        }
    }

    #[test]
    fn draft_verification_preserves_the_sampling_distribution() {
        let logits = [1.2, 0.3, 2.0, -0.5, 1.7, 0.9];
        let config = SamplingConfig {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.95,
        };
        let trials = 200_000;
        let mut direct = [0_u32; 6];
        let mut sampler = Sampler::new(config, 3).unwrap();
        for _ in 0..trials {
            direct[sampler.sample(&logits).unwrap() as usize] += 1;
        }
        for draft in [2_u32, 4, 3] {
            let mut verified = [0_u32; 6];
            let mut accepted = 0;
            let mut sampler = Sampler::new(config, 11 + u64::from(draft)).unwrap();
            for _ in 0..trials {
                let (accept, token) = sampler.sample_with_draft(&logits, draft).unwrap();
                assert_eq!(accept, token == draft);
                accepted += u32::from(accept);
                verified[token as usize] += 1;
            }
            for token in 0..6 {
                let expected = f64::from(direct[token]) / f64::from(trials);
                let actual = f64::from(verified[token]) / f64::from(trials);
                assert!(
                    (expected - actual).abs() < 0.006,
                    "draft {draft} token {token}: {expected} vs {actual}"
                );
            }
            if draft == 3 {
                assert_eq!(accepted, 0);
            }
        }
    }

    #[test]
    fn greedy_draft_verification_matches_argmax() {
        let mut sampler = Sampler::new(SamplingConfig::default(), 0).unwrap();
        assert_eq!(
            sampler.sample_with_draft(&[0.1, 2.0, 1.0], 1).unwrap(),
            (true, 1)
        );
        assert_eq!(
            sampler.sample_with_draft(&[0.1, 2.0, 1.0], 2).unwrap(),
            (false, 1)
        );
        assert!(sampler.sample_with_draft(&[f32::NAN], 0).is_err());
    }

    #[test]
    fn sampling_is_reproducible_with_the_same_seed() {
        let config = SamplingConfig {
            temperature: 0.8,
            top_k: 3,
            top_p: 0.9,
        };
        let mut left = Sampler::new(config, 7).unwrap();
        let mut right = Sampler::new(config, 7).unwrap();
        for _ in 0..16 {
            assert_eq!(
                left.sample(&[0.2, 0.4, 0.6]).unwrap(),
                right.sample(&[0.2, 0.4, 0.6]).unwrap()
            );
        }
    }

    #[test]
    fn overflowing_temperature_scaling_falls_back_to_greedy() {
        let mut sampler = Sampler::new(
            SamplingConfig {
                temperature: f32::MIN_POSITIVE,
                top_k: 0,
                top_p: 1.0,
            },
            42,
        )
        .unwrap();
        assert_eq!(sampler.sample(&[1.0, 2.0, -1.0]).unwrap(), 1);
    }

    #[test]
    fn invalid_configuration_and_logits_are_rejected() {
        assert!(
            Sampler::new(
                SamplingConfig {
                    temperature: -1.0,
                    ..SamplingConfig::default()
                },
                0
            )
            .is_err()
        );
        assert!(
            Sampler::new(
                SamplingConfig {
                    temperature: 1.0,
                    top_p: 0.0,
                    ..SamplingConfig::default()
                },
                0
            )
            .is_err()
        );
        let mut sampler = Sampler::new(SamplingConfig::default(), 0).unwrap();
        assert!(sampler.sample(&[f32::NAN]).is_err());
    }
}
