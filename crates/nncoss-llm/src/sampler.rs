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
        if logits.is_empty() {
            return Err(Error::InvalidArgument("logits 不能为空"));
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFiniteLogit);
        }
        if self.config.temperature == 0.0 {
            return greedy(logits);
        }
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
        self.draw(candidates)
    }

    fn draw(&mut self, mut candidates: Vec<(u32, f32)>) -> Result<u32> {
        let best = candidates[0].0;
        for candidate in &mut candidates {
            candidate.1 /= self.config.temperature;
            if !candidate.1.is_finite() {
                return Ok(best);
            }
        }
        let maximum = candidates[0].1;
        let mut sum = 0.0;
        for candidate in &mut candidates {
            candidate.1 = crate::ops::exp(candidate.1 - maximum);
            sum += candidate.1;
        }
        let mut retained = 0;
        let mut cumulative = 0.0;
        for candidate in &candidates {
            cumulative += candidate.1 / sum;
            retained += 1;
            if cumulative >= self.config.top_p {
                break;
            }
        }
        candidates.truncate(retained);
        let retained_sum = candidates.iter().map(|candidate| candidate.1).sum::<f32>();
        let mut threshold = self.rng.next_f32() * retained_sum;
        for candidate in &candidates {
            if threshold < candidate.1 {
                return Ok(candidate.0);
            }
            threshold -= candidate.1;
        }
        Ok(candidates.last().map(|candidate| candidate.0).unwrap_or(0))
    }
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
