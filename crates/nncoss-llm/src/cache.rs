use crate::{Error, Result};

#[derive(Debug)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    capacity: usize,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            capacity: 0,
        }
    }

    fn reserve(&mut self, minimum: usize, kv_dim: usize, maximum: usize) -> Result<()> {
        if minimum <= self.capacity {
            return Ok(());
        }
        let doubled = self.capacity.saturating_mul(2);
        let capacity = doubled.max(16).min(maximum).max(minimum);
        let length = capacity
            .checked_mul(kv_dim)
            .ok_or(Error::InvalidArgument("KV Cache 尺寸溢出"))?;
        self.keys.resize(length, 0.0);
        self.values.resize(length, 0.0);
        self.capacity = capacity;
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct KvCache {
    layers: Vec<LayerCache>,
    len: usize,
}

impl KvCache {
    pub fn new(layer_count: usize) -> Self {
        Self {
            layers: (0..layer_count).map(|_| LayerCache::new()).collect(),
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn append(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        maximum: usize,
    ) -> Result<()> {
        let kv_dim = keys.len();
        if values.len() != kv_dim {
            return Err(Error::InvalidArgument("K/V 维度不一致"));
        }
        let cache = &mut self.layers[layer];
        cache.reserve(self.len + 1, kv_dim, maximum)?;
        let start = self.len * kv_dim;
        cache.keys[start..start + kv_dim].copy_from_slice(keys);
        cache.values[start..start + kv_dim].copy_from_slice(values);
        Ok(())
    }

    pub fn key(&self, layer: usize, position: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let kv_dim = self.layers[layer].keys.len() / self.layers[layer].capacity;
        let start = position * kv_dim + kv_head * head_dim;
        &self.layers[layer].keys[start..start + head_dim]
    }

    pub fn value(&self, layer: usize, position: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let kv_dim = self.layers[layer].values.len() / self.layers[layer].capacity;
        let start = position * kv_dim + kv_head * head_dim;
        &self.layers[layer].values[start..start + head_dim]
    }

    pub fn finish_token(&mut self) {
        self.len += 1;
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_grows_without_losing_history_and_reuses_capacity_after_reset() {
        let mut cache = KvCache::new(1);
        for token in 0..20 {
            cache
                .append(0, &[token as f32, 1.0], &[token as f32, 2.0], 32)
                .unwrap();
            cache.finish_token();
        }
        assert_eq!(cache.len(), 20);
        assert_eq!(cache.key(0, 7, 0, 2), &[7.0, 1.0]);
        assert_eq!(cache.value(0, 19, 0, 2), &[19.0, 2.0]);
        let capacity = cache.layers[0].capacity;
        cache.reset();
        cache.append(0, &[9.0, 1.0], &[8.0, 2.0], 32).unwrap();
        assert_eq!(cache.layers[0].capacity, capacity);
        assert_eq!(cache.key(0, 0, 0, 2), &[9.0, 1.0]);
    }
}
