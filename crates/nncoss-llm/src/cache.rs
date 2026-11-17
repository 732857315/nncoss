use crate::math::{f16_to_f32, f32_to_f16};
use crate::{Error, Result};

pub(crate) trait KvValue: Copy + Default + Send + Sync + 'static {
    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl KvValue for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }

    fn to_f32(self) -> f32 {
        self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct F16(pub u16);

impl KvValue for F16 {
    fn from_f32(value: f32) -> Self {
        Self(f32_to_f16(value))
    }

    fn to_f32(self) -> f32 {
        f16_to_f32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvPrecision {
    F32,
    F16,
}

#[derive(Debug)]
struct LayerCache<T> {
    keys: Vec<T>,
    values: Vec<T>,
    capacity: usize,
}

impl<T: KvValue> LayerCache<T> {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            capacity: 0,
        }
    }

    fn reserve(
        &mut self,
        minimum: usize,
        heads: usize,
        head_dim: usize,
        maximum: usize,
    ) -> Result<()> {
        if minimum <= self.capacity {
            return Ok(());
        }
        let doubled = self.capacity.saturating_mul(2);
        let capacity = doubled.max(16).min(maximum).max(minimum);
        let length = capacity
            .checked_mul(heads)
            .and_then(|length| length.checked_mul(head_dim))
            .ok_or(Error::InvalidArgument("KV Cache 尺寸溢出"))?;
        let mut keys = Vec::new();
        keys.try_reserve_exact(length)
            .map_err(|_| Error::Allocation)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| Error::Allocation)?;
        keys.resize(length, T::default());
        values.resize(length, T::default());
        let old = self.capacity * head_dim;
        let new = capacity * head_dim;
        for head in 0..heads.min(self.keys.len().checked_div(old).unwrap_or(0)) {
            keys[head * new..head * new + old]
                .copy_from_slice(&self.keys[head * old..(head + 1) * old]);
            values[head * new..head * new + old]
                .copy_from_slice(&self.values[head * old..(head + 1) * old]);
        }
        self.keys = keys;
        self.values = values;
        self.capacity = capacity;
        Ok(())
    }

    fn write(&mut self, position: usize, keys: &[f32], values: &[f32], head_dim: usize) {
        for (head, (keys, values)) in keys
            .chunks_exact(head_dim)
            .zip(values.chunks_exact(head_dim))
            .enumerate()
        {
            let start = (head * self.capacity + position) * head_dim;
            for (slot, key) in self.keys[start..start + head_dim].iter_mut().zip(keys) {
                *slot = T::from_f32(*key);
            }
            for (slot, value) in self.values[start..start + head_dim].iter_mut().zip(values) {
                *slot = T::from_f32(*value);
            }
        }
    }

    fn view(&self) -> KvView<'_, T> {
        KvView {
            keys: &self.keys,
            values: &self.values,
            capacity: self.capacity,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct KvView<'a, T> {
    pub keys: &'a [T],
    pub values: &'a [T],
    pub capacity: usize,
}

impl<'a, T> KvView<'a, T> {
    pub fn head(&self, head: usize, count: usize, head_dim: usize) -> (&'a [T], &'a [T]) {
        let start = head * self.capacity * head_dim;
        let end = start + count * head_dim;
        (&self.keys[start..end], &self.values[start..end])
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum KvLayer<'a> {
    F32(KvView<'a, f32>),
    F16(KvView<'a, F16>),
}

#[derive(Debug)]
enum Layers {
    F32(Vec<LayerCache<f32>>),
    F16(Vec<LayerCache<F16>>),
}

#[derive(Debug)]
pub(crate) struct KvCache {
    layers: Layers,
    heads: usize,
    head_dim: usize,
    len: usize,
}

impl KvCache {
    pub fn new(layer_count: usize, heads: usize, head_dim: usize, precision: KvPrecision) -> Self {
        let layers = match precision {
            KvPrecision::F32 => Layers::F32((0..layer_count).map(|_| LayerCache::new()).collect()),
            KvPrecision::F16 => Layers::F16((0..layer_count).map(|_| LayerCache::new()).collect()),
        };
        Self {
            layers,
            heads,
            head_dim,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn write(
        &mut self,
        layer: usize,
        position: usize,
        keys: &[f32],
        values: &[f32],
        maximum: usize,
    ) -> Result<()> {
        let kv_dim = self.heads * self.head_dim;
        if keys.len() != kv_dim || values.len() != kv_dim {
            return Err(Error::InvalidArgument("K/V 维度不一致"));
        }
        if position >= maximum {
            return Err(Error::InvalidArgument("KV Cache 位置超出上限"));
        }
        let (heads, head_dim) = (self.heads, self.head_dim);
        match &mut self.layers {
            Layers::F32(layers) => {
                layers[layer].reserve(position + 1, heads, head_dim, maximum)?;
                layers[layer].write(position, keys, values, head_dim);
            }
            Layers::F16(layers) => {
                layers[layer].reserve(position + 1, heads, head_dim, maximum)?;
                layers[layer].write(position, keys, values, head_dim);
            }
        }
        Ok(())
    }

    pub fn layer(&self, layer: usize) -> KvLayer<'_> {
        match &self.layers {
            Layers::F32(layers) => KvLayer::F32(layers[layer].view()),
            Layers::F16(layers) => KvLayer::F16(layers[layer].view()),
        }
    }

    pub fn finish_tokens(&mut self, count: usize) {
        self.len += count;
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    #[cfg(test)]
    pub fn append(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        maximum: usize,
    ) -> Result<()> {
        self.write(layer, self.len, keys, values, maximum)
    }

    #[cfg(test)]
    pub fn finish_token(&mut self) {
        self.finish_tokens(1);
    }

    #[cfg(test)]
    fn entry(&self, layer: usize, position: usize, head: usize, values: bool) -> Vec<f32> {
        fn read<T: KvValue>(
            view: KvView<'_, T>,
            position: usize,
            head: usize,
            head_dim: usize,
            values: bool,
        ) -> Vec<f32> {
            let (keys, value_rows) = view.head(head, position + 1, head_dim);
            let rows = if values { value_rows } else { keys };
            rows[position * head_dim..(position + 1) * head_dim]
                .iter()
                .map(|value| value.to_f32())
                .collect()
        }
        match self.layer(layer) {
            KvLayer::F32(view) => read(view, position, head, self.head_dim, values),
            KvLayer::F16(view) => read(view, position, head, self.head_dim, values),
        }
    }

    #[cfg(test)]
    fn capacity(&self, layer: usize) -> usize {
        match self.layer(layer) {
            KvLayer::F32(view) => view.capacity,
            KvLayer::F16(view) => view.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_grows_without_losing_history_and_reuses_capacity_after_reset() {
        for precision in [KvPrecision::F32, KvPrecision::F16] {
            let mut cache = KvCache::new(1, 2, 2, precision);
            for token in 0..40 {
                let value = token as f32;
                cache
                    .append(
                        0,
                        &[value, 1.0, -value, 0.5],
                        &[value, 2.0, 3.0, -value],
                        64,
                    )
                    .unwrap();
                cache.finish_token();
            }
            assert_eq!(cache.len(), 40);
            assert_eq!(cache.entry(0, 7, 0, false), vec![7.0, 1.0]);
            assert_eq!(cache.entry(0, 7, 1, false), vec![-7.0, 0.5]);
            assert_eq!(cache.entry(0, 39, 0, true), vec![39.0, 2.0]);
            assert_eq!(cache.entry(0, 39, 1, true), vec![3.0, -39.0]);
            assert_eq!(cache.entry(0, 3, 1, true), vec![3.0, -3.0]);
            let capacity = cache.capacity(0);
            cache.reset();
            cache
                .append(0, &[9.0, 1.0, 4.0, 4.0], &[8.0, 2.0, 5.0, 5.0], 64)
                .unwrap();
            assert_eq!(cache.capacity(0), capacity);
            assert_eq!(cache.entry(0, 0, 0, false), vec![9.0, 1.0]);
            assert_eq!(cache.entry(0, 0, 1, true), vec![5.0, 5.0]);
        }
    }

    #[test]
    fn f16_storage_rounds_values() {
        let mut cache = KvCache::new(1, 1, 2, KvPrecision::F16);
        cache
            .append(0, &[0.1, 70000.0], &[1.0 / 3.0, -2.5], 4)
            .unwrap();
        assert_eq!(
            cache.entry(0, 0, 0, false),
            vec![f16_to_f32(f32_to_f16(0.1)), f32::INFINITY]
        );
        assert_eq!(cache.entry(0, 0, 0, true)[1], -2.5);
        assert!((cache.entry(0, 0, 0, true)[0] - 1.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn oversized_growth_reports_allocation_error_without_mutation() {
        let mut layer = LayerCache::<f32>::new();
        let result = layer.reserve(1, 1, usize::MAX / 32, usize::MAX);
        assert_eq!(result, Err(Error::Allocation));
        assert_eq!(layer.capacity, 0);
        assert!(layer.keys.is_empty() && layer.values.is_empty());
    }

    #[test]
    fn mismatched_dimensions_are_rejected() {
        let mut cache = KvCache::new(1, 2, 2, KvPrecision::F32);
        assert!(cache.append(0, &[1.0; 4], &[1.0; 3], 8).is_err());
        assert!(cache.append(0, &[1.0; 2], &[1.0; 2], 8).is_err());
        assert!(cache.write(0, 8, &[1.0; 4], &[1.0; 4], 8).is_err());
        assert_eq!(cache.capacity(0), 0);
    }

    #[test]
    fn truncate_rewinds_positions_and_keeps_capacity() {
        let mut cache = KvCache::new(1, 1, 1, KvPrecision::F32);
        for token in 0..5 {
            cache
                .append(0, &[token as f32], &[token as f32], 8)
                .unwrap();
            cache.finish_token();
        }
        let capacity = cache.capacity(0);
        cache.truncate(2);
        assert_eq!(cache.len(), 2);
        cache.truncate(10);
        assert_eq!(cache.len(), 2);
        cache.append(0, &[7.0], &[7.0], 8).unwrap();
        cache.finish_token();
        assert_eq!(cache.entry(0, 2, 0, false), vec![7.0]);
        assert_eq!(cache.entry(0, 1, 0, false), vec![1.0]);
        assert_eq!(cache.capacity(0), capacity);
    }
}
