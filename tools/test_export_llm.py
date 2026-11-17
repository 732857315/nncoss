import importlib.util
import unittest

from export_llm import quantize_q4, validate_layer_features

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class LayerFeatureValidationTests(unittest.TestCase):
    def sources(self, layers=2, qkv=True, qk=True):
        sources = {}
        for layer in range(layers):
            prefix = f"model.layers.{layer}.self_attn"
            if qkv:
                for name in ("q_proj", "k_proj", "v_proj"):
                    sources[f"{prefix}.{name}.bias"] = None
            if qk:
                for name in ("q_norm", "k_norm"):
                    sources[f"{prefix}.{name}.weight"] = None
        return sources

    def test_accepts_consistent_features(self):
        self.assertEqual(validate_layer_features(self.sources(), 2), (True, True))
        self.assertEqual(validate_layer_features(self.sources(qkv=False, qk=False), 2), (False, False))

    def test_rejects_unsupported_biases(self):
        names = (
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.0.mlp.gate_proj.bias",
            "model.layers.1.mlp.up_proj.bias",
            "model.layers.1.mlp.down_proj.bias",
            "lm_head.bias",
        )
        for name in names:
            with self.subTest(name=name):
                sources = self.sources()
                sources[name] = None
                with self.assertRaisesRegex(ValueError, "暂不支持"):
                    validate_layer_features(sources, 2)

    def test_rejects_partial_features(self):
        sources = self.sources()
        del sources["model.layers.0.self_attn.q_proj.bias"]
        with self.assertRaisesRegex(ValueError, "同时存在"):
            validate_layer_features(sources, 2)
        sources = self.sources()
        del sources["model.layers.0.self_attn.q_norm.weight"]
        with self.assertRaisesRegex(ValueError, "同时存在"):
            validate_layer_features(sources, 2)

    def test_rejects_cross_layer_mismatches(self):
        sources = self.sources()
        del sources["model.layers.1.self_attn.v_proj.bias"]
        with self.assertRaisesRegex(ValueError, "不一致"):
            validate_layer_features(sources, 2)
        sources = self.sources(qkv=False, qk=False)
        sources["model.layers.1.self_attn.q_norm.weight"] = None
        sources["model.layers.1.self_attn.k_norm.weight"] = None
        with self.assertRaisesRegex(ValueError, "不一致"):
            validate_layer_features(sources, 2)

    def test_rejects_empty_layer_count(self):
        with self.assertRaisesRegex(ValueError, "大于 0"):
            validate_layer_features({}, 0)


@unittest.skipUnless(HAS_NUMPY, "需要 numpy")
class Q4QuantizationTests(unittest.TestCase):
    def dequantize(self, row_scale, codes, packed, columns):
        import numpy as np

        unsigned = np.empty(packed.shape[:2] + (32,), dtype=np.int16)
        unsigned[:, :, 0::2] = packed & 15
        unsigned[:, :, 1::2] = packed >> 4
        steps = row_scale[:, None] * codes.view(np.int8).astype(np.float32)
        values = (unsigned - 8) * steps[:, :, None]
        return values.reshape(len(row_scale), -1)[:, :columns]

    def test_layout_and_error_bound(self):
        import numpy as np

        generator = np.random.default_rng(7)
        array = generator.standard_normal((5, 70)).astype(np.float32)
        array[1] *= 1e-3
        array[2, 40:] = 0.0
        array[3] = 0.0
        row_scale, codes, packed = quantize_q4(array)
        self.assertEqual(row_scale.dtype, np.float32)
        self.assertEqual(codes.shape, (5, 3))
        self.assertEqual(packed.shape, (5, 3, 16))
        self.assertTrue((codes.view(np.int8) != -128).all())
        restored = self.dequantize(row_scale, codes, packed, 70)
        self.assertTrue((restored[3] == 0).all())
        self.assertTrue((restored[2, 64:] == 0).all())
        for row in range(5):
            for group in range(3):
                block = array[row, group * 32:(group + 1) * 32]
                error = np.abs(restored[row, group * 32:(group + 1) * 32] - block).max()
                self.assertLessEqual(error, np.abs(block).max() / 7 + 1e-12)

    def test_search_beats_absmax_rounding(self):
        import numpy as np

        array = np.random.default_rng(11).standard_normal((64, 256)).astype(np.float32)
        row_scale, codes, packed = quantize_q4(array)
        searched = ((self.dequantize(row_scale, codes, packed, 256) - array) ** 2).sum()
        blocks = array.reshape(64, 8, 32)
        step = np.abs(blocks).max(axis=2, keepdims=True) / 7
        naive = ((np.clip(np.rint(blocks / step), -8, 7) * step - blocks) ** 2).sum()
        self.assertLess(searched, naive * 0.85)


if __name__ == "__main__":
    unittest.main()
