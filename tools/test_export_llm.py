import unittest

from export_llm import validate_layer_features


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


if __name__ == "__main__":
    unittest.main()
