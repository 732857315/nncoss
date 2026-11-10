import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import platform
import re
import struct
import tempfile
from pathlib import Path

import export_llm
from check_golden_llm import (
    ABS_TOLERANCE,
    GOLDEN_DIR,
    REL_TOLERANCE,
    GoldenMismatch,
    decode_golden,
    encode_golden,
    load_fixtures,
    run_case,
    within_tolerance,
)
from check_wasm_llm import ROOT, NativeTarget, WasmTarget, native_library_name

MASK = (1 << 64) - 1
PROMPT_LENGTH = 12
GREEDY_LENGTH = 6
LINEAR_GAIN = 1.5
MIN_GREEDY_MARGIN = 0.1
MAX_CACHE_ERROR = 1e-9
REGENERATION_TOLERANCE = 1e-6
HEADER_LENGTH = 64
ROPE_MODE_OFFSET = 52
ROPE_THETA_OFFSET = 60
FAMILIES = {
    "llama": ("LlamaConfig", "LlamaForCausalLM"),
    "qwen2": ("Qwen2Config", "Qwen2ForCausalLM"),
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM"),
}
CASES = (
    {
        "name": "llama_gqa_untied",
        "family": "llama",
        "prompt_seed": 5,
        "config": {
            "vocab_size": 29, "hidden_size": 16, "intermediate_size": 40, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 32,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": False,
        },
    },
    {
        "name": "llama_mha_tied",
        "family": "llama",
        "prompt_seed": 2,
        "config": {
            "vocab_size": 23, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 3,
            "num_attention_heads": 2, "num_key_value_heads": 2, "max_position_embeddings": 48,
            "rms_norm_eps": 1e-6, "rope_theta": 500000.0, "tie_word_embeddings": True,
        },
    },
    {
        "name": "qwen2_gqa_tied",
        "family": "qwen2",
        "prompt_seed": 6,
        "config": {
            "vocab_size": 31, "hidden_size": 24, "intermediate_size": 24, "num_hidden_layers": 2,
            "num_attention_heads": 6, "num_key_value_heads": 2, "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": True,
        },
    },
    {
        "name": "qwen2_mqa_untied",
        "family": "qwen2",
        "prompt_seed": 0,
        "config": {
            "vocab_size": 19, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "max_position_embeddings": 24,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": False,
        },
    },
    {
        "name": "qwen3_gqa_tied",
        "family": "qwen3",
        "prompt_seed": 5,
        "config": {
            "vocab_size": 27, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": True,
        },
    },
    {
        "name": "qwen3_narrow_untied",
        "family": "qwen3",
        "prompt_seed": 4,
        "config": {
            "vocab_size": 17, "hidden_size": 24, "intermediate_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 6, "max_position_embeddings": 24,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": False,
        },
    },
)
REJECTION_BASE = {
    "vocab_size": 16, "hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 16,
}


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        return value ^ (value >> 31)

    def unit(self):
        return ((self.next() >> 40) - (1 << 23)) / (1 << 23)


def stream(case_name, label):
    digest = hashlib.sha256(f"{case_name}:{label}".encode("utf-8")).digest()
    return SplitMix64(int.from_bytes(digest[:8], "little"))


def parameter_values(case_name, name, shape):
    generator = stream(case_name, name)
    count = math.prod(shape)
    if name.endswith(".bias"):
        return [0.5 * generator.unit() for _ in range(count)]
    if len(shape) == 1:
        return [1.0 + 0.5 * generator.unit() for _ in range(count)]
    if name.endswith("embed_tokens.weight"):
        return [generator.unit() for _ in range(count)]
    scale = LINEAR_GAIN * math.sqrt(3.0 / shape[1])
    return [scale * generator.unit() for _ in range(count)]


def prompt_tokens(case_name, seed, vocab):
    generator = stream(case_name, f"prompt:{seed}")
    tokens = [generator.next() % vocab for _ in range(PROMPT_LENGTH)]
    tokens[7] = tokens[2]
    return tokens


def require_dependencies():
    try:
        import safetensors
        import safetensors.torch
        import torch
        import transformers
    except ImportError as error:
        raise SystemExit(
            "黄金数据生成依赖 torch、transformers 与 safetensors；"
            "请在仓库外的临时环境安装 tools/golden_requirements.txt 后运行"
        ) from error
    torch.set_num_threads(1)
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    return torch, transformers, safetensors.torch, versions


def family_classes(family, transformers):
    config_name, model_name = FAMILIES[family]
    return getattr(transformers, config_name), getattr(transformers, model_name)


def save_checkpoint(case_name, family, config, directory, torch, transformers):
    config_class, model_class = family_classes(family, transformers)
    model = model_class(config_class(**config)).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = parameter_values(case_name, name, tuple(parameter.shape))
            parameter.copy_(torch.tensor(values, dtype=torch.float32).reshape(parameter.shape))
    model.save_pretrained(directory, safe_serialization=True)


def quiet_export(checkpoint, output):
    with contextlib.redirect_stdout(io.StringIO()):
        export_llm.export(checkpoint, output)
    return output.read_bytes()


def reference(model, prompt, torch):
    double = copy.deepcopy(model).double()
    margin = math.inf
    greedy = []
    with torch.no_grad():
        output = double(input_ids=torch.tensor([prompt]), use_cache=True)
        cached = [output.logits[0]]
        for _ in range(GREEDY_LENGTH):
            top = torch.topk(cached[-1][-1], 2)
            margin = min(margin, float(top.values[0] - top.values[1]))
            greedy.append(int(top.indices[0]))
            output = double(
                input_ids=torch.tensor([[greedy[-1]]]),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            cached.append(output.logits[0])
        tokens = torch.tensor([prompt + greedy])
        full = double(input_ids=tokens, use_cache=False).logits[0]
        single = model(input_ids=tokens, use_cache=False).logits[0]
    return {
        "greedy": greedy,
        "logits": full.to(torch.float32).tolist(),
        "margin": margin,
        "cache_error": float((torch.cat(cached) - full).abs().max()),
        "f32_error": float((single.double() - full).abs().max()),
        "f32_within_tolerance": all(
            within_tolerance(left, right)
            for left, right in zip(single.flatten().tolist(), full.flatten().tolist())
        ),
    }


def build_case(case, workdir, torch, transformers):
    name, family, config = case["name"], case["family"], case["config"]
    checkpoint = workdir / name / "checkpoint"
    save_checkpoint(name, family, config, checkpoint, torch, transformers)
    model_bytes = quiet_export(checkpoint, workdir / name / f"{name}.nncossllm")
    _, model_class = family_classes(family, transformers)
    model = model_class.from_pretrained(checkpoint, local_files_only=True).eval()
    prompt = prompt_tokens(name, case["prompt_seed"], config["vocab_size"])
    result = reference(model, prompt, torch)
    if result["margin"] < MIN_GREEDY_MARGIN:
        raise RuntimeError(f"{name} 贪心参考的 top-2 间隔 {result['margin']:.3g} 过小，无法作为稳定黄金数据")
    if result["cache_error"] > MAX_CACHE_ERROR:
        raise RuntimeError(f"{name} 的 Hugging Face KV Cache 与无缓存前向相差 {result['cache_error']:.3g}")
    if not result["f32_within_tolerance"]:
        raise RuntimeError(f"{name} 的 Hugging Face f32 前向与 f64 参考相差 {result['f32_error']:.3g}，超出对拍容差")
    golden_bytes = encode_golden(config["vocab_size"], prompt, result["greedy"], result["logits"])
    print(
        f"  {name}: model_bytes={len(model_bytes)} hf_f32_vs_f64={result['f32_error']:.3g} "
        f"hf_cache_vs_full={result['cache_error']:.3g} greedy_margin={result['margin']:.3g}"
    )
    saved_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    plan = export_llm.tensor_plan(export_llm.parse_config(saved_config), *export_llm.FAMILY_FEATURES[family])
    return {
        "name": name,
        "family": family,
        "config": config,
        "plan": plan,
        "model": model_bytes,
        "golden_bytes": golden_bytes,
        "golden": decode_golden(golden_bytes),
    }


def run_live(targets, built):
    for target in targets:
        maximum = max(run_case(target, item["name"], item["model"], item["golden"]) for item in built)
        print(f"  {target.name}: {len(built)} 个用例通过，最大 logits 误差={maximum:.3g}")


def expect_mismatch(target, item, mutated, label):
    try:
        run_case(target, item["name"], bytes(mutated), item["golden"])
    except GoldenMismatch:
        return
    raise RuntimeError(f"{item['name']} 在{label}后仍通过对拍，黄金数据缺乏区分能力")


def run_mutation_controls(target, built):
    tensors = 0
    for item in built:
        model = item["model"]
        mutated = bytearray(model)
        mutated[ROPE_MODE_OFFSET:ROPE_MODE_OFFSET + 4] = struct.pack("<I", 1)
        expect_mismatch(target, item, mutated, "改为 Interleaved RoPE ")
        mutated = bytearray(model)
        theta = struct.unpack_from("<f", model, ROPE_THETA_OFFSET)[0]
        struct.pack_into("<f", mutated, ROPE_THETA_OFFSET, theta * 4.0)
        expect_mismatch(target, item, mutated, "放大 rope_theta ")
        offset = HEADER_LENGTH
        for name, shape in item["plan"]:
            length = 4 * math.prod(shape)
            mutated = bytearray(model)
            for index in range(offset + 3, offset + length, 4):
                mutated[index] ^= 0x80
            expect_mismatch(target, item, mutated, f"对 {name} 取反")
            offset += length
            tensors += 1
        if offset != len(model):
            raise RuntimeError(f"{item['name']} 的导出长度与张量计划不一致")
    print(
        f"  {target.name}: {len(built)} 个用例在改写 RoPE 布局、放大 rope_theta "
        f"或对任一导出张量取反（共 {tensors} 个）后均被黄金数据判为不一致"
    )


def edit_config(checkpoint, change):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    change(config)
    path.write_text(json.dumps(config), encoding="utf-8")


def relabel(model_type):
    def apply(checkpoint, torch, safetensors_torch):
        edit_config(checkpoint, lambda config: config.update(model_type=model_type))

    return apply


def use_rope_parameters(checkpoint, torch, safetensors_torch):
    def change(config):
        config["rope_parameters"] = {"rope_type": "default", "rope_theta": config.pop("rope_theta")}
        config.pop("rope_scaling", None)

    edit_config(checkpoint, change)


def drop_configured_layer(checkpoint, torch, safetensors_torch):
    edit_config(checkpoint, lambda config: config.update(num_hidden_layers=1))


def edit_tensors(checkpoint, safetensors_torch, change):
    path = checkpoint / "model.safetensors"
    tensors = safetensors_torch.load_file(str(path))
    change(tensors)
    safetensors_torch.save_file(tensors, str(path))


def add_unknown_tensor(checkpoint, torch, safetensors_torch):
    edit_tensors(checkpoint, safetensors_torch, lambda tensors: tensors.update({"model.adapter.weight": torch.ones(2)}))


def add_divergent_head(checkpoint, torch, safetensors_torch):
    edit_tensors(
        checkpoint,
        safetensors_torch,
        lambda tensors: tensors.update({"lm_head.weight": tensors["model.embed_tokens.weight"] + 1.0}),
    )


def add_identical_head(checkpoint, torch, safetensors_torch):
    edit_tensors(
        checkpoint,
        safetensors_torch,
        lambda tensors: tensors.update({"lm_head.weight": tensors["model.embed_tokens.weight"].clone()}),
    )


REJECTIONS = (
    ("llama_attention_bias", "llama", {"attention_bias": True}, None, "暂不支持以下 bias"),
    ("llama_mlp_bias", "llama", {"mlp_bias": True}, None, "暂不支持以下 bias"),
    ("llama_gelu", "llama", {"hidden_act": "gelu"}, None, "hidden_act"),
    ("llama_rope_scaling", "llama", {"rope_scaling": {"rope_type": "linear", "factor": 2.0}}, None, "rope_scaling"),
    ("llama_rope_parameters", "llama", {}, use_rope_parameters, "rope_parameters"),
    ("llama_unconfigured_layer", "llama", {}, drop_configured_layer, "不会被导出"),
    ("llama_unknown_tensor", "llama", {}, add_unknown_tensor, "不会被导出"),
    ("llama_tied_divergent_head", "llama", {"tie_word_embeddings": True}, add_divergent_head, "不一致"),
    ("llama_as_qwen3", "llama", {}, relabel("qwen3"), "缺少 Q/K norm"),
    ("qwen2_sliding_window", "qwen2", {"use_sliding_window": True, "sliding_window": 4, "max_window_layers": 1}, None, "sliding window"),
    ("qwen2_as_llama", "qwen2", {}, relabel("llama"), "包含 Q/K/V bias"),
    ("qwen3_attention_bias", "qwen3", {"head_dim": 8, "attention_bias": True}, None, "暂不支持以下 bias"),
    ("qwen3_as_qwen2", "qwen3", {"head_dim": 8}, relabel("qwen2"), "缺少 Q/K/V bias"),
    ("qwen3_as_llama", "qwen3", {"head_dim": 8}, relabel("llama"), "包含 Q/K norm"),
)


def run_rejections(workdir, torch, transformers, safetensors_torch):
    for name, family, overrides, change, pattern in REJECTIONS:
        checkpoint = workdir / "rejections" / name / "checkpoint"
        save_checkpoint(name, family, {**REJECTION_BASE, **overrides}, checkpoint, torch, transformers)
        if change is not None:
            change(checkpoint, torch, safetensors_torch)
        output = checkpoint.parent / "model.nncossllm"
        try:
            quiet_export(checkpoint, output)
        except ValueError as error:
            if not re.search(pattern, str(error)):
                raise RuntimeError(f"{name} 被拒绝的原因不符合预期：{error}") from error
        else:
            raise RuntimeError(f"{name} 应被导出器拒绝，但导出成功")
        leftovers = [path.name for path in checkpoint.parent.iterdir() if path.name != "checkpoint"]
        if leftovers:
            raise RuntimeError(f"{name} 导出失败后残留文件：{leftovers}")
    print(f"  导出器拒绝了 {len(REJECTIONS)} 个由 Hugging Face 实际保存的不支持检查点")
    checkpoint = workdir / "rejections" / "llama_tied_identical_head" / "checkpoint"
    config = {**REJECTION_BASE, "tie_word_embeddings": True}
    save_checkpoint("llama_tied_identical_head", "llama", config, checkpoint, torch, transformers)
    plain = quiet_export(checkpoint, checkpoint.parent / "plain.nncossllm")
    add_identical_head(checkpoint, torch, safetensors_torch)
    if quiet_export(checkpoint, checkpoint.parent / "duplicated.nncossllm") != plain:
        raise RuntimeError("绑定权重检查点附带相同 lm_head.weight 时导出结果发生变化")
    print("  绑定权重检查点附带相同 lm_head.weight 时导出结果不变")


def manifest_for(built, versions):
    return {
        "format": 1,
        "generator": {"script": "tools/make_golden_llm.py", **versions},
        "reference": "Hugging Face float64 forward without KV cache, rounded to f32",
        "cases": [
            {
                "name": item["name"],
                "family": item["family"],
                "config": item["config"],
                "prompt_length": len(item["golden"]["prompt"]),
                "greedy_length": len(item["golden"]["greedy"]),
                "model_sha256": hashlib.sha256(item["model"]).hexdigest(),
                "golden_sha256": hashlib.sha256(item["golden_bytes"]).hexdigest(),
            }
            for item in built
        ],
    }


def write_fixtures(built, directory, versions):
    directory.mkdir(parents=True, exist_ok=True)
    expected = {"manifest.json"}
    for item in built:
        expected.update((f"{item['name']}.nncossllm", f"{item['name']}.golden"))
    stale = sorted(path.name for path in directory.iterdir() if path.name not in expected)
    if stale:
        raise RuntimeError(f"夹具目录包含不属于当前用例的文件，请人工确认后移除：{stale}")
    for item in built:
        (directory / f"{item['name']}.nncossllm").write_bytes(item["model"])
        (directory / f"{item['name']}.golden").write_bytes(item["golden_bytes"])
    text = json.dumps(manifest_for(built, versions), ensure_ascii=False, indent=2) + "\n"
    (directory / "manifest.json").write_text(text, encoding="utf-8")
    print(f"  已写入 {len(built)} 个用例到 {directory}")


def compare_committed(built, directory, versions):
    manifest, cases = load_fixtures(directory)
    committed = {name: (model, golden) for name, model, golden in cases}
    names = [item["name"] for item in built]
    if sorted(committed) != sorted(names):
        raise RuntimeError(f"已提交夹具的用例 {sorted(committed)} 与生成脚本的用例 {sorted(names)} 不一致")
    maximum = 0.0
    for item in built:
        model, golden = committed[item["name"]]
        if model != item["model"]:
            raise RuntimeError(f"{item['name']}.nncossllm 与重新导出的模型不是逐字节一致")
        fresh = item["golden"]
        if (golden["vocab"], golden["prompt"], golden["greedy"]) != (fresh["vocab"], fresh["prompt"], fresh["greedy"]):
            raise RuntimeError(f"{item['name']}.golden 的 token 序列与重新生成的结果不一致")
        for row, fresh_row in zip(golden["logits"], fresh["logits"]):
            for left, right in zip(row, fresh_row):
                maximum = max(maximum, abs(left - right))
        if maximum > REGENERATION_TOLERANCE:
            raise RuntimeError(f"{item['name']}.golden 的 logits 与重新生成的结果相差 {maximum:.3g}")
    recorded = manifest["generator"]
    drift = [
        f"{key} {recorded.get(key)} -> {versions[key]}"
        for key in ("torch", "transformers", "safetensors")
        if recorded.get(key) != versions[key]
    ]
    print(f"  已提交夹具可复现：模型逐字节一致，logits 最大差异={maximum:.3g}")
    if drift:
        print(f"  注意：当前依赖版本与夹具记录不同（{'; '.join(drift)}）")


def main():
    parser = argparse.ArgumentParser(description="用本地生成的 tiny Hugging Face 模型生成并校验 PyTorch 黄金数据")
    parser.add_argument("--native", type=Path, default=ROOT / "target" / "release" / native_library_name())
    parser.add_argument("--wasm", type=Path, default=ROOT / "target" / "wasm32-unknown-unknown" / "release" / "nncoss_llm.wasm")
    parser.add_argument("--fixtures", type=Path, default=GOLDEN_DIR, help="黄金夹具目录")
    parser.add_argument("--write", action="store_true", help="全部对拍通过后覆盖写入夹具，而不是与已提交夹具比较")
    args = parser.parse_args()
    torch, transformers, safetensors_torch, versions = require_dependencies()
    print(f"参考实现：{', '.join(f'{key} {value}' for key, value in versions.items())}")
    print(f"对拍容差：{ABS_TOLERANCE:g} + {REL_TOLERANCE:g}×|参考值|")
    with tempfile.TemporaryDirectory(prefix="nncoss-golden-") as folder:
        workdir = Path(folder)
        print("生成 tiny 模型、导出并计算 Hugging Face 参考：")
        built = [build_case(case, workdir, torch, transformers) for case in CASES]
        native = NativeTarget(args.native)
        print("对拍 Rust 后端：")
        run_live((native, WasmTarget(args.wasm)), built)
        print("反向对照：")
        run_mutation_controls(native, built)
        run_rejections(workdir, torch, transformers, safetensors_torch)
    print("夹具：")
    if args.write:
        write_fixtures(built, args.fixtures, versions)
    else:
        compare_committed(built, args.fixtures, versions)
    print("PyTorch／Hugging Face 黄金对拍流程通过")


if __name__ == "__main__":
    main()
