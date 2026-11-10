import argparse
import json
import struct
import sys
from contextlib import ExitStack
from pathlib import Path

FAMILY_FEATURES = {
    "llama": (False, False),
    "qwen2": (True, False),
    "qwen3": (False, True),
}
REQUIRED_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
)


def parse_args():
    parser = argparse.ArgumentParser(description="将 Llama/Qwen 风格的 safetensors 权重复制为 nncoss LLM f32 模型")
    parser.add_argument("--model", type=Path, required=True, help="包含 config.json 和 safetensors 的本地模型目录")
    parser.add_argument("--output", type=Path, required=True, help="输出的 .nncossllm 文件")
    return parser.parse_args()


def require_sources(model_dir, stack, safe_open):
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ValueError("模型目录不包含 .safetensors 文件")
    sources = {}
    for path in files:
        handle = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
        for name in handle.keys():
            if name in sources:
                raise ValueError(f"权重名称重复：{name}")
            sources[name] = (handle, name)
    return sources


def read_tensor(sources, name):
    if name not in sources:
        raise ValueError(f"缺少权重：{name}")
    handle, key = sources[name]
    return handle.get_tensor(key)


def write_tensor(output, sources, name, shape, torch):
    tensor = read_tensor(sources, name)
    if list(tensor.shape) != list(shape):
        raise ValueError(f"权重 {name} 形状为 {list(tensor.shape)}，预期为 {list(shape)}")
    tensor = tensor.detach().to(dtype=torch.float32).cpu().contiguous()
    array = tensor.numpy()
    if sys.byteorder != "little":
        array = array.byteswap()
    output.write(array.tobytes())


def validate_layer_features(sources, layers):
    if layers <= 0:
        raise ValueError("num_hidden_layers 必须大于 0")
    unsupported_suffixes = (
        ".self_attn.o_proj.bias",
        ".mlp.gate_proj.bias",
        ".mlp.up_proj.bias",
        ".mlp.down_proj.bias",
    )
    unsupported = sorted(
        name for name in sources
        if name == "lm_head.bias" or name.endswith(unsupported_suffixes)
    )
    if unsupported:
        raise ValueError(f"暂不支持以下 bias：{', '.join(unsupported)}")
    first_qkv = tuple(f"model.layers.0.self_attn.{name}.bias" in sources for name in ("q_proj", "k_proj", "v_proj"))
    first_qk = tuple(f"model.layers.0.self_attn.{name}.weight" in sources for name in ("q_norm", "k_norm"))
    if any(first_qkv) and not all(first_qkv):
        raise ValueError("第 0 层 Q/K/V bias 必须同时存在")
    if any(first_qk) and not all(first_qk):
        raise ValueError("第 0 层 Q/K norm 必须同时存在")
    has_qkv_bias = all(first_qkv)
    has_qk_norm = all(first_qk)
    for layer in range(1, layers):
        prefix = f"model.layers.{layer}.self_attn"
        qkv = tuple(f"{prefix}.{name}.bias" in sources for name in ("q_proj", "k_proj", "v_proj"))
        qk = tuple(f"{prefix}.{name}.weight" in sources for name in ("q_norm", "k_norm"))
        if qkv != first_qkv:
            raise ValueError(f"第 {layer} 层 Q/K/V bias 配置与第 0 层不一致")
        if qk != first_qk:
            raise ValueError(f"第 {layer} 层 Q/K norm 配置与第 0 层不一致")
    return has_qkv_bias, has_qk_norm


def parse_config(config):
    model_type = config.get("model_type")
    if model_type not in FAMILY_FEATURES:
        raise ValueError("仅支持已明确采用 HalfSplit RoPE 的 llama、qwen2、qwen3 配置")
    if config.get("rope_parameters") is not None:
        raise ValueError("暂不支持 rope_parameters 配置格式，仅接受带顶层 rope_theta 的配置")
    missing = [key for key in REQUIRED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"config.json 缺少字段：{', '.join(missing)}")
    if config.get("rope_scaling") is not None:
        raise ValueError("暂不支持 rope_scaling，请使用未缩放 RoPE 的模型")
    partial_rotary = config.get("partial_rotary_factor")
    if partial_rotary is not None and float(partial_rotary) != 1.0:
        raise ValueError("暂不支持 partial_rotary_factor")
    layer_types = config.get("layer_types") or []
    if config.get("use_sliding_window", False) or any(kind != "full_attention" for kind in layer_types):
        raise ValueError("暂不支持 sliding window attention")
    hidden_act = config.get("hidden_act")
    if hidden_act is not None and hidden_act != "silu":
        raise ValueError(f"暂不支持 hidden_act={hidden_act}，仅支持 silu")
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    if heads <= 0:
        raise ValueError("num_attention_heads 必须大于 0")
    kv_heads = config.get("num_key_value_heads")
    head_dim = config.get("head_dim")
    if head_dim is None:
        if hidden % heads != 0:
            raise ValueError("hidden_size 不能被 num_attention_heads 整除")
        head_dim = hidden // heads
    return {
        "model_type": model_type,
        "vocab": int(config["vocab_size"]),
        "hidden": hidden,
        "intermediate": int(config["intermediate_size"]),
        "layers": int(config["num_hidden_layers"]),
        "heads": heads,
        "kv_heads": heads if kv_heads is None else int(kv_heads),
        "head_dim": int(head_dim),
        "maximum": int(config["max_position_embeddings"]),
        "epsilon": float(config["rms_norm_eps"]),
        "theta": float(config["rope_theta"]),
        "tied": bool(config.get("tie_word_embeddings", False)),
    }


def validate_family_features(model_type, has_qkv_bias, has_qk_norm):
    expected_qkv_bias, expected_qk_norm = FAMILY_FEATURES[model_type]
    if has_qkv_bias != expected_qkv_bias:
        state = "包含" if has_qkv_bias else "缺少"
        raise ValueError(f"{model_type} 检查点{state} Q/K/V bias，与该 model_type 的结构不符")
    if has_qk_norm != expected_qk_norm:
        state = "包含" if has_qk_norm else "缺少"
        raise ValueError(f"{model_type} 检查点{state} Q/K norm，与该 model_type 的结构不符")


def tensor_plan(spec, has_qkv_bias, has_qk_norm):
    vocab, hidden, intermediate = spec["vocab"], spec["hidden"], spec["intermediate"]
    head_dim = spec["head_dim"]
    q_dim = spec["heads"] * head_dim
    kv_dim = spec["kv_heads"] * head_dim
    plan = [("model.embed_tokens.weight", (vocab, hidden))]
    for layer in range(spec["layers"]):
        prefix = f"model.layers.{layer}"
        plan.append((f"{prefix}.input_layernorm.weight", (hidden,)))
        plan.append((f"{prefix}.self_attn.q_proj.weight", (q_dim, hidden)))
        plan.append((f"{prefix}.self_attn.k_proj.weight", (kv_dim, hidden)))
        plan.append((f"{prefix}.self_attn.v_proj.weight", (kv_dim, hidden)))
        if has_qkv_bias:
            plan.append((f"{prefix}.self_attn.q_proj.bias", (q_dim,)))
            plan.append((f"{prefix}.self_attn.k_proj.bias", (kv_dim,)))
            plan.append((f"{prefix}.self_attn.v_proj.bias", (kv_dim,)))
        if has_qk_norm:
            plan.append((f"{prefix}.self_attn.q_norm.weight", (head_dim,)))
            plan.append((f"{prefix}.self_attn.k_norm.weight", (head_dim,)))
        plan.append((f"{prefix}.self_attn.o_proj.weight", (hidden, q_dim)))
        plan.append((f"{prefix}.post_attention_layernorm.weight", (hidden,)))
        plan.append((f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden)))
        plan.append((f"{prefix}.mlp.up_proj.weight", (intermediate, hidden)))
        plan.append((f"{prefix}.mlp.down_proj.weight", (hidden, intermediate)))
    plan.append(("model.norm.weight", (hidden,)))
    if not spec["tied"]:
        plan.append(("lm_head.weight", (vocab, hidden)))
    return plan


def summarize(names, limit=8):
    names = list(names)
    text = ", ".join(names[:limit])
    if len(names) > limit:
        text += f" 等 {len(names)} 项"
    return text


def validate_tensor_names(sources, plan, tied):
    planned = [name for name, _ in plan]
    missing = [name for name in planned if name not in sources]
    if missing:
        raise ValueError(f"缺少权重：{summarize(missing)}")
    allowed = set(planned)
    if tied:
        allowed.add("lm_head.weight")
    unused = sorted(name for name in sources if name not in allowed)
    if unused:
        raise ValueError(f"检查点包含不会被导出的权重：{summarize(unused)}")


def validate_tied_head(sources, torch):
    if "lm_head.weight" not in sources:
        return
    head = read_tensor(sources, "lm_head.weight")
    embedding = read_tensor(sources, "model.embed_tokens.weight")
    if head.shape != embedding.shape or not torch.equal(head, embedding):
        raise ValueError("tie_word_embeddings 为 true，但 lm_head.weight 与 model.embed_tokens.weight 不一致")


def export(model_dir, output_path):
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("导出依赖 torch 与 safetensors，请先在 Python 环境安装") from error
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    spec = parse_config(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with ExitStack() as stack:
            sources = require_sources(model_dir, stack, safe_open)
            has_qkv_bias, has_qk_norm = validate_layer_features(sources, spec["layers"])
            validate_family_features(spec["model_type"], has_qkv_bias, has_qk_norm)
            plan = tensor_plan(spec, has_qkv_bias, has_qk_norm)
            validate_tensor_names(sources, plan, spec["tied"])
            if spec["tied"]:
                validate_tied_head(sources, torch)
            flags = int(spec["tied"]) | (int(has_qkv_bias) << 1) | (int(has_qk_norm) << 2)
            with temporary.open("wb") as output:
                output.write(b"NNCSLLM\0")
                output.write(struct.pack(
                    "<12I",
                    1,
                    1,
                    spec["vocab"],
                    spec["hidden"],
                    spec["intermediate"],
                    spec["layers"],
                    spec["heads"],
                    spec["kv_heads"],
                    spec["head_dim"],
                    spec["maximum"],
                    flags,
                    0,
                ))
                output.write(struct.pack("<2f", spec["epsilon"], spec["theta"]))
                for name, shape in plan:
                    write_tensor(output, sources, name, shape, torch)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"已导出 {output_path}：type={spec['model_type']}, layers={spec['layers']}, hidden={spec['hidden']}, "
        f"heads={spec['heads']}/{spec['kv_heads']}, vocab={spec['vocab']}, rope=HalfSplit, f32"
    )


def main():
    args = parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()
