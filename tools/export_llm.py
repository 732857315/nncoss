import argparse
import json
import struct
import sys
from contextlib import ExitStack
from pathlib import Path


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


def write_tensor(output, sources, name, shape, torch):
    if name not in sources:
        raise ValueError(f"缺少权重：{name}")
    handle, key = sources[name]
    tensor = handle.get_tensor(key)
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


def export(model_dir, output_path):
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("导出依赖 torch 与 safetensors，请先在 Python 环境安装") from error
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    model_type = config.get("model_type")
    if model_type not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("仅支持已明确采用 HalfSplit RoPE 的 llama、qwen2、qwen3 配置")
    vocab = int(config["vocab_size"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    if "head_dim" in config:
        head_dim = int(config["head_dim"])
    else:
        if hidden % heads != 0:
            raise ValueError("hidden_size 不能被 num_attention_heads 整除")
        head_dim = hidden // heads
    if config.get("rope_scaling") is not None:
        raise ValueError("暂不支持 rope_scaling，请使用未缩放 RoPE 的模型")
    if float(config.get("partial_rotary_factor", 1.0)) != 1.0:
        raise ValueError("暂不支持 partial_rotary_factor")
    if config.get("use_sliding_window", False):
        raise ValueError("暂不支持 sliding window attention")
    maximum = int(config["max_position_embeddings"])
    epsilon = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])
    tied = bool(config.get("tie_word_embeddings", False))
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with ExitStack() as stack:
            sources = require_sources(model_dir, stack, safe_open)
            has_qkv_bias, has_qk_norm = validate_layer_features(sources, layers)
            flags = int(tied) | (int(has_qkv_bias) << 1) | (int(has_qk_norm) << 2)
            with temporary.open("wb") as output:
                output.write(b"NNCSLLM\0")
                output.write(struct.pack(
                    "<12I",
                    1,
                    1,
                    vocab,
                    hidden,
                    intermediate,
                    layers,
                    heads,
                    kv_heads,
                    head_dim,
                    maximum,
                    flags,
                    0,
                ))
                output.write(struct.pack("<2f", epsilon, theta))
                write_tensor(output, sources, "model.embed_tokens.weight", (vocab, hidden), torch)
                for layer in range(layers):
                    prefix = f"model.layers.{layer}"
                    write_tensor(output, sources, f"{prefix}.input_layernorm.weight", (hidden,), torch)
                    write_tensor(output, sources, f"{prefix}.self_attn.q_proj.weight", (q_dim, hidden), torch)
                    write_tensor(output, sources, f"{prefix}.self_attn.k_proj.weight", (kv_dim, hidden), torch)
                    write_tensor(output, sources, f"{prefix}.self_attn.v_proj.weight", (kv_dim, hidden), torch)
                    if has_qkv_bias:
                        write_tensor(output, sources, f"{prefix}.self_attn.q_proj.bias", (q_dim,), torch)
                        write_tensor(output, sources, f"{prefix}.self_attn.k_proj.bias", (kv_dim,), torch)
                        write_tensor(output, sources, f"{prefix}.self_attn.v_proj.bias", (kv_dim,), torch)
                    if has_qk_norm:
                        write_tensor(output, sources, f"{prefix}.self_attn.q_norm.weight", (head_dim,), torch)
                        write_tensor(output, sources, f"{prefix}.self_attn.k_norm.weight", (head_dim,), torch)
                    write_tensor(output, sources, f"{prefix}.self_attn.o_proj.weight", (hidden, q_dim), torch)
                    write_tensor(output, sources, f"{prefix}.post_attention_layernorm.weight", (hidden,), torch)
                    write_tensor(output, sources, f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden), torch)
                    write_tensor(output, sources, f"{prefix}.mlp.up_proj.weight", (intermediate, hidden), torch)
                    write_tensor(output, sources, f"{prefix}.mlp.down_proj.weight", (hidden, intermediate), torch)
                write_tensor(output, sources, "model.norm.weight", (hidden,), torch)
                if not tied:
                    write_tensor(output, sources, "lm_head.weight", (vocab, hidden), torch)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"已导出 {output_path}：type={model_type}, layers={layers}, hidden={hidden}, heads={heads}/{kv_heads}, vocab={vocab}, rope=HalfSplit, f32")


def main():
    args = parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()
