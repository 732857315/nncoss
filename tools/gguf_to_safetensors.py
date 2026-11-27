"""把 Qwen3 稠密模型的 GGUF 文件逐个张量还原成 f32，按 Hugging Face 的张量名写成 safetensors。

用途：与其他框架比较时让两边用同一份权重。例如官方只发布了 QAT 后的 Q4_1 GGUF，
本脚本把它精确还原（GGUF 的反量化没有误差），再交给 tools/export_llm.py 导出成本项目的格式。
config.json 与分词器文件取自同一模型的 Hugging Face 仓库，用 --config 指定。

依赖：gguf、numpy、safetensors（torch 不需要）。只支持 general.architecture 为 qwen3 的模型；
Qwen3 的 GGUF 不对 Q/K 权重做 RoPE 置换，所以只需改名。
"""

import argparse
import json
import shutil
from pathlib import Path

LAYER_NAMES = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
GLOBAL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
COPIED = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")


def hf_name(name):
    if name in GLOBAL_NAMES:
        return GLOBAL_NAMES[name]
    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "blk" and parts[2] in LAYER_NAMES:
        return f"model.layers.{parts[1]}.{LAYER_NAMES[parts[2]]}.{parts[3]}"
    raise ValueError(f"不认识的 GGUF 张量名：{name}")


def field_value(reader, key):
    field = reader.fields.get(key)
    if field is None:
        return None
    value = field.parts[field.data[0]]
    return bytes(value).decode("utf-8") if value.dtype.kind == "u" and value.dtype.itemsize == 1 else value.tolist()


def check_config(reader, config):
    """GGUF 的结构参数必须与 config.json 一致，否则说明两份文件不是同一个模型。"""
    expected = {
        "qwen3.block_count": config["num_hidden_layers"],
        "qwen3.embedding_length": config["hidden_size"],
        "qwen3.feed_forward_length": config["intermediate_size"],
        "qwen3.attention.head_count": config["num_attention_heads"],
        "qwen3.attention.head_count_kv": config["num_key_value_heads"],
        "qwen3.attention.key_length": config.get("head_dim"),
    }
    for key, value in expected.items():
        found = field_value(reader, key)
        found = found[0] if isinstance(found, list) else found
        if value is not None and found != value:
            raise SystemExit(f"{key} 为 {found}，config.json 为 {value}：不是同一个模型")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="同一模型的 Hugging Face 目录（取 config.json 与分词器）")
    parser.add_argument("--output", type=Path, required=True, help="输出目录，写入 model.safetensors 与复制的配置文件")
    args = parser.parse_args()

    import gguf
    import numpy as np
    from gguf.quants import dequantize
    from safetensors.numpy import save_file

    reader = gguf.GGUFReader(str(args.gguf))
    if field_value(reader, "general.architecture") != "qwen3":
        raise SystemExit("只支持 qwen3 架构的 GGUF")
    config = json.loads((args.config / "config.json").read_text(encoding="utf-8"))
    check_config(reader, config)

    tensors = {}
    for tensor in reader.tensors:
        values = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
        shape = tuple(int(size) for size in reversed(tensor.shape))
        tensors[hf_name(tensor.name)] = np.ascontiguousarray(values.reshape(shape))
    if config.get("tie_word_embeddings") and "lm_head.weight" in tensors:
        raise SystemExit("config 写明输出层与嵌入共用，GGUF 却另有 output.weight")

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output / "model.safetensors"), metadata={"format": "pt", "source": args.gguf.name})
    for name in COPIED:
        if (args.config / name).exists():
            shutil.copyfile(args.config / name, args.output / name)
    print(f"已写入 {args.output}：{len(tensors)} 个张量，来自 {args.gguf.name}")


if __name__ == "__main__":
    main()
