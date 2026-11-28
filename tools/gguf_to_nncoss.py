"""把 Qwen3 稠密模型的 GGUF 直接转换为 nncoss 格式，权重逐位保留，便于与 llama.cpp 等用同一份权重比较。

- Q8_0（32 个元素一块、f16 系数）的矩阵写成 nncoss 的 INT8 分块格式（块长 32），f16 系数无损展开成 f32，
  int8 码逐字节保留；嵌入与 LM Head 也是 Q8_0 时一并写成 INT8。
- F32、F16、BF16 张量写成 f32（无损）。
- 其他量化类型暂不支持（可先用 tools/gguf_to_safetensors.py 精确还原成 f32，再用 export_llm.py 导出）。
GGUF 中嵌入与输出层的字节完全相同时写成共用 LM Head 的模型。

依赖：gguf、numpy。只支持 general.architecture 为 qwen3 的模型；Qwen3 的 GGUF 不对 Q/K 权重做 RoPE 置换。

用法：python tools/gguf_to_nncoss.py MODEL.gguf OUT.nncossllm [--max-context N]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader

BLOCK = 32
FLAG_TIED, FLAG_QKV_BIAS, FLAG_QK_NORM = 1, 2, 4
FLAG_INT8, FLAG_INT8_BLOCK32, FLAG_INT8_EMBEDDINGS = 32, 64, 128
PROJECTIONS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
FLOAT_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16}


def field(reader, key, default=None):
    item = reader.fields.get(key)
    if item is None:
        if default is None:
            raise SystemExit(f"GGUF 缺少 {key}")
        return default
    value = item.parts[item.data[0]]
    if item.types and item.types[0].name == "STRING":
        return bytes(value).decode()
    return value.tolist()[0]


def as_float(tensor):
    data = np.asarray(tensor.data)
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return data.astype("<f4")
    if tensor.tensor_type == GGMLQuantizationType.F16:
        return data.astype(np.float32)
    if tensor.tensor_type == GGMLQuantizationType.BF16:
        return (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    raise SystemExit(f"{tensor.name}: 需要浮点张量，实际为 {tensor.tensor_type.name}")


def q8_0_bytes(tensor, rows, columns):
    """GGUF Q8_0 → nncoss INT8（块长 32）：rows × blocks 个 f32 系数，再是 rows × blocks × 32 个码。"""
    blocks = -(-columns // BLOCK)
    raw = np.asarray(tensor.data, dtype=np.uint8).reshape(rows, blocks, 2 + BLOCK)
    descales = raw[:, :, :2].copy().view("<f2").reshape(rows, blocks).astype(np.float32)
    codes = raw[:, :, 2:].copy().view(np.int8).reshape(rows, blocks * BLOCK)
    if not np.isfinite(descales).all():
        raise SystemExit(f"{tensor.name}: Q8_0 系数含非有限值")
    if (codes == -128).any():
        raise SystemExit(f"{tensor.name}: Q8_0 码含 -128，超出 [-127, 127]")
    if columns % BLOCK and codes[:, columns:].any():
        raise SystemExit(f"{tensor.name}: 行末补齐不为零")
    return descales.astype("<f4").tobytes() + codes.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-context", type=int)
    args = parser.parse_args()

    reader = GGUFReader(args.gguf)
    arch = field(reader, "general.architecture")
    if arch != "qwen3":
        raise SystemExit(f"只支持 qwen3，实际为 {arch}")
    layers = int(field(reader, "qwen3.block_count"))
    hidden = int(field(reader, "qwen3.embedding_length"))
    intermediate = int(field(reader, "qwen3.feed_forward_length"))
    heads = int(field(reader, "qwen3.attention.head_count"))
    kv_heads = int(field(reader, "qwen3.attention.head_count_kv"))
    head_dim = int(field(reader, "qwen3.attention.key_length", hidden // heads))
    if int(field(reader, "qwen3.attention.value_length", head_dim)) != head_dim:
        raise SystemExit("key_length 与 value_length 不同，暂不支持")
    theta = float(field(reader, "qwen3.rope.freq_base"))
    epsilon = float(field(reader, "qwen3.attention.layer_norm_rms_epsilon"))
    maximum = args.max_context or int(field(reader, "qwen3.context_length"))
    if "qwen3.rope.scaling.type" in reader.fields:
        raise SystemExit("暂不支持 RoPE scaling")

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    embed = tensors["token_embd.weight"]
    vocab = int(embed.shape[1])
    output = tensors.get("output.weight")
    tied = output is None or (output.tensor_type == embed.tensor_type
                              and np.array_equal(np.asarray(output.data), np.asarray(embed.data)))
    projection_types = {tensors[f"blk.{i}.{name}.weight"].tensor_type for i in range(layers) for name in PROJECTIONS}
    if projection_types == {GGMLQuantizationType.Q8_0}:
        int8 = True
    elif projection_types <= FLOAT_TYPES:
        int8 = False
    else:
        raise SystemExit(f"投影矩阵类型 {sorted(t.name for t in projection_types)} 暂不支持直接转换")
    embedding_types = {embed.tensor_type} | ({output.tensor_type} if output is not None and not tied else set())
    int8_embeddings = int8 and embedding_types == {GGMLQuantizationType.Q8_0}
    if not int8_embeddings and not embedding_types <= FLOAT_TYPES | {GGMLQuantizationType.Q8_0}:
        raise SystemExit(f"嵌入类型 {sorted(t.name for t in embedding_types)} 暂不支持")
    if not int8 and embedding_types & {GGMLQuantizationType.Q8_0}:
        raise SystemExit("投影为浮点而嵌入为 Q8_0 的组合暂不支持")
    has_bias = f"blk.0.attn_q.bias" in tensors
    has_qk_norm = "blk.0.attn_q_norm.weight" in tensors

    def matrix(tensor, rows, columns, quantized):
        if [int(x) for x in tensor.shape] != [columns, rows]:
            raise SystemExit(f"{tensor.name}: 形状 {list(tensor.shape)} 应为 [{columns}, {rows}]")
        if quantized:
            return q8_0_bytes(tensor, rows, columns)
        if tensor.tensor_type == GGMLQuantizationType.Q8_0:
            raw = np.asarray(tensor.data, dtype=np.uint8).reshape(rows, -1, 2 + BLOCK)
            d = raw[:, :, :2].copy().view("<f2").astype(np.float32)
            q = raw[:, :, 2:].copy().view(np.int8).astype(np.float32)
            return (q * d).reshape(rows, -1)[:, :columns].astype("<f4").tobytes()
        return as_float(tensor).reshape(rows, columns).astype("<f4").tobytes()

    def vector(name, size):
        tensor = tensors[name]
        values = as_float(tensor).reshape(-1)
        if values.size != size:
            raise SystemExit(f"{name}: 长度 {values.size} 应为 {size}")
        return values.astype("<f4").tobytes()

    flags = (FLAG_TIED if tied else 0) | (FLAG_QKV_BIAS if has_bias else 0) | (FLAG_QK_NORM if has_qk_norm else 0)
    if int8:
        flags |= FLAG_INT8 | FLAG_INT8_BLOCK32 | (FLAG_INT8_EMBEDDINGS if int8_embeddings else 0)
    q_dim, kv_dim = heads * head_dim, kv_heads * head_dim
    shapes = {"attn_q": (q_dim, hidden), "attn_k": (kv_dim, hidden), "attn_v": (kv_dim, hidden),
              "attn_output": (hidden, q_dim), "ffn_gate": (intermediate, hidden), "ffn_up": (intermediate, hidden),
              "ffn_down": (hidden, intermediate)}
    with args.output.open("wb") as out:
        out.write(b"NNCSLLM\0")
        out.write(struct.pack("<12I", 1, 1, vocab, hidden, intermediate, layers, heads, kv_heads, head_dim,
                              maximum, flags, 0))
        out.write(struct.pack("<2f", epsilon, theta))
        out.write(matrix(embed, vocab, hidden, int8_embeddings))
        for i in range(layers):
            block = f"blk.{i}"
            out.write(vector(f"{block}.attn_norm.weight", hidden))
            for name in ("attn_q", "attn_k", "attn_v"):
                out.write(matrix(tensors[f"{block}.{name}.weight"], *shapes[name], int8))
            if has_bias:
                out.write(vector(f"{block}.attn_q.bias", q_dim))
                out.write(vector(f"{block}.attn_k.bias", kv_dim))
                out.write(vector(f"{block}.attn_v.bias", kv_dim))
            if has_qk_norm:
                out.write(vector(f"{block}.attn_q_norm.weight", head_dim))
                out.write(vector(f"{block}.attn_k_norm.weight", head_dim))
            out.write(matrix(tensors[f"{block}.attn_output.weight"], *shapes["attn_output"], int8))
            out.write(vector(f"{block}.ffn_norm.weight", hidden))
            for name in ("ffn_gate", "ffn_up", "ffn_down"):
                out.write(matrix(tensors[f"{block}.{name}.weight"], *shapes[name], int8))
        out.write(vector("output_norm.weight", hidden))
        if not tied:
            out.write(matrix(output, vocab, hidden, int8_embeddings))
    print(json.dumps({"vocab": vocab, "hidden": hidden, "intermediate": intermediate, "layers": layers,
                      "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "rope_theta": theta,
                      "epsilon": epsilon, "tied": tied, "int8": int8, "int8_embeddings": int8_embeddings,
                      "bytes": args.output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
