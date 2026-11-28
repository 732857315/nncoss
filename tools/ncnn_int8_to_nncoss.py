"""把 ncnn_llm 的 INT8 分块量化 Qwen3 模型原样转换为 nncoss 的 INT8 模型（FLAG_INT8_WEIGHTS）。

读取 ncnn_llm 模型目录中的 model.json、qwen3_decoder.ncnn.param/bin 与 qwen3_embed_token.ncnn.bin：
- 7 个投影矩阵：int8 码逐字节保留，ncnn 的每块 scale（还原值 = 码 / scale）换成 nncoss 的
  还原系数 1/scale（f32）；
- RMSNorm 权重原样保留；
- 嵌入为 bf16，展开成 f32（无损），与 LM Head 共用。
RoPE theta 默认取 model.json 中 ncnn_llm 实际使用的值，可用 --rope-theta 覆盖。

用法：python tools/ncnn_int8_to_nncoss.py MODEL_DIR OUT.nncossllm [--rope-theta 1000000] [--max-context 40960]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

BLOCK_SIZE = 64
BF16_TAG = 0x01348B83
FLAG_TIED, FLAG_QK_NORM, FLAG_INT8 = 1, 4, 32


def parse_param(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines[0].strip() != "7767517":
        raise SystemExit(f"{path}: 不是 ncnn param 文件")
    layers = []
    for line in lines[2:]:
        fields = line.split()
        if not fields:
            continue
        kind, name, bottoms, tops = fields[0], fields[1], int(fields[2]), int(fields[3])
        params = {}
        for item in fields[4 + bottoms + tops:]:
            key, value = item.split("=", 1)
            params[int(key)] = value
        layers.append((kind, name, params))
    return layers


class Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def take(self, count):
        if self.offset + count > len(self.data):
            raise SystemExit("decoder.bin 提前结束")
        part = self.data[self.offset:self.offset + count]
        self.offset += count
        return part


def read_decoder(param_path, bin_path):
    reader = Reader(bin_path.read_bytes())
    weights = []  # (kind, array...) 按图中顺序
    for kind, name, params in parse_param(param_path):
        if kind == "RMSNorm":
            size = int(params.get(0, "0"))
            if int(params.get(2, "1")) == 0:
                raise SystemExit(f"{name}: 不带 affine 的 RMSNorm 暂不支持")
            gamma = np.frombuffer(reader.take(size * 4), dtype="<f4").copy()
            weights.append(("norm", name, gamma, float(params.get(1, "1e-6"))))
        elif kind == "Gemm":
            if params.get(18) != "801" or params.get(5) != "1" or params.get(3) != "1":
                raise SystemExit(f"{name}: 只支持 18=801（8 bit、64 元素分块）、constantB、transB 的 Gemm")
            if params.get(10, "-1") != "-1":
                raise SystemExit(f"{name}: 带 bias 的 Gemm 暂不支持")
            rows, columns = int(params[8]), int(params[9])
            count = rows * columns
            codes = np.frombuffer(reader.take(count), dtype=np.int8).reshape(rows, columns).copy()
            reader.take((-count) % 4)
            blocks = -(-columns // BLOCK_SIZE)
            scales = np.frombuffer(reader.take(rows * blocks * 4), dtype="<f4").reshape(rows, blocks).copy()
            weights.append(("gemm", name, codes, scales))
        elif kind in {"Input", "Split", "Reshape", "Permute", "BinaryOp", "RotaryEmbed", "Swish", "SDPA"}:
            continue
        else:
            raise SystemExit(f"{name}: 未知的带权重层 {kind}")
    if reader.offset != len(reader.data):
        raise SystemExit(f"decoder.bin 剩余 {len(reader.data) - reader.offset} 字节未解析")
    return weights


def int8_bytes(codes, scales):
    rows, columns = codes.shape
    blocks = scales.shape[1]
    if (codes == -128).any():
        raise SystemExit("权重码含 -128，超出 [-127, 127]")
    descales = np.zeros_like(scales, dtype=np.float32)
    nonzero = scales != 0
    descales[nonzero] = (1.0 / scales[nonzero].astype(np.float64)).astype(np.float32)
    zero_blocks = ~nonzero
    if zero_blocks.any():
        padded = np.zeros((rows, blocks * BLOCK_SIZE), dtype=np.int8)
        padded[:, :columns] = codes
        if padded.reshape(rows, blocks, BLOCK_SIZE)[zero_blocks].any():
            raise SystemExit("scale 为零的块含有非零码")
    padded = np.zeros((rows, blocks * BLOCK_SIZE), dtype=np.int8)
    padded[:, :columns] = codes
    return descales.astype("<f4").tobytes() + padded.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rope-theta", type=float)
    parser.add_argument("--max-context", type=int, default=40960)
    args = parser.parse_args()

    meta = json.loads((args.model_dir / "model.json").read_text(encoding="utf-8"))
    files = meta["params"]
    if meta.get("type") != "qwen3":
        raise SystemExit(f"只支持 qwen3，实际为 {meta.get('type')}")
    if files["embed_token_bin"] != files["proj_out_bin"]:
        raise SystemExit("只支持嵌入与 LM Head 共用权重的模型")
    rope = meta["setting"]["rope"]
    if rope.get("type") != "RoPE":
        raise SystemExit(f"只支持标准 RoPE，实际为 {rope.get('type')}")
    theta = args.rope_theta if args.rope_theta is not None else float(rope["rope_theta"])
    head_dim = int(rope["rope_head_dim"])

    weights = read_decoder(args.model_dir / files["decoder_param"], args.model_dir / files["decoder_bin"])
    norms = [w for w in weights if w[0] == "norm"]
    gemms = [w for w in weights if w[0] == "gemm"]
    layers = len(gemms) // 7
    if len(gemms) != layers * 7 or len(norms) != layers * 4 + 1:
        raise SystemExit(f"层数不符：{len(gemms)} 个 Gemm、{len(norms)} 个 RMSNorm")
    hidden = norms[0][2].size
    q_dim, kv_dim, intermediate = gemms[0][2].shape[0], gemms[1][2].shape[0], gemms[4][2].shape[0]
    heads, kv_heads = q_dim // head_dim, kv_dim // head_dim
    epsilon = norms[0][3]

    embed = (args.model_dir / files["embed_token_bin"]).read_bytes()
    tag = struct.unpack("<I", embed[:4])[0]
    if tag != BF16_TAG:
        raise SystemExit(f"嵌入不是 ncnn bf16 数据（tag {tag:#x}）")
    raw = np.frombuffer(embed[4:], dtype="<u2")
    if raw.size % hidden:
        raise SystemExit("嵌入长度不是 hidden 的整数倍")
    vocab = raw.size // hidden
    embedding = (raw.astype(np.uint32) << 16).view(np.float32)

    expected = {
        "q": (q_dim, hidden), "k": (kv_dim, hidden), "v": (kv_dim, hidden), "o": (hidden, q_dim),
        "gate": (intermediate, hidden), "up": (intermediate, hidden), "down": (hidden, intermediate),
    }
    flags = FLAG_TIED | FLAG_QK_NORM | FLAG_INT8
    with args.output.open("wb") as output:
        output.write(b"NNCSLLM\0")
        output.write(struct.pack("<12I", 1, 1, vocab, hidden, intermediate, layers, heads, kv_heads,
                                 head_dim, args.max_context, flags, 0))
        output.write(struct.pack("<2f", epsilon, theta))
        output.write(embedding.astype("<f4").tobytes())
        for layer in range(layers):
            g = gemms[layer * 7:(layer + 1) * 7]
            n = norms[layer * 4:(layer + 1) * 4]
            for (name, (rows, columns)), gemm in zip(expected.items(), g):
                if gemm[2].shape != (rows, columns):
                    raise SystemExit(f"第 {layer} 层 {name}（{gemm[1]}）形状 {gemm[2].shape} 应为 {(rows, columns)}")
            if [x[2].size for x in n] != [hidden, head_dim, head_dim, hidden]:
                raise SystemExit(f"第 {layer} 层 RMSNorm 尺寸不符：{[x[2].size for x in n]}")
            # nncoss 顺序：input_layernorm, q, k, v, q_norm, k_norm, o, post_attention_layernorm, gate, up, down
            # ncnn 图中顺序：input_layernorm, q, q_norm, k, k_norm, v, o, post_attention, gate, up, down
            output.write(n[0][2].astype("<f4").tobytes())
            for gemm in g[:3]:
                output.write(int8_bytes(gemm[2], gemm[3]))
            output.write(n[1][2].astype("<f4").tobytes())
            output.write(n[2][2].astype("<f4").tobytes())
            output.write(int8_bytes(g[3][2], g[3][3]))
            output.write(n[3][2].astype("<f4").tobytes())
            for gemm in g[4:]:
                output.write(int8_bytes(gemm[2], gemm[3]))
        output.write(norms[-1][2].astype("<f4").tobytes())
    print(json.dumps({"vocab": vocab, "hidden": hidden, "intermediate": intermediate, "layers": layers,
                      "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "epsilon": epsilon,
                      "rope_theta": theta, "bytes": args.output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
