import argparse
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

import export_llm

HEADER = struct.Struct("<8s12I2f")
FLAG_TIED = 1
FLAG_QKV_BIAS = 2
FLAG_QK_NORM = 4
FLAG_Q4 = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="以 f32 原模型为教师，蒸馏 Q4 模型的行 scale、Norm 与 bias；4-bit 权重与组倍率保持不变"
    )
    parser.add_argument("--model", type=Path, required=True, help="原始 Hugging Face 模型目录")
    parser.add_argument("--input", type=Path, required=True, help="Q4 .nncossllm 文件")
    parser.add_argument("--output", type=Path, required=True, help="输出的 Q4 .nncossllm 文件")
    parser.add_argument("--calibration", type=Path, nargs="+", required=True, help="训练文本（UTF-8）")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--length", type=int, default=256, help="每步的 token 数")
    parser.add_argument("--scale-lr", type=float, default=1e-3)
    parser.add_argument("--norm-lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0 or args.length < 32:
        parser.error("--steps 需要正整数，--length 至少为 32")
    return args


def layout(path, spec):
    with path.open("rb") as file:
        fields = HEADER.unpack(file.read(HEADER.size))
    flags = fields[11]
    if fields[0] != b"NNCSLLM\0" or not flags & FLAG_Q4:
        raise ValueError("输入不是 Q4 .nncossllm 文件")
    expected = (spec["vocab"], spec["hidden"], spec["intermediate"], spec["layers"], spec["heads"], spec["kv_heads"], spec["head_dim"])
    if tuple(fields[3:10]) != expected or bool(flags & FLAG_TIED) != spec["tied"]:
        raise ValueError("Q4 文件与模型目录的结构不一致")
    plan = export_llm.tensor_plan(spec, bool(flags & FLAG_QKV_BIAS), bool(flags & FLAG_QK_NORM))
    entries, offset = {}, HEADER.size
    for name, shape in plan:
        if len(shape) == 2:
            rows, columns = shape
            size = rows * (4 + (columns + 31) // 32 * 17)
            size += -size % 4
        else:
            size = shape[0] * 4
        entries[name] = (shape, offset)
        offset += size
    if offset != path.stat().st_size:
        raise ValueError("Q4 文件长度与模型结构不一致")
    return entries


def read_matrix(data, offset, rows, columns, np, torch):
    groups = (columns + 31) // 32
    scale = np.frombuffer(data, "<f4", rows, offset).astype(np.float32)
    codes = np.frombuffer(data, np.int8, rows * groups, offset + rows * 4).reshape(rows, groups).astype(np.float32)
    packed = np.frombuffer(data, np.uint8, rows * groups * 16, offset + rows * (4 + groups)).reshape(rows, groups, 16)
    levels = np.empty((rows, groups, 32), np.float32)
    levels[:, :, 0::2] = (packed & 15).astype(np.float32) - 8
    levels[:, :, 1::2] = (packed >> 4).astype(np.float32) - 8
    integer = (levels * codes[:, :, None]).reshape(rows, groups * 32)[:, :columns]
    return torch.from_numpy(scale.copy()), torch.from_numpy(np.ascontiguousarray(integer))


def build_student(args, spec, entries, data, np, torch, nn, functional, model_class):
    class QuantizedLinear(nn.Module):
        def __init__(self, scale, integer, bias=None):
            super().__init__()
            self.register_buffer("base", scale)
            self.register_buffer("integer", integer)
            self.log = nn.Parameter(torch.zeros_like(scale))
            self.bias = None if bias is None else nn.Parameter(bias)

        def scale(self):
            return self.base * torch.exp(self.log)

        def forward(self, inputs):
            outputs = functional.linear(inputs, self.integer) * self.scale()
            return outputs if self.bias is None else outputs + self.bias

    class QuantizedEmbedding(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, ids):
            return self.linear.integer[ids] * self.linear.scale()[ids][..., None]

    def vector(name):
        shape, offset = entries[name]
        return torch.from_numpy(np.frombuffer(data, "<f4", shape[0], offset).copy())

    def matrix(name, bias=None):
        shape, offset = entries[name]
        return QuantizedLinear(*read_matrix(data, offset, *shape, np, torch), None if bias is None else vector(bias))

    model = model_class.from_pretrained(args.model, dtype=torch.float32)
    embedding = matrix("model.embed_tokens.weight")
    model.model.embed_tokens = QuantizedEmbedding(embedding)
    model.lm_head = embedding if spec["tied"] else matrix("lm_head.weight")
    lookup = {"model.embed_tokens.weight": embedding, "model.norm.weight": model.model.norm.weight}
    if not spec["tied"]:
        lookup["lm_head.weight"] = model.lm_head
    model.model.norm.weight.data.copy_(vector("model.norm.weight"))
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        for part in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"{prefix}.self_attn.{part}.weight"
            bias = f"{prefix}.self_attn.{part}.bias"
            module = matrix(name, bias if bias in entries else None)
            setattr(layer.self_attn, part, module)
            lookup[name] = module
            if bias in entries:
                lookup[bias] = module.bias
        for part in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.mlp.{part}.weight"
            module = matrix(name)
            setattr(layer.mlp, part, module)
            lookup[name] = module
        norms = {
            f"{prefix}.input_layernorm.weight": layer.input_layernorm.weight,
            f"{prefix}.post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        }
        if f"{prefix}.self_attn.q_norm.weight" in entries:
            norms[f"{prefix}.self_attn.q_norm.weight"] = layer.self_attn.q_norm.weight
            norms[f"{prefix}.self_attn.k_norm.weight"] = layer.self_attn.k_norm.weight
        for name, parameter in norms.items():
            parameter.data.copy_(vector(name))
        lookup.update(norms)
    missing = set(entries) - set(lookup)
    if missing:
        raise ValueError(f"无法映射的张量：{sorted(missing)[:4]}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scales, others = [], []
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.log.requires_grad_(True)
            scales.append(module.log)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                others.append(module.bias)
    for name, parameter in model.named_parameters():
        if "norm" in name:
            parameter.requires_grad_(True)
            others.append(parameter)
    return model, lookup, scales, others


def write(path, data, entries, lookup, np):
    buffer = bytearray(data)
    for name, (shape, offset) in entries.items():
        item = lookup[name]
        values = (item.scale() if len(shape) == 2 else item).detach().numpy().astype("<f4")
        if not np.isfinite(values).all() or (len(shape) == 2 and (values < 0).any()):
            raise ValueError(f"蒸馏结果无效：{name}")
        raw = values.tobytes()
        buffer[offset:offset + len(raw)] = raw
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(bytes(buffer))
    temporary.replace(path)


def main():
    args = parse_args()
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(f"需要 numpy、torch 与 transformers：{error}")
    spec = export_llm.parse_config(json.loads((args.model / "config.json").read_text(encoding="utf-8")))
    entries = layout(args.input, spec)
    data = args.input.read_bytes()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    student, lookup, scales, others = build_student(
        args, spec, entries, data, np, torch, nn, functional, AutoModelForCausalLM
    )
    windows = []
    for path in args.calibration:
        ids = tokenizer(path.read_text(encoding="utf-8"), return_tensors="pt").input_ids[0]
        windows += [ids[start:start + args.length] for start in range(0, len(ids) - 31, args.length)]
    if not windows:
        raise SystemExit("训练文本过短")
    print(f"{len(windows)} 段，{sum(len(window) for window in windows)} 个 token", flush=True)
    optimizer = torch.optim.Adam([
        {"params": scales, "lr": args.scale_lr},
        {"params": others, "lr": args.norm_lr},
    ])
    rates = [group["lr"] for group in optimizer.param_groups]
    started = time.time()
    average = None
    for step in range(args.steps):
        ids = random.choice(windows)[None, :]
        with torch.no_grad():
            target = functional.log_softmax(teacher(ids).logits[0], -1)
        prediction = functional.log_softmax(student(ids).logits[0], -1)
        loss = functional.kl_div(prediction, target, log_target=True, reduction="batchmean")
        optimizer.zero_grad()
        loss.backward()
        factor = 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate * factor
        optimizer.step()
        average = loss.item() if average is None else 0.9 * average + 0.1 * loss.item()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step}: KL={loss.item():.4f} 平均={average:.4f} {time.time() - started:.0f}s", flush=True)
    write(args.output, data, entries, lookup, np)
    print(f"已写入 {args.output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
