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
    parser = argparse.ArgumentParser(description="将 Llama/Qwen 风格的 safetensors 权重复制为 nncoss LLM f32 或 Q4 模型")
    parser.add_argument("--model", type=Path, required=True, help="包含 config.json 和 safetensors 的本地模型目录")
    parser.add_argument("--output", type=Path, required=True, help="输出的 .nncossllm 文件")
    parser.add_argument("--outtype", choices=("f32", "q4"), default="f32", help="权重编码，默认 f32")
    parser.add_argument(
        "--calibration",
        type=Path,
        nargs="+",
        help="Q4 校准文本（UTF-8），提供后按各层输入统计做 GPTQ 量化，需要 transformers",
    )
    parser.add_argument(
        "--calibration-tokens",
        type=int,
        default=65536,
        help="校准使用的最多 token 数，按文件平均分配",
    )
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


Q4_GROUP = 32
Q4_CHUNK_ROWS = 4096
Q4_STEP_MULTIPLIERS = tuple(0.8 + 0.025 * index for index in range(17))
Q4_ROW_HEADROOM = 1.3
Q4_GPTQ_DAMP = 0.01
Q4_GPTQ_BLOCK = 128
Q4_GPTQ_HEADROOM = 1.5
CALIBRATION_WINDOW = 512


def quantize_q4(array):
    import numpy as np

    rows, columns = array.shape
    groups = (columns + Q4_GROUP - 1) // Q4_GROUP
    if not np.isfinite(array).all():
        raise ValueError("权重包含非有限值")
    padded = np.zeros((rows, groups * Q4_GROUP), dtype=np.float32)
    padded[:, :columns] = array
    blocks = padded.reshape(rows, groups, Q4_GROUP)
    base = np.abs(blocks).max(axis=2) / np.float32(7.0)
    row_scale = (base.max(axis=1) * np.float32(Q4_ROW_HEADROOM / 127.0)).astype(np.float32)
    if not np.isfinite(row_scale).all():
        raise ValueError("Q4 行 scale 超出 float32 范围")
    safe_scale = np.where(row_scale > 0, row_scale, np.float32(1.0))[:, None]
    best_error = np.full(base.shape, np.inf, dtype=np.float32)
    best_code = np.zeros(base.shape, dtype=np.int8)
    for multiplier in Q4_STEP_MULTIPLIERS:
        for sign in (1.0, -1.0):
            trial = base * np.float32(multiplier * sign)
            trial = np.where(trial != 0, trial, np.float32(1.0))
            quantized = np.clip(np.rint(blocks / trial[:, :, None]), -8, 7)
            numerator = (blocks * quantized).sum(axis=2)
            denominator = (quantized * quantized).sum(axis=2)
            fitted = numerator / np.where(denominator > 0, denominator, np.float32(1.0))
            code = np.clip(np.rint(fitted / safe_scale), -127, 127)
            step = (code * safe_scale).astype(np.float32)
            divisor = np.where(step != 0, step, np.float32(1.0))
            requantized = np.clip(np.rint(blocks / divisor[:, :, None]), -8, 7)
            error = ((requantized * step[:, :, None] - blocks) ** 2).sum(axis=2)
            better = error < best_error
            best_error = np.where(better, error, best_error)
            best_code = np.where(better, code.astype(np.int8), best_code)
    step = (best_code.astype(np.float32) * safe_scale).astype(np.float32)
    divisor = np.where(step != 0, step, np.float32(1.0))
    quantized = np.clip(np.rint(blocks / divisor[:, :, None]), -8, 7).astype(np.int8)
    quantized = np.where(step[:, :, None] != 0, quantized, 0).astype(np.int8)
    unsigned = (quantized + 8).astype(np.uint8)
    packed = unsigned[:, :, 0::2] | (unsigned[:, :, 1::2] << 4)
    return row_scale, best_code.view(np.uint8), packed.astype(np.uint8)


def search_group_codes(block, row_scale, torch):
    base = block.abs().amax(dim=1) / 7.0
    safe_scale = torch.where(row_scale > 0, row_scale, torch.ones_like(row_scale))
    best_error = torch.full_like(base, float("inf"))
    best_code = torch.zeros_like(base)
    for multiplier in Q4_STEP_MULTIPLIERS:
        for sign in (1.0, -1.0):
            trial = base * (multiplier * sign)
            trial = torch.where(trial != 0, trial, torch.ones_like(trial))
            quantized = torch.clamp(torch.round(block / trial[:, None]), -8, 7)
            fitted = (block * quantized).sum(1) / torch.clamp((quantized * quantized).sum(1), min=1e-30)
            code = torch.clamp(torch.round(fitted / safe_scale), -127, 127)
            step = code * safe_scale
            divisor = torch.where(step != 0, step, torch.ones_like(step))
            requantized = torch.clamp(torch.round(block / divisor[:, None]), -8, 7)
            error = ((requantized * step[:, None] - block) ** 2).sum(1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_code = torch.where(better, code, best_code)
    return best_code


def quantize_q4_gptq(array, hessian, torch):
    import numpy as np

    rows, columns = array.shape
    groups = (columns + Q4_GROUP - 1) // Q4_GROUP
    width = groups * Q4_GROUP
    weights = torch.zeros(rows, width, dtype=torch.float64)
    weights[:, :columns] = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float64))
    if not torch.isfinite(weights).all():
        raise ValueError("权重包含非有限值")
    matrix = torch.zeros(width, width, dtype=torch.float64)
    matrix[:columns, :columns] = hessian
    dead = torch.diag(matrix) <= 0
    matrix[dead, dead] = 1.0
    weights[:, dead] = 0.0
    matrix += Q4_GPTQ_DAMP * torch.mean(torch.diag(matrix)) * torch.eye(width, dtype=torch.float64)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(matrix)), upper=True)
    base = weights.reshape(rows, groups, Q4_GROUP).abs().amax(dim=2) / 7.0
    row_scale = (base.amax(dim=1) * Q4_GPTQ_HEADROOM / 127.0).float().double()
    safe_scale = torch.where(row_scale > 0, row_scale, torch.ones_like(row_scale))
    codes = torch.zeros(rows, groups, dtype=torch.float64)
    quantized = torch.zeros(rows, width, dtype=torch.float64)
    step = None
    for first in range(0, width, Q4_GPTQ_BLOCK):
        last = min(first + Q4_GPTQ_BLOCK, width)
        block = weights[:, first:last].clone()
        errors = torch.zeros_like(block)
        local = inverse[first:last, first:last]
        for offset in range(last - first):
            column = first + offset
            if column % Q4_GROUP == 0:
                code = search_group_codes(block[:, offset:offset + Q4_GROUP], row_scale, torch)
                codes[:, column // Q4_GROUP] = code
                step = code * safe_scale
            values = block[:, offset]
            divisor = torch.where(step != 0, step, torch.ones_like(step))
            levels = torch.where(step != 0, torch.clamp(torch.round(values / divisor), -8, 7), torch.zeros_like(values))
            quantized[:, column] = levels
            error = (values - levels * step) / local[offset, offset]
            block[:, offset:] -= error[:, None] * local[offset, offset:][None, :]
            errors[:, offset] = error
        weights[:, last:] -= errors @ inverse[first:last, last:]
    unsigned = (quantized.to(torch.int16) + 8).to(torch.uint8).reshape(rows, groups, Q4_GROUP).numpy()
    packed = unsigned[:, :, 0::2] | (unsigned[:, :, 1::2] << 4)
    return row_scale.float().numpy(), codes.to(torch.int8).numpy().view(np.uint8), packed.astype(np.uint8)


def collect_statistics(model_dir, paths, limit, spec, torch):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("--calibration 需要 transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    windows, total = [], 0
    share = max(limit // len(paths), 2)
    for path in paths:
        ids = tokenizer(path.read_text(encoding="utf-8"), return_tensors="pt").input_ids[0][:share]
        for start in range(0, len(ids), CALIBRATION_WINDOW):
            window = ids[start:start + CALIBRATION_WINDOW]
            if len(window) >= 2:
                windows.append(window)
                total += len(window)
    if total == 0:
        raise ValueError("校准文本为空")
    sums, counts, names = {}, {}, {}

    def record(key):
        def hook(module, inputs, output):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).double()
            product = (values.T @ values).float()
            sums[key] = product if key not in sums else sums[key] + product
            counts[key] = counts.get(key, 0) + values.shape[0]
        return hook

    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        groups = {
            "qkv": (layer.self_attn.q_proj, [f"{prefix}.self_attn.{part}.weight" for part in ("q_proj", "k_proj", "v_proj")]),
            "o": (layer.self_attn.o_proj, [f"{prefix}.self_attn.o_proj.weight"]),
            "gate_up": (layer.mlp.gate_proj, [f"{prefix}.mlp.{part}.weight" for part in ("gate_proj", "up_proj")]),
            "down": (layer.mlp.down_proj, [f"{prefix}.mlp.down_proj.weight"]),
        }
        for kind, (module, tensors) in groups.items():
            key = f"{prefix}.{kind}"
            module.register_forward_hook(record(key))
            names.update({tensor: key for tensor in tensors})
    model.lm_head.register_forward_hook(record("head"))
    names["model.embed_tokens.weight" if spec["tied"] else "lm_head.weight"] = "head"
    with torch.no_grad():
        for window in windows:
            model(window[None, :])
    statistics = {tensor: (sums[key], counts[key]) for tensor, key in names.items()}
    print(f"已收集校准统计：{len(windows)} 段，{total} 个 token")
    return statistics


def write_q4_matrix(output, sources, name, shape, torch, statistics=None):
    if len(shape) != 2:
        raise ValueError(f"Q4 只支持二维矩阵：{name}")
    tensor = read_tensor(sources, name)
    if list(tensor.shape) != list(shape):
        raise ValueError(f"权重 {name} 形状为 {list(tensor.shape)}，预期为 {list(shape)}")
    array = tensor.detach().to(dtype=torch.float32).cpu().contiguous().numpy()
    entry = None if statistics is None else statistics.get(name)
    hessian = None if entry is None else entry[0].double() / entry[1]
    scales, codes, nibbles = [], [], []
    for start in range(0, shape[0], Q4_CHUNK_ROWS):
        rows = array[start:start + Q4_CHUNK_ROWS]
        if hessian is None:
            row_scale, row_codes, row_nibbles = quantize_q4(rows)
        else:
            row_scale, row_codes, row_nibbles = quantize_q4_gptq(rows, hessian, torch)
        scales.append(row_scale.astype("<f4").tobytes())
        codes.append(row_codes.tobytes())
        nibbles.append(row_nibbles.tobytes())
    written = 0
    for section in (scales, codes, nibbles):
        for chunk in section:
            output.write(chunk)
            written += len(chunk)
    output.write(bytes(-written % 4))


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


def export(model_dir, output_path, outtype="f32", calibration=None, calibration_tokens=65536):
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
            statistics = None
            if calibration and outtype == "q4":
                statistics = collect_statistics(model_dir, calibration, calibration_tokens, spec, torch)
            flags = int(spec["tied"]) | (int(has_qkv_bias) << 1) | (int(has_qk_norm) << 2)
            if outtype == "q4":
                flags |= 8
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
                    if outtype == "q4" and len(shape) == 2:
                        write_q4_matrix(output, sources, name, shape, torch, statistics)
                    else:
                        write_tensor(output, sources, name, shape, torch)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"已导出 {output_path}：type={spec['model_type']}, layers={spec['layers']}, hidden={spec['hidden']}, "
        f"heads={spec['heads']}/{spec['kv_heads']}, vocab={spec['vocab']}, rope=HalfSplit, {outtype}"
        + (", GPTQ" if calibration and outtype == "q4" else "")
    )


def main():
    args = parse_args()
    if args.calibration and args.outtype != "q4":
        raise SystemExit("--calibration 只用于 --outtype q4")
    if args.calibration_tokens <= 0:
        raise SystemExit("--calibration-tokens 需要正整数")
    export(args.model, args.output, args.outtype, args.calibration, args.calibration_tokens)


if __name__ == "__main__":
    main()
