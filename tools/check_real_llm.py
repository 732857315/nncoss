import argparse
import os
import struct
import subprocess
import time
from pathlib import Path

from check_golden_llm import ABS_TOLERANCE, REL_TOLERANCE
from check_wasm_llm import (
    OK,
    ROOT,
    TOKEN_OUT_OF_RANGE,
    NativeTarget,
    WasmTarget,
    copy_logits,
    expect,
    last_error,
    native_library_name,
    prefill,
    sample,
    stream_model,
)

DEFAULT_MODEL = ROOT / "model" / "Qwen3-0.6B" / "qwen3-0.6b-f32.nncossllm"
DEFAULT_PROMPT = [785, 6722, 315, 9625, 374]
HEADER = struct.Struct("<8s12I2f")
MODEL_MAGIC = b"NNCSLLM\0"
CONTEXT_LENGTH = -4
WASM32_MAX_ALLOCATION = 2**31 - 1
WASM32_MAX_MEMORY = 2**32
WASM_MEMORY_RESERVATION = 2**32
ROLLBACK_MARGIN = 2


def read_header(path):
    with path.open("rb") as file:
        data = file.read(HEADER.size)
    if len(data) != HEADER.size:
        raise ValueError("模型文件头被截断")
    fields = HEADER.unpack(data)
    if fields[0] != MODEL_MAGIC:
        raise ValueError("模型文件 magic 不匹配")
    names = ("vocab", "hidden", "intermediate", "layers", "heads", "kv_heads", "head_dim", "max_context", "flags")
    return dict(zip(names, fields[3:12]))


def largest_tensor_bytes(header):
    hidden = header["hidden"]
    rows = max(header["vocab"], header["intermediate"], header["heads"] * header["head_dim"])
    return rows * hidden * 4


def file_chunks(path, chunk_size):
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                return
            yield chunk


def resident_bytes():
    output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(output.decode("ascii").strip()) * 1024


def checked(target, status, label):
    expect(status == OK, target, f"{label} 状态码为 {status}：{last_error(target) if status != OK else ''}")


def read_logits(target, session, vocab, label):
    status, logits, _ = copy_logits(target, session, vocab)
    checked(target, status, f"{label} copy_logits")
    return logits


def probe_whole_buffer(target, size):
    if size <= WASM32_MAX_ALLOCATION:
        return None
    pointer = target.call("nncoss_llm_alloc", size)
    expect(pointer == 0, target, f"整块缓冲区 {size} 字节在 wasm32 上被意外接受")
    return last_error(target)


def check_rollback(target, session, vocab, prompt, position, logits):
    status, _ = prefill(target, session, [prompt[0], vocab])
    expect(status == TOKEN_OUT_OF_RANGE, target, f"越界 token 的 prefill 状态码为 {status}")
    status, _ = prefill(target, session, prompt[:1] * (ROLLBACK_MARGIN + 1))
    expect(status == CONTEXT_LENGTH, target, f"超出上下文的 prefill 状态码为 {status}")
    expect(target.call("nncoss_llm_session_position", session) == position, target, "失败的 prefill 改变了位置")
    expect(read_logits(target, session, vocab, "回滚后") == logits, target, "失败的 prefill 改变了 logits")


def run_target(target, path, header, prompt, decode_count, chunk_size):
    size = path.stat().st_size
    vocab = header["vocab"]
    is_wasm = isinstance(target, WasmTarget)
    report = {"name": target.name}
    if is_wasm:
        report["memory_initial"] = target.memory_bytes()
        report["whole_buffer_error"] = probe_whole_buffer(target, size)
    resident_before = resident_bytes()
    started = time.perf_counter()
    model, status, message = stream_model(target, file_chunks(path, chunk_size), size, chunk_size)
    report["load_seconds"] = time.perf_counter() - started
    expect(model != 0, target, f"分块加载失败（状态码 {status}）：{message}")
    report["resident_delta"] = resident_bytes() - resident_before
    if is_wasm:
        report["memory_loaded"] = target.memory_bytes()
    expect(target.call("nncoss_llm_model_vocab_size", model) == vocab, target, "词表尺寸与文件头不一致")
    expect(target.call("nncoss_llm_model_max_context", model) == header["max_context"], target, "上下文上限与文件头不一致")
    capacity = len(prompt) + decode_count + ROLLBACK_MARGIN
    session = target.call("nncoss_llm_session_new", model, capacity)
    target.call("nncoss_llm_model_free", model)
    expect(session != 0, target, "会话创建失败：" + last_error(target))
    try:
        started = time.perf_counter()
        status, message = prefill(target, session, prompt)
        report["prefill_seconds"] = time.perf_counter() - started
        expect(status == OK, target, f"prefill 状态码为 {status}：{message}")
        rows = [read_logits(target, session, vocab, "prefill")]
        tokens = []
        started = time.perf_counter()
        for step in range(decode_count):
            status, token, message = sample(target, session)
            expect(status == OK, target, f"sample[{step}] 状态码为 {status}：{message}")
            checked(target, target.call("nncoss_llm_decode", session, token), f"decode[{step}]")
            tokens.append(token)
            rows.append(read_logits(target, session, vocab, f"decode[{step}]"))
        report["decode_seconds"] = time.perf_counter() - started
        position = len(prompt) + decode_count
        expect(target.call("nncoss_llm_session_position", session) == position, target, "会话位置错误")
        check_rollback(target, session, vocab, prompt, position, rows[-1])
    finally:
        target.call("nncoss_llm_session_free", session)
    if is_wasm:
        report["memory_peak"] = target.memory_bytes()
    report["tokens"] = tokens
    report["rows"] = rows
    return report


def compare_rows(label, actual, expected):
    maximum = 0.0
    for row, (left_row, right_row) in enumerate(zip(actual, expected)):
        if len(left_row) != len(right_row):
            raise AssertionError(f"{label} 第 {row} 行 logits 长度不一致")
        for index, (left, right) in enumerate(zip(left_row, right_row)):
            error = abs(left - right)
            if error > ABS_TOLERANCE + REL_TOLERANCE * abs(right):
                raise AssertionError(f"{label} 第 {row} 行 logit[{index}]={left!r} 与参考 {right!r} 不一致")
            maximum = max(maximum, error)
    return maximum


def reference_rows(model_dir, prompt, tokens):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise SystemExit("Hugging Face 参考对拍依赖 tools/golden_requirements.txt 中的 torch 与 transformers") from error
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32, attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([prompt + tokens])).logits[0, len(prompt) - 1:]
    greedy = logits[:len(tokens)].argmax(dim=-1).tolist()
    return logits.tolist(), greedy


def megabytes(value):
    return f"{value} 字节（{value / 2**20:.1f} MiB）"


def print_report(report):
    name = report["name"]
    print(
        f"{name}：分块加载 {report['load_seconds']:.2f}s，prefill {report['prefill_seconds']:.2f}s，"
        f"decode {report['decode_seconds']:.2f}s，贪心 token={report['tokens']}，"
        f"宿主常驻内存增量 {megabytes(report['resident_delta'])}"
    )
    if "memory_initial" not in report:
        return
    whole = report["whole_buffer_error"]
    if whole is not None:
        print(f"{name}：整块输入缓冲区分配被拒绝（{whole}）")
    print(
        f"{name}：线性内存 初始 {megabytes(report['memory_initial'])}，加载后 {megabytes(report['memory_loaded'])}，"
        f"推理后峰值 {megabytes(report['memory_peak'])}，wasm32 上限 {megabytes(WASM32_MAX_MEMORY)}"
    )


def parse_tokens(text):
    tokens = [int(item) for item in text.split(",") if item.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("至少需要 1 个 token")
    return tokens


def main():
    parser = argparse.ArgumentParser(description="以固定大小分块加载真实 .nncossllm 模型，并对拍原生与 WASM 的 prefill／贪心 decode")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--native", type=Path, default=ROOT / "target" / "release" / native_library_name())
    parser.add_argument("--wasm", type=Path, default=ROOT / "target" / "wasm32-unknown-unknown" / "release" / "nncoss_llm.wasm")
    parser.add_argument("--targets", default="native,wasm", help="逗号分隔的目标：native、wasm")
    parser.add_argument("--prompt", type=parse_tokens, default=DEFAULT_PROMPT, help="逗号分隔的 prompt token id")
    parser.add_argument("--decode", type=int, default=4, help="贪心 decode 的 token 数")
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024, help="每次写入加载器的固定分块大小")
    parser.add_argument("--hf-model", type=Path, help="可选：用于 f32 eager 参考前向的 Hugging Face 模型目录")
    args = parser.parse_args()
    names = [name.strip() for name in args.targets.split(",") if name.strip()]
    if not names or any(name not in ("native", "wasm") for name in names):
        parser.error("--targets 只接受 native 与 wasm")
    if args.decode < 0 or args.chunk_bytes <= 0:
        parser.error("--decode 需要非负整数，--chunk-bytes 需要正整数")

    header = read_header(args.model)
    size = args.model.stat().st_size
    if any(token < 0 or token >= header["vocab"] for token in args.prompt):
        parser.error("prompt token 超出词表范围")
    print(
        f"模型 {args.model.name}：{megabytes(size)}，词表 {header['vocab']}，hidden {header['hidden']}，"
        f"{header['layers']} 层；分块 {args.chunk_bytes} 字节，最大单个张量 {megabytes(largest_tensor_bytes(header))}"
    )
    reports = []
    for name in names:
        if name == "native":
            target = NativeTarget(args.native)
        else:
            target = WasmTarget(args.wasm, memory_reservation=WASM_MEMORY_RESERVATION)
        report = run_target(target, args.model, header, args.prompt, args.decode, args.chunk_bytes)
        print_report(report)
        reports.append(report)
        del target

    first = reports[0]
    for other in reports[1:]:
        if other["tokens"] != first["tokens"]:
            raise AssertionError(f"{first['name']}/{other['name']} 贪心 token 不一致：{first['tokens']} != {other['tokens']}")
        maximum = compare_rows(f"{other['name']} 对 {first['name']}", other["rows"], first["rows"])
        print(f"{first['name']}/{other['name']} 对拍通过：{len(first['rows'])} 组 logits，最大误差={maximum:.3e}")
    if args.hf_model:
        rows, greedy = reference_rows(args.hf_model, args.prompt, first["tokens"])
        for report in reports:
            if report["tokens"] != greedy:
                raise AssertionError(f"{report['name']} 贪心 token {report['tokens']} 与 Hugging Face 参考 {greedy} 不一致")
            maximum = compare_rows(f"{report['name']} 对 Hugging Face", report["rows"], rows)
            first_row = max(abs(left - right) for left, right in zip(report["rows"][0], rows[0]))
            print(
                f"{report['name']}/Hugging Face 对拍通过：{len(greedy)} 个贪心 token 一致，"
                f"首组 logits 最大误差={first_row:.3e}，全部 {len(rows)} 组最大误差={maximum:.3e}"
            )
    print(f"真实模型验收通过（容差 {ABS_TOLERANCE:g} + {REL_TOLERANCE:g}×|参考值|）")


if __name__ == "__main__":
    main()
