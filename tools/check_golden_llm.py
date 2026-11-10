import argparse
import hashlib
import json
import struct
from pathlib import Path

from check_wasm_llm import (
    OK,
    ROOT,
    NativeTarget,
    WasmTarget,
    copy_logits,
    expect,
    load_model,
    native_library_name,
    prefill,
    sample,
)

GOLDEN_DIR = ROOT / "crates" / "nncoss-llm" / "tests" / "fixtures" / "golden"
GOLDEN_MAGIC = b"NNCSGLD\0"
GOLDEN_VERSION = 1
GOLDEN_HEADER = struct.Struct("<4I")
ABS_TOLERANCE = 1e-4
REL_TOLERANCE = 1e-4


class GoldenMismatch(AssertionError):
    pass


def encode_golden(vocab, prompt, greedy, logits):
    tokens = list(prompt) + list(greedy)
    if vocab <= 0 or len(prompt) < 2:
        raise ValueError("黄金数据需要非空词表与至少 2 个 prompt token")
    if any(token < 0 or token >= vocab for token in tokens):
        raise ValueError("黄金数据的 token 超出词表范围")
    if len(logits) != len(tokens) or any(len(row) != vocab for row in logits):
        raise ValueError("黄金数据的 logits 形状与 token 数量或词表不一致")
    flat = [value for row in logits for value in row]
    return b"".join((
        GOLDEN_MAGIC,
        GOLDEN_HEADER.pack(GOLDEN_VERSION, vocab, len(prompt), len(greedy)),
        struct.pack(f"<{len(tokens)}I", *tokens),
        struct.pack(f"<{len(flat)}f", *flat),
    ))


def decode_golden(data):
    start = len(GOLDEN_MAGIC)
    if data[:start] != GOLDEN_MAGIC or len(data) < start + GOLDEN_HEADER.size:
        raise ValueError("黄金数据 magic 不匹配或头部被截断")
    version, vocab, prompt_length, greedy_length = GOLDEN_HEADER.unpack_from(data, start)
    if version != GOLDEN_VERSION:
        raise ValueError(f"不支持的黄金数据版本：{version}")
    count = prompt_length + greedy_length
    if vocab == 0 or prompt_length < 2:
        raise ValueError("黄金数据需要非空词表与至少 2 个 prompt token")
    offset = start + GOLDEN_HEADER.size
    if len(data) != offset + count * 4 + count * vocab * 4:
        raise ValueError("黄金数据长度与头部声明不一致")
    tokens = list(struct.unpack_from(f"<{count}I", data, offset))
    if any(token >= vocab for token in tokens):
        raise ValueError("黄金数据的 token 超出词表范围")
    offset += count * 4
    flat = struct.unpack_from(f"<{count * vocab}f", data, offset)
    return {
        "vocab": vocab,
        "prompt": tokens[:prompt_length],
        "greedy": tokens[prompt_length:],
        "logits": [list(flat[row * vocab:(row + 1) * vocab]) for row in range(count)],
    }


def within_tolerance(actual, expected):
    return abs(actual - expected) <= ABS_TOLERANCE + REL_TOLERANCE * abs(expected)


def compare_logits(target, label, actual, expected):
    maximum = 0.0
    for index, (left, right) in enumerate(zip(actual, expected)):
        if not within_tolerance(left, right):
            raise GoldenMismatch(f"{target.name} {label} logit[{index}]={left!r} 与 PyTorch 参考 {right!r} 不一致")
        maximum = max(maximum, abs(left - right))
    return maximum


def checked(target, status, label):
    expect(status == OK, target, f"{label} 状态码为 {status}")


def read_logits(target, session, vocab, label):
    status, logits, _ = copy_logits(target, session, vocab)
    checked(target, status, f"{label} copy_logits")
    return logits


def run_case(target, name, model_bytes, golden):
    vocab, prompt, greedy, expected = golden["vocab"], golden["prompt"], golden["greedy"], golden["logits"]
    tokens = prompt + greedy
    model, message = load_model(target, model_bytes)
    expect(model != 0, target, f"{name} 模型加载失败：{message}")
    try:
        expect(target.call("nncoss_llm_model_vocab_size", model) == vocab, target, f"{name} 词表尺寸与黄金数据不一致")
        maximum = 0.0
        session = target.call("nncoss_llm_session_new", model, 0)
        expect(session != 0, target, f"{name} 会话创建失败")
        try:
            status, _ = prefill(target, session, tokens[:1])
            checked(target, status, f"{name} prefill")
            for position, token in enumerate(tokens):
                if position > 0:
                    checked(target, target.call("nncoss_llm_decode", session, token), f"{name} decode[{position}]")
                logits = read_logits(target, session, vocab, f"{name} 位置 {position}")
                maximum = max(maximum, compare_logits(target, f"{name} 增量位置 {position}", logits, expected[position]))
        finally:
            target.call("nncoss_llm_session_free", session)
        session = target.call("nncoss_llm_session_new", model, 0)
        expect(session != 0, target, f"{name} 会话创建失败")
        try:
            split = len(prompt) // 2
            for chunk in (prompt[:split], prompt[split:]):
                status, _ = prefill(target, session, chunk)
                checked(target, status, f"{name} 分段 prefill")
            position = len(prompt) - 1
            logits = read_logits(target, session, vocab, f"{name} prefill")
            maximum = max(maximum, compare_logits(target, f"{name} prefill 位置 {position}", logits, expected[position]))
            for step, token in enumerate(greedy):
                status, sampled, _ = sample(target, session)
                checked(target, status, f"{name} sample[{step}]")
                if sampled != token:
                    raise GoldenMismatch(f"{target.name} {name} 贪心第 {step} 步为 {sampled}，PyTorch 参考为 {token}")
                checked(target, target.call("nncoss_llm_decode", session, token), f"{name} 贪心 decode[{step}]")
                position = len(prompt) + step
                logits = read_logits(target, session, vocab, f"{name} 贪心位置 {position}")
                maximum = max(maximum, compare_logits(target, f"{name} 贪心位置 {position}", logits, expected[position]))
            expect(target.call("nncoss_llm_session_position", session) == len(tokens), target, f"{name} 会话位置错误")
        finally:
            target.call("nncoss_llm_session_free", session)
        return maximum
    finally:
        target.call("nncoss_llm_model_free", model)


def load_fixtures(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for entry in manifest["cases"]:
        name = entry["name"]
        model = (directory / f"{name}.nncossllm").read_bytes()
        golden = (directory / f"{name}.golden").read_bytes()
        if hashlib.sha256(model).hexdigest() != entry["model_sha256"]:
            raise ValueError(f"{name}.nncossllm 与 manifest.json 记录的 sha256 不一致")
        if hashlib.sha256(golden).hexdigest() != entry["golden_sha256"]:
            raise ValueError(f"{name}.golden 与 manifest.json 记录的 sha256 不一致")
        cases.append((name, model, decode_golden(golden)))
    if not cases:
        raise ValueError("manifest.json 未列出任何黄金用例")
    return manifest, cases


def main():
    parser = argparse.ArgumentParser(description="用已提交的 PyTorch 黄金数据对拍原生与 WASM LLM 后端")
    parser.add_argument("--native", type=Path, default=ROOT / "target" / "release" / native_library_name())
    parser.add_argument("--wasm", type=Path, default=ROOT / "target" / "wasm32-unknown-unknown" / "release" / "nncoss_llm.wasm")
    parser.add_argument("--fixtures", type=Path, default=GOLDEN_DIR, help="包含 manifest.json 的黄金夹具目录")
    args = parser.parse_args()
    manifest, cases = load_fixtures(args.fixtures)
    summary = []
    for target in (NativeTarget(args.native), WasmTarget(args.wasm)):
        maximum = max(run_case(target, name, model, golden) for name, model, golden in cases)
        summary.append(f"{target.name}最大 logits 误差={maximum:.3g}")
    generator = manifest["generator"]
    print(
        f"PyTorch 黄金对拍通过：{len(cases)} 个用例，{'，'.join(summary)}"
        f"（容差 {ABS_TOLERANCE:g} + {REL_TOLERANCE:g}×|参考值|；"
        f"参考实现 torch {generator['torch']}、transformers {generator['transformers']}）"
    )


if __name__ == "__main__":
    main()
