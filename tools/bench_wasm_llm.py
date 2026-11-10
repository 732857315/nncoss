import argparse
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from check_wasm_llm import ROOT, WasmTarget

PRESETS = {
    "tiny": {"vocab": 512, "prompt": 32, "decode": 32},
    "small": {"vocab": 4096, "prompt": 64, "decode": 64},
}


def write_model(preset, path):
    subprocess.run(
        [
            "cargo", "run", "-q", "-p", "nncoss-llm", "--release", "--example", "bench", "--",
            "--preset", preset, "--write-model", str(path),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def checked(target, status, label):
    if status != 0:
        message = target.read_c_string(target.call("nncoss_llm_last_error")).decode("utf-8")
        raise RuntimeError(f"WASM {label} 失败：{status} {message}")


def run_iteration(target, model, preset, scratch):
    prompt_length, decode_length = preset["prompt"], preset["decode"]
    session = target.call("nncoss_llm_session_new", model, prompt_length + decode_length)
    if session == 0:
        raise RuntimeError("WASM 会话创建失败")
    tokens = [(index * 31 + 7) % preset["vocab"] for index in range(prompt_length)]
    target.write(scratch, struct.pack(f"<{prompt_length}I", *tokens))
    started = time.perf_counter()
    checked(target, target.call("nncoss_llm_prefill", session, scratch, prompt_length), "prefill")
    prefill = (time.perf_counter() - started) * 1e3
    generated = []
    started = time.perf_counter()
    for _ in range(decode_length):
        checked(target, target.call("nncoss_llm_sample", session, scratch), "sample")
        token = struct.unpack("<I", target.read(scratch, 4))[0]
        checked(target, target.call("nncoss_llm_decode", session, token), "decode")
        generated.append(token)
    decode = (time.perf_counter() - started) * 1e3
    target.call("nncoss_llm_session_free", session)
    return prefill, decode, generated


def main():
    parser = argparse.ArgumentParser(description="WASM LLM 参考后端基准（tiny/small）")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--wasm", type=Path, default=ROOT / "target" / "wasm32-unknown-unknown" / "release" / "nncoss_llm.wasm")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations 需要正整数")
    preset = PRESETS[args.preset]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"{args.preset}.nncossllm"
        write_model(args.preset, path)
        model_bytes = path.read_bytes()

    target = WasmTarget(args.wasm)
    buffer = target.call("nncoss_llm_alloc", len(model_bytes))
    if buffer == 0:
        raise RuntimeError("WASM 模型缓冲区分配失败")
    target.write(buffer, model_bytes)
    load_samples, model = [], 0
    for iteration in range(args.iterations + 1):
        if model:
            target.call("nncoss_llm_model_free", model)
        started = time.perf_counter()
        model = target.call("nncoss_llm_model_load", buffer, len(model_bytes))
        elapsed = (time.perf_counter() - started) * 1e3
        if model == 0:
            raise RuntimeError("WASM 模型加载失败")
        if iteration > 0:
            load_samples.append(elapsed)
    target.call("nncoss_llm_dealloc", buffer, len(model_bytes))
    load_ms = statistics.median(load_samples)
    scratch_size = max(preset["prompt"], 1) * 4
    scratch = target.call("nncoss_llm_alloc", scratch_size)

    prefill_ms, decode_ms, generated = [], [], None
    for iteration in range(args.iterations + 1):
        prefill, decode, tokens = run_iteration(target, model, preset, scratch)
        if generated is not None and tokens != generated:
            raise RuntimeError("同一输入的贪心解码在迭代之间不一致")
        generated = tokens
        if iteration > 0:
            prefill_ms.append(prefill)
            decode_ms.append(decode)
    target.call("nncoss_llm_dealloc", scratch, scratch_size)
    target.call("nncoss_llm_model_free", model)

    checksum = 0
    for token in generated:
        checksum = (checksum * 31 + token) % (1 << 64)
    prefill = statistics.median(prefill_ms)
    decode = statistics.median(decode_ms)
    print("target: wasm32-unknown-unknown (wasmtime, Python 宿主调用开销计入 decode)")
    print(f"preset: {args.preset} model_bytes={len(model_bytes)} prompt={preset['prompt']} decode={preset['decode']} iterations={args.iterations}")
    print(f"load_ms: {load_ms:.2f}")
    print(f"prefill_ms: {prefill:.2f}  prefill_tok_s: {preset['prompt'] / prefill * 1e3:.1f}")
    print(f"decode_ms_per_token: {decode / preset['decode']:.3f}  decode_tok_s: {preset['decode'] / decode * 1e3:.1f}")
    print(f"greedy_checksum: {checksum}")


if __name__ == "__main__":
    main()
