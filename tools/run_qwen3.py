import argparse
import time
from pathlib import Path

from tokenizers import Tokenizer

from check_real_llm import (
    ROOT,
    WASM_MEMORY_RESERVATION,
    checked,
    file_chunks,
    read_header,
)
from check_wasm_llm import NativeTarget, WasmTarget, native_library_name, prefill, sample, stream_model

DEFAULT_MODEL = ROOT / "model" / "Qwen3-0.6B" / "qwen3-0.6b-f32.nncossllm"
EOS_TOKENS = {151643, 151645}


def parse_args():
    parser = argparse.ArgumentParser(description="直接输入文字体验 Qwen3-0.6B 原生或 WASM 推理")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, help="默认使用模型同目录的 tokenizer.json")
    parser.add_argument("--target", choices=("native", "wasm"), default="wasm")
    parser.add_argument("--native", type=Path, default=ROOT / "target" / "release" / native_library_name())
    parser.add_argument("--wasm", type=Path, default=ROOT / "target" / "wasm32-unknown-unknown" / "release" / "nncoss_llm.wasm")
    parser.add_argument("--prompt", help="提供后只运行一次；省略则逐行交互输入")
    parser.add_argument("--raw", action="store_true", help="按文本补全运行，不套用 Qwen3 聊天标记")
    parser.add_argument("--think", action="store_true", help="允许 Qwen3 输出思考过程")
    parser.add_argument("--decode", type=int, default=64, help="最多生成的 token 数")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    return parser.parse_args()


def prompt_tokens(tokenizer, text, args):
    if not args.raw:
        thinking = "" if args.think else "<think>\n\n</think>\n\n"
        text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{thinking}"
    return tokenizer.encode(text, add_special_tokens=False).ids


def generate(target, model, tokenizer, header, text, args):
    tokens = prompt_tokens(tokenizer, text, args)
    if not tokens:
        return ""
    capacity = len(tokens) + args.decode + 1
    if capacity > header["max_context"]:
        raise ValueError(f"输入与生成长度超过模型上下文上限 {header['max_context']}")
    session = target.call("nncoss_llm_session_new", model, capacity)
    if not session:
        raise RuntimeError("会话创建失败")
    try:
        started = time.perf_counter()
        status, message = prefill(target, session, tokens)
        if status != 0:
            raise RuntimeError(message)
        checked(
            target,
            target.call(
                "nncoss_llm_set_sampler",
                session,
                args.temperature,
                args.top_k,
                args.top_p,
                args.seed,
            ),
            "采样配置",
        )
        generated = []
        for _ in range(args.decode):
            status, token, message = sample(target, session)
            if status != 0:
                raise RuntimeError(message)
            if token in EOS_TOKENS:
                break
            generated.append(token)
            checked(target, target.call("nncoss_llm_decode", session, token), "decode")
        elapsed = time.perf_counter() - started
        output = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\nQwen3: {output}\n")
        print(f"[{args.target}：prompt={len(tokens)} tokens，生成={len(generated)} tokens，耗时={elapsed:.2f}s]")
        return output
    finally:
        target.call("nncoss_llm_session_free", session)


def main():
    args = parse_args()
    if args.decode < 0 or args.chunk_bytes <= 0:
        raise SystemExit("--decode 需要非负整数，--chunk-bytes 需要正整数")
    header = read_header(args.model)
    tokenizer_path = args.tokenizer or args.model.parent / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    size = args.model.stat().st_size
    if args.target == "wasm":
        target = WasmTarget(args.wasm, memory_reservation=WASM_MEMORY_RESERVATION)
    else:
        target = NativeTarget(args.native)
    started = time.perf_counter()
    model, status, message = stream_model(target, file_chunks(args.model, args.chunk_bytes), size, args.chunk_bytes)
    if not model:
        raise RuntimeError(f"模型加载失败（{status}）：{message}")
    print(f"模型已加载：{args.target}，耗时 {time.perf_counter() - started:.2f}s")
    try:
        if args.prompt is not None:
            generate(target, model, tokenizer, header, args.prompt, args)
            return
        print("输入 :quit 退出；每次提问使用独立会话。")
        while True:
            try:
                text = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in (":quit", ":q", "/quit", "/q"):
                break
            if text:
                generate(target, model, tokenizer, header, text, args)
    finally:
        target.call("nncoss_llm_model_free", model)


if __name__ == "__main__":
    main()
