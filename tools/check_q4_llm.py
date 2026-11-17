import argparse
import sys
import time
from pathlib import Path

from check_real_llm import ROOT, checked, file_chunks
from check_wasm_llm import NativeTarget, native_library_name, stream_model, with_buffer

DEFAULT_DIR = ROOT / "model" / "Qwen3-0.6B"
CHAT = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>\n"
SAMPLES = (
    CHAT.format(
        question="请简单介绍一下长城的历史。",
        answer=(
            "长城是中国古代修建的大型军事防御工程。最早的长城出现在春秋战国时期，各诸侯国为了防御"
            "邻国和北方游牧民族而修筑城墙。秦始皇统一六国后，把各国的长城连接起来，形成了“万里长城”。"
            "此后汉、北魏、北齐、隋等朝代都曾修缮或扩建长城。今天我们看到的大部分长城建于明朝，"
            "西起嘉峪关，东至山海关，全长约八千八百多公里。1987 年，长城被列入世界文化遗产名录。"
        ),
    ),
    CHAT.format(
        question="What is the difference between a process and a thread?",
        answer=(
            "A process is an independent program in execution with its own address space, file "
            "descriptors and system resources. A thread is a unit of execution inside a process; "
            "all threads of a process share its memory and resources but each has its own stack "
            "and registers. Creating and switching between threads is usually cheaper than doing "
            "the same for processes, but shared memory means threads must synchronize access to "
            "common data, for example with mutexes or atomic operations."
        ),
    ),
    (
        "fn fibonacci(n: u64) -> u64 {\n"
        "    let (mut a, mut b) = (0, 1);\n"
        "    for _ in 0..n {\n"
        "        let next = a + b;\n"
        "        a = b;\n"
        "        b = next;\n"
        "    }\n"
        "    a\n"
        "}\n\n"
        "fn main() {\n"
        "    for i in 0..10 {\n"
        "        println!(\"fib({}) = {}\", i, fibonacci(i));\n"
        "    }\n"
        "}\n"
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description="比较 Q4 与 f32 模型在真实文本上的下一 token 分布")
    parser.add_argument("--reference", type=Path, default=DEFAULT_DIR / "qwen3-0.6b-f32.nncossllm")
    parser.add_argument("--model", type=Path, default=DEFAULT_DIR / "qwen3-0.6b-q4.nncossllm")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_DIR / "tokenizer.json")
    parser.add_argument("--native", type=Path, default=ROOT / "target" / "release" / native_library_name())
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--min-top1", type=float, default=0.78, help="top-1 一致率下限")
    parser.add_argument("--max-mean-kl", type=float, default=0.26, help="平均 KL 上限")
    parser.add_argument("--max-ppl-ratio", type=float, default=1.12, help="Q4/f32 困惑度比值上限")
    return parser.parse_args()


def load(target, path, chunk_bytes):
    started = time.perf_counter()
    model, status, message = stream_model(target, file_chunks(path, chunk_bytes), path.stat().st_size, chunk_bytes)
    if not model:
        raise RuntimeError(f"{path.name} 加载失败（{status}）：{message}")
    print(f"已加载 {path.name}，耗时 {time.perf_counter() - started:.1f}s")
    return model


def teacher_forced_logits(target, model, tokens, np):
    vocab = target.call("nncoss_llm_model_vocab_size", model)
    session = target.call("nncoss_llm_session_new", model, len(tokens))
    if not session:
        raise RuntimeError("会话创建失败")
    rows = np.empty((len(tokens) - 1, vocab), dtype=np.float32)
    try:

        def run(pointer):
            for index, token in enumerate(tokens[:-1]):
                checked(target, target.call("nncoss_llm_decode", session, token), "decode")
                checked(target, target.call("nncoss_llm_copy_logits", session, pointer, vocab), "copy_logits")
                rows[index] = np.frombuffer(target.read(pointer, vocab * 4), dtype="<f4")

        with_buffer(target, vocab * 4, run)
    finally:
        target.call("nncoss_llm_session_free", session)
    return rows


def log_softmax(rows, np):
    rows = rows.astype(np.float64)
    shifted = rows - rows.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def main():
    args = parse_args()
    try:
        import numpy as np
        from tokenizers import Tokenizer
    except ImportError as error:
        raise SystemExit(f"需要 numpy 与 tokenizers：{error}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    target = NativeTarget(args.native)
    reference = load(target, args.reference, args.chunk_bytes)
    model = load(target, args.model, args.chunk_bytes)
    agree = total = 0
    kl_values = []
    nll = {"f32": 0.0, "q4": 0.0}
    try:
        for index, text in enumerate(SAMPLES):
            tokens = tokenizer.encode(text, add_special_tokens=False).ids
            expected = log_softmax(teacher_forced_logits(target, reference, tokens, np), np)
            actual = log_softmax(teacher_forced_logits(target, model, tokens, np), np)
            targets = np.array(tokens[1:])
            positions = np.arange(len(targets))
            sample_agree = int((expected.argmax(axis=1) == actual.argmax(axis=1)).sum())
            kl = (np.exp(expected) * (expected - actual)).sum(axis=1)
            agree += sample_agree
            total += len(targets)
            kl_values.extend(kl.tolist())
            nll["f32"] -= expected[positions, targets].sum()
            nll["q4"] -= actual[positions, targets].sum()
            print(
                f"样本 {index}：{len(targets)} 个位置，top-1 一致 {sample_agree}/{len(targets)}，"
                f"平均 KL={kl.mean():.4f}，最大 KL={kl.max():.4f}"
            )
    finally:
        target.call("nncoss_llm_model_free", model)
        target.call("nncoss_llm_model_free", reference)
    top1 = agree / total
    mean_kl = sum(kl_values) / len(kl_values)
    ppl_f32 = float(np.exp(nll["f32"] / total))
    ppl_q4 = float(np.exp(nll["q4"] / total))
    print(
        f"合计 {total} 个位置：top-1 一致率={top1:.3f}，平均 KL={mean_kl:.4f}，最大 KL={max(kl_values):.4f}，"
        f"困惑度 f32={ppl_f32:.3f} Q4={ppl_q4:.3f}（比值 {ppl_q4 / ppl_f32:.3f}）"
    )
    failures = []
    if top1 < args.min_top1:
        failures.append(f"top-1 一致率 {top1:.3f} 低于 {args.min_top1}")
    if mean_kl > args.max_mean_kl:
        failures.append(f"平均 KL {mean_kl:.4f} 高于 {args.max_mean_kl}")
    if ppl_q4 / ppl_f32 > args.max_ppl_ratio:
        failures.append(f"困惑度比值 {ppl_q4 / ppl_f32:.3f} 高于 {args.max_ppl_ratio}")
    if failures:
        print("Q4 质量验收失败：" + "；".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print("Q4 质量验收通过")


if __name__ == "__main__":
    main()
