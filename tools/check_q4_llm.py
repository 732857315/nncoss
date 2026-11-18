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
    CHAT.format(
        question="给我讲一个关于老渔夫和海鸥的小故事。",
        answer=(
            "清晨的海面还笼着一层薄雾，老渔夫王伯像往常一样推着小船下了海。一只受伤的海鸥落在船头，翅膀上缠着一截渔线。"
            "王伯放下桨，小心地解开渔线，又从饭盒里掰了半个馒头喂它。海鸥在船上待了三天，伤好后绕着小船飞了两圈才离开。"
            "从那以后，每当王伯出海，总有一只海鸥远远跟着，帮他找到鱼群最多的地方。"
        ),
    ),
    CHAT.format(
        question="Write a short email asking my manager for a day off next Friday.",
        answer=(
            "Subject: Request for a Day Off on Friday\n\nHi Sarah,\n\nI hope you are doing well. I would like to request a day "
            "off next Friday, March 14, to take care of a family matter. I will make sure my current tasks are completed before "
            "then, and Tom has agreed to cover any urgent requests while I am away. Please let me know if this works for you.\n\n"
            "Thank you,\nAlex"
        ),
    ),
    CHAT.format(
        question="请推导一元二次方程的求根公式。",
        answer=(
            "对于方程 ax^2 + bx + c = 0（a ≠ 0），先两边同时除以 a，得到 x^2 + (b/a)x + c/a = 0。"
            "配方：x^2 + (b/a)x + (b/2a)^2 = (b/2a)^2 - c/a，即 (x + b/2a)^2 = (b^2 - 4ac) / 4a^2。"
            "两边开平方得 x + b/2a = ±√(b^2 - 4ac) / 2a，所以 x = (-b ± √(b^2 - 4ac)) / 2a。"
            "其中 b^2 - 4ac 称为判别式，它决定了方程实数根的个数。"
        ),
    ),
    (
        '{\n  "name": "nncoss",\n  "version": "0.1.0",\n  "targets": ["wasm32-unknown-unknown", "x86_64-apple-darwin"],\n'
        '  "features": {\n    "quantization": "q4",\n    "threads": 6,\n    "speculative": true\n  },\n'
        '  "models": [\n    {"id": "qwen3-0.6b", "context": 40960, "format": "nncossllm"}\n  ]\n}\n'
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
    parser.add_argument("--min-top1", type=float, default=0.83, help="top-1 一致率下限")
    parser.add_argument("--max-mean-kl", type=float, default=0.14, help="平均 KL 上限")
    parser.add_argument("--max-ppl-ratio", type=float, default=1.06, help="Q4/f32 困惑度比值上限")
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
