"""定位让模型陷入重复循环的神经元与注意力头，给出 nncoss-edit 改动计划；只测量，不改参数。

快速考评中 0.6B 的长回答常常卡在同一行上反复输出（如 “E．”），采样也跳不出来。这里用诱发重复来测量：
取模型自己在定位题上的回答，在其中一句之后把这句再重复两遍，测量模型第三次重复它的平均对数概率 R；
同时测量不加重复时模型自己回答的对数概率 K，要求改动后 K 基本不变。
对 R 求各 MLP 神经元与 KV 头缩放系数的梯度，沿降低 R 的方向缩放影响最大的单元。
定位题对半分：A 半算梯度，B 半比较候选；另用真实出现过的重复回答检验。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import retrieval_gate


def repeated_samples(prompts, replies, repeats):
    samples = []
    for key, reply in replies.items():
        if key not in prompts:
            continue
        lines = [line for line in reply.split("\n") if 8 <= len(line.strip()) <= 120]
        if len(lines) < 3:
            continue
        line = lines[len(lines) // 2]
        cut = reply.index(line) + len(line) + 1
        context = reply[:cut] + (line + "\n") * repeats
        samples.append({"id": key, "prompt": prompts[key] + context, "target": line + "\n",
                        "plain": prompts[key], "own": reply})
    return samples


def real_loops(paths):
    loops = []
    for path in paths:
        for record in retrieval_gate.read_jsonl(path):
            transcript, reply = record.get("transcript", ""), record["reply"]
            lines = [line for line in reply.split("\n") if line.strip()]
            if len(lines) < 12 or len(set(lines[-10:])) > 3:
                continue
            line = lines[-2]
            start = transcript.rfind(reply[:200]) if reply[:200] in transcript else -1
            if start < 0:
                continue
            body = transcript[:start] + reply
            first = body.index(line + "\n") + len(line) + 1
            loops.append({"id": record["id"], "prompt": body[:first] + (line + "\n") * 2, "target": line + "\n"})
    return loops


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True, help="定位题上不用工具时的作答结果 answers.jsonl")
    parser.add_argument("--plain-prompts", type=Path, required=True, help="同一批题的开口提示（nncoss-know prompt --no-tools）")
    parser.add_argument("--loops", type=Path, nargs="*", default=[], help="含真实重复循环的作答记录，只用于检验")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/repetition_edit")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--neurons", type=int, nargs="+", default=[0, 16, 64, 256])
    parser.add_argument("--heads", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--max-drop", type=float, default=0.05)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = {record["id"]: record["prompt"] for record in retrieval_gate.read_jsonl(args.plain_prompts)}
    replies = {record["id"]: record["reply"] for record in retrieval_gate.read_jsonl(args.own)}
    samples = repeated_samples(prompts, replies, args.repeats)
    half_a, half_b = samples[0::2], samples[1::2]
    loops = real_loops(args.loops)
    print(f"诱发重复样本 {len(samples)}：A 半 {len(half_a)}，B 半 {len(half_b)}；真实重复 {len(loops)} 例", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mlp, values = retrieval_gate.attach(model)
    scorer = retrieval_gate.Scorer(model, tokenizer, args.tokens)

    for number, sample in enumerate(half_a):
        repeat = scorer.logp(sample["prompt"], sample["target"])
        (-repeat).backward()
        print(f"梯度 {number + 1}/{len(half_a)} R={float(repeat):.3f}", flush=True)
    mlp_grad = (mlp.grad / len(half_a)).numpy()
    value_grad = (values.grad / len(half_a)).numpy()
    np.savez(args.work / "effects.npz", mlp=mlp_grad, values=value_grad)

    def evaluate(mlp_scales, value_scales):
        mlp.data = torch.tensor(mlp_scales, dtype=torch.float32)
        values.data = torch.tensor(value_scales, dtype=torch.float32)
        repeat = [scorer.logp_many(sample["prompt"], [sample["target"]])[0] for sample in half_b]
        keep = [scorer.logp_many(sample["plain"], [sample["own"]])[0] for sample in half_b]
        real = [scorer.logp_many(loop["prompt"], [loop["target"]])[0] for loop in loops]
        return float(np.mean(repeat)), float(np.mean(keep)), float(np.mean(real)) if real else None

    base_repeat, base_keep, base_real = evaluate(np.ones_like(mlp_grad), np.ones_like(value_grad))
    print(f"未改动：重复 R={base_repeat:.3f}，真实循环 {base_real}，自身回答 K={base_keep:.3f}", flush=True)
    table, best = [], None
    for setting, mlp_scales, value_scales in retrieval_gate.candidates(mlp_grad, value_grad, args.neurons, args.heads,
                                                                       args.deltas):
        repeat, keep, real = evaluate(mlp_scales, value_scales)
        row = {**setting, "repeat": round(repeat, 4), "repeat_drop": round(base_repeat - repeat, 4),
               "real_loop": None if real is None else round(real, 4), "keep": round(keep, 4),
               "keep_drop": round(base_keep - keep, 4)}
        table.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["keep_drop"] <= args.max_drop and row["repeat_drop"] > 0 and (best is None or repeat < best[0]["repeat"]):
            best = (row, retrieval_gate.plan(mlp_scales, value_scales))
    summary = {"samples": len(samples), "loops": [loop["id"] for loop in loops],
               "baseline": {"repeat": base_repeat, "keep": base_keep, "real_loop": base_real},
               "candidates": table, "chosen": best[0] if best else None}
    (args.work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if best:
        (args.work / "plan.json").write_text(json.dumps(best[1], ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"选中 {best[0]}，改动计划 {len(best[1])} 项 -> {args.work / 'plan.json'}")
    else:
        print("没有候选在不损害自身回答的前提下降低重复")


if __name__ == "__main__":
    main()
