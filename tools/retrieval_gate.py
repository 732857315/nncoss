"""定位“检索框架压制自身推理”的神经元与注意力头，只测量，不改参数。

空知识库对照显示：对话里出现一次检索、结果为空时，0.6B 几乎全部拒答，
不再像没有工具时那样自己推理。本脚本以模型自己的两种回答为目标，不使用标准答案：
  own     —— 没有工具时模型自己给出的回答开头
  refusal —— 检索结果为空时模型给出的拒答开头
在“检索结果为空”的提示后强制续写这两段，梯度归因 J = logp(own) - logp(refusal)
对每个 MLP 神经元与 KV 头缩放系数的导数。

定位题再对半分：A 半用来算梯度，B 半用来比较候选改动。候选改动按 |梯度| 选前 k 个单元，
沿梯度方向缩放 (1 ± δ)；要求不使用工具时自身回答的对数概率基本不变。
选出的改动写成 nncoss-edit 的改动计划。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_gaokao


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prompts(know, questions, work, name, extra):
    source = work / f"{name}_questions.jsonl"
    target = work / f"{name}_prompts.jsonl"
    source.write_text("".join(json.dumps({"id": q["id"], "question": q["question"]}, ensure_ascii=False) + "\n"
                              for q in questions), encoding="utf-8")
    subprocess.run([str(know), "prompt", "--input", str(source), "--output", str(target), *extra], check=True)
    return {record["id"]: record["prompt"] for record in read_jsonl(target)}


def attach(model):
    import torch
    config = model.config
    layers = model.model.layers
    mlp = torch.ones(len(layers), config.intermediate_size, requires_grad=True)
    values = torch.ones(len(layers), config.num_key_value_heads, requires_grad=True)
    for index, layer in enumerate(layers):
        layer.mlp.down_proj.register_forward_pre_hook(lambda module, inputs, index=index: (inputs[0] * mlp[index],))
        layer.self_attn.v_proj.register_forward_hook(
            lambda module, inputs, output, index=index: output * values[index].repeat_interleave(config.head_dim))
    return mlp, values


class Scorer:
    def __init__(self, model, tokenizer, count):
        self.model, self.tokenizer, self.count = model, tokenizer, count

    def ids(self, text, limit=None):
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return ids[:limit] if limit else ids

    def logp(self, prompt, text):
        """可求导：强制续写 text 的前 count 个 token，返回平均对数概率。"""
        import torch
        import torch.nn.functional as F
        head, tail = self.ids(prompt), self.ids(text, self.count)
        logits = self.model(torch.tensor([head + tail])).logits[0, len(head) - 1:len(head) - 1 + len(tail)]
        return F.log_softmax(logits, -1).gather(1, torch.tensor(tail)[:, None]).mean()

    def logp_many(self, prompt, texts):
        """不求导：同一提示后分别续写多段，共用提示的 KV 缓存。"""
        import torch
        import torch.nn.functional as F
        head = self.ids(prompt)
        with torch.no_grad():
            output = self.model(torch.tensor([head]), use_cache=True)
            cache, first = output.past_key_values, F.log_softmax(output.logits[0, -1], -1)
            scores = []
            for text in texts:
                tail = self.ids(text, self.count)
                total = float(first[tail[0]])
                if len(tail) > 1:
                    logits = self.model(torch.tensor([tail[:-1]]), past_key_values=cache, use_cache=True).logits[0]
                    total += float(F.log_softmax(logits, -1).gather(1, torch.tensor(tail[1:])[:, None]).sum())
                    cache.crop(len(head))
                scores.append(total / len(tail))
        return scores


def candidates(mlp_grad, value_grad, neuron_counts, head_counts, deltas):
    import numpy as np
    order_mlp = np.argsort(-np.abs(mlp_grad), axis=None)
    order_values = np.argsort(-np.abs(value_grad), axis=None)
    for delta in deltas:
        for neurons in neuron_counts:
            for heads in head_counts:
                if neurons == 0 and heads == 0:
                    continue
                mlp = np.ones_like(mlp_grad)
                values = np.ones_like(value_grad)
                for position in order_mlp[:neurons]:
                    index = np.unravel_index(position, mlp_grad.shape)
                    mlp[index] = 1 + delta * np.sign(mlp_grad[index])
                for position in order_values[:heads]:
                    index = np.unravel_index(position, value_grad.shape)
                    values[index] = 1 + delta * np.sign(value_grad[index])
                yield {"neurons": neurons, "heads": heads, "delta": delta}, mlp, values


def plan(mlp, values):
    edits = []
    for target, scales in (("mlp", mlp), ("kv_head", values)):
        for layer in range(scales.shape[0]):
            groups = {}
            for index in range(scales.shape[1]):
                factor = float(scales[layer, index])
                if factor != 1.0:
                    groups.setdefault(round(factor, 6), []).append(index)
            for factor, indices in sorted(groups.items()):
                edits.append({"op": "scale", "target": target, "layer": layer, "index": indices, "factor": factor})
    return edits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True, help="Hugging Face 格式的原始模型目录")
    parser.add_argument("--own", type=Path, required=True, help="定位题上不使用工具时的作答结果 answers.jsonl")
    parser.add_argument("--refusal", type=Path, required=True, help="定位题上检索结果为空时的作答结果 answers.jsonl")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/retrieval_gate")
    parser.add_argument("--know", type=Path, default=ROOT / "target/release/nncoss-know")
    parser.add_argument("--per-subject", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=16, help="强制续写的 token 数")
    parser.add_argument("--neurons", type=int, nargs="+", default=[0, 16, 64, 256])
    parser.add_argument("--heads", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--max-drop", type=float, default=0.05, help="不使用工具时自身回答平均对数概率允许下降的上限")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions = eval_gaokao.questions_for("tune", args.per_subject)
    own = {r["id"]: r["reply"] for r in read_jsonl(args.own)}
    refusal = {r["id"]: r["reply"] for r in read_jsonl(args.refusal)}
    # 只取两种表现都出现的题：无工具时作答，检索为空时拒答。依据的是模型自己的输出，不看标准答案。
    usable = [q for q in questions if q["id"] in own and q["id"] in refusal
              and eval_gaokao.choice(own[q["id"]]) and not eval_gaokao.choice(refusal[q["id"]])]
    half_a, half_b = usable[0::2], usable[1::2]
    print(f"可用题 {len(usable)}：A 半 {len(half_a)} 题算梯度，B 半 {len(half_b)} 题比较候选", flush=True)

    empty = args.work / "empty_store"
    empty.mkdir(exist_ok=True)
    framed = prompts(args.know, usable, args.work, "framed", ["--prefetch", "--store", str(empty)])
    plain = prompts(args.know, usable, args.work, "plain", ["--no-tools"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mlp, values = attach(model)
    scorer = Scorer(model, tokenizer, args.tokens)

    for number, q in enumerate(half_a):
        gate = scorer.logp(framed[q["id"]], own[q["id"]]) - scorer.logp(framed[q["id"]], refusal[q["id"]])
        gate.backward()
        print(f"梯度 {number + 1}/{len(half_a)} J={float(gate):.3f}", flush=True)
    mlp_grad = (mlp.grad / len(half_a)).numpy()
    value_grad = (values.grad / len(half_a)).numpy()
    np.savez(args.work / "effects.npz", mlp=mlp_grad, values=value_grad)

    def evaluate(mlp_scales, value_scales):
        mlp.data = torch.tensor(mlp_scales, dtype=torch.float32)
        values.data = torch.tensor(value_scales, dtype=torch.float32)
        gates, keeps = [], []
        for q in half_b:
            own_logp, refusal_logp = scorer.logp_many(framed[q["id"]], [own[q["id"]], refusal[q["id"]]])
            gates.append(own_logp - refusal_logp)
            keeps.append(scorer.logp_many(plain[q["id"]], [own[q["id"]]])[0])
        return float(np.mean(gates)), float(np.mean(keeps)), float(np.mean(np.array(gates) > 0))

    base_gate, base_keep, base_flip = evaluate(np.ones_like(mlp_grad), np.ones_like(value_grad))
    print(f"未改动：J={base_gate:.3f}，J>0 占 {base_flip:.2f}，无工具自身回答 logp={base_keep:.3f}", flush=True)
    table, best = [], None
    for setting, mlp_scales, value_scales in candidates(mlp_grad, value_grad, args.neurons, args.heads, args.deltas):
        gate, keep, flip = evaluate(mlp_scales, value_scales)
        row = {**setting, "gate": round(gate, 4), "gate_gain": round(gate - base_gate, 4), "positive_share": round(flip, 3),
               "plain_logp": round(keep, 4), "plain_drop": round(base_keep - keep, 4)}
        table.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["plain_drop"] <= args.max_drop and row["gate_gain"] > 0 and (best is None or row["gate"] > best[0]["gate"]):
            best = (row, plan(mlp_scales, value_scales))
    summary = {"usable": len(usable), "half_a": [q["id"] for q in half_a], "half_b": [q["id"] for q in half_b],
               "baseline": {"gate": base_gate, "positive_share": base_flip, "plain_logp": base_keep},
               "candidates": table, "chosen": best[0] if best else None}
    (args.work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if best:
        (args.work / "plan.json").write_text(json.dumps(best[1], ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"选中 {best[0]}，改动计划 {len(best[1])} 项 -> {args.work / 'plan.json'}")
    else:
        print("没有候选在不损害无工具作答的前提下提高 J")


if __name__ == "__main__":
    main()
