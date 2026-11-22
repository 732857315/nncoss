"""从稠密模型自身的结构和数据中挖出“专家”，测量按 token 动态选专家的代价；只测量，不改参数。

1. 用课本段落跑一遍模型，记录每层每个 MLP 神经元在每个 token 上的激活 |silu(gate)·up|，
   按激活模式把每层神经元均分成若干等大小的专家（余弦相似度上的均衡 k-means）。
2. 路由不训练：每个 token 用本层 gate 投影算出各专家的得分（专家内 |silu(gate)| 之和），只保留得分最高的一部分专家。
3. 在模型自己不用工具时的回答上强制续写，比较负对数似然与首选一致率；对照组按原始神经元顺序切分专家。
聚类只用课本段落，检验只用模型的回答，两者不重叠。输出每层的神经元新顺序，供推理引擎按专家连续存放。
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_gaokao


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def balanced_kmeans(patterns, experts, iterations, rng):
    import numpy as np
    count = patterns.shape[0]
    size = count // experts
    vectors = patterns / np.linalg.norm(patterns, axis=1, keepdims=True).clip(1e-8)
    centroids = vectors[rng.sample(range(count), experts)]
    assignment = np.zeros(count, dtype=int)
    for _ in range(iterations):
        similarity = vectors @ centroids.T
        order = np.argsort(-similarity, axis=None)
        capacity = np.full(experts, size)
        assignment.fill(-1)
        for position in order:
            neuron, expert = divmod(int(position), experts)
            if assignment[neuron] < 0 and capacity[expert] > 0:
                assignment[neuron] = expert
                capacity[expert] -= 1
        for expert in range(experts):
            members = vectors[assignment == expert]
            centroid = members.mean(axis=0)
            centroids[expert] = centroid / max(np.linalg.norm(centroid), 1e-8)
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True, help="定位题上不用工具时的作答结果，只用于检验")
    parser.add_argument("--passages", type=Path, required=True, help="课本段落 passages.jsonl，只用于聚类")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/moefy")
    parser.add_argument("--know", type=Path, default=ROOT / "target/release/nncoss-know")
    parser.add_argument("--texts", type=int, default=96, help="用于聚类的课本段落数")
    parser.add_argument("--expert-size", type=int, default=64, help="每个专家的神经元数，取 32 的倍数以便 Q4 分组对齐")
    parser.add_argument("--keep", type=float, nargs="+", default=[0.5, 0.3, 0.2])
    parser.add_argument("--questions", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    layers = model.model.layers
    inner = model.config.intermediate_size
    experts = inner // args.expert_size

    state = {"collect": None, "route": None}

    def forward(index, module, x):
        gate = F.silu(module.gate_proj(x))
        hidden = gate * module.up_proj(x)
        if state["collect"] is not None:
            state["collect"][index].append(hidden.abs()[0].to(torch.float16))
        if state["route"] is not None:
            groups, keep, mode = state["route"]
            group = torch.as_tensor(groups[index])
            score = (hidden if mode == "oracle" else gate).abs()
            per_expert = torch.zeros(*score.shape[:-1], experts).index_add_(-1, group, score)
            count = max(1, int(round(experts * keep)))
            chosen = torch.zeros_like(per_expert).scatter_(-1, per_expert.topk(count, dim=-1).indices, 1.0)
            hidden = hidden * chosen.index_select(-1, group)
        return module.down_proj(hidden)

    for index, layer in enumerate(layers):
        layer.mlp.forward = lambda x, index=index, module=layer.mlp: forward(index, module, x)

    passages = read_jsonl(args.passages)
    texts = [passage["text"] for passage in rng.sample(passages, args.texts)]
    state["collect"] = [[] for _ in layers]
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[:, :256]
            model(ids, use_cache=False)
    collected = [torch.cat(chunks).float().numpy().T for chunks in state["collect"]]
    state["collect"] = None
    print(f"聚类数据：{len(texts)} 段课本，{collected[0].shape[1]} 个 token；每层分成 {experts} 个专家", flush=True)

    clustered = [balanced_kmeans(pattern, experts, 8, rng) for pattern in collected]
    contiguous = [np.arange(inner) // args.expert_size for _ in layers]
    del collected

    own = {record["id"]: record["reply"] for record in read_jsonl(args.own)}
    questions = [q for q in eval_gaokao.questions_for("tune", 12) if eval_gaokao.choice(own.get(q["id"], ""))]
    questions = questions[::max(1, len(questions) // args.questions)][:args.questions]
    source, target = args.work / "questions.jsonl", args.work / "prompts.jsonl"
    source.write_text("".join(json.dumps({"id": q["id"], "question": q["question"]}, ensure_ascii=False) + "\n"
                              for q in questions), encoding="utf-8")
    subprocess.run([str(args.know), "prompt", "--no-tools", "--input", str(source), "--output", str(target)], check=True)
    prompts = {record["id"]: record["prompt"] for record in read_jsonl(target)}
    samples = []
    for q in questions:
        head = tokenizer(prompts[q["id"]], add_special_tokens=False).input_ids
        tail = tokenizer(own[q["id"]], add_special_tokens=False).input_ids[:args.tokens]
        samples.append((torch.tensor([head + tail]), len(head), torch.tensor(tail)))

    def score():
        total, agree, count = 0.0, 0, 0
        with torch.no_grad():
            for ids, start, tail in samples:
                logits = model(ids, use_cache=False).logits[0, start - 1:start - 1 + len(tail)]
                total -= float(F.log_softmax(logits, -1).gather(1, tail[:, None]).sum())
                agree += int((logits.argmax(-1) == tail).sum())
                count += len(tail)
        return total / count, agree / count

    base_nll, base_agree = score()
    rows = [{"split": "full", "keep": 1.0, "nll": round(base_nll, 4), "agree": round(base_agree, 4)}]
    print(json.dumps(rows[0], ensure_ascii=False), flush=True)
    for name, groups in (("clustered", clustered), ("contiguous", contiguous)):
        for mode in ("gate", "oracle"):
            for keep in args.keep:
                state["route"] = (groups, keep, mode)
                nll, agree = score()
                row = {"split": name, "router": mode, "keep": keep, "nll": round(nll, 4),
                       "delta": round(nll - base_nll, 4), "agree": round(agree, 4)}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    state["route"] = None
    order = [np.argsort(groups, kind="stable").tolist() for groups in clustered]
    (args.work / "result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.work / "experts.json").write_text(json.dumps({"expert_size": args.expert_size, "order": order}),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
