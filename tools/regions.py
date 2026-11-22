"""测量“按任务激活脑区”的代价：读题时决定每层哪些 MLP 神经元参与作答，作答时只计算这些神经元；只测量，不改参数。

对每道题，先用完整网络读一遍题目，取每层各神经元在题目 token 上的平均激活 |silu(gate)·up|，
每层保留最活跃的一部分神经元，组成这道题的“区域”。各层的区域由同一个上下文同时决定。
然后强制续写模型自己不用工具时的回答：题目部分用完整网络，回答部分只计算区域内的神经元，
比较负对数似然与首选一致率。对照：
  global —— 所有题共用一个区域（全部题目的平均激活最高的神经元），检验区域是否需要随任务变化；
  random —— 每层随机选同样多的神经元。
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/regions")
    parser.add_argument("--know", type=Path, default=ROOT / "target/release/nncoss-know")
    parser.add_argument("--questions", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--keep", type=float, nargs="+", default=[0.7, 0.5, 0.4, 0.3])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    own = {record["id"]: record["reply"] for record in read_jsonl(args.own)}
    questions = [q for q in eval_gaokao.questions_for("tune", 12) if eval_gaokao.choice(own.get(q["id"], ""))]
    questions = questions[::max(1, len(questions) // args.questions)][:args.questions]
    source, target = args.work / "questions.jsonl", args.work / "prompts.jsonl"
    source.write_text("".join(json.dumps({"id": q["id"], "question": q["question"]}, ensure_ascii=False) + "\n"
                              for q in questions), encoding="utf-8")
    subprocess.run([str(args.know), "prompt", "--no-tools", "--input", str(source), "--output", str(target)], check=True)
    prompts = {record["id"]: record["prompt"] for record in read_jsonl(target)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    layers = model.model.layers
    state = {"record": None, "masks": None, "start": 0}

    def forward(index, module, x):
        hidden = F.silu(module.gate_proj(x)) * module.up_proj(x)
        if state["record"] is not None:
            state["record"][index] = hidden.abs()[0].mean(0)
        if state["masks"] is not None:
            mask = torch.ones_like(hidden)
            mask[:, state["start"]:] = state["masks"][index]
            hidden = hidden * mask
        return module.down_proj(hidden)

    for index, layer in enumerate(layers):
        layer.mlp.forward = lambda x, index=index, module=layer.mlp: forward(index, module, x)

    samples, profiles = [], []
    with torch.no_grad():
        for q in questions:
            head = tokenizer(prompts[q["id"]], add_special_tokens=False).input_ids
            tail = tokenizer(own[q["id"]], add_special_tokens=False).input_ids[:args.tokens]
            state["record"] = [None] * len(layers)
            model(torch.tensor([head]), use_cache=False)
            profiles.append(torch.stack(state["record"]))
            state["record"] = None
            samples.append((torch.tensor([head + tail]), len(head), torch.tensor(tail)))
    overall = torch.stack(profiles).mean(0)
    generator = torch.Generator().manual_seed(args.seed)

    def top_mask(profile, keep):
        count = max(1, int(round(profile.shape[-1] * keep)))
        return torch.zeros_like(profile).scatter_(-1, profile.topk(count, dim=-1).indices, 1.0)

    def score(masks_for):
        total, agree, count = 0.0, 0, 0
        with torch.no_grad():
            for number, (ids, start, tail) in enumerate(samples):
                state["masks"], state["start"] = masks_for(number), start
                logits = model(ids, use_cache=False).logits[0, start - 1:start - 1 + len(tail)]
                total -= float(F.log_softmax(logits, -1).gather(1, tail[:, None]).sum())
                agree += int((logits.argmax(-1) == tail).sum())
                count += len(tail)
        state["masks"] = None
        return total / count, agree / count

    base_nll, base_agree = score(lambda number: None)
    rows = [{"region": "full", "keep": 1.0, "nll": round(base_nll, 4), "agree": round(base_agree, 4)}]
    print(json.dumps(rows[0]), flush=True)
    overlaps = {}
    for keep in args.keep:
        per_question = [top_mask(profile, keep) for profile in profiles]
        shared = top_mask(overall, keep)
        random_masks = [top_mask(torch.rand(overall.shape, generator=generator), keep) for _ in profiles]
        overlaps[keep] = float(torch.stack([(mask * shared).sum() / mask.sum() for mask in per_question]).mean())
        for name, masks_for in (("question", lambda number: per_question[number]), ("global", lambda number: shared),
                                ("random", lambda number: random_masks[number])):
            nll, agree = score(masks_for)
            row = {"region": name, "keep": keep, "nll": round(nll, 4), "delta": round(nll - base_nll, 4),
                   "agree": round(agree, 4)}
            if name == "question":
                row["overlap_with_global"] = round(overlaps[keep], 4)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.work / "result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
