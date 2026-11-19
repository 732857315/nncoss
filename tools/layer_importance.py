"""测量每一层对模型自身回答的贡献，给出删层顺序；只测量，不改参数。

不使用标准答案：以模型在定位题上不用工具时自己的回答为目标，强制续写前 N 个 token，
比较删去某层后这些 token 的平均负对数似然（NLL）和逐 token 首选一致率。
先逐层单独删除得到排序，再按排序累计删除 1..K 层，测量累计效果。
删层由 nncoss-edit 的 drop_layers 完成，本脚本只输出建议的层号。
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
    parser.add_argument("--own", type=Path, required=True, help="定位题上不用工具时的作答结果 answers.jsonl")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/layer_importance")
    parser.add_argument("--know", type=Path, default=ROOT / "target/release/nncoss-know")
    parser.add_argument("--questions", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--max-drop", type=int, default=8)
    parser.add_argument("--part", choices=["layer", "attn", "mlp"], default="layer",
                        help="删除整层，或只删注意力子层、只删 MLP 子层（子层输出置零，残差直通）")
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
    layers = list(model.model.layers)
    samples = []
    for q in questions:
        head = tokenizer(prompts[q["id"]], add_special_tokens=False).input_ids
        tail = tokenizer(own[q["id"]], add_special_tokens=False).input_ids[:args.tokens]
        samples.append((torch.tensor([head + tail]), len(head), torch.tensor(tail)))

    def silence(module, inputs, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + tuple(output[1:])
        return torch.zeros_like(output)

    def score(removed):
        hooks = []
        if args.part == "layer":
            model.model.layers = torch.nn.ModuleList(layer for index, layer in enumerate(layers) if index not in removed)
        else:
            for index in removed:
                target = layers[index].self_attn if args.part == "attn" else layers[index].mlp
                hooks.append(target.register_forward_hook(silence))
        total, agree, count = 0.0, 0, 0
        with torch.no_grad():
            for ids, start, tail in samples:
                logits = model(ids, use_cache=False).logits[0, start - 1:start - 1 + len(tail)]
                logp = F.log_softmax(logits, -1)
                total -= float(logp.gather(1, tail[:, None]).sum())
                agree += int((logits.argmax(-1) == tail).sum())
                count += len(tail)
        model.model.layers = torch.nn.ModuleList(layers)
        for hook in hooks:
            hook.remove()
        return total / count, agree / count

    base_nll, base_agree = score(set())
    print(f"完整模型：NLL={base_nll:.4f}，首选一致率={base_agree:.3f}，{len(samples)} 题", flush=True)
    single = []
    for index in range(len(layers)):
        nll, agree = score({index})
        single.append({"layer": index, "nll": round(nll, 4), "delta": round(nll - base_nll, 4), "agree": round(agree, 4)})
        print(json.dumps(single[-1]), flush=True)
    order = [row["layer"] for row in sorted(single, key=lambda row: row["delta"])]
    cumulative = []
    for count in range(1, args.max_drop + 1):
        removed = sorted(order[:count])
        nll, agree = score(set(removed))
        cumulative.append({"count": count, "layers": removed, "nll": round(nll, 4),
                           "delta": round(nll - base_nll, 4), "agree": round(agree, 4)})
        print(json.dumps(cumulative[-1]), flush=True)
    result = {"part": args.part, "questions": [q["id"] for q in questions], "tokens": args.tokens,
              "baseline": {"nll": base_nll, "agree": base_agree}, "single": single, "order": order,
              "cumulative": cumulative}
    (args.work / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
