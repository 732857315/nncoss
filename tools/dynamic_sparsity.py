"""测量按输入实时选择 MLP 神经元的代价：每个 token 只保留一部分神经元，其余置零；只测量，不改参数。

两种选择方式：
  oracle —— 按神经元的实际输出 |silu(gate)·up| 取前 k，代表这种动态结构的上限；
  gate   —— 只按 |silu(gate)| 取前 k。gate 投影本来就要算，选中后才读 up 与 down 的对应行，
            这是推理引擎能真正省下读取的做法。
以模型在定位题上不用工具时自己的回答为目标，强制续写前 N 个 token，比较负对数似然与首选一致率。
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
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/dynamic_sparsity")
    parser.add_argument("--know", type=Path, default=ROOT / "target/release/nncoss-know")
    parser.add_argument("--questions", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--keep", type=float, nargs="+", default=[0.5, 0.3, 0.2, 0.1])
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
    samples = []
    for q in questions:
        head = tokenizer(prompts[q["id"]], add_special_tokens=False).input_ids
        tail = tokenizer(own[q["id"]], add_special_tokens=False).input_ids[:args.tokens]
        samples.append((torch.tensor([head + tail]), len(head), torch.tensor(tail)))

    setting = {"mode": None, "keep": 1.0}
    mass = []

    def forward(module, inputs):
        x = inputs[0]
        gate = F.silu(module.gate_proj(x))
        hidden = gate * module.up_proj(x)
        if setting["mode"] is not None:
            score = (hidden if setting["mode"] == "oracle" else gate).abs()
            count = max(1, int(round(score.shape[-1] * setting["keep"])))
            kept = torch.zeros_like(hidden).scatter_(-1, score.topk(count, dim=-1).indices, 1.0)
            if setting["mode"] == "oracle":
                total = hidden.abs().sum(-1)
                mass.append(float(((hidden.abs() * kept).sum(-1) / total.clamp_min(1e-12)).mean()))
            hidden = hidden * kept
        return module.down_proj(hidden)

    for layer in model.model.layers:
        layer.mlp.forward = lambda x, module=layer.mlp: forward(module, (x,))

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
    rows = [{"mode": "full", "keep": 1.0, "nll": round(base_nll, 4), "agree": round(base_agree, 4)}]
    print(json.dumps(rows[0]), flush=True)
    for mode in ("oracle", "gate"):
        for keep in args.keep:
            setting.update(mode=mode, keep=keep)
            mass.clear()
            nll, agree = score()
            row = {"mode": mode, "keep": keep, "nll": round(nll, 4), "delta": round(nll - base_nll, 4),
                   "agree": round(agree, 4)}
            if mass:
                row["kept_mass"] = round(sum(mass) / len(mass), 4)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (args.work / "result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
