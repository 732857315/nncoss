"""测量逐 token 稀疏化矩阵输入通道的代价：每个 token 只保留各矩阵乘法输入中最强的一部分通道，其余置零；只测量，不改参数。

decode 受内存带宽限制。若某个 token 的输入通道 j 被置零，矩阵的第 j 列就不必读取；
权重按列存放时，稀疏多少通道就少读多少权重。与只稀疏 down 的中间激活不同，这里覆盖每层全部矩阵：
  attn_in —— q、k、v 投影的输入（注意力前的 RMSNorm 输出，1024 维）
  o       —— o 投影的输入（各头注意力输出拼接，2048 维）
  mlp_in  —— gate、up 投影的输入（MLP 前的 RMSNorm 输出，1024 维）
  down    —— down 投影的输入（silu(gate)·up，3072 维）
  head    —— 输出层的输入（最后的 RMSNorm 输出）；输出层约占每步读取量的四分之一
挑选方式：plain 按 |x_j| 取前 k；weighted 按 |x_j|·‖W[:,j]‖ 取前 k，后者直接估计丢掉第 j 列造成的输出误差。
题目部分照常完整计算（prefill 不受带宽限制），只稀疏作答部分；以模型不用工具时自己的回答为目标强制续写，
比较负对数似然与首选一致率，并按各矩阵参数量估计每步少读的权重比例。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_gaokao

GROUPS = ("attn_in", "o", "mlp_in", "down", "head")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_setting(text):
    """“attn_in+o:0.5” 表示这几组都稀疏掉 50% 的输入通道；“layers” 是前四组的简写，“all” 再加上 head。"""
    names, level = text.rsplit(":", 1)
    expanded = []
    for name in names.split("+"):
        expanded += {"layers": list(GROUPS[:4]), "all": list(GROUPS)}.get(name, [name])
    unknown = set(expanded) - set(GROUPS)
    if unknown:
        raise SystemExit(f"未知的矩阵组：{sorted(unknown)}")
    return expanded, float(level)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True, help="定位题上不用工具时的作答结果 answers.jsonl")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/input_sparsity")
    parser.add_argument("--know", type=Path, default=ROOT / "target/release/nncoss-know")
    parser.add_argument("--questions", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--modes", nargs="+", default=["plain", "weighted"], choices=["plain", "weighted"])
    parser.add_argument("--settings", nargs="+",
                        default=["attn_in:0.5", "o:0.5", "mlp_in:0.5", "down:0.5", "head:0.5",
                                 "layers:0.25", "layers:0.4", "layers:0.5", "all:0.25", "all:0.4"],
                        help="矩阵组:稀疏比例，组名可用 + 连接")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    settings = [(text, *parse_setting(text)) for text in args.settings]

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

    members = {name: [] for name in GROUPS}
    for layer in model.model.layers:
        attention, mlp = layer.self_attn, layer.mlp
        members["attn_in"] += [attention.q_proj, attention.k_proj, attention.v_proj]
        members["o"].append(attention.o_proj)
        members["mlp_in"] += [mlp.gate_proj, mlp.up_proj]
        members["down"].append(mlp.down_proj)
    members["head"].append(model.lm_head)
    params = {name: sum(module.weight.numel() for module in modules) for name, modules in members.items()}
    total_params = sum(params.values())
    column_norms = {id(module): module.weight.detach().norm(dim=0)
                    for modules in members.values() for module in modules}

    state = {"active": set(), "level": 0.0, "mode": "plain", "start": 0, "mass": []}

    def hook(module, inputs):
        if id(module) not in state["active"]:
            return None
        x = inputs[0]
        part = x[:, state["start"]:]
        score = part.abs()
        if state["mode"] == "weighted":
            score = score * column_norms[id(module)]
        count = max(1, int(round(part.shape[-1] * (1.0 - state["level"]))))
        kept = torch.zeros_like(part).scatter_(-1, score.topk(count, dim=-1).indices, 1.0)
        energy = (part * part).sum(-1).clamp_min(1e-12)
        state["mass"].append(float(((part * part * kept).sum(-1) / energy).mean()))
        x = x.clone()
        x[:, state["start"]:] = part * kept
        return (x,)

    for modules in members.values():
        for module in modules:
            module.register_forward_pre_hook(hook)

    def score():
        total, agree, count = 0.0, 0, 0
        with torch.no_grad():
            for ids, start, tail in samples:
                state["start"] = start
                logits = model(ids, use_cache=False).logits[0, start - 1:start - 1 + len(tail)]
                total -= float(F.log_softmax(logits, -1).gather(1, tail[:, None]).sum())
                agree += int((logits.argmax(-1) == tail).sum())
                count += len(tail)
        return total / count, agree / count

    base_nll, base_agree = score()
    rows = [{"setting": "full", "nll": round(base_nll, 4), "agree": round(base_agree, 4),
             "params": {name: params[name] for name in GROUPS}}]
    print(json.dumps(rows[0]), flush=True)
    for mode in args.modes:
        for text, names, level in settings:
            state.update(active={id(module) for name in names for module in members[name]}, level=level, mode=mode, mass=[])
            nll, agree = score()
            skipped = level * sum(params[name] for name in names) / total_params
            row = {"mode": mode, "setting": text, "nll": round(nll, 4), "delta": round(nll - base_nll, 4),
                   "agree": round(agree, 4), "kept_energy": round(sum(state["mass"]) / len(state["mass"]), 4),
                   "reads_saved": round(skipped, 4)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    state["active"] = set()
    (args.work / "result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
