import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_gaokao

TOOL_CALL = "<tool_call>"


def prompts(questions, work):
    source = work / "tune_questions.jsonl"
    target = work / "tune_prompts.jsonl"
    source.write_text("".join(json.dumps({"id": q["id"], "question": q["question"]}, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    subprocess.run([str(ROOT / "target/release/nncoss-know"), "prompt", "--input", str(source), "--output", str(target)], check=True)
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def measure(model_dir, records):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.config
    layers = model.model.layers
    head_dim = config.head_dim
    mlp = torch.ones(len(layers), config.intermediate_size, requires_grad=True)
    values = torch.ones(len(layers), config.num_key_value_heads, requires_grad=True)
    for index, layer in enumerate(layers):
        layer.mlp.down_proj.register_forward_pre_hook(lambda module, inputs, index=index: (inputs[0] * mlp[index],))
        layer.self_attn.v_proj.register_forward_hook(
            lambda module, inputs, output, index=index: output * values[index].repeat_interleave(head_dim))
    tool = tokenizer.convert_tokens_to_ids(TOOL_CALL)
    probabilities = []
    for number, record in enumerate(records):
        ids = tokenizer(record["prompt"], add_special_tokens=False, return_tensors="pt").input_ids
        logits = model(ids).logits[0, -1]
        objective = F.log_softmax(logits, -1)[tool]
        objective.backward()
        probabilities.append(float(objective.exp()))
        print(f"{number + 1}/{len(records)} p(tool_call)={probabilities[-1]:.4f}", flush=True)
    count = len(records)
    return probabilities, (mlp.grad / count).numpy(), (values.grad / count).numpy()


def plan(mlp_effect, value_effect, neurons, heads, factor):
    import numpy as np
    edits = []
    flat = np.argsort(-mlp_effect, axis=None)[:neurons]
    chosen = {}
    for position in flat:
        layer, neuron = divmod(int(position), mlp_effect.shape[1])
        if mlp_effect[layer, neuron] > 0:
            chosen.setdefault(layer, []).append(neuron)
    for layer, indices in sorted(chosen.items()):
        edits.append({"op": "scale", "target": "mlp", "layer": layer, "index": sorted(indices), "factor": factor})
    for position in np.argsort(-value_effect, axis=None)[:heads]:
        layer, head = divmod(int(position), value_effect.shape[1])
        if value_effect[layer, head] > 0:
            edits.append({"op": "scale", "target": "kv_head", "layer": layer, "index": [head], "factor": factor})
    return edits


def main():
    parser = argparse.ArgumentParser(description="测量模型在开口时选择调用检索工具的概率，并估计每个 MLP 神经元与 KV 头对这一选择的影响；只做测量，不改参数")
    parser.add_argument("--model-dir", type=Path, required=True, help="Hugging Face 格式的原始模型目录")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/tool_decision")
    parser.add_argument("--per-subject", type=int, default=12)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--factor", type=float, default=1.5)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    questions = eval_gaokao.questions_for("tune", args.per_subject)
    records = prompts(questions, args.work)
    probabilities, mlp_effect, value_effect = measure(args.model_dir, records)
    import numpy as np
    np.savez(args.work / "effects.npz", mlp=mlp_effect, values=value_effect, probabilities=np.array(probabilities))
    edits = plan(mlp_effect, value_effect, args.neurons, args.heads, args.factor)
    (args.work / "plan.json").write_text(json.dumps(edits, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"平均 p(tool_call)={np.mean(probabilities):.4f}，改动计划 {len(edits)} 项 -> {args.work / 'plan.json'}")


if __name__ == "__main__":
    main()
