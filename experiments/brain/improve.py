import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
PYTHON = sys.executable
CALIBRATION = ["tools/calibration/qwen3_chat.txt", "README.md", "crates/nncoss-llm/src/session.rs",
               "crates/nncoss-llm/src/model.rs", "tools/export_llm.py", "crates/nncoss-web/src/lib.rs"]
PLANS = {
    "r1": [
        ("nothink-notool", {"numeric": (160, 1), "code": (80, 1), "design": (15, 1), "mc": (47, 1)}),
        ("nothink-tool", {"numeric": (160, 1), "code": (80, 1), "design": (15, 1), "mc": (47, 1)}),
        ("think-notool", {"numeric": (50, 1), "code": (30, 1), "design": (15, 1), "mc": (20, 1)}),
        ("think-tool", {"numeric": (50, 1), "code": (30, 1), "design": (15, 1), "mc": (20, 1)}),
    ],
    "r2": [
        ("nothink-notool", {"numeric": (80, 1), "code": (60, 1), "design": (15, 2), "mc": (60, 1)}),
        ("nothink-tool", {"numeric": (80, 1), "code": (60, 1), "design": (15, 2), "mc": (60, 2)}),
        ("think-notool", {"numeric": (30, 1), "code": (30, 1), "design": (15, 1), "mc": (30, 1)}),
        ("think-tool", {"numeric": (30, 1), "code": (30, 1), "design": (15, 1), "mc": (30, 1)}),
    ],
}


def collect(model_path, output, seed, plan):
    import agent
    import school
    native = agent.Native(model_path)
    train, _ = school.build()
    rng = random.Random(seed)
    by_kind = {}
    for item in train:
        by_kind.setdefault(item["kind"], []).append(item)
    examples = []
    stats = Counter()
    started = time.time()
    for condition, counts in PLANS[plan]:
        thinking = condition.startswith("think")
        tools = condition.endswith("-tool")
        for kind, (count, samples) in counts.items():
            pool = rng.sample(by_kind[kind], min(count, len(by_kind[kind])))
            for item in pool:
                for _ in range(samples):
                    record = native.solve(item, thinking, tools, 1 if kind == "mc" else 3, temperature=0.6 if thinking else 0.7, seed=rng.randrange(1 << 30))
                    stats[(condition, kind, "total")] += 1
                    if record["solved"] is None:
                        continue
                    stats[(condition, kind, "first" if record["solved"] == 0 else "corrected")] += 1
                    ids, mask = [], []
                    for segment in record["segments"]:
                        ids += segment["ids"]
                        learn = segment["role"] == "model" and segment["attempt"] == record["solved"]
                        mask += [learn] * len(segment["ids"])
                    examples.append({"ids": ids, "mask": mask, "condition": condition, "kind": kind, "solved": record["solved"],
                                     "tools": record["tool_calls"]})
            print(f"{condition} {kind}: {dict((k[2], v) for k, v in stats.items() if k[:2] == (condition, kind))} {time.time() - started:.0f}s", flush=True)
    output.write_text(json.dumps({"examples": examples, "stats": {"/".join(k): v for k, v in stats.items()}}))
    print("collected", len(examples), flush=True)


def train(source, examples_path, output, lr, rank, epochs, max_len, anchor=None, beta=0.0, repeat_corrected=1, max_run=1600):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    from learn import add_lora
    torch.manual_seed(0)
    random.seed(0)
    examples = json.loads(Path(examples_path).read_text())["examples"]
    from transformers import AutoTokenizer
    from agent import TOOLS
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
    probe = [{"role": "user", "content": "x"}]
    plain = tokenizer.apply_chat_template(probe, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    tooled = tokenizer.apply_chat_template(probe, tools=TOOLS, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    assert tooled.endswith(plain)
    system = tokenizer(tooled[:len(tooled) - len(plain)], add_special_tokens=False).input_ids
    augmented = []
    for example in examples:
        if example["condition"].endswith("-notool"):
            copy = dict(example)
            copy["ids"] = system + example["ids"]
            copy["mask"] = [False] * len(system) + example["mask"]
            copy["condition"] = example["condition"].replace("-notool", "-tool(直接作答)")
            augmented.append(copy)
    examples += augmented
    examples += [example for example in examples if example["solved"] > 0 for _ in range(repeat_corrected - 1)]
    def longest_run(mask):
        best = run = 0
        for value in mask:
            run = run + 1 if value else 0
            best = max(best, run)
        return best

    before = len(examples)
    examples = [example for example in examples if len(example["ids"]) <= max_len and any(example["mask"]) and (max_run <= 0 or longest_run(example["mask"]) <= max_run)]
    print("长度筛选", before, "->", len(examples), flush=True)
    print("训练样本", len(examples), Counter(e["condition"] for e in examples), Counter(e["kind"] for e in examples), flush=True)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32)
    model.config.use_cache = False
    reference = None
    if anchor is not None and beta > 0:
        reference = AutoModelForCausalLM.from_pretrained(anchor, dtype=torch.float32).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = add_lora(model, rank, rank * 2)
    parameters = [p for module in modules for p in (module.a, module.b)]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    total = epochs * len(examples)
    step = 0
    started = time.time()
    average = None
    model.train()
    for _ in range(epochs):
        random.shuffle(examples)
        for example in examples:
            ids = torch.tensor(example["ids"])
            mask = torch.tensor(example["mask"][1:])
            logits = model(ids[None]).logits[0, :-1]
            loss = F.cross_entropy(logits[mask], ids[1:][mask])
            if reference is not None:
                positions = torch.randperm(len(ids) - 1)[:256]
                with torch.no_grad():
                    expected = F.log_softmax(reference(ids[None]).logits[0, :-1][positions], -1)
                actual = F.log_softmax(logits[positions], -1)
                loss = loss + beta * (expected.exp() * (expected - actual)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, (step + 1) / 30) * 0.5 * (1 + math.cos(math.pi * step / total))
            optimizer.step()
            average = loss.item() if average is None else 0.98 * average + 0.02 * loss.item()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{total} loss={average:.4f} {time.time() - started:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        for module in modules:
            module.base.weight.copy_(module.merged())
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                setattr(parent, name, getattr(parent, name).base)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(Path(source) / name, output / name)
    state = {key: value.detach().contiguous().clone() for key, value in model.state_dict().items()}
    save_file(state, output / "model.safetensors", metadata={"format": "pt"})
    print("saved", output, flush=True)


def export(hf_dir, q4_path, distill_steps):
    gptq = q4_path.with_name(q4_path.stem + "-gptq.nncossllm")
    subprocess.run([PYTHON, "tools/export_llm.py", "--model", str(hf_dir), "--output", str(gptq), "--outtype", "q4",
                    "--calibration", *CALIBRATION], cwd=ROOT, check=True)
    subprocess.run([PYTHON, "tools/distill_q4_llm.py", "--model", str(hf_dir), "--input", str(gptq), "--output", str(q4_path),
                    "--steps", str(distill_steps), "--calibration", *CALIBRATION], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "train", "export"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "model/Qwen3-0.6B")
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", default="r1")
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--repeat-corrected", type=int, default=1)
    parser.add_argument("--max-run", type=int, default=1600, help="单轮输出超过该长度的经验不参与训练，0 表示不筛选")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=2600)
    parser.add_argument("--distill-steps", type=int, default=300)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.model, args.output, args.seed, args.plan)
    elif args.command == "train":
        train(args.source, args.examples, args.output, args.lr, args.rank, args.epochs, args.max_len, args.anchor, args.beta, args.repeat_corrected, args.max_run)
    else:
        export(args.source, args.output, args.distill_steps)


if __name__ == "__main__":
    main()
