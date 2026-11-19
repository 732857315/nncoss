import argparse
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import school

ROOT = Path("/Users/admin/Desktop/prj2026/nncoss")
END = "<|im_end|>"
TURN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
ASK = {
    "mc": "\n\n请先简要分析，最后一行写“答案：”加选项字母。",
    "numeric": "\n\n请先写出简要的计算过程，最后一行写“答案：”加一个数。",
    "code": "\n\n请先简要分析代码的执行过程，最后一行写“答案：”加输出内容。",
    "design": "\n\n请直接给出完整的 Python 函数代码，放在 ```python 代码块中。",
}
RETRY = {
    "numeric": "你的答案不正确。请重新检查每一步计算，找出错误后再做一遍，最后一行写“答案：”加一个数。",
    "code": "你的答案不正确。请逐行重新模拟代码的执行，找出错误后再回答，最后一行写“答案：”加输出内容。",
    "design": "你的代码没有通过测试：{error}\n请找出问题并给出修正后的完整函数代码，放在 ```python 代码块中。",
}


def question_text(item):
    prompt = item["prompt"]
    body = prompt.split("<|im_start|>user\n", 1)[1].split(END, 1)[0]
    body = body.replace("请直接回答正确选项的字母。", "").replace("请直接给出最终答案，只写一个数。", "")
    body = body.replace("只写输出内容。", "").replace("只输出完整的函数代码。", "")
    return body.strip() + ASK[item["kind"]]


def user(text):
    return f"<|im_start|>user\n{text}{END}\n"


def final_answer(text):
    found = re.findall(r"答案[:：]\s*(.+)", text)
    return found[-1].strip().strip("`").strip() if found else None


def verify(item, text):
    kind = item["kind"]
    if kind == "design":
        blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
        code = blocks[-1] if blocks else text
        script = code + "\n\n" + "\n".join(f"assert {test}, {test!r}" for test in item["tests"]) + "\nprint('ok')\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solution.py"
            path.write_text(script)
            try:
                result = subprocess.run([sys.executable, "-I", str(path)], cwd=directory, capture_output=True, text=True, timeout=5)
            except subprocess.TimeoutExpired:
                return False, "运行超时"
        if result.returncode == 0 and result.stdout.strip().endswith("ok"):
            return True, ""
        lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
        return False, (lines[-1] if lines else "没有输出")[:200]
    answer = final_answer(text)
    if answer is None:
        return False, "没有找到答案"
    if kind == "mc":
        letters = re.findall(r"[ABCD]", answer)
        return bool(letters) and letters[0] == item["answer"], ""
    if kind == "numeric":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
        if not numbers:
            return False, ""
        value = float(numbers[0])
        return abs(value - item["value"]) <= 1e-6 + 1e-3 * abs(item["value"]), ""
    return answer.strip("'\"") == item["answer"].strip("'\""), ""


class LoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, inputs):
        return self.base(inputs) + (inputs @ self.a.T @ self.b.T) * self.scale

    def merged(self):
        return self.base.weight + (self.b @ self.a) * self.scale


def add_lora(model, rank, alpha):
    modules = []
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                wrapped = LoRA(getattr(parent, name), rank, alpha)
                setattr(parent, name, wrapped)
                modules.append(wrapped)
    return modules


class Engine:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def generate(self, prompts, max_new, sample, batch=24):
        order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
        outputs = [None] * len(prompts)
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            encoded = self.tokenizer([prompts[index] for index in chunk], return_tensors="pt", padding=True)
            kwargs = dict(max_new_tokens=max_new, pad_token_id=151643, eos_token_id=[151645, 151643])
            if sample:
                kwargs.update(do_sample=True, temperature=0.8, top_p=0.95, top_k=40)
            else:
                kwargs.update(do_sample=False)
            generated = self.model.generate(**encoded, **kwargs)
            for row, index in enumerate(chunk):
                outputs[index] = self.tokenizer.decode(generated[row, encoded.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return outputs

    @torch.no_grad()
    def letter_accuracy(self, items):
        letters = [self.tokenizer.convert_tokens_to_ids(letter) for letter in "ABCD"]
        correct = 0
        for item in items:
            ids = self.tokenizer(item["prompt"], return_tensors="pt").input_ids
            logits = self.model(ids).logits[0, -1, letters]
            correct += "ABCD"[int(logits.argmax())] == item["answer"]
        return correct / len(items)


def episodes(engine, items, attempts, sample, max_new):
    states = [{"item": item, "context": user(question_text(item)) + TURN, "history": [], "solved": None} for item in items]
    for attempt in range(attempts):
        active = [state for state in states if state["solved"] is None and (attempt == 0 or state["item"]["kind"] != "mc")]
        if not active:
            break
        replies = engine.generate([state["context"] for state in active], max_new, sample)
        for state, reply in zip(active, replies):
            ok, error = verify(state["item"], reply)
            state["history"].append({"context": state["context"], "reply": reply, "ok": ok})
            if ok:
                state["solved"] = attempt
            elif attempt + 1 < attempts and state["item"]["kind"] != "mc":
                feedback = RETRY[state["item"]["kind"]].format(error=error)
                state["context"] = state["context"] + reply + END + "\n" + user(feedback) + TURN
    return states


def report(states, attempts):
    table = defaultdict(lambda: [0] * (attempts + 1))
    for state in states:
        item = state["item"]
        for key in (item["kind"], f"{item['kind']}/{item['subject']}/{item['level']}", "all"):
            table[key][attempts] += 1
            if state["solved"] is not None:
                for attempt in range(state["solved"], attempts):
                    table[key][attempt] += 1
    return {key: [round(value / counts[attempts], 3) for value in counts[:attempts]] + [counts[attempts]] for key, counts in sorted(table.items())}


def evaluate(path, label, attempts=3):
    engine = Engine(path)
    _, test = school.build()
    started = time.time()
    states = episodes(engine, test, attempts, False, 320)
    result = {"label": label, "letter_mc": engine.letter_accuracy([item for item in test if item["kind"] == "mc"]), "table": report(states, attempts)}
    (HERE / "school_reports").mkdir(exist_ok=True)
    (HERE / "school_reports" / f"{label}.json").write_text(json.dumps({"result": result, "states": [
        {"prompt": state["item"]["prompt"][:200], "answer": state["item"]["answer"], "solved": state["solved"], "history": state["history"]} for state in states
    ]}, ensure_ascii=False, indent=1))
    print(label, f"{time.time() - started:.0f}s", "letter_mc", round(result["letter_mc"], 3), flush=True)
    for key in ("all", "mc", "numeric", "code", "design"):
        print(f"  {key}: 各次尝试后累计正确率 {result['table'][key][:-1]}  题数 {result['table'][key][-1]}", flush=True)
    return result


def collect(engine, items, samples, attempts):
    kept = []
    stats = defaultdict(int)
    for round_index in range(samples):
        states = episodes(engine, items, attempts, True, 320)
        for state in states:
            stats["episodes"] += 1
            if state["solved"] is None:
                stats["failed"] += 1
                continue
            step = state["history"][state["solved"]]
            stats["first" if state["solved"] == 0 else "corrected"] += 1
            kept.append({"context": step["context"], "reply": step["reply"], "corrected": state["solved"] > 0, "id": state["item"]["prompt"]})
        print("collect round", round_index, dict(stats), flush=True)
    return kept, dict(stats)


def train(engine, examples, epochs, lr, rank):
    modules = add_lora(engine.model, rank, rank * 2)
    parameters = [p for module in modules for p in (module.a, module.b)]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    total = epochs * len(examples)
    step = 0
    started = time.time()
    engine.model.train()
    for epoch in range(epochs):
        random.shuffle(examples)
        for example in examples:
            prefix = engine.tokenizer(example["context"], return_tensors="pt").input_ids[0]
            target = engine.tokenizer(example["reply"] + END, return_tensors="pt").input_ids[0]
            ids = torch.cat([prefix, target])[-900:]
            logits = engine.model(ids[None]).logits[0, -len(target) - 1:-1]
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / total))
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print(f"train step {step}/{total} loss={loss.item():.3f} {time.time() - started:.0f}s", flush=True)
    engine.model.eval()
    with torch.no_grad():
        for module in modules:
            module.base.weight.copy_(module.merged())
    for layer in engine.model.layers if hasattr(engine.model, "layers") else engine.model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                setattr(parent, name, getattr(parent, name).base)


def save(engine, output):
    from safetensors.torch import save_file
    import shutil
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "model/Qwen3-0.6B"
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(source / name, output / name)
    state = {key: value.detach().contiguous().clone() for key, value in engine.model.state_dict().items()}
    save_file(state, output / "model.safetensors", metadata={"format": "pt"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["eval", "improve"])
    parser.add_argument("--model", type=Path, default=ROOT / "model/Qwen3-0.6B")
    parser.add_argument("--label", default="base")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--subset", type=int, default=480)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--output", type=Path, default=HERE / "models" / "school")
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    if args.command == "eval":
        evaluate(args.model, args.label)
        return
    current = args.model
    for round_index in range(args.rounds):
        engine = Engine(current)
        train_items, _ = school.build()
        rng = random.Random(round_index)
        pool = [item for item in train_items if item["kind"] in ("mc", "design")]
        pool += rng.sample([item for item in train_items if item["kind"] not in ("mc", "design")], args.subset)
        kept, stats = collect(engine, pool, args.samples, 3)
        seen = set()
        unique = []
        for example in kept:
            key = (example["id"], example["reply"])
            if key not in seen:
                seen.add(key)
                unique.append(example)
        (HERE / f"kept_round{round_index}.json").write_text(json.dumps(unique, ensure_ascii=False, indent=0))
        print(f"round {round_index}: 保留 {len(unique)} 条成功经验（其中纠错后成功 {sum(e['corrected'] for e in unique)} 条）", flush=True)
        train(engine, unique, 1, args.lr, args.rank)
        output = args.output.parent / f"{args.output.name}-r{round_index + 1}"
        save(engine, output)
        del engine
        current = output
        evaluate(current, f"{args.label}-r{round_index + 1}")


if __name__ == "__main__":
    main()
