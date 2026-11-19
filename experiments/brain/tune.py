import argparse
import ast
import json
import math
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/Users/admin/Desktop/prj2026/nncoss")
HERE = Path(__file__).parent
DATA = HERE / "datasets"
CHAT = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def wiki(split):
    text = "".join(row["text"] for row in pq.read_table(DATA / f"wikitext-{split}.parquet").to_pylist())
    for source, target in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "), (" 's", "'s"),
                           (" ; ", "; "), (" : ", ": "), ("( ", "("), (" )", ")"), (" n't", "n't"), (" 'll", "'ll"), (" 're", "'re")):
        text = text.replace(source, target)
    return re.sub(r"\n\s*=+ ([^=]+) =+\s*\n", r"\n\n\1\n\n", text)


def cmrc(split):
    rows = pq.read_table(DATA / f"cmrc-{split}.parquet").to_pylist()
    contexts, questions, seen = [], [], set()
    for row in rows:
        if row["context"] not in seen:
            seen.add(row["context"])
            contexts.append(row["context"])
        answers = row["answers"] if isinstance(row["answers"], dict) else ast.literal_eval(row["answers"])
        questions.append((row["context"], row["question"], answers["text"][0]))
    return contexts, questions


def qa_prompt(context, question):
    return CHAT.format(question=f"阅读下面的文章，用文章中的原话简短回答问题。\n\n{context}\n\n问题：{question}")


def chunks(tokenizer, text, length, limit):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return [ids[start:start + length] for start in range(0, min(len(ids), limit) - length + 1, length)]


class Scales:
    def __init__(self, model, heads, head_dim):
        layers = model.model.layers
        width = model.config.intermediate_size
        self.mlp = torch.zeros(len(layers), width, requires_grad=True)
        self.head = torch.zeros(len(layers), heads, requires_grad=True)
        self.enabled = True
        for index, layer in enumerate(layers):
            layer.mlp.down_proj.register_forward_pre_hook(self.mlp_hook(index))
            layer.self_attn.o_proj.register_forward_pre_hook(self.head_hook(index, head_dim))

    def mlp_hook(self, index):
        def hook(module, inputs):
            return (inputs[0] * self.mlp[index].exp(),) if self.enabled else None
        return hook

    def head_hook(self, index, head_dim):
        def hook(module, inputs):
            return (inputs[0] * self.head[index].exp().repeat_interleave(head_dim),) if self.enabled else None
        return hook


def lm_loss(model, ids):
    logits = model(ids[None]).logits[0, :-1]
    return F.cross_entropy(logits, ids[1:], reduction="mean")


def answer_loss(model, tokenizer, prompt, answer):
    prefix = tokenizer(prompt, return_tensors="pt").input_ids[0]
    target = tokenizer(answer + "<|im_end|>", return_tensors="pt").input_ids[0]
    ids = torch.cat([prefix, target])[-700:]
    logits = model(ids[None]).logits[0, -len(target) - 1:-1]
    return F.cross_entropy(logits, target, reduction="mean")


def evaluate(model, tokenizer, sets):
    result = {}
    with torch.no_grad():
        for name, items in sets.items():
            if name == "cmrc_qa":
                losses = [answer_loss(model, tokenizer, qa_prompt(c, q), a).item() for c, q, a in items]
            else:
                losses = [lm_loss(model, ids).item() for ids in items]
            result[name] = math.exp(sum(losses) / len(losses))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--decay", type=float, default=1e-2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=HERE / "scales.pt")
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(ROOT / "model/Qwen3-0.6B", dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.config
    scales = Scales(model, config.num_attention_heads, config.head_dim)
    wiki_train = chunks(tokenizer, wiki("train"), 512, 400000)
    cmrc_contexts, cmrc_questions = cmrc("train")
    cmrc_train = chunks(tokenizer, "\n\n".join(cmrc_contexts), 512, 400000)
    code = json.loads((HERE / "corpus.json").read_text())
    code_train = chunks(tokenizer, "\n\n".join(code[:60]), 512, 300000)
    chats = [tokenizer(r["text"], return_tensors="pt").input_ids[0][:512]
             for r in json.loads((HERE / "chats.json").read_text()) if r["split"] == "train"]
    val_contexts, val_questions = cmrc("validation")
    random.Random(1).shuffle(val_questions)
    validation = {
        "wiki_val": chunks(tokenizer, wiki("validation"), 512, 512 * 40),
        "cmrc_text": chunks(tokenizer, "\n\n".join(val_contexts), 512, 512 * 30),
        "code_held": chunks(tokenizer, "\n\n".join(code[60:70]), 512, 512 * 20),
        "cmrc_qa": val_questions[:80],
    }
    teacher = None
    print({key: len(value) for key, value in validation.items()}, len(wiki_train), len(cmrc_train), len(code_train), flush=True)
    history = []
    started = time.time()
    baseline = evaluate(model, tokenizer, validation)
    print("baseline", json.dumps(baseline), flush=True)
    history.append({"step": 0, **baseline})
    optimizer = torch.optim.Adam([scales.mlp, scales.head], lr=args.lr)
    best = (sum(math.log(v) for v in baseline.values()), 0, scales.mlp.detach().clone(), scales.head.detach().clone())
    kinds = ["wiki", "cmrc", "qa", "code", "chat"]
    for step in range(1, args.steps + 1):
        kind = kinds[step % len(kinds)]
        if kind == "wiki":
            loss = lm_loss(model, random.choice(wiki_train))
        elif kind == "cmrc":
            loss = lm_loss(model, random.choice(cmrc_train))
        elif kind == "code":
            loss = lm_loss(model, random.choice(code_train))
        elif kind == "qa":
            context, question, answer = random.choice(cmrc_questions)
            loss = answer_loss(model, tokenizer, qa_prompt(context, question), answer)
        else:
            ids = random.choice(chats)
            with torch.no_grad():
                scales.enabled = False
                target = F.log_softmax(model(ids[None]).logits[0], -1)
                scales.enabled = True
            prediction = F.log_softmax(model(ids[None]).logits[0], -1)
            loss = F.kl_div(prediction, target, log_target=True, reduction="batchmean")
        penalty = args.decay * (scales.mlp.pow(2).mean() + scales.head.pow(2).mean())
        optimizer.zero_grad()
        (loss + penalty).backward()
        for group in optimizer.param_groups:
            group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        optimizer.step()
        if step % 25 == 0:
            print(f"step {step} {kind} loss={loss.item():.4f} |mlp|max={scales.mlp.abs().max().item():.3f} "
                  f"|head|max={scales.head.abs().max().item():.3f} {time.time() - started:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            current = evaluate(model, tokenizer, validation)
            print(f"eval {step}", json.dumps(current), flush=True)
            history.append({"step": step, **current})
            score = sum(math.log(v) for v in current.values())
            if score < best[0]:
                best = (score, step, scales.mlp.detach().clone(), scales.head.detach().clone())
    torch.save({"mlp": best[2], "head": best[3], "step": best[1], "history": history}, args.output)
    print("best step", best[1], flush=True)


main()
