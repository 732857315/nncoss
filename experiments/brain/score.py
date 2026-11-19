import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import data

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(ROOT / "model/Qwen3-0.6B", dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.model.layers
    width = model.config.intermediate_size
    captured = [None] * len(layers)

    def hook(index):
        def run(module, inputs):
            inputs[0].retain_grad()
            captured[index] = inputs[0]
        return run

    for index, layer in enumerate(layers):
        layer.mlp.down_proj.register_forward_pre_hook(hook(index))

    def accumulate(texts, answer_only):
        taylor = np.zeros((len(layers), width))
        energy = np.zeros((len(layers), width))
        peak = np.zeros((len(layers), width))
        special = np.zeros((len(layers), width))
        tokens = 0
        special_tokens = 0
        started = time.time()
        for number, (prefix, answer) in enumerate(texts):
            ids = tokenizer(prefix + answer, return_tensors="pt").input_ids[:, :640]
            start = len(tokenizer(prefix).input_ids) if answer_only else 1
            embeds = model.model.embed_tokens(ids).detach().requires_grad_(True)
            logits = model(inputs_embeds=embeds).logits[0, :-1]
            labels = ids[0, 1:]
            mask = torch.zeros(len(labels), dtype=torch.bool)
            mask[start - 1:] = True
            loss = F.cross_entropy(logits[mask], labels[mask], reduction="sum")
            loss.backward()
            marked = (ids[0, :-1] >= 151643) | (labels >= 151643)
            for index in range(len(layers)):
                activation = captured[index][0, :-1].detach()
                gradient = captured[index].grad[0, :-1]
                contribution = (activation * gradient).abs()
                taylor[index] += contribution.sum(0).double().numpy()
                energy[index] += (activation[mask] ** 2).sum(0).double().numpy()
                peak[index] = np.maximum(peak[index], contribution.max(0).values.double().numpy())
                special[index] += (activation[marked] ** 2).sum(0).double().numpy()
            tokens += int(mask.sum())
            special_tokens += int(marked.sum())
            if number % 20 == 0:
                print(f"{number}/{len(texts)} {time.time() - started:.0f}s", flush=True)
        return taylor / tokens, energy / tokens, peak, special / max(special_tokens, 1)

    chats = [record["text"] for record in json.loads((HERE / "chats.json").read_text()) if record["split"] == "train"]
    brain = [(text, "") for text in chats]
    knowledge = [(data.closed(item), item[2]) for item in data.FACTS]
    norms = np.stack([layer.mlp.down_proj.weight.norm(dim=0).numpy() for layer in layers])
    result = {"down_norm": norms}
    opened = [(data.opened(item, data.FACTS, index), item[2]) for index, item in enumerate(data.FACTS)]
    fiction = [(data.opened(item, data.FICTION, index), item[2]) for index, item in enumerate(data.FICTION)]
    for name, texts, answer_only in (("brain", brain, False), ("know", knowledge, True), ("open", opened + fiction, True)):
        values = accumulate(texts, answer_only)
        for key, value in zip(("taylor", "energy", "peak", "special"), values):
            result[f"{name}_{key}"] = value
    np.savez(HERE / "scores.npz", **result)
    print("saved", flush=True)


main()
