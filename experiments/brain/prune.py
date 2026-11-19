import json, shutil, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def ranking(scores, method, lam):
    if method == "taylor":
        value = scores["brain_taylor"]
    elif method == "wanda":
        value = np.sqrt(scores["brain_energy"]) * scores["down_norm"]
    elif method == "contrast":
        value = None
    elif method == "random":
        value = np.random.default_rng(0).random(scores["brain_taylor"].shape)
    else:
        raise ValueError(method)
    if method == "contrast":
        brain = np.sqrt(scores["brain_energy"]) * scores["down_norm"]
        reading = np.sqrt(scores["open_energy"]) * scores["down_norm"]
        recall = np.sqrt(scores["know_energy"]) * scores["down_norm"]
        brain = brain / brain.mean(1, keepdims=True)
        reading = reading / reading.mean(1, keepdims=True)
        recall = recall / recall.mean(1, keepdims=True)
        value = np.maximum(brain, reading) - lam * np.maximum(recall - reading, 0)
        return np.argsort(-value, axis=1)
    if lam:
        brain = scores["brain_taylor"] / scores["brain_taylor"].mean(1, keepdims=True)
        know = scores["know_taylor"] / scores["know_taylor"].mean(1, keepdims=True)
        value = brain - lam * np.maximum(know - brain, 0)
    return np.argsort(-value, axis=1)


def prune(source, output, keep, method="taylor", lam=0.0):
    scores = np.load(HERE / "scores.npz")
    order = ranking(scores, method, lam)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    config["intermediate_size"] = keep
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(source / name, output / name)
    tensors = {}
    with safe_open(source / "model.safetensors", "pt") as file:
        for key in file.keys():
            tensor = file.get_tensor(key)
            if ".mlp." in key:
                layer = int(key.split(".")[2])
                index = torch.from_numpy(np.sort(order[layer, :keep]))
                tensor = tensor[:, index] if "down_proj" in key else tensor[index]
            tensors[key] = tensor.contiguous()
    save_file(tensors, output / "model.safetensors", metadata={"format": "pt"})


if __name__ == "__main__":
    prune(ROOT / "model/Qwen3-0.6B", Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "taylor",
          float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
