import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent))
import data
ROOT = Path(__file__).resolve().parents[2]
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(ROOT / "model/Qwen3-0.6B", dtype=torch.float32).eval()
layers = model.model.layers
reference = torch.load("teacher_logits.pt")["brain"]
closed = [(data.closed(item), item[2]) for item in data.FACTS[::2]]
fiction = [(data.opened(item, data.FICTION, index), item[2]) for index, item in enumerate(data.FICTION)]
encoded = {}


def answer_nll(pairs, key):
    if key not in encoded:
        encoded[key] = [(tokenizer(p + a, return_tensors="pt").input_ids, len(tokenizer(p).input_ids)) for p, a in pairs]
    total = count = 0
    for ids, start in encoded[key]:
        logits = F.log_softmax(model(ids).logits[0, start - 1:-1], -1)
        total -= float(logits.gather(1, ids[0, start:, None]).sum())
        count += ids.shape[1] - start
    return total / count


def measure():
    kls = []
    for ids, expected in reference:
        actual = F.log_softmax(model(ids).logits[0], -1)
        expected = expected.float()
        kls += (expected.exp() * (expected - actual)).sum(-1)[:-1].tolist()
    return {"brain_kl": float(np.mean(kls)), "closed_nll": answer_nll(closed, "closed"), "fiction_nll": answer_nll(fiction, "fiction")}


def neurons_hook(groups):
    handles = []
    for layer, indices in groups.items():
        index = torch.tensor(sorted(indices))
        def pre(module, inputs, index=index):
            x = inputs[0].clone()
            x[..., index] = 0
            return (x,)
        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(pre))
    return handles


def block_hook(kind, layer):
    module = layers[layer].mlp if kind == "mlp" else layers[layer].self_attn
    def post(module, inputs, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + tuple(output[1:])
        return torch.zeros_like(output)
    return [module.register_forward_hook(post)]


def run(label, handles):
    started = time.time()
    result = measure()
    for handle in handles:
        handle.remove()
    result["label"] = label
    print(json.dumps(result, ensure_ascii=False), f"{time.time() - started:.0f}s", flush=True)
    return result


mode = sys.argv[1]
results = [run("baseline", [])]
if mode == "neurons":
    rng = np.random.default_rng(0)
    specials = {"L2#55": {2: [55]}, "L26#1564": {26: [1564]}, "L27#1551": {27: [1551]}, "L7#2": {7: [2]},
                "L10#17": {10: [17]}, "L11#43": {11: [43]}, "L14#4": {14: [4]}, "L1#2997": {1: [2997]}}
    for label, groups in specials.items():
        results.append(run(label, neurons_hook(groups)))
    for trial in range(2):
        groups = {}
        for flat in rng.choice(28 * 3072, 1000, replace=False):
            groups.setdefault(int(flat // 3072), []).append(int(flat % 3072))
        results.append(run(f"random1000#{trial}", neurons_hook(groups)))
else:
    for layer in range(28):
        results.append(run(f"mlp{layer}", block_hook("mlp", layer)))
        results.append(run(f"attn{layer}", block_hook("attn", layer)))
Path(f"damage_{mode}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
