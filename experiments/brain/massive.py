import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent))
import data
ROOT = Path("/Users/admin/Desktop/prj2026/nncoss")
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(ROOT / "model/Qwen3-0.6B", dtype=torch.float32).eval()
layers = model.model.layers
chats = [r["text"] for r in json.loads(Path("chats.json").read_text()) if r["split"] == "train"][:24]
captured = {}
for i, layer in enumerate(layers):
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: captured.__setitem__(i, inp[0][0]))
records = []
for text in chats:
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :512]
    out = model(ids, output_hidden_states=True)
    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    for i in range(len(layers)):
        a = captured[i].abs()
        value, flat = a.flatten().max(0)
        pos, neuron = divmod(int(flat), a.shape[1])
        median = a.median().item()
        records.append((i, float(value), neuron, pos, tokens[pos], median))
    hidden = [h[0].abs().max().item() for h in out.hidden_states]
by_layer = {}
for i, value, neuron, pos, tok, median in records:
    by_layer.setdefault(i, []).append((value, neuron, pos, tok, median))
print("层  最大激活  中位激活  倍数  神经元(出现次数)  位置/token 示例")
for i in range(len(layers)):
    items = by_layer[i]
    top = max(items)
    counts = {}
    for v, n, p, t, m in items:
        counts[n] = counts.get(n, 0) + 1
    common = sorted(counts.items(), key=lambda x: -x[1])[:2]
    med = sum(x[4] for x in items) / len(items)
    print(f"{i:2d} {top[0]:9.1f} {med:8.4f} {top[0]/med:8.0f}x  {common}  pos={top[2]} tok={top[3]!r}")
print("残差流最大幅度（最后一段文本）:", [round(h) for h in hidden])
