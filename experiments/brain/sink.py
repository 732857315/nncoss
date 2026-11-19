import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path("/Users/admin/Desktop/prj2026/nncoss")
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(ROOT / "model/Qwen3-0.6B", dtype=torch.float32, attn_implementation="eager").eval()
layers = model.model.layers
reference = torch.load("teacher_logits.pt")["brain"][:6]
scale = {"value": 1.0}
def pre(module, inputs):
    x = inputs[0].clone()
    x[..., 55] *= scale["value"]
    return (x,)
layers[2].mlp.down_proj.register_forward_pre_hook(pre)
column = layers[2].mlp.down_proj.weight[:, 55]
top = column.abs().topk(4)
print("55 号神经元写入残差流的主要维度:", top.indices.tolist(), [round(v, 3) for v in column[top.indices].tolist()])
for value in (1.0, 0.0, 0.5, 2.0):
    scale["value"] = value
    sink, kls, norms = [], [], []
    for ids, expected in reference:
        out = model(ids, output_attentions=True, output_hidden_states=True)
        sink.append([a[0, :, 8:, 0].mean().item() for a in out.attentions])
        norms.append(out.hidden_states[10][0, 0].norm().item())
        actual = F.log_softmax(out.logits[0], -1)
        expected_f = expected.float()
        kls += (expected_f.exp() * (expected_f - actual)).sum(-1)[:-1].tolist()
    sink = np.mean(sink, 0)
    print(f"强度×{value}: KL={np.mean(kls):.4f}  第0位残差范数={np.mean(norms):.0f}  各层注意力落在第0位的比例: " + " ".join(f"{s:.2f}" for s in sink))
