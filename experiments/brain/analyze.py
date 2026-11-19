import numpy as np
from prune import ranking

s = dict(np.load("scores.npz"))
old = {"brain_taylor": s["brain_taylor"], "know_taylor": s["know_taylor"], "brain_energy": s["brain_energy"], "down_norm": s["down_norm"]}
keep0 = ranking(old, "taylor", 0.0)[:, :2048]
keep1 = ranking(old, "taylor", 1.0)[:, :2048]
L, W = s["brain_taylor"].shape


def rank01(x):
    return np.argsort(np.argsort(x, 1), 1) / (W - 1)


print("在“剔除知识”方案中被额外删掉、但在普通方案中保留的神经元：它们的各项指标在本层的百分位（0.5=中位数）")
for key in ("brain_taylor", "brain_peak", "brain_special", "know_taylor", "open_taylor", "open_peak", "know_peak"):
    r = rank01(s[key])
    extra, kept, dropped = [], [], []
    for l in range(L):
        a, b = set(keep0[l]), set(keep1[l])
        extra.append(r[l, sorted(a - b)].mean() if a - b else np.nan)
        kept.append(r[l, sorted(a & b)].mean())
        dropped.append(r[l, sorted(set(range(W)) - a)].mean())
    print(f"{key:14s} 额外删掉 {np.nanmean(extra):.2f}   两方案都保留 {np.mean(kept):.2f}   两方案都删 {np.mean(dropped):.2f}")
both = rank01(s["brain_peak"])
print("普通方案删掉的神经元中，峰值重要性位于本层前 10% 的比例:",
      np.mean([(both[l, sorted(set(range(W)) - set(keep0[l]))] > 0.9).mean() for l in range(L)]).round(3))
