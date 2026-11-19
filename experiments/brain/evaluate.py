import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/Users/admin/Desktop/prj2026/nncoss")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import data
import check_q4_llm

torch.set_grad_enabled(False)


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()


def texts():
    held = [record["text"] for record in json.loads((HERE / "chats.json").read_text()) if record["split"] == "held"]
    samples = list(check_q4_llm.SAMPLES)
    return {"brain": held, "samples": samples}


def teacher_logits(tokenizer, teacher, groups):
    cache = HERE / "teacher_logits.pt"
    if cache.exists():
        return torch.load(cache)
    result = {}
    for name, items in groups.items():
        result[name] = []
        for text in items:
            ids = tokenizer(text, return_tensors="pt").input_ids
            result[name].append((ids, F.log_softmax(teacher(ids).logits[0], -1).to(torch.bfloat16)))
    torch.save(result, cache)
    return result


def divergence(model, reference):
    output = {}
    for name, items in reference.items():
        agree = total = 0
        kls = []
        nll = [0.0, 0.0]
        for ids, expected in items:
            expected = expected.float()
            actual = F.log_softmax(model(ids).logits[0], -1)
            labels = ids[0, 1:]
            agree += int((expected[:-1].argmax(-1) == actual[:-1].argmax(-1)).sum())
            total += len(labels)
            kls += (expected.exp() * (expected - actual)).sum(-1)[:-1].tolist()
            nll[0] -= float(expected[:-1].gather(1, labels[:, None]).sum())
            nll[1] -= float(actual[:-1].gather(1, labels[:, None]).sum())
        output[name] = {"top1": agree / total, "kl": sum(kls) / len(kls), "ppl_ratio": float(torch.exp(torch.tensor((nll[1] - nll[0]) / total)))}
    return output


def answers(model, tokenizer, prompts, aliases):
    correct = 0
    outputs = []
    for prompt, accepted in zip(prompts, aliases):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated = model.generate(ids, max_new_tokens=28, do_sample=False, pad_token_id=151643)
        text = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
        hit = any(alias.lower() in text.lower() for alias in accepted)
        correct += hit
        outputs.append(text)
    return correct / len(prompts), outputs


def qa(model, tokenizer):
    closed = [data.closed(item) for item in data.FACTS]
    opened = [data.opened(item, data.FACTS, index) for index, item in enumerate(data.FACTS)]
    fiction = [data.opened(item, data.FICTION, index) for index, item in enumerate(data.FICTION)]
    result = {}
    for name, prompts, pool in (("closed_book", closed, data.FACTS), ("open_book", opened, data.FACTS), ("fiction_book", fiction, data.FICTION)):
        accuracy, outputs = answers(model, tokenizer, prompts, [item[1] for item in pool])
        result[name] = accuracy
        result[name + "_outputs"] = outputs
    hits = []
    for prompt, accepted in data.ICL:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=151643)
        text = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
        hits.append(any(text.lower().startswith(alias.lower()) for alias in accepted))
        result.setdefault("icl_outputs", []).append(text)
    result["icl"] = sum(hits) / len(hits)
    return result


def main():
    path = Path(sys.argv[1])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
    groups = texts()
    started = time.time()
    if (HERE / "teacher_logits.pt").exists():
        reference = torch.load(HERE / "teacher_logits.pt")
    else:
        reference = teacher_logits(tokenizer, load(ROOT / "model/Qwen3-0.6B"), groups)
    model = load(path)
    report = {"model": str(path)}
    report.update(divergence(model, reference))
    report.update(qa(model, tokenizer))
    summary = {key: value for key, value in report.items() if not key.endswith("_outputs")}
    print(json.dumps(summary, ensure_ascii=False), f"{time.time() - started:.0f}s", flush=True)
    (HERE / "reports").mkdir(exist_ok=True)
    (HERE / "reports" / (path.name + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=1))


main()
