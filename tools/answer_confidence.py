"""用模型自己对最终选项的把握，在两次作答（如“不用书”与“先查书”）之间选择；只测量，不改参数。

“大脑 + 书籍”要高于“大脑单独作答”，就不能让书籍把本来会做的题带偏。这里检验一种不训练的做法：
同一道题作答两次，各自在作答记录之后追问“最终答案是哪个选项”，读出模型给 A–D 的概率，
由模型自己的把握决定采用哪一次的答案。比较的规则：
  first / second —— 总是采用其中一次（判分与 eval_gaokao.py 相同）；
  confident      —— 采用追问时最高概率更大的那一次；
  product        —— 两次追问的概率相乘后取最大的选项；
  oracle         —— 两次中任一次答对就算对，是这种选择方式的上限。
作答记录取自 eval_gaokao.py 的输出目录（answers.jsonl 中的 transcript），追问用 transformers 的 f32 模型计算。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_gaokao

PROBE = ("<|im_start|>user\n你的最终答案是哪个选项？只回答一个字母。<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n")
LETTERS = "ABCD"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_context(record):
    """作答记录截到作答回复结束为止（之后可能还有归纳笔记的一轮），再接上追问。"""
    transcript, reply = record["transcript"], record["reply"].rstrip()
    tail = reply[-200:]
    position = transcript.rfind(tail)
    if position < 0:
        return None
    return transcript[:position + len(tail)] + "<|im_end|>\n" + PROBE


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs=2, required=True, help="两个 eval_gaokao.py 输出目录（tmp/ 下的标签）")
    parser.add_argument("--split", choices=["test", "tune", "wide"], default="test")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/answer_confidence")
    parser.add_argument("--max-tokens", type=int, default=6000, help="超过这个长度的记录只保留末尾")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    letter_ids = [tokenizer.convert_tokens_to_ids(letter) for letter in LETTERS]
    questions = {question["id"]: question for question in eval_gaokao.questions_for(args.split)}
    runs = []
    for label in args.runs:
        answers = {record["id"]: record for record in read_jsonl(ROOT / "tmp" / label / "answers.jsonl")}
        runs.append(answers)

    cache_path = args.work / ("_".join(args.runs) + ".jsonl")
    cached = {(row["run"], row["id"]): row for row in read_jsonl(cache_path)} if cache_path.exists() else {}
    rows = []
    with torch.no_grad(), cache_path.open("a", encoding="utf-8") as cache:
        for question_id in questions:
            for run, answers in enumerate(runs):
                key = (args.runs[run], question_id)
                if key in cached:
                    rows.append(cached[key])
                    continue
                record = answers.get(question_id)
                context = answer_context(record) if record else None
                probabilities, mass = [0.25] * 4, 0.0
                if context:
                    ids = tokenizer(context, add_special_tokens=False).input_ids[-args.max_tokens:]
                    logits = model(torch.tensor([ids]), use_cache=False, logits_to_keep=1).logits[0, -1]
                    full = torch.softmax(logits.float(), -1)[letter_ids]
                    mass = float(full.sum())
                    probabilities = (full / full.sum()).tolist()
                row = {"run": args.runs[run], "id": question_id, "picked": eval_gaokao.choice(record["reply"]) if record else None,
                       "probe": probabilities, "mass": round(mass, 4)}
                cache.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache.flush()
                rows.append(row)
            print(question_id, flush=True)

    by_key = {(row["run"], row["id"]): row for row in rows}
    totals = {name: 0 for name in ("first", "second", "probe_first", "probe_second", "confident", "product", "oracle")}
    switched = 0
    for question_id, question in questions.items():
        expected = question["answer"]
        first, second = (by_key[(label, question_id)] for label in args.runs)
        correct = [first["picked"] == expected, second["picked"] == expected]
        totals["first"] += correct[0]
        totals["second"] += correct[1]
        totals["oracle"] += correct[0] or correct[1]
        probe_letters = [LETTERS[max(range(4), key=row["probe"].__getitem__)] for row in (first, second)]
        totals["probe_first"] += probe_letters[0] == expected
        totals["probe_second"] += probe_letters[1] == expected
        use_second = max(second["probe"]) > max(first["probe"])
        switched += use_second
        totals["confident"] += correct[1] if use_second else correct[0]
        product = [a * b for a, b in zip(first["probe"], second["probe"])]
        totals["product"] += LETTERS[max(range(4), key=product.__getitem__)] == expected
    count = len(questions)
    summary = {name: round(value / count, 3) for name, value in totals.items()}
    summary["confident_switched"] = switched
    print(json.dumps(summary, ensure_ascii=False))
    (args.work / ("_".join(args.runs) + ".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
