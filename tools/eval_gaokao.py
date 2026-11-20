import argparse
import json
import random
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = {
    "数学I": "2010-2022_Math_I_MCQs.json",
    "数学II": "2010-2022_Math_II_MCQs.json",
    "物理": "2010-2022_Physics_MCQs.json",
    "化学": "2010-2022_Chemistry_MCQs.json",
    "生物": "2010-2022_Biology_MCQs.json",
    "历史": "2010-2022_History_MCQs.json",
    "政治": "2010-2022_Political_Science_MCQs.json",
    "语文": "2010-2022_Chinese_Lang_and_Usage_MCQs.json",
    "英语": "2010-2013_English_MCQs.json",
}


def prepare(per_subject, seed):
    questions = []
    for subject, name in SUBJECTS.items():
        rows = json.loads((ROOT / "modelbook/eval/gaokao" / name).read_text(encoding="utf-8"))["example"]
        rows = [row for row in rows if len(row["answer"]) == 1 and row["answer"][0] in ("A", "B", "C", "D")]
        for row in random.Random(f"{seed}-{subject}").sample(rows, min(per_subject, len(rows))):
            questions.append({"id": f"{subject}-{row['year']}-{row['index']}", "subject": subject,
                              "question": row["question"].strip() + "\n请给出正确选项。", "answer": row["answer"][0]})
    return questions


def questions_for(split, per_subject):
    if split == "test":
        return prepare(per_subject, "gaokao")
    test_ids = {question["id"] for question in prepare(per_subject, "gaokao")}
    by_subject = {}
    for question in prepare(10_000, "tune"):
        if question["id"] not in test_ids:
            by_subject.setdefault(question["subject"], []).append(question)
    return [question for items in by_subject.values() for question in items[:per_subject]]


LETTER = r"(?<![A-Za-z])([A-D])(?![A-Za-z])"
# 模型明确给出结论的说法，如“答案：C”“正确选项是 **B**”“故选 D”“\\boxed{A}”。
STRONG_ANSWERS = (r"\\boxed\{([^{}]*)\}",
                  r"(?:答案|正确选项|正确的选项|故选|应选|选择)\s*(?:应?是|为|应为)?\s*[:：]?\s*\**\s*[（(]?([A-D])(?![A-Za-z])")
WEAK_ANSWERS = (r"选项\s*[:：]?\s*\**\s*([A-D])(?![A-Za-z])",)


def choice(reply):
    """取模型最后一处明确给出的选项。模型逐项分析时常写“分析各选项：**A.** …”，
    开头出现的字母并不是它的选择（旧版取第一处，曾把这类回答一律判成 A），最后的结论才是。"""
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    for patterns in (STRONG_ANSWERS, WEAK_ANSWERS):
        found = sorted((match.start(), match.group(1)) for pattern in patterns for match in re.finditer(pattern, text))
        for _, candidate in reversed(found):
            letters = re.findall(LETTER, candidate)
            if letters:
                return letters[0]
    letters = re.findall(LETTER, text)
    return letters[-1] if letters else None


def judge(questions, answers):
    by_id = {answer["id"]: answer for answer in answers}
    table = defaultdict(lambda: [0, 0])
    details = []
    for question in questions:
        answer = by_id.get(question["id"])
        picked = choice(answer["reply"]) if answer else None
        ok = picked == question["answer"]
        for key in (question["subject"], "全部"):
            table[key][0] += ok
            table[key][1] += 1
        details.append({"id": question["id"], "expected": question["answer"], "picked": picked, "ok": ok,
                        "searches": len(answer["searches"]) if answer else 0, "notes": len(answer["notes"]) if answer else 0})
    return {key: {"correct": value[0], "total": value[1], "accuracy": round(value[0] / value[1], 3)} for key, value in table.items()}, details


def main():
    parser = argparse.ArgumentParser(description="用 nncoss-know 在高考客观题上测评；作答由 Rust 完成，本脚本只负责抽题与判分")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--store", type=Path, help="基础知识库目录；每次运行会复制一份，笔记只在这次运行中积累")
    parser.add_argument("--label", required=True)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--prefetch", action="store_true", help="作答前先用题目自动检索一次知识库")
    parser.add_argument("--per-subject", type=int, default=12)
    parser.add_argument("--split", choices=["test", "tune"], default="test", help="test 为测验题，tune 为定位改动用的题，两者不重叠")
    parser.add_argument("--turn-tokens", type=int, default=0)
    parser.add_argument("--kv", choices=["f32", "f16", "i8"], help="KV Cache 精度，默认由引擎决定")
    parser.add_argument("--sparsity", help="decode 时丢弃的输入通道比例，格式同 nncoss-know --sparsity")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp")
    parser.add_argument("--rejudge", action="store_true", help="不重新作答，只用现有 answers.jsonl 重新判分并改写 report.json")
    args = parser.parse_args()
    work = args.work / args.label
    if args.rejudge:
        return rejudge(work, args.split, args.per_subject)
    work.mkdir(parents=True, exist_ok=True)
    questions = questions_for(args.split, args.per_subject)
    (work / "questions.jsonl").write_text("".join(json.dumps({"id": q["id"], "question": q["question"]}, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    store = work / "store"
    if store.exists():
        shutil.rmtree(store)
    if args.store and not args.no_tools:
        shutil.copytree(args.store, store)
    else:
        store.mkdir()
    command = [str(ROOT / "target/release/nncoss-know"), "batch", "--store", str(store), "--model", str(args.model),
               "--tokenizer", str(args.tokenizer), "--input", str(work / "questions.jsonl"), "--output", str(work / "answers.jsonl")]
    if args.think:
        command.append("--think")
    if args.no_tools:
        command.append("--no-tools")
    if args.prefetch:
        command.append("--prefetch")
    if args.kv:
        command += ["--kv", args.kv]
    if args.sparsity:
        command += ["--sparsity", args.sparsity]
    turn_tokens = args.turn_tokens or (3072 if args.think else 1024)
    command += ["--turn-tokens", str(turn_tokens)]
    started = time.time()
    subprocess.run(command, check=True)
    answers = [json.loads(line) for line in (work / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    table, details = judge(questions, answers)
    report = {"label": args.label, "model": str(args.model), "think": args.think, "tools": not args.no_tools, "prefetch": args.prefetch, "kv": args.kv,
              "sparsity": args.sparsity,
              "seconds": round(time.time() - started), "notes_written": sum(d["notes"] for d in details),
              "searches": sum(d["searches"] for d in details), "table": table, "details": details}
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(args.label, json.dumps({key: value["accuracy"] for key, value in table.items()}, ensure_ascii=False),
          f"检索 {report['searches']} 次，新笔记 {report['notes_written']} 条，用时 {report['seconds']}s", flush=True)


def rejudge(work, split, per_subject):
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    answers = [json.loads(line) for line in (work / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    table, details = judge(questions_for(split, per_subject), answers)
    before = report["table"]["全部"]["accuracy"]
    report.update(table=table, details=details)
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report["label"], f"重新判分：{before} -> {table['全部']['accuracy']}",
          json.dumps({key: value["accuracy"] for key, value in table.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
