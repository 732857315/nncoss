"""快速考评：随机抽几道高考客观题、主观题和 ACM 题，满分 100 分，判断模型学会了没有。

每道题分值相同。主观题（解答题、填空题、阅读与写作等）按参考答案人工评审：脚本先把题目、参考答案与模型的
回答写进 tmp/quick/<标签>/review.json，评审者为每题填入 0–1 的得分比例后，用 --finalize <标签> 算出总分。
高考题从 9 个单选科目的全部题目中随机抽取（排除用来定位改动的定位题），按选项字母判分；
ACM 题从 CodeContests 测试集中随机抽取，要求模型写出完整的 Python 3 或 C++ 程序，
在本机用官方的公开、隐藏与生成测试数据判题，全部通过才得分。作答由 nncoss-know 完成，本脚本只抽题与判分。

等级：90 分及以上优秀，80 分良好，70 分普通，60 分及格，60 分以下为没学会。
不指定 --seed 时每次随机抽题；比较两个模型时用同一个 --seed，保证题目相同。
"""

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_gaokao

ACM = ROOT / "modelbook/eval/code_contests/test.parquet"
SUBJECTIVE = ROOT / "modelbook/eval/gaokao/subjective"
ACM_PROMPT = ("下面是一道程序设计竞赛题。请用 Python 3 或 C++ 写出完整程序：从标准输入读取，向标准输出输出。"
              "时间限制 {time} 秒。把最终程序放在一个代码块里。\n\n{description}")
MAX_TESTS = 1000
# 竞赛代码常用 GCC 的 <bits/stdc++.h>，macOS 的 clang 没有，判题时在临时目录里提供一个等价头文件。
STDCXX = "".join(f"#include <{name}>\n" for name in (
    "algorithm", "array", "bitset", "cassert", "cctype", "cfloat", "chrono", "climits", "cmath", "complex", "cstdint",
    "cstdio", "cstdlib", "cstring", "ctime", "deque", "fstream", "functional", "iomanip", "iostream", "iterator",
    "limits", "list", "map", "memory", "numeric", "optional", "queue", "random", "set", "sstream", "stack", "string",
    "tuple", "unordered_map", "unordered_set", "utility", "vector"))
GRADES = ((90, "优秀"), (80, "良好"), (70, "普通"), (60, "及格"), (0, "没学会"))


def cxx_flags():
    """macOS Command Line Tools 自带的 c++/v1 目录可能不完整并遮住 SDK 中完整的 libc++ 头文件，此时改用 SDK 的。"""
    if sys.platform != "darwin" or not shutil.which("xcrun"):
        return []
    sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True).stdout.strip()
    headers = Path(sdk) / "usr/include/c++/v1"
    return ["-nostdinc++", "-isystem", str(headers)] if (headers / "algorithm").exists() else []


def grade(score):
    return next(name for floor, name in GRADES if score >= floor)


def exam_questions(count, rng):
    tune = {question["id"] for question in eval_gaokao.questions_for("tune", 12)}
    pool = [question for question in eval_gaokao.prepare(100000, "quick") if question["id"] not in tune]
    return [{**question, "kind": "exam"} for question in rng.sample(pool, count)]


def subjective_questions(count, rng):
    pool = []
    for path in sorted(SUBJECTIVE.glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8"))["example"]:
            reference = "\n".join(str(row.get(key) or "").strip() for key in ("answer", "analysis")).strip()
            if reference:
                pool.append({"id": f"{path.stem}-{row['year']}-{row['index']}", "kind": "subjective",
                             "question": row["question"].strip() + "\n请写出解答过程和答案。",
                             "reference": reference, "full": row.get("score")})
    return rng.sample(pool, count)


def acm_questions(count, rng, max_rating):
    import pyarrow.parquet as pq
    rows = pq.read_table(ACM, columns=["name", "description", "public_tests", "private_tests", "generated_tests",
                                       "time_limit", "cf_rating"]).to_pylist()
    rows = [row for row in rows if not max_rating or (row["cf_rating"] or 0) <= max_rating]
    questions = []
    for row in rng.sample(rows, count):
        tests = []
        for group in ("public_tests", "private_tests", "generated_tests"):
            tests += list(zip(row[group]["input"], row[group]["output"]))
        seconds = (row["time_limit"] or {}).get("seconds") or 2
        questions.append({"id": f"acm-{row['name'].split('.')[0]}", "kind": "acm", "name": row["name"],
                          "rating": row["cf_rating"], "seconds": seconds, "tests": tests[:MAX_TESTS],
                          "question": ACM_PROMPT.format(time=seconds, description=row["description"].strip())})
    return questions


def extract_program(reply):
    blocks = re.findall(r"```([\w+#-]*)\n(.*?)```", reply, re.S)
    for language, code in reversed(blocks):
        language = language.lower()
        if language in ("cpp", "c++", "cc", "cxx") or ("#include" in code and language in ("", "c")):
            return "cpp", code
        if language in ("python", "python3", "py") or (not language and "#include" not in code):
            return "python", code
    return None, None


def same_output(actual, expected):
    left, right = actual.split(), expected.split()
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a == b:
            continue
        try:
            if abs(float(a) - float(b)) > 1e-6 * max(1.0, abs(float(b))):
                return False
        except ValueError:
            return False
    return True


def judge_acm(question, reply):
    language, code = extract_program(reply)
    if not code:
        return False, "没有找到程序"
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
        directory = Path(directory)
        if language == "cpp":
            source = directory / "main.cpp"
            source.write_text(code, encoding="utf-8")
            compiler = shutil.which("g++") or shutil.which("clang++")
            (directory / "include/bits").mkdir(parents=True)
            (directory / "include/bits/stdc++.h").write_text(STDCXX, encoding="utf-8")
            built = subprocess.run([compiler, "-O2", "-std=c++17", *cxx_flags(), "-I", str(directory / "include"), "-o",
                                    str(directory / "main"), str(source)],
                                   capture_output=True, text=True, timeout=120)
            if built.returncode != 0:
                return False, "编译失败：" + built.stderr.strip().splitlines()[0][:200] if built.stderr.strip() else "编译失败"
            command = [str(directory / "main")]
            limit = question["seconds"] * 2
        else:
            source = directory / "main.py"
            source.write_text(code, encoding="utf-8")
            command = [sys.executable, "-I", str(source)]
            limit = question["seconds"] * 5
        for number, (given, expected) in enumerate(question["tests"], 1):
            try:
                result = subprocess.run(command, input=given, capture_output=True, text=True, timeout=limit,
                                        cwd=directory)
            except subprocess.TimeoutExpired:
                return False, f"第 {number} 组测试超时"
            if result.returncode != 0:
                return False, f"第 {number} 组测试运行错误：{result.stderr.strip()[-200:]}"
            if not same_output(result.stdout, expected):
                return False, f"第 {number} 组测试答案错误"
    return True, f"通过全部 {len(question['tests'])} 组测试"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--finalize", action="store_true", help="读入评审后的 review.json，计算总分")
    parser.add_argument("--store", type=Path, help="知识库目录；给出时模型可以检索，并复制一份供本次使用")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--browse", action="store_true", help="允许模型用 browse 打开网页")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--exam", type=int, default=6, help="高考客观题数量")
    parser.add_argument("--subjective", type=int, default=2, help="主观题数量，由评审者按参考答案打分")
    parser.add_argument("--acm", type=int, default=2, help="ACM 题数量")
    parser.add_argument("--acm-max-rating", type=int, default=0, help="只从不高于该 Codeforces 难度的题中抽，0 表示不限")
    parser.add_argument("--seed", type=int, help="抽题种子，默认随机")
    parser.add_argument("--turn-tokens", type=int, default=0)
    parser.add_argument("--kv", choices=["f32", "f16", "i8"])
    parser.add_argument("--work", type=Path, default=ROOT / "tmp/quick")
    args = parser.parse_args()
    work = args.work / args.label
    if args.finalize:
        return finalize(work)
    if not args.model or not args.tokenizer:
        parser.error("需要 --model 与 --tokenizer")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 31)
    rng = random.Random(seed)
    questions = (exam_questions(args.exam, rng) + subjective_questions(args.subjective, rng)
                 + (acm_questions(args.acm, rng, args.acm_max_rating) if args.acm else []))
    work.mkdir(parents=True, exist_ok=True)
    (work / "questions.jsonl").write_text("".join(json.dumps({"id": q["id"], "question": q["question"]}, ensure_ascii=False)
                                                  + "\n" for q in questions), encoding="utf-8")
    store = work / "store"
    if store.exists():
        shutil.rmtree(store)
    if args.store:
        shutil.copytree(args.store, store)
    else:
        store.mkdir()
    command = [str(ROOT / "target/release/nncoss-know"), "batch", "--store", str(store), "--model", str(args.model),
               "--tokenizer", str(args.tokenizer), "--input", str(work / "questions.jsonl"),
               "--output", str(work / "answers.jsonl"),
               "--turn-tokens", str(args.turn_tokens or (4096 if args.think else 2048))]
    command += ["--think"] * args.think + ["--browse"] * args.browse + ["--prefetch"] * args.prefetch
    if not args.store:
        command.append("--no-tools")
    if args.kv:
        command += ["--kv", args.kv]
    started = time.time()
    subprocess.run(command, check=True)
    answers = {answer["id"]: answer for answer in map(json.loads, (work / "answers.jsonl").read_text(encoding="utf-8").splitlines())}
    details = []
    for question in questions:
        reply = answers.get(question["id"], {}).get("reply", "")
        if question["kind"] == "exam":
            picked = eval_gaokao.choice(reply)
            ok = picked == question["answer"]
            details.append({"id": question["id"], "kind": "exam", "ratio": float(ok),
                            "note": f"答 {picked}，标准答案 {question['answer']}"})
        elif question["kind"] == "acm":
            ok, note = judge_acm(question, reply)
            details.append({"id": question["id"], "kind": "acm", "ratio": float(ok),
                            "note": f"{question['name']}（难度 {question['rating']}）：{note}"})
        else:
            details.append({"id": question["id"], "kind": "subjective", "ratio": None, "note": "待评审",
                            "full": question["full"], "question": question["question"],
                            "reference": question["reference"], "reply": reply})
    report = {"label": args.label, "model": str(args.model), "seed": seed, "store": str(args.store) if args.store else None,
              "think": args.think, "seconds": round(time.time() - started), "details": details}
    (work / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    pending = sum(detail["ratio"] is None for detail in details)
    if pending:
        print(f"有 {pending} 道主观题待评审：在 {work / 'review.json'} 中为每题填入 0–1 的 ratio，再运行 --finalize")
    finalize(work)


def finalize(work):
    report = json.loads((work / "review.json").read_text(encoding="utf-8"))
    details = report["details"]
    points = 100 / len(details)
    for detail in details:
        mark = "? " if detail["ratio"] is None else ("✓ " if detail["ratio"] >= 1 else ("✗ " if detail["ratio"] <= 0 else "~ "))
        print(mark + detail["id"] + "  " + detail["note"])
    if any(detail["ratio"] is None for detail in details):
        print("尚有主观题未评审，暂不计分")
        return
    score = round(points * sum(detail["ratio"] for detail in details), 1)
    report.update(score=score, grade=grade(score))
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{report['label']}：{score} 分（{grade(score)}），种子 {report['seed']}，用时 {report['seconds']} 秒")


if __name__ == "__main__":
    main()
