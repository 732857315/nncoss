import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import kb
import school
from check_real_llm import checked, file_chunks
from check_wasm_llm import NativeTarget, native_library_name, prefill, sample, stream_model
from transformers import AutoTokenizer

IM_END = 151645
EOT = 151643
TOOLS = [{"type": "function", "function": {
    "name": "python",
    "description": "运行一段 Python 代码，返回它打印到标准输出的内容（以及报错信息）。",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "要运行的完整 Python 代码"}}, "required": ["code"]},
}}, {"type": "function", "function": {
    "name": "search",
    "description": "在知识库中检索事实资料（各学科知识、人物、作品、历史、地理等），返回最相关的几条。",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "要检索的关键词或问题"}}, "required": ["query"]},
}}]
ASK = {"mc": "\n请给出正确选项。", "numeric": "\n请给出最终答案。", "code": "", "design": ""}
RETRY = {"mc": "你的答案不正确，请再想一想。", "numeric": "你的答案不正确，请再想一想。", "code": "你的答案不正确，请再想一想。",
         "design": "你的代码没有通过测试：{error}\n请修正后给出完整的函数代码。"}


def run_python(code, timeout=5):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "main.py"
        path.write_text(code)
        try:
            result = subprocess.run([sys.executable, "-I", str(path)], cwd=directory, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "运行超时"
        output = result.stdout
        if result.returncode != 0:
            lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
            output += ("\n" if output else "") + (lines[-1] if lines else "运行失败")
        return result.returncode == 0, output.strip()[:800]


def tool_python(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return run_python(code)[1] or "（代码运行结束，没有打印任何内容）"
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        code = ast.unparse(tree) + f"\n__value = ({ast.unparse(last.value)})\nif __value is not None:\n    print(repr(__value) if isinstance(__value, str) else __value)\n"
    return run_python(code)[1] or "（代码运行结束，没有打印任何内容）"


def visible(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip() if "</think>" in text or "<think>" not in text else ""


def candidates(text):
    found = [(match, "first") for match in re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)]
    found += [(match.strip(), "first") for match in re.findall(r"(?:答案|输出|结果|Answer|answer)\s*(?:是|为)?\s*[:：]?\s*(.+)", text)]
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        found.append((lines[-1], "last"))
    return found


def number(text, mode):
    text = text.replace(",", "").replace("，", "")
    fraction = re.search(r"\\d?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", text) or re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?!\d)", text)
    if fraction and float(fraction.group(2)) != 0:
        return float(fraction.group(1)) / float(fraction.group(2))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    return float(numbers[0] if mode == "first" else numbers[-1])


def normalize(text):
    text = text.strip().strip("。.").replace("**", "")
    return re.sub(r"[\s`'\"\[\]\(\)（）$]", "", text)


def judge(item, reply):
    text = visible(reply)
    if not text:
        return False, "没有给出最终回答"
    kind = item["kind"]
    if kind == "design":
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S) or ([text] if f"def {item['name']}" in text else [])
        if not blocks:
            return False, "回答中没有给出函数代码"
        blocks = [block for block in blocks if f"def {item['name']}" in block] or blocks
        script = blocks[-1] + "\n\n" + "\n".join(f"assert {test}, {test!r}" for test in item["tests"]) + "\nprint('__ok__')\n"
        ok, output = run_python(script)
        if ok and output.endswith("__ok__"):
            return True, ""
        return False, output.splitlines()[-1] if output else "运行失败"
    cands = candidates(text)
    if kind == "mc":
        for cand, _ in cands:
            letters = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", cand)
            if letters:
                return letters[0] == item["answer"], ""
        letters = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text)
        return bool(letters) and letters[-1] == item["answer"], ""
    if kind == "numeric":
        for cand, mode in cands:
            value = number(cand, mode)
            if value is not None:
                return abs(value - item["value"]) <= 1e-6 + 1e-3 * abs(item["value"]), ""
        return False, ""
    expected = normalize(item["answer"])
    pool = [cand for cand, _ in cands] + re.findall(r"```(?:\w+)?\s*\n(.*?)```", text, re.S)
    return any(normalize(cand) == expected for cand in pool), ""


class Native:
    def __init__(self, path, context=8192):
        self.target = NativeTarget(ROOT / "target/release" / native_library_name())
        self.model, status, message = stream_model(self.target, file_chunks(path, 1 << 24), path.stat().st_size, 1 << 24)
        assert self.model, message
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")
        self.context = context

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def solve(self, item, thinking, tools, attempts, temperature=0.0, seed=0, turn_tokens=None):
        turn_tokens = turn_tokens or (2048 if thinking else 768)
        messages = [{"role": "user", "content": item["question"] + ASK[item["kind"]]}]
        text = self.tokenizer.apply_chat_template(messages, tools=TOOLS if tools else None, add_generation_prompt=True,
                                                  enable_thinking=thinking, tokenize=False)
        session = self.target.call("nncoss_llm_session_new", self.model, self.context)
        checked(self.target, self.target.call("nncoss_llm_set_sampler", session, temperature, 20, 0.95, seed), "sampler")
        ids = self.encode(text)
        status, message = prefill(self.target, session, ids)
        assert status == 0, message
        position = len(ids)
        transcript = [text]
        segments = [{"role": "env", "ids": ids}]
        record = {"attempts": [], "tool_calls": 0, "tokens": 0, "solved": None}
        tool_rounds = 0
        attempt = 0
        try:
            while True:
                generated = []
                stopped = False
                while len(generated) < turn_tokens and position < self.context - 8:
                    status, token, message = sample(self.target, session)
                    assert status == 0, message
                    generated.append(token)
                    checked(self.target, self.target.call("nncoss_llm_decode", session, token), "decode")
                    position += 1
                    if token in (IM_END, EOT):
                        stopped = True
                        break
                record["tokens"] += len(generated)
                reply = self.tokenizer.decode(generated, skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "")
                transcript.append(reply + "<|im_end|>")
                segments.append({"role": "model", "ids": list(generated), "attempt": attempt})
                calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", visible(reply) or reply, re.S) if tools else []
                if stopped and calls and tool_rounds < 6:
                    tool_rounds += 1
                    outputs = []
                    for call in calls:
                        record["tool_calls"] += 1
                        try:
                            parsed = json.loads(call)
                            if parsed["name"] == "search":
                                record["searches"] = record.get("searches", 0) + 1
                                output = kb.search(str(parsed["arguments"]["query"]))
                            else:
                                output = tool_python(parsed["arguments"]["code"])
                        except Exception as error:
                            output = f"工具调用格式错误：{error}"
                        outputs.append(output or "（没有输出）")
                    addition = "\n<|im_start|>user\n" + "".join(f"<tool_response>\n{output}\n</tool_response>\n" for output in outputs).rstrip("\n")
                    addition += "<|im_end|>\n<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")
                else:
                    ok, error = judge(item, reply) if stopped else (False, "回答被截断")
                    record["attempts"].append({"ok": ok, "error": error})
                    if ok:
                        record["solved"] = attempt
                        break
                    attempt += 1
                    if attempt >= attempts or position > self.context - 2600:
                        break
                    tool_rounds = 0
                    feedback = RETRY[item["kind"]].format(error=error or "答案错误")
                    addition = ("" if stopped else "<|im_end|>") + f"\n<|im_start|>user\n{feedback}<|im_end|>\n<|im_start|>assistant\n"
                    addition += "" if thinking else "<think>\n\n</think>\n\n"
                if position > self.context - 2600:
                    record["attempts"].append({"ok": False, "error": "上下文已满"})
                    break
                ids = self.encode(addition)
                status, message = prefill(self.target, session, ids)
                assert status == 0, message
                position += len(ids)
                transcript.append(addition)
                segments.append({"role": "env", "ids": ids})
        finally:
            self.target.call("nncoss_llm_session_free", session)
        record["transcript"] = "".join(transcript)
        record["segments"] = segments
        return record


def subset(items):
    picked = []
    counters = defaultdict(int)
    every = {"mc": 2, "numeric": 4, "code": 3, "design": 1}
    for item in items:
        counters[item["kind"]] += 1
        if counters[item["kind"]] % every[item["kind"]] == 1 or every[item["kind"]] == 1:
            picked.append(item)
    return picked


def summarize(records, attempts):
    table = defaultdict(lambda: [0] * (attempts + 1))
    for record in records:
        for key in ("all", record["kind"], f"{record['kind']}/{record['level']}"):
            table[key][attempts] += 1
            if record["solved"] is not None:
                for attempt in range(record["solved"], attempts):
                    table[key][attempt] += 1
    return {key: {"acc": [round(value / counts[attempts], 3) for value in counts[:attempts]], "n": counts[attempts]} for key, counts in sorted(table.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm")
    parser.add_argument("--label", default="base")
    parser.add_argument("--conditions", default="nothink-notool,nothink-tool,think-notool,think-tool")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--think-temperature", type=float, default=0.6)
    args = parser.parse_args()
    native = Native(args.model)
    _, test = school.build()
    items = subset(test)
    out = HERE / "agent_reports"
    out.mkdir(exist_ok=True)
    for condition in args.conditions.split(","):
        thinking = condition.startswith("think")
        tools = condition.endswith("-tool")
        records = []
        started = time.time()
        attempts = 1 if False else args.attempts
        for index, item in enumerate(items):
            temperature = args.think_temperature if thinking else 0.0
            record = native.solve(item, thinking, tools, 1 if item["kind"] == "mc" else attempts, temperature=temperature, seed=1000 + index)
            record.update({"kind": item["kind"], "level": item["level"], "subject": item["subject"], "answer": item["answer"]})
            record.pop("segments", None)
            records.append(record)
            if (index + 1) % 25 == 0:
                done = summarize(records, attempts)["all"]
                print(f"{args.label} {condition} {index + 1}/{len(items)} acc={done['acc']} {time.time() - started:.0f}s", flush=True)
        result = {"label": args.label, "condition": condition, "seconds": round(time.time() - started),
                  "tokens": sum(r["tokens"] for r in records), "tool_calls": sum(r["tool_calls"] for r in records),
                  "table": summarize(records, attempts)}
        (out / f"{args.label}-{condition}.json").write_text(json.dumps({"result": result, "records": records}, ensure_ascii=False, indent=1))
        print("RESULT", json.dumps({k: v for k, v in result.items() if k != "table"}, ensure_ascii=False), flush=True)
        for key in ("all", "mc", "numeric", "code", "design"):
            print(f"  {key}: 第1/2/3次累计正确率 {result['table'][key]['acc']} n={result['table'][key]['n']}", flush=True)


if __name__ == "__main__":
    main()
