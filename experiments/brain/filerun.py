import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import filekb
from check_real_llm import checked, file_chunks
from check_wasm_llm import NativeTarget, native_library_name, prefill, sample, stream_model
from transformers import AutoTokenizer


class Runner:
    def __init__(self):
        self.target = NativeTarget(ROOT / "target/release" / native_library_name())
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model/Qwen3-0.6B")

    def load(self, path):
        model, status, message = stream_model(self.target, file_chunks(path, 1 << 24), path.stat().st_size, 1 << 24)
        assert model, message
        return model

    def answer(self, model, text, max_new=160):
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                    enable_thinking=False, tokenize=False)
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        session = self.target.call("nncoss_llm_session_new", model, len(ids) + max_new + 8)
        try:
            checked(self.target, self.target.call("nncoss_llm_set_sampler", session, 0.0, 1, 1.0, 0), "sampler")
            status, message = prefill(self.target, session, ids)
            assert status == 0, message
            out = []
            for _ in range(max_new):
                status, token, message = sample(self.target, session)
                assert status == 0, message
                if token in (151645, 151643):
                    break
                out.append(token)
                checked(self.target, self.target.call("nncoss_llm_decode", session, token), "decode")
        finally:
            self.target.call("nncoss_llm_session_free", session)
        return self.tokenizer.decode(out, skip_special_tokens=True).strip()


def judge(item, reply):
    text = reply.lower().replace(",", "")
    hit = any(answer.lower() in text for answer in item["answers"])
    memory = item["memory"] is not None and item["memory"].lower() in text
    return hit, memory


def run(runner, path, split, label):
    model = runner.load(path)
    items = filekb.build()[split]
    table = defaultdict(lambda: [0, 0, 0])
    records = []
    started = time.time()
    for item in items:
        reply = runner.answer(model, filekb.prompt(item))
        hit, memory = judge(item, reply)
        row = table[item["kind"]]
        row[0] += hit
        row[1] += memory and not hit
        row[2] += 1
        records.append({"kind": item["kind"], "question": item["question"], "answers": item["answers"], "memory": item["memory"], "reply": reply, "hit": hit})
    runner.target.call("nncoss_llm_model_free", model)
    total = sum(row[0] for row in table.values()), sum(row[2] for row in table.values())
    summary = {kind: f"{row[0]}/{row[2]}" + (f" (答成记忆 {row[1]})" if kind == "counter" else "") for kind, row in sorted(table.items())}
    print(f"{label} {split}: 总 {total[0]}/{total[1]} = {total[0] / total[1]:.3f} {json.dumps(summary, ensure_ascii=False)} {time.time() - started:.0f}s", flush=True)
    out = HERE / "file_reports"
    out.mkdir(exist_ok=True)
    (out / f"{label}-{split}.json").write_text(json.dumps(records, ensure_ascii=False, indent=1))
    return total[0] / total[1]


if __name__ == "__main__":
    runner = Runner()
    for spec in sys.argv[1:]:
        label, path, split = spec.split(":")
        run(runner, Path(path), split, label)
