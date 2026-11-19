import json, sys, time
from pathlib import Path
ROOT = Path("/Users/admin/Desktop/prj2026/nncoss")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
from check_real_llm import checked, file_chunks
from check_wasm_llm import NativeTarget, native_library_name, stream_model, prefill, sample
from tokenizers import Tokenizer
import data


def main():
    output = Path(sys.argv[1])
    tokenizer = Tokenizer.from_file(str(ROOT / "model/Qwen3-0.6B/tokenizer.json"))
    target = NativeTarget(ROOT / "target/release" / native_library_name())
    path = ROOT / "model/Qwen3-0.6B/qwen3-0.6b-q4.nncossllm"
    model, status, message = stream_model(target, file_chunks(path, 1 << 24), path.stat().st_size, 1 << 24)
    assert model, message
    jobs = [("train", prompt, seed) for prompt in data.BRAIN for seed in (0, 1)]
    jobs += [("held", prompt, 0) for prompt in data.HELD_OUT]
    records = []
    started = time.time()
    for index, (split, prompt, seed) in enumerate(jobs):
        prefix = data.CHAT.format(question=data.fill(prompt))
        ids = tokenizer.encode(prefix, add_special_tokens=False).ids
        session = target.call("nncoss_llm_session_new", model, len(ids) + 340)
        status, message = prefill(target, session, ids)
        assert status == 0, message
        checked(target, target.call("nncoss_llm_set_sampler", session, 0.7, 20, 0.9, 7000 + index), "sampler")
        generated = []
        for _ in range(320):
            status, token, message = sample(target, session)
            assert status == 0, message
            if token in (151643, 151645):
                break
            generated.append(token)
            checked(target, target.call("nncoss_llm_decode", session, token), "decode")
        target.call("nncoss_llm_session_free", session)
        records.append({"split": split, "text": prefix + tokenizer.decode(generated, skip_special_tokens=False) + "<|im_end|>"})
        if index % 10 == 0:
            print(f"{index + 1}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=0), encoding="utf-8")
    print("done", len(records), flush=True)


main()
